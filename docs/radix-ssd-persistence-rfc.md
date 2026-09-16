# RFC: Radix-Tree SSD Persistence (cold disk tier beneath the in-RAM radix KV cache)

## 1. Motivation

The in-RAM radix KV cache (`RadixKVCacheManager`) makes same-process
prefix hits cheap, but it is volatile: a server restart loses every entry and
the next request re-prefills the full shared prefix. On this machine the
Qwen3.8-27B 4-bit weights + KV + head occupy ~15 GB of 48 GB unified memory,
so the in-RAM cache is bounded by `MemoryAdmissionPolicy` and cannot absorb a
large multi-session working set.

This RFC adds a **cold disk tier**: eligible radix entries serialize to disk
on graceful shutdown and on LRU eviction. After a restart, matching prefixes
restore from disk (lazy tensor load) instead of re-prefilling. The goal is a
post-restart TTFT for a warm prefix that is within 2× a warm in-RAM hit and
≥ 30% faster than a cold re-prefill.

## 2. Goals / non-goals

In scope:
- Serialize eligible (completed, non-cancelled) radix entries to a local SSD
  directory (per-node safetensors + JSON sidecar + a single tree index).
- Restore the tree skeleton on startup (metadata only, no tensor loads).
- Lazy-load tensors on a `matchPrefix` hit, on the model lane.
- Cross-restart key discipline (weight digest, KV config/scheme + tail,
  tokenizer/template, KV format) — any mismatch is a miss.
- Disk TTL (`--kv-ssd-ttl-seconds`, default 86400), byte budget
  (`--kv-ssd-cache-gb`, default 8), dir (`--kv-ssd-cache-dir`, default
  `~/.qwen38-mtp/kv-ssd`), each with env overrides.
- Memory accounting: a restored entry counts against `MemoryAdmissionPolicy`
  / `purgeIfMemoryPressure` exactly like a RAM entry.

Non-goals (first cut):
- No write-through (cache stays write-back).
- No compression (MLX array bytes are written as-is).
- No network/NFS, no multi-process sharing, no encryption.
- No continuous batching; single active generation lane unchanged.

## 3. Design

### 3.1 Serialization granularity: per-node (decision)

The radix tree's LCP match semantics depend on **which nodes carry state**,
including *internal* trimmed-state nodes created by `insert` splits (see
`RadixKVCacheManager.insert`). The benchmark's partial matches (e.g. the 2000
token shared system prefix) hit internal nodes that carry a **trimmed** state,
not a full session state.

If only complete sessions (leaves) were persisted and re-inserted on restore,
the internal trimmed-state nodes would be **stateless** in the skeleton
(re-insertion splits produce no state until tensors are present), and partial
matches at internal nodes would break — violating "LCP semantics must survive
restore" (the `RadixCacheBenchmarkTests` interleaved A/B/C scenario).

**Decision:** serialize **every node that has state** (leaves *and* internal
trimmed nodes), keyed by its full token prefix. On restore the tree topology
is rebuilt node-for-node (same segments, same parent links, same state
presence), so LCP semantics are preserved trivially. This also keeps the
weight-gated round-trip simple: a node's tensors are exactly its `state`/
`metaState` (see §3.2), with no trim-at-restore.

The existing `RadixCacheBenchmarkTests` must pass **unmodified** (they use
`cache: []` dummy payloads and exercise pure tree LCP semantics, which this
design leaves intact).

### 3.2 Per-node payload (what is serialized)

A radix `CacheEntry` (see `RadixKVCacheManager.CacheEntry`) contains:
`tokens: [Int]`, `cache: [any KVCache]`, `hidden: MLXArray`, `primary: Int`,
`top2: ([Int], [Double])`, `config: ResolvedKVCacheConfig`.

The `KVCache` protocol already exposes the serialization hooks
(`state: [MLXArray]` and `metaState: [String]`, `KVCache.swift:60-63`). The
cache *structure* is reconstructed by `model.newCache(parameters:)`
(`Qwen38MTPTarget.swift`), which returns fresh caches of the correct
types/dimensions (16 `KVCacheSimple`/`RotatingKVCache` full-attention + 48
`MambaCache` GDN for this model); each cache's `state`/`metaState` is then
set from the saved data.

Per node, two files:
- `<nodeID>.safetensors` — all `MLXArray`s: `"hidden"` plus one entry per
  `KVCache` state array, keyed `"cache.<i>.<j>"`.
- `<nodeID>.json` — JSON sidecar: `segment` (token ids), `parent` (node id),
  `config` (resolved KV config), `key_dims` (§3.6), `created_at`, `primary`,
  `top2`, and per-cache `meta_state: [String]` + `count` (number of state
  arrays, to know how many to load).

### 3.3 On-disk layout

```
<ssd-dir>/
  index.json                 # tree topology + per-node metadata (visibility gate)
  nodes/
    <nodeID>.safetensors
    <nodeID>.json
```

`index.json` is a flat list of nodes (built by a DFS of the tree):
```
{
  "version": 1,
  "key_dims": { "weight_digest", "template_hash" },   # uniform across tree
  "nodes": [
    { "id": 0, "segment": [int...], "parent": null,
      "has_state": true, "file": "nodes/<nodeID>",
      "created_at": iso8601, "bytes": <int> }
    ...
  ]
}
```
`nodeID` is a SHA-256 of the canonical JSON of `(key_dims, fullTokenPrefix)`,
so re-writing the same node is idempotent (same file names).

### 3.4 Atomicity (a half-written entry must never be matched)

`index.json` is the **visibility commit point** (written via temp + rename):
1. Write `nodes/<nodeID>.safetensors` and `nodes/<nodeID>.json` to their final
   names (best-effort, idempotent).
2. Atomically replace `index.json` (temp + rename) to include the node.

An entry is matchable **iff** it is in `index.json` **and** both of its
`nodes/` files exist (verified on load; missing → miss). A crash between (1)
and (2) leaves orphaned `nodes/` files with no index reference; they are never
matched and are reclaimed by the byte-budget LRU sweep (§3.7). No entry can
ever be half-matched.

### 3.5 Write points (write-back)

- **Graceful shutdown** (`ModelShutdownHandler.shutdownAsync`, after
  `scheduler.drain`): DFS the live tree, serialize every node with state,
  write `index.json`. Entries from cancelled/in-flight sessions are never
  stored (the existing `store()` only runs on successful completion), so the
  tree contains only eligible entries.
- **LRU eviction** (`evictLRULeaf`): before dropping the evicted leaf's
  in-RAM state, serialize it to `nodes/` and atomically add it to the index
  (in-memory copy + a flush). The in-memory index is the source of truth and
  is flushed on eviction and shutdown.

### 3.6 Restore flow (startup + lazy load)

- **Startup**: read `index.json`. Rebuild the radix tree **skeleton** — same
  nodes, same segments, same parent links, same `has_state` flags, `key_dims`,
  `created_at`. **No tensor loads.** Each stateful node gets a `diskState`
  reference (file path + key dims) and its in-RAM `cache` stays `nil`.
- **Match**: `matchPrefix` treats a node as matchable if it has state in RAM
  (`cache != nil`) **or** on disk (`diskState != nil`), and `config` matches
  and the disk TTL has not expired (§3.8). A disk hit returns a
  `DiskStateRef` (node id + file path + key dims) plus the prefix count.
- **Lazy load** (model lane, in `MLXGenerator`): read `nodes/<id>.safetensors`
  + `.json`, reconstruct the cache via `model.newCache(parameters:)` and set
  each `state`/`metaState`, build the `CacheEntry`, and hand it to
  `session.begin` — which runs the **existing** `trimmableOffset == prefixCount`
  desync check. A desync falls back to full prefill (existing path).
- **Failure → miss, never an error**: any missing file, safetensors/JSON parse
  failure, key-dimension mismatch, TTL expiry, or `trimmableOffset` desync
  yields `prefixCount = 0` / no entry. The request always degrades to a normal
  prefill.

### 3.7 Byte budget + orphan cleanup

`--kv-ssd-cache-gb` (default 8). Before/after a flush, sweep `nodes/`: delete
(orphaned, or LRU-oldest) files until total bytes ≤ budget. Orphans (files not
in `index.json`) are removed first. This is best-effort and never blocks the
generation lane (runs on the model lane during shutdown / idle eviction).

### 3.8 TTL

`--kv-ssd-ttl-seconds` (default 86400) applies to **disk-resident** entries,
measured from `created_at` (write time). The 300 s RAM default
(`tokenizationCacheTTLSeconds` / `samplingParams.ttlSeconds`) is too short for
restart persistence. On a disk hit the entry is loaded into RAM and becomes an
ordinary RAM entry (RAM TTL from that point).

### 3.9 Memory accounting

A lazy load is a disk read into RAM. Before loading, if
`Memory.activeMemory > memoryAdmissionPolicy.memoryLimitBytes * pressure`
(> 90% by default), the load is treated as a **miss** (no load); otherwise it
loads and the entry counts as a RAM entry in the tree, subject to the existing
`purgeIfMemoryPressure` / `evictLRULeaf` eviction (which now serializes to disk
before dropping, per §3.5).

### 3.10 Key discipline (cross-restart identity)

A disk hit is a **miss** if any of these differs from the current process:
- **Weight digest** — SHA-256 tree over the weights dir (reuses
  `MLXGenerator.weightIdentity()`, memoized).
- **KV config/scheme + tail** — `ResolvedKVCacheConfig` (already checked by
  `matchPrefix`'s `config ==`), which also encodes the **KV format** (f16 vs
  quantized + group size + quantizedKVStart).
- **Tokenizer/template** — a hash of the chat template string (redundant with
  the token-id prefix, but cheap insurance).

`index.json` stores `key_dims` (uniform across the tree); a disk hit compares
them against the current process's values. Mismatch on any dimension → miss.
The token-id prefix itself is matched by the tree walk (as today).

### 3.11 Config surface

New `ServerConfig` fields (with env overrides), following existing conventions:
- `--kv-ssd-cache-dir` / `QWEN_KV_SSD_CACHE_DIR` (default `~/.qwen38-mtp/kv-ssd`)
- `--kv-ssd-cache-gb` / `QWEN_KV_SSD_CACHE_GB` (default 8)
- `--kv-ssd-ttl-seconds` / `QWEN_KV_SSD_TTL_SECONDS` (default 86400)

A `--kv-ssd-disabled` (env `QWEN_KV_SSD_ENABLED=0`) turns the tier off
entirely (no reads, no writes) — the safe default-off escape hatch.

## 4. Testing

### 4.1 Pure-Swift (no weights)

Extend/add tests exercising the index, skeleton, and failure paths with dummy
payloads (mirroring `RadixCacheBenchmarkTests`):
- **Index round-trip**: build a tree, DFS-serialize to a temp dir, rebuild the
  skeleton, and assert the tree topology (segments + parent links + state
  presence) is identical and `RadixCacheBenchmarkTests`-style LCP match counts
  are unchanged.
- **Corrupt file → miss**: truncate/corrupt a `nodes/*.safetensors` or `.json`,
  assert the matching prefix becomes 0 (miss), no throw.
- **Key-dimension mismatch → miss**: change `weight_digest`/`template_hash` in
  the index, assert miss.
- **Budget LRU**: force the byte budget below the working set, assert orphaned
  files are reclaimed first and the total is within budget.
- **TTL expiry**: set `created_at` older than the TTL, assert disk entries are
  not matched.
- **Cancelled exclusion**: assert `store()` is the only writer (cancelled /
  in-flight sessions never produce entries) — covered by the existing
  store-on-success invariant + a unit test that a cancelled path stores nothing.
- **Mid-write crash**: write state files, crash before the index commit, assert
  the entries are not matched and the orphan sweep reclaims them.

### 4.2 Weight-gated (`QWEN_RUN_WEIGHT_TESTS=1`)

- Real-array save→load round-trip: store a completed session, serialize the
  node, rebuild the skeleton, lazy-load, and assert the reconstructed
  `CacheEntry` has `trimmableOffset(reusableCache) == prefixCount` and the
  cache `state`/`metaState` match the original bit-for-bit (MLX array
  equality on every `state` array).
- Generation through a restored session is **token-identical** to a fresh
  session on the same prompt (greedy, temp 0).

These run in a **separate `swift test` invocation** from other weight-gated
tests (two 14 GB model loads in parallel deadlock the MLX eval lock).

## 5. Benchmark extension

Extend the radix e2e benchmark with a **restart step**: warm the prefix set,
gracefully shut down (flush to disk), restart the server (skeleton restore),
re-issue the same prompts, and measure post-restart TTFT. Report:
- post-restart TTFT (warm prefix, disk-restored),
- warm in-RAM TTFT (baseline),
- cold re-prefill TTFT (no cache).

## 6. Acceptance

Post-restart TTFT for a warm prefix:
- **within 2× a warm in-RAM hit**, and
- **≥ 30% faster than a cold re-prefill**.

Measured by the §5 restart benchmark on the M5 Pro, reported with the exact
commands and a `benchmarks/results/...` artifact.
