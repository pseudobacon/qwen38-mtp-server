# Prefix Cache (Phase B)

Bounded, RAM-only, token-prefix KV/session cache for TTFT / prefill-latency
improvement. This document is the Phase B deliverable: it audits the
pre-existing `RadixKVCacheManager`, records the hardening added in this change
set, the test evidence, and the TTFT finding.

---

## 1. What already exists

`Sources/HTTPServer/Generation/RadixKVCacheManager.swift` is a radix-tree
(token-prefix) store of **completed-session** state (tokens + KV cache +
hidden/primary/top2 + the resolved KV config + namespace + timestamp).

- **Store-on-success**: `MLXGenerator` stores a session's state only on
  successful completion (no failed/aborted sessions pollute the tree).
- **CoW reuse**: a `begin()` hit clones the cached state into the new session's
  private state (copy-on-write); the tree state is never mutated in place by a
  live session.
- **Partial (prefix) match**: `matchPrefix` walks the tree by longest-common
  prefix and returns the deepest node whose state is present, unexpired
  (TTL), config-matching, and (fully matched OR trimmable).
- **LRU-leaf eviction under memory pressure**: `purgeIfMemoryPressure`
  evicts the oldest `lastAccessed` leaf until the shared memory budget is met,
  then `Memory.clearCache()` (a measured recovery/eviction action, per policy).
- **Bounded**: TTL (default 5 min), LRU eviction, and (now) a per-cache byte
  budget. No SSD persistence; no cross-process sharing.

### The exact-prefix constraint (drives the TTFT shape)

The recurrent (gated-delta) layers are **not trimmable**: `begin` requires
`trimmableOffset(reusableCache) == prefixCount`. So a hit is only usable at a
stored history that is an **exact token-prefix** of the new seed. This is
satisfied by **same-thread multi-turn continuation** (turn N+1's seed = turn N's
history + new user message) and *not* by interleaved shared-prefix (a new
thread after a shared system prompt diverges before the stored history's end).
This is the primary TTFT win, quantified in Section 4.

---

## 2. Audit findings → hardening (this change set)

| Finding | Hardening added |
|---------|-----------------|
| **Namespace gap**: the match key was `ResolvedKVCacheConfig` (KV geometry only — scheme, K/V format, group size, quantized start). Model/head/tokenizer/template identity was *not* part of the key, so a cache from model A could be served to model B if the model ever changed (latent: the model never reloads per process, but the invariant should be enforced, not assumed). | Added `namespace: String` to `CacheEntry`/`RadixNode`; `matchPrefix` now requires `node.namespace == namespace`. `MLXGenerator` threads its existing `cacheNamespace` (model + head + template) into both `matchPrefix` and `store`. |
| **No per-cache byte budget**: only the *global* memory-admission limit bounded the tree. The radix cache could, in principle, grow to consume the entire shared budget and starve active sessions. | Added `RadixKVCacheManager(maxCacheBytes:)`. `store` evicts LRU leaves until the tree's estimated footprint is under the budget. `MLXGenerator` sets it to `memoryLimitBytes / 4` so cached prefixes can't starve weights + active sessions. The global pressure eviction remains as a shared-budget backstop. |
| **No metrics**: no hit/miss/eviction/bytes exposure. | Added lifetime counters (`hits`, `misses`, `evictions`, `stores`) and a `Metrics` snapshot (`entries`, `estimatedBytes`, `hitRate`). |

The per-cache byte accounting (`totalEstimatedBytes`) uses the same conservative
per-token KV bound as the existing eviction path (full prefix length per leaf ×
`estimatedBytesPerToken`), so it is a safe over-estimate.

---

## 3. Test evidence (B3)

`Tests/HTTPServerTests/RadixKVCacheManagerTests.swift` — **11 tests, all green**
(pure Swift, no weights):

- Pre-existing (8): exact-history match, exact-prefix requirement, branch
  split, shorter-splits-longer, TTL expiration, config-mismatch miss, LRU
  eviction, clear.
- **New (3)**:
  - `namespaceMismatchMisses` — a model-B lookup does not see a model-A entry
    (and does not evict it).
  - `metricsCountHitsMissesEvictionsAndBytes` — hit/miss/store/bytes counters
    and `hitRate`.
  - `byteCapEvictsLRUOnStore` — a `maxCacheBytes` cap evicts the LRU leaf on
    store so the tree stays under budget.

Full server suite: **125 tests, 0 failures** (`swift test --filter HTTPServerTests`).

---

## 4. TTFT fixture (B4)

`benchmarks/prefix_ttft.py` (+ `benchmarks/run_prefix_ttft.sh`) is a
**fixture** for wall-clock TTFT: turn 1 (cold full prefill of a ~4K-token
system prompt), turn 2 (reuse turn-1's actual history as an exact token-prefix
+ short tail), and a cold control (same length, different system prompt).

**The match-level proof of cache hits is `RadixKVCacheManagerTests`
(`effectiveHitRatio > 0` on interleaved sessions, `RadixCacheBenchmarkTests`),
not the wall-clock harness.** The end-to-end TTFT is deliberately *not* cited as
a number here, for two measured reasons:

1. **Exact-prefix is a hard requirement** (Section 1): turn 2 hits only if the
   model's generated tokens re-tokenize bit-exactly when fed back as text.
   Text-based multi-turn round-trips are not guaranteed bit-exact (special-token
   / merge boundaries), so the harness's turn-2 is a necessary-not-sufficient
   probe.
2. **Measurement entanglement**: at ~4K tokens the prefill dominates (~tens of
   seconds), the model emits empty-content deltas, and the SSE role-framing line
   arrives when the stream opens (before prefill) — so a clean
   time-to-first-content number requires the server's SSE to not buffer, which
   is a separate concern.

Run the harness against the loaded model for a machine-specific TTFT; the
meaningful quantity is `ttft_turn2_hit_s` vs `ttft_cold_control_s` *within one
run* (never cross-session, per project rules). This is a **benchmark, not a
correctness gate**; it is not part of the test suite.

---

## 5. Do-not-claim

- No cross-session absolute TTFT is cited as a stable number (per project
  rules); the *ratio* `hit / cold-control` within one run is the meaningful
  quantity.
- The cache is single-stream (no continuous batching) and single-process
  (RAM-only, no SSD). It reuses the allocator cache between requests; it does
  not clear the global runtime memory cache between requests.

---

## 6. Reproduction

```
# B3 tests
 cd qwen38-mtp-server && swift test --filter RadixKVCacheManagerTests
# B4 TTFT (starts the release server, runs the harness, stops it)
 cd qwen38-mtp-server && swift build -c release --product qwen38-mtp-server
 cd qwen38-mtp-server && bash benchmarks/run_prefix_ttft.sh
```

Binary under test: `e448b2e2bbbfa7174c9d24e42108f5f721cb0a76b588f3fa3be7c1cbe1e167ab`.
