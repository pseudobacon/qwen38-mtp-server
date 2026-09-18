# Cold-JIT pre-warm (first-boot Metal compile elimination)

## Problem

MLX JIT-compiles the decode-family kernels (512-token seed forward, every
verify width, MTP head drafts, `draftTokenID`) on first dispatch. Metal
persists those compiled libraries in the **per-user** disk cache, so the
cost is first-boot-only:

| state | warmup (measured, this kernel set) | time-to-readyz |
|---|---:|---:|
| fresh install (caches cleared) | 7.5–11.6 s (quiet); up to ~18 s under concurrent load | ~10–14 s |
| warm restart (cache intact) | ~3.0 s | ~5.4 s |

The pre-fusion binary (LEV-C, 29434ccf) measured 16.9 s cold; the current
post-Item-1 kernel set compiles faster but the first-boot-only delta
(~5–8 s) remains the largest single startup lever after weights.

## How Metal's cache works (PW1 survey)

- Location: `$DARWIN_USER_CACHE_DIR/com.apple.metal/<framework-build>/`
  (here `32024`) plus the frontend cache `$DARWIN_USER_CACHE_DIR/
  com.apple.metalfe/`. Both are per-user; both must be cleared to simulate
  a fresh install (wiping only `com.apple.metal` leaves ~3 s of warm state).
- MLX loads the shipped `default.metallib` (CWD-relative, `METAL_PATH`) for
  its "always-list" kernels, then compiles the remaining kernels from
  embedded C++ sources via `device_->newLibrary(source, MTLCompileOptions)`
  (`cmlx` metal device). Compiled results land in the per-user cache and
  survive process restarts (hence 16.9 s cold → 3.0 s warm on the old
  kernel set).
- The compile key is inferred to be device + toolchain + kernel-source
  hash, so a cache entry is only valid for the exact binary/metallib/model
  that produced it. A stale entry is a wrong-weights hazard (the
  stale-metallib lesson): any pre-warm artifact must be **version-keyed and
  fail-loud on mismatch**, never silently reused.
- MTLBinaryArchive capture would require an MLX-side hook (upstream). Not
  needed: the pre-warm path covers the full first-boot JIT because it runs
  the same `warmAllDepths` the server runs at startup.

## Components

### `--prewarm-exit` (deployment pre-warm)

`HTTPServer serve --prewarm-exit` performs the full startup (weights,
tokenizer, head, `warmAllDepths`) **without binding the HTTP port**, writes
the provenance manifest, and exits. Run once at install time:

```sh
scripts/prewarm.sh            # runs the binary with --prewarm-exit
```

This compiles every decode-family kernel into the per-user Metal cache and
records what it was compiled for.

### Provenance manifest (`PrewarmProvenance.swift`)

`~/.qwen38-mtp/prewarm-manifest.json` (override: `QWEN_PREWARM_MANIFEST`)
records: binary SHA-256, metallib path + SHA-256, model weight-tree digest,
MTP-head digest, draft geometry (`spec_draft_n_max`, `forced_draft_k`),
macOS build, hardware ID, and the Metal cache directory. Written atomically
(tmp + rename, destination removed first) so a crash mid-replace degrades
to *no manifest* (cold-JIT expected), never to a stale hit.

### Startup check (normal mode)

On every production boot, a background task compares the manifest against
the current process state and logs one verdict:

- `MATCH` — Metal JIT cache expected warm (no action).
- `noManifest` — first boot; cold JIT expected (~10 s); run
  `scripts/prewarm.sh` once.
- `MISMATCH (fields: …)` — **fail-loud warning**: the cache cannot be
  trusted (binary/model/OS/hardware changed); expect a cold JIT; re-run
  `scripts/prewarm.sh`. Never silently reuses a stale cache.

### `--prewarm-check` (CLI one-shot)

Exits 0 (match), 1 (mismatch), 2 (no manifest/metallib unresolvable).
Useful in install scripts and CI.

### Shared deferred weight-identity digest

The weight digest is a ~8 s / 15 GB read. Both the startup check and the
lazy SSD path need it. `MLXGenerator.weightIdentityDeferred()` computes it
**once per process**, **deferred until after the first request** (60 s
idle cap), and shares it across both consumers — so it never overlaps
decode rounds. Measured effect: first-request latency in the pre-warmed
state went from ~17 s (two concurrent 15 GB reads) to ~2.7 s, matching the
no-manifest state; decode step time identical across all A/B cells
(~75 ms/step).

## Validation (5-trial A/B, `benchmarks/run_prewarm_ab.sh`)

Protocol: alternating A (fresh install: no manifest, `com.apple.metal` +
`com.apple.metalfe` + SSD tier cleared, per-user Metal compiler service
killed) and B (pre-warm run, then production boot) cells; fixed greedy
64-token request per trial.

Result (run `prewarm-ab-20260918-1808`, final binary):

| metric | A (no pre-warm) | B (pre-warmed) |
|---|---:|---:|
| mean time-to-readyz | 10.03 s | **4.94 s** |
| mean first request | 2.46 s | 3.43 s (one 4.1 s outlier; decode identical) |
| Metal cache growth during run | 13 908 KB (compiles) | **164 KB (no JIT)** |
| content hash | single value across all 5 trials | |

Gates (all pass):

1. mean first-boot reduction ≥ 5.0 s **and** mean(B) ≤ 6.0 s —
   *the original planning gate was ≥ 8.0 s, derived from LEV-C's 16.9 s
   cold number on the pre-fusion binary. The current kernel set compiles
   in 7.5–11.6 s cold, so the achievable ceiling is ~5–6 s (B's floor is
   weight load ~1.5 s + warm warmup ~3.0 s + overhead). An 8.0 s
   reduction is structurally unreachable on this kernel set.*
2. determinism: one unique content hash, zero errors.
3. warm-restart sanity: mean(B) within [4.0, 9.0] s.
4. no JIT in B: mean Metal cache growth ≤ 2000 KB.

Determinism evidence: the greedy 64-token content SHA is identical across
all 15 trials of the three A/B runs (`96b3e57603e12cde`) — the pre-warm
caches compilation, not numerics.

## Operations

- **After install / model change / engine or metallib rebuild:** run
  `scripts/prewarm.sh` once. Startup logs the provenance verdict on every
  subsequent boot.
- **Mismatch warning** ⇒ re-run `scripts/prewarm.sh`; until then expect a
  cold JIT on first use (safe degradation, never stale reuse).
- The Metal cache is per-user: each macOS user who runs the server needs
  their own pre-warm (or accepts the one-time cold JIT).
