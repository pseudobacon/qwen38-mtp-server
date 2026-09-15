# HANDOFF — MTP exactness (Phase A) + prefix cache (Phase B) (COMPLETE, pending merge)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**:
> `agent-checkpoint.sh` ran successfully in both repositories (exit 0) and
> wrote `.dsh/last-agent-checkpoint` in each before this file was finalized.
> Markers in the Checkpoint markers section below.

## Objective and acceptance criteria

Two-phase task.
- **Phase A** — audit speculative-decoding exactness (T=0 greedy
  width-consistency; T>0 stochastic sampling correctness), produce a
  compatibility contract, fix small self-contained defects.
- **Phase B** — audit + harden the bounded, RAM-only, token-prefix
  KV/session cache for TTFT / prefill-latency improvement (the cache already
  exists; no SSD, no continuous batching, no kernel/head/quant/weight/MLX
  changes).

Both phases are code + test + doc; no product-behavior change beyond the
self-contained F1 defect fix (Phase A) and the cache hardening (Phase B).

## Result

**Phase A (COMPLETE).**
- T=0 greedy exact (modulo per-width near-tie ulps — pre-existing, documented
  in DRAFT-DEPTH-POLICY.md). T>0 exact rejection sampling (q/p share the same
  sampling controls).
- **F1 defect fixed**: `hasNonDefaultPenalties` was computed but never enforced
  in depth selection; added `effectiveMTPEnabled` (mtpEnabled &&
  !hasNonDefaultPenalties) used at both `decodeDepth` sites.
- **A4**: extracted `acceptanceAlpha`/`residualLogits`/`applySamplingFilters`
  as pure static methods (bit-identical); 8 math unit tests + an env-gated
  (`QWEN_MTP_DIST_HARNESS=1`) distributional-parity harness. Top-8 carried-mass
  Δ: T=0.8 token0 0.047 / token1 0.031; T=1.0 token0 0.016 / token1 0.078 — all
  under the 0.15 gate. Full-support TVD is informational only (tail-dominated
  at N=64 over the 248k vocab).
- Contract: `benchmarks/MTP-CORRECTNESS-CONTRACT.md`.

**Phase B (COMPLETE).**
- Audited `RadixKVCacheManager` (radix-tree token-prefix store; store-on-
  success; CoW via begin clone; TTL; LRU-leaf eviction under global memory
  pressure). Key constraint: recurrent (gated-delta) layers are **not**
  trimmable → a hit requires the stored history to be an **exact token-prefix**
  of the new seed → the TTFT win is same-thread multi-turn continuation, not
  interleaved shared-prefix.
- Hardened: (1) **namespace** on `CacheEntry`/`RadixNode`, `matchPrefix`
  requires `node.namespace == namespace`, MLXGenerator threads `cacheNamespace`
  (model+head+template) into `matchPrefix`+`store` (closes the latent
  cross-model serve gap); (2) **per-cache byte budget** `maxCacheBytes` (LRU
  eviction on store; set to `memoryLimitBytes/4`); (3) **metrics** lifetime
  hits/misses/evictions/stores + `Metrics` snapshot.
- B3: 3 new `RadixKVCacheManagerTests` (namespaceMismatchMisses,
  metricsCountHitsMissesEvictionsAndBytes, byteCapEvictsLRUOnStore).
  `RadixKVCacheManagerTests` **11 green**; full `HTTPServerTests` **125 green**.
- B4 fixture: `benchmarks/prefix_ttft.py` + `run_prefix_ttft.sh`. The
  **match-level** proof of cache hits is `RadixKVCacheManagerTests` (hit ratio
  > 0), not the wall-clock harness (which is entangled with SSE buffering and
  text round-trip exactness — documented, not cited as a number).
- Doc: `benchmarks/PREFIX-CACHE.md`.

## Git state

- `qwen38-mtp-server` (this repo): branch `feature/mtp-exactness-prefix-cache`,
  HEAD `9bcffbf` (prefix-cache hardening). Working tree clean.
- `../mlx-swift-lm`: branch `feature/mtp-exactness-prefix-cache`, HEAD `38f2bd2`
  (A4 harness). Working tree clean. (Engine commit precedes server commit, per
  protocol.)
- **Both pending merge to `main`** (auto-merge rule; this is the one remaining
  step).

## Commands / verification

```
# engine
cd ../mlx-swift-lm && swift test --filter Qwen38MTPDiagnosticTests   # 3 green (incl A4 harness, ~108s)
cd ../mlx-swift-lm && swift test --filter Qwen38MTPKernelTests        # 10 green
# server
cd qwen38-mtp-server && swift test --filter RadixKVCacheManagerTests # 11 green
cd qwen38-mtp-server && swift test --filter HTTPServerTests          # 125 green
cd qwen38-mtp-server && swift build -c release --product qwen38-mtp-server
```

## Unresolved risks / caveats

- Phase A stochastic MTP is **not** claimed distribution-exact beyond the A4
  top-8 mass gate (Δ < 0.15); the distribution harness is env-gated and
  N=64 (not a full-support proof).
- Phase B end-to-end TTFT is a fixture, not a cited number: the cache hit
  requires bit-exact token round-trip (text multi-turn not guaranteed) and the
  wall-clock is entangled with SSE buffering + empty-content deltas. The match-
  level proof is the radix tests.
- Cross-session absolute tok/s and TTFT are non-comparable (thermal); ratios
  within one run only.

## Do-not-repeat

- Do not claim stochastic MTP exact without the A4 verification.
- Do not use full-support TVD as the A4 gate — tail-dominated at N=64 over
  248k vocab; use top-8 carried mass.
- Do not change production sampling behavior (top-p non-standard transform) as
  part of the exactness audit.
- Do not run `head -N` on checkpoint output (SIGPIPE aborts before the marker
  write); redirect to a file.
- Do not cite cross-session absolute tok/s or TTFT as a conclusion.
- Do not create top-level `@Test` functions depending on global MLXRandom state
  — they race other suites under concurrent cross-suite execution.

## Next step

**Merge both repos to `main`** (auto-merge rule): engine `38f2bd2` first, then
server `9bcffbf`; delete both feature branches. (This is the only remaining
step; all tests are green and both working trees are clean.)

## Checkpoint markers

- engine: 2026-09-15T02:54:58+01:00
- server: 2026-09-15T02:54:52+01:00

The fresh-checkpoint procedure completed.