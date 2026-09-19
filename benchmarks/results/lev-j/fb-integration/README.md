# LEV-J Phase FB — Engine Integration (gated, default OFF)

**Run ID:** lev-j / fb-integration
**Date:** 2026-09-19
**Status:** ⚠️ INTEGRATION COMPLETE; **DETERMINISM GATE FAILS** — the flash
kernel is intermittently non-deterministic. **LEV-J is BLOCKED on a kernel
correctness defect.** The FA GO was premature.

## What was built

- `Libraries/MLXLMCommon/FlashSDPA.swift` — the deterministic flash-SDPA kernel
  (ported from `FlashBench`), the geometry gate, per-prefill dispatch counters,
  the `MLX_FLASH_SDPA` env var (default OFF; `ENABLE_BIT_EXACT=1` forces OFF),
  the loud load-time engagement log, and `warmFlashSDPAKernel()`.
- `Libraries/MLXLMCommon/AttentionUtils.swift` — `chunkedCausalPrefill` routes
  each prefill tile to `flashSDPA` when `Qwen38FlashSDPA.canEngage(...)` holds;
  otherwise it falls back to the incumbent dense `MLXFast.scaledDotProductAttention`
  (byte-for-byte unchanged). Dispatch counters (routed/fallback + Q-width
  histogram) are logged per prefill.
- `Libraries/MLXLLM/Models/Qwen38MTPBlockSession.swift` — `warmAllDepthShapes`
  calls `Qwen38FlashSDPA.logEngagement()` + `warmFlashSDPAKernel()` so the Metal
  JIT compiles at warmup (not on the first request).
- `Tests/MLXLMTests/Qwen38FlashSDPATests.swift` — forward/reverse tolerance +
  ulp histogram vs the dense incumbent, bit-exact determinism (40 concurrent
  pairs, the correct gate for an intermittent race), and the geometry
  negative-control gate.

## Unit-test results (2026-09-19, this session)

| test | result |
|------|--------|
| `testFlashMatchesDenseForward` (Q=2048, prefix=8192) | ✅ PASS (max\|diff\| ≤ 0.0625) |
| `testFlashMatchesDenseReverse` (reversed q/k/v) | ✅ PASS |
| `testGateGeometry` (decode Q=1, verify Q=9, head_dim≠256, non-causal, offset≠0) | ✅ PASS |
| `testFlashDeterminismAcrossBlockCount` (40 concurrent pairs, ≥128 blocks) | ❌ **FAILS INTERMITTENTLY** |

## The determinism defect (BLOCKING)

The flash kernel is **not** bit-exact. Two dispatches of the **same** input
occasionally produce **entirely different** outputs (whole-output corruption of
the softmax, not a 1-ulp flip). Measured:

- First run after build: FAIL (a ≠ b).
- Full-suite run: FAIL with `totalDiff = 500,705,160` ≈ 40/40 pairs × ~12.5 M
  elements — i.e. when the race fires, **every** output element is wrong.
- Stress (5 consecutive runs after the single-writer fix): run 1 FAIL, runs
  2-5 PASS. → ~20 % per-run failure in this window.

When the race fires it corrupts the running max/sum (`m_run`/`l_run`), so the
entire softmax is wrong — consistent with a stale read of the shared `t_scores`
by the Phase-3 online-softmax update.

### Fix attempted (insufficient)

The FA notes attribute the original race to "redundant 256-thread writes racing
the next-iteration read" and apply a single-writer fix to `m`/`l` (moved to
registers). I extended the **same single-writer pattern** to the shared-memory
` t_part ` (lane 0 only) and `t_scores ` (thread j only) — matching the
production `Qwen35Kernels` pattern. **This did not eliminate the race** (stress
run 1 still failed). The residual race is in deeper `threadgroup_barrier`
ordering of the online-softmax cross-block state and was not fixable by the
single-writer change.

## Consequence

- **The FA GO was premature.** FA reported "Determinism: bit-exact = true", but
  that check (single-shot, cross-process) got lucky; the race is intermittent
  and whole-output-corrupting. The kernel does **not** meet the non-negotiable
  determinism requirement.
- **LEV-J cannot proceed to FC/FD** on this kernel. The gate is
  `MLX_FLASH_SDPA=0` (default OFF), so the deployed path is the incumbent dense
  SDPA, byte-for-byte unchanged — the integration is safe to keep (it is inert
  until the kernel is proven deterministic).

## Next steps (options, in rough order of preference)

1. **Prove determinism with a barrier audit.** Trace the exact
   `threadgroup_barrier` ordering of `t_part`/`t_scores`/register state across
   the block boundary; the residual race is a stale shared read in Phase 3.
   Candidate: a full second barrier after Phase 3's register update, or a
   `mem_flags::mem_device` scope on the barriers.
2. **Replace online softmax with a provably-deterministic 2-pass kernel**
   (pass 1: max reduction; pass 2: weighted sum). Removes the cross-block
   running state entirely at the cost of a second K/V read. The kernel is
   compute-bound (FA measured ~25 % of peak FLOPs), so the extra bandwidth has
   headroom — but it must be re-benchmarked against the c_flash bar.
3. **Re-scope LEV-J.** If the race cannot be fixed with confidence, close LEV-J
   as REJECT-on-correctness (not on performance) and record that the FA GO was
   an over-optimistic determinism read.

## Reproduce

```
cd mlx-swift-lm && swift build --build-tests
swift test --filter Qwen38FlashSDPATests   # run several times; the
                                           # determinism gate fails intermittently
```
