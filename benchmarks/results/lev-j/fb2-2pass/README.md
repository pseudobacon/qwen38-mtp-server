# LEV-J Phase FB2 — deterministic 2-pass flash SDPA kernel

**Run ID:** lev-j / fb2-2pass
**Date:** 2026-09-19
**Status:** ❌ **STOP — the 2-pass kernel is NOT deterministic.** The
structural claim "deterministic by construction" is **FALSE**. Per the task
instruction ("If any determinism failure appears in the 2-pass… STOP and report
rather than iterate fixes"), this phase halts on the finding. LEV-J remains
BLOCKED.

## What was built

`Libraries/MLXLMCommon/FlashSDPA.swift` — a 2-pass kernel (separate launches):

- **Pass 1 (`lev_j_flash_sdpa_max`):** per-row max of the (pre-scaled) scores →
  `m` [nq, Q] fp32, in device memory.
- **Pass 2 (`lev_j_flash_sdpa_sum`):** reads `m`, computes
  `out = Σ exp(score−m)·v / Σ exp(score−m)` in a fixed-order per-thread
  accumulation → `out` [nq, Q, D] bf16.

The two passes are separate `MLXFast.metalKernel` launches; `m` is the only
cross-pass state (device memory, launch boundary = global barrier). The Swift
API (`flashSDPA`, gate, counters, warm-up) is unchanged. The 1-pass source is
retained for reference.

## The structural argument (as designed) — and its flaw

The design rule the task set: "no shared memory written by one threadgroup and
read by another… no such sharing at all (separate allocations per pass, or a
single kernel-internal two-phase with a hard threadgroup_barrier between phases
and no overlap)." The 2-pass satisfies this at the **cross-threadgroup** and
**cross-pass** level:

1. Each threadgroup handles ONE (q_row, head); no cross-threadgroup sharing.
2. Pass 1 and pass 2 are separate launches; `m` in device memory is the only
   cross-pass state (launch boundary = global barrier).
3. The cross-BLOCK running state (pass 1 `m`; pass 2 `l`, `o_acc`) is in
   per-thread registers.
4. Pass 1 is a MAX (order-independent); pass 2 is a fixed-order accumulation.

**The flaw — the structural argument was incomplete.** It argued the cross-block
and cross-pass state is safe, but it did NOT establish that the **intra-block
score computation** (`t_part`/`t_scores`, the 8-simdgroup dot-product sum per
64-key block) is deterministic. That intra-block computation uses shared memory
with `threadgroup_barrier` ordering — the *same* code the 1-pass used — and it
is the source of the non-determinism. The 2-pass removed the cross-block online
softmax (the 1-pass's amplification), but the intra-block score race is shared
by both. So the 2-pass is **not** structurally deterministic.

## Determinism gate results (the gate that caught the 1-pass race)

Strengthened per FB2-B: 4 block counts (16/65/128/513), 100 concurrent pairs
each, GPU-side diff (1-int readback), plus 3× dispatch hash, plus the geometry
gate.

| geometry | result |
|----------|--------|
| 16 blocks (prefix 1024) | ✅ bit-exact (100/100 pairs) |
| 65 blocks (prefix 4160) | ❌ **100/100 pairs differ, totalDiff=39,168,000** |
| 128 blocks (prefix 8192) | ❌ **100/100 pairs differ, totalDiff=39,118,500** |
| 513 blocks (prefix 32768) | ❌ **100/100 pairs differ, totalDiff=39,116,500** |

The failure is **block-count dependent** (16 OK, ≥65 bad) — the same scaling as
the 1-pass/FA race signature. The diff is **deterministic** (identical
totalDiff across separate processes), i.e. two dispatches of the same input
differ in the same ~391.7 K of 393.2 K elements every time.

## Re-examination of the structural claim (required by the task)

The claim "deterministic by construction" is **FALSE**, and the reason is
specific: the intra-block score reduction (`t_part` → `t_scores`, the 8-simdgroup
partial sum) is a shared-memory reduction whose `threadgroup_barrier` ordering
does not, empirically, make the write→read visible in a way that keeps two
dispatches bit-identical at ≥65 blocks. This is *not* the online-softmax
cross-block amplification (which the 2-pass correctly removed) — it is the
per-block score computation, which is identical in the 1-pass and 2-pass.

Therefore neither the 1-pass nor the 2-pass is deterministic. The root cause is
in the shared per-block score reduction, not in the online-softmax running
state.

## Consequence

- **The FA GO (Phase FA) was premature** — its single-shot cross-process
  determinism check got lucky; the kernel has been non-deterministic all along
  (block-count dependent, ~≥65 blocks).
- **The 2-pass did not fix it** — it removed the online-softmax amplification
  but not the intra-block score race.
- **LEV-J remains BLOCKED** on a kernel correctness defect that is in the
  shared per-block score reduction (t_part/t_scores), present in both the
  1-pass and 2-pass designs.
- The production path stays incumbent dense SDPA (gate OFF) — byte-for-byte
  unchanged.

## Next steps (options)

1. **Pin down the intra-block score-reduction race.** It is the 8-simdgroup
   `t_part → t_scores` sum. Candidate: the `threadgroup_barrier(
   mem_flags::mem_threadgroup)` after the `t_part`/`t_scores` writes does not
   order the write→read across the simdgroup sum at scale. Try
   `mem_flags::mem_device` scope, or a second barrier, or a
   `simdgroup_barrier` around the `simd_sum`. This is a deep Metal shared-memory
   ordering issue and the two prior single-writer fixes did not help.
2. **Eliminate the cross-simdgroup score sum from shared memory entirely.**
   Restructure so each simdgroup owns a full 256-dim dot product for its keys
   (no 8-way shared sum), trading shared memory for more per-simdgroup work.
   Larger rewrite; must re-bench vs the c_flash bar.
3. **REJECT-on-correctness** and record: the FA GO was an over-optimistic
   determinism read; both the 1-pass and 2-pass are non-deterministic because of
   the shared per-block score reduction; LEV-J is closed on combined
   correctness-velocity grounds with the race analysis as evidence.

## Reproduce

```
cd mlx-swift-lm && swift build --build-tests
swift test --filter testFlashDeterminismAcrossBlockCount
#   16 blocks pass; 65/128/513 blocks fail 100/100 pairs (totalDiff ~39M)
```
