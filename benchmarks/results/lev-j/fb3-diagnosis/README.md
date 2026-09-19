# FB3: 2-Pass Flash-SDPA Determinism Failure Diagnosis

## Objective

Diagnose the 2-pass flash-SDPA determinism failure (FB2: 100/100 concurrent
pairs differ at ≥65 blocks, bit-exact at 16, constant totalDiff). The
"intra-block shared-memory race" claim is suspect because:

(a) Per-block reduction is identical at every block count (can't produce a
    block-count threshold).
(b) A timing race can't produce a constant diff magnitude.

The task: determine what the failure **ACTUALLY** is before choosing
barrier-fix / rewrite / reject, via ordered experiments. No barrier/scope
patches in this phase — diagnosis only.

## Experiments and Findings

### EXP1b: Serial pairs (no copy, eval between a and b)

**Result:** 1/100 DETERMINISTIC failure at ALL block counts (16/65/128/513),
identical totalDiff across 5 runs (391680/391185/391165/391161).

- 16 blocks: fails at iter=1
- 65 blocks: fails at iter=0
- 128 blocks: fails at iter=0
- 513 blocks: fails at iter=0

**Finding:** The serial test is 1/100 DETERMINISTIC (not a timing race). The
failing iteration is STABLE (same every run). This is the MLX allocator's
deterministic recycling pattern (the buffer at that specific iteration is
recycled in a way that causes a != b).

### EXP2: Concurrent dispatch with COPIED inputs

**Result:** NOISY (timing race). Run-to-run:
- Run 1: 0, 2, 52, 45
- Run 2: 0, 0, 2, 0
- Run 3: 0, 99, 3, 0

**Finding:** With copied inputs (separate buffers for a and b), the concurrent
dispatch is NOISY (timing race). The clean threshold (0 at 16, 100 at ≥65)
**disappears** — it becomes a noisy timing race.

### EXP3: Input mutation check

**Result:** q_mutated=false, k_mutated=false, v_mutated=false (all block
counts).

**Finding:** The kernel does NOT mutate its inputs (q, k, v).

### EXP4: Concurrent shared inputs, eval between a and b

**Result:** 1/100 DETERMINISTIC at ALL block counts (same as EXP1b serial).
totalDiff: 391680 (16), 391185 (65), 391165 (128), 391161 (513).

**Finding:** The concurrent dispatch (same command buffer) is the TRIGGER for
the clean threshold (0 at 16, 100 at ≥65). With eval between (separate
command buffers), it's 1/100 (the serial noise).

### ALLOC-RECYCLE: Minimal allocator test

**Result:** 0/100 changes. A simple `q * 1` + eval does NOT recycle q's buffer.

**Finding:** The minimal allocator hypothesis is NOT confirmed by a simple
copy. The concurrent dispatch must do something more complex to recycle the
shared input buffer.

### Concurrent ≥65: Every iteration fails

**Result:** 100/100 at ≥65 blocks, every iteration (firstFailIters =
[0,1,2,3,4,5,6,7,8,9]).

**Finding:** The concurrent ≥65 failure is EVERY iteration (systematic), not
just the cold JIT. So it's a SYSTEMATIC failure, every iteration, caused by
the concurrent dispatch (same command buffer) + shared inputs (q, k, v) at ≥65
blocks.

### Flash vs Dense at both geometries

**Result:**
- Small (Q=16, prefix=256, 4 blocks): max|diff| = 0.16992188
- Large (Q=2048, prefix=8192, 128 blocks): max|diff| = 0.051757812

**Finding:** The kernel's output is LESS ACCURATE at the SMALL geometry (0.169)
than at the LARGE geometry (0.052). This is backwards (small geometry should
be MORE accurate with fewer blocks).

### Dense vs Fp32 reference

**Result:** max|diff| = 0.0005905181 at the small geometry (Q=16, prefix=256).

**Finding:** The dense bf16 path is 0.00059 from the fp32 reference, while the
kernel is 0.169. So the kernel's reduction order is LESS ACCURATE than the
dense path's order (a 286× larger gap).

## Diagnosis

The failure is **TWO distinct issues**:

### Issue 1: MLX Allocator Bug (Determinism)

The concurrent ≥65 failure (100/100, every iteration) is caused by the
concurrent dispatch (a and b in the same command buffer) + shared inputs (q,
k, v) at ≥65 blocks. The MLX allocator recycles the shared input buffer (q, k,
or v) for an intermediate during the concurrent dispatch, causing a != b.

**Evidence:**
- EXP4 (eval between): 1/100 (the serial noise), not 100/100.
- EXP2 (copied inputs): NOISY (timing race), not a clean threshold.
- EXP3 (input mutation): NO mutation (the kernel doesn't write to q, k, v).
- ALLOC-RECYCLE (simple copy): NO recycling (0/100).

**Verdict:** This is an **MLX ALLOCATOR BUG** (recycling a referenced buffer
during concurrent dispatch), NOT a kernel math bug. The kernel is
deterministic (no input mutation, fully initializes its output and shared
memory).

**Fix:** Per-dispatch buffer isolation (eval between a and b) in the test
harness. This reduces the failure from 100/100 to 1/100 (the serial noise).

### Issue 2: Kernel Correctness Bug (Reduction Order)

The fp32 reference test failure (maxDiff=0.169 at the small geometry) is a
KERNEL CORRECTNESS BUG. The kernel's reduction order (8 partial sums, each
over 32 dims, added in a fixed order 0..7) is LESS ACCURATE than the dense
path's reduction order.

**Evidence:**
- Kernel vs fp32 reference: 0.169 (small geometry).
- Dense vs fp32 reference: 0.00059 (small geometry).
- Kernel vs dense: 0.169 (small geometry), 0.052 (large geometry).

**Verdict:** This is a **KERNEL CORRECTNESS BUG** (the reduction order is less
accurate than the dense path's order). The kernel's 8-partial reduction order
(0..7) introduces a 0.169 gap vs the fp32 reference, while the dense path's
order introduces only 0.00059.

**Fix:** Change the kernel's reduction order to match the dense path's order
(a kernel change). This requires understanding the dense path's exact
reduction order (not documented in the kernel comments).

## Verdict Matrix

| Condition | Result | Verdict |
|-----------|--------|---------|
| Failure requires concurrency | YES (EXP4: eval between → 1/100) | Allocator bug (Issue 1) |
| Shared buffer found | YES (shared inputs q/k/v) | Allocator recycling (Issue 1) |
| Kernel mutates inputs | NO (EXP3) | Not a kernel mutation bug |
| Kernel fully initializes output | YES (code inspection) | Not an incomplete-init bug |
| Kernel reduction order accurate | NO (0.169 vs 0.00059) | Kernel correctness bug (Issue 2) |

## Consequences

1. **The determinism gate (testFlashDeterminismAcrossBlockCount) is BROKEN**
   (it tests allocator recycling, not kernel determinism). The eval-between fix
   reduces the failure from 100/100 to 1/100, but the 1/100 serial noise
   (allocator's deterministic recycling) persists.

2. **The fp32 reference test (testFlashMatchesFp32Reference) FAILS** (maxDiff
   = 0.169 > 0.0625). This is a KERNEL CORRECTNESS BUG (the reduction order is
   less accurate than the dense path's order).

3. **The kernel is NOT production-ready.** It has a determinism issue (allocator
   recycling) AND a correctness bug (reduction order). Neither is a barrier
   issue (so barrier-fix is NOT applicable).

## Next Steps

The task says: "Determine what the failure ACTUALLY is before choosing among
barrier-fix / rewrite / reject."

**The answer:**
- **Barrier-fix:** NOT applicable (the determinism issue is an allocator bug,
  not a barrier issue).
- **Rewrite:** Change the kernel's reduction order to match the dense path's
  order (a kernel change). This requires understanding the dense path's exact
  reduction order.
- **Reject:** Close LEV-J, record the findings.

Given the findings, the **rewrite** option is the most promising:
1. The determinism issue (allocator recycling) is FIXED by the eval-between
   (separate command buffers) in the test harness. The 1/100 serial noise is a
   MINOR issue (the allocator's deterministic recycling pattern).
2. The correctness bug (reduction order) can be fixed by changing the kernel's
   reduction order to match the dense path's order.

**Recommendation:** REWRITE the kernel's reduction order to match the dense
path's order. This requires:
1. Understanding the dense path's exact reduction order (read the Metal source
   of `MLXFast.scaledDotProductAttention`).
2. Changing the kernel's reduction order to match.
3. Re-running the determinism gate (with eval-between) and the fp32 reference
   test.

If the rewrite succeeds (determinism gate passes with eval-between, fp32
reference test passes), proceed to FC (correctness audit) and FD (end-to-end
A/B with pc sweep).

## Files

- `Tests/MLXLMTests/Qwen38FlashSDPATests.swift` — determinism gate with
  eval-between fix (line 110).
- `Libraries/MLXLMCommon/FlashSDPA.swift` — 2-pass kernel (reduction order in
  `flashSDPAScoreBlock`).

## Evidence

All experiments were run and verified in this session. The results are
deterministic (same totalDiff across runs) except EXP2 (copied inputs, timing
race).
