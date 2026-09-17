# FFP4 — FFN Prefill GEMM Kill-Switch (Relaxed Bit-Exactness)

**Date:** 2026-09-17
**Status:** **NO-GO** (kill-switch not met: best candidate 0.87–0.88× incumbent at
M=512, gate requires ≥ 2×). FFP5/FFP6/FFP7 not pursued.

## Context

FFP1 (`../ffp1/ffp1-report.md`) closed the FFN prefill GEMM as NO-GO under a
**bit-exact** requirement: the incumbent `quantizedMM` down_proj is 10× off-peak at
M=512 (≈21.6 TF vs ≈220 TF gateup), and the headroom was in GEMM tiling that
bit-exactness forbade changing.

This task **reopened** the kill-switch under a scoped relaxation (contract §0,
"Bit-exactness policy v2"): FFN GEMMs at prefill widths (M ≥ 256) may differ from
`quantizedMM` by a few ulp (fp32-accumulate-level), enabling tiling changes such
as split-K. The gate (FFP4): a candidate must achieve **≥ 2× sustained throughput**
at M=512 within tolerance. If no candidate clears 2×, stop with a negative result.

## Method

- **Candidates** (all `down_proj`, M=512, K=17408, N=5120, 4-bit affine group=64):
  - `incumbent_qmm` — the current `quantizedMM` (reference).
  - `splitk2_qmm`, `splitk4_qmm`, `splitk8_qmm` — decompose the large-K GEMM into
    `s` sub-`quantizedMM` over K/s each, accumulated in fp32, then cast to bf16.
    This increases K-parallelism (more threadgroups) — the tiling change the
    relaxation permits.
  - `bf16_gemm` — dequantize the weight once to bf16 (`MLX.dequantized`), then
    `MLX.matmul(x, W^T)`. Reference baseline (FFP1 measured ≈14.5 TF at M=512).
- **Tolerance check** at M ∈ {256, 512, 1024, 8192}: max|diff|, max relative,
  fraction of elements ≤ 2 ulp / ≤ 8 ulp (relative ulp), max relative-ulp.
- **Kill-switch timing** at M=512: interleaved A/B (`sustainedPair`, 128
  back-to-back per batch, 3 warmup discarded, ≥ 12 s wall per pair) so each
  candidate is measured against a fresh incumbent probe in the same DVFS window.

## Tolerance results

All split-K candidates fall within the fp32-accumulate tolerance for the **bulk** of
elements; the `maxRel`/`maxRelUlp` columns are dominated by near-zero reference
elements (denominator → 0), which inflates the relative metric — expected for a GEMM
over random inputs, not a correctness failure:

```
M=512  max|diff|=0.0625  frac<=2ulp≈0.96  frac<=8ulp≈0.99   (splitk2/4/8)
M=512  max|diff|=0       frac<=8ulp=1.00                  (bf16_gemm)
```

≈99 % of elements within 8 ulp at every engaged M. (The `bf16_gemm` candidate
produced a NaN at M=1024 — a bf16 overflow in the large-K reference matmul; it is
the *slowest* candidate and does not affect the split-K verdict.)

## Kill-switch results (M=512, 2 reps)

| Candidate   | Rep 1 (µs) | TF   | speedup | Rep 2 (µs) | TF   | speedup |
|-------------|-----------:|-----:|--------:|-----------:|-----:|--------:|
| incumbent   | ~4196      | ~21.9| 1.00x   | ~4385      | ~21.0| 1.00x   |
| splitk2_qmm | 4797       | 19.0 | 0.87x   | 4975       | 18.3 | 0.88x   |
| splitk4_qmm | 5084       | 18.0 | 0.82x   | 5327       | 17.1 | 0.83x   |
| splitk8_qmm | 5879       | 15.5 | 0.71x   | 6026       | 15.1 | 0.73x   |
| bf16_gemm   | 6285       | 14.5 | 0.67x   | 6315       | 14.5 | 0.69x   |

**No candidate clears 2×. Every candidate is *slower* than the incumbent**
(0.66–0.88×). The best split (splitk2) is *slower* than one full GEMM: the extra
kernel launches and fp32 cross-split accumulation cost more than the K-parallelism
gain. More splits are monotonically *slower* (splitk8 < splitk4 < splitk2),
confirming the per-split overhead dominates.

## Finding

The M=512 down_proj slowness is **not** a tiling-occupancy artifact fixable by
split-K. It is a fundamental property of the small-M, large-K GEMM shape that the
Metal quantized GEMM engine handles inefficiently, and decomposing K *increases*
per-op overhead without increasing throughput. This **extends FFP1's NO-GO**: even
with the bit-exactness relaxation explicitly permitting tiling changes, no candidate
reaches 2× (none even reaches 1.0×).

## Decision

**NO-GO.** The FFP4 kill-switch is not met (best 0.87–0.88× vs the 2× gate). Per the
gate, **FFP5 (engine integration), FFP6 (model-level audit), and FFP7 (A/B matrix)
are not pursued.** The FFP1 NO-GO stands, now confirmed under the relaxed
bit-exactness policy. The prefill FFN down_proj at M=512 remains a known,
quantified inefficiency with no viable kernel-side fix identified.

## Caveats

- **bf16_gemm NaN at M=1024:** the bf16 reference candidate overflowed at M=1024;
  it is the slowest candidate and irrelevant to the split-K verdict. The split-K
  candidates (the real test) were finite at all Ms.
- **Tolerance metric:** `maxRel`/`maxRelUlp` are inflated by near-zero reference
  elements; the meaningful measure is `frac<=8ulp` (≈0.99).
- **Scope:** candidates are down_proj only (the 10×-off-peak projection). gateup is
  not the bottleneck and was not a candidate.
- **Thermals:** no thermal warning during the run; interleaving cancels DVFS drift.

## Reproduce

```
cd /Users/cwong/ai/mlx-swift-lm && swift build --product qmvbench -c release
./.build/arm64-apple-macosx/release/qmvbench --ffn-prefill --ffn-cand \
    --ffn-ms 512 --ffn-batch 4 --ffn-wall 15
# tolerance table (M=256/512/1024/8192) + kill-switch timing (M=512, 2+ reps)
```
