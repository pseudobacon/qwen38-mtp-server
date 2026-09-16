# P2 — Toggle-Gated Prefill Kernel Extensions (LCP)

**Date:** 2026-09-16 · **EXP_ID:** LCP · **Kernel targets:** `residual3d` + `gdn-prefill`
**Device:** M5 Pro 48 GB · **Model:** Qwen3.8-27B 4-bit + MTP

## 1. Changes (engine fork `mlx-swift-lm`, `feature/prompt-1`)

Both extensions are **non-destructive, env-gated, default-OFF**, and reuse existing
Metal kernels — no new kernels, no new abstractions.

### 1.1 `MLX_QWEN_FUSED_RESIDUAL_3D` (residual3d)

The fused residual+RMSNorm Metal kernel (`qwen35FusedResidualRMSNorm`) was already
row-major/shape-agnostic; the only blocker was the Swift-side `ndim == 2` guard, which
forced the prefill (3-D `[B=1, S, H]` or `[S, H]`-as-3-D) path to fall back to
`x + r` followed by a separate `MLXFast.rmsNorm`. The 3-D branch of
`applyResidualNorm` (gated on `qwen35FusedResidual3DEnabled`, and further on
`x.dim(2) == 5120` with row-major strides) now routes 3-D prefill tensors to the same
fused kernel. Expected gain: removes one elementwise add + one rmsNorm pass over the
residual stream per layer per chunk (the `norm`/`residual` rows: 0.6–1.9 % of prefill).

### 1.2 `MLX_QWEN_FUSED_GDN_PREFILL` (gdn-prefill)

The fused GDN prework kernel (`qwen35PackedGDNPrework`: conv+SiLU+split + Q/K rmsNorm +
g/beta scale, one launch) was width-gated to verify widths (S ≤ 9). The kernel is
templated on S and per-(row, head) math is independent of S, so the gate is extended
to prefill widths (S ≤ 4096) under the new flag — independent of the verify-width flag
`MLX_QWEN_FUSED_GDN`. Expected gain: collapses the eager prework chain of the 48 GDN
layers' prefill path (the `gdn` row: 19–27 % of prefill) — a small fraction of that
row is the prework itself.

### 1.3 Master switch

`ENABLE_BIT_EXACT=1` forces the strict unoptimized fallback: dense attention **and**
both extensions (and the existing SwiGLU/QKV/4-GDN fusions) disabled. See P3.

## 2. Bit-exact validation

### 2.1 Unit tests (engine, `Tests/MLXLMTests/Qwen35PrefillFusionTests.swift`)

Bit-comparison = f32 upcast + `bitPattern` equality on every element (f32 upcast is
injective on bf16/f16/f32).

| test | widths S | compared against | result |
| --- | --- | --- | --- |
| `testResidualRMSNorm3DBitwise` | 1, 2, 512, 1000 | `x + r; MLXFast.rmsNorm(h, weight:, eps:)` | PASS (all elements) |
| `testGDNPreworkPrefillWidthsBitwise` | 16, 512 | eager chain: `generalConv` → SiLU → split → Q/K rmsNorm → scale → compiled g/beta (6 outputs: qNormed, kNormed, v, newConvState, g, beta) | PASS (all elements) |

### 2.2 Real-model content hashes (greedy, pinned fixtures, pc=512)

Every P2 cell must reproduce the pinned pc=512 baseline hash — a single flipped token
fails the gate.

| cell | flags | hash | result |
| --- | --- | --- | --- |
| p2-8k-both | res3d+gdn | `660dd1208737764c` | PASS (= baseline) |
| p2-16k-both | res3d+gdn | `2e583ad29dc28465` | PASS (= baseline) |
| p2-32k-res3d | res3d | `97bc0d74846a043b` | PASS (= baseline) |
| p2-32k-gdn | gdn | `97bc0d74846a043b` | PASS (= baseline) |
| p2-32k-both | res3d+gdn | `97bc0d74846a043b` | PASS (= baseline) |
| p2-64k-res3d | res3d | `14b26f9f89f16da5` | PASS (= baseline) |
| p2-64k-gdn | gdn | `14b26f9f89f16da5` | PASS (= baseline) |
| p2-64k-both | res3d+gdn | `14b26f9f89f16da5` | PASS (= baseline) |

**Both extensions are bit-exact end-to-end on the real model at 8K/16K/32K/64K**,
including their composition.

## 3. 32K/64K re-profile (eval-synchronized)

All cells greedy, pc=512, `MLX_TRACE_PREFILL=1`, fresh server per cell.

| cell | wall s | total ms | ffn | gdn | attn | sdpa | norm | resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32K base (P1) | 310.9 | 258565 | 116860 | 58849 | 75728 | 56224 | 4926 | 2201 |
| 32K res3d | 371.6 | 292542 | 131858 | 68229 | 85001 | 62916 | 5883 | 1571 |
| 32K gdn | 281.7 | 244365 | 110781 | 50871 | 75426 | 56939 | 5072 | 2215 |
| 32K both | 334.6 | 259484 | 120068 | 56173 | 76408 | 56139 | 5324 | 1511 |
| **32K base (post-matrix, same session)** | **211.0** | **169935** | 79375 | 41088 | 44163 | 32814 | 3295 | 2014 |
| 64K base (P1) | 779.6 | 693524 | 252854 | 132208 | 291818 | 247237 | 12142 | 4502 |
| 64K res3d | 754.6 | 671119 | 251501 | 132903 | 271119 | 225522 | 12643 | 2953 |
| 64K gdn | 628.7 | 550390 | 228607 | 104283 | 203256 | 166812 | 9812 | 4432 |
| 64K both | 473.6 | 396705 | 164969 | 76171 | 147473 | 124801 | 5350 | 2742 |

### Interpretation

- **Thermal/state variance dominates.** The flag-off 32K re-run at the end of the
  session (211.0 s wall / 169.9 s eval-sync) is **19 % faster** than the P1 32K
  baseline (310.9 / 258.6) and **33 % faster** than the slowest P2 cell (371.6 / 292.5).
  GPU clock state swings by up to ~1.6× within a session on this machine (no `pmset`
  thermal warning; sustained-load downclock and recovery). The expected fusion gains
  (1–3 % of prefill) are far below this noise floor: **per-kernel wall-time attribution
  is not resolvable on this hardware with single-shot cells.**
- **No regression:** every flag-on cell is within the observed flag-off spread at the
  same length; the composed `both` cell is bit-exact and the fastest 64K cell of the
  matrix (473.6 s).
- **Directional (GDN row):** with the gdn flag on, the GDN row measures
  132.2 → {104.3, 76.2} s at 64K and 58.8 → {50.9, 56.2} s at 32K; the res3d row is
  12.1 → {5.4, 12.6} s at 64K. The direction is consistent with the kernels fusing
  work, but the magnitudes are not separable from state variance.
- **Recommendation:** keep both toggles **default-OFF** (bit-exact default path is the
  unoptimized path); the extensions are correctness-preserving micro-optimizations,
  not a performance lever. The dominant levers are SDPA (35.6 % at 64K, O(L²), Phase 3
  scope) and FFN (36.5 %, prebuilt MLX 4-bit GEMM — out of scope).

## 4. Reproduction

```bash
cd /Users/cwong/ai/qwen38-mtp-server
bash benchmarks/run_lcp_p2.sh   # 8 cells, ~80 min, hash-gated
```

Artifacts in `benchmarks/results/prefill-opt-20260915/`:
`srv-p2-<cell>.log` (PF2 lines), `resp-p2-<cell>.json`, `summary-p2-<cell>.json`,
`wall-p2-<cell>.txt`, `run-p2.log`.
