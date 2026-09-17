# MCP1 — FFN M-curve (per-token cost vs prefill width)

**Date:** 2026-09-17
**Target:** Qwen3.8-27B 4-bit, layer 0 FFN (gate/up fused wide, down_proj)
**Model:** `/Users/cwong/ai/qwen38-mtp-server/weights` (MLX-4bit, 4-bit affine, group=64)
**Bench:** `qmvbench --ffn-prefill --ffn-pair` (release build, interleaved DVFS-fair)
**Binary SHA-256:** `a4b56e2b0238cf41df81cc103088bb5f92d9d7d0b2ca5e4de9180b72663fa7aa`
**Thermal:** `pmset -g therm` — no thermal/performance warning before or after every run.

## Objective

FFP1/FFP4 established the M=512 down_proj inefficiency (21.6 TF vs ~220 TF gateup) as
fundamental to the small-M/large-K shape class and closed the kernel axis. This probe
asks the orthogonal, **config-only** question: is that inefficiency a *fixed* property of
the shape, or an M-dependent penalty that a larger `--prefill-chunk-size` amortizes?

The sweep trades on **per-token cost** (`µs/GEMM ÷ M`), which is what the prefill wall
actually spends. If per-token FFN cost drops ≥ 25% at some M ≥ 1024, a larger chunk may
cut total prefill wall with zero kernel/source change → GO to the MCP2 sweep.

## FFN geometry (layer 0, verified from safetensors)

| GEMM | N | K | FLOPs / GEMM |
|---|---|---|---|
| gateup_wide (fused gate‖up) | 34816 | 5120 | 2·M·5120·34816 |
| down_proj | 5120 | 17408 | 2·M·17408·5120 |

## Methodology

- **Sustained no-sync, DVFS-fair interleaved** (`--ffn-pair`): gateup and down_proj
  alternate at the batch level (a,b,a,b,…) so both shapes sit in the same DVFS window.
  One `MLX.eval` per batch; 128/64/32/16 back-to-back GEMMs per batch (back-to-back
  scaled down with M to cap the resident gateup output buffer at ~9 GB and avoid memory
  compression, which corrupted an earlier M=2048 run at batch=128).
- **Reps:** 2 independent runs for M=512/1024 (the decision region); 1 run each for
  M=2048/4096/8192 (the far end, which is clearly not the minimum and need not be
  double-replicated). ≥ 70 batches per M (M=8192: 330).
- **Inputs:** `MLXRandom.normal` bf16 (dense GEMM throughput is input-value-independent).
- **Per-token cost** = mean per-GEMM µs ÷ M. This is the metric the sweep trades on.

## Curve (per-token cost, µs/token)

| M | gateup µs | gateup µs/tok | down µs | down µs/tok | **FFN µs/tok** | down TF |
|---|-----------|---------------|---------|-------------|----------------|---------|
| 512  | 1079.5 (avg of 1043.7/1115.4) | 2.108 | 4686.0 (avg 4647.9/4724.1) | **9.153** | **11.261** | ~19.5 |
| 1024 | 4115.4 (avg 3855.6/4375.2) | 4.019 | 2465.0 (avg 2336.9/2593.0) | **2.407** | **6.426** ← min | ~74 |
| 2048 | 10076.3 | 4.920 | 5341.3 | 2.608 | 7.528 | 68.3 |
| 4096 | 21700.6 | 5.298 | 11847.4 | 2.892 | 8.190 | 61.6 |
| 8192 | 41823.3 | 5.105 | 23438.9 | 2.861 | 7.966 | 62.3 |

**down_proj per-token:** 9.153 → 2.407 (M=512→1024) = **−73.7%**, then rises to a
plateau ~2.86–2.89 at M=4096/8192. **Minimum at M=1024.**
**gateup per-token:** rises monotonically 2.108 → ~5.1 (the large-N shape gets *less*
efficient per token at larger M).
**FFN total per-token:** 11.261 → **6.426** (M=512→1024) = **−42.9%**, minimum at
**M=1024**, then rises to ~8.0–8.2 at M=4096/8192.

## Finding

The M=512 down_proj anomaly is **M-dependent, not a fixed shape property**: at M=1024 the
same down_proj GEMM runs at ~74 TF (vs 19.5 TF at M=512), i.e. the per-token cost falls
73.7%. The FFN per-token cost has a clear interior minimum at **M=1024** (6.426 µs/tok):
below it, the down_proj M=512 tiling anomaly dominates; above it, the gateup per-token
cost rises and erodes the down_proj gain.

## Decision: **GO** to MCP2

The down_proj per-token cost at M=1024 (2.407) is 73.7% below M=512 (9.153) — far beyond
the 25% threshold. The curve is not flat; it has a sharp interior minimum at M=1024.

### Predicted optimal pc (stated before running MCP2)

The FFN per-token minimum is at **M=1024**, so the predicted optimal prefill chunk size is
**pc = 1024**. The MCP2 cell set `{512, 1024, 2048}` brackets the minimum (1024 is in the
set; 2048 is the "above-minimum" check; 4096 is not added because MCP1 predicts the
crossing point is *below* 2048, at 1024).

### Predicted total prefill savings

FFN per-token drops 42.9% (11.261 → 6.426 µs/tok). FFN is ~45% of the 32K prefill wall
@pc=512. Predicted 32K prefill reduction ≈ **45% × 42.9% ≈ 19%** — well above the 5% MCP2
gate. (The attention scores buffer at pc=1024 is ~1.6 GB @32K, ~2× the pc=512 value;
admission control models this — MCP2 verifies it still admits chunked 32K/64K and 507-
rejects dense.)

### SDPA curve (cost side) — not separately measured, by argument

Total causal-attention work is O(L²) and **independent of pc** (∑ᵢ pc·(i·pc) over chunks =
L²/2 for any pc). GDN and norms are O(L), also pc-independent. So the only pc-dependent
prefill cost is FFN, whose minimum is at pc=1024. The SDPA per-Q-token curve between the
endpoints would not move the optimum to first order; it is not measured (kept the probe
zero-kernel and cheap). MCP2's per-phase breakdown at each pc will empirically confirm
that SDPA/GDN/norms times are pc-invariant and that the total prefill minimum tracks the
FFN minimum.

## Caveats

- **DVFS across M:** the pair mode makes the gateup/down comparison fair *within* a run;
  across M (separate runs) the DVFS baseline drifts. The M=512→M=1024 drop (74%) dwarfs
  any baseline drift, and M=1024 vs M=2048/4096 (the "is the minimum at 1024" question)
  are close in M and measured in comparable regimes, so the interior minimum at 1024 is
  robust.
- **M=2048/4096/8192 single-rep:** the far end is one run each. It is clearly above the
  M=1024 minimum (7.5–8.2 vs 6.43 µs/tok), so no rep-2 was needed there.
- **M=8192 run was slow** (~22 min, sustained deep downclocking; std ~23–27%). It is the
  far end of the curve, not the decision region; its per-token (2.86 down / 5.11 gateup)
  is consistent with the M=4096 plateau.
- Cross-session absolutes are never conclusions; MCP2 measures the in-session paired
  32K prefill wall per pc.

## Reproduce

```
cd /Users/cwong/ai/mlx-swift-lm
B=.build/arm64-apple-macosx/release/qmvbench
# decision region (2 reps), batch sized to cap the gateup output buffer ~9 GB:
$B --ffn-prefill --ffn-pair --ffn-ms 512,1024 --ffn-batch 128 --ffn-wall 60   # rep 1 + 2
$B --ffn-prefill --ffn-pair --ffn-ms 2048   --ffn-batch 64  --ffn-wall 60
$B --ffn-prefill --ffn-pair --ffn-ms 4096   --ffn-batch 32  --ffn-wall 60
$B --ffn-prefill --ffn-pair --ffn-ms 8192   --ffn-batch 16  --ffn-wall 60
pmset -g therm   # before/after; expect no thermal warning
```
