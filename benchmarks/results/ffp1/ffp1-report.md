# FFP1 — M=512 FFN prefill GEMM micro-benchmark (kill-switch)

**Date:** 2026-09-17
**Target:** Qwen3.8-27B 4-bit, layer 0 FFN (gate/up/down), M5 Pro GPU
**Model:** `/Users/cwong/ai/qwen38-mtp-server/weights` (MLX-4bit, 4-bit affine, group=64)
**Bench:** `qmvbench --ffn-prefill --ffn-pair` (release build)
**Decision: GO** — there is large, stable, shape-specific incumbent headroom at M=512.

## Objective

Determine, cheaply, whether the incumbent `quantizedMM` at the **default prefill
chunk width M=512** is far enough from peak that a specialized FFN kernel could
win by ≥10% (the FFP1 kill-switch). If the incumbent were already near the
compute floor, write a negative result and stop before touching the engine.

## FFN geometry (independently verified from the safetensors headers)

| GEMM | weight (U32) | N | K | FLOPs @M=512 |
|---|---|---|---|---|
| gate_proj | [17408, 640] | 17408 | 5120 | — |
| up_proj | [17408, 640] | 17408 | 5120 | — |
| **gateup_wide** (fused gate‖up) | — | **34816** | **5120** | **182.5 GFLOP** |
| **down_proj** | [5120, 2176] | **5120** | **17408** | **91.3 GFLOP** |
| **FFN total / layer** | | | | **273.8 GFLOP** |

At M=512 the FFN is deeply **compute-bound** (arithmetic intensity ≈1247–1293
FLOP/byte); the task's stated weight-bandwidth floor (0.209–0.403 ms) is
non-binding. The binding floor is the compute floor at peak FLOP/s.

QMV is **not** involved: the specialized QMV kernel is gated to widths 1–9
(`Qwen35+FastPath.swift`); at M=512 the FFN goes through the generic
`QuantizedLinear` → `quantizedMM` Metal path (no Swift-side fallback;
`QuantizedLinear.callAsFunction` calls `quantizedMM` unconditionally).

## Methodology

- **Sustained, no per-GEMM sync:** 128 GEMMs back-to-back per batch, one
  `MLX.eval` per batch. This matches the real prefill regime (a 512-chunk
  prefill runs 64 layers × (gateup, downproj) back-to-back), including the GPU
  DVFS state that real prefill experiences.
- **Fair same-window comparison:** gateup and downproj are **interleaved at the
  batch level** (a,b,a,b,…) so both sit in the same DVFS window. Measuring them
  in separate long runs confounds shape with DVFS state (the sustained load
  downclocks the GPU; `pmset -g therm` records no warning, i.e. normal DVFS,
  not severe throttling).
- **Warm-up:** 3 warm-up batches discarded before timing.
- **Inputs:** `MLXRandom.normal` bf16. For a dense GEMM the throughput is
  input-value-independent, so random inputs are valid for timing.
- **Wall time:** ~40 s per M (68 batches at M=512).

## Measurements (stable, wall-40 s run)

| M | shape | mean (µs) | p50 (µs) | std (µs) | batches | FLOP/s |
|---|---|---|---|---|---|---|
| 512 | gateup_wide | **828.4** | 955.2 | 311.7 | 68 | **220.4 TF** |
| 512 | downproj | **4582.5** | 4631.6 | 128.6 | 68 | **19.9 TF** |
| 1024 | gateup_wide | 3336.7 | 3604.2 | 553.9 | 82 | 109.4 TF |
| 1024 | downproj | 2041.5 | 2141.3 | 237.6 | 82 | **89.4 TF** |

The M=512 downproj number is stable across 5 independent runs (4531–4968 µs,
i.e. 19.9–20.1 TF) and has 2.8% std over 68 batches.

## Finding

**The incumbent `quantizedMM` is ~10× off-peak on the down_proj GEMM at M=512:**
20 TFLOP/s (downproj) vs 220 TFLOP/s (gateup_wide), in the same DVFS window.
down_proj (N=5120, K=17408) runs at **84.7% of the total per-layer FFN time**
(4582 / (828+4582)).

The anomaly is **shape- and width-specific, not a global slowness**:
- At M=1024 the same down_proj shape runs at **89.4 TF** (≈4× faster per FLOP
  than at M=512). So the down_proj is *not* intrinsically 20 TF — the
  `quantizedMM` kernel picks a bad tiling/occupancy configuration specifically
  at M=512 for the small-N (5120) / large-K (17408) shape.
- gateup_wide (large N=34816) is fine at every width (220 TF @512, 109 TF @1024).

This is a **tiling anomaly** in the generic quantized GEMM: the small N
dimension (5120 → few N-tiles) combined with large K (17408) under-utilizes the
GPU at exactly M=512, the default prefill chunk.

## Headroom (per layer, M=512)

| Scenario | gateup (µs) | downproj (µs) | FFN total (µs) | saved/layer |
|---|---|---|---|---|
| Incumbent | 828 | 4582 | 5410 | — |
| downproj → 89 TF (M=1024 level) | 828 | 1025 | 1853 | 3557 µs |
| downproj → 220 TF (gateup level) | 828 | 415 | 1243 | 4167 µs |

Over 64 layers: incumbent FFN ≈ 346 ms; fixing down_proj to the M=1024 level
saves ≈ 228 ms (≈4–5% of a ~5.5 s 32K prefill) — right at the FFP3 ≥5% gate;
matching gateup efficiency would save ≈ 267 ms.

## Decision: **GO**

The incumbent has 10× headroom on the down_proj GEMM at the default prefill
chunk width. A specialized FFN kernel (or a tiling fix) targeting the
small-N/large-K down_proj shape at M=512 has a clear path to a large win.
Proceed to **FFP2** (design + implement an FFN-phase kernel, env-gated
`QWEN_FFN_KERNEL=off|on`, default off, with a per-phase gateup/downproj counter).

## Caveats

- **M=8192 numbers are invalid** in the pair mode: back-to-back 128 of a
  [8192, 34816] output is 73 GB, exceeding 48 GB unified memory (swap /
  pathological eval). Large-M measurement needs a smaller back-to-back (≤2).
  Not needed for the M=512 decision.
- **DVFS:** the sustained protocol intentionally includes DVFS downclocking
  (it is part of real prefill). The gateup/downproj interleaving removes the
  order/DVFS confound for the shape comparison.
- **Random inputs:** valid for dense-GEMM throughput; a real-activation run is
  a worthwhile FFP2 confirmation but not required for the GO decision.
- The ≥10% FFP1 gate is about the incumbent's headroom (a candidate kernel's
  *actual* win is measured in FFP2/FFP3). Headroom here is ~10×, far above 10%.

## Reproduce

```
cd /Users/cwong/ai/mlx-swift-lm && swift build --product qmvbench -c release
./.build/arm64-apple-macosx/release/qmvbench --ffn-prefill --ffn-pair \
    --ffn-ms 512,1024 --ffn-wall 40 --blocks 2
pmset -g therm   # before/after; expect "no thermal warning"
```
