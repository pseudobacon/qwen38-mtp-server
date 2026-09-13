# Qwen 3.8 gate+up / QKV fusion diagnosis — final report

Hardware: MacBook M5 Pro, 48 GB unified memory.
Model: Qwen 3.8-27B 4-bit (group 64), 64 layers (48 GDN + 16 full attention),
hidden 5120, intermediate 17408, fully dense.

---

## §0 — Pinned-fixture verdict

- Fixture: `benchmarks/prompts/essay-1024.txt`,
  SHA-256 `7ed683f87be0835c751505e2ee7dfc18fd922b93bcc32fad05d86c158cfb040e`.
- Protocol: greedy (temperature 0, `enable_thinking: false`, `max_tokens: 1024`);
  observed `finish_reason: length` in every cell.
- **The pinned essay fixture does NOT reproduce the recorded 645/1008/380.**
  It reproduces **599 accepted / 1086 proposed / 426 rounds**
  (acceptedPerStep 1.4061), stream hash
  `949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe`.
- The recorded 645/1008/380 (stream hash `139acb9d…`) belongs to the
  **specdec prompt** (`benchmarks/prompts/specdec-800.txt`) and was exactly
  reproduced under the rolled-back build (`benchmarks/results/resolve.jsonl`).
- Consequence: the pre-session "A1−A0 = +6.19 ms QKV regression" check was
  **cross-prompt and therefore confounded**. All in-session deltas below are
  the valid measurements.

---

## Item A — 2×2 same-session fusion matrix

Fixture `essay-1024.txt` (`7ed683f8…`), greedy 1024. Protocol: 6 reps of the
interleaved cell order A0(0/0) → A3(1/1) → A1(1/0) → A2(0/1); rep 1 discarded
as warmup; 5 measured reps per cell; fresh server start per cell;
`QWEN_MTP_STEP_TRACE=1`; port 18099.

All 24 cell outputs are **bit-identical**: same 1024-token stream, hash
`949b9423bd851233…`, 599/1086/426, `finish_reason: length`. Fusion engagement
verified on this binary (load-time summary: backbone swiGLU 64/64, qkv 16/16
attention layers, gdn 48/48).

avgStepMs = per-request MTP-STEP-SUMMARY field; min/max are per-rep extremes of the same field (reps 2–6).

| Cell | Fused QKV | Fused SW | avgStepMs mean | min | max | Δ vs A0 (ms) | tEvalAvg (ms) | TTLT (tok/s) | accepted/proposed/rounds | stream hash |
|------|-----------|----------|----------------|------|------|--------------|---------------|--------------|--------------------------|-------------|
| A0 | 0 | 0 | 143.042 | 134.790 | 164.483 | — | 131.1 | 16.89 | 599/1086/426 | 949b9423… |
| A1 | 1 | 0 | 143.304 | 134.600 | 152.421 | +0.262 | 131.3 | 16.81 | 599/1086/426 | 949b9423… |
| A2 | 0 | 1 | 142.967 | 138.174 | 147.587 | −0.076 | 130.8 | 16.81 | 599/1086/426 | 949b9423… |
| A3 | 1 | 1 | 145.429 | 135.032 | 151.562 | +2.387 | 133.1 | 16.55 | 599/1086/426 | 949b9423… |

(Per-rep `stepAvg` means: A0 142.943, A1 143.194, A2 142.854, A3 145.300.)

In-session verdict: **both fusions are latency-neutral.** |Δ| ≤ 2.4 ms
(≤ 1.7 % of the ~143 ms step), well inside per-rep noise (per-rep Δ swings
−21.8…+15.2 ms; run band 134.8–164.5 ms).

**Provenance of the table above.** The first matrix (02:41 binary) predates
the QMV dispatch fix and the fusion merge (`0514b11`, 02:49): with the pre-fix
dispatch a fused decode would have written only part of each 32-column output
tile and diverged from the known-correct greedy hash — since all 24 cells
matched the correct hash, **that run was fully eager** (its Δs −0.652/+0.271/
−0.892 ms were eager-vs-eager noise and are discarded). A first re-run on the
verified fusion-engaged binary was discarded for timing contamination
(parallel test runs overlapped its last cells: spikes 239.6/196.1/178.9 ms).
The table above is a clean re-run with no parallel builds/tests.

Caveat: cross-session absolute values are not comparable (this session
sustained ~135–165 ms vs ~116–124 ms in the prior session); only in-session
Δs are valid.

---

## Item B — Standalone QMV microbenchmark (`qmvbench`)

Calls `qwen35RoutedLinear` / `qwen35RoutedQuantizedMM` directly (no
reimplementation) on one real gate/up pair (layer 3 of Qwen3.8-27B-4bit):

- narrow projection: N = 17408, K = 5120, 4-bit group-64, packed K = 640
  uint32, scales/biases [N, 80] bf16;
- fused wide: N = 34816 — `global` (all gate rows then all up rows) and
  `interleaved` (per 32-row kernel tile: `[gate tile b, up tile b, …]`);
- kernel row tile 32; M ∈ {1, 4}; one shard loaded via CPU stream.

Protocol per condition: 100 warmup + 1000 timed iterations, randomized
interleaved order, 3 whole blocks (3000 samples); `ContinuousClock` with
device sync before and after each call.

**Correctness:** all kernel guards pass by construction; at both M = 1 and
M = 4 every routed condition is bit-identical to the incumbent
`QuantizedLinear`/`quantizedMM` output, the wide outputs split back to the
exact narrow outputs, and the interleaved fused tensor is a verified row
permutation of the global fused tensor.

M = 1 — microseconds per iteration (mean / min / p50 / std over 3000 samples),
GB/s at mean:

| condition | mean | min | p50 | std | GB/s |
|---|---|---|---|---|---|
| narrow_gate_routed | 384.92 | 312 | 380 | 30.15 | 130.36 |
| narrow_up_routed | 385.63 | 311 | 380 | 35.88 | 130.13 |
| wide_global_routed | 572.60 | 491 | 563 | 47.81 | 175.25 |
| wide_interleaved_routed | 571.56 | 491 | 564 | 35.38 | 175.57 |
| narrow_gate_fallback | 366.63 | 303 | 361 | 33.65 | 136.87 |
| narrow_up_fallback | 366.01 | 303 | 361 | 29.34 | 137.10 |
| wide_global_fallback | 553.31 | 480 | 545 | 63.59 | 181.36 |

M = 4 — microseconds per iteration (mean / min / p50 / std over 3000 samples),
GB/s at mean:

| condition | mean | min | p50 | std | GB/s |
|---|---|---|---|---|---|
| narrow_gate_routed | 464.06 | 379 | 458 | 29.38 | 108.42 |
| narrow_up_routed | 464.25 | 381 | 458 | 29.20 | 108.38 |
| wide_global_routed | 724.89 | 628 | 713 | 41.35 | 138.77 |
| wide_interleaved_routed | 726.65 | 635 | 713 | 87.33 | 138.43 |
| narrow_gate_fallback | 545.38 | 471 | 541 | 37.16 | 92.26 |
| narrow_up_fallback | 546.43 | 465 | 542 | 34.04 | 92.08 |
| wide_global_fallback | 904.30 | 824 | 898 | 38.67 | 111.23 |

**Conclusion (one paragraph).** The candidate-owned routed QMV kernel is
bit-identical to the incumbent `quantizedMM` at every tested geometry. At the
decode batch size M = 1 it is slightly slower than the incumbent per
projection (~385 µs vs ~366 µs, ≈ 5 %; effective ~130 vs ~137 GB/s), so at
M = 1 the fusion brings no micro-level win and a small kernel penalty. At
M = 4 the kernel reverses: ~15 % faster on a single projection (464 vs 545 µs)
and ~20 % faster on the fused wide tensor (725 vs 904 µs, ~139 vs ~111 GB/s).
So the routed kernel is net-positive only for M ≥ 4, and the fused wide
dispatch beats the incumbent by ~18–20 % there. The per-expert interleaved
layout is within measurement noise of the global layout at both M values
(M = 1: 571.6 vs 572.6 µs; M = 4: 726.7 vs 724.9 µs): it neither recovers nor
loses the decode range at the microbenchmark level. Caveat: this is a
per-projection microbenchmark (layer 3, one shard, per-iteration device sync
included); in-model decode reaches the routed kernel only for 2-D `x`
(3-D batched `x` falls back to `quantizedMM` by the `ndim == 2` guard,
bit-identical either way).

---

## Item C — env-gated interleaved layout (`MLX_QWEN_SWIGLU_LAYOUT`)

Knob: `MLX_QWEN_SWIGLU_LAYOUT ∈ {global (default), interleaved}`; read at
prepare time only; the fused forward path splits the output tensor on the
stored layout; the eager fallback is bit-identical and any fusion fallback
is logged loudly at model load.

Cells: Cglobal (`MLX_QWEN_FUSED_SWIGLU=1`, layout global) vs Cint
(`MLX_QWEN_FUSED_SWIGLU=1 MLX_QWEN_SWIGLU_LAYOUT=interleaved`), 6 reps of the
interleaved order Cglobal → Cint, rep 1 discarded, same fixture and protocol
as Item A, run beside A2 (Cglobal ≡ A2). Run on the post-dispatch-fix binary
in which fusion is verified engaged (see below).

**Fusion engagement evidence.** Load-time log of the current binary (both
layouts):

```
MLXLM: fusion prepare summary: backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 0 qkv 0
MLXLM: fused gate+up SwiGLU projection fell back to eager projections (layout <L>): a projection is not a stock QuantizedLinear instance
```

The summary line proves gate+up fusion engaged in **all 64 backbone layers**,
QKV fusion in all 16 full-attention layers, and GDN input-projection fusion in
all 48 GDN layers. The single fallback line comes from the **MTP head layer
only**: the head tree is BF16 (`mtp-head/model.safetensors` carries
`layers.0.mlp.gate_proj.weight BF16 [17408, 5120]`), so its projections are
not `QuantizedLinear` and fusion is ineligible there by design (documented).
Because the fused path is bit-identical, engagement is provable this way
without changing any hot-path behavior.

**All 12 cells bit-identical**: 599/1086 accepted (426 rounds), stream hash
`949b9423bd851233…`, `finish_reason: length` in every cell — the interleaved
layout is bit-exact end-to-end.

avgStepMs = per-request MTP-STEP-SUMMARY field; min/max are per-rep extremes
(reps 2–6).

| Cell | layout | avgStepMs mean | min | max | Δ vs Cglobal (ms) | tEvalAvg (ms) | TTLT (tok/s) | accepted/proposed/rounds | stream hash |
|------|--------|----------------|------|------|---------------------|---------------|--------------|--------------------------|-------------|
| Cglobal | global | 150.578 | 126.864 | 166.173 | — | 137.48 | 16.10 | 599/1086/426 | 949b9423… |
| Cint | interleaved | 152.187 | 133.479 | 160.276 | +1.609 | 138.87 | 15.85 | 599/1086/426 | 949b9423… |

Per-rep Δ (Cint − Cglobal, ms): +6.61, +8.48, +1.35, −2.50, −5.90 — the mean
+1.61 ms (+1.1 %) is inside the run's own noise band (Cglobal spans
126.9–166.2 ms across reps due to sustained-run thermal drift), so the
interleaved layout is **latency-neutral in-session**.

Memory cost: interleaved source views are materialized row-gather copies —
2 × (17408·640·4 + 2·17408·80·2) ≈ **100.8 MB per gate/up layer pair**
(≈ 6.5 GB across 64 layers). Fits in 48 GB; documented per spec.

---

## Conclusions

**(a) Net win / regression per fusion (in-session, fusion engaged).**
Fusion engagement is verified on the benchmarked binary (load-time summary:
backbone swiGLU 64/64, qkv 16/16 attention layers, gdn 48/48; only the BF16
MTP head falls back by design). Decode steps (2-D `x`) dispatch through the
routed QMV kernel; prefill/verify (3-D batched `x`) falls back to
`quantizedMM`, bit-identical.
- QKV fusion (A1): Δ **+0.262 ms** vs A0 → **latency-neutral** (no net win,
  no regression). The previously recorded +6.19 ms "regression" was
  cross-prompt and does not reproduce in-session.
- gate+up fusion (A2): Δ **−0.076 ms** vs A0 → **latency-neutral**.
- Both (A3): Δ **+2.387 ms** (+1.7 %) → inside the per-rep noise band
  (±22 ms); no net regression.
- Micro-level: the routed QMV kernel is ≈ 5 % slower than the incumbent at
  M = 1 and ≈ 15–20 % faster at M = 4 (Item B, bit-exact). End-to-end, the
  MTP-STEP metric is dominated by the draft/verify loop, so the per-M kernel
  effects do not translate into a measurable step delta in this session.

**(b) gate+up regression cause.** With fusion engaged, the clean in-session
matrix shows gate+up fusion at −0.076 ms vs A0 (A2), so there is no
in-session regression and no cause to identify. The previously recorded
+3–9 % decode-range regression does not reproduce under in-session control. The only
real, quantified costs are: (i) the M = 1 kernel penalty (≈ 5 % per
projection, not expressed end-to-end here), and (ii) for the interleaved
variant only, +≈ 100.8 MB/layer (≈ 6.5 GB total).

**(c) Does the interleaved layout recover the +3–9 % decode range?** No.
At the microbenchmark level, interleaved is within noise of global at both M
values (M = 1: 571.6 vs 572.6 µs; M = 4: 726.7 vs 724.9 µs). End-to-end,
the Item C cells are bit-exact in every rep and the in-session Δ vs the
global layout is +1.61 ms mean (+1.1 %), inside the run's thermal noise band
(per-rep Δ swings ±8.5 ms) — no recovery of the previously recorded
+3–9 % decode range, and no regression either.
