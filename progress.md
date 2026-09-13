# qwen38-mtp-server — Progress

A speculative-decoding server for Qwen 3.8 / 3.5 architectures on Apple Silicon MLX.

1. **Engine layer** — `../mlx-swift-lm`: custom MTP draft/verification block session (`Qwen38MTPBlockSession`) and model definitions.
2. **Model-state layer** — `MLXFastModel`: weight loading, KV-cache state management, tied-embedding sanitization, MTP head attachment.
3. **Server layer** — `HTTPServer`: Vapor-based OpenAI-compatible API (`/v1/chat/completions`), SSE streaming, memory admission control, observability endpoints.

## Architecture baseline (v1.0)

- **Verified features**: OpenAI SSE streaming fully compliant; parameter handling (temperature, max_tokens, `enable_thinking`); per-request KV-cache and tokenizer-cache isolation; MTP draft/verification alignment matching engine distributions (93.5% raw-token acceptance on the diagnostic benchmark).
- **Test suites**: `HTTPServerTests` 107/107 (the previously recorded 108 was a grep artifact — the `testRecoveryPolicyBypassedWhenDisabled` name contains "passed"), `MLXLMTests` MTP diagnostic suites, all green.
- **v1.0 diagnostic step-latency profile** (eager upstream backbone, per decode round): `avgStepMs` 151 ms, `tEvalMs` ~163 ms (roadmap estimate), TTLT 14.7–17.0 tok/s. Superseded absolute values — see the benchmarking-history note below.

## v1.1-performance progress log

### Checkpoint 1 — compiled activation micro-fusions (DONE)

Ported the four `compile(shapeless: true)` fusion blocks into the active fork (eager fallback gated by `MLXHardwareInfo.isCompiledDecodeSupported`, env override `MLX_COMPILED_DECODE`):

- `qwen35CompiledFusedSwiGLU` — `silu(gate) * up` (dense MLP, MoE shared expert).
- `qwen35CompiledSigmoidMultiply` — `x * sigmoid(gate)` (attention output gate, MoE shared-expert gate).
- `qwen35CompiledGatedDeltaGBeta` — GDN prologue (`g`, `beta`) computed once, reused for recurrence and MTP replay tape.
- `qwen35CompiledGatedDeltaPostNorm` — GDN post-norm (`preciseSwiGLU` + RMSNorm in one node; S==1 keeps eager `RMSNormGated`).

Small fixed shapes, immune to the Tahoe Metal JIT zero-result bug that affects whole-model compilation. Bit-identical to the eager chains.

### Checkpoint 2a — pinned QK RMSNorm + RoPE Metal kernel (DONE)

`qwen35_attention_qk_rms_rope_bf16_v1` (`Qwen35Kernels.swift`): fused Q & K RMSNorm + partial 64-dim RoPE in one Metal pass. Active via `Qwen35Attention.forwardFastPath` when compiled decode is on, Qwen 3.8-27B geometry applies (`usesFusedQKPreparation`: 24 Q heads, 4 KV heads, 256-dim heads, 64-dim partial RoPE, 10M base, default RoPE type), scalar offset, `L ≤ 32`, BF16. Bit-exact with the eager `qNorm().transposed()` + RoPE path.

### Checkpoint 2b / 2b-fix — fused residual+RMSNorm, routed QMV kernels, fast-path extraction (DONE)

- Fused residual + RMSNorm MSL kernel (`qwen35FusedResidualRMSNorm`, with the xsums sidecar variant): one launch computing the BF16-rounded residual `h = x + r` and the weight-scaled RMS norm of `h`, bit-exact with the eager chain. Row-parallel; accepts any row count.
- Candidate-owned 4-bit affine-4 / group-64 wide QMV dispatch (`Qwen35CustomQMV` / `qwen35RoutedLinear`): bit-exact replica of the incumbent `quantizedMM` wide kernel for the routed geometry. Covered M ∈ 2…9; Checkpoint 2c added the native M = 1 case (IPG 1, live-sums arm).
- **2b-fix**: five QMV dispatch defects fixed — uint32 dtype guard, packed-K dimension checks, kernel array count/order, 32-row output tile offset, per-M `ipg` grid (fixes OOB at M = 5). Also fixed the MLXFast grid-convention bug (see Benchmarking history): `MLXFast.metalKernel` treats `grid` as the *total* thread count; the dispatch was passing threadgroup counts, so pre-fix the kernel wrote only 4 of every 32 output columns. A trivial probe (2 of 512 slots written) confirmed the bug; both arms now dispatch `grid: (⌈m/ipg⌉·32, (n/32)·8, 1)`, `threadGroup: (32, 8, 1)`.
- Fast-path extraction: `Qwen35-FastPath.swift` and `Qwen35Kernels.swift` hold all custom additions; `Qwen35.swift` kept close to vendor shape with 1-line hooks (fork diff reduced from ~676 to ~30 lines).
- New standalone executable target `qmvbench` (see Benchmarking) — the first consumer that actually drives the routed QMV kernel end-to-end; previously the kernel was effectively dead code (no in-model or test path exercised it with the dispatch bug present).

### Checkpoint 2c — native M=1 wide QMV dispatch (DONE)

Single-token decode routes through the candidate-owned wide QMV kernel (`Qwen35CustomQMV.widths` includes 1; `ipg(1) = 1`, live-sums arm). `Qwen35Attention.projectPreRope` routes q/k/v and `mergeHeadsAndProject(routed: true)` routes o-proj through `qwen35RoutedLinear`.

### Checkpoint 2d — fused packed projections (DONE, measured neutral)

Both packed projections live in main via the `FusedQuantizedLinearProjection` machinery (storage-sharing views, net-0 memory, per-module invalidation on parameter updates):

- **Item 1, fused W_qkv** (`MLX_QWEN_FUSED_QKV`, default ON): q/k/v packed into one quantized matmul, sliced on output. Geometry: q_proj [12288, 640], k/v [1024, 640] packed U32 → [14336, 640]. `Qwen35FusedQKVProjectionTests` 8/8.
- **Item 2, fused W_gate+up** (`MLX_QWEN_FUSED_SWIGLU`, default ON): gate/up packed into one quantized matmul, sliced at half; `down_proj` stays eager. Geometry: [17408, 640] × 2 → fused U32 [34816, 640], all 64 layers. `Qwen35FusedSwiGLUProjectionTests` 8/8.
- MTP head is BF16 → fusion ineligible there by design (loud load-time log documents it).
- Both fusions are bit-exact end-to-end (identical greedy stream and acceptance under `Qwen38MTPDiagnosticTests`, 93.46%, logit divergence unchanged).

**Performance verdict (authoritative, in-session fusion matrix — see Benchmarking):** both fusions are **latency-neutral** at this geometry (dense 17408×5120 MLPs, GQA 12288/1024/1024): QKV +0.26 ms, gate+up −0.08 ms, both +2.39 ms against a ~143 ms step, inside the run's noise band. Retained default-ON: bit-exact, net-0 memory, fewer dispatches, with working rollback knobs. The interim cross-run comparisons that suggested a QKV regression were confounded (see Benchmarking history) and are retracted.

### Item C — env-gated interleaved gate+up layout (DONE — implemented, measured, rejected, removed)

`MLX_QWEN_SWIGLU_LAYOUT=interleaved` (default `global`) built per-expert-block interleaved fused weights via row-gather materialized copies. Bit-exact end-to-end, but latency-neutral at both the micro and end-to-end level, at a cost of ~100.8 MB per gate/up layer pair (~6.5 GB over 64 layers). **Verdict: no gain, +6.5 GB → rejected; the gate, layout plumbing, fuse path, and interleaved tests were removed from the engine in `901d2ca` (2026-09-13).** Historical Item C data remains in `benchmarks/FUSION_REPORT.md` and `benchmarks/results/itemC.jsonl`.

## Authoritative benchmark results

**Benchmarking methodology (applies to all future runs).**

- **Pinned prompt fixtures**: `benchmarks/prompts/*.txt`, read from file at request time; SHA-256 recorded with every result. Active fixtures:
  - `essay-1024.txt` (38 prompt tokens) — SHA-256 `7ed683f87be0835c751505e2ee7dfc18fd922b93bcc32fad05d86c158cfb040e`; greedy stream 599/1086 over 426 rounds, acceptedPerStep 1.4061, stream hash `949b9423…`.
  - `specdec-800.txt` (62 prompt tokens) — stream 645/1008 over 380 rounds, acceptedPerStep 1.6974, stream hash `139acb9d…`.
- **Determinism gate**: greedy (temperature 0.0), `enable_thinking: false`, `max_tokens: 1024`, `finish_reason: length`; every cell must reproduce the fixture's accepted counts and stream hash, else the run is discarded as a correctness/protocol failure, not reported as a performance delta.
- **Session discipline**: fresh server per cell (port 18099), `QWEN_MTP_STEP_TRACE=1`, rep 1 discarded as warmup, 5 measured reps, interleaved cell order so thermal drift hits cells evenly, no parallel builds/tests during timing. **Cross-session absolute latencies are not comparable** (observed 116–124 ms vs 135–165 ms bands for identical code under different thermal conditions) — only in-session deltas are valid. Fusion engagement is verified from the load-time summary (`backbone swiGLU 64/64 qkv 16/16 gdn 48/48; head …`), not from RSS.
- **Result artifacts**: `benchmarks/results/*.jsonl`, `benchmarks/FUSION_REPORT.md`.

### Headline throughput (conditions-labeled — not cross-comparable)

Phase 3 TTLT uses the pinned definition 1024 / wall-seconds (the 22.28 row is 1024 / decode-seconds, as originally recorded).

| Number | Conditions | Provenance |
|---|---|---|
| **18.92 tok/s** (1024 / 54.20 s wall, 141.62 ms mean step) | `specdec-800` fixture, acceptance 1.6974 (2.70 tok/round), in-session thermal ramp | Phase 3 B2 — **first valid specdec headline on current main** (post-QMV-fix, post-layout-removal binary) |
| **17.55 tok/s** (1024 / 58.48 s wall, 136.56 ms mean step) | `essay-1024` fixture, acceptance 1.4061 (2.41 tok/round) | Phase 3 B1 — current-main re-baseline, comparable to A0 |
| 16.64 tok/s (1024 / 61.66 s wall, 143.91 ms mean step) | `essay-1024` fixture, `MLX_COMPILED_DECODE=0` (compiled fast paths OFF) | Phase 3 B3 — ablation cell |
| 22.28 tok/s (1024 tok / 45.96 s decode, 120.86 ms steps) | `specdec-800` fixture, acceptance 1.6974 (2.70 tok/round), cooler session | 2d Item 3 final run — **pre-QMV-grid-fix binary**; the routed kernel never executed in that run |
| 16.55–16.89 tok/s (143.0–145.4 ms steps) | `essay-1024` fixture, acceptance 1.4061 (2.41 tok/round), sustained thermal load | Item A clean fusion matrix — current fixed binary, fusion engaged |

The old essay-vs-specdec gap decomposed as acceptance × step-time (×0.74). The new **B2-vs-22.28** gap is purely step time: acceptance is identical (645/1008/380, same stream hash `139acb9d…` in both), so 22.28 × (120.86/141.62) ≈ 19.0 ≈ B2's measured 19.05 (1024/decode-seconds: 19.046). That +17.2% step-time delta is the unseparated sum of this session's thermal state and the M=1 routed-QMV dispatch (≈5% slower per projection at M=1 per `qmvbench`) — consistent with the standing decision that cross-session absolute latencies are not comparable.

### Fusion matrix (2×2, in-session, fusion-engaged binary, essay-1024 fixture)

All 24 cells bit-identical (599/1086/426, hash `949b9423…`).

| Cell | Fused QKV | Fused SW | avgStepMs (mean) | Δ vs A0 | TTLT (tok/s) |
|---|---|---|---|---|---|
| A0 | 0 | 0 | 143.042 | — | 16.89 |
| A1 | 1 | 0 | 143.304 | +0.262 ms | 16.81 |
| A2 | 0 | 1 | 142.967 | −0.076 ms | 16.81 |
| A3 | 1 | 1 | 145.429 | +2.387 ms | 16.55 |

### Phase 3 — dual-fixture re-baseline + compiled-path ablation (2026-09-13)

One session, release build (server `83f82ab` / engine `901d2ca`), port 18099, greedy (`temperature: 0`, `enable_thinking: false`, `max_tokens: 1024`, `finish_reason: length`), `QWEN_MTP_STEP_TRACE=1`, fresh server per cell, 6 reps per cell interleaved B1→B2→B3, rep 1 discarded as warmup (5 measured), `pmset -g therm` logged before every rep, no parallel builds/tests, no mid-matrix rebuild. All 18 reps reproduced the pinned streams exactly (B1/B3: 599/1086/426 hash `949b9423…`; B2: 645/1008/380 hash `139acb9d…`) — zero gate failures. Fusion engagement confirmed at load time in every cell (`backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head 0/0` — BF16 head ineligible by design).

Raw per-rep records (with the pre-rep thermal line): `benchmarks/results/rebaseline-essay.jsonl` (B1), `rebaseline-specdec.jsonl` (B2), `ablation-compiled-off.jsonl` (B3). Driver: `benchmarks/run_phase3.sh`; stats/merge: `benchmarks/phase3_report.py`; run log: `.tmp/phase3-run.log`, thermal log: `.tmp/phase3-thermal.log`.

| Cell | Fixture | Config | avgStepMs (m/mn/MX) | tEvalAvg | tGraphBuildAvg | tCacheStateAvg | tHostReadAvg | wall s | TTLT 1024/wall |
|---|---|---|---|---|---|---|---|---|---|
| B1 | essay-1024 | default (fusions ON, compiled decode ON) | 136.559 / 124.108 / 142.341 | 124.627 / 112.447 / 130.310 | 10.310 / 10.049 / 10.438 | 1.533 / 1.511 / 1.553 | 0.012 / 0.011 / 0.013 | 58.483 / 53.121 / 60.893 | 17.553 / 16.816 / 19.277 |
| B2 | specdec-800 | default | 141.623 / 131.800 / 145.854 | 129.618 / 120.185 / 133.487 | 11.132 / 10.815 / 11.475 | 0.772 / 0.720 / 0.844 | 0.014 / 0.009 / 0.019 | 54.202 / 50.358 / 55.834 | 18.921 / 18.340 / 20.334 |
| B3 | essay-1024 | `MLX_COMPILED_DECODE=0` | 143.909 / 134.182 / 153.648 | 131.516 / 122.357 / 140.433 | 10.658 / 10.242 / 11.236 | 1.619 / 1.493 / 1.830 | 0.015 / 0.010 / 0.022 | 61.657 / 57.423 / 65.733 | 16.639 / 15.578 / 17.832 |

Accepted/proposed/rounds are constant across all reps of a cell (B1: 599/1086/426, B2: 645/1008/380, B3: 599/1086/426). m/mn/MX = mean/min/max over the 5 measured reps (2–6). The session shows an in-session thermal ramp (mean steps rise ~15–20 ms from rep 1 to rep 6 in every cell); the interleaved order spreads that drift evenly across cells, so only in-session deltas are valid.

**Ablation verdict (in-session, B3 − B1):** compiled decode OFF costs **+7.350 ms/step (+5.38%)** and **−0.914 tok/s (−5.21%)** on the essay fixture. This is the combined controlled contribution of every component gated by `MLX_COMPILED_DECODE` — Checkpoint 1's compiled activation micro-fusions, Checkpoint 2a's QK-RoPE fused kernel, and the fused residual+RMSNorm kernel — measured in an otherwise identical session. The packed W_qkv / W_gate+up fusions (separate env knobs, default ON) and the routed QMV kernels (ungated) are active in both cells, so they are not part of this delta.

**Binary-provenance note:** the release binary on disk (built 18:38) predated engine commit `901d2ca`; SwiftPM's incremental state had not invalidated the two changed engine modules. A forced recompile of `FusedQuantizedLinear.swift` + `Qwen35+FastPath.swift` plus relink produced a **different** binary (SHA-256 `88e27643…` vs `03231f16…`), and that is the binary this matrix ran on. The stale-binary risk flagged in the handoff is closed: the headline numbers above are from the current codebase.

### QMV microbenchmark (`qmvbench`)

Layer-3 gate/up pair, 100 warmup + 1000 timed × 3 blocks, per-call device sync. Routed kernel is bit-identical to incumbent `quantizedMM` everywhere. At M = 1 it is ~5% slower per projection (~385 vs ~366 µs); at M = 4 it is ~15% faster (~464 vs ~545 µs) and the fused wide dispatch ~20% faster (~725 vs ~904 µs). Interleaved layout within noise of global at both M. **Caveat: per-call sync measures serialized latency, not pipeline throughput — A2's null end-to-end result shows these per-kernel deltas do not survive async submission at this geometry.** A throughput mode (N calls per sync) is the standing methodology gap.

### Reference step-component profile (in-session, ~143 ms round)

`tEvalAvg` ~131 ms, `tGraphBuildAvg` ~10–11 ms, `tCacheStateAvg` ~0.9–1.7 ms, `tHostReadAvg` ~0.02 ms.

## Benchmarking history (superseded results — kept for provenance only)

The v1.0-baseline, 2a, 2b, and 2c tables previously in this file are **retracted as measurement artifacts**, for two reasons established on 2026-09-13:

1. **Prompt confound.** The 2c baseline ran the 38-token essay prompt (599/1086/426, 1.4061) while the 2d Item 1/2/3 runs ran the 62-token specdec prompt (645/1008/380, 1.6974). The reported "2c → Item 1 +6.19 ms QKV regression" and the "TTLT win" were cross-prompt comparisons. Provenance was established with a rolled-back all-fusion-off build (`benchmarks/results/resolve.jsonl`): essay reproduces 599/1086/426 exactly; specdec-800 reproduces 645/1008/380 exactly. Their absolute step latencies (~116–124 ms) additionally reflect a cooler thermal session and are not comparable to later runs.
2. **Binary-vintage / dead-kernel confound.** The 2a–2d checkpoints predate the QMV dispatch grid fix, and the routed kernel was never exercised in-model (decode reaches it only via 2-D `x`; verify falls back to `quantizedMM` by the `ndim == 2` guard). Correctness was preserved throughout because the eager/fallback paths carried those runs.

The engineering work of those checkpoints (kernels, fusions, tests, refactors) stands as described above; only their **timing tables** are invalid. The headline-throughput and fusion-matrix tables above are the only valid comparisons.

## Current status and roadmap

**Done:** Checkpoints 1, 2a, 2b, 2b-fix, 2c, 2d (both packed projections, merged to main in both repos), Item C interleaved layout (implemented, measured, rejected; removed from the engine in `901d2ca`), `qmvbench` microbenchmark target, prompt-fixture + determinism benchmarking protocol, MLXFast grid-convention bug fix, Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13 — first valid headline tok/s measured on current main, plus per-rep thermal logging wired into the harness).

**Decisions on record:**

- Fused W_qkv and W_gate+up: keep, default ON (latency-neutral, bit-exact, net-0 memory, rollback knobs `MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU`).
- Interleaved gate+up layout: rejected (no gain, +6.5 GB row-gather copies); removed from the engine in `901d2ca` (2026-09-13). Historical data in `benchmarks/FUSION_REPORT.md` and `benchmarks/results/itemC.jsonl`.

**Open items (in priority order):**

1. **Item D — route the verify pass through the candidate QMV kernel.** The `ndim == 2` guard currently sends verify (3-D batched `x`, the dominant `tEvalAvg` share) to incumbent `quantizedMM`; a free reshape to `[M, K]` would dispatch it through the routed kernel at M ∈ 2…9, where the microbench says it is ~15–20% faster. Go in with tempered expectations (see the A2 null result) and gate on an end-to-end A/B win, not the microbench alone.
2. **`qmvbench` throughput mode** — batch N calls per sync so per-kernel numbers reflect async pipeline conditions; required before trusting any future per-kernel delta.
3. **Thermal control for benchmarks** — per-rep `pmset -g therm` logging is now wired (`benchmarks/run_phase3.sh`); shorter run blocks / cooldowns remain, so absolute numbers become comparable across sessions.
4. **Attention-layer kernels and acceptance-rate work** — the remaining path toward the `tEvalMs` 24 ms / 30 tok/s target; weight-packing is measured out as a lever at this geometry.