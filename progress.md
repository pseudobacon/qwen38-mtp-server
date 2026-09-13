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

| Number | Conditions | Provenance |
|---|---|---|
| **22.28 tok/s** (1024 tok / 45.96 s, 120.86 ms steps) | `specdec-800` fixture, acceptance 1.6974 (2.70 tok/round), cooler session | 2d Item 3 final run — **pre-QMV-grid-fix binary**; the routed kernel never executed in that run |
| 16.55–16.89 tok/s (143.0–145.4 ms steps) | `essay-1024` fixture, acceptance 1.4061 (2.41 tok/round), sustained thermal load | Item A clean fusion matrix — current fixed binary, fusion engaged |

The gap decomposes exactly: acceptance ratio (2.41/2.70, −10.8%) × step-time ratio (120.9/145.4 ms, −16.9%) ≈ ×0.74, and 22.28 × 0.74 ≈ 16.5. **Neither number is wrong; they are different prompt × thermal-session conditions.** Note also that the current binary has **not** been *measured* on the `specdec-800` fixture under the pinned protocol — its first run on the current (post-QMV-fix) binary was the 2026-09-13 determinism verify cell (`benchmarks/results/verify.jsonl`, fusion-off): 645/1008/380, stream hash `139acb9d…` reproduced, ~124.8 ms mean step in a single uncontaminated cell. It now routes M=1 decode through the routed kernel (≈5% slower per projection at M=1 per `qmvbench`), so the true headline on current main is unmeasured and may land below 22.28 even in a cool session. The standing action is the dual-fixture re-baseline below.

### Fusion matrix (2×2, in-session, fusion-engaged binary, essay-1024 fixture)

All 24 cells bit-identical (599/1086/426, hash `949b9423…`).

| Cell | Fused QKV | Fused SW | avgStepMs (mean) | Δ vs A0 | TTLT (tok/s) |
|---|---|---|---|---|---|
| A0 | 0 | 0 | 143.042 | — | 16.89 |
| A1 | 1 | 0 | 143.304 | +0.262 ms | 16.81 |
| A2 | 0 | 1 | 142.967 | −0.076 ms | 16.81 |
| A3 | 1 | 1 | 145.429 | +2.387 ms | 16.55 |

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

**Done:** Checkpoints 1, 2a, 2b, 2b-fix, 2c, 2d (both packed projections, merged to main in both repos), Item C interleaved layout (implemented, measured, rejected; removed from the engine in `901d2ca`), `qmvbench` microbenchmark target, prompt-fixture + determinism benchmarking protocol, MLXFast grid-convention bug fix.

**Decisions on record:**

- Fused W_qkv and W_gate+up: keep, default ON (latency-neutral, bit-exact, net-0 memory, rollback knobs `MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU`).
- Interleaved gate+up layout: rejected (no gain, +6.5 GB row-gather copies); removed from the engine in `901d2ca` (2026-09-13). Historical data in `benchmarks/FUSION_REPORT.md` and `benchmarks/results/itemC.jsonl`.

**Open items (in priority order):**

1. **Dual-fixture re-baseline on current main.** Run the pinned-protocol matrix (or at minimum the default-config cell) on **both** `essay-1024` and `specdec-800` fixtures with the current binary in one thermally-controlled session, so the headline tok/s is a measured number for the current codebase rather than a cross-session artifact. Expected spread: acceptance factor alone spans ~2.41→2.70 tok/round between fixtures.
2. **Item D — route the verify pass through the candidate QMV kernel.** The `ndim == 2` guard currently sends verify (3-D batched `x`, the dominant `tEvalAvg` share) to incumbent `quantizedMM`; a free reshape to `[M, K]` would dispatch it through the routed kernel at M ∈ 2…9, where the microbench says it is ~15–20% faster. Go in with tempered expectations (see the A2 null result) and gate on an end-to-end A/B win, not the microbench alone.
3. **`qmvbench` throughput mode** — batch N calls per sync so per-kernel numbers reflect async pipeline conditions; required before trusting any future per-kernel delta.
4. **Thermal control for benchmarks** — log thermal state per run (`pmset -g therm`), shorter run blocks, or cooldowns, so absolute numbers become comparable across sessions.
5. **Attention-layer kernels and acceptance-rate work** — the remaining path toward the `tEvalMs` 24 ms / 30 tok/s target; weight-packing is measured out as a lever at this geometry.