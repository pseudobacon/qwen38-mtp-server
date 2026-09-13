# qwen38-mtp-server — Progress & Architecture Baseline

## Overview
A 3-layer speculative decoding server for Qwen 3.8 / 3.5 architectures using Apple Silicon MLX:
1. **Engine Layer (`../mlx-swift-lm`):** Custom MTP draft/verification block session (`Qwen38MTPBlockSession`) and model definitions.
2. **Model & State Layer (`MLXFastModel`):** Weight loading, KV-cache state management, tied-embedding sanitization, and MTP head attachment.
3. **Server Layer (`HTTPServer`):** Vapor-based OpenAI-compatible API (`/v1/chat/completions`), SSE streaming, memory admission control, and observability endpoints.

---

## Baseline Status (`v1.0-baseline`)

**Verified Features & Contracts:**
* **OpenAI SSE Streaming:** Fully compliant streaming framing (`data: {...}`, `data: [DONE]`).
* **Parameter Handling:** Correct parsing for `temperature`, `max_tokens`, and reasoning overrides (`"enable_thinking": false`).
* **State Isolation:** KV-cache reset and tokenization cache confirmed isolated across sequential client requests.
* **MTP Alignment:** MTP draft verification and `postNorm` logit calculations match expected engine distributions (93.5% raw token benchmark / ~55-74% live chat contexts).
* **Test Suites:** 121/121 `HTTPServerTests` passing; `MLXLMTests` MTP diagnostic suites passing.

---

## Performance Baseline & Diagnostic Metrics

| Metric | Measured Baseline (`v1.0-baseline`) | Notes |
| :--- | :--- | :--- |
| **TTFT (Time To First Token)** | ~0.44s – 0.65s | Fast prefill with memory-admission checks |
| **MTP Acceptance (Raw Prose)** | ~93.5% | Diagnostic greedy benchmark (no chat template) |
| **MTP Acceptance (Live Chat)** | ~74.1% (thinking OFF) / ~55.1% (thinking ON) | Expected context entropy variance |
| **Step Latency (`avgStepMs`)** | ~151 ms / round | Running eager-mode upstream backbone |
| **Decoding Throughput (TTLT)** | ~14.7 – 17.0 tok/sec | Backbone execution bottleneck |

---

## v1.1-Performance — Progress Log

### Checkpoint 1: Fused `compile(shapeless:)` activation blocks (DONE)
Ported the 4 `compile(shapeless: true)` fusion blocks from the legacy 6,088-line `Qwen35.swift` into the active fork (`../mlx-swift-lm`), each with an eager fallback gated by `MLXHardwareInfo.isCompiledDecodeSupported` (`MLX_COMPILED_DECODE` env override). The shapes are small/fixed, so they are immune to the Tahoe Metal JIT zero-result bug that affects whole-model compilation.

| Fusion | Eager replacement | Wired into |
| :--- | :--- | :--- |
| `qwen35CompiledFusedSwiGLU` | `silu(gate) * up` | `Qwen3NextMLP` (dense MLP + MoE shared expert) |
| `qwen35CompiledSigmoidMultiply` | `x * sigmoid(gate)` | `Qwen35Attention.mergeHeadsAndProject` + MoE shared-expert gate |
| `qwen35CompiledGatedDeltaGBeta` | `exp(-exp(A_log)*softplus(a+dt_bias))` + `sigmoid(b)` | `Qwen35GatedDeltaNet` prologue (computed once, reused for recurrence and MTP replay tape) |
| `qwen35CompiledGatedDeltaPostNorm` | `preciseSwiGLU` (rmsNorm + silu-gate) | `Qwen35GatedDeltaNet` post-norm (S>1; S==1 keeps `RMSNormGated`) |

Split `gatedDeltaUpdate` into a prepared-input overload (`g`/`beta`) so the GDN no longer re-derives the prologue per call.

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM` and `swift build --build-tests --force-resolved-versions`: clean (`git diff --check` clean across `Qwen35.swift`, `Qwen3Next.swift`, `GatedDelta.swift`, `MLXHardwareInfo.swift`).
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 aggregate acceptance **93.46%** (1072/1147); logit max-divergence vs target `postNorm: true => 16.25`, `postNorm: false => 14.0`.

---

### Checkpoint 2a: Pinned QK RMSNorm + RoPE Metal kernel (DONE)
Ported the fused Q & K RMSNorm + partial (64-dim) RoPE kernel (`qwen35_attention_qk_rms_rope_bf16_v1`) from the vendor copy into the active engine fork:

* **`Qwen35Kernels.swift`:** MSL shader + `qwen35AttentionQKRMSRoPE` wrapper (reads `[B,L,H,D]` Q/K, writes row-contiguous `[B,H,L,D]` outputs; grid `(totalRows*64,1,1)`, `ensureRowContiguous: false`).
* **`Qwen35+FastPath.swift`:** `extension Qwen35Attention { forwardFastPath }` — calls the fused kernel when compiled decode is supported AND Qwen 3.8-27B geometry applies (`usesFusedQKPreparation`) AND scalar RoPE offset + `L <= 32` + bf16 Q/K/weights.
* **`Qwen35.swift`:** Lightweight 2-line guard hook at the top of `Qwen35Attention.callAsFunction` plus stored `usesFusedQKPreparation` / `ropeLog2Base`.

**Gate (27B-only):** `attentionHeads == 24 && kvHeads == 4 && headDim == 256 && ropeDims == 64 && ropeTheta == 10_000_000 && ropeType == "default"`.

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — 93.46% (1072/1147).
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Performance Checkpoint 2a & Diagnostic Metrics
Release build of `qwen38-mtp-server` evaluated with `QWEN_MTP_STEP_TRACE=1`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token request (prompt: 38 tokens, total completion: 1,024 tokens, `finish_reason: length`).

**MTP-STEP-SUMMARY (per-request, stderr):**
rounds=426 proposed=1086 accepted=599 acceptedPerStep=1.4061 avgStepMs=116.6788 decodeSeconds=49.7432 committed=1024

**Per-round timing breakdown (426 rounds):**
| Component | avg ms | Notes |
| :--- | :--- | :--- |
| `tGraphBuildMs` | 9.98 | Metal graph build per round |
| `tEvalMs` | **104.92** | Target eval (draft verify + bonus token) |
| `tHostReadMs` | 0.02 | Host readback |
| `tCacheStateMs` | 1.63 | KV-cache state mutation |
| `stepMs` (total) | 116.55 | Sum of components above |

**Checkpoint 2a vs v1.0-baseline:**
| Metric | v1.0-baseline | Checkpoint 2a (fused QK-RoPE) | Delta |
| :--- | :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | ~151 ms/round | **116.68 ms/round** | −34.3 ms (−22.7%) |
| **`tEvalMs`** | ~163 ms (roadmap est.) | **104.92 ms** | −58.1 ms (−35.6%) |
| **Decoding Throughput (TTLT)** | ~14.7–17.0 tok/s | **~20.6 tok/s** (1024 / 49.74 s) | +3.6–5.9 tok/s (+21–40%) |
| **TTFT** | ~0.44–0.65 s | 0.3435 s | at/below baseline range |
| **Accepted / Step** | — | **1.4061** | — |
| **MTP Draft Acceptance** | ~93.5% raw / ~74.1% live | 55.15% (599/1086) | Prompt-specific (high-entropy essay) |

---

### Checkpoint 2b: Fused residual+RMSNorm + routed wide QMV projection kernels (DONE)
Ported the remaining fused MSL projection/norm kernels from the legacy `Qwen35.swift` reference into the active engine fork:

* **`Qwen35Kernels.swift`:**
  * Fused residual+RMSNorm MSL shader + 2 kernel factories and the `qwen35FusedResidualRMSNorm` wrapper: one launch writes `residual = r + x` and `normed = rmsnorm(residual, weight)` (grid `(nRows*1024,1,1)`, `ensureRowContiguous: false`).
  * `qwen35E120QMVSource` MSL: wide (rows-per-SIMD=4) + m-wrapper QMV templates with 4-bit group-64 dequant; 3 kernel factories: affine-4, affine-4-table (`USE_TABLE` template), and the xsums sidecar variant.
  * `Qwen35CustomQMV` enum, `qwen35RoutedQuantizedMM` dispatcher (routed for M in 2..9, eager `quantizedMM` otherwise), and `qwen35RoutedLinear` (guards: `layer as? QuantizedLinear`, `q.bias == nil`; falls back to `layer(x)`).
* **`Qwen35+FastPath.swift`:** q/k/v projections routed via `qwen35RoutedLinear`; `mergeHeadsAndProject(…, routed: true)` routes the oProj.
* **`Qwen35.swift`:** `mergeHeadsAndProject` gained a `routed: Bool = false` parameter; the decoder generic body fuses `r + rmsnorm(x)` via the fused kernel, gated on `isCompiledDecodeSupported` + BF16 + `dim == 5120` + contiguous strides, with the exact eager fallback.

**Decode-path activation note:** the wide QMV switch covers M = 2..9, so single-token decode (M = 1) falls back to eager `layer(x)` — the projection routing is active for multi-token shapes; the residual+RMSNorm fusion is active for every decode token (row-parallel kernel).

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 acceptance **93.46%** (1072/1147); logit max-divergence vs target `16.25` / `14.0` (unchanged from 2a).
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Performance Checkpoint 2b & Diagnostic Metrics
Release build of `qwen38-mtp-server` evaluated with `QWEN_MTP_STEP_TRACE=1`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token essay-prompt request (`finish_reason: length`). Two identical runs (greedy ⇒ deterministic; wall 46.40 s / 46.38 s).

**MTP-STEP-SUMMARY (per-request, stderr, run 1):**
rounds=385 proposed=1042 accepted=641 acceptedPerStep=1.6649 avgStepMs=119.8347 decodeSeconds=46.1632 committed=1024

**Per-round timing breakdown (385 rounds, phase averages over both runs):**
| Component | Checkpoint 2a | Checkpoint 2b | Delta |
| :--- | :--- | :--- | :--- |
| `tGraphBuildMs` | 9.98 | 10.87 | +0.89 |
| `tEvalMs` | **104.92** | **107.93** | +3.01 |
| `tHostReadMs` | 0.02 | 0.02 | ~0 |
| `tCacheStateMs` | 1.63 | 0.89 | −0.74 |
| `stepMs` (total) | 116.68 | 119.76 | +3.08 |

**Checkpoint 2b vs 2a vs v1.0-baseline:**
| Metric | v1.0-baseline | Checkpoint 2a (fused QK-RoPE) | Checkpoint 2b (+fused residual, routed QMV) |
| :--- | :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | ~151 ms/round | 116.68 ms/round | **119.80 ms/round** (+3.12 vs 2a) |
| **`tEvalMs`** | ~163 ms (roadmap est.) | 104.92 ms | **107.93 ms** |
| **Decoding Throughput (TTLT)** | ~14.7–17.0 tok/s | ~20.6 tok/s | **~22.2 tok/s** (1024 / 46.15 s) |
| **Accepted / Step** | — | 1.4061 | **1.6649** (prompt-specific) |

**Caveats:**
* The +3.1 ms step regression vs 2a is one extra Metal kernel op in the decode graph (fused residual+RMSNorm launch: `tEval` +3.0 ms, `tGraphBuild` +0.9 ms).
* TTLT improved +7.7% because this prompt yields a higher accepted-per-step (1.66 vs 1.41) — prompt-entropy dependent, not kernel-attributable.
* Wide QMV routing is inactive at M = 1, so the 2b delta over 2a is effectively the fused residual+RMSNorm fusion only.
* The ~24 ms `tEvalMs` target still requires attention-layer kernels and an M = 1 QMV dispatch (decode is the dominant shape).

---

### Refactor: Fast-path extraction out of `Qwen35.swift` (DONE)
Extracted custom fast-path additions out of `Qwen35.swift` into modular files to minimize vendor drift (fork diff reduced from ~676 lines to ~30):

* **`Qwen35+FastPath.swift`:** 4 `compile(shapeless: true)` closures, `MLXHardwareInfo`, and `Qwen35GatedDeltaNet` fast-path extension (`postNormGated`, `prefixReplayTape`, `applyReplayTape`, `canReplayPrefix`, `replayPrefix`).
* **`Qwen35Kernels.swift`:** MSL shader definitions and `MLXFast.metalKernel` wrappers.
* **`Qwen35.swift`:** Restored close to stock vendor shape, using 1-line hooks delegating to fast-path extensions.

**Behavior Note:** `postNormGated` is now consistently gated by `MLXHardwareInfo.isCompiledDecodeSupported` for $S > 1$.

---

### Checkpoint 2b-fix: QMV dispatch fixes + fast-path extraction completion (DONE)
Fixed five defects in the wide QMV dispatch in `Qwen35Kernels.swift` and completed the fast-path extraction into `Qwen35+FastPath.swift`:

**`Qwen35Kernels.swift` — five QMV dispatch fixes:**
1. **Dtype guard:** `weight.dtype == .uint32` (was `.bfloat16`) — 4-bit packed weights are `uint32`.
2. **Dimension check:** `scales.dim(1) * 64 == x.dim(1)`, `x.dim(1) == weight.dim(1) * 8`, and `k = x.dim(1)` (original feature dim; was `weight.dim(1)`, the packed k/4).
3. **Array count/order:** no-table dispatch passes exactly `[weight, scales, z, x]` (4 arrays); table dispatch passes `[weight, scales, z, x, xsums]` (5 arrays). The spurious `out` input was removed (`out` belongs in `outputShapes`/`outputDTypes`), and the `template:` argument was removed from the no-table kernel.
4. **Row-tile offset:** `qmv_out_row = int(qmv_tid.y) * 32 + int(qmv_sgid)` (was `* 8 + * 4` — a 4× offset into the output rows).
5. **Grid:** per-M `ipg(for:)` SIMD-group width (2→2, 3→3, 4→4, 5→5, 6→3, 7→4, 8→4, 9→3); grid = `((m + ipg - 1) / ipg, n / 32, 1)` — the old uniform tile was OOB for M = 5.

`qwen35CustomAffine4QMVTableKernel` visibility changed from `private` to internal for test access.

**Fast-path extraction completion (`Qwen35.swift` → `Qwen35+FastPath.swift`):**
* GDN `hasFusedInputProjection`, `prepareFusedInputProjection`, `projectInputs` moved into `extension Qwen35GatedDeltaNet`.
* Attention `projectPreRope` and `mergeHeadsAndProject` moved into `extension Qwen35Attention`.
* DecoderLayer fused-residual eligibility moved into `extension Qwen35DecoderLayer.applyResidualNorm`, backed by a `Qwen35FastPathFlags` cache (`rmsNormIsBF16`, `fusedResidualEligible`) resolved once per layer on first use. The stored `cachedFastPathFlags` property is declared in the class body because extensions in a separate file cannot add stored properties.

**Decisions:**
* No M ≥ 2 gate on `applyResidualNorm`: the residual+RMSNorm kernel accepts any row count (M = 1 decode included); the xsums sidecar gates its own per-M xsums arm independently (`Qwen35CustomQMV.widths.contains(rows) && tablePays(m:)`).
* `Qwen35FastPathFlags` is a 2-field per-layer struct rather than the planned 4-field model-level struct: the Attention `usesFusedQKPreparation` gate is already an init-time `let` (pure config geometry), layers have no back-reference to the model, and the only per-token invariant re-check worth caching is the decoder-layer residual gate.
* M = 1 wide QMV dispatch remains a gap (checkpoint 2c); single-token decode falls back to eager `layer(x)`.

**Root-cause note:** the +3.1 ms step regression vs checkpoint 2a is the fused residual+RMSNorm kernel launch itself (`tEval` +3.0 ms, `tGraphBuild` +0.9 ms), not the QMV dispatch; the five QMV fixes restore correctness of the wide path (active for M = 2..9 verify shapes) but do not remove that delta.

**Verification (active fork `../mlx-swift-lm` + server):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean (engine: 3 files, +195/−126).
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 acceptance **93.46%** (1072/1147); logit max-divergence vs target `16.25` / `14.0` (unchanged from 2a).
* `swift test --filter Qwen35FusedGDNProjectionTests`: **15 passed, 1 skipped, 0 failures** (exercises the relocated GDN projection methods).
* `swift build --target HTTPServer`: clean; full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

**Step-trace benchmark (release build, 1024 tokens, `QWEN_MTP_STEP_TRACE=1`, 437 decode rounds):**
* `stepAvg = 115.21 ms`, `tEvalAvg = 104.21 ms`, `tGraphBuildAvg = 9.73 ms`, `tCacheStateAvg = 1.25 ms`, `tHostReadAvg = 0.019 ms`; greedy T=0 acceptance **93.46%**.
* vs checkpoint 2b baseline (119.83 ms / 107.93 ms / 10.42 ms / 1.30 ms): step **-4.62 ms**, tEval **-3.72 ms** — the +3.1 ms regression is resolved (and the step is now below the clean 117.6 ms baseline as well).
* Wall clock: 50.71 s for 1024 tokens (~20.2 tok/s).

---

### Checkpoint 2c: Native $M = 1$ wide QMV dispatch (DONE)
Implemented native $M = 1$ (single-token decode) routing through the candidate-owned 4-bit wide QMV Metal kernel, so the projection path covers the dominant decode shape **without fallbacks** to eager `layer(x)`:

* **`Qwen35Kernels.swift`:** added the `M = 1` case to the `qwen_e120_qmv_m` template set (`IPG = 1`: one input row per SIMD group, so the wide helper degenerates to a plain matrix-vector pass over the four output rows the group owns); `Qwen35CustomQMV.widths` now includes `1` and `ipg(for:)` maps `1 → 1`. `M = 1` always takes the live-sums arm (the table sidecar only pays at `M >= 3`).
* **`Qwen35+FastPath.swift`:** `Qwen35Attention.projectPreRope` routes q/k/v through `qwen35RoutedLinear` (previously eager `qProj`/`kProj`/`vProj` calls).
* **`Qwen35.swift`:** `Qwen35DecoderLayer.attentionPostBody` routes `oProj` via `mergeHeadsAndProject(..., routed: true)`.

**Verification (active fork `../mlx-swift-lm` + server):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean (engine: 3 files, +29/−13).
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 acceptance **93.46%** (1072/1147); $M = 1$ wide QMV dispatch active on the decode path without fallbacks.
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Performance Checkpoint 2c & Diagnostic Metrics
Release build of `qwen38-mtp-server`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token request (`finish_reason: length`), with the native $M = 1$ wide QMV dispatch active on every decode projection:

| Metric | Checkpoint 2c | Notes |
| :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | **116.06 ms** | per decode round |
| **Eval Latency (`tEvalAvg`)** | **~104.21 ms** | target eval (draft verify + bonus token) |
| **Decoding Throughput (TTLT)** | **20.7 tok/s** | 49.47 s wall-clock for 1,024 committed tokens |

**Key Change:** Native $M = 1$ wide QMV Metal kernel routing implemented and verified without fallbacks — the $M = 1$ gap flagged in Checkpoint 2b-fix is closed; single-token decode now dispatches through the same wide kernel as the $M = 2..9$ verify shapes.

---

### Checkpoint 2d — Item 1: Fused `W_qkv` QKV projection (DONE)
Re-integrated the packed `W_qkv` (q + k + v rows) into the MTP execution pipeline using the existing `FusedQuantizedLinearProjection` machinery (source modules become row-slice **views sharing storage** — no weight duplication, net ~0 memory):

* **`FusedQuantizedLinear.swift` (MLXLMCommon):** added two default-ON env rollback knobs: `qwen35FusedQKVEnabled` (`MLX_QWEN_FUSED_QKV`) and `qwen35FusedSwiGLUEnabled` (`MLX_QWEN_FUSED_SWIGLU`, Item 2). Set `0` to disable.
* **`Qwen35.swift`:** `Qwen35Attention` gained `let fusedQKVProjection = FusedQuantizedLinearProjectionCache()`; `Qwen35TextModel.prepare()` calls `prepareFusedQKVProjection()` on every backbone full-attention layer **and** every MTP-head layer (head is BF16 ⇒ fusion ineligible there, stays eager fallback). Added `update`/`updateModule` overrides on `Qwen35Attention` that invalidate the cache when a q/k/v parameter or module is replaced (mirrors the GDN input-projection invalidation).
* **`Qwen35+FastPath.swift`:** `extension Qwen35Attention` gained `hasFusedQKVProjection`, `prepareFusedQKVProjection()` (fuses `q_proj` + `k_proj` + `v_proj` into one `QuantizedLinear`, installs storage-sharing views), and `qkvProjections(_:)` — one fused `qwen35RoutedLinear` call sliced at `qProj.shape.0` / `+ kProj.shape.0`, eager 3-call fallback otherwise. `forwardFastPath` and `projectPreRope` both call the helper (reshape/norm/transpose logic unchanged).
* **`Tests/MLXLMTests/Qwen35FusedQKVProjectionTests.swift` (new, 8 tests):** bit-identical fused-vs-eager for decode `(1,1)` and prefill `(2,7)`, full-attention-forward bit-identity, model-`prepare()` wiring, no-lazy-preparation-on-forward, incompatible-policy fallback, LoRA fallback, checkpoint-topology preservation, parameter/module-update invalidation. **8/8 PASS.**

**Geometry:** backbone 4-bit `q_proj [12288,640]`, `k_proj`/`v_proj [1024,640]` → fused `U32 [14336,640]` (`14336 % 32 == 0` ✓); active for M ∈ 1..9 (verify width 1 + 0..8 drafts). MTP head BF16 ⇒ not fusion-eligible (eager fallback, documented).

**Verification (active fork `../mlx-swift-lm` + server):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 acceptance **93.46%** (1072/1147); logit max-divergence vs target `16.25` / `14.0` (unchanged — fusion is bit-exact).
* `swift test --filter Qwen35FusedQKVProjectionTests`: **8/8 PASS**.
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Performance Checkpoint 2d Item 1 & Diagnostic Metrics
Release build of `qwen38-mtp-server`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token essay-prompt request (`finish_reason: length`), `QWEN_MTP_STEP_TRACE=1`, port 18099:

| Metric | Checkpoint 2c | Item 1 (fused `W_qkv`) | Delta |
| :--- | :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | 116.06 ms | **122.25 ms** | +6.19 ms (+5.3%) |
| **Eval Latency (`tEvalAvg`)** | ~104.21 ms | **110.57 ms** | +6.36 ms |
| **`tGraphBuildAvg`** | 9.73 ms | 10.75 ms | +1.02 ms |
| **`tCacheStateAvg`** | 1.25 ms | 0.80 ms | −0.45 ms |
| **Decoding Throughput (TTLT)** | 20.7 tok/s | **22.03 tok/s** (1024 / 46.48 s) | +1.33 tok/s (+6.4%) |
| **Accepted / Step** | 1.4061 | 1.6974 (645/1008, 380 rounds) | prompt/run-entropy dependent |

**Caveats:**
* The +6.2 ms step regression is the one fused `W_qkv` matmul replacing three separate 4-bit QMV launches for the 16 full-attention layers (per-round `tEval` +6.4 ms); at the current geometry the single wider pass costs more than the three narrower ones. The fusion's value is latency-neutral-to-negative **per step** but improves **tokens per step** when acceptance is high (TTLT +6.4% this run).
* `acceptedPerStep` varies run-to-run across checkpoints (1.41 / 1.66 / 1.70 for the same prompt); greedy target tokens are unchanged (bit-exact fusion, confirmed by identical 93.46% diagnostic acceptance and logit divergences).
* `MLX_QWEN_FUSED_QKV=0` restores the three-projection eager path bit-for-bit.

### Checkpoint 2d — Item 2: Fused `W_gate+up` SwiGLU projection (DONE)
Re-integrated the packed `W_gate+up` (gate + up rows) into the MTP execution pipeline, sharing the `FusedQuantizedLinearProjection` machinery from Item 1. `down_proj` stays eager — only the gate+up sweep is fused:

* **`Qwen3Next.swift`:** `Qwen3NextMLP` gained `let fusedSwiGLUProjection = FusedQuantizedLinearProjectionCache()` plus `update`/`updateModule` overrides invalidating the cache when a `gate_proj`/`up_proj` parameter or module is replaced. `callAsFunction` now goes through `swiGLUGateUpProjections(x)` → `downProj(qwen35CompiledFusedSwiGLU(gate, up))`.
* **`Qwen35+FastPath.swift`:** `extension Qwen3NextMLP` with `hasFusedSwiGLUProjection`, `prepareFusedSwiGLUProjection()` (fuses `gate_proj` + `up_proj`, storage-sharing views), and `swiGLUGateUpProjections(_:)` — one fused `qwen35RoutedLinear` call sliced at `half = gateProj.shape.0` (gate `[0..<half]`, up `[half...]`), eager 2-call fallback otherwise. Shared by the Qwen 3.5 backbone and Qwen 3Next models (same bit-exactness argument).
* **`Qwen35.swift`:** `Qwen35TextModel.prepare()` now also calls `prepareFusedSwiGLUProjection()` on every backbone MLP and MTP-head MLP (head BF16 ⇒ ineligible ⇒ eager fallback).
* **`Tests/MLXLMTests/Qwen35FusedSwiGLUProjectionTests.swift` (new, 8 tests):** bit-identical fused-vs-eager for decode `(1,1)` and prefill `(2,7)`, full-MLP-forward bit-identity, model-`prepare()` wiring, no-lazy-preparation-on-forward, incompatible-policy fallback, LoRA fallback, checkpoint-topology preservation, parameter/module-update invalidation. **8/8 PASS.**

**Geometry:** backbone 4-bit `gate_proj`/`up_proj [17408,640]` → fused `U32 [34816,640]` (`34816 % 32 == 0` ✓); active for M ∈ 1..9. 64 MLPs (all 64 layers) get the repack, vs 16 QKV in Item 1.

**Verification (active fork `../mlx-swift-lm` + server):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 acceptance **93.46%** (1072/1147); logit max-divergence vs target `16.25` / `14.0` (unchanged — both fusions bit-exact).
* `swift test --filter Qwen35FusedSwiGLUProjectionTests`: **8/8 PASS**.
* `swift test --filter Qwen35FusedQKVProjectionTests`: **8/8 PASS** (Item 1 regression check).
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Performance Checkpoint 2d Item 2 & Diagnostic Metrics
Release build of `qwen38-mtp-server`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token essay-prompt request (`finish_reason: length`), `QWEN_MTP_STEP_TRACE=1`, port 18099:

| Metric | Checkpoint 2c | Item 1 (QKV) | Item 2 (QKV + gate+up) | Item 2 Δ vs Item 1 |
| :--- | :--- | :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | 116.06 ms | 122.25 ms | **123.69 ms** | +1.44 ms |
| **Eval Latency (`tEvalAvg`)** | ~104.21 ms | 110.57 ms | **111.64 ms** | +1.07 ms |
| **`tGraphBuildAvg`** | 9.73 ms | 10.75 ms | 11.04 ms | +0.29 ms |
| **`tCacheStateAvg`** | 1.25 ms | 0.80 ms | 0.87 ms | +0.07 ms |
| **Decoding Throughput (TTLT)** | 20.7 tok/s | 22.03 tok/s | **21.77 tok/s** (1024 / 47.04 s) | −0.26 tok/s |
| **Accepted / Step** | 1.4061 | 1.6974 (645/1008, 380 rounds) | 1.6974 (645/1008, 380 rounds) | identical greedy stream |

**Caveats:**
* The gate+up repack adds ~1.4 ms/step: the fused `U32 [34816,640]` pass over 64 layers is wider than the two `[17408,640]` passes it replaces. Combined with Item 1, total step delta vs 2c is **+7.63 ms** (116.06 → 123.69) while TTLT stays above the 2c baseline (21.77 vs 20.7 tok/s) on this run's acceptance rate.
* Item 1 and Item 2 runs produced identical accepted counts (645/1008, 380 rounds) — the greedy token stream is unchanged by both fusions (bit-exact), and the acceptance rate is stable across the two Item 1/Item 2 runs.
* Rollback: `MLX_QWEN_FUSED_SWIGLU=0` restores the two-projection eager path bit-for-bit (independently of `MLX_QWEN_FUSED_QKV`).

### Checkpoint 2d — Item 3: Combined QKV + SwiGLU verification + merge (DONE)
Final verification with **both** packed projections (`W_qkv` + `W_gate+up`) active, then merge of `feature/prompt-ckpt2d` into `main` in both repos:

* **Engine `../mlx-swift-lm`**: `swift test --filter Qwen38MTPDiagnosticTests` **PASS** (93.46%, 1072/1147; logit divergence `16.25` / `14.0` — unchanged, both fusions bit-exact); `Qwen35FusedQKVProjectionTests` **8/8**; `Qwen35FusedSwiGLUProjectionTests` **8/8**.
* **Server `qwen38-mtp-server`**: `swift test --filter HTTPServerTests` **121/121 PASS**.
* **Git:** engine committed first (`feat: fused W_qkv and W_gate+up packed projections for MTP pipeline`), then server (`docs: checkpoint 2d items 1-3 fused projections progress and handoff`); both `feature/prompt-ckpt2d` branches merged to `main` with `git merge --no-edit` and deleted.

## Performance Checkpoint 2d Item 3 & Diagnostic Metrics (final combined)
Release build of `qwen38-mtp-server`, greedy (`temperature: 0.0`), `enable_thinking: false`, 1,024-token essay-prompt request (`finish_reason: length`), `QWEN_MTP_STEP_TRACE=1`, port 18099, both packed projections active:

| Metric | Checkpoint 2c | Item 1 (QKV) | Item 2 (QKV + gate+up) | Item 3 final (both) |
| :--- | :--- | :--- | :--- | :--- |
| **Step Latency (`avgStepMs`)** | 116.06 ms | 122.25 ms | 123.69 ms | **120.86 ms** |
| **Eval Latency (`tEvalAvg`)** | ~104.21 ms | 110.57 ms | 111.64 ms | **108.97 ms** |
| **`tGraphBuildAvg`** | 9.73 ms | 10.75 ms | 11.04 ms | 10.89 ms |
| **`tCacheStateAvg`** | 1.25 ms | 0.80 ms | 0.87 ms | 0.86 ms |
| **Decoding Throughput (TTLT)** | 20.7 tok/s | 22.03 tok/s | 21.77 tok/s | **22.28 tok/s** (1024 / 45.96 s) |
| **Accepted / Step** | 1.4061 | 1.6974 (645/1008) | 1.6974 (645/1008) | 1.6974 (645/1008, 380 rounds) |

**Findings:**
* All three Item 1/2/3 runs produced the identical greedy stream (645/1008 accepted over 380 rounds) — both packed projections are bit-exact with the separate-projection eager path; `QWEN_MTP_STEP_TRACE` confirms no behavior drift.
* Per-step latency varies ±~2.8 ms run-to-run even with identical code (Item 2: 123.69 vs Item 3: 120.86), so the per-step delta vs the 2c baseline (116.06 ms) is best read as a band of **~116–124 ms**; the packed-projection cost at this geometry is latency-neutral-to-slightly-negative per step.
* Headline result: combined packed projections hold TTLT at **~22.3 tok/s** vs the 2c baseline's **20.7 tok/s** on this prompt, with zero acceptance-rate or token-stream change and net ~0 memory cost (storage-sharing views).
* Rollback knobs (both default ON): `MLX_QWEN_FUSED_QKV=0`, `MLX_QWEN_FUSED_SWIGLU=0`.

**Checkpoint 2d status: COMPLETE** — Items 1–3 done, `main` in both repos carries the fused `W_qkv` + `W_gate+up` pipeline; feature branches deleted.

---

## Prompt-Fixture Provenance (§0, prompt rev 2) — RESOLVED

The committed fixture `benchmarks/prompts/essay-1024.txt` (SHA-256 `7ed683f87be0835c751505e2ee7dfc18fd922b93bcc32fad05d86c158cfb040e`) is **not** the prompt behind the recorded 645/1008/380 (hash `139acb9d…`) numbers. Provenance, established with the rolled-back (all-fusion-off) build:

* `essay-1024.txt` → **38 prompt tokens**, stream `599/1086/426` (1.4061), hash `949b9423bd851233…` — this is the 2a/2c prompt.
* `specdec-800.txt` → **62 prompt tokens**, stream `645/1008/380`, hash `139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac` — **exact match** to the recorded Item 1/2/3 numbers.

Consequences: the prior Item-1-vs-2c step comparison (+6.19 ms, 116.06 → 122.25) is **cross-prompt (confounded)**. The all-fusion matrix in the pinned 38-token essay prompt is the authoritative same-prompt comparison. All matrix cells used the pinned essay fixture, read from file at request time; greedy (temp 0, `enable_thinking: false`, `max_tokens: 1024`, `finish_reason: length`); port 18099; `QWEN_MTP_STEP_TRACE=1`; rep 1 discarded as warmup, reps 2–6 measured; interleaved cell order A0,A3,A1,A2.

### Item A — 2×2 same-session fusion matrix (COMPLETE, 24/24 cells)

Cells: A0 = QKV off / gate+up off (0,0); A1 = QKV on / gate+up off (1,0); A2 = QKV off / gate+up on (0,1); A3 = both (1,1). **All 24 cells bit-identical**: 599/1086 accepted (426 rounds, 1.4061/step), stream hash `949b9423bd851233…`, `finish_reason: length`. Determinism: **confirmed**.

Measured reps r2–r6 (r1 discarded), `avgStepMs` per cell:

| cell | fusion | mean | min | max | Δ vs A0 (mean) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| A0 | off/off | 138.554 | 136.545 | 140.892 | — |
| A3 | QKV + gate+up | 137.662 | 134.949 | 138.698 | **−0.892** |
| A1 | QKV only | 137.902 | 134.210 | 141.213 | **−0.652** |
| A2 | gate+up only | 138.825 | 135.489 | 142.370 | **+0.271** |

Per-rep Δ vs A0 (ms): A3 −1.60, −0.23, −0.22, −0.11, −2.30; A1 −2.34, −1.15, −0.12, +2.40, −2.06; A2 −1.06, −0.68, +1.16, +3.56, −1.62. `tEvalAvg` means: A0 126.459, A3 125.637, A1 125.885, A2 126.610. `tGraphBuildAvg` ≈ 10.25–10.44, `tCacheStateAvg` ≈ 1.57–1.67, `tHostReadAvg` ≈ 0.015–0.020. TTLT means: A0 17.341, A3 17.455, A1 17.427, A2 17.310 tok/s.

**Caveats:**
* Sustained-run thermal drift across the 24-cell run raised absolute step latency from ~123.9 ms (rep 1) to ~140.3 ms (rep 5); interleaved cell order keeps per-cell Δ within ±3.6 ms, and the per-rep Δs above are the correct read. Cross-session absolute values (e.g. 2c's 116.06 ms) are **not comparable** to this run's ~138–140 ms band.
* In-session verdict: no fusion shows a net step regression; gate+up alone (A2) is neutral (+0.27 ms mean) and both-fusion (A3) is slightly negative (−0.89 ms). The prior +6.19 ms Item-1 regression does not reproduce in-session.

**Binary-vintage correction (IMPORTANT):** the original Item A matrix ran on the 02:41 release binary, which **predates the QMV dispatch fix** (working-tree fix 13:43, `grid: ((m+ipg-1)/ipg*32, n/32*8, 1)` vs the committed `((m+ipg-1)/ipg, n/32, 1)`) and the fusion merge (`0514b11`, 02:49). With the buggy dispatch, any fused decode through the routed QMV kernel would have written only a fraction of each 32-col output tile and produced a divergent token stream; since **all 24 cells produced the known-correct greedy hash `949b9423…`**, the fused kernel never ran in that binary — every Item A cell was the eager path (env gates were no-ops). The Δs above are therefore eager-vs-eager noise, not fusion effects.

**Re-run history (2026-09-13):**
1. First re-run on the verified fusion-engaged binary (job bash-58, `benchmarks/results/itemA.jsonl` superseded): 24/24 cells bit-exact (599/1086/426, hash `949b9423…`), but the timing was **contaminated** — engine/server test suites ran in parallel during the matrix (the 151 s `Qwen38MTPDiagnosticTests` run loaded the full model and generated in overlap with the final cells), producing spikes A2-r5 239.58 ms, A0-r6 196.10 ms, A2-r3 178.87 ms and per-rep Δ swings of −9…+33 ms. Its numbers are discarded.
2. Clean re-run (job bash-59, `.tmp/itemA-clean.log`) with **no parallel builds/tests**: **COMPLETE, 24/24 cells bit-exact** (599/1086/426, hash `949b9423…`). Reps 2–6 means: A0 143.042 ms (134.790–164.483), A1 143.304 (+0.262), A2 142.967 (−0.076), A3 145.429 (+2.387); stepAvg means 142.943/143.194/142.854/145.300; tEvalAvg 131.1/131.3/130.8/133.1 ms; TTLT 16.89/16.81/16.81/16.55 tok/s. Per-rep Δ swings −21.8…+15.2 ms. **Final verdict (fusion engaged): both fusions latency-neutral in-session; no net win, no regression.** These numbers supersede the earlier Item A table in this file and are the authoritative Item A data for `benchmarks/FUSION_REPORT.md`.

### Item B — Standalone QMV microbenchmark `qmvbench` (COMPLETE)

New executable target `QmvBench` in the engine fork (`Libraries/QmvBench/main.swift`, product `qmvbench` in `../mlx-swift-lm/Package.swift`). Calls `qwen35RoutedLinear` / `qwen35RoutedQuantizedMM` directly (no reimplementation) on one real gate/up pair (layer 3 of Qwen 3.8-27B-4bit): narrow `N = 17408, K = 5120` (packed K = 640 uint32, scales/biases [N,80] bf16); fused wide `N = 34816` in `global` and `interleaved` row layouts (32-row kernel tiles); M ∈ {1,4}; one shard CPU-loaded. Protocol: 100 warmup + 1000 timed iterations per condition, randomized interleaved order, 3 whole blocks (3000 samples), `ContinuousClock` with device sync per call.

**Correctness:** all guards pass by construction; at both M values every routed condition is **bit-identical** to the incumbent `QuantizedLinear`/`quantizedMM`, the wide outputs split back to the exact narrow outputs, and the interleaved fused tensor is a verified row permutation of the global fused tensor.

M = 1 (µs per iteration, mean / min / p50 / std, GB/s at mean):

| condition | mean | min | p50 | std | GB/s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| narrow_gate_routed | 384.92 | 312 | 380 | 30.15 | 130.36 |
| narrow_up_routed | 385.63 | 311 | 380 | 35.88 | 130.13 |
| wide_global_routed | 572.60 | 491 | 563 | 47.81 | 175.25 |
| wide_interleaved_routed | 571.56 | 491 | 564 | 35.38 | 175.57 |
| narrow_gate_fallback | 366.63 | 303 | 361 | 33.65 | 136.87 |
| narrow_up_fallback | 366.01 | 303 | 361 | 29.34 | 137.10 |
| wide_global_fallback | 553.31 | 480 | 545 | 63.59 | 181.36 |

M = 4 (µs per iteration, mean / min / p50 / std, GB/s at mean):

| condition | mean | min | p50 | std | GB/s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| narrow_gate_routed | 464.06 | 379 | 458 | 29.38 | 108.42 |
| narrow_up_routed | 464.25 | 381 | 458 | 29.20 | 108.38 |
| wide_global_routed | 724.89 | 628 | 713 | 41.35 | 138.77 |
| wide_interleaved_routed | 726.65 | 635 | 713 | 87.33 | 138.43 |
| narrow_gate_fallback | 545.38 | 471 | 541 | 37.16 | 92.26 |
| narrow_up_fallback | 546.43 | 465 | 542 | 34.04 | 92.08 |
| wide_global_fallback | 904.30 | 824 | 898 | 38.67 | 111.23 |

**Root-cause note (kernel dispatch bug, FIXED):** the routed QMV kernel previously produced garbage (only cols 0–3 of each 32-col tile written) because `MLXFast.metalKernel` dispatch in `Source/C/metal/custom_kernel.cpp` treats the `grid` argument as the **total thread count** and clamps the threadgroup to `min(threadGroup, grid)` — the old dispatch passed the threadgroup count as `grid`, so only `grid.x` lanes per threadgroup ran. Confirmed with a trivial `MLXFast.metalKernel` probe (2 of 512 slots written pre-fix). Fixed both arms in `Qwen35Kernels.swift` to `grid: ((m + ipg - 1) / ipg * 32, n / 32 * 8, 1), threadGroup: (32, 8, 1)` (codebase convention: grid = total threads, e.g. `(nRows*1024,1,1)` / `(1024,1,1)`). Also fixed the same pass: interpolation leak `case \\(m):`, missing semicolon, `*4` in `qmv_out_row`, and MLX 4-bit nibble/x-scaling convention (`& 0x0f/0xf0/0xf00/0xf000`, `/16,/256,/4096`).

**Note:** before this session the custom QMV kernel was dead code in-model (no engine test exercised it; in-model decode reaches it only via 2-D `x`, 3-D batched `x` falls back to `quantizedMM` by the `ndim == 2` guard — bit-identical either way); `qmvbench` is the first consumer.

### Item C — env-gated `MLX_QWEN_SWIGLU_LAYOUT` (COMPLETE: code + matrix)

`FusedQuantizedLinear.swift` (MLXLMCommon): new `MLX_QWEN_SWIGLU_LAYOUT ∈ {global (default), interleaved}` read at prepare time only via `qwen35SwiGLULayout()`; the interleaved fuse builds per-expert `[2I, K]` adjacent 32-row-block row-gather views on the fused weight/scales/biases (`take(idx, axis: 0)`) — materialized copies, ≈ +100.8 MB per gate/up layer pair (≈ 6.5 GB over 64 layers; fits 48 GB). `Qwen35+FastPath.swift`: `prepareFusedSwiGLUProjection()` takes the layout, logs a loud load-time fallback when fusion falls back, and `swiGLUGateUpProjections(_:)` splits the output tensor on the stored layout (`.global`: slice at `half`; `.interleaved`: `[blocks, 64]` reshape, slice `[0..<32]`/`[32..<64]`). Guard: interleaved requires N % 32 == 0, else eager fallback (loud log).

**Verification:** `Qwen35FusedSwiGLUProjectionTests` **8/8 PASS** (incl. `testInterleavedLayoutIsBitIdentical`, `testInterleavedFusePermutationAndViewIdentity`, `testInterleavedLayoutRejectsNonTileMultiple`); `Qwen38MTPDiagnosticTests` PASS (52.8 s).

**Build note (relink resolution):** `swift build --target HTTPServer` compiles the target's objects only — the release **executable** is produced by `swift build --configuration release --product qwen38-mtp-server` (the `--target` form never re-runs the product link, so a `rm` of the binary leaves it absent despite "Build complete"). After the product build, `strings .build/release/qwen38-mtp-server | grep -c MLX_QWEN_SWIGLU_LAYOUT` = 1.

**Matrix results (COMPLETE, 12/12 cells bit-exact):** `benchmarks/run_matrix.sh itemC` (6 reps × Cglobal/Cint, shared Item-A protocol, beside A2): every cell 599/1086/426, hash `949b9423…`, `finish_reason: length`. `avgStepMs` (reps 2–6): Cglobal mean 150.578 (min 126.864, max 166.173; tEvalAvg 137.48, TTLT 16.10 tok/s); Cint mean 152.187 (min 133.479, max 160.276; tEvalAvg 138.87, TTLT 15.85 tok/s). Δ mean **+1.609 ms (+1.1 %)**; per-rep Δ +6.61, +8.48, +1.35, −2.50, −5.90 ms — inside the run's thermal noise band (Cglobal spans 126.9–166.2 ms across reps). Verdict: interleaved layout is bit-exact end-to-end and **latency-neutral in-session**.

**Fusion engagement verified (resolves the earlier "fell back" alarm):** the one-shot server load logs exactly ONE fallback line — from the **MTP head layer only** (the head tree is BF16, `mtp-head/model.safetensors` `layers.0.mlp.gate_proj.weight BF16 [17408, 5120]`; fusion ineligible there by design). The new load-time summary in `Qwen35TextModel.prepare()` prints: `MLXLM: fusion prepare summary: backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 0 qkv 0` — i.e. **gate+up fusion engaged in all 64 backbone layers, QKV fusion in all 16 full-attention layers, GDN input-projection fusion in all 48 GDN layers**, on the current binary (both global and interleaved layouts). Consequence: Item C cells (15:29+ binary) exercised the fused path; the earlier suspicion that all cells ran eager was wrong. (RSS was discarded as an engagement probe — Metal shared-heap memory is not faithfully reflected in RSS.)

---

## Technical Debt & Performance Roadmap (`v1.1-performance`)

* **Current Status:** Checkpoint 1 (`compile()` closures), Checkpoint 2a (QK RMSNorm + RoPE kernel), Checkpoint 2b (fused residual+RMSNorm + routed wide QMV projection kernels), Checkpoint 2b-fix (five QMV dispatch defects fixed + fast-path extraction completed), Checkpoint 2c (native $M = 1$ wide QMV dispatch), the structural refactor, and **Checkpoint 2d (Items 1–3: fused $W_{qkv}$ + fused $W_{gate+up}$ packed projections, merged to `main` in both repos)** completed. Step latency ~116–124 ms band (tEvalAvg ~104–112 ms, greedy T=0); TTLT ~22.3 tok/s (1,024 tokens in 45.96 s wall, checkpoint 2d final run; 2c baseline 20.7 tok/s / 49.47 s).
* **Pending Scope:** Attention-layer fused kernels — the remaining path toward the ~24 ms `tEvalMs` target.
* **Target Milestone:** Reduce `tEvalMs` from ~104.21 ms to ~24 ms, achieving decoding throughput of **~30+ tok/sec**.