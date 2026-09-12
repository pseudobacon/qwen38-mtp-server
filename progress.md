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
Ported the 4 `compile(shapeless: true)` fusion blocks from the legacy 6,088-line `Qwen35.swift`
into the active fork (`../mlx-swift-lm`), each with an eager fallback gated by
`MLXHardwareInfo.isCompiledDecodeSupported` (`MLX_COMPILED_DECODE` env override). The shapes are
small/fixed, so they are immune to the Tahoe Metal JIT zero-result bug that affects whole-model
compilation.

| Fusion | Eager replacement | Wired into |
| :--- | :--- | :--- |
| `qwen35CompiledFusedSwiGLU` | `silu(gate) * up` | `Qwen3NextMLP` (dense MLP + MoE shared expert) |
| `qwen35CompiledSigmoidMultiply` | `x * sigmoid(gate)` | `Qwen35Attention.mergeHeadsAndProject` + MoE shared-expert gate |
| `qwen35CompiledGatedDeltaGBeta` | `exp(-exp(A_log)*softplus(a+dt_bias))` + `sigmoid(b)` (previously computed ×2) | `Qwen35GatedDeltaNet` prologue — computed once, reused for the recurrence and the MTP replay tape |
| `qwen35CompiledGatedDeltaPostNorm` | `preciseSwiGLU` (rmsNorm + silu-gate) | `Qwen35GatedDeltaNet` post-norm (S>1; S==1 keeps `RMSNormGated`) |

Also split `gatedDeltaUpdate` into a prepared-input overload (`g`/`beta`) so the GDN no longer
re-derives the prologue per call.

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM` and `swift build --build-tests --force-resolved-versions`: clean
  (`git diff --check` clean; 4 files: `Qwen35.swift`, `Qwen3Next.swift`, `GatedDelta.swift`,
  new `MLXLMCommon/MLXHardwareInfo.swift`).
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 aggregate acceptance
  **93.46%** (1072/1147); logit max-divergence vs target `postNorm: true => 16.25`,
  `postNorm: false => 14.0` (finite, aligned). Confirms the `compile(shapeless:)` fusions
  (incl. `MLXFast.rmsNorm` in the post-norm) are bit-exact at runtime.
* **tEvalMs / TTLT step-trace measurement: PENDING** — requires launching the server with
  `QWEN_MTP_STEP_TRACE=1` + a 1,024-token request (Checkpoint 3). The full ~163→24 ms recovery
  is expected to need both Checkpoint 1 (this) **and** Checkpoint 2 (20 pinned MSL kernels).

### Checkpoint 2a: Pinned QK RMSNorm + RoPE Metal kernel (`qwen35_attention_qk_rms_rope_bf16_v1`) (DONE)
Ported the fused Q & K RMSNorm + partial (64-dim) RoPE kernel from the `qwen-mtp-server`
vendor copy into the active fork:

* **`Qwen35Kernels.swift`:** full MSL shader + `qwen35AttentionQKRMSRoPE` wrapper (reads
  `[B,L,H,D]` Q/K, writes row-contiguous `[B,H,L,D]` outputs; grid `(totalRows*64,1,1)`,
  `ensureRowContiguous: false`).
* **`Qwen35+FastPath.swift`:** new `extension Qwen35Attention { forwardFastPath }` — calls the
  fused kernel when compiled decode is supported AND the Qwen 3.8-27B geometry applies
  (`usesFusedQKPreparation`) AND a scalar RoPE offset + `L <= 32` + bf16 Q/K/weights;
  otherwise the exact eager `projectPreRope` + `applyRotaryPosition` path (bit-identical to
  the vendor-shaped `callAsFunction`).
* **`Qwen35.swift`:** 2-line guard hook at the top of `Qwen35Attention.callAsFunction`
  (`if MLXHardwareInfo.isCompiledDecodeSupported { return forwardFastPath(...) }`) plus stored
  `usesFusedQKPreparation` / `ropeLog2Base`. Eager path unchanged.

Gate (27B-only): `attentionHeads == 24 && kvHeads == 4 && headDim == 256 && ropeDims == 64
&& ropeTheta == 10_000_000 && ropeType == "default"` (ropeType from
`ropeScaling["type"] ?? ["rope_type"]`, default `"default"`).

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — greedy T=0 aggregate acceptance
  93.46% (1072/1147); logit max-divergence vs target `postNorm: true => 16.25`,
  `postNorm: false => 14.0` (finite, aligned).
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

Remaining Checkpoint 2 scope: the other pinned attention fusions (beyond QK RMSNorm + RoPE)
are still to be ported; `tEvalMs` / TTLT step-trace re-measurement remains pending (Checkpoint 3).

### Refactor: fast-path extraction out of `Qwen35.swift` (DONE)
Moved all custom fast-path additions out of `Qwen35.swift` into dedicated files to keep the
vendor-shaped file close to upstream (fork diff shrank from ~676 lines to ~30):

* **New `Libraries/MLXLLM/Models/Qwen35+FastPath.swift`:** the four `compile(shapeless: true)`
  fusion closures + gated wrappers (`qwen35CompiledFusedSwiGLU`, `qwen35CompiledSigmoidMultiply`,
  `qwen35CompiledGatedDeltaGBeta`, `qwen35CompiledGatedDeltaPostNorm`), `MLXHardwareInfo` (moved
  from `MLXLMCommon`), and a `Qwen35GatedDeltaNet` fast-path extension
  (`postNormGated`, `prefixReplayTape`, `applyReplayTape`, `canReplayPrefix`, `replayPrefix`).
* **New `Libraries/MLXLLM/Models/Qwen35Kernels.swift`:** designated home for the pinned MSL
  shaders / `MLXFast.metalKernel` defs. `Qwen35.swift` currently contains no raw MSL shader
  code (the Checkpoint 2 kernels will be added here), so the file is a documented placeholder.
* **`Qwen35.swift`:** inline fast-path logic in `forward`/`callAsFunction` replaced with
  1-line hooks (`qwen35CompiledGatedDeltaGBeta`, `prefixReplayTape(...)`,
  `postNormGated(out, gate: z, sequence: S)`, `applyReplayTape(replayTape, to: cache)`);
  fusion section and replay-prefix methods removed; the stored `postNorm` property stays
  (stored properties cannot live in an extension).
* **Deleted `Libraries/MLXLMCommon/MLXHardwareInfo.swift`** (content moved to MLXLLM;
  no other target referenced it).

**Behavior note:** `postNormGated` is now gated by `MLXHardwareInfo.isCompiledDecodeSupported`
for S>1 (previously the compiled post-norm node ran for S>1 regardless of the
`MLX_COMPILED_DECODE` env var; the eager `RMSNormGated` fallback is bit-identical, so this
makes the env-var opt-out complete and consistent with the other three fusions).

**Verification (active fork `../mlx-swift-lm`):**
* `swift build --target MLXLLM`: clean; `git diff --check`: clean.
* `swift test --filter Qwen35GDNDecodeBitwiseTests`: **PASS** (decodeConv bit-pinned).
* `swift test --filter Qwen35FusedGDNProjection`: 15 tests, 1 skipped, 0 failures
  (full GDN forward bit-identical for decode/prefill, VLM forward, checkpoint topology).
* Sanitize / CompiledDecodeLifecycle / MRoPE / DirectExpertReduction suites: 13 tests, 0 failures.
* Full server suite `swift test --filter HTTPServerTests`: **121/121 PASS**.

## Technical Debt & Performance Roadmap (`v1.1-performance`)

* **Primary Bottleneck Identified:** Step latency (~151 ms) is dominated by standard eager-mode backbone execution in `Qwen35.swift` (1,692 lines).
* **Target Optimization:** Port the legacy fused backbone (6,088 lines) containing 20 pinned MSL kernels (`qwen35_attention_qk_rms_rope_bf16_v1`) and 4 `compile(shapeless:)` fusion blocks into `mlx-swift-lm`.
* **Expected Recovery:** Reduce `tEval` from ~163 ms to ~24 ms, restoring target decoding throughput to **~25+ tok/sec**.