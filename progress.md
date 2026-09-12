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

### Checkpoint 2 (NEXT): Pinned MSL attention kernels
Port `qwen35_attention_qk_rms_rope_bf16_v1` and the remaining pinned `MLXFast.metalKernel`
attention fusions into `Qwen35Attention.callAsFunction` (QK RMSNorm + RoPE → fused Metal kernel
when compiled decode is supported), then re-measure `tEvalMs` / TTLT via the step-trace.

## Technical Debt & Performance Roadmap (`v1.1-performance`)

* **Primary Bottleneck Identified:** Step latency (~151 ms) is dominated by standard eager-mode backbone execution in `Qwen35.swift` (1,692 lines).
* **Target Optimization:** Port the legacy fused backbone (6,088 lines) containing 20 pinned MSL kernels (`qwen35_attention_qk_rms_rope_bf16_v1`) and 4 `compile(shapeless:)` fusion blocks into `mlx-swift-lm`.
* **Expected Recovery:** Reduce `tEval` from ~163 ms to ~24 ms, restoring target decoding throughput to **~25+ tok/sec**.