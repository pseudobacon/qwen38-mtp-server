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

## Technical Debt & Performance Roadmap (`v1.1-performance`)

* **Primary Bottleneck Identified:** Step latency (~151 ms) is dominated by standard eager-mode backbone execution in `Qwen35.swift` (1,692 lines).
* **Target Optimization:** Port the legacy fused backbone (6,088 lines) containing 20 pinned MSL kernels (`qwen35_attention_qk_rms_rope_bf16_v1`) and 4 `compile(shapeless:)` fusion blocks into `mlx-swift-lm`.
* **Expected Recovery:** Reduce `tEval` from ~163 ms to ~24 ms, restoring target decoding throughput to **~25+ tok/sec**.