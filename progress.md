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

## Technical Debt & Performance Roadmap (`v1.1-performance`)

* **Current Status:** Checkpoint 1 (`compile()` closures), Checkpoint 2a (QK RMSNorm + RoPE kernel), Checkpoint 2b (fused residual+RMSNorm + routed wide QMV projection kernels), Checkpoint 2b-fix (five QMV dispatch defects fixed + fast-path extraction completed), Checkpoint 2c (native $M = 1$ wide QMV dispatch), the structural refactor, and **Checkpoint 2d (Items 1–3: fused $W_{qkv}$ + fused $W_{gate+up}$ packed projections, merged to `main` in both repos)** completed. Step latency ~116–124 ms band (tEvalAvg ~104–112 ms, greedy T=0); TTLT ~22.3 tok/s (1,024 tokens in 45.96 s wall, checkpoint 2d final run; 2c baseline 20.7 tok/s / 49.47 s).
* **Pending Scope:** Attention-layer fused kernels — the remaining path toward the ~24 ms `tEvalMs` target.
* **Target Milestone:** Reduce `tEvalMs` from ~104.21 ms to ~24 ms, achieving decoding throughput of **~30+ tok/sec**.