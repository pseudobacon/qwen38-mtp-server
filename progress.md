# qwen38-mtp-server — Progress

A speculative-decoding server for Qwen 3.8 / 3.5 architectures on Apple Silicon MLX.

1. **Engine layer** — `../mlx-swift-lm`: custom MTP draft/verification block session (`Qwen38MTPBlockSession`) and model definitions.
2. **Model-state layer** — `MLXFastModel`: weight loading, KV-cache state management, tied-embedding sanitization, MTP head attachment.
3. **Server layer** — `HTTPServer`: Vapor-based OpenAI-compatible API (`/v1/chat/completions`), SSE streaming, memory admission control, observability endpoints.

## Architecture baseline (v1.0)

- **Verified features**: OpenAI SSE streaming fully compliant; parameter handling (temperature, max_tokens, `enable_thinking`); per-request KV-cache and tokenizer-cache isolation; MTP draft/verification alignment matching engine distributions (93.5% raw-token acceptance on the diagnostic benchmark).
- **Test suites**: `HTTPServerTests` 121/121 (the previously recorded 107/108 counts are stale — 108 was a grep artifact, the suite has since grown to 121 with the Item A/C/D work; verified green on `4af4e73`), engine `MLXLMTests` MTP diagnostic suites all green (`Qwen38MTPDiagnosticTests` 1 test, fusion projection suites 9/9/15).
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
- **Session discipline**: fresh server per cell (port 18099), `QWEN_MTP_STEP_TRACE=1`, rep 1 discarded as warmup, 5 measured reps, interleaved cell order so thermal drift hits cells evenly, no parallel builds/tests during timing, per-rep thermal state logged (pmset -g therm, wired into benchmarks/run_phase3.sh). 
- **Cross-session absolute latencies are not comparable** (observed 116–124 ms vs 135–165 ms bands for identical code under different thermal conditions) — only in-session deltas are valid. Fusion engagement is verified from the load-time summary (`backbone swiGLU 64/64 qkv 16/16 gdn 48/48; head …`), not from RSS.
- **Binary and repo provenance**: every cell's JSONL record carries
  `binary_sha256` (the release binary actually launched), `binary_mtime`,
  `server_head`, `engine_head`, and `server_dirty` / `engine_dirty` flags,
  emitted by `benchmarks/run_cell.sh`. All cells in one matrix must share a
  single `binary_sha256` — a mismatch means the binary changed mid-matrix and
  the run is discarded. Before any timing: force-recompile changed engine
  modules (SwiftPM incremental state may not invalidate them — the Phase 3
  stale-binary trap, `03231f16…` vs `88e27643…`), and headline runs require
  both worktrees `clean`.
- **Result artifacts**: `benchmarks/results/*.jsonl`, `benchmarks/FUSION_REPORT.md`.

### Headline throughput (conditions-labeled — not cross-comparable)

Phase 3 TTLT uses the pinned definition 1024 / wall-seconds (the 22.28 row is 1024 / decode-seconds, as originally recorded).

| Number | Conditions | Provenance |
|---|---|---|
| **21.32 tok/s** (1024 / 48.08 s decode, 111.36 ms mean step, acceptance 1.3759) | `essay-1024`, default config (QMV verify ON, adaptive draft depth 3, **4-bit MTP head default ON**), 5 measured reps, binary `c56ca6ea…` | **W4 A/B matrix q4 cell (2026-09-14)** — current-main headline, essay (post-W4 default; binary `9753a41e…` default cells reproduce the pinned stream bit-exact) |
| **23.45 tok/s** (1024 / 43.83 s decode, acceptance 1.6693) | `specdec-800`, default config (adaptive depth 3, 4-bit MTP head default ON), 5 measured reps, same binary | **W4 A/B matrix q4 cell (2026-09-14)** — current-main headline, specdec |
| 19.90 tok/s (1024 / 51.47 s decode, 120.63 ms mean step, acceptance 1.4061) | `essay-1024`, default config (QMV verify ON, adaptive draft depth 3, **BF16 head**), 5 measured reps, binary `11a8e61e…` | W3 sweep k3 cell (2026-09-14) — superseded by the W4 default (BF16-head state; the in-session W4 matrix BF16 cell measured 19.63) |
| 25.44 tok/s (1024 / 40.26 s decode, acceptance 1.6974) | `specdec-800`, default config (adaptive depth 3, BF16 head), single probe rep, binary `11a8e61e…` | W3 specdec confirmation (2026-09-14) — superseded by the W4 default (BF16-head state; the in-session W4 matrix BF16 cell measured 22.11) |

**Headline refresh (DONE 2026-09-14):** the W4 default (4-bit MTP head ON) supersedes the W3 BF16-head rows above (kept as provenance; the W4 matrix's in-session BF16 cells are the valid same-session comparison). The W3 k=2 essay cell (21.26 tok/s) remains the fastest valid *forced-depth* essay configuration but **fails the specdec determinism gate** (W3 section); the default adaptive k=3 remains the recommended configuration and the only one certifiably bit-exact on both fixtures.
| 16.64 tok/s (1024 / 61.66 s wall, 143.91 ms mean step) | `essay-1024` fixture, `MLX_COMPILED_DECODE=0` (compiled fast paths OFF) | Phase 3 B3 — ablation cell |
| 22.28 tok/s (1024 tok / 45.96 s decode, 120.86 ms steps) | `specdec-800` fixture, acceptance 1.6974 (2.70 tok/round), cooler session | 2d Item 3 final run — **pre-QMV-grid-fix binary**; the routed kernel never executed in that run |
| 16.55–16.89 tok/s (143.0–145.4 ms steps) | `essay-1024` fixture, acceptance 1.4061 (2.41 tok/round), sustained thermal load | Item A clean fusion matrix — current fixed binary, fusion engaged |

The old essay-vs-specdec gap decomposed as acceptance × step-time (×0.74). The new **B2-vs-22.28** gap is purely step time: acceptance is identical (645/1008/380, same stream hash `139acb9d…` in both), so 22.28 × (120.86/141.62) ≈ 19.0 ≈ B2's measured 19.05 (1024/decode-seconds: 19.046). That +17.2% step-time delta is the unseparated sum of this session's thermal state and the M=1 routed-QMV dispatch (the ≈5%-slower-at-M=1 serialized-microbench number was not reproduced under the sustained conditions W5 later measured — `benchmarks/results/w5-throughput.txt`) — consistent with the standing decision that cross-session absolute latencies are not comparable.

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

**Ablation verdict (in-session, B3 − B1):** compiled decode OFF costs **+7.350 ms/step (+5.38%)** and **−0.914 tok/s (−5.21%)** on the essay fixture. This is the combined controlled contribution of every component gated by `MLX_COMPILED_DECODE` — Checkpoint 1's compiled activation micro-fusions, Checkpoint 2a's QK-RoPE fused kernel, and the fused residual+RMSNorm kernel — measured in an otherwise identical session. The packed W_qkv / W_gate+up fusions (separate env knobs, default ON) and the routed QMV kernels (ungated) are active in both cells, so they are not part of this delta. **Verdict toward the 24 ms `tEvalMs` target:** the family is measured out as a *material* lever — a real but small, in-session, bit-exact win (≈7 ms/step; ≈6.9 ms on `tEval`, ≈5–6% of the ≈125 ms `tEval`, closing only ≈7% of the ≈100 ms gap to 24 ms), not a route to the target on its own. The path to 24 ms is the attention/acceptance work (open item 4); the family is retained default-ON as a confirmed win, not as a path to 24 ms.

**Binary-provenance note:** the release binary on disk (built 18:38) predated engine commit `901d2ca`; SwiftPM's incremental state had not invalidated the two changed engine modules. A forced recompile of `FusedQuantizedLinear.swift` + `Qwen35+FastPath.swift` plus relink produced a **different** binary (SHA-256 `88e27643…` vs `03231f16…`), and that is the binary this matrix ran on. The stale-binary risk flagged in the handoff is closed: the headline numbers above are from the current codebase.

### Phase 3 — Item D: route the verify pass through the candidate QMV kernel (2026-09-14) — final classification: +12.2% win, default ON

**Design (engine `b900aad`):** env gate `MLX_QWEN_QMV_VERIFY` (original default OFF, loud load-time log line; flipped to ON by the decision below). ON: 3-D verify `[B, L, K]` with `B·L ∈ 2..9` is reshaped (free row-major view behind a stride guard; non-contiguous inputs materialized and counted) to `[B·L, K]` and dispatched through the candidate `qwen35RoutedQuantizedMM`; M = 1 (2-D decode and `B·L == 1`) falls back to incumbent `layer(x)` (W5: a wash under sustained conditions). OFF is exactly the pre-Item D behavior (3-D falls back via the `ndim == 2` guard; M = 1 routes as before). Per-dispatch counters (`Qwen35QMVVerifyDispatch`): routed/fallback B·L-width histograms plus `materialized`, emitted in the per-request MTP-STEP-SUMMARY (parsed into per-rep JSONL) and at server shutdown. GDN `in_proj` fused path untouched. Server `4ca9589` + `4af4e73` (the latter fixes a `String(format:) %s` bug that emitted the qmv tokens as raw bytes, dropped by the parser).

**Original A/B (2026-09-14) — NULL, INVALID: flush confound.** Protocol: same release binary `1a71b2c9…` in both cells (env-var-only difference), essay-1024 fixture, port 18099, greedy, `QWEN_MTP_STEP_TRACE=1`, fresh server per cell, 6 reps per cell interleaved with alternating start cell, rep 1 discarded, 5 measured, `pmset -g therm` before every rep (no thermal warning in any snapshot), no parallel builds/tests, no mid-matrix rebuild; all 12 reps bit-exact (599/1086/426, hash `949b9423…`). Reported result: mean Δ −0.252 ms/step (D1 marginally *slower*), 4/5 pairs favor D1, decode wall 58.644 vs 58.750 s, TTLT 17.475 vs 17.446 tok/s. This run is **not a valid measurement**: the guard's row-major stride probe is `x.asData(access: .noCopy).strides`, and `MLXArray.asData(access:)` begins with `self.eval()` (mlx-swift `MLXArray+Bytes.swift`) — so **every routed dispatch flushes the GPU pipeline mid-graph-build** (96 flushes per verify forward). In D1, ~119 ms of GPU wait moves from the step-trace's eval phase into the graphBuild phase (tEvalAvg 125.573 → 6.302 ms; tGraphBuildAvg 10.346 → 129.854 ms) and host build + GPU execution serialize layer-by-layer; the per-rep phase components sum to `stepAvg` exactly in both cells (a pipeline non-translation, protocol class (i)). D0 never reaches the probe (the fast-path's own `asData` check is short-circuited by an `ndim == 2` guard that verify's 3-D input fails), which is why the shift is D1-only. `materialized=0` and the 99.1% routed share rule out engagement/copy artifacts (class (ii) does not apply). Per-rep provenance: `benchmarks/results/itemd-D0-essay.jsonl` / `itemd-D1-essay.jsonl`; driver `benchmarks/run_itemd.sh`.

**Guard fix (engine `c87fc6b`):** the per-call `x.asData(access: .noCopy).strides` contiguity probe is replaced by `Qwen35RowMajorCache` (`Qwen35+FastPath.swift`): a lock-protected, shape-keyed cache that pays the one `asData` probe on the first observation of a shape (warm-up, where a flush is harmless) and returns the cached decision afterwards. A stale "contiguous" entry is safe by construction — the QMV kernel wrappers are `ensureRowContiguous: true`, so the worst case is an unnecessary copy inside the reshape, never a wrong result. The same pattern replaces the two latent per-call stride probes in `applyResidualNorm`.

**Hot-path `asData` audit (all call sites classified):**

| Location | Classification | Disposition |
|---|---|---|
| `Qwen35Kernels.swift` Item D guard | **Hidden flush** — 96 `eval()`s per verify forward in D1; the Item D artifact | Fixed: cached per-shape decision |
| `Qwen35+FastPath.swift` `applyResidualNorm` stride probes | **Dead code** — every live caller passes 3-D `[B,S,H]`, so the `ndim == 2` guard declines before the probe fires; latent landmine | Fixed with the same cache pattern |
| `UserInput.swift` image processing | Non-hot-path | None |
| `Qwen38MTPBlockSession` `asArray` / `.item()` readouts | Intentional post-`eval` host reads (the `tHostRead` phase) | None |

**Flush-free rerun (2026-09-14):** same protocol as the original A/B, plus two new per-rep gates — single flush-free binary `55fa97e8…` in both cells (env-var-only difference), essay-1024 fixture, port 18099, greedy, `QWEN_MTP_STEP_TRACE=1`, fresh server per cell, 6 reps per cell interleaved, rep 1 discarded (warmup), 5 measured, `pmset -g therm` before every rep (no thermal warning in any snapshot), no parallel builds/tests, no mid-matrix rebuild. Every rep additionally gated on the new **phase-sum validation** (`tEval + tGraphBuild + tCacheState + tHostRead ≈ stepAvg` within max(1 ms, 1%)) and `qmvVerifyMaterialized = 0`. Phase 2 gates before the matrix, on this binary: fusion projection tests green in both explicit knob states (QKV 9/9, SwiGLU 9/9, GDN 15/15 in `swift test` processes with and without the env set); `Qwen38MTPDiagnosticTests` at 93.46% / 16.25 / 14.0 in both states; `run_matrix.sh verify` bit-exact on both fixtures with zero routed dispatches (explicit OFF env), zero materializations, phase-sums exact.

| rep | D0 avgStepMs | D1 avgStepMs | Δ (D0−D1) | D0 tEvalAvg | D1 tEvalAvg | D0 tGraphBuild | D1 tGraphBuild | D0 wall s | D1 wall s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 130.666 | 108.668 | +21.998 | 118.925 | 96.630 | 10.116 | 10.419 | 55.981 | 46.607 |
| 3 | 137.002 | 121.408 | +15.595 | 125.211 | 109.273 | 10.175 | 10.535 | 58.679 | 52.033 |
| 4 | 137.546 | 120.780 | +16.766 | 125.717 | 108.405 | 10.206 | 10.734 | 58.905 | 53.368 |
| 5 | 137.845 | 123.343 | +14.502 | 125.964 | 111.093 | 10.236 | 10.621 | 59.046 | 52.864 |
| 6 | 138.242 | 123.786 | +14.457 | 126.443 | 111.575 | 10.174 | 10.590 | 59.199 | 53.038 |
| **mean** | **136.260** | **119.601** | **+16.659** | **124.452** | **107.395** | **10.182** | **10.580** | **58.362** | **51.582** |

TTLT: D0 17.62 vs D1 20.11 tok/s. All 12 reps bit-exact (599/1086/426, hash `949b9423…`), phase-sums exact to 0.00 ms in every rep, `materialized=0` in every rep. **The artifact is gone:** the D1 warmup rep shows tEvalAvg 93.1 ms and tGraphBuildAvg 10.4 ms — the D0 regime — versus the pre-fix D1's 6.3 / 129.9 ms. Engagement identical to the original D1: routed `2:1632, 3:15552, 4:24096` (41 280 dispatches, 99.1% of all 41 664) in every D1 rep.

**Decision (2026-09-14): `MLX_QWEN_QMV_VERIFY` default ON — rollback knob `MLX_QWEN_QMV_VERIFY=0`.** The keep gate (mean improvement ≥ 2.0 ms/step AND ≥ 4 of 5 paired reps favoring D1) passes decisively: **+16.659 ms/step mean (12.2%)** and **5/5 paired reps** favor D1. The original null was entirely the flush artifact; with it removed, the qmvbench 15–20% kernel win at M = 2..4 transfers end-to-end. Note the direction of the thermal bias: D0 always runs first in each pair (colder start), yet still loses — the win is robust to in-run thermal drift (both cells drift upward across reps). Default flip (engine `a5f102f`; server unchanged): `Qwen35QMVVerifyRouting.enabled` now returns true when the env is unset; `MLX_QWEN_QMV_VERIFY=0` (or any non-on token) still selects the pre-Item D baseline for A/B cells. Post-flip verification on the new binary `11a8e61e…`: fusion projection tests green in both explicit states (default run = ON), `Qwen38MTPDiagnosticTests` 93.46% in both states, and `run_matrix.sh verify` (env unset → default ON) bit-exact on both fixtures with engaged routing (essay `2:3264, 3:31104, 4:48192`) and `materialized=0`. The 2× routed dispatch count versus the original D1 spot cell is the fusion-state difference only (this cell runs `FUSED_QKV=0 FUSED_SWIGLU=0`, six dispatches per full-attention layer, versus the default-ON fusions' three); both agree with the same two-full-forward-per-round structure (16 full-attention layers × 3 or 6 dispatches).

Records: `benchmarks/results/itemd-rerun-D0.jsonl`, `itemd-rerun-D1.jsonl`; driver `benchmarks/run_itemd.sh`; run log `/tmp/itemd-rerun-driver.log`.

### QMV microbenchmark (`qmvbench`)

Layer-3 gate/up pair, 100 warmup + 1000 timed × 3 blocks, per-call device sync. Routed kernel is bit-identical to incumbent `quantizedMM` everywhere (re-verified at every M in the W5 run). At M = 1 it is ~5% slower per projection (~385 vs ~366 µs); at M = 4 it is ~15% faster (~464 vs ~545 µs) and the fused wide dispatch ~20% faster (~725 vs ~904 µs). Interleaved layout within noise of global at both M.

**W5 — sustained throughput mode (`--throughput 128`, 2026-09-14):** 128 back-to-back submissions per timed batch, one device sync per batch, 1000 timed batches, same geometry and bit-exactness checks. Raw log: `benchmarks/results/w5-throughput.txt`. Sustained µs/call (routed vs fallback, mean):

| M | narrow gate / up (routed / fallback) | wide fused (routed / fallback) | routed win |
|---|---|---|---|
| 1 | 162.0 / 162.4 ; 141.4 / 151.7 | 321.8 / 303.7 | wash (wide slightly favors fallback) |
| 2 | 165.8 / 197.1 ; 166.5 / 197.1 | 342.2 / 404.2 | ~16% |
| 4 | 252.5 / 359.8 ; 250.1 / 363.3 | 499.0 / 745.9 | ~30% |
| 8 | 496.9 / 698.7 ; 496.1 / 697.8 | 990.9 / 1426.2 | ~29% |
| 9 | 572.7 / 856.0 ; 570.6 / 856.2 | 1140.8 / 1740.6 | ~33% |

Two conclusions. (1) The per-call-sync protocol carried a fixed **~170–235 µs/call sync overhead** (serialized − sustained at every condition); the Item D-era micro numbers were inflated by it, and the "~5% slower at M = 1" penalty does not exist under sustained conditions (routed ≈ fallback at M = 1, consistent with the M = 1 flip to incumbent). (2) **The M = 2..9 kernel win survives — and grows to ~30% — under sustained async submission**, confirming the W1 end-to-end keep decision and resolving the original A2 null question (the null was the flush artifact, not async submission). Sustained bandwidth at M = 1 reaches 310–355 GB/s (vs 142–186 GB/s serialized).

### Reference step-component profile (in-session, ~143 ms round)

`tEvalAvg` ~131 ms, `tGraphBuildAvg` ~10–11 ms, `tCacheStateAvg` ~0.9–1.7 ms, `tHostReadAvg` ~0.02 ms.

### W2 — tEval profile: Metal System Trace + headbench (2026-09-14) — DONE

Deliverable: **`benchmarks/PROFILE.md`**. One greedy essay-1024 request captured under
`xctrace` Metal System Trace (driver `benchmarks/run_w2trace.sh`; trace
`/tmp/w2-profile.trace`, exports `/tmp/w2-*.xml`): **GPU utilization 98.2 %**
(10 817 ms busy over an 11 010 ms span) — decode is GPU-bound, not dispatch-bound.
CPU is ~2.56 s over ~11 s wall (server 14.1 %, malloc 4.8 %, AGX 4.2 %, swiftCore
3.7 %) — not the bottleneck; `tGraphBuild` ≈ 10.5 ms/round (~10 %) is the dominant
CPU-side cost. **Stated limitation:** this trace template does not enable the Metal
shader profiler, so it contains no per-kernel GPU intervals (command-level only);
per-kernel bucketing therefore comes from micro-benchmark measurement +
bytes/bandwidth estimates, each labeled measured or estimated in PROFILE.md.

New engine tool **`headbench`** (`Libraries/HeadBench`, product `headbench`) loads
the backbone + head exactly the way the server does and times the exact per-round
session calls (median of 50 timed reps, release build): head forward (flush/step)
**16.38 / 16.26 ms**, draft projection `draftTokenID` **3.95 ms**, verify
`lm_head` **4.33 → 7.94 ms** (M = 2..5). Per-round head-family cost: **d=1 24.66
ms, d=2 45.94 ms, d=3 67.47 ms, d=4 88.89 ms**. Raw-matmul reference (`--raw`):
bf16 GEMV floor for one head forward **7.36 ms @ ~115 GB/s**; the ~9 ms remainder
of the 16.38 ms head forward is layer structure (gated attention, RoPE, q/k
norms, the eager non-fused SwiGLU path — load summary `head swiGLU 0 qkv 0` —
per-rep fresh KV-cache update, ~25 kernel launches), an estimate pending a
per-kernel profile. 4-bit group-32 reference at the backbone gate shape: 0.42 ms
@ ~105 GB/s.

Per-round decomposition (d=2 class, tEval ≈ 105 ms): backbone verify forward (3
rows, 4-bit) ~55–59 ms (~55 %, **estimated** from 14.42 GB ÷ 200–275 GB/s —
implied bandwidth back-computed from the W3 sweep at all four depths: 215 / 247 /
273 / 200 GB/s, all inside the measured 4-bit kernel band), MTP head family
40.6 ms (~39 %, **measured**), verify `lm_head` 5.4 ms (~5 %, **measured**), gaps/
launches the remainder. Weight traffic from the safetensors headers: backbone
body **13.70 GB/forward** (16 FA × 209.4 MB + 48 GDN × 215.7 MB); `lm_head` 4-bit
payload **635.7 MB** (U32 [248320, 640], group 64) + 79.5 MB scales/biases;
`embed_tokens` likewise 635.7 MB; grand total 15.13 GB. **Correction:** the earlier
note "`lm_head` … 317.8 MB" was wrong (2-byte element assumption); the payload is
635.7 MB.

**W4 trigger: MET.** Gate was ≥ ~5 ms/round on the draft-head path; measured
24.66 ms/round (d=1) and 45.94 ms/round (d=2) — triggered by 5–9×. `lm_head` is
already 4-bit: report-only, no further quantization headroom (and quantizing it
would change committed tokens). Next fused-kernel levers in size order: (1) 4-bit
MTP head (W4; ~14 ms/round at d=2, 849 MB → ~300 MB), (2) SwiGLU/QKV fusions
extended to the head (launch-level win), (3) backbone 4-bit GEMV bandwidth
(200–275 GB/s in-pipeline vs 310–355 GB/s sustained in qmvbench), (4)
`tGraphBuild` ~10.5 ms CPU (secondary).

### W3 — MTP draft-depth sweep (2026-09-14) — **correctness stop at depth 5; valid optimum k=2 (21.26 tok/s, conditional)**

**Protocol.** `benchmarks/run_w3sweep.sh` over `run_cell.sh` (fresh server per cell, port
18099, greedy temp 0, `enable_thinking: false`, max_tokens 1024, `finish_reason: length`,
`QWEN_MTP_STEP_TRACE=1`, `pmset -g therm` before every rep, no parallel work, no mid-sweep
rebuild, binary `11a8e61e…` identical across all cells — verified per-cell SHA in the
driver log). The committed stream must equal the fixture hash in every cell and rep
(accepted/proposed/rounds are free — they vary with depth — and are logged). 6 reps per
cell, rep 1 discarded as warmup, interleaved with rotating start cell. Cell configs:
k1 `QWEN_MTP_DRAFT_K=1`; k2 `QWEN_MTP_DRAFT_K=2`; k3 no env/no flag (adaptive cost model,
offered 3 — **the current value**); k4 `--spec-draft-n-max 8` + `QWEN_MTP_DRAFT_K=4`.

**Correctness stop (NEW BUG — top open item).** The depth-5 cell (`QWEN_MTP_DRAFT_K=5`,
`--spec-draft-n-max 8`) committed stream `da7bb159…` — NOT the essay hash `949b9423…`.
Depth 6 reproduced the *same* wrong stream (deterministic). Depths 1–4 are bit-exact in
every rep of the sweep. The first divergence is the first drafted token of the **first
verify round** (the prefill's token 1 matches in all cells). Provenance:
`benchmarks/results/w3-probe-k5.jsonl`, `w3-probe-k6-qmvoff.jsonl`,
`w3-probe-k6-nofuse.jsonl`, and the diverged `w3-draftk-essay-k6.jsonl` record.

Exclusion probes (fresh server, single rep each):

| probe | env | stream | verdict |
|---|---|---|---|
| k5 default | `QWEN_MTP_DRAFT_K=5` | `da7bb159…` | DIVERGED |
| k6, QMV off | `QWEN_MTP_DRAFT_K=6 MLX_QWEN_QMV_VERIFY=0` | `da7bb159…` | DIVERGED — routed QMV verify exonerated |
| k6, no fast path | `QWEN_MTP_DRAFT_K=6 MLX_QWEN_QMV_VERIFY=0 MLX_QWEN_FUSED_QKV=0 MLX_QWEN_FUSED_SWIGLU=0` | `da7bb159…` | DIVERGED — fused QKV/SwiGLU exonerated |

**Root-cause hypothesis (static, strong).** Full-attention verify at width 6..9 dispatches
a single `MLXFast.scaledDotProductAttention` with qL = 6..9 (plain `KVCacheSimple`
`update` + one SDPA — the final branch of `attentionWithCacheUpdate` in
`MLXLMCommon/AttentionUtils.swift`). At qL·gqa = 36..54 > 32 the fused-vector SDPA path is
left, so the wide verify attention is not bit-identical to the serial (qL=1) path and a
greedy argmax can flip. The engine's own design comments
(`Qwen38MTPBlockSession.swift` ~1428–1436 and ~1996–1998; the SDPA warm-up comment
~755–770) specify an "exactness chunk": 6..9-row causal verify attention must be split
into two ≤5-row SDPA calls with byte-identical bottom-right-aligned windows — and the
warm-up code still compiles exactly the qL∈{1..5} shapes that chunk would dispatch —
**but that split does not exist anywhere in the current code** (single `attentionWithCacheUpdate`
definition, no `updateAndAttend` in the LLM model). The empirical boundary matches the
qL·gqa = 32 edge exactly: width ≤5 (qL·gqa ≤ 30) is bit-exact, width 6 (36) diverges at
round 1. Fix direction (separate task): implement the chunked verify SDPA (two ≤5-row
calls, second call over the cache extended by the first chunk's K/V), then re-run the
determinism gate at k5–k8 before any performance claim at those depths. Until then,
draft depths ≥5 are **unsound** and the `--spec-draft-n-max 8` / `QWEN_MTP_DRAFT_K`
surface above 4 is broken, not merely slow.

**Essay-1024 results (valid subset, 5 measured reps per cell; all reps bit-exact
`949b9423…`, phaseSumOK, materialized=0):**

| k | rounds | accepted | acc/step | stepAvg ms | decode s | tok/s |
|---|---|---|---|---|---|---|
| 1 | 577 | 448 | 0.7764 | 92.08 | 53.20 | 19.25 |
| 2 | 459 | 567 | 1.2353 | 104.76 | 48.16 | **21.26** |
| 3 (adaptive, current) | 426 | 599 | 1.4061 | 120.63 | 51.47 | 19.90 |
| 4 | 390 | 636 | 1.6308 | 161.27 | 62.97 | 16.26 |

Per-rep data: `benchmarks/results/w3-draftk-essay-k{1,2,3,4}.jsonl` (last 5 records are the
measured reps r2–r6). Depth distributions: k1 `1:577`, k2 `2:459`, k3 `1:16, 2:160, 3:250`,
k4 `4:390`.

**Reading.** Within the valid set, **k=2 is the essay optimum: 21.26 tok/s, +6.8 % over the
current default (k=3 adaptive, 19.90)** and +10.4 % over k=1. The acceptance curve is
sub-linear in depth: each extra drafted token adds ~40 ms of verify cost (stepAvg
92→105→121→161 ms) while removing far fewer than a proportional number of rounds
(577→459→426→390). k=4 is strictly worse than the default. The pre-fix sweep cells at
k=6/k8 are INVALID (wrong stream) and are reported only as bug evidence — their
timing-looking numbers must never be used as performance data. If the SDPA exactness
chunk is implemented and re-verified, the sweep must be re-run at k ∈ {5,6,8} before
claiming any deep-depth optimum.

**Specdec-800 confirmation (fresh server, single rep per k; determinism gate + reference
timing only, not headline-grade statistics):**

| k | rounds | accepted | acc/step | decode s | tok/s | gate |
|---|---|---|---|---|---|---|
| 1 | 562 | 462 | 0.8221 | 45.95 | 22.29 | PASS (`139acb9d…`) |
| 2 | 427 | 598 | 1.4005 | 39.26 | 26.08 | **FAIL** — deterministic wrong stream `06882d85…` (rerun reproduces the same hash; divergence at char 4847/5131 ≈ 94.5 %) |
| 3 (adaptive, current) | 380 | 645 | 1.6974 | 40.26 | 25.44 | PASS |
| 4 | 334 | 690 | 2.0659 | 47.24 | 21.68 | **FAIL** — deterministic wrong stream `48728382…` (divergence at char 4883/5131 ≈ 95.2 %) |

Provenance: `benchmarks/results/w3-draftk-specdec-k1.jsonl`,
`w3-draftk-specdec-k2.jsonl`, `w3-probe-specdec-k2-rerun.jsonl`,
`w3-probe-specdec-k3.jsonl`, `w3-probe-specdec-k4.jsonl`.

**Specdec reading.** Forced depths k=2 and k=4 are not bit-exact on specdec-800 either —
with a *different* signature from the k5/k6 essay bug: divergence at ~95 % of the stream
(late-position knife-edge argmax flips), deterministic per prompt (same wrong hash across
fresh server runs), and the k2/k4 specdec wrong streams differ from the essay k5/k6 wrong
stream. Combined with the essay results: **no forced depth other than k=1 is
certifiably bit-exact on both fixtures in the current evidence**; the default adaptive
k=3 is the only configuration bit-exact on both fixtures and the recommended
configuration. The k=2 essay optimum is therefore **conditional**: it is the fastest
valid essay cell, but until the specdec k=2/k4 divergence is understood and resolved, k=2
must not be promoted as the default. (Note the specdec k2 wrong stream at 26.08 tok/s is
fastest-in-class on specdec — the divergence costs nothing in the measured stream, which
is exactly why the hash gate, not the timing, is the acceptance criterion.)


**Trigger (W2).** headbench measured the BF16 head family at 24.66 ms/round (d=1) and
45.94 ms/round (d=2) — 5–9× the ~5 ms/round gate. The head is a full transformer
decoder layer (fc + one full-attention layer + MLP, 849.3 MB BF16) whose raw matmul
floor is 7.36 ms per forward at ~115 GB/s on top of ~9 ms of eager layer-structure
cost (gated attention, rope, q/k norms, eager SwiGLU, KV update, ~25 launches).

**Artifact.** `benchmarks/make_q4_head.py` quantizes the pinned BF16 head to 4-bit
group-64 affine in the backbone's exact layout (U32 `[out, in/8]` weights + BF16
`[out, in/64]` scales/biases for `fc`, `q/k/v/o_proj`, `gate/up/down_proj`; the seven
RMSNorms stay BF16) → `mtp-head/q4/` (238.9 MB; the BF16 tree is untouched).
`verifyHeadTree` needed no engine change: the head tree carries bare keys + a 31-key
index, and the backbone config's `perLayerQuantization` (4-bit/64/affine) drives the
`quantize(model:)` walk, which converts every `Linear` with a `.scales` key into a
`QuantizedLinear` — including the head's.

**Knob + default flip.** `MLX_QWEN_MTP_HEAD_QUANT` (server, `MLXGenerator.init`):
`1/true/on` forces the 4-bit tree (fails loudly if missing); `0/false/off` forces the
pinned BF16 tree (rollback); **unset = default ON** (post-verdict) — 4-bit tree if
present, otherwise a loudly logged BF16 fallback (a fresh checkout runs
`benchmarks/make_q4_head.py` once). Every load logs `MLXLM: MTP head selected: …`;
`run_cell.sh` records `head_selected` + `fusion_summary` per rep so engagement is
proven, not assumed.

**Head fusion engagement.** Fusion eligibility requires stock `QuantizedLinear`
instances: the BF16 head was ineligible (`head swiGLU 0 qkv 0`), the 4-bit head
engages both fusions. `fusion_summary` in every matrix cell: BF16 state
`backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 0 qkv 0`; q4 state
`…; head swiGLU 1 qkv 1`. The head's quantized linears also route through the Item D
QMV verify kernel (expected interaction, recorded not fought).

**Diagnostic runs (recorded, not gated).** `Qwen38MTPDiagnosticTests` gained
`QWEN_MTP_HEAD_TEST_PATH` / `QWEN_MTP_TEST_DEPTH` env overrides (depth default now 4 —
the largest width W3 verified safe; 8 crossed the bug-B territory) and records
per-prompt + overall committed-stream hashes (printed, never asserted):

| state | depth | acceptance (3 prompts) | aggregate | overall committed hash |
|---|---|---|---|---|
| BF16 | 8 (historical default) | 94.35 / 93.07 / 93.00 | 93.46 % | `80ffe841…` |
| q4 | 8 | 96.62 / 93.07 / 92.27 | 93.97 % | `33ac90ab…` |
| BF16 | 4 | 91.10 / 93.85 / 96.80 | 93.97 % | `06e40dd4…` |
| q4 | 4 | 94.56 / 93.85 / 95.60 | 94.68 % | `2860dc69…` |
| BF16 | 1 | 89.06 / 96.88 / 100.00 | 95.31 % | `b609ae55…` |
| q4 | 1 | 92.19 / 96.88 / 100.00 | 96.35 % | `67c3f591…` |

**Why the diagnostic is not a W4 committed-stream gate.** The diagnostic's three short
prompts are width-sensitive *within a fixed head state*: BF16 depth 8 ≠ depth 4 ≠
depth 1 streams, and at depth 1 (verify width 2) prompt 1 still diverges between head
states while prompts 2–3 agree. That is the same batched-verify-vs-serial
bit-exactness failure family as W3's bug A (specdec k=2/k4 at width 3–5) and bug B
(width ≥ 6): committed-stream identity depends on verify-batch geometry/content
through knife-edge argmax flips, and the diagnostic prompts sit on those edges.
The valid W4 gate is the A/B matrix below — the two benchmark fixtures at k=3, where
the W3 evidence says bit-exactness holds. (The 93.46 % figure remains a recorded
reference value, never a code assertion.)

**A/B matrix (Phase 3 protocol).** Binary `c56ca6ea…` (the only difference is the env
var), k=3 default adaptive policy (no `QWEN_MTP_DRAFT_K`, no `--spec-draft-n-max`),
6 reps per cell per fixture (rep 1 warmup, 5 measured), interleaved with rotating
start cell, fresh server per cell, port 18099, greedy, per-rep gates: committed
stream = fixture hash, `head_selected` proves the intended tree loaded, `phaseSumOK`,
`qmvVerifyMaterialized = 0`, thermal snapshot before every rep. **All 24 reps GATE
PASS.**

| fixture | state | rounds | acc/step | step ms | tEval | tGraphBuild | decode s | tok/s | gate |
|---|---|---|---|---|---|---|---|---|---|
| essay-1024 | BF16 | 426.0 | 1.4061 | 122.51 | 110.09 | 10.80 | 52.27 | 19.63 | PASS `949b9423…` |
| essay-1024 | q4 | 431.0 | 1.3759 | **111.36** | 106.08 | **3.55** | **48.08** | **21.32** | PASS `949b9423…` |
| specdec-800 | BF16 | 380.0 | 1.6974 | 121.81 | 109.75 | 11.26 | 46.35 | 22.11 | PASS `139acb9d…` |
| specdec-800 | q4 | 384.0 | 1.6693 | **113.96** | 109.63 | **3.51** | **43.83** | **23.45** | PASS `139acb9d…` |

**Delta.** Essay **+1.69 tok/s (+8.6 %)**; specdec **+1.34 tok/s (+6.1 %)**. The q4
draft is marginally less accurate (acc/step 1.3759/1.6693 vs 1.4061/1.6974; +5/+4
rounds) but the step-time win dominates. The delta is carried almost entirely by
tGraphBuild (−7.25 / −7.75 ms — the fused 4-bit head graph is smaller: 1 fused QKV +
1 fused SwiGLU vs 3 + 5 eager ops, plus quantized-linear nodes); tEval is within
in-session noise (−4.01 / −0.12 ms).

**headbench confirmation (isolated head family).**

| call | BF16 median | q4 median |
|---|---|---|
| flush(F=1) | 16.38 ms | **1.52 ms** |
| step(F=1) | 16.26 ms | **1.48 ms** |
| proj1(M=1) (shared 4-bit lm_head) | 3.95 ms | 3.98 ms |
| verifyM(M=3) (shared 4-bit lm_head) | 5.41 ms | 5.56 ms |
| per-round d=1 | 24.66 ms | **9.85 ms** |
| per-round d=2 | 45.94 ms | **16.52 ms** |

The isolated head-body collapse (−14.9 ms per forward) does not transfer 1:1 into the
in-pipeline tEval delta (within noise) — the W2 bucketing built the per-round head
family number from exactly these isolated costs, so the in-pipeline attribution is
looser than the isolated one (same pattern as W5's serialized-vs-sustained finding).
Recorded as an open observation, not blocking.

**Post-flip verification (default ON, binary `9753a41e…`).** Default cells (env unset)
load the q4 head and reproduce both pinned streams bit-exact (essay `949b9423…` 24.6
tok/s; specdec `139acb9d…` 26.02 tok/s — single reps, this thermal session, reference
grade only). Rollback cell `MLX_QWEN_MTP_HEAD_QUANT=0` loads BF16 and is bit-exact
(essay `949b9423…`).

**Verdict (rule: ≥ 2 % net tok/s on BOTH fixtures AND bit-exact committed stream →
keep, default ON).** Essay +8.6 % and specdec +6.1 %, 24/24 reps bit-exact, engagement
proven per rep → **KEEP. `MLX_QWEN_MTP_HEAD_QUANT` default flipped to ON** (loud BF16
fallback if the q4 tree is missing; `=0` is the rollback knob). Memory: head 849.3 →
238.9 MB (−610 MB), enabling the fused head paths that the BF16 head could not use.

Provenance: `benchmarks/results/w4-ab-essay-{bf16,q4}.jsonl`,
`w4-ab-specdec-{bf16,q4}.jsonl` (4 files × 6 reps, rep 1 warmup per state),
`w4-diag-{bf16,q4}-{d1,d4,depth8}.txt`, `benchmarks/make_q4_head.py`,
`benchmarks/run_w4ab.sh`.


## Benchmarking history (superseded results — kept for provenance only)

The v1.0-baseline, 2a, 2b, and 2c tables previously in this file are **retracted as measurement artifacts**, for two reasons established on 2026-09-13:

1. **Prompt confound.** The 2c baseline ran the 38-token essay prompt (599/1086/426, 1.4061) while the 2d Item 1/2/3 runs ran the 62-token specdec prompt (645/1008/380, 1.6974). The reported "2c → Item 1 +6.19 ms QKV regression" and the "TTLT win" were cross-prompt comparisons. Provenance was established with a rolled-back all-fusion-off build (`benchmarks/results/resolve.jsonl`): essay reproduces 599/1086/426 exactly; specdec-800 reproduces 645/1008/380 exactly. Their absolute step latencies (~116–124 ms) additionally reflect a cooler thermal session and are not comparable to later runs.
2. **Binary-vintage / dead-kernel confound.** The 2a–2d checkpoints predate the QMV dispatch grid fix, and the routed kernel was never exercised in-model (decode reaches it only via 2-D `x`; verify falls back to `quantizedMM` by the `ndim == 2` guard). Correctness was preserved throughout because the eager/fallback paths carried those runs.

The engineering work of those checkpoints (kernels, fusions, tests, refactors) stands as described above; only their **timing tables** are invalid. The headline-throughput and fusion-matrix tables above are the only valid comparisons.

## Current status and roadmap

**Done:** Checkpoints 1, 2a, 2b, 2b-fix, 2c, 2d (both packed projections, merged to main in both repos), Item C interleaved layout (implemented, measured, rejected; removed from the engine in `901d2ca`), `qmvbench` microbenchmark target, prompt-fixture + determinism benchmarking protocol, MLXFast grid-convention bug fix, Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13; per-rep thermal logging wired into the harness), **Item D verify-pass QMV routing** (implemented; **final classification: +12.2% win, default ON** — the original A/B null was the `asData` flush artifact, root-caused; the flush-free rerun kept D1; engine `b900aad` → `c87fc6b` → `a5f102f`, server `4ca9589`+`4af4e73` → `31032ec` + evidence commits; detail: Phase 3 Item D section), **W5 `qmvbench` sustained-throughput mode** (implemented + measured; engine `a5f102f`), **W2 tEval profile** (DONE 2026-09-14 — `benchmarks/PROFILE.md`; `headbench` tool added to the engine; W4 trigger MET; headline refreshed), **W3 draft-depth sweep** (DONE 2026-09-14 — k=1..4 valid and bit-exact on essay; **k≥5 correctness stop** (top open item); essay optimum k=2 21.26 tok/s, conditional on the specdec k=2/k4 divergence; specdec: only k=1 and default k=3 bit-exact), **W4 MTP-head 4-bit quantization** (DONE 2026-09-14 — 4-bit head tree 238.9 MB generated; A/B matrix 24/24 reps bit-exact; essay +8.6 % / specdec +6.1 %; **verdict KEEP, `MLX_QWEN_MTP_HEAD_QUANT` default flipped ON**; detail: W4 section).

**Decisions on record:**

- Fused W_qkv and W_gate+up: keep, default ON (latency-neutral, bit-exact, net-0 memory, rollback knobs `MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU`).
- Interleaved gate+up layout: rejected (no gain, +6.5 GB row-gather copies); removed from the engine in `901d2ca` (2026-09-13). Historical data in `benchmarks/FUSION_REPORT.md` and `benchmarks/results/itemC.jsonl`.
- Item D verify-routing knob `MLX_QWEN_QMV_VERIFY`: **KEEP, default ON** (2026-09-14; final classification in the Phase 3 Item D section). The original A/B was NULL due to a measurement artifact: the guard's per-call `asData` stride probe calls `self.eval()` on every routed dispatch, serializing the verify pipeline and shifting ~119 ms of GPU wait from the tEval phase into the graphBuild phase. After replacing the probe with a cached per-shape contiguity decision (`Qwen35RowMajorCache`, engine `c87fc6b`), the flush-free rerun shows **+16.659 ms/step mean (12.2%), 5/5 paired reps, bit-exact, 99.1% routed, zero materializations, phase-sums exact in all 12 reps** — the qmvbench 15–20% kernel win at M = 2..4 does transfer end-to-end. Default is now ON; `MLX_QWEN_QMV_VERIFY=0` is the rollback knob (selects the pre-Item D baseline for A/B cells). Per-rep provenance for both the invalid original A/B and the valid flush-free rerun is in the `benchmarks/results/itemd-*.jsonl` records.
- W1 hot-path `asData` audit: all call sites classified (one hidden flush — fixed; two dead-code probes — fixed with the same pattern; `UserInput` non-hot-path; MTP session readouts are intentional post-`eval` host reads). No remaining per-call tensor metadata in the forward path.
- W4 MTP-head 4-bit quantization: **KEEP, default ON** (2026-09-14). The pinned head was BF16 (849.4 MB) — W2's headbench measured 24.66 / 45.94 ms/round at d=1/2 (trigger MET). The 4-bit group-64 tree (`mtp-head/q4/`, 238.9 MB, `benchmarks/make_q4_head.py`) A/B'd against the BF16 head at k=3 on both fixtures: **essay 19.63 → 21.32 tok/s (+8.6 %), specdec 22.11 → 23.45 tok/s (+6.1 %), 24/24 reps bit-exact, head fusion engaged (`head swiGLU 1 qkv 1`), zero materializations, phase-sums exact.** Win carried by tGraphBuild (−7.2/−7.8 ms); tEval within noise; the isolated head-body collapse (16.38 → 1.52 ms/forward) does not transfer 1:1 in-pipeline (recorded observation). `MLX_QWEN_MTP_HEAD_QUANT`: unset = default ON (loud BF16 fallback if the q4 tree is missing), `1` force q4, `0` rollback BF16. `lm_head` report-only (already 4-bit, 635.7 MB payload — quantizing it changes committed tokens). The diagnostic's short prompts are width-sensitive within a fixed head state (same bug A/B family), so they are recorded, not gated; the A/B matrix at k=3 is the committed-stream gate.
- MTP-head SwiGLU/QKV fusions: engaged automatically by the W4 4-bit head (stock `QuantizedLinear` eligibility); the BF16 head remains eager-fallback by design. No separate task needed.

**Open items (in priority order — W1, W2, W3, W4, W5 done 2026-09-14; next: correctness bug A, then bug B):**

1. **Correctness bug A — specdec even-k (k=2, k=4) committed-stream divergence (report-only until fixed).** Late-position knife-edge argmax flips at ~95 % of the stream, deterministic per prompt, distinct from bug B's wrong streams. Same batched-verify-vs-serial bit-exactness family; W4's diagnostic runs confirmed the diagnostic's short prompts are width-sensitive even at verify width 2 within a fixed head state, i.e. the boundary is content/fixture-sensitive, not a clean width threshold. Root-cause + regression test is the next engine task; then re-run the specdec k∈{1,2,3,4} determinism gate.
2. **Correctness bug B — verify width ≥ 6 commits wrong tokens (report-only until fixed).** Root-cause hypothesis (strong, static + empirical): the "exactness chunk" split specified in the engine's design comments — 6..9-row causal verify SDPA as two ≤5-row calls — does not exist in `attentionWithCacheUpdate`; at qL·gqa > 32 the fused-vector SDPA path is left and wide verify attention is not bit-identical to serial. Boundary matches exactly (width 5 = 30 ≤ 32 clean; width 6 = 36 diverges at round 1); QMV verify and fused QKV/SwiGLU exonerated by exclusion probes. Full detail + provenance: W3 section. Fix is a separate task (chunked verify SDPA + re-run the determinism gate at k5–k8); **the `--spec-draft-n-max` / `QWEN_MTP_DRAFT_K` surface above 4 is broken, not merely slow, until then.**
3. **W5 — `qmvbench --throughput N`: DONE 2026-09-14** — sustained throughput measured at M ∈ {1,2,4,8,9}, narrow/wide × routed/fallback (table in the QMV microbenchmark section): ~170–235 µs/call sync overhead in the serialized protocol; the routed kernel's M = 2..9 win grows to ~30% under sustained conditions; M = 1 is a wash. The M = 16/17 extension is **moot**: W3's optimum sits at k=2, not the sweep ceiling.
4. **Thermal control for benchmarks** — per-rep `pmset -g therm` logging is wired; shorter run blocks / cooldowns remain, so absolute numbers become comparable across sessions.
5. **Attention-layer kernels and acceptance-rate work** — the remaining path toward the `tEvalMs` 24 ms / 30 tok/s target; weight-packing is measured out as a lever at this geometry.
6. **Re-sweep draft depths k ∈ {5,6,8} after the item-2 fix** — the current k=6/k8 records are bug evidence only.