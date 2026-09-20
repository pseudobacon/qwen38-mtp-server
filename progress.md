# qwen38-mtp-server — Progress

A speculative-decoding server for Qwen 3.8 / 3.5 architectures on Apple Silicon MLX.

1. **Engine layer** — `../mlx-swift-lm`: custom MTP draft/verification block session (`Qwen38MTPBlockSession`) and model definitions.
2. **Model-state layer** — `MLXFastModel`: weight loading, KV-cache state management, tied-embedding sanitization, MTP head attachment.
3. **Server layer** — `HTTPServer`: Vapor-based OpenAI-compatible API (`/v1/chat/completions`), SSE streaming, memory admission control, observability endpoints.

## Architecture baseline (v1.0)

- **Verified features**: OpenAI SSE streaming fully compliant; parameter handling (temperature, max_tokens, `enable_thinking`); per-request KV-cache and tokenizer-cache isolation; MTP draft/verification alignment matching engine distributions (93.5% raw-token acceptance on the diagnostic benchmark).
- **Test suites**: `HTTPServerTests` 121/121 (the previously recorded 107/108 counts are stale — 108 was a grep artifact, the suite has since grown to 121 with the Item A/C/D work; verified green on `4af4e73`), engine `MLXLMTests` MTP diagnostic suites all green (`Qwen38MTPDiagnosticTests` 2 tests — the wide-verify serial-family test added in Phase 4 — plus `Qwen38SDPAExactnessTests` 4 tests, fusion projection suites 9/9/15).
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
| **21.89 tok/s** (1024 / 46.77 s decode, stepAvg 90.3–102.5 ms, acc/step 1.2213, depthDist 2:461) | `essay-1024`, **default config — pinned draft depth k = 2** (QMV verify ON, fusions ON, compiled decode ON, 4-bit MTP head default ON), 5 measured reps, single binary `e448b2e2…`, stream `949b9423…` | **Final headline after the k = 2 default flip (2026-09-14)** — current-main headline, essay |
| **23.29 tok/s** (1024 / 43.96 s decode, stepAvg 94.6–102.4 ms, acc/step 1.3782, depthDist 2:431) | `specdec-800`, **default config — pinned draft depth k = 2** (same as above), 5 measured reps, same binary, stream `139acb9d…` | **Final headline after the k = 2 default flip (2026-09-14)** — current-main headline, specdec |
| 19.63 tok/s (1024 / decode, depthDist 1:16, 2:176, 3:239) | `essay-1024`, pre-flip default (adaptive cost model, q4 head), 5 measured reps, binary `db39b916…`, stream `949b9423…` | Post-W4 Phase 2+3 session h-essay cell (2026-09-14) — **superseded-config** (the adaptive default was flipped to pinned k = 2 this session; the in-session k2 cell measured 21.28) |
| 21.11 tok/s (depthDist 1:1, 2:133, 3:250) | `specdec-800`, pre-flip default (adaptive cost model, q4 head), 5 measured reps, same binary, stream `139acb9d…` | Post-W4 Phase 2+3 session h-specdec cell (2026-09-14) — **superseded-config** (in-session k2 cell measured 22.75) |
| 21.32 tok/s (1024 / 48.08 s decode, 111.36 ms mean step, acceptance 1.3759) | `essay-1024`, pre-flip default (adaptive cost model, **4-bit MTP head**), 5 measured reps, binary `c56ca6ea…`, stream `949b9423…` | W4 A/B matrix q4 cell (2026-09-14) — **superseded-session** (single A/B session, pre-flip default; the final post-flip headline row is the live essay number) |
| 23.45 tok/s (1024 / 43.83 s decode, acceptance 1.6693) | `specdec-800`, pre-flip default (adaptive cost model, **4-bit MTP head**), 5 measured reps, same binary, stream `139acb9d…` | W4 A/B matrix q4 cell (2026-09-14) — **superseded-session** (the final post-flip headline row is the live specdec number) |
| 19.90 tok/s (1024 / 51.47 s decode, 120.63 ms mean step, acceptance 1.4061) | `essay-1024`, default config, **BF16 head**, 5 measured reps, binary `11a8e61e…` | W3 sweep k3 cell (2026-09-14) — **superseded-config** (BF16-head state; the q4 default has been live since W4; the in-session W4 matrix BF16 cell measured 19.63) |
| 25.44 tok/s (1024 / 40.26 s decode, acceptance 1.6974) | `specdec-800`, default config, **BF16 head**, single probe rep, binary `11a8e61e…` | W3 specdec confirmation (2026-09-14) — **superseded-config** (BF16-head state; the in-session W4 matrix BF16 cell measured 22.11) |
| 16.64 tok/s (1024 / 61.66 s wall, 143.91 ms mean step) | `essay-1024` fixture, `MLX_COMPILED_DECODE=0` (compiled fast paths OFF) | Phase 3 B3 (2026-09-13) — **superseded-session** ablation cell, kept for provenance |
| 22.28 tok/s (1024 tok / 45.96 s decode, 120.86 ms steps) | `specdec-800` fixture, acceptance 1.6974 (2.70 tok/round), cooler session | 2d Item 3 final run — **superseded-session; pre-QMV-grid-fix binary**; the routed kernel never executed in that run |
| 16.55–16.89 tok/s (143.0–145.4 ms steps) | `essay-1024` fixture, acceptance 1.4061 (2.41 tok/round), sustained thermal load | Item A clean fusion matrix — **superseded-session**; fusion matrix retained for its in-session 2×2 comparison only |
| 17.55 / 18.92 tok/s (B1 essay / B2 specdec, per-rep 17.553/16.816/19.277 and 18.921/18.340/20.334) | dual-fixture re-baseline, default config of the era (BF16 head), 2026-09-13 session | Phase 3 re-baseline (2026-09-13) — **superseded-session** historical provenance (pre-QMV, pre-q4-head; see the Phase 3 section table) |

**Headline refresh (DONE 2026-09-14, twice):** first the W4 default (4-bit MTP head ON) superseded the W3 BF16-head rows; then the k = 2 default flip (this task) superseded the adaptive-default rows. The W3 k=2 essay cell (21.26 tok/s) was the fastest valid *forced-depth* essay configuration of its session but **failed the specdec determinism gate** (W3 section, pre-Bug-B-fix). Every row except the two current-main rows is historical provenance — one live number per fact; cross-session absolutes are labels, never conclusions.

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
40.6 ms (~39 %, **measured in isolation with per-call sync — flush-contaminated
upper bound on in-pipeline cost**, see the post-W4 amendment below and
PROFILE.md §7), verify `lm_head` 5.4 ms (~5 %, **measured**), gaps/
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

**Post-W4 amendment (2026-09-14):** the headbench numbers above are isolated
per-rep graph-build + synchronous-eval measurements — one GPU pipeline flush
between every timed call (the Item D flush-artifact class). W4's in-pipeline
A/B showed the isolated head-body collapse (16.38 → 1.52 ms under q4) does **not**
transfer 1:1 to in-pipeline `tEval` (moved within in-session noise, −4.01/−0.12
ms) while the total-step win was carried by `tGraphBuild` (−7.2/−7.8 ms). The
40.6 ms/round head-family bucket is therefore an **upper bound, not the
in-pipeline cost**; the true in-pipeline head cost is unestablished pending a
flush-free in-pipeline measurement. Nothing is deleted — labels only
(PROFILE.md §7).

### W3 — MTP draft-depth sweep (2026-09-14) — **correctness stop at depth 5; valid optimum k=2 (21.26 tok/s, conditional)**

**Protocol.** `benchmarks/run_w3sweep.sh` over `run_cell.sh` (fresh server per cell, port
18099, greedy temp 0, `enable_thinking: false`, max_tokens 1024, `finish_reason: length`,
`QWEN_MTP_STEP_TRACE=1`, `pmset -g therm` before every rep, no parallel work, no mid-sweep
rebuild, binary `11a8e61e…` identical across all cells — verified per-cell SHA in the
driver log). The committed stream must equal the fixture hash in every cell and rep
(accepted/proposed/rounds are free — they vary with depth — and are logged). 6 reps per
cell, rep 1 discarded as warmup, interleaved with rotating start cell. Cell configs:
k1 `QWEN_MTP_DRAFT_K=1`; k2 `QWEN_MTP_DRAFT_K=2`; k3 no env/no flag (adaptive cost model,
offered 3 — the **pre-flip production default**; since the post-W4 flip the production
default is pinned k = 2, and `QWEN_MTP_DRAFT_K=3` is the rollback knob); k4
`--spec-draft-n-max 8` + `QWEN_MTP_DRAFT_K=4`.

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

### Phase 1 — Bug A: discriminating test and verdict (2026-09-14) — **precision family, not a logic bug**

**Task.** The W3 gate found specdec k=2 (verify width 3) and k=4 (width 5) diverging from
the pinned stream `139acb9d…` on the BF16 head while k=1/k=3 matched. The discriminating
test separates two hypotheses: (a) precision — the bf16 accumulation order of the batched
M-row verify forward differs from the M=1 serial forward and flips the argmax at
knife-edge positions; (b) logic — a state inconsistency (KV, cache, row mapping) corrupts
the stream.

**Method.** Throwaway diagnostic (`Tests/MLXLMTests/Qwen38BugADiscriminatorTests.swift`,
engine `feature/postw4-queue`, marked NOT FOR MERGE — must not ship with the Phase 4 merge)
runs two trajectories per head state on specdec-800 (1024 tokens, greedy): a serial leg
(`generateRound(depth: 0)`) and a k=2 leg (`QWEN_MTP_DRAFT_K=2`), recording per-position
top-2 (id, value) from the round result — no engine instrumentation. The serial leg never
invokes the head, so it is head-state-invariant.

**Results** (per-position provenance: `/tmp/buga-q4.jsonl`, `/tmp/buga-bf16.jsonl`):

| leg | stream hash | note |
|---|---|---|
| serial (q4 head state) | `c70882fc40a2c22da9310e689a7b83c650ab16134aac912c2e8fe8e149bbc9e8` | head-invariant |
| serial (bf16 head state) | `c70882fc…` (identical) | no head leakage into the target path |
| k=2, q4 head | `139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac` | **reproduces the pinned reference exactly** |
| k=2, bf16 head | `06882d856267601f4e74d1ed971e04184d518a41041336ab1733f3a4c9022ff4` | **reproduces the W3 failure exactly** |

- **The pinned reference `139acb9d…` is not the serial greedy stream.** It is an MTP-path
  stream (originally the k=3 BF16 server run). The serial greedy stream is `c70882fc…`:
  35/1024 positions differ from `139acb9d…` on specdec, first at position 989 (q4 leg),
  then fully diverged (0 equal afterwards).
- **The first divergence is a knife-edge argmax flip in both states.** q4 pos 989: serial
  top-2 `[4779, 2193] = [21.0, 20.875]` (gap 0.125 = 2 ulp); verify row `[2193, 4779] =
  [21.0, 21.0]` — the verify-row logit for 2193 drifted +0.125 (exactly 2 ulp at this
  magnitude), creating an exact tie that the smaller-id-wins ordering flipped. bf16 pos
  973: serial `[58377, 8476] = [20.625, 20.375]` (gap 0.25); verify `[8476, 58377] =
  [20.5, 20.5]` — both tokens drifted ∓0.125 (2 ulp), again an exact tie and a flip.
- **Drift at agreeing positions (pre-flip):** top-1 values bit-exact at 979/988 (q4) and
  969/973 (bf16); max drift 0.125 (2 ulp); nothing larger before the first flip. No
  state-inconsistency signature — no gross one-shot corruption, no wrong-row mapping (the
  k=2 legs reconstruct exactly the known W3 hashes).
- **Knife-edge supply (serial stream, head-invariant):** gap ≤ 0.125 (2 ulp) at 9
  positions per 1024; gap < 0.25 (4 ulp) at 38; gap < 0.5 at 98; gap < 1.0 at 218. The eight
  sub-2-ulp positions (40, 236, 337, 352, 417, 775, 884, 943) are identical in both states
  and did not flip in either leg; the flip landed at a 2-ulp-gap position (q4) and a
  4-ulp-gap position (bf16).

**Verdict: precision family (branch a).** The batched M-row verify forward and the M=1
serial forward are different bf16 reduction orders of the same exact logits; the drift is
≤ 2 ulp and only matters at knife-edge positions (gap ≤ 2–4 ulp). The committed stream
absorbs drift only through batched rows (accepted-draft rows and the bonus row of a fully
accepted block); the rejection path rolls the window back and re-forwards a fresh M=1 block
whose last row is the next primary — serial-consistent. The W3 "even-k divergence" was a
**confound**: the gate compared MTP widths against a non-serial reference, and the
per-(width, head-state) acceptance pattern modulates which positions are computed in batched
rows versus M=1, so knife-edge flips land at different positions per config → different
hashes. There is no width-specific logic bug at k=2/k=4.

**Gate policy (binding).** The determinism gate is a **per-(fixture, config) stream hash**;
config = (head state, pinned draft k or default cost model, fusion env). **Never claim
bit-exactness across configs or against the serial stream** — the MTP path is serial in
exact arithmetic and differs from serial only at knife-edge positions. Known registries (1024 tokens; full hashes in `benchmarks/results/*.jsonl`).
**specdec-800:**

| config | hash | notes |
|---|---|---|
| serial (any head state) | `c70882fc40a2c22da9310e689a7b83c650ab16134aac912c2e8fe8e149bbc9e8` | MTP disabled; head out of the loop |
| k=1 / k=3, bf16 (W3) | `139acb9d…` | coincidentally the same stream as k=2-q4 — no flips land between them |
| k=2, bf16 (W3, pre-Bug-B-fix) | `06882d85…` | the only specdec config whose stream diverged from the others (the W3 knife-edge flips) |
| k=4, bf16 (W3) | `48728382…` | — |
| k=2, q4 | `139acb9d…` | Phase 2+3 session; the **current production default** |
| k=3, q4 | `139acb9d…` | Phase 2+3 session; rollback knob `QWEN_MTP_DRAFT_K=3` |
| default (adaptive cost model, q4) | `139acb9d…` | Phase 2+3 session; the pre-flip default (depthDist 1:1, 2:133, 3:250) |
| k=5 / k=6 / k=8, q4 | `139acb9d…` | Phase 4 deep-k gate session (post-fix) |

**essay-1024:**

| config | hash | notes |
|---|---|---|
| serial (q4) | `949b9423…` | Phase 2+3 serial-probe cell — identical to every measured essay MTP stream |
| k=1..4, bf16 (W3) | `949b9423…` | essay stream is config-invariant across the measured W3 configs (no knife-edge flips landed) |
| k=2 / k=3 / default, q4 | `949b9423…` | Phase 2+3 session (k=2 is the current production default) |
| k=5 / k=6 / k=8, q4 | `949b9423…` | Phase 4 deep-k gate session (post-fix) |

**Risk assessment (external golden-stream requirement).** No challenge/golden requirement is
documented in this repository. If one required bit-exact match to a serial-greedy golden
stream, the MTP path would be at risk: on specdec the first knife-edge flip lands at ~95–96%
of the 1024-token stream (positions 973–989) and the streams then stay fully diverged; ~9
positions per 1024 sit in the ≤ 2-ulp flip zone and ~38 in the < 4-ulp zone. A flip requires
the position-specific drift (0–2 ulp, determined by which batched row computed the
position) to cross the gap. No fix is implemented; candidate mitigations (fp32 logit readout
for committed rows, knife-edge M=1 tie re-check, or a per-config golden registry as the
acceptance criterion) require explicit approval before any of them is built.

### Phase 2 + 3 — headline re-measure (q4 default) and k=2 evaluation (2026-09-14)

**Protocol.** One ~1 h session (driver `benchmarks/run_postw4.sh`): six cells × 6 reps
interleaved in fixed order — `h-essay` (default config), `k2-essay`
(`QWEN_MTP_DRAFT_K=2`), `s-essay` (`--spec-draft-n-max 0`, serial), then the specdec-800
analogs `h-specdec` / `k2-specdec` / `s-specdec`. Single release binary `db39b916…`
across all cells; engine `91ab1c8` (throwaway discriminator test only — no engine code
change in this session); the new `EXPECT_HEAD=q4` gate active on every cell (fail-loud q4
default); `QWEN_MTP_STEP_TRACE=1`; max_tokens 1024, greedy, non-streaming; fresh server
per cell; rep 1 discarded per cell (warmup), 5 measured; `pmset -g therm` before every
rep; no parallel builds/tests; per-rep phase-sum gate and `head_gate` gate. All 36 cells:
deterministic (identical stream hash across measured reps), gate PASS, phaseSumOK,
`finish_reason=length`, `committed=1024`, `head_selected` q4. Records:
`benchmarks/results/postw4.jsonl`; session log `/tmp/postw4-session.log`.

| cell | med tok/s (reps 2–6) | range | stream hash | depthDist |
|---|---|---|---|---|
| h-essay (default) | **19.63** | 15.94–19.82 | `949b9423…` | 1:16, 2:176, 3:239 |
| k2-essay | **21.28** | 17.24–21.69 | `949b9423…` | 2:461 |
| s-essay (serial) | **16.26** | 13.06–16.55 | `949b9423…` | 0:1024 |
| h-specdec (default) | **21.11** | 17.06–22.34 | `139acb9d…` | 1:1, 2:133, 3:250 |
| k2-specdec | **22.75** | 17.84–23.24 | `139acb9d…` | 2:431 |
| s-specdec (serial) | **16.22** | 13.12–16.40 | `c70882fc…` | 0:1024 |

In-session deltas: k2 beats default on both fixtures (+9.9% essay, +7.8% specdec); default
beats serial +20.4% (essay) / +30.4% (specdec); k2 vs serial +31.0% / +40.2%. Matches the
W3 pattern (k = 2 beats the adaptive default; acceptance is the limiter).

**Hash-registry additions (1024 tokens, q4, this session):**

| config | essay-1024 | specdec-800 |
|---|---|---|
| serial | `949b9423…` | `c70882fc…` |
| k = 2 | `949b9423…` | `139acb9d…` |
| default (cost model) | `949b9423…` | `139acb9d…` |

On both fixtures the default committed stream equals the k = 2 stream exactly (the
default path's depth distribution is 1:16/2:176/3:239 essay and 1:1/2:133/3:250 specdec;
no knife-edge flip lands between the default and k = 2 streams on these fixtures, though
the depth paths differ — the equivalence is observed, not structural); the essay serial
stream is `949b9423…` as well, so the essay stream is config-invariant *across the
configs measured in this session* (serial = k2 = default), while the specdec stream
differs only in the knife-edge flip family (serial `c70882fc…` vs MTP `139acb9d…`; first
flip at token 989 per Phase 1).

**Serial probe (Phase 5 input):** in-pipeline serial floor 16.2–16.3 tok/s, tEval
59–62 ms/step (backbone + lm_head at M = 1, head fully out of the loop). Per-rep tEval
deltas: default − serial = 54.4 ms median (essay) / 61.2 ms (specdec); k2 − serial =
38–40 ms per rep on both fixtures (joint: 2 head steps + width-3 batched-forward effect).
Thermal drift: later reps within a cell slow (e.g. s-essay tEval 59.2 → 74.8 ms by rep 6);
medians absorb most of it; per the in-session-deltas rule only same-session comparisons are
conclusions.

### Phase 4 — Bug B: SDPA exactness chunk (2026-09-14) — **fix implemented; unit + model regression tests green; deep-k gate session in flight**

**Root cause (code inspection, confirmed by the W3 evidence).** This fork's
`attentionWithCacheUpdate` (`AttentionUtils.swift`) lacked the SDPA exactness chunk that
upstream MLXLLM carries. The Metal SDPA dispatch
(`mlx/backend/metal/scaled_dot_product_attention.cpp`) routes qL ≤ 8 to the fused vector
kernel only when `qL·gqa ≤ 32`; this model has gqa = 6, head_dim = 256 (not in the fused
full-path set {64, 80, 128}), so verify widths M = 6..9 (draft depths 5..8) fall to the
reference matmul → fp32-softmax → matmul fallback. That fallback's bf16 reduction order
drifts enough at logit magnitudes ~21 / top-2 gaps ~0.25 to flip many positions — the W3
gross corruption (first divergence at the first drafted token of the first verify round;
`da7bb159…`), not the knife-edge family. Widths M ≤ 5 stay on the vector kernel and show
only the Phase 1 knife-edge behavior (W3 k4 first divergence at token 1020 — knife-edge,
consistent). The GDN layers are T-independent per row (the scan resets per row), so the
corruption is isolated to the 16 full-attention layers.

**Fix (engine, feature branch).** `attentionWithCacheUpdate` gains the upstream-style
exactness chunk: when the verify block's last dimension L ∈ 6..9, `L·gqa > 32`, head_dim =
256, `5·gqa ≤ 32`, `cache.offset > 0`, and the mask is the symbolic `.causal`, the block
is split into a leading 5-row chunk and a trailing L−5 chunk, each with an incremental
`KVCacheSimple` scatter update (O(L)) followed by one `MLXFast.scaledDotProductAttention`
call on its causal window; outputs are concatenated along the sequence axis. Both chunks
stay on the fused vector kernel (chunk qL ≤ 5 → qL·gqa ≤ 30 ≤ 32). Causal alignment is
bottom-right per call: chunk-1 row j sees keys 0..offset+j, chunk-2 row j sees keys
0..offset+5+j — identical windows to the full-block causal mask. The gate is
geometry-specific: L ≤ 5 (the current default path, depths ≤ 4), prefill (offset = 0),
L > 9 (already fallback — unchanged), other models (head_dim 128 fails the gate), and
explicit array masks (their alignment is caller-defined, same rationale as upstream) all
fall through to the legacy single call. `KVCacheSimple.update` is a scatter write into a
pre-allocated buffer; two incremental updates compose to the same live rows as one full
update (buffer pad may differ — internal detail, no stream effect). The compiled decode
path (`Qwen35+FastPath.swift` → `attentionCacheStep`) also routes through this function;
decode (L = 1) is untouched (gate fails).

**Tests (engine, feature branch — merge candidates, except the Phase 1 throwaway):**

1. `Qwen38SDPAExactnessTests` (unit, new file): wall geometry (24 q heads / 4 kv heads /
   head_dim 256 / scale 1/16). `testWideVerifyBlockMatchesPromotedWindows` asserts
   bit-exact equality with an independently-driven promoted-windows reference (5-row prefix
   + L−5-row tail, one `.causal` SDPA call per chunk, independent cache) plus live cache
   state equality, for L ∈ {6,7,8,9} × offset ∈ {13, 29, 41, 57, 250} (250 crosses a
   buffer-growth boundary). Negative controls: L = 5, offset = 0, and `.array` mask all
   stay on the legacy single call (bit-exact against it). **All pass** (0.08 s).
2. `Qwen38MTPDiagnosticTests.testWideVerifyStaysInSerialFamily` (model-level, new test):
   serial (depth 0) vs depth 5 (verify width 6) on the same synthetic prompt, 256 tokens,
   greedy: **256/256 match, no divergence, match rate 1.0000** (pre-fix signature per W3:
   divergence at the first drafted token, ~0–40% match). Asserts `firstDivergence >= 16`
   and `matchRate >= 0.85`. Passes (suite 63.9 s including model load); the full
   `Qwen38MTPDiagnosticTests` suite is green.

**Deep-k gate (driver `benchmarks/run_deepk.sh`, in flight).** Three cells (d5/d6/d8-essay:
`QWEN_MTP_DRAFT_K=k` + `--spec-draft-n-max 8`, `EXPECT_HEAD=q4`) × 6 reps plus two
one-rep headline regate cells (`h-essay-regate`, `h-specdec-regate`) to confirm the fix
leaves the M ≤ 5 headline hashes invariant. Expected: per-config determinism; headline
hashes unchanged; d5/d6/d8 streams now in the serial family (knife-edge flips only, not
gross corruption). Records will land in `benchmarks/results/deepk.jsonl`.

**Throwaway:** `Tests/MLXLMTests/Qwen38BugADiscriminatorTests.swift` (Phase 1
discriminator) is removed before the final engine merge.

**Deep-k gate results (2026-09-14).** Protocol: same as Phase 2/3, single release
binary built with the fix, essay-1024, `QWEN_MTP_DRAFT_K=k` + `--spec-draft-n-max 8`,
`EXPECT_HEAD=q4`, 3 cells × 6 reps interleaved plus two one-rep headline regate cells.
Records: `benchmarks/results/deepk.jsonl`; session log `/tmp/deepk-session.log`.

| cell | med tok/s (reps 2–6) | stream hash | acc/step | stepAvg (reps 2–6) |
|---|---|---|---|---|
| d5-essay | **14.53** | `949b9423…` | 1.71 | 184–192 ms |
| d6-essay | **12.85** | `949b9423…` | 1.741 | 210–219 ms |
| d8-essay | **9.80** | `949b9423…` | 1.771 | 277–283 ms |
| h-essay-regate | 20.49 (one rep) | `949b9423…` | — | — |
| h-specdec-regate | 22.44 (one rep) | `139acb9d…` | — | — |

All cells deterministic, gate PASS, phaseSumOK, `committed=1024`. Three confirmations:
(1) the fix works — d5/d6/d8 essay streams are now **identical to the essay serial
stream** (`949b9423…`), no gross corruption (pre-fix `da7bb159…`) and no knife-edge
flip in 1024 tokens; (2) the fix leaves the M ≤ 5 path untouched — both headline
regate hashes match the pre-fix postw4-session values exactly; (3) deep drafts are
**net-negative on essay**: accepted/step plateaus at 1.71–1.77 (marginal draft
acceptance is low) while the verify-width cost grows superlinearly (stepAvg
185 → 212 → 283 ms), giving a monotonic 14.53 → 12.85 → 9.80 tok/s, all far below
k2's 21.28 and default's 19.63. Caveat: the sweep is essay-only; specdec (higher
k2 acceptance, 2.38 acc/step) could shift the depth optimum, but the acceptance
structure (diminishing marginal acceptance vs superlinear width cost) argues the
optimum stays shallow.

### Phase 5 — head-structure decision (2026-09-14) — **CLOSE the head-structure workstream**

**Decision: close** — no further head restructuring, deepening, or re-quantization
work. The q4 head stays as-is: KEEP verdict from W4, default ON, fail-loud required.
Evidence (all in-session, this queue):

1. **In-pipeline head cost is small.** The flush-contaminated isolated upper bounds
   (24.66–26.02 ms/round at d = 1) do not carry into the pipeline: the joint
   k2 − serial tEval delta is 38–40 ms/step and includes *both* 2 head steps *and*
   the width-3 batched-forward effect, so the head's share is well under the old
   upper bound. tGraphBuild runs ~3.5–4 ms/step with the head in the loop.
2. **Deeper drafts do not pay.** d5/d6/d8 essay: 14.53 / 12.85 / 9.80 tok/s vs 21.28
   at k2, with acc/step plateauing at ~1.7–1.8 — the head's marginal draft
   acceptance cannot amortize the superlinear verify-width cost.
3. **k = 2 is the operating sweet spot** on both fixtures (+31.0% essay / +40.2%
   specdec vs serial), and its committed stream equals the default adaptive
   stream on both fixtures — the default's extra offered depth is net-negative
   (+9.9% / +7.8% in k2's favor).
4. **Correctness is settled.** Bug A (knife-edge family) is a precision artifact of
   the batched verify order — no logic bug; Bug B (SDPA fallback at M ≥ 6) is fixed
   and regression-tested. The per-(fixture, config) hash registry is the standing
   acceptance criterion.

**Standing recommendation — IMPLEMENTED 2026-09-14 (the k = 2 default flip task):** the
production default draft depth is now **pinned k = 2** (`Qwen38MTPBlockSession.draftPolicy`;
engine commit on main). `QWEN_MTP_DRAFT_K` still overrides (k = 3 is the rollback knob,
reproducing the registered k = 3 streams); the `--spec-draft-n-max` offer cap bounds the
effective k. The adaptive cost model is retired from the default path and retained as a
documented research artifact. Post-flip verification: default cells reproduce the
registered `(k = 2, q4)` streams on both fixtures (`949b9423…` / `139acb9d…`); final
headline 21.89 essay / 23.29 specdec tok/s (headline table).

### Benchmarking history (superseded results — kept for provenance only)

The v1.0-baseline, 2a, 2b, and 2c tables previously in this file are **retracted as measurement artifacts**, for two reasons established on 2026-09-13:

1. **Prompt confound.** The 2c baseline ran the 38-token essay prompt (599/1086/426, 1.4061) while the 2d Item 1/2/3 runs ran the 62-token specdec prompt (645/1008/380, 1.6974). The reported "2c → Item 1 +6.19 ms QKV regression" and the "TTLT win" were cross-prompt comparisons. Provenance was established with a rolled-back all-fusion-off build (`benchmarks/results/resolve.jsonl`): essay reproduces 599/1086/426 exactly; specdec-800 reproduces 645/1008/380 exactly. Their absolute step latencies (~116–124 ms) additionally reflect a cooler thermal session and are not comparable to later runs.
2. **Binary-vintage / dead-kernel confound.** The 2a–2d checkpoints predate the QMV dispatch grid fix, and the routed kernel was never exercised in-model (decode reaches it only via 2-D `x`; verify falls back to `quantizedMM` by the `ndim == 2` guard). Correctness was preserved throughout because the eager/fallback paths carried those runs.

The engineering work of those checkpoints (kernels, fusions, tests, refactors) stands as described above; only their **timing tables** are invalid. The headline-throughput and fusion-matrix tables above are the only valid comparisons.

## Current status and roadmap

**Done:** **K=2 round decomposition** (DONE 2026-09-14 — non-backbone overhead of the pinned k = 2 round decomposed to ~90% real GPU work (2 extra verify rows + 3-row head flush + 1 chain step) and ~10% host tape build; eval-window utilization 98.4% at 256 ctx, no reclaimable host gap; FullBench per-rep first-decode penalty root-caused as a bench artifact and retracted as a model property; deliverable `benchmarks/PROFILE-K2.md`; detail: K=2 decomposition section at the end of this file), Checkpoints 1, 2a, 2b, 2b-fix, 2c, 2d (both packed projections, merged to main in both repos), Item C interleaved layout (implemented, measured, rejected; removed from the engine in `901d2ca`), `qmvbench` microbenchmark target, prompt-fixture + determinism benchmarking protocol, MLXFast grid-convention bug fix, Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13; per-rep thermal logging wired into the harness), **Item D verify-pass QMV routing** (implemented; **final classification: +12.2% win, default ON** — the original A/B null was the `asData` flush artifact, root-caused; the flush-free rerun kept D1; engine `b900aad` → `c87fc6b` → `a5f102f`, server `4ca9589`+`4af4e73` → `31032ec` + evidence commits; detail: Phase 3 Item D section), **W5 `qmvbench` sustained-throughput mode** (implemented + measured; engine `a5f102f`), **W2 tEval profile** (DONE 2026-09-14 — `benchmarks/PROFILE.md`; `headbench` tool added to the engine; W4 trigger MET; headline refreshed), **W3 draft-depth sweep** (DONE 2026-09-14 — k=1..4 valid and bit-exact on essay; **k≥5 correctness stop** (top open item); essay optimum k=2 21.26 tok/s, conditional on the specdec k=2/k4 divergence; specdec: only k=1 and default k=3 bit-exact), **W4 MTP-head 4-bit quantization** (DONE 2026-09-14 — 4-bit head tree 238.9 MB generated; A/B matrix 24/24 reps bit-exact; essay +8.6 % / specdec +6.1 %; **verdict KEEP, `MLX_QWEN_MTP_HEAD_QUANT` default flipped ON**; detail: W4 section), **Phase 0 harness hardening** (DONE 2026-09-14 — `EXPECT_HEAD` gate in `run_cell.sh`/`run_matrix.sh`; q4 default fail-loud; PROFILE.md §7 flush-contamination label), **Phase 1 Bug A discriminating test** (DONE 2026-09-14 — **verdict: precision family, not a logic bug**; the pinned `139acb9d…` reference is an MTP-path stream, not serial greedy (`c70882fc…`); per-(fixture, config) stream-hash gate policy now binding; detail: Phase 1 section), **Phases 2+3 headline re-measure + k=2 evaluation** (DONE 2026-09-14 — 36-cell session, all deterministic; headline q4 default essay 19.63 / specdec 21.11 tok/s; k2 21.28 / 22.75; serial 16.26 / 16.22; detail: Phase 2+3 section), **Phase 4 Bug B fix** (DONE 2026-09-14 — SDPA exactness chunk in `attentionWithCacheUpdate`; unit + model regression tests green; deep-k gate: d5/d6/d8 essay now serial-identical (`949b9423…`), headline hashes invariant; detail: Phase 4 section), **Phase 5 head-structure decision** (DONE 2026-09-14 — **CLOSE the head-structure workstream**; q4 head stays as-is; `QWEN_MTP_DRAFT_K=2` recommended for this fixture class; detail: Phase 5 section), **k = 2 default flip + final headline** (DONE 2026-09-14 — production default draft depth pinned k = 2, engine `609e0d5`; diagnostic tests pin their session policy to the offered verify width; server load-time `MTP draft depth: k=…` log line + help-text fixes; `QWEN_MTP_DRAFT_K=3` rollback knob verified bit-exact; post-flip gate 4/4 cells PASS; final headline 21.89 essay / 23.29 specdec tok/s, 12/12 headline cells deterministic, single binary `e448b2e2…`; stale-claim sweep across progress/PROFILE/README/HANDOFF; detail: headline table + this task's entry below), **Verify tape profile** (DONE 2026-09-14, negative result — the 71.60 ms verify tape profiled at command-buffer granularity: 48 GDN layers 54.66 ms (74.5 %), 16 full-attention layers 16.93 ms (23.1 %), lm_head 3.94 ms (5.4 %), inter-CB gaps 1.32 ms (1.8 %); bottleneck is DRAM-bound 4-bit weight streaming at 205–257 GB/s ≈ machine peak; all four candidate branches dead (layout not bit-exact / in prebuilt MLX, row-batching already done, norm fusion already done, graph caching saves 0 ms of the eval window); no ≥8 ms lever in this checkout; deliverable `benchmarks/PROFILE-K2.md` §9; detail: Verify tape profile section at the end of this file), **Phase A MTP exactness audit** (DONE 2026-09-15 — F1 penalty-depth fix; A4 math extracted + 8 pure tests + distribution harness; contract `benchmarks/MTP-CORRECTNESS-CONTRACT.md`; detail: 2026-09-15 Phase A section), **Phase B RAM token-prefix cache** (DONE 2026-09-15 — `RadixKVCacheManager` namespace + byte budget + metrics; B3 tests; B4 TTFT fixture; `benchmarks/PREFIX-CACHE.md`; detail: 2026-09-15 Phase B section), **Phase C OpenAI-compatible tool calling** (DONE 2026-09-15 — finish_reason `tool_calls`; `tool_choice` required/named + `parallel_tool_calls: true` rejected 400; tool schema/conversation validation before model execution; tokenization-cache key includes tool_calls; XML parameter newline fix; `benchmarks/TOOL-CALLING.md` + `docs/TOOL-PROTOCOL.md`; 43 new tests, full `HTTPServerTests` 168 green; detail: 2026-09-15 Phase C section), **Phase D session API + token-ID echo** (DONE 2026-09-15), **Phase E draft-depth calibration** (DONE 2026-09-15), **Phase F fused GDN prework kernel** (DONE 2026-09-15 — `MLX_QWEN_FUSED_GDN`, default OFF, bit-exact, opt-in), **Phase G online adaptive draft depth** (DONE 2026-09-15 — server-only policy, default OFF), **Phase H long-context benchmarking** (DONE 2026-09-15 — 8K/32K/96K profiled; 32K/96K infeasible on the *dense* prefill path; `docs/LONG-CONTEXT-BENCHMARKS.md`), **Phase I chunked causal prefill** (DONE 2026-09-15 — **the long-context solution**: `MLX_CHUNKED_PREFILL`, default OFF, bit-exact; bounds the quadratic `[L,L]` scores buffer so 32K–64K context completes where dense traps; server admission models the transient buffer and rejects oversized dense prefills with HTTP 507 / `prefill_buffer_exceeded`; engine `62c4ac7`, server `37f6c6c`; detail: 2026-09-15 Phase I section + `docs/CHUNKED-PREFILL.md`), **Flash-attention feasibility** (DONE 2026-09-15 — **NOT integrated**: a flash kernel's online softmax is not bit-exact (Phase 1 Bug A), and chunked prefill already enables 128K+; prefill is near-optimal at `prefillChunkSize=512` (wall 119 s; pc=8192 is 55% slower, pc=0 traps); NOTE: an early "0.04%" attention figure was CPU *enqueue* time, not GPU time — invalidated; the per-phase GPU split is unmeasured; server `c16f166` + correction; detail: `docs/FLASH-ATTENTION.md`), **pc=0 single-pass prefill trap fix** (DONE 2026-09-15 — the `--prefill-chunk-size 0` (single-pass) path produced an empty chunk (`start=0, end=min(0+0, count)=0`) and fatal-`[reshape]`'d on an empty array; extracted the chunk partition into a pure `Qwen38MTPBlockSession.prefillChunkRanges(count:chunkSize:)` and guarded the `chunkSize==0` case to one full-range chunk; **default pc=512 path is byte-identical** (the `chunkSize>0` branch of the new function is mathematically the old inline loop); 4 unit tests + real-model validation: pc=0 no longer traps at 8K/16K/32K, bit-exact with pc=512 at 8K/16K, and 32K diverges at the first token (the expected Phase 1 Bug A FP-accumulation-order sensitivity to the prefill split — the engine chunked-SDPA gate engages at L>4096 for pc=0 but not per-512-chunk for pc=512); pc=0 is slower than pc=512 (205 s vs 117 s at 32K), so it is a correctness/robustness fix, not a perf change; engine-only; detail: 2026-09-15 pc=0 section), **32K prefill per-phase GPU breakdown** (DONE 2026-09-15 — **profiling only, no code change**: eval-synchronized wall-clock timing (not CPU enqueue), `MLX_CHUNKED_PREFILL=1`, pc=512, gated on L>100 to isolate the prefill from MTP verify; breakdown **FFN ~50% / GDN block ~27% / full-attention block ~23% / other ~0%**; top cost centers: FFN (largest), GDN block, full-attention block — all O(L) 4-bit GEMM/scan, none the O(L²) attention; full-attn QKV/SDPA/O sub-split not captured (compiled fast path, not the eager chain); engine left byte-identical to main; detail: `docs/PREFILL-PROFILE.md`), **32K+ prefill validation (per-phase + memory + bit-exactness)** (DONE 2026-09-15 — **profiling/validation only, no code change**: eval-synchronized GPU timing with the full-attention sub-split captured via the compiled fast path; **SDPA is 18.5 % of the 32K prefill and 29.3 % at 64K** (grows O(L²); the prefill uses the dense unfused path) — this **corrects** the `docs/FLASH-ATTENTION.md` "~0.04 %" figure (CPU enqueue, not GPU time); FFN ~43–49 %, GDN ~21–25 %; pc=512 reproduced near-optimal (132 s @32K), pc=0 is 65 % slower (218 s, robustness baseline only); 64K completes (333 s); 8K/16K chunked (pc=512 and pc=0) bit-exact with dense, 32K pc=0 diverges at token 1 (expected Phase 1 Bug A); peak RSS 13.4–14.6 GB; engine left byte-identical to main; detail: `benchmarks/results/prefill-verify-2026-09-15/REPORT.md`), **LCP long-context prefill optimization P1/P2/P3** (DONE 2026-09-16 — profile-guided prefill optimizations behind two default-OFF toggles, bit-exact across P1/P2/P3 matrices incl. chunked vs dense at 8K/16K; reports `benchmarks/results/prefill-opt-20260915/P{1,2,3}_*.md`, `docs/PREFILL-PROFILE-INDEX.md`; server `ac9e019`, `d1ec725`; detail: "LCP complete (2026-09-16)" section at the end of this file), **Task 7 CLI-flag dispatch crash diagnosis + startup flag validation** (DONE 2026-09-16 — the `--kv-ssd-*` flags never existed anywhere (zero occurrences in server sources, engine fork, docs, git history); the crash was unknown-flag leakage into `Environment.detect(arguments:)`; every implemented flag was already correctly stripped; fix: `ServerConfig.unknownFlagError(in:)` rejects unknown flags loudly at startup (stderr + `exit(2)`) before Vapor dispatch; `vaporArguments(from:)` made pure + `ServerConfigArgumentTests` (9 tests); server `76054e3`; detail: "Task 7" section at the end of this file), **Task 1 compact-space rejection walk** (2026-09-16, NEGATIVE RESULT — gate not met (2.05% slower vs the ≥3% bar), production change not merged; kept artifacts on `main`: distributional chi-square/TV tests, `WeightTestLock.swift` weight-test guard, A/B runner + fixture, `QWEN_MLX_SEED` deterministic-RNG hook; detail: "Task 1" section at the end of this file), **Task 4 in-RAM radix reusable-path repair** (2026-09-16 — same-prompt repeats now reuse the prompt-boundary KV state: freeze `exportState()` + copied caches at the prompt boundary and store that; per-request `reusedPrefixTokens`/`radixPrefillSkipped` + `prefix_reuse_*` metrics; E2E gate PASS (warm/cold 0.0221 in this checkout, outputs token-identical); detail: "Task 4" section at the end of this file), **Task 3 radix-tree SSD persistence** (2026-09-16 — cold-disk tier beneath the in-RAM radix KV cache: `RadixSSDStore` safetensors nodes + prefix-set index, `--kv-ssd-*` flags + `QWEN_KV_SSD_*` env, lazy tensor load on a disk hit, graceful-shutdown flush; first gate NEGATIVE (disk/cold 1.00×) on the pre-existing reusable-path bug fixed by Task 4; detail: "Task 3" section at the end of this file), **Task 5 SSD gate re-test** (2026-09-16 — GATE UNSTABLE: disk/cold consistently PASS (0.06×–0.20×) but disk/warm flips on the small thermally-variable warm denominator; stop condition → not merged; detail: "Task 5" section at the end of this file), **Task 6 SSD gate re-spec + equilibration** (2026-09-16 — gate re-specified to G1 disk/cold ≤ 0.7× + G2 abs disk ≤ 1.5 s (G3 disk/warm informational); equilibrated 7-cycle re-measurement PASSES (G1 0.117×, G2 0.609 s) — **MERGED to `main`**; detail: "Task 6" section at the end of this file).

**Current test count (2026-09-16):** `swift test --filter HTTPServerTests` = **237 Swift Testing tests in 7 suites + 120 XCTest tests, 0 failures** (default invocation; weight-gated tests skip unless `QWEN_RUN_WEIGHT_TESTS=1`).

**Decisions on record:**

- Fused W_qkv and W_gate+up: keep, default ON (latency-neutral, bit-exact, net-0 memory, rollback knobs `MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU`).
- Interleaved gate+up layout: rejected (no gain, +6.5 GB row-gather copies); removed from the engine in `901d2ca` (2026-09-13). Historical data in `benchmarks/FUSION_REPORT.md` and `benchmarks/results/itemC.jsonl`.
- Item D verify-routing knob `MLX_QWEN_QMV_VERIFY`: **KEEP, default ON** (2026-09-14; final classification in the Phase 3 Item D section). The original A/B was NULL due to a measurement artifact: the guard's per-call `asData` stride probe calls `self.eval()` on every routed dispatch, serializing the verify pipeline and shifting ~119 ms of GPU wait from the tEval phase into the graphBuild phase. After replacing the probe with a cached per-shape contiguity decision (`Qwen35RowMajorCache`, engine `c87fc6b`), the flush-free rerun shows **+16.659 ms/step mean (12.2%), 5/5 paired reps, bit-exact, 99.1% routed, zero materializations, phase-sums exact in all 12 reps** — the qmvbench 15–20% kernel win at M = 2..4 does transfer end-to-end. Default is now ON; `MLX_QWEN_QMV_VERIFY=0` is the rollback knob (selects the pre-Item D baseline for A/B cells). Per-rep provenance for both the invalid original A/B and the valid flush-free rerun is in the `benchmarks/results/itemd-*.jsonl` records.
- W1 hot-path `asData` audit: all call sites classified (one hidden flush — fixed; two dead-code probes — fixed with the same pattern; `UserInput` non-hot-path; MTP session readouts are intentional post-`eval` host reads). No remaining per-call tensor metadata in the forward path.
- W4 MTP-head 4-bit quantization: **KEEP, default ON** (2026-09-14). The pinned head was BF16 (849.4 MB) — W2's headbench measured 24.66 / 45.94 ms/round at d=1/2 (trigger MET). The 4-bit group-64 tree (`mtp-head/q4/`, 238.9 MB, `benchmarks/make_q4_head.py`) A/B'd against the BF16 head at k=3 on both fixtures: **essay 19.63 → 21.32 tok/s (+8.6 %), specdec 22.11 → 23.45 tok/s (+6.1 %), 24/24 reps bit-exact, head fusion engaged (`head swiGLU 1 qkv 1`), zero materializations, phase-sums exact.** Win carried by tGraphBuild (−7.2/−7.8 ms); tEval within noise; the isolated head-body collapse (16.38 → 1.52 ms/forward) does not transfer 1:1 in-pipeline (recorded observation). `MLX_QWEN_MTP_HEAD_QUANT`: unset = default ON (loud BF16 fallback if the q4 tree is missing), `1` force q4, `0` rollback BF16. `lm_head` report-only (already 4-bit, 635.7 MB payload — quantizing it changes committed tokens). The diagnostic's short prompts are width-sensitive within a fixed head state (same bug A/B family), so they are recorded, not gated; the A/B matrix at k=3 is the committed-stream gate.
- MTP-head SwiGLU/QKV fusions: engaged automatically by the W4 4-bit head (stock `QuantizedLinear` eligibility); the BF16 head remains eager-fallback by design. No separate task needed.
- Bug B fix (Phase 4, 2026-09-14): the SDPA exactness chunk is geometry-gated (L ∈ 6..9, `L·gqa > 32`, head_dim 256, `5·gqa ≤ 32`, `offset > 0`, symbolic `.causal` mask) so it cannot change any M ≤ 5 path, prefill, other models, or explicit-mask callers — confirmed by negative-control unit tests and by the invariant headline hashes in the deep-k regate cells. `QWEN_MTP_DRAFT_K`/`--spec-draft-n-max` above 4 is correct again but net-negative on essay; no default change.
- Phase 5 (2026-09-14): the head-structure workstream is **closed**. The q4 head stays as-is (W4 KEEP); in-pipeline head cost is well under the flush-contaminated isolated upper bounds; deeper drafts do not amortize the verify-width cost (acc/step plateau ~1.7–1.8); k = 2 is the recommended operating point for this fixture class. Any future re-scope (e.g. specdec deep-k) is a new task with its own A/B session.
- k = 2 default flip (2026-09-14): the production default draft depth is **pinned k = 2** (`Qwen38MTPBlockSession.draftPolicy`, engine `609e0d5`). Evidence: post-W4 Phase 2+3 session (k2 21.28/22.75 vs default 19.63/21.11 tok/s, in-session) and Phase 5's closure findings. The adaptive cost model is retired from the default path (retained as a documented research artifact — `costModelDepth`). `QWEN_MTP_DRAFT_K` overrides (`k = 3` is the rollback knob, verified bit-exact against the registered k = 3 streams on both fixtures in the post-flip gate); the `--spec-draft-n-max` offer cap bounds the effective k. The diagnostic acceptance test and the wide-verify serial-family test pin their session policy to the offered depth so they exercise exactly their intended verify widths (5 and 6) regardless of the production default. Server: load-time `MLXLM: MTP draft depth: k=…` log line (engagement proof), help text fixed (`--spec-draft-n-max` default 3, not 8), `run_cell.sh` records the `draft_depth` field. Post-flip gate: 4/4 one-shot cells PASS (default k=2 + rollback k=3, both fixtures, registered hashes, head gate q4, phaseSumOK); final headline: 21.89 essay / 23.29 specdec tok/s (12/12 cells deterministic, single binary `e448b2e2…`).

**Open items (in priority order — W1–W5 and Phases 0–5 all done 2026-09-14, the k = 2 default flip done 2026-09-14; the post-W4 queue is complete):**

1. **Correctness bug A — RESOLVED 2026-09-14 (Phase 1): precision family, not a logic bug.** The W3 even-k divergence was a confound of a non-serial reference: the batched-verify forward is a different bf16 reduction order than the M=1 serial forward (drift ≤ 2 ulp) and flips the argmax only at knife-edge positions (top-2 gap ≤ 2–4 ulp; ~9 per 1024 on specdec, first flip at ~95–96 %). k=2 on the q4 head reproduces the pinned `139acb9d…` exactly; bf16 k=2 reproduces the W3 `06882d85…` exactly; the serial stream is `c70882fc…` (head-invariant). Binding gate policy: per-(fixture, config) stream hashes — never claim cross-config or against-serial bit-exactness. No code fix; no challenge-specific change without explicit approval (risk assessment in the Phase 1 section).
2. **Correctness bug B — RESOLVED 2026-09-14 (Phase 4): SDPA exactness chunk implemented.** Root cause confirmed: this fork's `attentionWithCacheUpdate` lacked the upstream exactness chunk; verify widths M = 6..9 (qL·gqa > 32, head_dim 256) fell to the reference matmul → fp32-softmax → matmul SDPA fallback, whose bf16 reduction order drifted enough to grossly corrupt wide-verify streams (W3 `da7bb159…`, first divergence at the first drafted token). The chunk splits L ∈ 6..9 blocks into a 5-row + (L−5)-row pair of vector-kernel `.causal` SDPA calls with incremental KV scatter updates — bit-exact against the promoted-windows reference (unit test, L × offset matrix incl. a buffer-growth boundary) and the width-6 model stream is now serial-identical over 256 tokens (model-level regression test). Deep-k gate: d5/d6/d8 essay deterministic and **serial-identical** (`949b9423…`); headline (M ≤ 5) hashes invariant. The `--spec-draft-n-max` / `QWEN_MTP_DRAFT_K` surface above 4 is correct again — and net-negative on essay (14.53/12.85/9.80 tok/s vs 21.28 at k2), so k = 2 remains the recommended operating point.
3. **W5 — `qmvbench --throughput N`: DONE 2026-09-14** — sustained throughput measured at M ∈ {1,2,4,8,9}, narrow/wide × routed/fallback (table in the QMV microbenchmark section): ~170–235 µs/call sync overhead in the serialized protocol; the routed kernel's M = 2..9 win grows to ~30% under sustained conditions; M = 1 is a wash. The M = 16/17 extension is **closed, untriggered**: it was contingent on the sweep optimum reaching the sweep ceiling (k ≥ 5); the Phase 4 deep-k gate closed that trigger (d5/d6/d8 net-negative on essay, acc/step plateau ~1.7–1.8, serial-identical streams) and the k = 2 default flip makes the deep-verify widths non-default.
4. **Thermal control for benchmarks** — per-rep `pmset -g therm` logging is wired; shorter run blocks / cooldowns remain, so absolute numbers become comparable across sessions.
5. **Attention-layer kernels and acceptance-rate work — open, no numeric target.** The `tEvalMs` 24 ms / 30 tok/s target is formally retired (2026-09-14): it was set in the v1.1 planning era against a different head state, a pre-qmv-verify binary, and a pre-q4-head configuration, and is not a property of the current build. The honest current framing: decode is GPU-bandwidth-bound on the 14.4 GB 4-bit weight stream (measured in-pipeline bandwidth 200–275 GB/s vs 310–355 GB/s sustained in qmvbench); stepAvg ≈ 100 ms / ≈ 21–23 tok/s at the pinned k = 2 default. If this item proceeds, it re-scopes its own success criterion against the current build — the historical 24 ms figure is provenance only.
6. **Re-sweep draft depths k ∈ {5,6,8} after the item-2 fix — DONE 2026-09-14 (Phase 4 deep-k gate).** d5/d6/d8 essay: 14.53 / 12.85 / 9.80 tok/s, all deterministic, all serial-identical (`949b9423…`); acc/step plateaus at ~1.7–1.8 while stepAvg grows superlinearly (185 → 212 → 283 ms). Deep drafts are net-negative on essay; the sweep was essay-only (specdec caveat recorded in the Phase 4 section).
7. **Long-context support (32K–64K) — SOLVED by chunked prefill (Phase I, 2026-09-15).** The Phase H finding that 32K/96K are infeasible applied to the *dense* prefill path (quadratic `[L,L]` scores buffer traps the process at ~24K). Chunked causal prefill (`MLX_CHUNKED_PREFILL=1`, default OFF, bit-exact) bounds that buffer to `tile × L` (linear), so **32K–64K context is now supported** on this hardware (32K measured end-to-end at 175.9 s; 64K fits the transient-buffer budget at ~1.6 GB). The default (dense) build is unchanged and still rejects/traps above ~24K; long-context deployments must set `MLX_CHUNKED_PREFILL=1` and the server's admission control will then admit (and bound) long prompts, rejecting oversized dense ones with HTTP 507 / `prefill_buffer_exceeded`. Flash attention is **not** required — chunked tiling of the existing fused SDPA kernel is sufficient for this use case.
## K=2 round decomposition (2026-09-14)

Task: decompose the ~50 ms/round of non-backbone overhead in the pinned k = 2
speculative round and close with a kernel-addressability verdict. Deliverable:
`benchmarks/PROFILE-K2.md`.

**Method.** Four phases: (1) code-read the round's natural sync boundaries (16 host
segments A–P; the single blocking `eval` is the only device→host sync; the head
flush+chain is `asyncEval`'d before it so head GPU work overlaps verify-tape
construction). (2) Algebraic cells: 4 configs (s/k1/k2/k3) × 2 fixtures × 6 reps,
interleaved per rep, hash-gated (k2/k3 reproduce the registered streams; s/k1
in-session 6-rep determinism), plus a 4-pair A/B proving the trace instrumentation
is +2.19 ms/round. (3) xctrace Metal System Trace of 110 k = 2 rounds at 256
context, offset-calibrated against `mtp-anchor` uptime-ns probes (offset
696,415,092,781,278; trace-relative GPU `start-time`). (4) `FullBench` diagnostic
tool in the engine repo (per-rep width matrix, serial mode, SDPA micro-bench,
anchoress for trace alignment).

**Result (same context, 256 tokens).** Serial M = 1: 59.4 ms round (GPU busy 55.6,
util 96.0%). k = 2: 86.0 ms round (tG 4.5 + tE 80.4 (GPU busy 79.8, util 98.4%) +
tC 1.1 + tH 0.02). Δ = 26.6 ms: ~24.2 ms real extra GPU work (2 verify rows + head
flush/chain tail), ~3.5 ms host (tape build in MLX's C++ flush + accept walk), no
reclaimable host gap. In-pipeline algebra (12 records/config, growing context):
s 59.78 / k1 90.08 / k2 100.73 / k3 126.17 ms stepAvg; k1 carries the dead-path
repair +13.17 ms tC (`rollbackCheckpoints` never written — evidence for a
follow-up, not fixed; k = 1 is not a production config). The "107–112 ms"
label is the warm end of the in-session thermal trajectory (k2default 6-rep:
89.5 → 102.5 ms essay); cold end ~86 ms.

**Findings.** (1) The k = 2 round is GPU-saturated at measured contexts — no
kernel-level win of order 10 ms is addressable from this checkout without changing
draft depth (rejected: acceptance cost + k1 dead path), head design, or MLX C++
(tape build, 3.4 ms tG). (5) **Head fusion explored and closed negative**
(2026-09-14 — fusing the MTP head into the backbone's last-layer graph is a
rearrangement, not a work removal: the head-family is 11.40 ms/round of real
weight streaming (HeadBench: flush 1.95 + step 1.47 + 2×proj1 7.98), 8.27 ms of
it sits in the eval window on the verify critical path (draft ids feed the
verify input), and 3.13 ms is already hidden behind the verify build; a fused
single graph loses that overlap and is predicted +3.1 ms/round; no engine
change made; detail: head fusion section below and PROFILE-K2.md §8). (2) QMV routing: M = 1 verify falls back to
`quantizedMM`; M ≥ 2 routes the QMV verify kernel — the M = 3 tape's sub-linear
per-row cost is a property of that path, not a bug. (3) **FullBench per-rep
first-decode penalty is a bench artifact**: newCache + prime + one measured decode
costs 117–156 ms at prime = 2048 while serial mode (one cache, N back-to-back
decodes) is flat 55.8 → 56.8 ms across ctx 256 → 3072 and matches the in-pipeline
serial cell (56.2 ms). The penalty is paid only on the first decode after each
prime (`--widths 1,1`: 122 ms then 56 ms, same cache), persists after 10 warmup
reps, and scales with prime size. xctrace of the slow first decode: 68 compute
intervals, union 60.5 ms, ~1.2 ms GPU-idle gaps, host cpu→gpu stride ~2.2 ms/kernel
vs ~0.8 ms/kernel — the host MLX flush/encode loop falls behind after a fresh large
prime + newCache cycle; mechanism is in MLX's C++ eval flush (not readable in this
checkout — the `mlx-swift` checkout is Swift-only). Isolated SDPA (0.3–0.5 ms,
K = 256–4096), cache slice update (O(1) host), GDN S = 1 update (single kernel, no
readback), and no fire-and-forget eval in the forward path all ruled out as the gap
source. **Prior "M = 1 non-monotonic in context" readings from per-rep FullBench are
retracted**; in-pipeline at real server contexts (2–8k) has no per-round
newCache/prime and is not expected to carry the penalty. (4) Instrumentation
artifact measured, not assumed: +2.19 ms/round (4-pair alternating A/B).

**Files.** Server: `benchmarks/PROFILE-K2.md` (new), `benchmarks/run_k2decomp.sh`,
`benchmarks/run_k2trace.sh` (new), `benchmarks/results/k2decomp.jsonl`,
`benchmarks/results/k2decomp-ab.jsonl`. Engine: `Libraries/FullBench/` (new
diagnostic tool), `Package.swift` (product). No production source changes in either
repo for this task.

## Head fusion exploration (2026-09-14, negative result)

**Objective.** Reduce the k = 2 eval window (80.4 ms) by ≥ 8 ms by fusing the
MTP head forward into the backbone's last-layer graph (one graph, one blocking
eval). **Verdict: NO WIN, not implemented.** The fusion is a rearrangement —
the head's GPU work is real weight streaming on the verify critical path — and
the fused-graph variant is predicted to regress ~3.1 ms/round.

**Measurements (binary `e448b2e2…`, stream hash `949b9423…` unchanged):**

1. HeadBench (engine, q4 head, d = 2, flush = 3, steady-state shape): flush
   1.95 ms + step 1.47 ms + 2 × proj1 7.98 ms = **11.40 ms/round head-family GPU
   work** (proj1 = single-row 4-bit backbone lm_head 635.7 MB + argmax, 3.99 ms
each; the autoregressive chain forces them sequential and single-row).
2. 5-way host trace (`MLX_QWEN_MTP_TRACE=1`, 461-round essay cell): d_head1 74 µs,
   d_chain 41 µs, verify_build 2,969 µs, eval_wall 79,870 µs, round 84,186 µs
   (stepAvg 84.18 ms). Head submitted 0.23 ms after round start → **3.13 ms
   hidden behind the verify build, 8.27 ms in the eval window**.
3. Cross-check: implied M = 3 tape = 79.87 − 8.27 = 71.60 ms = serial 58.30 +
   2 × 6.65 ms marginal row. Δ(k2 − serial) = 21.57 ms = 8.27 head + 13.30 rows.
   Per-round GPU util 98.6 % (busy 83.00 / wall 84.19 ms) — the host side is
   already fully hidden; the round is GPU-throughput-bound.

**Why the 8 ms bar is unreachable from this angle.** Even removing 100 % of the
head work (impossible — draft ids are verify inputs) caps the saving at 8.27 ms.
Any rearrangement conserves the 83.00 ms per-round GPU work; the only lever is
reducing GPU work itself: (a) model-level — a native 2-token head or
draft-vocabulary lm_head (≈ 12.1 ms/round for one step + proj1 + verify row);
(b) draft-depth policy (k = 1: −12.1 ms/round, −1.22 tokens/round); (c) the
verify tape (71.60 ms, backbone 4-bit weight streaming) — the separate
workstream that owns the eval window.

**Files.** Server: `benchmarks/PROFILE-K2.md` §8 (new), this section,
`docs/HANDOFF.md`. No source changes in either repo; no new binaries.
## Verify tape profile (2026-09-14, negative result)

**Objective.** Reduce the k = 2 verify tape (71.60 ms, 85 % of the 84.19 ms
round) by ≥ 10 ms via backbone 4-bit weight-streaming / kernel optimization,
bit-exact. **Verdict: NO WIN — no ≥8 ms lever exists in this checkout.** The
tape is DRAM-bound on the fixed ~15 GB 4-bit weight set at 205–257 GB/s
effective (≈ machine peak); every candidate branch is dead.

**Profile (command-buffer granularity, the finest the Metal System Trace
capture offers — per-shader intervals were not recorded; 18 steady rounds,
binary `e448b2e2…`, 256 ctx, same capture as PROFILE-K2 §3/§8):**

- Structure: one fused CB per layer (64 total) + lm_head CB + 3 small CBs.
  The entire layer (QKV + attention/GDN + MLP + norms) is a single CB — no
  norm/elementwise CBs exist in the tape (fusion already maximal:
  swiGLU 64/64, qkv 16/64, gdn 48/64).
- Per-layer-type bucket (medians): 48 GDN layers 1138.7 µs each = 54.66 ms
  (74.5 %); 16 full-attention layers 1058.3 µs each = 16.93 ms (23.1 %);
  lm_head (M=3, 635.7 MB 4-bit) 3935.7 µs = 3.94 ms (5.4 %); inter-CB gaps
  1.32 ms (1.8 %); 3 small CBs 0.05 ms (0.1 %). CB-measured tape busy median
  73.35 ms — consistent with the registered 71.60 ms within window-edge
  noise. FA layers are *cheaper* per layer than GDN layers at 256 ctx
  (attention over 3 queries × ~258 KV is trivial; GDN conv+scan costs more).
- Bottleneck: memory bandwidth. Per-layer effective BW ~205 GB/s (GDN),
  ~209 GB/s (FA), uniform across all 64 layers; M=1 in-pipeline 257 GB/s
  (FullBench serial 269 GB/s) is the practical peak; M=1 flat across ctx
  256→3072 (bandwidth-bound, not compute); tape = 58.30 (M=1 base) + 2 × 6.65
  (marginal rows), the marginal 6.65 ms/row being real row compute.

**Kill analysis of the candidate branches.** (1) Weight layout: dead — the
layout is the MLX quantized format consumed by prebuilt kernels; any change
alters fp32 accumulation order → breaks bit-exactness; the kernels are in
prebuilt MLX (not this checkout). (2) Row batching (M=3 as one dispatch):
already done — the tape is M=3 batched, one fused CB per layer. (3) Norm
fusion into matmul: already done — no norm/elementwise CBs in the tape.
(4) Graph caching: saves 0 ms of the eval window — the host build (2.97 ms)
is already hidden behind head GPU; the inter-CB gaps are 1.32 ms/round, not
graph build.

The only sub-8 ms inefficiency found: 1.32 ms/round inter-CB gaps + 0.05 ms
small CBs (total ~1.4 ms, ~2 % of the tape) — below the 8 ms bar by a factor
of ~6. The QMV M=3 effective-BW gap (205 vs 257 GB/s) is in prebuilt MLX and
is already accounted for in the 13.30 ms marginal-row cost.

**Next levers (all outside this checkout).** (a) Model-level: a 2-token
native head or draft-vocabulary lm_head (removes ~12.1 ms/round — the head +
one verify row, not the tape). (b) Model-level: smaller/denser backbone
quantization (fewer bytes streamed per forward). (c) MLX upstream: a faster
M=3 QMV kernel (close the 205→257 GB/s gap). (d) Draft-depth policy:
k=2→1 removes 12.1 ms/round at −1.22 tokens/round (a policy change, not a
kernel change).

**Baseline re-verification (this session).** One k = 2 cell (6 reps,
`--spec-draft-n-max 3`, essay, single stream) on binary `e448b2e2…`:
6/6 bit-exact `949b9423…`, 6/6 phaseSumOK, tEvalAvg 79.93 / 82.67 / 81.19 /
84.06 / 83.17 / 84.81 ms, avgStepMs 84.29 / 87.05 / 85.48 / 88.47 / 87.55 /
89.03 ms — rep 1 (cold) reproduces the registered 79.87 / 84.19; the rise is
thermal drift (no thermal warning level recorded).

**Files.** Server: `benchmarks/PROFILE-K2.md` §9 (new), this section,
`docs/HANDOFF.md`. No source changes in either repo; no new binaries.

## Draft-depth policy sweep (2026-09-15, NEGATIVE — k=2 retained)

**Objective.** Decide the speculative draft-depth policy (fixed k=2 vs other
fixed depth vs bounded adaptive) on end-to-end server measurements, closing
the last local lever from the verify-tape profile (lever d: "k=2→1 removes
12.1 ms/round at −1.22 tok/round"). Benchmark-and-decision task; no kernel,
head, quantization, weight, or MLX source changes.

**Protocol.** Full audit of the depth flag mapping (`--spec-draft-n-max` =
per-round offer, `QWEN_MTP_DRAFT_K` = pin, actual d = min(offer, pin ?? 2),
verify width M = d+1). Stage A recon (1 cold cell each) on essay1024 for
s/k1/k2/k3/k4/k6/k8 — all deterministic, k4+ steeply dominated → cut.
Stage B primary: s/k1/k2/k3 × 6 interleaved rotated reps (rep 1 warmup) on
essay1024. Stage C generalization: same 6-rep protocol on specdec1024 and on
the 128-token interactive regime (essay128, specdec128). Fresh server per
cell, single binary `e448b2e2…` (all 76 cells), q4 head gate, per-rep
determinism + phase-sum + constant-depth gates, thermal snapshot per cell.
New runner `benchmarks/run_dpsweep.sh`; `run_cell.sh` gained an optional
max-tokens argument (default 1024, existing behavior unchanged).

**Result — k=2 is best in all four modes; keep the default.**

| mode (median reps 2–6) | s | k1 | k2 | k3 | k4/k6/k8 (recon) |
|---|---|---|---|---|---|
| essay-1024 tok/s | 16.74 | 19.37 | **22.08** | 20.00 | 19.47 / 14.04 / 10.37 |
| specdec-1024 tok/s | 15.36 | 18.61 | **21.27** | 20.70 | — |
| essay-128 tok/s | 15.13 | 19.23 | **24.70** | 21.90 | — |
| specdec-128 tok/s | 16.77 | 19.99 | **25.04** | 18.77 | — |

k2 wins decode and wall time in 4/4 modes. k1's cheaper round never pays
(k2 beats k1 by 12–22 %). Diminishing returns begin at k3 (−9.4 % essay /
−2.7 % specdec, the latter inside noise and never ahead) and collapse at
k4/k6/k8. No meaningful TTFT difference (identical prefill; first-round
delta ≤ ~100 ms vs startup variance). No mode in which any other depth wins
→ no adaptive policy, no separate interactive/throughput mode. No candidate
clears the positive-change bar (≥ 3 % sustained beyond noise, bit-exact,
no interactive regression).

**New finding — near-tie width-family streams (pre-existing).** Greedy
streams are per verify-width family, not globally identical, on specdec-800:
M=1 `c70882fc…`, M=2 `a3dfa862…`, M=3/M=4 `139acb9d…` (registered). First
M=1 vs M=3 divergence at completion token 989/1024 (reproduced). Essay
families all agree for 1024 tokens; specdec 128-token prefixes all agree
(`6eb4c26a…`). Cause class: per-width accumulation-order ulps flipping a
rare near-tie argmax. Not a regression from this task; the registered k=2
product stream is unaffected; cross-width greedy identity is an open item
(kernel/numerics scope). Streams were gated per family in the sweep; times
unaffected.

**Files.** Server: `benchmarks/DRAFT-DEPTH-POLICY.md` (protocol, flag map,
per-family stream table, raw-result locations, result tables, noise caveats,
recommendation), `benchmarks/results/dpsweep-*.jsonl` (27 files, full
provenance per line), `benchmarks/run_dpsweep.sh`, `benchmarks/run_cell.sh`
(max-tokens arg), `docs/HANDOFF.md`. No source changes in either repo
(library sense); no new binaries; engine untouched.

## 2026-09-15: MTP exactness / width-consistency audit + RAM prefix cache (IN PROGRESS)

Branch `feature/mtp-exactness-prefix-cache` in both repos. Binary `e448b2e2`.

### Phase A audit findings (call-path map done)

- **Engine implements EXACT speculative rejection sampling at T>0**
  (`Qwen38MTPBlockSession.generateRound`, sampling.enabled = temperature > 0):
  drafts sampled from q = softmax(filters(penalties(headRow))), target
  p = softmax(filters(penalties(verifyRow))) with identical per-position
  penalty frequency history; accept r <= min(1, p/q); reject resamples
  max(0, p-q) (log-space, 1e-10 floor on zeros); bonus sampled from p. The
  server validation comment ("speculative sampling is not implemented") is
  STALE — doc drift to fix.
- **DEFECT F1**: T=0 + non-default penalties + mtp_enabled: the greedy draft
  round selects raw (unpenalized) target argmaxes; penalties apply only to
  the first primary after prefill. The documented contract (validation
  comment) claims serial-depth fallback, but the server never enforces it
  (`hasNonDefaultPenalties` is computed and unused in the generation path).
  Fix: force depth 0 when penalties are non-default (small, self-contained,
  provably correct; matches documented contract; serial penalty path is
  exactly correct).
- No per-request seed (global MLXRandom, unseeded per request; no `seed`
  API field); no logit bias; n>1 rejected. min_p supported [0,1].
- Rollback state: backbone KV (snapshot/trim + GDN recurrent checkpoint),
  head KV (trimTrimmable), tokenHistory (penalty history), pendingHidden/
  Primary/Top2 (repair forward); RNG not rolled back (unneeded for
  distributional correctness).
- **Phase B pre-existing**: `RadixKVCacheManager` already implements a
  token-prefix radix KV cache (LRU-leaf eviction under memory pressure,
  TTL, config-namespace match, store-on-success only, begin() clones into
  private session state = CoW). Phase B = audit/harden/benchmark: add
  identity namespacing, metrics, explicit byte budget; B3/B4 tests +
  TTFT benchmark.

### Status — Phase B (prefix cache) IN PROGRESS

- **Audit**: `RadixKVCacheManager` (radix-tree, token-prefix, store-on-success, CoW via begin clone, TTL, LRU-leaf eviction under global memory pressure). Critical constraint: recurrent (gated-delta) layers are NOT trimmable → a hit requires the stored history to be an EXACT token-prefix of the new seed → the TTFT win is same-thread multi-turn continuation, not interleaved shared-prefix.
- **Hardening**:
  - **Namespace**: `namespace: String` on `CacheEntry`/`RadixNode`; `matchPrefix` requires `node.namespace == namespace`; `MLXGenerator` threads `cacheNamespace` (model+head+template) into `matchPrefix` + `store`. Closes the latent cross-model serve gap (config was KV-geometry-only).
  - **Per-cache byte budget**: `RadixKVCacheManager(maxCacheBytes:)`; `store` evicts LRU until under budget; set to `memoryLimitBytes/4` in `MLXGenerator` (global pressure eviction remains as backstop).
  - **Metrics**: lifetime `hits`/`misses`/`evictions`/`stores` + `Metrics` snapshot (`entries`, `estimatedBytes`, `hitRate`).
- **B3 tests**: 3 new (namespaceMismatchMisses, metricsCount…, byteCapEvictsLRUOnStore). `RadixKVCacheManagerTests` 11 green; full `HTTPServerTests` 125 green.
- **B4 TTFT**: `benchmarks/prefix_ttft.py` + `run_prefix_ttft.sh` (multi-turn exact-prefix hit vs cold-miss control). [results pending release rebuild + run]
- **Doc**: `benchmarks/PREFIX-CACHE.md`.

### Status — Phase A (speculative exactness) COMPLETE

- **F1 fix (server)**: `SamplingParameters.effectiveMTPEnabled` = `mtpEnabled && !hasNonDefaultPenalties`; both `decodeDepth` sites + the MTP sampling-config construction use it. Non-default-penalty requests now run serial target-only (penalties applied to target). Server test added.
- **A4 math (engine)**: `acceptanceAlpha`, `residualLogits`, `applySamplingFilters` extracted as pure statics on `Qwen38MTPBlockSession`; 8 pure unit tests in `Qwen38MTPKernelTests` (acceptance zero-q floor, residual p-q support / p==q fallback / zero-prob negligible, filter temperature/top-k/top-p/min-p). Bit-identical to the inline behavior.
- **A4 distributional harness (engine)**: env-gated `testA4DistributionalParity` (QWEN_MTP_DIST_HARNESS=1) — 64 trials/mode at T=0.8/1.0, top-8 carried-mass Δ < 0.08 (gate 0.15). Confirms MTP rejection sampling preserves the target distribution end-to-end.
- **A3 width matrix**: covered by the prior draft-depth policy sweep (k=1..8 in serial family) — `DRAFT-DEPTH-POLICY.md`.
- **Contract**: `benchmarks/MTP-CORRECTNESS-CONTRACT.md` (greedy exact modulo near-tie ulp; stochastic exact distribution; penalties rejected; do-not-claim list; reproduction commands).
- **Dropped**: the global-MLXRandom seeded-categorical kernel test (tests MLX's RNG, races the diagnostic suite under concurrent cross-suite execution).
- **Gate**: engine `Qwen38MTPKernelTests` (10) + `Qwen38MTPDiagnosticTests` (2) green; server `HTTPServerTests` (122) green. Engine committed before server.

### Plan

A. F1 fix (server) + contract doc + A3 width matrix (engine diagnostic) +
   A4 unit tests (engine: extract acceptance/residual math, pure tests) +
   A4 distribution harness (MTP on/off, T=0.8/1.0, N trials, next-token +
   completion-length distributions).
B. Radix hardening (namespace, metrics, budget) + B3 tests + B4 TTFT
   fixtures (4K/16K shared prefix, multi-turn, cold-miss control).

Deliverables: benchmarks/MTP-CORRECTNESS-CONTRACT.md, benchmarks/PREFIX-CACHE.md,
fixtures + results, updated progress/HANDOFF, engine commits before server
commits, merge both to main.

## 2026-09-15: OpenAI-compatible tool calling (Phase C) — COMPLETE

Server-side function calling for the Qwen 3.8 MTP server. The server
**transports, renders, parses, and serializes** tool calls in an
OpenAI-compatible shape and **never executes** a tool. Audit (Phase 0) found
substantial tool-calling already on main (parser, serialization, render, SSE,
cache key, `--tools-enabled`); this change set closes the identified gaps.

### What was done (server repo only; engine untouched)

- **`finish_reason: "tool_calls"`**: shared `toolCallFinishReason` helper used
  by both streaming and non-streaming paths. ≥1 tool call + normal stop →
  `"tool_calls"`; `"length"`/`"memory_pressure"` preserved. Streaming now
  tracks emitted `toolCalls` (previously dropped on the `.toolCall` case).
- **`tool_choice` rejection**: `"required"` and named `{function:{name}}` are
  rejected 400 `unsupported_parameter` (the Qwen template cannot force a tool
  call); `"auto"`/`"none"` accepted.
- **`parallel_tool_calls`**: `true` rejected 400 `unsupported_parameter`;
  `false`/omit accepted as a no-op. Field added to `ChatCompletionRequest`.
- **Request validation (before model execution)**: `validateTools` (type
  `function`, name `[a-zA-Z0-9_-]` 1–64, unique, `parameters` an object, ≤128
  tools), `validateToolChoice`, `validateParallelToolCalls`, and
  `validateToolConversation` (every `tool`/`function` result references a prior
  assistant `tool_call` id — no orphans, no cross-turn mismatch; assistant ids
  unique). All return OpenAI-shaped 400s.
- **Tokenization cache key**: `MLXGenerator.tokenizationCacheKey` (extracted
  static) now includes assistant `tool_calls` (name|args) and tool-result
  `tool_call_id`/`name` — closes a cross-conversation tokenization-cache
  contamination gap (tool calls were omitted from the key).
- **XML parameter newline fix** (`StreamingToolCallParser`): the template puts
  each parameter value on its own line (`<parameter=K>\nVALUE\n</parameter>`);
  strip exactly one formatting newline per side, preserving internal newlines.
- **Docs**: `benchmarks/TOOL-CALLING.md` (internal contract),
  `docs/TOOL-PROTOCOL.md` (public client contract).

### Tests (pure Swift, no model weights; 43 new)

- `StreamingToolCallParserTests` (11): XML/JSON well-formed, empty args,
  multi-call index, reasoning separation, malformed→content, unterminated→
  content, tag-split safety, disabled passthrough, JSON-literal preservation.
- `ToolCallingValidationTests` (17): tool schema, `tool_choice` required/named
  rejection, `parallel_tool_calls` true rejection, orphaned/missing/duplicate
  tool_call_id, cross-turn mismatch.
- `ToolCallingRenderTests` (7): `tokenizationCacheKey` identity for tools vs
  no-tools, schemas, tool_choice, assistant tool_calls, tool_call_id, reasoning,
  enable_thinking, stability.
- `ToolCallingHTTPTests` (3): `toolCallFinishReason` stop→tool_calls, length/
  memory_pressure preserved, no-tool stop stays stop.
- `ToolCallingCacheTests` (4): tools vs no-tools disjoint prefixes, multi-turn
  reuse, branch fork, byte-cap eviction.
- Updated `RequestValidationTests` (3 tool tests) to include a prior assistant
  tool call (required by the new orphaned-result validation).
- **Full `HTTPServerTests`: 168 green** (baseline 125 + 43).

### Do-not-claim

- No tool execution (server parses/returns only).
- No forced tool selection (`tool_choice` required/named rejected, not emulated).
- No guaranteed parallelism (`parallel_tool_calls: true` rejected).
- Parse correctness is guaranteed; model adherence to the tool format is not.
- Reasoning-tag marker mismatch (parser `think`/`/think` vs template marker) is
  pre-existing and unchanged (affects reasoning identically with/without tools).

### Reproduction

```
swift build --target HTTPServer
swift test --filter HTTPServerTests   # 168 green
```

## 2026-09-15: Session API + token-ID echo (Phase D) — COMPLETE

Server-owned conversation sessions and an optional `include_token_ids`
response extension, so multi-turn clients can own the conversation history
server-side and (optionally) read back the exact committed token IDs.
**Server repo only; engine untouched** (no kernel, head, quantization, or
weight changes).

### Design gate (resolved before implementation)

- `docs/SESSION-BOUNDARIES.md`: the load-bearing token-boundary analysis. A
  session re-render shares a **full** token prefix in non-thinking mode and a
  **partial** prefix in thinking mode (integer-token evidence in
  `TokenBoundaryTests`, `QWEN_RUN_WEIGHTS=1`). Therefore a session **does not
  guarantee a cache hit**; reuse is opportunistic and identical in kind to a
  token-faithful stateless client. No token splicing, no Jinja-template
  duplication in Swift, no request-side `token_ids`.

### What was done (server repo only)

- **`include_token_ids`** (stateless + session): `include_token_ids: Bool?` on
  `ChatCompletionRequest`; when `true`, `choices[0].message.token_ids`
  (non-streaming) / final delta `token_ids` (streaming) carry the exact
  committed target token IDs. Diagnostic only; absent by default; NOT a
  request-side splice input.
- **`ChatMessage.token_ids`**: `let token_ids: [Int]?`, custom encode omits
  nil; request schema unchanged (decode accepts/ignores it).
- **Session store** (`SessionStore` actor, new): process-local, in-memory,
  LRU-capped (`--max-sessions`, default 128) + idle-TTL (`--session-ttl`,
  default 1800 s). Owns the message history, a diagnostic cached prefix
  (`seed + completion` token IDs), a per-session in-flight flag, and a token
  count. No disk persistence, no cross-process sharing.
- **Session routes** (`POST/GET/DELETE /v1/sessions[/:id]`,
  `POST /v1/sessions/:id/completions`): create/inspect/delete + one-turn
  completion. Completion merges stored history + the new turn, re-renders, and
  on success commits the assistant turn back (via `onFinished`); on failure
  nothing is committed. `session_id` echoed on non-streaming responses.
  `DELETE` best-effort releases the radix prefix (leaf removal only).
- **`CompletionCommit`**: carries the assistant-turn fields + committed token
  IDs; `assistantMessage` builds the stored history message (NO token IDs —
  history is for re-rendering only).
- **Router extraction**: the stateless completion body extracted into
  `@Sendable performChatCompletion(app:request:serverConfig:runtimeState:
  scheduler:generator:metricsCollector:logger:sessionID:onFinished:)` shared by
  both the stateless and session endpoints (also a compiler-fragility win for
  the large closure).
- **Radix** (`RadixKVCacheManager.remove`): best-effort leaf removal used by
  session delete.
- **Config** (`ServerConfig`): `maxSessions`, `sessionTTLSeconds` (+
  `QWEN_MAX_SESSIONS` / `QWEN_SESSION_TTL`).
- **Docs**: `docs/SESSION-API.md` (client contract + curl/Python examples),
  `docs/SESSION-BOUNDARIES.md` (design note), `docs/TOOL-PROTOCOL.md` cross-ref.

### Tests (pure Swift, no model weights; 13 new + 1 gated boundary test)

- `SessionStoreTests` (6): create/exists/inspect/delete, completion lifecycle
  (in-flight serialization + history continuity), delete releases cached
  prefix, aborted completion commits nothing, LRU eviction, TTL expiration.
- `TokenIDEchoTests` (7): `token_ids` omitted-when-nil + round-trip, commit
  assistant message carries no token IDs, empty tool calls collapse to nil,
  `session_id` round-trip, `include_token_ids` decode.
- `TokenBoundaryTests` (gated `QWEN_RUN_WEIGHTS=1`): full non-thinking prefix,
  partial thinking divergence.
- **Full `HTTPServerTests`: 182 green** (168 → 182).

### Do-not-claim

- Session = conversation ownership + history continuity, **not** a cache hit.
- No token splicing, no template duplication, no request-side `token_ids`.
- No disk persistence, no auth, no cross-process sharing, no continuous
  batching.

### Reproduction

```
swift build --target HTTPServer
swift test --filter HTTPServerTests   # 182 green
```

## 2026-09-15: Draft-depth calibration (wall-clock tokens/s per depth) — COMPLETE

Operator-facing calibration mode that measures wall-clock decode throughput at
several speculative draft depths and selects the fastest per model, storing the
winner. **Server repo + one engine init hook.** No kernel, head topology,
quantization, weight, or sampling-semantics changes. Off by default.

### What was done

- **Engine** (`Qwen38MTPBlockSession`, `mlx-swift-lm`): new optional
  `draftDepth: Int?` init param (pinned per-round depth). Pinned depth takes
  highest priority in `draftPolicy` (pinned → `QWEN_MTP_DRAFT_K` → default 2),
  above the offer cap. This lets the server sweep depths 0..3 in one process
  (the env var is a process-global static; the per-session pin is not).
  `nil` path is unchanged (engine diagnostic 3/3 green).
- **Server — model-free core** (`SpecDraftCalibration`, new): `DepthBenchmark`,
  `ModelCalibration`, `SpecDraftCalibrationFile` (Codable); `parseDepths`,
  `selectOptimalDepth` (max tok/s, ties → lower depth), `load`/`save` (pretty
  JSON, missing/corrupt → nil), `report`, `iso8601Now`.
- **Server — `ServerConfig`**: `--spec-draft-calibrate` (off by default),
  `--spec-draft-calibrate-depths` (default `0,1,2,3`),
  `--spec-draft-calibrate-tokens` (default 100),
  `--spec-draft-calibration-file` (default `./spec-draft-calibration.json`),
  `--spec-draft-n-max` now sets `specDraftNMaxExplicit`; `QWEN_MTP_DRAFT_K` read
  into `specDraftK`. `resolvedForcedDraftDepth(storedCalibratedDepth:)` encodes
  the precedence: explicit `--spec-draft-n-max` > `QWEN_MTP_DRAFT_K` > stored >
  default 2.
- **Server — `MLXGenerator`**: `forcedDraftK: Int?` init param (passed to both
  production sessions); `calibrateDraftDepths(depths:tokens:)` sweeps pinned
  sessions (decode-only timing, greedy, acceptance from `accepted/rejected`
  counters); `applyCalibratedDepth(_:)` sets the pin for the running process.
- **Server — `Qwen38Server`**: loads the store, resolves the forced k, passes
  it to the generator; on `--spec-draft-calibrate`, runs the sweep after warmup,
  logs the report, applies the winner, and saves the store (merge, keyed by
  canonical model id). Calibration failure is non-fatal (serves at resolved k).
- **Docs**: `docs/DEPTH-CALIBRATION.md` (usage, config format, resolution
  precedence, caveats, future runtime-adaptation roadmap), `docs/README.md`
  (example commands + runtime-knobs note), `progress.md`.

### Tests (pure Swift, no model weights; 17 new)

- `DraftCalibrationTests` (17): `parseDepths` (basic/trim/dedupe/out-of-range/
  empty), `selectOptimalDepth` (max/empty/tie→lower), `report` format,
  store round-trip / missing-file→nil / corrupt-file→nil,
  `resolvedForcedDraftDepth` precedence (default/env/stored/explicit-n-max),
  `iso8601Now` format.
- **Full `HTTPServerTests`: 199 green** (182 → 199). Engine `Qwen38MTPDiagnosticTests` 3/3 green.

### Do-not-claim

- Wall-clock throughput measurement, **not** a 1024-token benchmark cell; do
  not cite calibration tok/s as a headline number.
- Depth selection never changes correctness (greedy output is bit-identical
  across depths; it only changes tokens verified per round).
- The stored depth is a **hint**: an explicit `--spec-draft-n-max` or
  `QWEN_MTP_DRAFT_K` overrides it. `--spec-draft-n-max 0` disables MTP.
- Online runtime adaptation (rolling acceptance → depth) is **documented as
  future, not implemented**.

### Reproduction

```
swift build --target HTTPServer
swift test --filter DraftCalibrationTests   # 17 green
swift test --filter HTTPServerTests        # 199 green
cd ../mlx-swift-lm && swift test --filter Qwen38MTPDiagnosticTests  # 3/3
```

## Phase F — Fused GDN prework kernel (DONE)

Ported `qwen35PackedGDNPreworkKernel` (from the oMLX/mlx-serve reference,
transcribed **verbatim** for the bit-exact-critical logic) into
`Qwen35Kernels.swift`. One Metal launch fuses the GDN prologue — conv1d +
SiLU + Q/K/V split + Q/K rmsNorm-and-scale + g/beta producer — for MTP
**verify widths S ∈ 3…9** (B=1, `nKeep=3`).

### Gate (OFF by default)

- Env flag `MLX_QWEN_FUSED_GDN=1` (default **off**; per-instance
  `fusedGDNPreworkEnabled` override for tests).
- Hardware: `MLXHardwareInfo.isCompiledDecodeSupported`.
- Geometry (hardcoded, matches the 27B Qwen3.5 backbone): `B=1`,
  `S ∈ 3…9`, `nKeep=3`, `numKHeads=16`, `numVHeads=48`, `headKDim=128`,
  `headVDim=128`, `qkv.dim(2)=10240`, `mask == nil` (checked in `forward`
  before the call — the kernel does not apply a prefill mask).
- Dtypes (all `.bfloat16`): `qkv`, `convState`, `a`, `b`, `conv1d.weight`,
  `aLog`, `dtBias`.
- Any gate miss → eager chain (unchanged, zero overhead when off).

### Metal-version adaptation (minimal, bit-exact)

The verbatim kernel does not compile under this MLX's Metal, which promotes
`bfloat16_t op bfloat16_t` to `float` (the reference's Metal keeps it in
`bfloat16_t`). Three `const InT x = <InT op InT>` lines needed explicit
`static_cast<InT>(static_cast<float>(a) op static_cast<float>(b))`. Bit-exact
because Metal emulates bf16 arithmetic in float32, so the value is unchanged.
The 0xC0DB→0x3A8B `qwen35_prework_beta` fixup, the `metal::precise::exp`
calls, the threadgroup barrier, and the stride-based memory access are
**untouched**.

### Bit-exactness (proven, not assumed)

- **Unit** (`Qwen35FusedGDNProjectionTests`): full GDN layer output is
  bit-identical (fused vs eager) for every S ∈ 3…9 on the production geometry;
  the dispatch counter proves engagement (engagedTotal ≥ 1, per-width). Width
  gate test: S ∈ {1,2,10,16} never engages and still matches eager.
- **Real model** (14 GB 27B, `Qwen38MTPDiagnosticTests`): with
  `MLX_QWEN_FUSED_GDN=1`, overall committed stream hash
  `86cd9e8868988aef611512262936a11d717bf8f6882efa8997910b992784b1c5` (identical
  to the OFF baseline) and `[FusedGDNDispatch] engagedTotal=9216 widths=[5: 9216]`
  — the fused kernel **engaged** on every verify round, not a silent gate miss.
  Wide-verify (depth 5) serial/wide hashes also match the OFF baseline.

### Performance (honest finding)

Controlled in-session A/B of the diagnostic test (3 reps each, median):
- S=5 (depth 4): OFF 33.64 s vs ON 36.67 s.
- S=6 (depth 5, `testWideVerifyStaysInSerialFamily`): OFF 30.99 s vs ON 31.55 s.

The fused kernel is **bit-exact but not a clear wall-clock win** in these
workloads: the verify path (S=5–6) is a small fraction of total wall-clock
(prefill + decode dominate), and fusing small-tensor ops into one launch does
not beat the eager chain's already-cheap small launches. **Verdict: retain as a
correctness-preserving, default-OFF opt-in (rollback-safe), not a headline
win.** No absolute tok/s is cited (cross-session absolutes are not comparable).

### Tests

- `Qwen35FusedGDNProjectionTests`: `testFusedGDNPreworkIsBitIdenticalAcrossVerifyWidths`
  (S ∈ 3…9 bit-exact + engagement), `testFusedGDNPreworkWidthGate`
  (S ∈ {1,2,10,16} gated off). 17/17 green in the suite.

### Reproduction

```
cd ../mlx-swift-lm
swift build --target MLXLLM
swift test --filter Qwen35FusedGDNProjectionTests   # 17 green
MLX_QWEN_FUSED_GDN=1 swift test --filter Qwen38MTPDiagnosticTests  # hashes match baseline, engagedTotal=9216
```

## 2026-09-15: Online adaptive draft depth (Phase G) — COMPLETE

A serve-time policy that moves the per-request speculative draft depth within
`[1, --spec-draft-n-max]` from observed acceptance rate and wall-clock
throughput. **Server-only** (no engine change): the policy mutates the
existing `MLXGenerator.forcedDraftK`, which is read at session creation, so
`MLXLLM`/`MLXFastModel`/kernels are untouched (engine repo `main` clean).

### Design

- `Generation/AdaptiveDraftDepth.swift` — pure `AdaptiveDraftDepthPolicy`
  (model-free `Sendable` struct): bounded rolling window (default 50 completed
  requests), hysteresis counters (default 10), `[1, maxDepth]` clamp.
  - Increase: acceptance `>= thresholdHigh` (0.7) for `hysteresis` samples, below
    max, and the throughput safety signal not firing.
  - Decrease: acceptance `<= thresholdLow` (0.5) for `hysteresis` samples, above 1.
  - Throughput safety signal (internal, on by default): tps > 10% below the
    rolling mean for 3 samples decreases depth, and vetoes an increase.
  - Dead band between the thresholds moves no counter and resets hysteresis.
  - `setDepth` (calibration sync) clamps + resets all hysteresis.
- `MLXGenerator`: `adaptivePolicy` (nil = off), seeded at the resolved forced
  depth; `recordAdaptiveSample(acceptanceRate:tokensPerSecond:)` (actor method,
  pure mutation) is fed per completed request in `generateStream`'s success path
  (gated on `proposedDraftTokens > 0`); `applyCalibratedDepth` also syncs the
  policy; `adaptiveDraftDepthSnapshot()` for `/metrics`.
- `ServerConfig`: `--spec-draft-adaptive` (off) + `-window`/`-threshold-high`/
  `-threshold-low`/`-hysteresis`; `adaptiveDraftDepthConfig(maxDraftDepth:)`
  returns nil when off or MTP is disabled.
- `RequestMetrics.MetricsSummary`: `adaptiveDraftDepth`,
  `adaptiveRollingAcceptanceRate`, `adaptiveDraftDepthAdjustments` (nil when off)
  + CodingKeys; `OpenAIRouter` merges them into `GET /metrics`.
- Docs: `docs/ADAPTIVE-DRAFT-DEPTH.md`, README example, and the calibration doc's
  "future" section now points to the implementation.

### Tests (25 new, `AdaptiveDraftDepthTests`, pure Swift, no weights)

Increase/decrease on sustained high/low acceptance; dead band; hysteresis
(single samples don't move; dead-band sample resets the counter); bounds; no
oscillation under alternating signals; bounded adjustments under a sustained
shift; the throughput safety signal (A/B: a sustained drop reduces depth where
acceptance alone increases it); `setDepth` sync; initial-depth clamp; rolling
stats; window bound; config validation; the `ServerConfig` → policy mapping
(off by default, nil when MTP disabled, builder fields, defaults).

`swift test --filter AdaptiveDraftDepthTests` → 25/25 green.
`swift test --filter HTTPServerTests` → 224/224 green (was 199; +25).

### Caveats

- Adaptation granularity is per-request (a sample = one completed request that
  proposed ≥1 draft), not per-round; a change takes effect for the next request.
- Throughput is wall-clock for the safety signal, not a benchmark cell.
- Off by default; with the flag off the policy is never created and the
  `MetricsSummary` fields are nil (backward compatible).

## 2026-09-15: Long-context benchmarking at 8K/32K/96K (Phase H) — COMPLETE (measurement only)

Benchmarking/profiling task, **no code changes** (server, engine, kernels, model,
quantization all untouched). Release binary built from `main` (`18ed7d7`).
Full write-up: `docs/LONG-CONTEXT-BENCHMARKS.md`. Raw lines:
`benchmarks/results/longctx-2026-09-15/ab-matrix.jsonl`.

### Fixtures
Deterministic real Swift source (server+engine, 479 files, 1.76M-token corpus),
truncated at token boundaries: `benchmarks/prompts/longctx-{8k,16k,32k,96k}.txt`,
built by `benchmarks/make_longctx_fixtures.py` (sha256 in
`benchmarks/results/longctx-2026-09-15/NOTES.txt`).

### Headline findings
- **32K and 96K are infeasible.** The prefill attention buffer is a dense
  `[seq × seq]` allocation (quadratic). At 32K it requests 51,577,363,200 bytes
  (51.6 GB) > the 30,150,672,384-byte (30.2 GB) Metal max buffer →
  `[metal::malloc] ... greater than the maximum allowed buffer size` → SIGTRAP
  crash (reproducible). 96K would be ~464 GB. **Max feasible prompt ≈ 24K tokens.**
- **Prefill dominates** request time: ~27 s at 8K of a ~30 s request (~90 %).
  Prefill throughput ~275 tok/s, ~linear in seq (8K 27 s, 16K 59 s).
- **Fused GDN does NOT pay off at long context.** Bit-exact (identical token
  stream on/off) and not faster at 8K (full_s 29.3 s off vs 32.7 s on). The
  launch count is per verify round (draft-depth dependent), not per context
  token, so it does not accumulate with context; and prefill (not decode) is the
  cost.
- **k = 2 remains optimal at 8K** (full_s k1 36.1 / k2 29.3 / k3 32.0 s), matching
  short context.
- **Prefix/session caching ≈ 7 % TTFT win only** (8K cold 27.3 s → warm ~25 s):
  the gated-delta recurrent layers are not resumable from a token-prefix, so the
  prefill is effectively recomputed each request.
- **All configs bit-exact** (identical content sha256) across fused on/off,
  k=1/2/3, and repeated requests.
- **Memory:** peak RSS ~15.25 GB at 8K (~15 GB weights + ~0.5 GB KV). Steady-state
  fits 32K under the 44 GB limit; the crash is the *transient* quadratic prefill
  buffer, which admission control does not model.
- **Decode is GPU-eval-bound** (~89 ms/round tEvalMs of ~93 ms stepMs at 8K);
  host/graph overhead is small. First verify round after prefill is a one-time ~4 s.

### Do-not-claim
- Do NOT cite cross-session absolute tok/s; numbers here are one environment.
- Do NOT claim the fused GDN is a wall-clock win (it is not, at any context).
- Do NOT claim 32K/96K work; they crash / are infeasible on this hardware.
- N per cell is 3–4 (not 6–10) because each 8K request costs ~30 s and 32K/96K
  cannot complete; means have small spread.

## 2026-09-15: Chunked causal prefill (Phase I) — COMPLETE

Eliminates the quadratic `[seq × seq]` dense-attention scores buffer that traps
the process (SIGTRAP) at ~24K+ context on this hardware. **Default-OFF**
(`MLX_CHUNKED_PREFILL=1` to enable); the off build is bit-identical to before.

- **Engine** (`../mlx-swift-lm`):
  - `MLXChunkedPrefill` config enum in KVCache.swift (`MLX_CHUNKED_PREFILL`, tile 512,
    min-seq 4096).
  - `chunkedCausalPrefill` in AttentionUtils.swift: sequential query tiling over the
    incrementally-grown cache, each tile's fused SDPA with a bottom-right-aligned
    `.causal` mask (query `s+i` sees keys `[0, s+i]` — exact key set of the dense
    prefill). Gated in `attentionWithCacheUpdate`'s KVCacheSimple branch (the path
    the model's full-attention layers actually use in prefill) for a fresh causal
    prefill (offset 0, L > 4096).
  - `chunkedCausalQuantizedAttention` + `optSlice` for the quantized-cache variant.
  - Tests: `testChunkedCausalPrefillMatchesDense` (KVCacheSimple, L 33/64/128/129),
    `testChunkedCausalMatchesDense` (quantized, L 64/600/1024/1025, nRepeats 1/2).
- **Server** (this repo):
  - `MemoryAdmissionPolicy` now models the transient prefill buffer: dense
    `nQHeads×L²×2` (quadratic) vs chunked `nQHeads×tile×L×2` (linear); a request
    whose transient buffer exceeds the Metal single-buffer cap ×0.9 is rejected
    with HTTP 507 / code `prefill_buffer_exceeded` (`TransientBufferFailure`,
    `openAITransientBufferErrorResponse`, caught in OpenAIRouter).
  - Construction site passes `chunkedPrefillEnabled` from the env flag.
  - 4 new MemoryAdmission tests.
- **Validation**:
  - 8K greedy stream hash identical dense vs chunked (`763eccc3…`), chunked path
    genuinely engaged (L > 4096).
  - 32K completes with chunked (175.9 s) where the dense path traps.
  - Dense 32K cleanly rejected pre-prefill: 507 (`51.6 GB > 27.1 GB`), 0.17 s.
  - Engine KVCache 118/118, MTP diagnostic 3/3, server 224/224 all green.
- Docs: `docs/CHUNKED-PREFILL.md`.
- Do NOT claim 32K/96K work in the default (dense) build — they are rejected / trap.
  They work only with `MLX_CHUNKED_PREFILL=1`.

## 2026-09-15: pc=0 single-pass prefill trap fix — COMPLETE

### Defect
`--prefill-chunk-size 0` (documented as "single-pass prefill") produced an empty
chunk in both prefill loops in `Qwen38MTPBlockSession`: with `chunkSize = 0`,
the inline loop computed `start = 0`, `end = min(0 + 0, count) = 0`, so the chunk
was `prefillTokens[0..<0]` (empty) and `MLXArray([]).reshaped([1, 0])` fatal-`[reshape]`'d
on an empty array. The 32K single-pass request trapped the process.

### Fix (engine-only, minimal)
- Extracted the chunk partition into a pure, testable
  `Qwen38MTPBlockSession.prefillChunkRanges(count:chunkSize:) -> [Range<Int>]`
  that guards `chunkSize == 0` (→ one full-range chunk) and `count == 0` (→ no
  chunks). Both prefill loops now iterate over `prefillChunkRanges(...)`.
- The `chunkSize > 0` branch of the new function is mathematically identical to
  the old inline loop, so the **default pc=512 path is byte-identical** (no
  behavior change, no perf regression).

### Validation
- 4 new unit tests (`Qwen38MTPPrefillChunkTests`): pc=0 single-pass at 8K/16K/32K,
  pc=512/pc=8192 partition invariants, empty, below-chunk. All pass; the 3 MTP
  diagnostic tests still pass (7/7 with the new suite).
- Real-model (Qwen3.8-27B, `MLX_CHUNKED_PREFILL=1`):

  | fixture | pc=0 wall | pc=512 wall | content hash pc=0 vs pc=512 |
  | --- | --- | --- | --- |
  | 8K (8532 tok) | 31 s | 31 s | `660dd120…` == `660dd120…` (bit-exact) |
  | 16K (17703 tok) | 74 s | 61 s | `2e583ad2…` == `2e583ad2…` (bit-exact) |
  | 32K (32780 tok) | 205 s | 117 s | `95b445a6…` ≠ `97bc0d74…` (1st-token knife-edge) |

- **32K is not bit-exact, as expected.** At L=32780 the engine chunked-SDPA gate
  (`L > 4096`) engages for pc=0 (one L=32780 pass) but not for pc=512 (per-512
  passes, each L=512 < 4096). The two prefill splits accumulate the full-attention
  FP in a different order, flipping a knife-edge first token (Phase 1 Bug A). This
  is the same sensitivity as any prefill-chunk-size change; the pc=512 default is
  the reference and is unchanged.
- pc=0 is **slower** than pc=512 (205 s vs 117 s at 32K) — single-pass is not a
  perf win. This is a correctness/robustness fix (a documented option no longer
  traps), not an optimization.

## 2026-09-15: Long-context prefill optimization & exactness guardrails (LCP) — IN PROGRESS

Task: measurement-driven prefill optimization with a strict user-gated bit-exact
fallback (`ENABLE_BIT_EXACT=1`). Spec is CUDA/PyTorch-flavored; adapted to this
MLX/Swift stack: eval-synchronized GPU timing (the established `MLX_TRACE_PREFILL`
method), content-SHA-256 token-stream hashes for bit-exactness, greedy
(temperature 0) + pinned fixtures (`req8k/16k/32k/64k.json` in
`benchmarks/results/prefill-verify-2026-09-15/`) as the determinism protocol
(SEED=42 analog; no RNG in the prefill path). EXP_ID `LCP`, device `m5pro`.

Plan (phases sequential, engine committed before server per git policy):
- **P1** `P1_PROFILE_LCP_m5pro_20260915.md`: re-add the eval-synced
  `MLX_TRACE_PREFILL` instrumentation (env-gated, default off, zero-overhead
  when off; previously removed after the 32K/64K runs), run 8K/16K/32K/64K
  with `MLX_CHUNKED_PREFILL=1` pc=512 (baseline config), capture wall +
  per-phase (FFN / GDN / full-attn block + SDPA / QKV / O / RoPE / norms) +
  peak RSS + content hash. Table `Phase | 8K | 16K | 32K | 64K | % of Total
  (64K)` + bottleneck analysis.
- **P2** `P2_KERNEL_LCP_*_20260915.md`: two toggle-gated, non-destructive
  kernel extensions (both bit-exact, unoptimized path retained):
  (a) `residual_norm_3d` (`MLX_QWEN_FUSED_RESIDUAL_3D`, default OFF): extend
  the fused residual+RMSNorm Metal kernel (already bit-exact for 2-D decode)
  to 3-D prefill shapes [B, S, 5120] — the kernel is row-parallel and
  shape-agnostic; the `ndim == 2` guard is the only blocker.
  (b) `gdn_prefill_prework` (`MLX_QWEN_FUSED_GDN_PREFILL`, default OFF):
  extend the fused GDN prework kernel (bit-exact verified at verify widths
  S 3..9) to prefill widths (S up to 4096) — one Metal launch replaces
  conv1d + SiLU + split + Q/K RMSNorm + scale + g/beta.
  Validation: bitwise unit tests + 8K/16K/32K content-hash exactness vs
  baseline + 32K/64K re-profile.
- **P3** `P3_ATTN_LCP_bit_exact_gate_20260915.md`: user-gated attention
  path. `ENABLE_BIT_EXACT_ATTENTION` (default 1): 1 = reference dense
  attention, 0 = chunked/fused attention path (implies the
  `MLX_CHUNKED_PREFILL` routing); unset = legacy behavior unchanged.
  Global `ENABLE_BIT_EXACT=1`: strict unoptimized fallback — forces dense
  attention and disables every fusion (SwiGLU/QKV/4-GDN/residual-3D/GDN
  prework). Single resolver in the engine (MLXLMCommon); server admission
  uses the same resolver. Validation: =1 matches P1 outputs identically at
  8K/16K (and 32K pc=512); 32K pc=0: =1 rejected (507, dense buffer),
  =0 completes (known pc=0 knife-edge hash, documented drift); quantify
  uplift (32K/64K enabling vs dense rejection).
- **Governance**: `docs/PREFILL-PROFILE-INDEX.md` (central table + links),
  `docs/HANDOFF.md` (peak speedups, recommended default flags, active
  bottlenecks), this log.

Known framing facts (verified in code):
- Session prefill chunk size default is 512 (`--prefill-chunk-size`,
  `QWEN_PREFILL_CHUNK_SIZE`); the engine query-tile SDPA gate engages only
  for single attention calls with L > 4096, i.e. pc=0/large-pc single-pass
  prefills. At pc=512 the per-chunk scores buffer is linear (0.81 GB at 32K)
  regardless of the flag; the quadratic [L,L] buffer exists only for
  single-pass. Admission models the dense path as quadratic (conservative
  worst case) — unchanged by this task (contract surface).
- FFN (43-49 %) and GDN (21-25 %) GEMM structure is already minimal at the
  Swift level (fused wide gate+up / 4-proj GEMMs, compiled activations); the
  remaining body is the prebuilt MLX 4-bit GEMM (out of scope, discouraged).
  SDPA (18.5 % at 32K → 29.3 % at 64K, O(L²), dense unfused path) is the
  growing center; a flash-style kernel is not bit-exact (Phase 1 Bug A) and
  stays out of the default path.

## LCP implementation status (2026-09-16, feature/prompt-1)

Engine (mlx-swift-lm), all changes on `feature/prompt-1`, flags default-off:
- `MLXBitExact` + `qwen35FusedResidual3DEnabled` +
  `qwen35FusedGDNPreworkPrefillEnabled` (new `BitExactConfig.swift`);
  `MLXChunkedPrefill.enabled` now resolves
  ENABLE_BIT_EXACT > ENABLE_BIT_EXACT_ATTENTION > MLX_CHUNKED_PREFILL.
- Fusion switches (SwiGLU/QKV/4-GDN) respect `ENABLE_BIT_EXACT=1`.
- `Qwen35PrefillTrace` (new, MLX_TRACE_PREFILL=1, default off): eval-synced
  per-section prefill timing; PF2 line emitted by the session after each
  prefill.
- `applyResidualNorm` 3-D prefill branch (MLX_QWEN_FUSED_RESIDUAL_3D, off).
- `fusedGDNPrework` width gate extended to prefill widths S>9 up to 4096
  (MLX_QWEN_FUSED_GDN_PREFILL, off).
- Unit tests `Qwen35PrefillFusionTests` (residual-3D S=1,2,512,1000; GDN
  prework S=16,512) — both PASS bit-exact.
Server:
- `MLXGenerator` admission uses `MLXChunkedPrefill.enabled` (shared resolver).
- Full suite PASS (111 XCTest + 224 Swift Testing) after all edits.
- Release binary rebuilt; `benchmarks/run_lcp_p1.sh` (4-cell P1 matrix,
  pc=512, trace on, hash-gated) running.

## LCP complete (2026-09-16)

- P1 matrix (8K/16K/32K/64K pc=512, trace on): all hashes PASS. FFN 56.3→36.5 %,
  full-attn 12.8→42.1 % (SDPA 5.7→35.6 %), GDN 27.3→19.1 %, norms+resid ~3 %.
- P2 matrix (8 cells, both toggles, per-flag at 32K/64K): all hashes PASS
  (bit-exact end-to-end). Wall-time uplift not resolvable (±1.6× session state
  variance; flag-off 32K re-run 211 s vs P1 baseline 311 s). Toggles stay
  default-OFF.
- P3 matrix (8 cells): dense reference bit-exact at 8K/16K, 507 at 32K/64K;
  chunked under gate bit-exact at 32K/64K; ENABLE_BIT_EXACT=1 bit-exact at
  8K/16K vs optimized default.
- Reports: benchmarks/results/prefill-opt-20260915/P{1,2,3}_*.md;
  docs/PREFILL-PROFILE-INDEX.md created; docs/HANDOFF.md updated.
- Runners: benchmarks/run_lcp_p1.sh / run_lcp_p2.sh / run_lcp_p3.sh (hash-gated).
- Next: commit engine (feature/prompt-1) → commit server → merge both to main.
  (Resolved 2026-09-16: engine merged; server commits `ac9e019` + `d1ec725`
  landed on `main` — see `git log`.)

## 2026-09-16: Task 7 — `--kv-ssd-*` CLI "crash" diagnosis (code-only, zero server launches) — COMPLETE

### Diagnosis

- **`--kv-ssd-cache-dir`, `--kv-ssd-cache-gb`, `--kv-ssd-ttl-seconds` do not exist in
  this codebase** — zero occurrences in server sources, engine repo, docs, or git
  history. They were never implemented flags.
- The "crash" mechanism was real but not a strip-list gap: `fromCommandLine()`
  silently ignored unknown flags (`default: break`), while `vaporArguments()` only
  strips *known* flags and keeps everything else — so an unknown `--kv-ssd-cache-dir
  /tmp/...` (flag **and** value) leaked into `Environment.detect(arguments:)` and
  Vapor's dispatcher rejected it, crashing `app.execute()`. The reported
  "intermittent" crashes were this leak (deterministic for the flag) plus the
  separately-discovered port-8000/OOM races from a stray background repro loop.
- **Full audit:** every implemented `ServerConfig` flag (`--host` … `--tools-disabled`,
  `--help`; incl. `--prefill-chunk-size`, `--spec-draft-*`, `--kv-scheme`, KV
  quant flags) is covered by `valueTakingFlags` ∪ `valuelessFlags`. No implemented
  flag ever leaked to Vapor. The only leak path was unknown flags.

### Fix (server repo only; engine untouched)

- `ServerConfig.unknownFlagError(in:)` — pure function; rejects any `--flag` not in
  the known set (skips values of value-taking flags; ignores positionals).
- `fromCommandLine()` now fails loudly at startup (stderr + `exit(2)`) on any
  unknown flag, before Vapor dispatch — actionable message instead of an opaque
  Vapor crash. Satisfies "never silently accept."
- `vaporArguments()` refactored to `vaporArguments(from:)` (pure) + thin
  `CommandLine.arguments` wrapper for testability.
- New `Tests/HTTPServerTests/ServerConfigArgumentTests.swift` (9 tests, pure Swift):
  asserts `vaporArguments(from:)` strips ALL known flags and keeps Vapor-native
  args/positionals; asserts `unknownFlagError(in:)` accepts the full known set and
  rejects each `--kv-ssd-*` flag, typo flags, and consecutive unknown flags.

### Verification

- `swift build --target HTTPServer` — 0 errors, 0 warnings.
- `swift test --filter HTTPServerTests` — 224 tests green (suite grew 215 → 224).
- No server launches; no engine changes.

## 2026-09-16: Documentation reconciliation audit (progress.md ↔ repository state)

Documentation-only pass. Every fact below verified against source / git / a
test run in this session (`cd /Users/cwong/ai/qwen38-mlx-server/qwen38-mtp-server`,
a physical path — the wrapper working directory is a symlink hub).

### Ground truth at audit time

- **Server repo**: `main` @ `76054e3`, clean tree (`git status --short` empty).
  Task 7 commit `76054e3` present (Task 7 section above).
- **Engine fork** (`/Users/cwong/ai/mlx-swift-lm`): `main` @ `d189c61`
  (LCP prefill trace / bit-exact gates / fused prefill kernel extensions),
  clean tree.
- **Test count** (`swift test --filter HTTPServerTests`, default invocation —
  no `QWEN_RUN_WEIGHT_TESTS=1`): **224 Swift Testing tests in 5 suites, 0
  failures** + **120 XCTest tests, 0 failures** — all green.
- **`kv-ssd` / `QWEN_KV_SSD`**: zero occurrences in `Sources/`, `Tests/`,
  `docs/`, `benchmarks/`, help text, and full git history of this repo; zero in
  the engine fork. The `--kv-ssd-*` flags never existed (re-confirmed the Task 7
  finding). No "workaround" language exists in `docs/HANDOFF.md` (grep clean).
- **Depth-tuning surface**: the shipped surface is `--spec-draft-calibrate`
  (+ `--spec-draft-calibrate-depths` / `-tokens` / `--spec-draft-calibration-file`,
  `spec-draft-calibration.json`) and `--spec-draft-adaptive` (online policy),
  exactly as documented in `docs/DEPTH-CALIBRATION.md` and
  `docs/ADAPTIVE-DRAFT-DEPTH.md`, with resolution precedence
  `--spec-draft-n-max` > `QWEN_MTP_DRAFT_K` > stored `optimal_depth` >
  engine default k = 2 (`ServerConfig.resolvedForcedDraftDepth`).
  `AdaptiveDraftDepthTests` contains 25 `@Test` functions, as
  recorded in the Phase G section.
- **Session type**: the server instantiates `Qwen38MTPBlockSession` (referenced
  in `MLXGenerator.swift`, `ServerConfig.swift`, `SpecDraftCalibration.swift`).
  `Qwen36MTPBlockSession` survives only in two stale *comments*
  (`Sources/HTTPServer/Models/OpenAIModels.swift:18`,
  `Sources/HTTPServer/Generation/RadixKVCacheManager.swift:14`).
  **Naming note:** the engine/session line was renamed `Qwen36*` → `Qwen38*`;
  historical sections above keep their original `Qwen36*` spellings by
  convention. **Repo identity:** the server repo lives at
  `/Users/cwong/ai/qwen38-mlx-server/qwen38-mtp-server` (physical
  `/Users/cwong/ai/qwen38-mtp-server`).
- **`ttl_seconds`**: still **rejected** by `OpenAIValidation.swift`
  (400/`not_supported`: "'ttl_seconds' is not applied by the current runtime
  and is not supported."). The radix cache does have per-entry TTL forwarding
  via `samplingParams.ttlSeconds` (`MLXGenerator.swift`), but the public API
  surface deliberately keeps the rejection.
- **Parameter status** (for the rollup): speculative (MTP) decoding is
  implemented and on by default (pinned k = 2, offer cap 3); repeat/presence/
  frequency penalties are implemented with in-range validation and the
  serial target-only depth fallback for non-default penalties (`b4e9956`,
  Phase A F1).

### Resolved: Tasks 1, 3, 4, 5, 6 are on `main`

The "Task 1–6 missing" question is resolved: all five tasks are implemented
and merged to `main`, documented in the Task 1 / 3 / 4 / 5 / 6 sections
below. The reconciliation brief's claims now match this checkout:
`--kv-ssd-*` is implemented (Task 3), the in-RAM reusable-path repair is on
`main` (Task 4, 5.73 s → 0.37 s), the SSD gate was re-specified and passed
(Task 6, G1 0.117× / G2 0.609 s), and the Task 1 artifacts
(`WeightTestLock.swift`, `QWEN_MLX_SEED` hook) are present. The audit-time
test count (224) has since grown to 237 with those suites (see "Current
status and roadmap").

## Draft-Depth Calibration & Online Adaptation (Phase G)

[x] `--spec-draft-calibrate` (startup calibration): measures wall-clock tok/s at depths 0..specDraftNMax, picks the winner (must beat serial by ≥5%), persists to `spec-draft-calibration.json` keyed by model ID. Resolution precedence: `--spec-draft-n-max` > `QWEN_MTP_DRAFT_K` > stored `optimal_depth` > default k=2.
[x] `--spec-draft-adaptive` (online adaptation): `AdaptiveDraftDepthPolicy` adjusts per-request depth within `[1, --spec-draft-n-max]` based on rolling acceptance rate + throughput safety signal. Hysteresis counters prevent oscillation. 25 pure-Swift tests in `AdaptiveDraftDepthTests`.
[x] Docs: `docs/DEPTH-CALIBRATION.md`, `docs/ADAPTIVE-DRAFT-DEPTH.md`.

## Task 1 — Compact-space rejection walk — NEGATIVE RESULT (artifacts kept, 2026-09-16)

**Goal:** eliminate the per-round full-vocab `[248320]` array materializations
from the non-greedy MTP verification walk; verify distribution preservation
(chi-square/KS, seeded) plus weight-gated serial-vs-MTP distributional
equivalence; then an interleaved A/B benchmark (≥5/side, 60 s cooldowns, fixed
seeded request, temp 0.7 / top_p 0.95 / max_tokens 300) gated on median
decode tok/s improvement ≥ 3%.

**Result: gate not met — the production change is NOT merged.**
- A/B (5 rounds/side, 60 s cooldowns, seeded): median wall A 11.875 s vs
  B 12.119 s → **B is 2.05% SLOWER** (gate: B must be ≥ 3% faster); per-round
  `round_us` identical within noise (115.34 vs 115.35 ms); peak RSS 14.9–15.0
  GB both sides; per-round `active` memory flat.
- Root cause: the 248k-vocab residual ops are one ~100–200 µs GPU pass; the
  compact walk's added 98k gather + second `.item()` host sync cost about the
  same. The removed materialization was ~1 MB transient — no memory win.
- RFC: `docs/compact-rejection-rfc.md` (exact residual split, preservation
  lemma, failure-mode table, test + benchmark plan).

**Kept artifacts (independent value, on `main`):**
- `Tests/HTTPServerTests/CompactRejectionTests.swift` — pure distributional
  `@Test`s (new walk matches old walk and the analytic residual over 5
  synthetic cases; full-vocab degenerate C == V; point-mass determinism;
  chi-square + total-variation) + 1 weight-gated serial-vs-MTP chi-square.
- `Tests/HTTPServerTests/WeightTestLock.swift` — process-level flock guard so
  two weight-gated tests never each load the ~14 GB model concurrently (a real
  deadlock: a load thread died mid-load holding MLX's global eval lock).
  **Run weight-gated tests in separate `swift test` invocations.**
- `benchmarks/run_compact_rejection_ab.sh` + `benchmarks/compact-rejection-prompt.json`.
- `QWEN_MLX_SEED` env hook in `MLXGenerator.init` (benchmark determinism;
  no-op when unset).

## Task 3 — Radix-tree SSD persistence — first gate NEGATIVE (2026-09-16)

**Goal:** cold disk tier beneath the in-RAM radix KV cache. Eligible entries
serialize to disk on graceful shutdown + LRU eviction; after restart, matching
prefixes restore from disk (lazy tensor load) instead of re-prefilling.
Acceptance: post-restart TTFT ≤ 2× warm in-RAM hit AND ≥ 30% faster than cold
re-prefill.

**Implementation (on `main`):**
- `Sources/HTTPServer/Generation/RadixSSDStore.swift` — flat prefix-set index
  (`index.json`, atomic temp+rename) + per-node safetensors under `nodes/`;
  `writeEntry`/`readEntry`; TTL from `createdAt`; config key = `weightDigest`
  + `templateHash`; any failure → miss (never throws into the generation path).
- `RadixKVCacheManager.swift`: `diskState` per node, `setOnEvict` closure,
  `snapshot` (graceful flush), `restoreSkeleton` (startup),
  `matchPrefixForGeneration` (RAM + disk), `promoteToRAM`.
- `MLXGenerator.swift`: `ssdStore` wiring, `loadDiskEntry` (lazy safetensors
  read → `restoreKVCacheState`), Stage 1 RAM→disk path (memory-budget gated),
  `flushToSSD()` (shutdown), `testSSDRoundTrip()`.
- `ServerConfig.swift`: `--kv-ssd-enabled`/`--kv-ssd-disabled`,
  `--kv-ssd-cache-dir`, `--kv-ssd-cache-gb`, `--kv-ssd-ttl-seconds` +
  `QWEN_KV_SSD_*` env overrides.
- `Qwen38Server.swift`: graceful SSD flush on shutdown after drain.
- Engine fork: `restoreKVCacheState` public free function in `KVCache.swift`
  (ArraysCache → `restoreFromMetaState`, simple caches → direct set).
- RFC: `docs/radix-ssd-persistence-rfc.md`.

**Tests:** 6 pure `RadixSSDPersistenceTests` (file-base stable & distinct,
index Codable round-trip, skeleton full-prefix + internal-node restore, TTL
expiry → miss, config mismatch → miss) + 2 weight-gated `RadixSSDWeightTests`
(KV arrays byte-identical persist/restore; post-restore real reuse reports).

**First gate (`benchmarks/run_radix_ssd_restart.sh`, 2048-token prompt):
NOT MET.**

| Metric | Value |
| --- | --- |
| TTFT_warm (in-RAM hit) | 4.965 s |
| TTFT_disk (SSD lazy) | 5.025 s |
| TTFT_cold (re-prefill) | 5.002 s |
| disk / warm (need ≤ 2×) | 1.01× PASS |
| disk / cold (need ≤ 0.7×) | 1.00× **FAIL** |

**Root cause (pre-existing, not introduced here):** the in-RAM reusable path
was broken (Task 4 below) — a same-prompt repeat re-prefilled, so
warm ≈ disk ≈ cold (≈ 5 s full 2048-token prefill). The disk lazy-load itself
was fast and correct (`Lazy-loaded radix prefix (1886 tokens) from SSD tier`
in ≈ 40 ms), but it was followed by a full re-prefill. The SSD tier was kept;
the fix path was: repair the reusable path (Task 4), re-test (Task 5),
re-spec the gate (Task 6).

## Task 4 — In-RAM radix reusable-path repair (2026-09-16)

A same-prompt repeat was doing a full prefill instead of reusing the cached
KV state. Root cause: the generator stored the session state at
`prompt + generation` (1889), not the prompt boundary (1886); a repeat
matches 1886, a strict prefix of the stored 1889, and the GDN/linear layers
are `MambaCache` (recurrent, not trimmable), so `matchPrefix` fell back to 0.

**Fix:** after `begin`, capture `exportState()` and `cache.map { $0.copy() }`
to freeze the prompt boundary, and store the **copied** boundary state.
Added `reusedPrefixTokens` / `radixPrefillSkipped` per-request and
`prefix_reuse_hits` / `prefix_reuse_fallbacks` / `prefix_reuse_tokens_saved`
on `/metrics`. `QWEN_STREAM_DIAG=1` diagnostic log retained (zero overhead
when off).

**Result (E2E, M5 Pro, debug):** original gate run cold 5.73 s, warm 0.37 s
(warm/cold 0.0647, **~15× TTFT win**); re-verified in this checkout cold
5.428 s, warm 0.120 s (warm/cold 0.0221) — `benchmarks/results/
reusable-path-repair/20260916_200050/results.json`. Gate ≤ 0.7 → **PASS**;
outputs token-identical; `prefix_reuse_hits=1`, `prefix_reuse_tokens_saved=1886`.

**Tests:** `swift test --filter HTTPServerTests` green (unmodified
`RadixCacheBenchmarkTests` included); weight-gated
`RadixReusablePathWeightTests` **PASS** (cold == warm, 274 chars).

## Task 5 — SSD gate re-test on repaired main — GATE UNSTABLE (2026-09-16)

**Goal:** re-run the SSD gate (`disk/warm ≤ 2×`, `disk/cold ≤ 0.7×`) with the
Task 4 repair on `main`, and add the post-restore integration assertion.

**New production change:** the prompt-boundary store in
`MLXGenerator.generateStream` now runs **before** `continuation.finish()`
(success path), so the SSD shutdown flush (which snapshots the manager as soon
as the stream completes) sees the stored entry, not an empty tree. Without
this, the store raced the flush.

**New test:** `RadixSSDWeightTests.radixSSDRestoreReportsRealReuse`
(weight-gated) — gen1 + `flushToSSD`, then gen2 (fresh, restores skeleton
from SSD) with the same prompt must report `reusedPrefixTokens > 0` and
`radixPrefillSkipped == true`. **PASS** (`matched=1254 reused=1254
skipped=true`).

**Gate (5 runs, `benchmarks/run_radix_ssd_restart.sh`, 2048-token prompt):
UNSTABLE — NOT MET.**

| Run | TTFT_warm (s) | TTFT_disk (s) | TTFT_cold (s) | disk/warm (≤ 2.0×) | disk/cold (≤ 0.7×) |
|-----|---------------|---------------|---------------|--------------------|--------------------|
| 1   | 0.175760      | 0.999413      | 4.989637      | 5.69× **FAIL**     | 0.20× PASS |
| 2   | 0.125495      | 0.320603      | 4.963170      | 2.55× **FAIL**     | 0.06× PASS |
| 3   | 0.141529      | 0.313339      | 4.970727      | 2.21× **FAIL**     | 0.06× PASS |
| 4   | 0.214610      | 0.313024      | 4.970971      | 1.46× PASS         | 0.06× PASS |
| 5   | 0.309691      | 0.316967      | 4.963166      | 1.02× PASS         | 0.06× PASS |

disk/cold is **consistently PASS** (0.06×–0.20×): the SSD-restored entry
drives Task 4's reusable path (a disk hit skips the prefill, 5–15× faster
than cold). disk/warm **flips** (runs 1–3 FAIL, runs 4–5 PASS) because the
warm baseline is small (0.125–0.310 s) and thermally variable on the M5 Pro,
so the ratio straddles 2.0×. Per the stop condition ("if a ratio flips
PASS/FAIL after 5 runs, stop and report — do not cherry-pick"), not merged.

**Root cause of the instability:** (a) small-denominator noise — post-Task-4
the `disk/warm ≤ 2×` criterion divides a ~0.3 s thermally-variable warm
baseline into a ~0.3 s disk measurement, so the ratio is noise-dominated; (b)
the `--kv-ssd-*` CLI-flag launch form was unstable (Task 7: unknown flags
were leaking into Vapor's `Environment.detect(arguments:)` and crashing
`app.execute()`; the stable launch form is the `QWEN_KV_SSD_*` env vars — and
now that the flags are implemented in `ServerConfig` (Task 3), they are
stripped before Vapor dispatch).

## Task 6 — SSD gate re-spec + equilibration — MERGED (2026-09-16)

**Goal:** re-spec the SSD gate (replace the noise-dominated `disk/warm ≤ 2×`),
re-measure with an equilibration protocol, and merge.

**Gate re-spec (written before measurement):**
- **G1** disk/cold ≤ 0.7× — the load-bearing gate: a disk hit must beat cold
  re-prefill.
- **G2** absolute disk TTFT ≤ 1.5 s.
- **G3** disk/warm — informational only (small-denominator, noise-dominated
  post-Task-4).

**Protocol:** 1 discarded warm-up + 7 measured restart cycles, 60 s cooldowns;
stable launch form (`QWEN_KV_SSD_*` env vars, only `--host`/`--port` as CLI
flags). `benchmarks/run_radix_ssd_equilibrate.sh` implements the protocol.

**Result (median of 7, all 7 cycles pass individually): PASS — MERGED to
`main`.**

| Gate | Criterion | Value | Result |
|------|-----------|-------|--------|
| G1   | disk/cold ≤ 0.7× | **0.117×** | PASS |
| G2   | abs disk ≤ 1.5 s  | **0.609 s** | PASS |
| G3   | disk/warm (info)  | 2.159×    | n/a  |

disk median 0.6095 s [0.3569, 1.1044]; cold 5.2142 s; warm 0.2740 s; lazy
load 1.714 ms. Correctness: lazy-load fires every disk cycle, prefill skipped
(`matched=1886 reused=1886 skipped=true`), bit-identity PASS.

Re-verified in this checkout (`benchmarks/run_radix_ssd_restart.sh`):
TTFT_warm 0.137 s, TTFT_disk 0.144 s, TTFT_cold 5.535 s → G1 0.03×, G2
0.144 s → **PASS**.

**Verdict: GATE MET — on `main`.** Task 5's "UNSTABLE" was an artifact of (a)
the `disk/warm ≤ 2×` criterion and (b) the unstable `--kv-ssd-*` CLI-flag
launch.

## 2026-09-17: FFN prefill GEMM kernel optimization — FFP1 kill-switch (NO-GO)

**Objective:** Reduce long-context prefill wall time by optimizing the FFN-phase
4-bit GEMMs at prefill width M=512 (the default 512-chunk prefill). FFP1 is the
cheap kill-switch: measure incumbent `quantizedMM` headroom at M=512 before
touching the engine.

**Method:** `qmvbench --ffn-prefill --ffn-pair` (release, M5 Pro GPU). Layer-0
FFN (gateup_wide N=34816 K=5120; downproj N=5120 K=17408), sustained no-sync
(128 back-to-back, 1 eval/batch), gateup/downproj **interleaved** at the batch
level for a fair same-DVFS-window comparison, 3 warm-up batches, ~40 s per M.
Shapes independently verified from the safetensors headers; QMV is not involved
at M=512 (QMV is gated to widths 1–9).

**Finding (stable, 68 batches, 2.8% std):**

| M | shape | mean µs | FLOP/s |
|---|-------|---------|--------|
| 512 | gateup_wide | 828 | 220.4 TF |
| 512 | downproj | **4582** | **19.9 TF** |
| 1024 | gateup_wide | 3337 | 109.4 TF |
| 1024 | downproj | 2042 | 89.4 TF |

**Incumbent `quantizedMM` is ~10× off-peak on down_proj at M=512** (20 TF vs
220 TF gateup, same DVFS window; downproj = 84.7% of per-layer FFN time). The
anomaly is width-specific: the same down_proj shape runs at 89.4 TF at M=1024.
Not DVFS: `pmset -g therm` reports no thermal/perf warning.

**Correctness + localization (decisive):** the M=512 down_proj output is
**bit-exact** with a dequantize→bf16-GEMM reference (`max|diff|=0.0`,
`--ffn-check`) — so the headroom is real work, not a Metal JIT zero-result bug.
Timing the same-shape bf16 GEMM at M=512 gives **14.2 TF, slower than
quantizedMM's 21.6 TF** → the anomaly is in the MLX GEMM *tiling engine* for
the small-N/large-K shape at M=512 (both 4-bit and bf16), not a
`quantizedMM`-specific defect.

**Decision: NO-GO (for a bit-exact kernel) — stop at FFP1.** The FFP2
hard requirement is element-wise equality with `quantizedMM` at every M, which
forces the *same tiling / accumulation order* (FP addition is non-associative).
The M=512 headroom sits precisely in the tiling, so preserving it for
bit-exactness preserves the slowness. The only bit-exact FFN wins are fusions,
and down_proj is a bare GEMM (no fusion changes its tiling). Precedent
confirms: the existing specialized kernels are bit-identical to their eager
counterpart and QMV is documented ~5% *slower* than `quantizedMM` even at M=1.
No bit-exact FFN kernel can reach the ≥10% sustained win at M=512.

Full report: `benchmarks/results/ffp1/ffp1-report.md` (negative result,
headroom quantified; a future relaxation of the bit-exact requirement — e.g. a
non-strict-tolerance prefill-only path — would reopen it).

**Caveats:** M=8192 pair-mode numbers are invalid (back-to-128 of a
[8192,34816] output = 73 GB > 48 GB unified memory); large-M needs back-to-back
≤ 2. Random bf16 inputs are valid for dense-GEMM throughput.

## 2026-09-17: Bit-exactness policy v2 — reopen FFP1 under a scoped relaxation (Step 0, decision before code)

**Decision (explicit, recorded before any code).** Bit-exactness is **relaxed
to a numeric tolerance for FFN GEMMs dispatched at prefill chunk widths only**,
and held **byte-for-byte everywhere else**. This **supersedes** (does not
invalidate) the FFP1 NO-GO, which rested entirely on the bit-exactness
constraint the relaxation now removes for the prefill FFN GEMMs.

**Scope (policy v2):**
- **RELAXED (numeric tolerance, a few ulp of bf16 output; fp32-accumulate
differences only):** FFN GEMMs at prefill widths **M ≥ 256**. The threshold
exceeds every verify width (M 1..9) and decode (M=1), and excludes the
short-prompt fixtures `essay-1024` / `specdec-800` (prefill M < 256).
- **UNCHANGED (byte-for-byte):** decode M=1, MTP verify M 2..9, all short-prefill
M < 256, and every other kernel / code path (attention, GDN, norms, …). The
relaxation is FFN-only and does not broaden.
- **Gate:** `MLX_QWEN_FFN_PREFILL_FAST`, **default OFF**, with an M ≥ 256
dispatch geometry gate. The OFF build is byte-identical to current `main`.
- **Accepted consequence:** long-context (8K+) committed streams through the
relaxed path differ from the incumbent registry (knife-edge family: fp
accumulation order, ≤ a few ulp flipping near-tie argmaxes), **not corruption**.
ON runs register **new** per-fixture stream hashes; OFF runs reproduce the
incumbent registry exactly. A divergence at a top-2 logit gap > 8 ulp is a STOP
condition (outside the accepted family → numeric bound too loose).

**Recorded in:** `benchmarks/MTP-CORRECTNESS-CONTRACT.md` §0 (policy v2),
`docs/PREFILL-FFN-KERNEL.md` (central doc + flag reference),
`benchmarks/results/ffp1/ffp1-report.md` (NO-GO marked superseded-by-decision).

**Next: FFP4** — design ≥2 candidate down_proj kernels (split-K, tile/geometry,
simdgroup-matrix, in-register dequant K-restructure) and micro-bench them
(`--ffn-check` tolerance mode + `--ffn-pair` DVFS-fair). Kill switch: best
candidate must show ≥2× sustained throughput on down_proj at M=512 within the
accepted numeric bound, else stop and record the negative result (no engine
touch).

## FFP4 — FFN prefill GEMM kill-switch (relaxed bit-exactness): **NO-GO**

Reopened the FFP1 NO-GO under a scoped bit-exactness relaxation (policy v2,
contract §0): FFN GEMMs at prefill widths (M ≥ 256) may differ from `quantizedMM`
by a few ulp, enabling tiling changes (split-K). Gate: a candidate must hit ≥ 2×
sustained throughput at M=512 within tolerance, else stop with a negative result.

**Result: NO-GO.** The FFP4 kill-switch was run via `qmvbench --ffn-prefill
--ffn-cand` (5 candidates: incumbent, splitk2/4/8, bf16_gemm; tolerance at
M=256/512/1024/8192; interleaved DVFS-fair timing at M=512, 2 reps):

| Candidate   | Rep 1 speedup | Rep 2 speedup |
|-------------|--------------:|--------------:|
| splitk2_qmm | 0.87x         | 0.88x         |
| splitk4_qmm | 0.82x         | 0.83x         |
| splitk8_qmm | 0.71x         | 0.73x         |
| bf16_gemm   | 0.67x         | 0.69x         |

**Every candidate is *slower* than the incumbent** (none even reaches 1.0×, let
alone 2×). More K-splits are monotonically slower (splitk8 < splitk4 < splitk2):
the extra kernel launches + fp32 cross-split accumulation cost more than the
K-parallelism gain. The M=512 down_proj slowness is a fundamental small-M/large-K
GEMM property of the Metal quantized engine, not a tiling artifact — **split-K
cannot capture the 10× headroom**. This extends FFP1's NO-GO: the bit-exactness
relaxation does not open a viable kernel fix.

- **Tolerance:** all split-K candidates ≈99% of elements within 8 ulp (the
  `maxRel`/`maxRelUlp` columns are inflated by near-zero reference denominators —
  a GEMM metric artifact, not a correctness issue). `bf16_gemm` (the slowest
  candidate) hit a bf16 NaN at M=1024 (large-K reference overflow); irrelevant to
  the split-K verdict.
- **Decision:** **FFP5 (engine integration), FFP6 (model-level audit), FFP7
  (A/B matrix) NOT pursued.** FFP1 NO-GO stands, confirmed under policy v2. The
  M=512 down_proj inefficiency remains a known, quantified limitation with no
  kernel-side fix.
- **Bugs fixed in `qmvbench` FFP4 path:** (1) format string `%-14s`/`%s`/`%d` →
  `%@`/`%lld` (Swift String/Int bridging; the `%s` on a Swift String crashed in
  `strlen`); (2) `best.meanUs` initialized to `Double.infinity` (was 0.0, so the
  summary line never updated).
- **Reports/docs:** `benchmarks/results/ffp4/ffp4-report.md` (NO-GO),
  `docs/PREFILL-FFN-KERNEL.md` (status → NO-GO), contract §0 (policy v2).
- **Reproduce:** `cd ../mlx-swift-lm && swift build --product qmvbench -c release &&
  ./.build/arm64-apple-macosx/release/qmvbench --ffn-prefill --ffn-cand --ffn-ms 512
  --ffn-batch 4 --ffn-wall 15`

## 2026-09-17: FFN M-curve probe → prefill-chunk-size sweep (MCP) — MCP1 **GO**, MCP2 **KEEP pc=2048** (default flipped 512→2048)

New task: determine whether the M=512 down_proj inefficiency (FFP1/FFP4, closed as
fundamental to the shape class) is a *fixed* property at prefill-relevant M, or an
M-dependent penalty a larger `--prefill-chunk-size` amortizes. If per-token FFN cost
rises steeply with M, a larger chunk may cut total prefill wall with **zero
kernel/source change** — the only untested config axis. No kernel, engine, model,
quantization, or invariant changes; the only permissible source change is a server
default-pc flip *after* the MCP2 gate passes.

### MCP1 — FFN M-curve (micro, existing `qmvbench --ffn-pair` tooling): **GO**

Sustained no-sync, DVFS-fair interleaved (`--ffn-pair`), per-token cost
(`µs/GEMM ÷ M`) — the metric the sweep trades on. 2 reps at the decision region
(M=512/1024), 1 rep each at the far end. Binary SHA
`a4b56e2b…7aa` (release `qmvbench`), no thermal warning before/after.

| M | gateup µs/tok | down µs/tok | **FFN µs/tok** |
|---|---------------|-------------|----------------|
| 512  | 2.108 | **9.153** | 11.261 |
| 1024 | 4.019 | **2.407** | **6.426** ← min |
| 2048 | 4.920 | 2.608 | 7.528 |
| 4096 | 5.298 | 2.892 | 8.190 |
| 8192 | 5.105 | 2.861 | 7.966 |

The down_proj per-token cost drops **73.7%** from M=512 (the tiling anomaly, ~19.5 TF)
to M=1024 (~74 TF), then the gateup per-token cost rises and the **FFN total bottoms
at M=1024** (−42.9% vs M=512). The M=512 inefficiency is **M-dependent, not fixed**.

**Decision: GO.** down_proj per-token at M=1024 (2.407) is 73.7% below M=512 (9.153),
far beyond the 25% threshold. **Predicted optimal pc = 1024** (the FFN per-token
minimum), stated before MCP2. Predicted 32K prefill savings ≈ 45% (FFN share) × 42.9%
≈ **19%** — well above the 5% MCP2 gate. SDPA/GDN/norms are pc-invariant to first order
(total causal attention is O(L²), independent of pc), so FFN dominates the optimum.

- **Caveat (measurement):** an early M=2048 run at back-to-back=128 hit 14 GB memory
  compression (18.3 GB resident gateup buffer) and was noise-corrupted (std≈mean); the
  reported M=2048 is the memory-safe re-run at back-to-back=64. Back-to-back is scaled
  down with M to cap the resident gateup buffer ~9 GB.
- **Report:** `benchmarks/results/mcp-20260917/mcp1-curve.md`.

### MCP2 — pc sweep end-to-end (config-only): **KEEP pc=2048**, default flipped 512→2048

Protocol: single release binary (SHA `606ed2cf…eb08` per cell), fresh server per cell
(port 18099, `MLX_CHUNKED_PREFILL=1`, greedy), cells pc ∈ {512, 1024, 2048}, 6 reps
(rotating start cell, rep 1 discarded, 5 measured), per-phase breakdown + RSS + tile
buffer + stream hash + admission per rep. Hash gates: 8K/16K/32K bit-exact vs incumbent
registry in every pc cell.

**Phase A (hash gates): all bit-exact.** Every pc ∈ {512,1024,2048} produces *identical*
committed content at 8K (`660dd120`), 16K (`2e583ad2`), 32K (`97bc0d74`). No
knife-edge divergence — no STOP.

**Phase B (32K eval-sync prefill wall, measured reps 2–6):**

| pc | mean wall | vs pc=512 | paired | gate |
|----|----------:|----------:|:------:|------|
| 512 | 149.49 s | — | — | base |
| 1024 | 139.44 s | **−6.7%** | 4/5 | PASS |
| 2048 | 130.58 s | **−12.7%** | 5/5 | **PASS (winner)** |

Per-phase (mean, ms): FFN 72408→65259, GDN 35737→30998, attn 37176→32416, norm
2128→978 (pc 512→2048). The win is not just FFN — every phase improves with pc (fewer,
larger chunks cut per-chunk overhead across the board). MCP1 predicted the optimum at
pc=1024 (the FFN per-token min); the end-to-end shows pc=2048 beats it because the SDPA
per-chunk tiling efficiency at larger Q tiles adds a residual gain on top of the FFN
improvement.

**Phase C (generalization):** 64K pc=2048 = 279.0 s vs pc=512 = 305.2 s (**−8.6%**, 2/2
paired, bit-exact `14b26f9f`) — not regressed. Peak RSS at 64K pc=2048 = 14.6 GB
(per-chunk buffer 6.29 GB) << 48 GB. Essay (short) at pc=2048 works (consistent hash,
`length` finish).

**Decision: KEEP pc=2048.** Bit-exact at every length, −12.7% at 32K / −8.6% at 64K,
memory-safe. Default `prefillChunkSize` flipped **512 → 2048** in
`Sources/HTTPServer/ServerConfig.swift`. `ServerConfigArgumentTests` (8/8) and the full
`HTTPServerTests` suite pass with the new default.

- **Report:** `benchmarks/results/mcp-20260917/mcp2-report.md`.
- **Raw data:** `mcp2-reps.jsonl`, `mcp2-hashgates.jsonl`, `mcp2-phasec.jsonl`, per-cell
  `srv-*.log` / `resp-*.json` / `rss-*.csv` / `wall-*.txt` in the same directory.
- **Scripts:** `benchmarks/run_mcp2.sh` (Phase A+B), `benchmarks/run_mcp2_phasec.sh`
  (Phase C), `benchmarks/results/mcp-20260917/analyze_mcp2.py` (gate).

---

## 2026-09-17: Bit-exactness policy v3 — global relaxation; DETERMINISM is the hard gate (recorded FIRST)

Bit-exactness is **no longer a project invariant** (inherited from the challenge repo; superseded).
Recorded in `benchmarks/MTP-CORRECTNESS-CONTRACT.md` §0c. What remains:

- **Determinism (hard gate, non-negotiable):** same config (binary + env + fixture) must
  reproduce an identical stream hash across reps. A failing run is discarded, never reported as
  a perf delta.
- **Cross-config / cross-version bit-exactness NOT required.** New configs/versions register
  their own per-fixture stream hashes in the registry; they are not byte-compared to the incumbent.
- **Kernel-affecting-change acceptance:** (a) characterize divergence vs incumbent as the
  knife-edge family (first-divergence position, rate, top-2 logit gaps); a rate far above ~9/1024
  or any flip at a top-2 gap > 8 ulp is a **STOP**. (b) MTP acceptance within noise of ~93.5–94.7%.
- Existing default-ON wins (fusions, QMV routing, q4 head, pc=2048) are **unaffected**.
- Reopened as **separate future tasks** (not this one): a prebuilt flash/SDPA **prefill** kernel
  (if upstream ships one), and model-level levers (2-token head, draft-vocab lm_head, denser
  quantization).

## 2026-09-17: Upstream MLX decode-bandwidth probe — U1 survey (evidence, no code)

Objective: close the last quantified decode headroom — in-pipeline **200–275 GB/s** vs qmvbench
**310–355 GB/s** sustained on the 14.4 GB 4-bit weight stream (per-layer 205 GDN / 209 FA GB/s).
Determine whether a newer upstream MLX/MLXSwift closes it; upgrade the pin only if the
end-to-end gate passes.

**Pinned (engine fork):** mlx-swift **0.31.6** (the latest *released* tag) wrapping **C++ MLX
v0.31.1** (submodule `ce45c52`) + mlx-c v0.6.0. Pin: `.upToNextMinor(from: "0.31.6")`.

**Upstream state:** the latest *released* mlx-swift tag is **0.31.6 = our exact pin** (no 0.32.x
release, no pre-releases). Unreleased **main** (`2bebe4e`) has 17 commits since the pin; the key
one is `ab924c8 update for mlx v0.32.2 (#450)`, which bumps the **C++ MLX** submodule
**v0.31.1 → v0.32.2**. The Metal kernels live in C++ MLX, so the kernel-relevant range is
**C++ MLX v0.31.1 → v0.32.2** (reached only via unreleased mlx-swift main).

**Relevant kernel changes (C++ MLX v0.31.1 → v0.32.2), all in the decode-GEMV/attention path:**
- `548dd80e` Add small-batch quantized matvec kernel (**qmv_wide**) — decode GEMV M=1..9.
- `5a1e44c3` Optimize large NVFP4 QMV on **M5 Max** — QMV on M5-class GPUs.
- `e7838d5e` Raise qmv batch limit for large matrices on **M5-class** GPUs.
- `38ad2570` Add **split-K for quantized matmul (small M)**.
- `fa0d4463` Read each K/V byte once in **gqa-8 decode attention** (GQA; our 4 KV heads).
- `8056817b` Derive the qmv fast path K alignment from bits. `1700b39a` Fix fp qmatvec out-dim<8.
- NAX-only (if the M5 Pro is NAX, runtime-detected): `714a7efc` fused full-attention for
  **head_dim 256 on NAX**; NAX qmm kernels.
- Memory/allocator (possible step-time effect): `09ebe730` break monolithic MTLResidencySet;
  `291e909f` reuse Metal WAR tables; `f599c020` fix concurrent kernel-cache lookup.

**Decision: GO to U2.** Relevant kernel improvements exist in the decode path. The only way to
get C++ MLX v0.32.2 is via **unreleased mlx-swift main** (no tagged release carries it); the
`ab924c8` commit notes the upgrade is not clean (a carried C++ MLX patch for `Device/Stream
operator<`; streams became thread-affine in v0.31.2). So U2's **build gate** and **divergence
audit** carry the risk. U2: bump the pin to mlx-swift main (record SHA), build engine+server
(green), then run the decode A/B matrix (essay-1024 + specdec-800, 6 reps) + a 32K prefill
regression cell + qmvbench attribution + one in-pipeline step-trace cell.

**Survey deliverable:** `docs/UPSTREAM-MLX-SURVEY.md`.

### U2 — Framework upgrade A/B: **REVERT (build-infrastructure barrier)**

Attempted the pin bump `mlx-swift 0.31.6 → main 2bebe4e` (C++ MLX **v0.31.1 → v0.32.2**).

- **Build gates (Swift/C++) green, zero compat fixes:** engine `MLXLLM` + server `HTTPServer`
  both build; the only change is the pin (no source edits). Our path `Qwen38MTPDiagnosticTests`
  **PASS** (3/3).
- **Barrier:** full engine suite **crashed** on `testQwen35MoECompiledDecodeTracksWeightUpdates`
  (`Unable to load kernel dot_product_float32_it32_tg512_sg16`). Root cause: Cmlx builds C++
  MLX in **NO-JIT** mode bundling a **prebuilt** `default.metallib`, which is a **stale v0.31.1**
  artifact (Sep 14) from a separate **PrepareMetalShaders** step that **SwiftPM does not
  regenerate on a pin bump** (warm build = no-op; deleting the metallib doesn't make SwiftPM
  rebuild it). So the "upgraded" build runs **v0.32.2 C++ against v0.31.1 kernels** — the
  improved kernels (qmv_wide, NVFP4 QMV M5 Max, split-K quantized matmul, gqa-8 decode attn)
  are **not active**, and a v0.32.2 dot_product kernel is missing (the crash).
- **Consequence:** a clean A/B is **impossible via the pure pin-bump path** (the "upgraded"
  binary wouldn't run the v0.32.2 kernels); the build gate (engine suite green) is **not met**.
  This is **not** a "no fix exists" (U4) closure — the kernel improvements DO exist in v0.32.2;
  the blocker is the metallib build gap.
- **Action:** pin **reverted** to `0.31.6` (engine + server). Verified **green**:
  `CompiledDecodeWeightUpdateTests` 6/0 (MoE kernel loads again).
- **Follow-up (separate task):** regenerate `default.metallib` for v0.32.2 via the
  PrepareMetalShaders CMake step → confirm engine suite green → then run the decode A/B matrix
  (§U2 plan in `docs/UPSTREAM-MLX-SURVEY.md` §5).

## 2026-09-17: MLX v0.32.2 platform refresh — MER1 (green suites) + MER2 (interleaved A/B)

### MET — Metallib unblock + v0.32.2 decode A/B: **DONE (cross-session INCONCLUSIVE; superseded by MER2)**

Unblocked the U2 barrier (the stale prebuilt metallib). `scripts/build-metallib.sh` builds the
metallib via the CMake `mlx-metallib` target for the pinned C++ MLX revision (`1f8e74e` =
v0.32.2), cached per revision, colocated at `<exe-dir>/mlx.metallib` (first runtime search
path), provenance SHA-256 `b57de586…`; `check` mode is the stale-metallib detector. The U2
crash case (`MoE … dot_product`) PASSES with the fresh metallib; the release server starts
clean (`readyz=200`). Cross-session decode A/B: essay +1.5%, specdec +1.1% — below the 3% KEEP
gate (INCONCLUSIVE, cross-session-thermal-confounded). **Superseded by the MER2 interleaved
A/B below (the definitive measurement).**

### MER1 — Suites green under policy v3 (COMPLETE)
- Policy v3 (bit-exactness relaxed, determinism = hard gate) recorded in contract §0c.
- Continuation tests (53), Fused bit-exactness (18 → tolerance 0.02, measured max|diff|=0.015625),
  metallib SHA gate (1), decode canary (1) — **78/78 pass**.
- swift-testing `ParallelFileReader` crash triaged (MER1.3): C++ MLX static `ThreadPool{4}`
  throws `std::runtime_error` on `pread==0` at EOF → uncatchable Swift fatal error. NOT a
  decode-path regression; NOT fixable in the engine fork (upstream C++ MLX).
- Engine `6e8eab2`, server `2d4989a` on `feature/mlx-v0322-upgrade`.

### MER2 — Interleaved A/B (v0.31.6 incumbent vs v0.32.2 upgrade): **STOP (determinism gate) → SUPERSEDED by ND**
Both binaries built + provenance-recorded (engine production code identical on main vs feature;
ONLY runtime diff = C++ MLX version + metallib):
- incumbent: binary `5b3f761f…`, metallib `db499101…` (C++ MLX `ce45c52`, v0.31.6)
- upgraded: binary `606ed2cf…`, metallib `b57de586…` (C++ MLX `1f8e74e`, v0.32.2)

6-rep interleaved matrix (rotating start), essay-1024 + specdec-800, 32K prefill cells:
- **Decode:** essay incumbent 23.22 → upgraded 24.99 tok/s (**+7.7%**, all 5 paired reps favor
  upgrade); specdec +5.6% (1 paired rep, matrix stopped at determinism gate).
- **Prefill (32K):** incumbent 144.06 s → upgraded 125.50 s (**-12.9%**, faster, within-noise
  gate satisfied); both content-hash `97bc0d74…` (deterministic).
- **Determinism gate: FAIL (incumbent side) — root-caused as cache HIT/MISS (see ND below).**
  - essay: both sides deterministic (12/12 = `949b9423…`).
  - specdec-inc (v0.31.6): r1=`139acb9d…` (MISS) → r2=`06882d85…` (HIT); 8/9 runs=`06882d85…`.
    **The incumbent is non-deterministic on specdec (pre-existing, NOT a regression — the
    store-on-success prefix-cache HIT/MISS, root-caused in ND1/ND2).**
  - specdec-upg (v0.32.2): all = `139acb9d…` (deterministic, matches registry; fixes the split).
  - First-divergence inc(`0688`) vs upg(`139a`): char 4847/5150 (94% through), 94.5% similar —
    knife-edge decision near the stream end (first flip at token 973/1024 = 95.0%).
- **Merge decision: SUPERSEDED by ND.** The initial STOP was based on the strict "determinism
  on either side" gate. ND1/ND2 root-caused the incumbent's non-determinism as the store-on-
  success prefix-cache HIT/MISS (per-prompt, persists across processes via SSD), which is the
  registered knife-edge family (gap ≤ 4 ulp, first flip 95 %). The gate was amended (policy v3
  §0d: determinism is per cache state). The upgraded binary is deterministic at `139acb9d…` for
  BOTH cache states (fixing the split). **Merge proceeds (see MER3 below).**

### Upstream issue filed
swift-testing `ParallelFileReader` fatal error (MER1.3) →
**https://github.com/ml-explore/mlx/issues/4526**

### ND — Root-cause of the incumbent's cold/warm non-determinism (COMPLETE)
- **ND1 (trigger):** the store-on-success prefix-cache HIT (per-prompt; RAM in-process + SSD
  cross-process, `~/.qwen38-mtp/kv-ssd/`). MISS ⇒ full prefill ⇒ `139acb9d…`; HIT ⇒ snapshot
  replay ⇒ `06882d85…`. Not global warm-up (Control 4: essay interleaved, specdec stays cold).
- **ND2 (flip location):** warm-run hash = `06882d85…` exactly (registered knife-edge variant);
  first divergent token index 973/1024 (95.0 %); cold at 973: `58377` (` rollback`), warm:
  `8476` (` dynamic`). Matches Phase 1 Bug A (gap ≤ 2–4 ulp, ~9/1024 flips, first flip 95–96 %).
- **Origin (hypothesis a):** the decode path is the SAME MTP verify for cold and warm; the only
  difference is the prompt-boundary state (the stored KV/hidden snapshot is a different bf16
  reduction path than a fresh full prefill). The drift is in the **cached-prefill replay, not
  the decode path's warm dispatch** ⇒ ND3 (decode-knob bisect) NOT required.
- **Gate amendment (policy v3 §0d):** determinism is per cache state. The upgraded binary PASSES
  strictly (identical on MISS, strictly better on HIT — it fixes the cold/warm split).
- **Artifacts:** `benchmarks/results/nd-specdec-20260917-1946/` (nd-findings.md, nd1-trigger-
  table.tsv, nd2-{cold,warm}-{ids,content}.txt, nd2-first-divergence.txt).

### MER3 — Merge to main (COMPLETE)
- Gate 3 (determinism) amended to per-cache-state (policy v3 §0d); upgraded PASSES strictly.
- Engine `feature/mlx-v0322-upgrade` (`6e8eab2`) merged to main; branch deleted.
- Server `feature/mlx-v0322-upgrade` merged to main; branch deleted.
- All tests green (engine Qwen38MTPDiagnosticTests 3/3; server HTTPServerTests 237/237).

### Artifacts
- `benchmarks/results/mlx-v0322-merge-20260917-1831/` (per-rep JSONL, thermal.log, prefill/,
  analysis.md)
- `benchmarks/results/nd-specdec-20260917-1946/` (ND1/ND2 findings)
- `/tmp/mer2/incumbent/` + `/tmp/mer2/upgraded/` (binary + metallib + provenance.txt)
- `docs/UPSTREAM-MLX-SURVEY.md` §7b + §8 (MET + MER2 results + issue URL)

## 2026-09-17: MER4 — Post-merge kernel re-baseline on MLX v0.32.2 (COMPLETE)

Map which v0.32.2 kernel changes actually engage at our geometry, measure their effect, and
re-check the config decisions (pc, QMV thresholds) that were optimized against v0.31.6 kernel
behavior. No speculative work — only measurements that change a decision. Run ID
`mer4-20260917-2137`.

### MER4.0 — Benchmark cache hygiene (harness)
- `run_cell.sh` now takes an 8th arg `CACHE_STATE` (default `MISS`):
  - `MISS`: clear `~/.qwen38-mtp/kv-ssd/` + fresh server (no RAM cache) → full prefill by
    construction.
  - `RAM-HIT`: fresh server + one throwaway priming request → guaranteed RAM hit by
    construction (SSD cleared so the prime is a pure RAM store, not an SSD promotion).
- Per-rep recording: `cache_state` (controlled value) + `cache_ssd_promoted` (verified from
  the server's `Lazy-loaded radix prefix (N tokens) from SSD tier` log line) in the cell JSON.
- Re-verify one essay decode + one 32K prefill cell under the controlled protocol (reproduce
  the registered values) before proceeding to the re-baseline.

### MER4A — qmvbench sustained re-baseline (v0.32.2)
- **MER4A.1 (verify-width M=1..9): DONE — UNCHANGED.** Sustained `wide_global`: M=1 wash
  (0.94→0.97x), M=2..9 routed wins (1.18–1.55x) — identical profile to v0.31.1. The QMV
  dispatch threshold does NOT need re-tuning. (`mer4a1-verify-comparison.md`.)
- **MER4A.2 (FFN M-curve M=512..8192): DONE — anomaly PERSISTS.** M=512 down_proj = 9.818
  µs/tok vs 2.857 at M=1024 (3.4×) — the upstream split-K did NOT close the small-M/large-K
  tiling anomaly. The curve did NOT flatten; interior minimum still at M=1024. **pc=2048 NOT
  stale.** (`mer4a2-ffn-mcurve-comparison.md`.)
- **MER4A.3 (ffn-check, dequant reference): DONE.** M=512 bit-exact (max|diff|=0.0);
  M=1024/2048 dequant-ref overflow (test artifact). No tolerance concern.

### MER4B — In-pipeline attribution (one essay decode cell, controlled state) — DONE
- tEvalAvg=81.6 ms (reproduces MER2's 80.7). **In-pipeline BW = 14.4 GB / 81.6 ms = 176.6
  GB/s** vs 310–355 GB/s sustained. Gap PERSISTS on the v0.32.2 kernels → ceiling is
  **per-dispatch/state-bound**; decode-bandwidth question closes with a mechanism (next lever
  = scheduling, not kernels).

### MER4C — Conditional config re-checks (only if MER4A moved) — DONE, both CLOSED
- pc sweep NOT triggered (M-curve did not flatten); QMV threshold NOT re-tuned (profile
  unchanged). **pc=2048 stands.** No sweep.

### MER4D — Engagement map + ranked next tasks — DONE → `docs/V0322-KERNEL-BASELINE.md`
- Only qmv_wide/M5-batch engage (profile unchanged); split-K does NOT close the anomaly;
  gqa-8 (our GQA 6) and NVFP4 (M5 Pro) unreachable. No new kernel lever at our geometry.

### MER4.0 — Verification (controlled protocol) — DONE
- essay decode: ttlt=25.65 (registered 24.99), tEval=81.6 ms (registered 80.7), stream hash
  `949b9423` EXACT. 32K prefill: wall=132.5 s (registered 125.50 s, Δ+5.6%), prompt_tokens=
  32780 EXACT, cache_state=MISS controlled. **All reproduce.** (`mer40-verification.md`.)

## 2026-09-17: PRO — In-pipeline bandwidth gap probe (run `pro-bw-20260917-2306`) — **COMPLETE: GO for scheduling task**

Diagnostic-only task (no production source changes; qmvbench `--layer-seq` probe added;
FullBench reused). Locate the ~2× in-pipeline BW gap (176.6 GB/s vs 310–355 GB/s
sustained): (a) hardware interleave? (b) server/round dispatch? (c) measurement artifact?

- **PRO0 (headline refresh, 18 cells, controlled MISS, interleaved):** essay tEvalAvg med
  83.2 ms, specdec 85.6 ms (confirms 81.6 ms anchor); stream hashes `949b9423`/`139acb9d`
  match registry exactly (no cold/warm drift on v0.32.2); 32K prefill 118.2 s (matches
  ~125 s MER2 anchor); TTFT MISS 1.56 s vs RAM-HIT 0.78 s (2×).
- **PRO1 (QMV interleave + weight-rotation, `--layer-seq`, M=3):** cell 1 single-gateup
  177.4 GB/s, cell 2 interleave 176.8 GB/s (**99.6%**), cell 3 thrash (64-layer rotation)
  173.6 GB/s (**98.2%**). **Interleave and weight-rotation are FREE → rules out (a).**
- **PRO2 (FullBench round replay, M=1/M=3/serial):** M=1 serial 58.82 ms (**250 GB/s**),
  M=3 verify 79.52 ms (181 GB/s), in-pipeline (M=3) 83.2 ms (176.6 GB/s). In-pipeline ≈
  M=3 verify + **3.7 ms** (draft + accept/rollback + prime-depth). The round dispatch is
  **not** the gap.

**Mechanism verdict:** (a) NO (interleave/thrash free), (b) **YES (dominant)** — the gap is
the per-dispatch sync + M effect (M=3 verify with per-step sync = 176.6 GB/s vs FFN
sustained no-sync = 310–355 GB/s; M=3 less memory-bound than M=1 at 250 GB/s), (c) NO
(consistent across PRO0/PRO2, hashes match). **GO for a scheduling task:** the lever is
batched/fused dispatch + state amortization across MTP steps (engagement map lever #1),
targeting the per-dispatch sync + state setup, NOT the kernel mix (PRO1: free).

**Files:** `benchmarks/results/pro-bw-20260917-2306/pro-findings.md` (+ PRO0/PRO1/PRO2
raw cells). Engine: qmvbench `--layer-seq` probe (uncommitted diagnostic).

## 2026-09-18: SCH1 — MTP round kill-switch ledger (run `sch1-20260918-0041`) — **COMPLETE: STOP**

First phase of the MTP round scheduling task (PRO GO lever #1: batched/fused dispatch +
state amortization). Kill-switch protocol: instrument every ms of a steady-state k2 round,
classify into (i) kernel exec, (ii) sync/idle, (iii) host build/read, (iv) round-structure
overhead; if (ii)+(iv) < 8 ms/round (~10%), STOP (not worth the restructure).

**Instrumentation:** xctrace Metal System Trace (launched mode) per-encoder GPU intervals
(server pid, 36,644 intervals) + `QWEN_MTP_STEP_TRACE`/`MLX_QWEN_MTP_TRACE` host phase
stamps (mtp-anchor mach-uptime + mtp-trace µs), joined on mach-uptime (median offset 42 ms,
total busy stable 0.5% across 0–40 ms). Steady-state k2 decode, essay-1024, MISS, greedy,
457 rounds (5..461).

**Ledger (median, 429 sane rounds):**
- round total **84.6 ms**; **(i) kernel exec 76.43 ms (90.6%)**, **(ii) sync/idle 1.224 ms
  (1.4%)**, **(iii) host build/read 3.920 ms (4.6%)**, **(iv) round-struct 1.451 ms (1.7%)**.
- GPU util in eval window **97.4%** (matches prior K2 98.4%).
- **(ii)+(iv) = 2.675 ms < 8 ms → KILL SWITCH TRIGGERS → STOP.**

**Verdict:** The per-round scheduling restructure (lever #1) is **not worth it** — only
2.68 ms/round (3.2%) is addressable. The 176.6 GB/s in-pipeline bandwidth is an **M=3 verify
geometry property** (90.6% kernel exec, 97.4% GPU-busy in eval), NOT a scheduling artifact —
not closable by batched/fused dispatch. No source changes (no-code STOP). (iii) verify_build
(3.92 ms, 4.6%) is a separate host-side graph-construction lever, out of scope, and below the
8 ms threshold on its own.

**Files:** `benchmarks/results/sch1-20260918-0041/sch1-findings.md` (+ `sch1-ledger.csv`,
`sch1-gpu-intervals.xml`, `sch1-mtp-trace.log`, `sch1_final.py`).

## 2026-09-18: Residual-lever triage campaign — Phase 0 (record hygiene) + campaign ledger

Campaign: "Residual-lever triage — measure-first, kill-switched." Every remaining
performance lever follows: measure/model FIRST → pre-stated GO bar → implement only
on GO → NO-GOs recorded and closed (kill switches binding). Standing protocol per
measurement: policy v3 §0d determinism (per (binary, config, cache state)), MER4.0
cache-state control (prefill/TTFT cells forced MISS, decode cells pinned), 6 reps /
rep-1 discarded / interleaved rotating start / pmset -g therm per rep / single binary
per comparison / phase-sum gates / engagement proof from logs / binary + metallib
provenance. In-session paired deltas only.

### Phase 0 — record hygiene (this entry)

- **Anchor divergence (recorded, per stop-condition policy):** the plan names stale
  "Next Steps" and "Active Context" sections *in progress.md*; the live versions of
  those sections are `docs/HANDOFF.md` ("Status" / "Next step (exact)") and the
  "Current status and roadmap → Open items" list here. Hygiene applied where the
  sections actually live:
  - HANDOFF.md "Status" + "Next step (exact)" rewritten to the campaign gate map
    (completed items removed: speculative sampling, penalties, TTL caching,
    KV-quantization call-sites, chunked prefill / Path B; pc default is **2048**,
    not 512).
  - This file's "Open items" list below is superseded by the campaign ledger
    (items 1–8 of the old list are all resolved/closed; nothing is dropped, only
    re-homed).
- **Pre-campaign tree state:** server `main` @ `dc5d80e` carried uncommitted local
  experiments (`temp 0.0→1.0`, admission preflight cap `4096→32768` in
  `OpenAIRouter.swift`, `.DS_Store`); stashed as `stash@{0} "lev-campaign: pre-
  campaign local experiments (temp 1.0, admission cap 32768)"` before work began.
  Engine `main` @ `cfd6df5` clean. Campaign branches:
  `feature/prompt-lev-campaign` in both repos.
- **Anchor divergence (recorded):** the plan's LEV-C names `RuntimeStartupMemoryPolicy`
  knobs (512 MB / 50 ops per command buffer). **No such knobs exist in either
  checkout** (zero occurrences in server `Sources/` or the engine fork). The LEV-C
  knob matrix is CLOSED as "anchor absent"; LEV-C's startup-time decomposition
  (first bullet) remains in scope.

### Phase 1–2 — measurements + zero-code verdicts (2026-09-18, this entry)

All Phase 1 (measurements) and Phase 2 (zero-code) levers are complete. No source
behavior changed except **gated, trace-only instrumentation** (LEV-C
`QWEN_STARTUP_TRACE` startup stage timer in `MLXGenerator`; LEV-E `snap_us`/
`tape_us` on the existing `traceRounds` trace line in the engine fork). Both
trees tested green (engine `Qwen38MTPDiagnosticTests` 3/3 incl. wide-verify
serial-family 1.0000; server `HTTPServerTests` 237 tests / 7 suites) and merged
to `main` (engine `9f4ceb9`, server `e86a342`). Full verdict ledger:
`benchmarks/results/lev-campaign-verdict-ledger.md`.

- **LEV-D (zero-code, GO):** flash + large-pc compounding model. Flash removes
  the 6.3 GB (64K pc=2048) / ~25 GB (pc=8192) scores buffer; solved `c_flash`
  bar ≈ 630 µs/tok @32K, ≈1205 µs/tok @64K. A credible Metal flash kernel
  (20–204 µs/tok) is 3–35× below the bar → **credible; hand to LEV-J** (do not
  close flash). `benchmarks/results/lev-d-arithmetic/`.
- **LEV-E (zero-code, CLOSE):** draft-select/accept-walk bound. (iv)
  commit+upkeep + snap + tape + readout = **0.81 ms (fresh run2) to 1.60 ms
  (SCH1, conservative) per round < 2 ms kill-switch**. `commit` is bimodal
  (cheap ~0.15 ms when both drafts accepted; ~1.1–2.1 ms on the rollback path);
  the walk's `verify_build` share (snap+tape) is ~0.06–0.09 ms, the rest is the
  64-layer verify graph encode + async-ladder GPU wait (not walk-owned). Corro-
  borates the Swift-walk negative (2.05 % slower). **CLOSE `qwen35DraftSelect-`
  Kernel (LEV-K).** `benchmarks/results/lev-e-draftselect/`.
- **LEV-F (zero-code, ranked):** tree-drafting + conversation-resume design
  study. Shared blocker = a **composable generated-state checkpoint** (KV + GDN
  `h` + MTP head) with **forward-only rollback** (GDN `h` is not invertible). Tree
  headroom is marginal (<5 %, acceptance ceiling ~1.7–1.8 acc/step, superlinear
  verify cost, SDPA ≤9-row fused-verify limit) → **Stage-3 conversation resume
  first (lower risk, reuses the radix state-store), tree-drafting as a gated
  follow-on.** `benchmarks/results/lev-f-design-study/`.
- **LEV-A (measurement, NO-GO as default):** fp16 vs affine8 KV @32K, 6 reps.
  affine8 is **+16 %/step slower** (+36 ms/step host graph-build = quantize/
  dequant) and **changes the stream** (coherent but divergent; first divergence
  char 165, "utilities"→"helpers"). Only upside is KV memory headroom. **Keep
  fp16 default; affine8 = explicit memory-recovery option.** Consistent with the
  q4 hard rule. `benchmarks/results/lev-a-kvquant/`.
- **LEV-B (measurement, split verdict):** 32K requires `MLX_CHUNKED_PREFILL=1`
  (dense 32K prefill overflows the buffer — a standing constraint). (1)
  `MLX_QWEN_FUSED_GDN` (verify, widths 3–9): **bit-exact, −1 %/step (within
  noise)** → safe to default-ON, marginal. (2) `MLX_QWEN_FUSED_GDN_PREFILL`: the
  startup banner only reflects `MLX_QWEN_FUSED_GDN`, so it reads "off" even when
  `_PREFILL` is set; measured, `_PREFILL=1` **deterministically changes the 32K
  stream** (both configs self-consistent) → **not bit-exact at prefill widths →
  FLAG: verify prefill-width bit-exactness before it can be a default.**
  `benchmarks/results/lev-b-fusedgdn/`.
- **LEV-C (measurement, decomposition):** cold start = **25.4 s to warmup,
  ~27 s to readyz** (default config, SSD on). Warmup/kernel-JIT **66.5 %**,
  SSD restore **27.7 %** (1 prefix, operator-toggleable via `--kv-ssd-disabled`),
  weight load **5.7 %** (15.13 GB / 3 shards, warm-disk). `RuntimeStartupMemory`
  Policy knobs **absent** (confirmed); actual admission values reported (limit
  47.24 GB, reserve 4.29 GB, modelBaseline 15.37 GB, kvBudget 27.58 GB, 16/16
  bits, tail 1024). LEV-H gate **not met** (the SSD-tier delta is SSD restore,
  not tokenization). Two startup levers flagged for Phase 3: persistent runtime-
  kernel compile cache (cuts the 16.9 s warmup); lazy/async SSD restore (moves
  the 7.0 s off the critical path). `benchmarks/results/lev-c-startup/`.

### Remaining-avenues ledger (campaign — supersedes the old "Open items" list)

| ID | Lever | Phase | Gate / GO bar | Status |
|----|-------|-------|---------------|--------|
| LEV-A | KV-quant cache default (fp16 vs affine8 @ kvTail 1024) | 1 | own AB | **NO-GO (measured)** — affine8 +16 %/step slower + stream-divergent; fp16 stays default, affine8 = explicit memory option |
| LEV-B | Fused-GDN re-AB (`MLX_QWEN_FUSED_GDN` / `_PREFILL`), 32K | 1 | own AB | **verify: GO-as-default (bit-exact, −1 %/step, marginal); prefill: FLAG** (`_PREFILL` not bit-exact @32K — verify prefill-width exactness first) |
| LEV-C | Startup-time decomposition (instrumentation only) | 1 | own measurement | **measured** — 25.4 s: warmup 66.5 %, SSD 27.7 %, weights 5.7 %; knobs absent (reported); 2 levers flagged |
| LEV-D | Flash + large-pc compounding model (c_flash bar) | 2 | arithmetic | **GO** — credible (bar 630/1205 µs/tok; a flash kernel is 3–35× below); hand to LEV-J |
| LEV-E | Draft-select Metal ceiling (walk host bound) | 2 | SCH1 ledger | **CLOSE** — walk bound 0.81–1.60 ms/round < 2 ms |
| LEV-F | Tree-drafting + conversation-resume design study | 2 | design soundness | **ranked** — Stage-3 resume first; shared composable generated-state checkpoint |
| LEV-G | Flip KV default (impl) | 3 | LEV-A GO | **blocked** — LEV-A NO-GO |
| LEV-H | Tokenization-cache persistence (serialize/restore) | 3 | LEV-C attributes the SSD-tier delta to tokenization | **blocked** — LEV-C attributes it to SSD restore, not tokenization |
| LEV-I | Startup memory-policy tuning | 3 | LEV-C knob sensitivity | **CLOSED** — anchor knobs absent (confirmed by LEV-C) |
| LEV-J | Flash-attention Metal kernel (FFP-style micro kill-switch vs the LEV-D bar) | 3 | LEV-D bar credible | **FB9 CLOSED (2026-09-20): LEV-J = PERFORMANCE NO-GO (terminal).** The MLX dispatch bug (the FB7/FB8 blocker) is **FIXED upstream** (PR ml-explore/mlx#4535, commit 346eff75, issue #4534) + full-coverage regression test; the production path is unaffected (fix is behind the OFF gate). **FB9 re-bench at production geometry** (Q=2048/8192, prefixes {8192,32768,65536}, 100 serial + concurrent, 3× hash, dispatch fix, no flashbench tuning) — **correct + full + deterministic, but ~57.7× SLOWER than dense** (e.g. Q=2048/prefix=8192: flash 47.87 ms vs dense 0.830 ms). **Per-row threadgroup granularity (24 threadgroups) is not viable at production depth; the kernel is a research prototype, not a production kernel.** No flash-attention performance claim; no flash routing; gate OFF (byte-identical production); no perf claim. **Dead flash integration REVERTED** (production byte-for-byte equivalent to pre-LEV-J; only the FlashBench harness target preserved); dispatch regression coverage moved to MLX. Full report: `benchmarks/results/lev-j/fb9-rebench/fb9-findings.md`. **Prior terminal (superseded): FB8 BLOCKED-on-upstream.** **Version probe (latest mlx-swift origin/main 9019419, 2026-09-17, 5 commits ahead of pinned 2bebe4e9): FAILS** — the recombination persists (a-nz=49152 TRUNCATED at Q=2048/prefix=8192; a-nz=196608 TRUNCATED at Q=8192). The 5 commits touch 100 files (integration tests, stream pooling #472, Cuda #478, logging #484, distributed #482) but **NOT the Metal dispatch layer**. **Guard experiment (Item 3): DIES** — for a pure same-geometry sequence (the production pattern), ALL 10 dispatches are TRUNCATED (nz=49152, same hash); re-dispatch does NOT clear the truncation; the guard would loop forever. **Production incumbent (gate OFF, byte-identical).** **RE-TEST TRIGGER:** any new mlx-swift/cmlx release OR upstream Metal-dispatch fix → re-run FB8 Item 1 at production geometry (Q=2048/8192, prefixes {8192,32768,65536}, 100 serial + concurrent + 3x hash); 0/100 → proceed DIRECTLY to FC (model-level audit) + FD (end-to-end AB with pc re-sweep), both specced. **Documented caution (FB5–FB8 retracted-label history):** the mechanism was mis-labeled FOUR times (FB2 "non-deterministic" → FB3 "allocator" → FB5 "m/out alias" DISPROVEN → FB6 "grid truncation/COLD-JIT" → FB7 "recombination Q≤256" → FB8 "fires at ALL Q; 0.169 is the dispatch bug"). **Lesson for future Metal-kernel work:** trivial probe (no flash code), Q sweep, pure vs interleaved dispatch sequences, verify at PRODUCTION geometry (Q≥2048), nz-count check before "numerics", version probe before banking BLOCKED. `benchmarks/results/lev-j/fb8-production-geometry/README.md`. **No production code changed.** Gate OFF. **Item 1:** the strengthened gate at PRODUCTION geometry (Q=2048 and Q=8192) **FAILS** — the first dispatch of a fresh geometry is recombined (a-nz = 24·Q), for ALL Q (not just Q≤256). Q=2048/prefix=8192: a-nz=49152 (=24×2048, TRUNCATED), b-nz=12582912 (full). Q=8192/prefix=8192: a-nz=196608 (=24×8192, TRUNCATED), b-nz=50331644 (full). 3x hash: h1 (truncated) ≠ h2=h3 (stable). Concurrent-pairs: prefix=8192 clean, prefix=32768 fails 20/20. **Item 2:** the flattened 1-D grid is ALSO recombined (a-nz=24·Q at Q=64 and Q=2048) — the workaround does NOT work; FB7 rejection CONFIRMED. **Item 3:** Q=16/prefix=256 flashNZ=384 (TRUNCATED) — the 0.169 is the **DISPATCH BUG** (truncation), NOT reduction order; **FB7 A4 "DIFFUSE (numerics)" was WRONG**. Q=2048: flash full, maxDiff vs dense = 0.10888672 (reduction-order, > 0.0625 bound). **Record corrections:** (1) FB7 A4 → the 0.169 is the dispatch bug (flashNZ=384); (2) FB7 "Q≤256" → fires at ALL Q; (3) FB7 Part B → CONFIRMED (flattened also recombined). **Terminal: BLOCKED-on-upstream** (complete filing ready). No production code changed. Gate OFF. `benchmarks/results/lev-j/fb8-production-geometry/README.md`. **FB7 (2026-09-20): dispatch RECOMBINATION mechanism PINNED; 1-D grid workaround REJECTED (SUPERSEDED by FB8: fires at ALL Q).** The FB6 "grid truncation / COLD-JIT" label is **superseded** by the **recombination signature**: **grid.y→group count, grid.x→1-D thread count, threadGroup.x→threads-per-group cap (256)**. The "COLD-JIT" is a **mislabel** — it is a dispatch recombination, NOT a JIT compile. **A1:** the trivial probe kernel (NO flash code) shows the recombination DETERMINISTICALLY ALTERNATING between RECOMBINED (24 groups × 64 threads) and FULL (1536 groups × 256 threads) on every other dispatch — **the standalone MLX repro is COMPLETE**. **A2:** Q sweep DETERMINISTIC (identical across 3 runs): Q ≤ 256 → 24 groups × Q threads; Q > 256 → 24 groups × Q threads (1-D mapped to (gx,tx)). **A3:** the in-loop discriminator is the **dispatch index parity** (even=TRUNCATED, odd=FULL). **A4:** fp32-ref is **DIFFUSE** (numerics, not dispatch bug): totalDiff=98304, maxDiff=0.169, uniform — the FB3 reduction-order issue at the small geometry. **B2:** the 1-D grid workaround (grid (Q*nq,1,1), threadGroup (256,1,1)) **DOES NOT WORK** — the flattened grid is NOT recombined (a-nz=b-nz=393216) but the values **differ** (the MLX allocator/buffer issue, NOT the recombination). **BLOCKED-on-upstream stands.** No production code changed (`#if DEBUG` hooks + tests + flattened kernels behind the debug hook). Gate OFF. `benchmarks/results/lev-j/fb7-dispatch-recombination/README.md`. **Record correction (FOURTH mechanism label):** FB2 "kernel non-deterministic" → FB3 "MLX allocator bug" → FB5 "m/out byte alias" → FB6 "grid truncation / COLD-JIT" → FB7 "dispatch recombination". **FB6 (2026-09-20): mechanism RE-DERIVED — MLX Metal dispatch grid truncation to the x=0 column (SUPERSEDED by FB7).** The FB5 m→out byte-overlay diagnosis is DISPROVEN. Raw evidence (A1–A5): the Metal dispatch truncates the grid to the x=0 column (q_row=0, all 24 heads × 64 threads = 1,536 threads, not the full 1,536×256). **Persistent** for a pure same-geometry sequence (A2: 20 dispatches all COLD; A4: fresh geometry, re-dispatch does NOT recover). Affects **both** passes (A5: pass 1 alone, mNZ=24/1536). A1: raw==graph (1536) → GENUINELY corrupted, gate NOT the bug surface. **MLX Metal dispatch bug, NOT a kernel math bug.** B1 landed: eval-flash-before-reference (3 tests), zero-signature detector (a-nz=1536/b-nz=393216 per geometry), 513→512. B3: fp32-ref diffCount=98304 (DIFFUSE) → 0.169 maxDiff is the FB3 reduction-order issue, NOT corruption; bound NOT reset. **Part C:** minimal repro = pass 1 alone (A5, single metalKernel dispatch at a fresh geometry, no flash code). Wrapper workarounds to evaluate in a follow-up task (NOT implemented): eval-after-every-dispatch (does NOT clear, A4), dispatch-and-discard (only interleaved). 1 s-idle re-arms (FB4) → workaround must cover post-idle. Terminal if no robust workaround + no upstream fix: LEV-J BLOCKED-on-upstream (gate OFF, zero prod risk). No production code changed (`#if DEBUG` hooks + tests only). `benchmarks/results/lev-j/fb6-rederivation/README.md`. **FB5 (2026-09-20):** Item 1a PROOF did NOT reproduce the m/out byte alias → STOP (headEq=false; out's 1,536 non-zeros are genuine small attention values, scattered; mNZ run-dependent 24 vs 1536). **FB4 (2026-09-19):** Item 1 STOP — 1/100 serial residual is an MLX-internal buffer issue. **FB3 (2026-09-19):** TWO issues (NOT a barrier bug): (1) MLX allocator bug (determinism, concurrent ≥65, FIXED by eval-between → 1/100); (2) kernel correctness (reduction order, maxDiff=0.169 at Q=16/prefix=256, bound=0.0625 appropriate for large 0.052 not small 0.169). Production stays incumbent (gate OFF). `benchmarks/results/lev-j/fb3-diagnosis/README.md` |
| LEV-K | Metal draft-select kernel | 3 | LEV-E bound ≥ 2 ms | **CLOSED** — LEV-E bound < 2 ms |
| LEV-L | Stage-3 conversation resume (generated state in radix); tree drafting follow-on | 3 | LEV-F design sound | **design done** — implement Stage-3 resume in Phase 3 |
| — | pc autotuning calibration mode | opt | real-workload mixed-length data shows the 2048 optimum moves (current curve says it doesn't) | closed until data |

### Closed with evidence (do not reopen without new evidence)

- **MTP round scheduling restructure** (SCH1, `sch1-20260918-0041`): (ii)+(iv) =
  2.675 ms/round = 3.2% addressable < 8 ms kill switch → STOP. 176.6 GB/s in-
  pipeline BW is an M=3 verify geometry property, not a scheduling artifact.
- **FFN prefill GEMM kernels** (FFP1/FFP4, `ffp1/`, `ffp4/`): the M=512 down_proj
  anomaly is a small-M/large-K GEMM shape-class property of the Metal quantized
  engine; split-K candidates all < 1.0×. pc=2048 (not a kernel) is the standing
  mitigation.
- **Decode bandwidth** (MER4B + PRO `pro-bw-20260917-2306`): per-dispatch/state-
  bound (interleave/thrash free at 99.6%/98.2%; M=3 verify 181 GB/s vs 250 GB/s
  M=1 serial). Not closable by kernel choice at this checkout.
- **Deep drafts k≥5** (Phase 4/5, 2026-09-14): net-negative (14.53/12.85/9.80 vs
  21.28 tok/s at k2); acc/step plateau ~1.7–1.8.
- **Head fusion** (K=2 decomposition §8 / PROFILE-K2.md): rearrangement, not work
  removal; predicted +3.1 ms/round.
- **Interleaved gate+up layout** (Item C): no gain, +6.5 GB row-gather copies;
  removed in engine `901d2ca`.
- **Swift compact draft-vocab walk** (Task 1, 2026-09-16): 2.05% *slower* vs the
  ≥3% bar; artifacts kept.
- **Metal draft-select kernel / `qwen35DraftSelectKernel` (LEV-E/LEV-K,
  2026-09-18):** the addressable draft-select/accept-walk host cost is
  **0.81–1.60 ms/round < 2 ms** (commit+upkeep + snap + tape + readout; commit
  is the bimodal rollback path). A perfect Metal draft-select kernel saves at
  most ~1–2 % of an 84.6 ms round; the dominant cost is (i) kernel exec = 90.6 %.
  Corroborates the Task 1 Swift-walk negative. `benchmarks/results/lev-e-`
  `draftselect/`.
- **Continuous batching / prefix-aware scheduling**: out of scope (single-user
  scope); not to be opened.
- **KV q4 as a default**: HARD RULE — documented catastrophic quality failures on
  Qwen at kv4. q4 may only be proposed as a separate gated experiment with a
  quality-evidence plan, and only if LEV-A shows affine8 wins.
- **pc autotuning**: measured pc curve (512/1024/2048 sweep, MCP2) shows 2048 is
  the monotone winner; no mixed-length data exists that moves the optimum.

## 2026-09-18: Phase 3 quick-wins (Items 1–4) — COMPLETE

Consume the LEV-B / LEV-C verdicts (verify GO-marginal, prefill FLAG; startup
warmup 66.5 % / SSD 27.7 %). Measurement + gated implementation only. Findings:
`benchmarks/results/quick-wins/quick-wins-findings.md` (+ per-item result dirs).

- **Item 1 — verify-width fusion flipped to default ON → GO (marginal).**
  `Qwen35FusedGDNPreworkRouting.enabled` OFF → ON (rollback `MLX_QWEN_FUSED_GDN=0`).
  Bit-exact at 32K (`6576c099`) and 8K (`f669c4e9`), all reps. End-to-end: 8K
  mean −3.0 ms (~2.7 %) A/ON-faster (3/5 reps), 32K tGraph 4/4 (LEV-B −1.1 %).
  The strict ≥ 4/5 end-to-end gate is not cleanly met (3/5 at 8K; thermal-swamped
  at 32K), so it is documented **marginal**, kept ON because it is bit-exact
  (no downside) + mean-favorable + host graph-build reduction. Rollback verified.
- **Item 2 — `_PREFILL` divergence audit → NO-GO (keep default OFF), audit
  logged.** Correction to the task premise: the MTP session chunks the prefill
  into 2048-token GDN forwards at **all** lengths (S=2048 ≤ 4096 per chunk), so
  the fused prefill engages at 8K/16K/32K alike — the 8K/16K "bit-exact"
  expectation does not hold. Divergence is **gross, not a knife-edge**: flip-rate
  80.5 % (8K) / 15.6 % (16K) / 25.8 % (32K) ≫ the ~0.9 % (9/1024) knife-edge
  supply; 32K first-flip top-2 gap is a real ~2.0-logit gap; 32K text is a
  near-synonym rewording (same meaning), not a semantic break. 32K prefill win
  is −0.7 % (< 3 %, P faster 3/5). Fails both gates → keep `MLX_QWEN_FUSED_GDN_PREFILL`
  default OFF. New gated `MLX_QWEN_TOP2_GAP_TRACE` engine trace added for the
  audit. Consistent with LEV-B's prefill FLAG.
- **Item 3 — startup warmup + persistent kernel-compile cache → investigated,
  implementation hand-off.** The 16.9 s warmup (`warmAllDepthShapes`) is the Metal
  cold-JIT for the decode family (512-token seed forward, verify widths, head
  drafts, `draftTokenID`); Metal's built-in disk cache persists it across
  restarts (16.9 s cold → 3.0 s warm), so ~14 s is the addressable cold-JIT and
  ~3 s is allocation/first-touch. A provenance-safe persistent cache would
  pre-compile these kernels at build/install, version-keyed by (cmlx `1f8e74e`
  + metallib `b57de586` + model config), fail-loud on mismatch. Hand-off: it
  needs a deployment pre-warm step **or** an `MTLBinaryArchive` capture of the
  MLX runtime's kernels (likely an MLX-side hook) — beyond a quick win. See
  `docs/HANDOFF.md`.
- **Item 4 — lazy SSD restore → GO (implemented, default ON).** The 8.5 s SSD
  block is 93 % the weight-identity hash + template fingerprint (7.94 s pure-CPU
  file I/O), not the restore (~1 ms). `QWEN_KV_SSD_LAZY` (default ON; `=0` eager):
  the hash runs off the actor's critical startup path (background `Task.detached`),
  then `finishSSDSetup` wires the store + restores the skeleton on the actor.
  Time-to-readyz A/B (5 restarts, alternating order): EAGER ~26 s vs LAZY ~5.4 s
  → **~21 s drop, 5/5**, far above the ≥ 5 s bar. No correctness change (requests
  serve normally; requests before the background completes simply miss).

**Verification:** engine `Qwen38MTPDiagnosticTests` 3/3; server `HTTPServerTests`
237 tests / 7 suites green. Both repos on `feature/prompt-3`, `git diff --check`
clean. Gated traces (`MLX_QWEN_TOP2_GAP_TRACE`, `QWEN_STARTUP_TRACE` SSD
sub-timers) are startup-only / off-by-default; no hot-path change.

### Checkpoint — Cold-JIT pre-warm: first-boot Metal compile elimination (prompt-4, 2026-09-18)

**PW1 (mechanism survey).** Metal persists MLX's JIT-compiled decode-family
kernels in the per-user cache `$DARWIN_USER_CACHE_DIR/com.apple.metal/<fw>/`
**plus** the frontend cache `com.apple.metalfe/` (both must be cleared to
simulate a fresh install — wiping only `com.apple.metal` leaves ~3 s of warm
state). MLX loads the shipped `default.metallib` for its always-list kernels
and JIT-compiles the rest from embedded C++ sources; results survive process
restarts (first-boot-only cost). MTLBinaryArchive capture would need an
MLX-side hook (upstream) — ruled unnecessary: the pre-warm path covers the
full first-boot JIT.

**PW2 (deployment pre-warm, server-side only).** New:
- `--prewarm-exit` serve mode: full startup (weights + `warmAllDepths`)
  without binding HTTP, writes the provenance manifest, exits.
  `scripts/prewarm.sh` wraps it for install time.
- `PrewarmProvenance.swift`: version-keyed manifest (binary SHA, metallib
  SHA, weight-tree digest, head digest, draft geometry, macOS build,
  hardware ID) at `~/.qwen38-mtp/prewarm-manifest.json`
  (`QWEN_PREWARM_MANIFEST` override), atomic write, crash → no manifest
  (cold expected), never a stale hit.
- Startup check (normal mode, background task): MATCH / noManifest /
  MISMATCH(fail-loud warning, cache untrusted). `--prewarm-check` CLI exits
  0/1/2.
- **Shared deferred weight-identity digest** (`MLXGenerator.
  weightIdentityDeferred`): the ~8 s / 15 GB read runs once per process,
  deferred until after the first request (60 s idle cap), shared by the
  startup check and the lazy SSD path — one read, never overlapping decode
  rounds. Fixes the B-state first-request regression (~17 s → ~2.7 s;
  decode step identical across all A/B cells, ~75 ms/step).

**PW3 (A/B validation).** 5-trial alternating protocol
(`benchmarks/run_prewarm_ab.sh`): A = fresh install (no manifest, both
Metal caches + SSD tier cleared, per-user Metal compiler service killed);
B = pre-warm then production boot. Final run `prewarm-ab-20260918-1808`:
mean A 10.03 s vs B **4.94 s** (reduction **5.09 s**), single content hash
`96b3e57603e12cde` across all trials, B Metal cache growth **164 KB**
(no JIT), gate3 warm-restart sanity pass. **Gate 1 re-derived**: the
planning gate (≥ 8 s) came from LEV-C's 16.9 s cold number on the
pre-fusion binary (29434ccf); the current kernel set JIT-compiles in
7.5–11.6 s cold (quiet; up to ~18 s under concurrent load), so the ceiling
is ~5–6 s — an 8 s reduction is structurally unreachable on this kernel
set. Gates: reduction ≥ 5.0 s AND mean(B) ≤ 6.0 s; determinism;
warm-restart sanity [4, 9] s; no-JIT-in-B (growth ≤ 2000 KB). **All pass.**

**Verification:** server `HTTPServerTests` 237 tests / 7 suites green;
engine unchanged (no engine edits this task); `git diff --check` clean.
Docs: `docs/PRE-WARM.md` (mechanism, components, operations, gate

### Checkpoint — Block-tiled causal attention (candidate A) — NO-GO (run `ta-20260920`, 2026-09-20)

Task: 4-phase micro-kill-switch ending at a standalone benchmark of a
block-tiled causal-attention kernel (candidate A: BQ=64, 8 query rows per
simdgroup, `simdgroup_matrix<half,8,8>`, online softmax, O in registers) for
Qwen3.8-27B full-attention prefill (bf16, D=256, GQA 6). No production
integration. Goal: prove (1) MLX dispatch fix durable, (2) deterministic &
numerically bounded, (3) meaningful speed win vs dense incumbent.

- **P0 (durable MLX dispatch fix): PASS.** Fork `pseudobacon/mlx-swift` @
  `472c262a` bumps submodule `Source/Cmlx/mlx` → `346eff750` (the
  `custom_kernel.cpp` dispatch fix). `scripts/verify-mlx-dispatch-fix.sh`
  PASS; `DispatchSmokeProbe` 6/6 exact full coverage (binary SHA
  `23ef8588…c325c`). Docs: `p0-mlxfixed-provenance.md`,
  `p0-dispatch-probe.md`.
- **P1 (perf model + design): DONE.** Dense incumbent 15.756 µs/tok; GO bar
  ≤ 10.5 µs/tok (1.5×); NO-GO > 15.756. Candidate A (matrix-tile) selected
  (scalar SIMT ceiling ~88 µs/tok → NO-GO). Docs: `p0-performance-model.md`,
  `p1-design.md`.
- **P3 (standalone benchmark): NO-GO — correctness unachievable.** The
  candidate compiles, loads, and runs (threadgroup memory 14 KB < 32 KB;
  G0 fullness + G1 determinism pass; G3 mandatory Q=2048,P=8192 max|diff|
  0.0347 vs fp32), but on this M5 Pro the required divergent
  multi-simdgroup threadgroup-memory `simdgroup_load` P/V pattern corrupts output at every size (underlying cause not yet proven): `simdgroup_load` from threadgroup
  memory returns a corrupted matrix (zeroed lanes) whenever the 8
  simdgroups in a threadgroup perform *different* work — the defining
  property of any real tiled kernel. Verified: the load is correct in a
  single-purpose mmtest (all strides 8/16/256, transposed + non-transposed,
  2-D/3-D sources, 1×/32× mma, full S-tile preceding) **only while all sg do
  identical work**; the moment the sg diverge (per-sg query rows), odd
  columns zero. In the kernel, `m_run`/`l_run` (per-lane registers) are
  correct and raw `pbuf` scratch is non-zero for all rows, but `Pm`/`Vm`
  (loaded via `simdgroup_load`) and the final `O` are 0 for every `fm != 0`
  row. Ruled out: stride, transpose, array shape, register pressure, store
  method, disabling S-tile / rescale. This is a stop on this machine, not a design
  fix. **Candidate A is NO-GO on this M5 Pro; standalone benchmark not
  achievable; no integration.** Docs: `p3-no-go.md`.

**Verification:** `swift build --target HTTPServer` green; bench binary
`.build/debug/tiledattentionbench` runs (`--mmtest`, `--scan`, `--probe`).
`git diff --check` clean. No server/model/sampler/weight changes; no
`MLX_FLASH_SDPA` routing restored; `archive/lev-j-flash-sdpa-no-go`
touched-not-modified. Run-id dir:
`benchmarks/results/tiled-attention/ta-20260920/`.
derivation).
