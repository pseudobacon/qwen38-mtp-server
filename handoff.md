# qwen38-mtp-server — Active Handoff (Checkpoint 2d)

_Last updated after Item 2 (fused `W_gate+up`) benchmark._

## Objective & Acceptance Criteria
Complete **Checkpoint 2d — Feature Parity Restorations**: restore original server performance by re-integrating the packed linear weight projections (`W_qkv`, `W_gate+up`) into the MTP execution pipeline (M ∈ 1..9), in three items:
1. **Item 1 — fused `W_qkv`** (QKV projection): implement, test both repos, release benchmark, progress row, handoff update. **DONE.**
2. **Item 2 — fused `W_gate+up`** (SwiGLU projection): same protocol. **DONE.**
3. **Item 3 — combined QKV + SwiGLU verification + merge**: full tests, merge `feature/prompt-ckpt2d` → `main` in both repos, delete feature branches. **PENDING.**

Per-item protocol: (1) implementation + `swift test` both repos; (2) release benchmark (build release, `QWEN_MTP_STEP_TRACE=1` server on port 18099, 1,024-token greedy essay request, extract `avgStepMs`, tok/s, wall-clock); (3) append row to `progress.md`; (4) update this file. Context guard: if context ≈ 80% or > 10 execution turns mid-item, stage + commit `wip: context handoff snapshot` and prompt user to fork/new session.

## Branch & Commit Status
* **Both repos:** `feature/prompt-ckpt2d` (cut from `main`).
* **Engine `mlx-swift-lm`:** HEAD `dfacc19` (`feat: add MSL M=1 wide QMV dispatch support`); uncommitted Items 1+2 changes: `Libraries/MLXLMCommon/FusedQuantizedLinear.swift`, `Libraries/MLXLLM/Models/Qwen35.swift`, `Libraries/MLXLLM/Models/Qwen35+FastPath.swift`, `Libraries/MLXLLM/Models/Qwen3Next.swift`, new `Tests/MLXLMTests/Qwen35FusedQKVProjectionTests.swift` + `Tests/MLXLMTests/Qwen35FusedSwiGLUProjectionTests.swift`; untracked pre-existing `docs/`. `git diff --check`: clean (4 files, +222/−10).
* **Server `qwen38-mtp-server`:** HEAD `cbffd7d` (`docs: update progress log for Checkpoint 2c benchmark`); uncommitted: `progress.md` (Item 1 + Item 2 rows), `handoff.md`; untracked pre-existing `docs/`. No server source changes (engine-only).
* Multi-repo policy: commit engine fork **first**, then server; merge both to `main` and delete feature branches only after Item 3.

## Test Status (Items 1+2)
* `swift build --target MLXLLM`: clean.
* `swift test --filter Qwen38MTPDiagnosticTests`: **PASS** — 93.46% (1072/1147) greedy T=0 acceptance; logit max-divergence `postNorm: true => 16.25`, `false => 14.0` — identical to 2c baseline (both fusions bit-exact).
* `swift test --filter Qwen35FusedQKVProjectionTests`: **8/8 PASS**.
* `swift test --filter Qwen35FusedSwiGLUProjectionTests`: **8/8 PASS**.
* `swift test --filter HTTPServerTests` (server): **121/121 PASS**.

## Latest Step Latency
* **Item 2 (fused `W_qkv` + fused `W_gate+up`), release build, 1,024-token request:** `avgStepMs = 123.69` (380 rounds), `tEvalAvg = 111.64`, `tGraphBuildAvg = 11.04`, `tCacheStateAvg = 0.87`, `decodeSeconds = 47.04`, TTLT **21.77 tok/s**, `acceptedPerStep = 1.6974` (645/1008).
* **Item 1 (fused `W_qkv` only):** `avgStepMs = 122.25`, `decodeSeconds = 46.48`, **22.03 tok/s**, same 645/1008 acceptance.
* **2c baseline:** `avgStepMs = 116.06`, `decodeSeconds = 49.47`, **20.7 tok/s**.
* Deltas: QKV fusion **+6.19 ms/step**; gate+up fusion adds **+1.44 ms/step** (fused `U32 [34816,640]` over 64 layers). Combined +7.63 ms/step vs 2c, but TTLT stays above the 2c baseline on this run's acceptance (21.77 vs 20.7 tok/s).
* Item 1 and Item 2 runs produced identical accepted counts (645/1008, 380 rounds) — greedy token stream unchanged (bit-exact).

## Key Files
* `mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedLinear.swift` — `FusedQuantizedLinearProjection` / `FusedQuantizedLinearProjectionCache`; env knobs `qwen35FusedQKVEnabled` (`MLX_QWEN_FUSED_QKV`, default ON), `qwen35FusedSwiGLUEnabled` (`MLX_QWEN_FUSED_SWIGLU`, default ON).
* `mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` — `Qwen35Attention.fusedQKVProjection` cache + `update`/`updateModule` invalidation; `Qwen35TextModel.prepare()` wires backbone + MTP-head layers for both fusions (head BF16 ⇒ both ineligible ⇒ eager fallback).
* `mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3Next.swift` — `Qwen3NextMLP.fusedSwiGLUProjection` cache + `update`/`updateModule` invalidation; `callAsFunction` via `swiGLUGateUpProjections`.
* `mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35+FastPath.swift` — `extension Qwen35Attention` (`hasFusedQKVProjection`, `prepareFusedQKVProjection`, `qkvProjections`) and `extension Qwen3NextMLP` (`hasFusedSwiGLUProjection`, `prepareFusedSwiGLUProjection`, `swiGLUGateUpProjections`); `forwardFastPath` + `projectPreRope` call the QKV helper.
* `mlx-swift-lm/Tests/MLXLMTests/Qwen35FusedQKVProjectionTests.swift` + `Qwen35FusedSwiGLUProjectionTests.swift` — 8 bit-exactness/lifecycle tests each.
* `qwen38-mtp-server/progress.md` — checkpoint log (Item 1 + Item 2 rows appended).
* Benchmark harness: `/tmp/run_bench_release.sh <port> <tag>` (release binary, `QWEN_MTP_STEP_TRACE=1`, 1,024-token greedy essay request, polls `/readyz`, parses `MTP-STEP-SUMMARY` + phase averages).

## Decisions & Evidence
* Fusions reuse the existing GDN machinery: source modules become row-slice **views sharing storage** → net ~0 memory (temporary double allocation during `prepare` only: ~147 MB QKV, ~1.45 GB gate+up).
* Fusions apply to **backbone 4-bit layers only** (16 QKV + 64 gate+up). MTP head is BF16 ⇒ not fusion-eligible ⇒ eager fallback (documented in `Qwen35TextModel.prepare()` and `progress.md`).
* Bit-exactness: per-row dequant identical ⇒ diagnostic acceptance/logit divergences unchanged (93.46%, 16.25/14.0) and identical greedy streams across Item 1/Item 2 benchmark runs (645/1008, 380 rounds).
* `down_proj` stays eager (scope: only the gate+up sweep fused per prompt).

## Unresolved Risks
* Combined step latency is +7.63 ms vs 2c (per-step cost of the wider fused matmuls). Item 3 numbers must be read with that caveat; TTLT is the headline metric and is above the 2c baseline on this run's acceptance.
* `acceptedPerStep` varies run-to-run across checkpoints (1.41 / 1.66 / 1.70 for the same prompt) — use release builds and the same prompt/params for comparisons.
* Port 18099 can be left occupied by a stale server from a previous run — `pkill -f "qwen38-mtp-server serve"` before re-running the benchmark.

## Operations That Must Not Be Repeated
* Do not run the benchmark while an old server instance holds port 18099.
* Do not edit source via shell scripts (use edit tools); do not touch response/metrics code this checkpoint.
* Do not merge to `main` before Item 3's full tests pass.

## Next Step
**Item 3 — combined QKV + SwiGLU verification + merge**: (1) re-run full verification with both fusions ON: engine `Qwen38MTPDiagnosticTests` + `Qwen35FusedQKVProjectionTests` + `Qwen35FusedSwiGLUProjectionTests`, server `HTTPServerTests`; (2) one final combined release benchmark (port 18099, tag `ckpt2d-item3`, 1,024-token greedy essay request) and append the Item 3 row to `progress.md`; (3) commit engine fork first (`git add` the 4 modified engine files + 2 new test files; plain message, e.g. `feat: fused W_qkv and W_gate+up packed projections for MTP pipeline`), then server repo (`progress.md` + `handoff.md`); (4) `git checkout main && git merge --no-edit feature/prompt-ckpt2d` in both repos, delete feature branches; (5) update this file to mark Item 3 DONE.
