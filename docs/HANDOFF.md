# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a task-state checkpoint, not a complete conversation transcript.
> Verify all claims against the working tree before acting.

## Prior task — Phase 3 dual-fixture re-baseline + compiled-path ablation (COMPLETE, 2026-09-13)

Objective: first valid headline tok/s for the current codebase (the 22.28 tok/s record was the pre-QMV-fix binary on the specdec fixture), plus the combined in-session contribution of Checkpoint 1 + 2a + the compiled fast paths. Three cells, one session, thermal discipline:

| Cell | Fixture | Config | Purpose |
|---|---|---|---|
| B1 | `essay-1024.txt` | default (fusions ON, `MLX_COMPILED_DECODE` default ON) | re-baseline, comparable to A0 |
| B2 | `specdec-800.txt` | default | first-ever specdec measurement on the current binary |
| B3 | `essay-1024.txt` | `MLX_COMPILED_DECODE=0` | ablation: compiled micro-fusions + QK-RoPE fast path + compiled decode segments OFF |

Protocol as pinned: release build, port 18099, greedy (temp 0, `enable_thinking: false`, `max_tokens: 1024`, `finish_reason: length`), prompt from the pinned fixture, `QWEN_MTP_STEP_TRACE=1`, fresh server per cell, 6 reps per cell with rep 1 discarded (5 measured), interleaved B1→B2→B3 across reps, no parallel builds/tests, no mid-matrix rebuild. `pmset -g therm` logged before every rep; per-rep determinism gate enforced (essay 599/1086/426 `949b9423…`; specdec 645/1008/380 `139acb9d…`).

- **All 18 reps bit-exact** — zero gate failures. Fusion engaged in every cell (`backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head 0/0`).
- **Headlines (1024 / wall-seconds, mean over reps 2–6):** B1 essay **17.55 tok/s** (136.559 ms steps); B2 specdec **18.92 tok/s** (141.623 ms steps); B3 ablation **16.64 tok/s** (143.909 ms steps).
- **Ablation verdict (B3 − B1, in-session):** **+7.350 ms/step (+5.38%)**, **−0.914 tok/s (−5.21%)** — the combined controlled contribution of everything gated by `MLX_COMPILED_DECODE` (Checkpoint 1 micro-fusions + Checkpoint 2a QK-RoPE kernel + fused residual+RMSNorm). Packed fusions and routed QMV are ungated and active in both cells — not part of the delta.
- **B2 vs 22.28:** acceptance identical (same stream hash); the entire gap is step time (+17.2%) — session thermal state + M=1 routed-QMV dispatch (≈5%/projection at M=1 per `qmvbench`), consistent with the cross-session non-comparability decision.
- **Stale-binary trap resolved:** the 18:38 release binary predated engine `901d2ca` and SwiftPM had not invalidated the two changed engine modules. Forced recompile of `FusedQuantizedLinear.swift` + `Qwen35+FastPath.swift` + relink produced a **different** binary (SHA-256 `88e27643…` vs `03231f16…`); that is the binary the matrix ran on. Lesson: force-recompile changed engine modules (or compare relink hashes) before benchmarking.
- Artifacts: `benchmarks/results/rebaseline-essay.jsonl`, `rebaseline-specdec.jsonl`, `ablation-compiled-off.jsonl` (6 per-rep records each, thermal line merged); driver `benchmarks/run_phase3.sh`; stats `benchmarks/phase3_report.py`; run/thermal logs `.tmp/phase3-run.log`, `.tmp/phase3-thermal.log`.

## Prior task — interleaved gate+up layout removal (COMPLETE, 2026-09-13)

Objective: remove the rejected interleaved gate+up layout (Item C of the fusion diagnosis) from the engine, then re-verify with a dual-fixture re-baseline gate. **All acceptance criteria met.**

- Removed from `../mlx-swift-lm`: the `MLX_QWEN_SWIGLU_LAYOUT` env gate, `FusedQuantizedLinearLayout` and the row-gather materialized fuse path in `FusedQuantizedLinear.swift`; the stored layout param and the `.interleaved` split branch in `Qwen35+FastPath.swift` (back to global slice-at-half); the `wide_interleaved_routed` qmvbench condition; the three interleaved-specific tests. Loud load-time fallback logging for genuine fallback cases (BF16 MTP head) retained. Default path was always global → behavior-neutral.
- Removed from `benchmarks/run_matrix.sh`: the `itemC` phase and `INT` env spec (the knob no longer exists; the phase would compare two identical cells). `verify` now covers **both** pinned fixtures. Historical data unchanged: `benchmarks/FUSION_REPORT.md`, `benchmarks/results/itemC.jsonl`.
- Engine commit `901d2ca` (`refactor: remove rejected interleaved gate+up layout`), fast-forward merged to engine `main`; server docs commit `7d04ecf`; feature branch `feature/prompt-ckpt3a` cut and deleted in both repos.
- Verification on the rebuilt binary: SwiGLU 8/8, QKV 8/8, GDN 14/14, `Qwen38MTPDiagnosticTests` 1/1 (acceptance 93.46%, logit divergence 16.25/14.0 — unchanged); `run_matrix.sh verify` reproduces essay-1024 599/1086/426 (hash `949b9423…`) and specdec-800 645/1008/380 (hash `139acb9d…`) — the **first specdec run on the post-QMV-fix binary**, correctness signal green.
- Server suite: `HTTPServerTests` **107 passed / 0 failed** — the previously recorded 108 was a grep artifact (the `testRecoveryPolicyBypassedWhenDisabled` name contains "passed"); confirmed by strict counts and a source cross-check.

## Prior task — fusion diagnosis (COMPLETE, 2026-09-13)

Objective was the rev-2 benchmark harness diagnosing Qwen 3.8 gate+up / QKV fusion (Items A–C) on the `mlx-swift-lm` fork + `qwen38-mtp-server`, plus the final report. **All acceptance criteria met**; full tables in `benchmarks/FUSION_REPORT.md`, task log in `progress.md`.

### §0 — Prompt-fixture provenance (resolved)

- Pinned `benchmarks/prompts/essay-1024.txt` (SHA-256 `7ed683f87be0835c751505e2ee7dfc18fd922b93bcc32fad05d86c158cfb040e`, 38 prompt tokens) reproduces **599/1086/426**, acceptedPerStep 1.4061, stream hash `949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe` — this is the 2a/2c prompt, NOT the recorded 645/1008/380.
- The recorded **645/1008/380** (hash `139acb9d…`) belongs to `benchmarks/prompts/specdec-800.txt` (62 prompt tokens), exactly reproduced under the rolled-back all-fusion-off build (`benchmarks/results/resolve.jsonl`).
- Consequence: the pre-session "2c → Item 1 +6.19 ms QKV regression" was cross-prompt confounded and is retracted; in-session deltas are the valid measurements.

### Fusion engagement (resolved)

The single load-time fallback log line is the **BF16 MTP head layer only** (`mtp-head/model.safetensors` carries BF16 projections — ineligible by design). The load-time summary in `Qwen35TextModel.prepare()` prints `backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 0 qkv 0` — gate+up fusion engaged in all 64 backbone layers, QKV in all 16 full-attention layers, GDN input-projection fusion in all 48 GDN layers.

### Item A — 2×2 same-session fusion matrix (clean run)

- 24/24 cells bit-exact (599/1086/426, hash `949b9423…`). Reps 2–6 means: A0 143.042, A1 143.304 (+0.262 ms), A2 142.967 (−0.076 ms), A3 145.429 (+2.387 ms); per-rep Δ swings −21.8…+15.2 ms → **both fusions latency-neutral in-session**.
- Invalidations, with evidence: the original 02:41 binary predated the QMV dispatch fix + fusion merge → ran fully eager (proven: correct greedy hash under a dispatch that would have diverged); the first re-run was discarded for timing contamination (parallel test runs; spikes 239.6/196.1/178.9 ms). Contaminated logs kept as evidence in `.tmp/itemA-rerun.log`.
- Authoritative data: `benchmarks/results/itemA.jsonl`.

### Item B — standalone QMV microbenchmark (`qmvbench`)

Complete; all 7 conditions bit-exact to incumbent `quantizedMM` at M=1 and M=4. M=1: routed ≈5% slower than incumbent per projection (~385 vs ~366 µs); M=4: routed ≈15–20% faster; interleaved ≈ global at both M. Caveat: per-call device sync measures serialized latency, not pipeline throughput (A2's null end-to-end result demonstrates this).

### Item C — interleaved gate+up layout (implemented, rejected, removed)

12/12 cells bit-exact. Cglobal 150.578 ms, Cint 152.187 ms, Δ +1.609 ms (+1.1%), inside the run's thermal noise band. Interleaved does NOT recover the +3–9% decode range and regresses nothing — but materializes ~100.8 MB per gate/up layer pair (~6.5 GB) of row-gather copies → **rejected; the gate and layout code were removed from the engine in `901d2ca` (2026-09-13)**.

### Tests (all green)

Engine: `Qwen35FusedSwiGLUProjectionTests` 8/8, `Qwen35FusedQKVProjectionTests` 8/8, `Qwen35FusedGDNProjectionTests` 14/14, `Qwen38MTPDiagnosticTests` 1/1 (acceptance 93.46%, logit divergence 16.25/14.0). Server: `HTTPServerTests` 107 passed / 0 failed (the previously recorded 108 was a grep artifact — see the prior-task section above). `git diff --check` clean in both repos.

## Goal (next task)

- Objective: **Item D — route the verify pass through the candidate QMV kernel.** The `ndim == 2` guard currently sends verify (3-D batched `x`, the dominant `tEvalAvg` share) to incumbent `quantizedMM`; a free reshape to `[M, K]` would dispatch it through the routed kernel at M ∈ 2…9, where the microbench says it is ~15–20% faster. Gate on an end-to-end A/B win, not the microbench alone (the A2 null result is the cautionary precedent).
- Acceptance criteria: bit-exact greedy streams on both fixtures (essay 599/1086/426 `949b9423…`; specdec 645/1008/380 `139acb9d…`); in-session A/B (default vs Item D ON) on at least one fixture, 5 measured reps per cell, via the Phase 3 harness pattern; keep only on an end-to-end win.
- Constraints / non-goals: same standing rules — no parallel builds/tests during timing cells; no `print()` in hot paths (load-time prints allowed); engine commits before server; `git merge --no-edit`; no python/sed source edits.

## Repository checkpoint

- Last verified 2026-09-13 20:23 BST — fresh checkpoint output in the Fresh checkpoint section below.
- Branches: server `main` at `4fb92ca` (Phase 3 merged fast-forward; `feature/prompt-phase3` deleted); engine `../mlx-swift-lm` `main` at `901d2ca` (unchanged in this task).
- Worktrees: clean in both repos; the template-only `mlx-swift-lm/docs/` was deleted during docs housekeeping.
- Environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/`; venv `/tmp/benchvenv` (HF tokenizer for stream hashes — recreate if missing, `/tmp` is wiped on reboot).
- Required services: none (benchmark server is started and pkill'd per cell by `benchmarks/run_cell.sh`).
- Workspace: `qwen38-mlx-server/` is a **non-repo DSH workspace** containing symlinks to both repos. Run all git and benchmark operations inside the real repos (`/Users/cwong/ai/qwen38-mtp-server`, `/Users/cwong/ai/mlx-swift-lm`); never `git init` or commit inside the workspace. Keep both symlinks at the same level (the engine is referenced as `../mlx-swift-lm`).

## Docs housekeeping (COMPLETE)

- Delete `mlx-swift-lm/docs/` entirely — done (untracked template only, zero history cost, keeps the fork diff minimal).
- Authoritative docs live in the server repo, reached from the workspace via symlink: `docs/HANDOFF.md` (this file, rolling task checkpoint), `docs/README.md` (orientation: layout, build/test, env knobs, agent conventions), `progress.md` (status log), `benchmarks/FUSION_REPORT.md` (final report). This file carries compact completed-state detail; full tables stay in FUSION_REPORT.md / progress.md — never two live copies of the same number.
- Fix stale test counts in `progress.md` — done: `HTTPServerTests` 107/107 (the previously recorded 108 was a grep artifact; 121 was stale), SwiGLU 8/8, GDN 14/14 (already present).

## Current state

- Completed: fusion diagnosis (see above); interleaved layout removal (engine `901d2ca`); dual-fixture §0 verify on the post-removal binary; docs housekeeping; **Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13)** — first valid headline tok/s measured on current main (B1 essay 17.55, B2 specdec 18.92, B3 ablation 16.64 tok/s; ablation delta +7.35 ms/step / −5.21% for the `MLX_COMPILED_DECODE`-gated fast paths; all 18 reps bit-exact; per-rep thermal logged).
- In progress: none.
- Not started: Item D (verify-pass QMV routing); `qmvbench` throughput mode (N calls per sync); thermal cooldowns between run blocks.
- Current hypothesis / diagnosis: the B2 (18.92 tok/s) vs pre-fix 22.28 gap is purely step-time (+17.2%, acceptance identical) — the unseparated sum of session thermal state and the M=1 routed-QMV dispatch (≈5%/projection at M=1 per `qmvbench`). "Current main may land below 22.28" is confirmed in-session; cross-session comparability remains open (standing decision).

## Important files

| Path | Why it matters | Current state |
|---|---|---|
| `benchmarks/FUSION_REPORT.md` | Final fusion-diagnosis report | Final, committed |
| `benchmarks/run_matrix.sh` | Matrix runner: phases `verify\|resolve\|itemA`; `verify` covers both pinned fixtures (itemC phase removed with the layout) | Committed (`7d04ecf`) |
| `benchmarks/run_cell.sh` | Single-cell harness (pkill, readyz poll, one fixture request, JSON line) | Committed |
| `benchmarks/prompts/essay-1024.txt` | Pinned fixture, SHA-256 `7ed683f8…`, stream 599/1086/426 | Committed |
| `benchmarks/prompts/specdec-800.txt` | Pinned fixture, stream 645/1008/380, hash `139acb9d…` | Committed |
| `benchmarks/results/itemA.jsonl` / `itemC.jsonl` / `resolve.jsonl` | Authoritative runs + provenance resolution | Committed |
| `benchmarks/results/verify.jsonl` | Dual-fixture §0 verify on the post-removal binary (both hashes reproduced) | Committed (`7d04ecf`) |
| `benchmarks/results/rebaseline-essay.jsonl` / `rebaseline-specdec.jsonl` / `ablation-compiled-off.jsonl` | Phase 3 per-rep records (6 each, thermal line merged, all bit-exact) | Committed (Phase 3) |
| `benchmarks/run_phase3.sh` | Phase 3 driver: 3 cells × 6 reps interleaved, per-rep `pmset -g therm` snapshot + determinism gate | Committed (Phase 3) |
| `benchmarks/phase3_report.py` | Phase 3 post-processing: thermal merge into JSONL + mean/min/max over measured reps 2–6 | Committed (Phase 3) |
| `progress.md` | Task log, authoritative tables, open items | Committed |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | Load-time fusion engagement summary print | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35Kernels.swift` | QMV dispatch grid fix + E120 lane-decode fixes | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35+FastPath.swift` | Global slice-at-half SwiGLU split (interleaved split branch removed) | Committed (`901d2ca`) |
| `../mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedLinear.swift` | Global-layout fusion + loud load-time fallback diagnostics (interleaved layout removed) | Committed (`901d2ca`) |
| `../mlx-swift-lm/Libraries/QmvBench/main.swift` | Item B microbenchmark target (interleaved condition removed) | Committed (`901d2ca`) |

## Decisions (standing)

| Decision | Reason | Rejected alternative |
|---|---|---|
| Keep both packed fusions default ON | Bit-exact, latency-neutral, net-0 memory, rollback knobs | Rolling back |
| Interleaved gate+up layout rejected and removed | No gain at micro or end-to-end level; +6.5 GB row-gather copies; removed in engine `901d2ca` (2026-09-13) | Keeping the env knob active |
| Cross-session absolute latencies invalid | Thermal/system state differs (116–124 vs 135–165 ms bands observed) | Comparing absolutes across sessions |
| Engagement proven via load-time summary | RSS not reliable on Metal shared heaps | RSS probing |
| Discard contaminated runs, keep as evidence | Only uncontaminated in-session Δs are valid | Outlier-trimming or reuse |
| Engine commits before server | AGENTS.md commit-order dependency | Server-first commit |
| Docs live in `qwen38-mtp-server`; workspace is symlink-only | `qwen38-mlx-server` is a non-repo DSH workspace — docs must version with the work; fork stays vendor-shaped | Moving docs into the workspace |

## Commands and evidence

```text
$ swift build --target MLXLLM                                  # clean (6.4 s)
$ swift build --configuration release --product qwen38-mtp-server   # clean (70 s)
$ swift test --filter HTTPServerTests                          # 107 passed, 0 failed
$ cd ../mlx-swift-lm && swift test --filter Qwen35FusedSwiGLUProjectionTests   # 8/8
$ swift test --filter Qwen35FusedQKVProjectionTests             # 8/8
$ swift test --filter Qwen35FusedGDNProjectionTests             # 14/14
$ swift test --filter Qwen38MTPDiagnosticTests                 # 1/1 (acceptance 93.46%, logit divergence 16.25/14.0)
$ bash benchmarks/run_matrix.sh verify                         # essay 599/1086/426 hash 949b9423...; specdec 645/1008/380 hash 139acb9d... (both reproduced)
$ bash benchmarks/run_phase3.sh                                # Phase 3: 18 reps, all determinism gates PASS
$ /tmp/benchvenv/bin/python benchmarks/phase3_report.py <cell.jsonl> <B1|B2|B3> ...   # thermal merge + per-cell mean/min/max
$ ./scripts/agent-checkpoint.sh                                # fresh checkpoint for this file (run per repo)
$ git log -1 --oneline                                         # engine 901d2ca, server 4fb92ca
```

Useful artifacts: `benchmarks/results/*.jsonl` (Phase 3: `rebaseline-essay.jsonl`, `rebaseline-specdec.jsonl`, `ablation-compiled-off.jsonl`); `.tmp/phase3-run.log`, `.tmp/phase3-thermal.log` (per-rep thermal + gate results); `.tmp/itemA-clean.log`, `.tmp/itemA-rerun.log` (contaminated, kept as evidence), `.tmp/probe-summary.log` (engagement summary line).

## Risks and blockers

- Known issue: cross-session absolute step latencies not comparable (thermal/system state); only in-session Δs are valid.
- Resolved (2026-09-13): specdec-800's first run on the current (post-QMV-fix, post-layout-removal) binary was the §0 verify cell — it reproduced 645/1008/380 with stream hash `139acb9d…`, so the correctness signal is green.
- Resolved (2026-09-13, Phase 3): stale-binary trap — the 18:38 release binary predated engine `901d2ca` and SwiftPM had not invalidated the two changed engine modules; a forced recompile + relink produced a different binary (SHA-256 `88e27643…` vs `03231f16…`), which the matrix ran on. **Before benchmarking: force-recompile changed engine modules (delete their `.o` files or compare relink hashes) so the binary provably matches engine HEAD.**
- Environment: `/tmp/benchvenv` does not survive a reboot — recreate before computing stream hashes if missing.
- Do not rerun / stateful: no parallel builds/tests during `run_matrix.sh` timing cells; do not rebuild the server binary mid-matrix; no git operations inside the workspace directory.

## Exact next step

1. Item D: relax the `ndim == 2` guard in the Qwen 3.5/3.8 verify path so the 3-D batched `x` is reshaped to `[M, K]` and dispatched through the routed QMV kernel at M ∈ 2…9 (free reshape; bit-exact tests first). Then run an in-session A/B (default vs Item D ON) on at least one fixture with the Phase 3 harness pattern (fresh server per cell, 6 reps interleaved, rep 1 warmup, per-rep thermal + determinism gate); keep only on an end-to-end win.
2. Before any timing: confirm the pinned streams on the freshly rebuilt binary (`bash benchmarks/run_matrix.sh verify`), and force-recompile changed engine modules before benchmarking (stale-binary trap — see Phase 3 section).

## Fresh checkpoint

Fresh-checkpoint procedure completed successfully (2026-09-13 20:23 BST):

```
# Repository checkpoint (server)
- Repository: /Users/cwong/ai/qwen38-mtp-server
- Branch: main
- HEAD: 4fb92ca
- Git status: clean
- Latest commit: 4fb92ca feat: phase 3 dual-fixture rebaseline + compiled-path ablation

# Repository checkpoint (engine)
- Repository: /Users/cwong/ai/mlx-swift-lm
- Branch: main
- HEAD: 901d2ca
- Git status: clean
- Latest commit: 901d2ca refactor: remove rejected interleaved gate+up layout
```

Both worktrees clean of tracked and untracked changes (`.tmp/` transient logs are gitignored); feature branch `feature/prompt-phase3` merged and deleted in the server repo; no engine changes this task. _(Regenerate via `scripts/agent-checkpoint.sh` — run inside each repo — at the start of the next session.)_