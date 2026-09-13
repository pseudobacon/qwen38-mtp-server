# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a task-state checkpoint, not a complete conversation transcript.
> Verify all claims against the working tree before acting.

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

### Item C — interleaved gate+up layout (implemented, rejected)

12/12 cells bit-exact. Cglobal 150.578 ms, Cint 152.187 ms, Δ +1.609 ms (+1.1%), inside the run's thermal noise band. Interleaved does NOT recover the +3–9% decode range and regresses nothing — but materializes ~100.8 MB per gate/up layer pair (~6.5 GB) of row-gather copies → **rejected; do not enable `MLX_QWEN_SWIGLU_LAYOUT=interleaved`**.

### Tests (all green)

Engine: `Qwen35FusedSwiGLUProjectionTests` 11/11, `Qwen35FusedQKVProjectionTests` 8/8, `Qwen35FusedGDNProjectionTests` 14/14, `Qwen38MTPDiagnosticTests` 1/1 (~151 s). Server: `HTTPServerTests` 108 passed / 0 failed. `git diff --check` clean in both repos.

## Goal (next task)

- Objective: **dual-fixture re-baseline on current main** — run the pinned-protocol benchmark on both fixtures (`essay-1024.txt`, `specdec-800.txt`) with the default fusion configuration on the current binary in one thermally-controlled session, producing the first valid headline tok/s for the current codebase (the 22.28 tok/s record is from the pre-QMV-fix binary and the specdec fixture; current main has never been benchmarked on specdec). Optionally follow with **Item D** — verify-pass reshape to 2-D `[M, K]` so the routed QMV kernel handles M ∈ 2…9 (see progress.md open items).
- Acceptance criteria: both fixtures reproduce their recorded streams exactly (essay: 599/1086/426, hash `949b9423…`; specdec: 645/1008/380, hash `139acb9d…`); 5 clean measured reps per fixture, interleaved order, no parallel builds/tests; thermal state logged per rep (`pmset -g therm`); results appended to `benchmarks/results/` and the headline-throughput table in `progress.md` updated with the measured numbers.
- Constraints / non-goals: same standing rules — no parallel builds/tests during timing cells; no `print()` in hot paths (load-time prints allowed); engine commits before server; `git merge --no-edit`; no python/sed source edits; do not enable `MLX_QWEN_SWIGLU_LAYOUT=interleaved` (rejected).

## Repository checkpoint

- Last verified 2026-09-13 (TZ Asia/Shanghai local machine clock) — fresh checkpoint output in the Fresh checkpoint section below.
- Branches: server `main` at `d071300` (feature branch merged and deleted); engine `../mlx-swift-lm` `main` at `f730e87` (fast-forward of `0514b11`; `feature/fusion-diagnosis` deleted).
- Worktrees: clean apart from untracked `.tmp/` in both repos, and the template-only `mlx-swift-lm/docs/` (to be deleted this session).
- Environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/`; venv `/tmp/benchvenv` (HF tokenizer for stream hashes — recreate if missing, `/tmp` is wiped on reboot).
- Required services: none (benchmark server is started and pkill'd per cell by `benchmarks/run_cell.sh`).
- Workspace: `qwen38-mlx-server/` is a **non-repo DSH workspace** containing symlinks to both repos. Run all git and benchmark operations inside the real repos (`/Users/cwong/ai/qwen38-mtp-server`, `/Users/cwong/ai/mlx-swift-lm`); never `git init` or commit inside the workspace. Keep both symlinks at the same level (the engine is referenced as `../mlx-swift-lm`).

## Docs housekeeping (part of this task)

- Delete `mlx-swift-lm/docs/` entirely (untracked template only — zero history cost, keeps the fork diff minimal).
- Authoritative docs live in the server repo, reached from the workspace via symlink: `docs/HANDOFF.md` (this file, rolling task checkpoint), `docs/README.md` (orientation: layout, build/test, env knobs, agent conventions), `progress.md` (status log), `benchmarks/FUSION_REPORT.md` (final report). This file carries compact completed-state detail; full tables stay in FUSION_REPORT.md / progress.md — never two live copies of the same number.
- Fix stale test counts in `progress.md`: HTTPServerTests 108 (not 121), SwiGLU 11/11 (not 8/8), add GDN 14/14.

## Current state

- Completed: prior task (see above); docs reorganization decided, pending execution.
- In progress: none.
- Not started: dual-fixture re-baseline; Item D (verify-pass QMV routing); `qmvbench` throughput mode (N calls per sync); thermal logging wiring.
- Current hypothesis / diagnosis: current main may land **below** 22.28 tok/s on the specdec fixture even in a cool session — M=1 decode now dispatches through the routed kernel, which `qmvbench` measures ~5% slower per projection at M=1.

## Important files

| Path | Why it matters | Current state |
|---|---|---|
| `benchmarks/FUSION_REPORT.md` | Final fusion-diagnosis report | Final, committed |
| `benchmarks/run_matrix.sh` | Matrix runner: phases `verify\|resolve\|itemA\|itemC` | Committed |
| `benchmarks/run_cell.sh` | Single-cell harness (pkill, readyz poll, one fixture request, JSON line) | Committed |
| `benchmarks/prompts/essay-1024.txt` | Pinned fixture, SHA-256 `7ed683f8…`, stream 599/1086/426 | Committed |
| `benchmarks/prompts/specdec-800.txt` | Pinned fixture, stream 645/1008/380, hash `139acb9d…` | Committed |
| `benchmarks/results/itemA.jsonl` / `itemC.jsonl` / `resolve.jsonl` | Authoritative runs + provenance resolution | Committed |
| `progress.md` | Task log, authoritative tables, open items | Committed |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | Load-time fusion engagement summary print | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35Kernels.swift` | QMV dispatch grid fix + E120 lane-decode fixes | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedLinear.swift` | Env gates + layouts + fallback diagnostics | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/QmvBench/main.swift` | Item B microbenchmark target | Committed (f730e87) |

## Decisions (standing)

| Decision | Reason | Rejected alternative |
|---|---|---|
| Keep both packed fusions default ON | Bit-exact, latency-neutral, net-0 memory, rollback knobs | Rolling back |
| Interleaved gate+up layout rejected | No gain at micro or end-to-end level; +6.5 GB copies | Keeping the env knob active |
| Cross-session absolute latencies invalid | Thermal/system state differs (116–124 vs 135–165 ms bands observed) | Comparing absolutes across sessions |
| Engagement proven via load-time summary | RSS not reliable on Metal shared heaps | RSS probing |
| Discard contaminated runs, keep as evidence | Only uncontaminated in-session Δs are valid | Outlier-trimming or reuse |
| Engine commits before server | AGENTS.md commit-order dependency | Server-first commit |
| Docs live in `qwen38-mtp-server`; workspace is symlink-only | `qwen38-mlx-server` is a non-repo DSH workspace — docs must version with the work; fork stays vendor-shaped | Moving docs into the workspace |

## Commands and evidence

```text
$ swift test --filter HTTPServerTests                          # 108 passed, 0 failed
$ cd ../mlx-swift-lm && swift test --filter Qwen35FusedSwiGLUProjectionTests   # 11/11
$ swift test --filter Qwen35FusedQKVProjectionTests             # 8/8
$ swift test --filter Qwen35FusedGDNProjectionTests            # 14/14
$ swift test --filter Qwen38MTPDiagnosticTests                 # 1/1 (~151 s)
$ bash benchmarks/run_matrix.sh verify                         # §0 provenance check, both fixtures
$ ./scripts/agent-checkpoint.sh                                 # fresh checkpoint for this file
$ git log -1 --oneline                                          # both repos: engine f730e87, server d071300
```

Useful artifacts: `benchmarks/results/*.jsonl`; `.tmp/itemA-clean.log`, `.tmp/itemA-rerun.log` (contaminated, kept as evidence), `.tmp/probe-summary.log` (engagement summary line).

## Risks and blockers

- Known issue: cross-session absolute step latencies not comparable (thermal/system state); only in-session Δs are valid.
- Expected wrinkle: specdec-800 has never run on the current binary — treat the first run as exploratory; if the stream hash doesn't reproduce `139acb9d…`, stop and investigate before any timing.
- Environment: `/tmp/benchvenv` does not survive a reboot — recreate before computing stream hashes if missing.
- Do not rerun / stateful: no parallel builds/tests during `run_matrix.sh` timing cells; do not rebuild the server binary mid-matrix; no git operations inside the workspace directory.

## Exact next step

1. Land the docs housekeeping: delete `mlx-swift-lm/docs/`, place `docs/README.md`, correct the test counts in `progress.md`.
2. Run `bash benchmarks/run_matrix.sh verify` on current main; confirm both fixtures reproduce their recorded streams.
3. Run the dual-fixture benchmark (5 reps each, interleaved, thermal state logged) and update the headline-throughput table in `progress.md`.
4. Refresh this handoff with the results before any context reset.

## Fresh checkpoint

Fresh-checkpoint procedure completed successfully (2026-09-13 17:40 BST):

```
# Repository checkpoint (server)
- Repository: /Users/cwong/ai/qwen38-mtp-server
- Branch: main
- HEAD: d071300
- Git status: ?? .tmp/  (untracked transient logs only)
- Latest commit: d071300 feat: fusion diagnosis benchmark harness and final report; correct Item A provenance

# Repository checkpoint (engine)
- Repository: /Users/cwong/ai/mlx-swift-lm
- Branch: main
- HEAD: f730e87
- Git status: ?? .tmp/  ?? docs/  (untracked transient logs + unfilled template only)
- Latest commit: f730e87 fix: QMV dispatch grid and E120 lane decode; add fused SwiGLU layout gate and load-time fusion diagnostics; add qmvbench microbenchmark target
```

Both worktrees clean of tracked changes; feature branches merged and deleted in both repos. _(Regenerate via `./scripts/agent-checkpoint.sh` at the start of the next session; the engine's untracked `docs/` line above disappears after this session's housekeeping.)_