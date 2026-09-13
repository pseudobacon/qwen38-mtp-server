# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a task-state checkpoint, not a complete conversation transcript.
> Verify all claims against the working tree before acting.

## Goal

- Objective: Benchmark harness (rev 2) diagnosing Qwen 3.8 gate+up / QKV fusion — Items A–C — on the `mlx-swift-lm` fork + `qwen38-mtp-server`, plus final markdown report (§0 verdict, A/B/C tables, conclusions (a) net win/regression per fusion, (b) gate+up regression cause, (c) does interleaved layout recover the +3–9 % decode range).
- Acceptance criteria: §0 prompt provenance resolved; Item A 2×2 matrix with bit-exact cells and in-session Δs on a fusion-verified binary; Item B standalone `qmvbench` bit-exact at M=1/4; Item C `MLX_QWEN_SWIGLU_LAYOUT=interleaved` bit-exact and measured vs global; report complete; engine committed+merged before server; tests green.
- Constraints / non-goals: no sandbox escalation (approval prompts disabled); engine commits before server; `git merge --no-edit`; no python/sed source edits; no `print()` in hot paths (load-time prints allowed); zsh-safe commit messages; narrowest test after edit batches; no parallel builds/tests during a timing matrix run.

## Repository checkpoint

- Updated: 2026-09-13 (TZ Asia/Shanghai local machine clock) — fresh checkpoint run recorded below in the checkpoint section.
- Branch (server): `main` after feature-branch merge (feature branch deleted).
- Branch (engine `../mlx-swift-lm`): `main` at `f730e87` (fast-forward of `0514b11`); `feature/fusion-diagnosis` deleted.
- HEAD (server): see `git log -1 --oneline` in this repo.
- Worktree: clean apart from untracked `.tmp/` (transient logs) in both repos.
- Git status: `git status --short` — `progress.md`, `benchmarks/`, `docs/` committed in the server feature branch before merge.
- Uncommitted changes: none expected after the final commit.
- Relevant environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/`; venv `/tmp/benchvenv` (HF tokenizer for stream hashes).
- Required services / local processes: none (benchmark server is started and pkill'd per cell by `benchmarks/run_cell.sh`).

## Current state

- Completed:
  - §0: pinned essay fixture reproduces 599/1086/426 (hash `949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe`), NOT the recorded 645/1008/380 (that belongs to `benchmarks/prompts/specdec-800.txt`, hash `139acb9d…` — resolved in `benchmarks/results/resolve.jsonl`).
  - Fusion engagement resolved: the single load-time fallback log line is the **BF16 MTP head layer only** (ineligible by design). New load-time summary in `Qwen35TextModel.prepare()`: `backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 0 qkv 0` — all quantized backbone layers fused (qkv = 16 full-attn layers; gdn = 48 GDN input-projection layers).
  - Item A (clean, fusion-engaged binary): 24/24 cells bit-exact (599/1086/426, hash `949b9423…`). Reps 2–6 means: A0 143.042, A1 143.304 (+0.262 ms), A2 142.967 (−0.076 ms), A3 145.429 (+2.387 ms); per-rep Δ swings −21.8…+15.2 ms → **both fusions latency-neutral in-session**.
  - Item A invalidations: original 02:41 binary predated the QMV dispatch fix + fusion merge → ran fully eager (proven: correct greedy hash under a dispatch that would have diverged); first re-run discarded for timing contamination (parallel test runs; spikes 239.6/196.1/178.9 ms).
  - Item B (`qmvbench`): complete; all 7 conditions bit-exact at M=1 and M=4. M=1: routed ≈5 % slower than incumbent per projection; M=4: routed ≈15–20 % faster; interleaved ≈ global at both M.
  - Item C: complete; 12/12 cells bit-exact; Cglobal 150.578 ms, Cint 152.187 ms, Δ +1.609 ms (+1.1 %) inside noise → interleaved does NOT recover the +3–9 % decode range and regresses nothing.
  - Tests: engine SwiGLU fusion 11/11, QKV fusion 8/8, GDN fusion 14/14, `Qwen38MTPDiagnosticTests` 1/1; server `swift test --filter HTTPServerTests` 108 passed / 0 failed.
- In progress: none.
- Not started: none.
- Current hypothesis / diagnosis: none — report final (`benchmarks/FUSION_REPORT.md`).

## Important files

| Path | Why it matters | Current state |
|---|---|---|
| `benchmarks/FUSION_REPORT.md` | Final report: §0, Items A/B/C tables + conclusions (a)–(c) | Final, committed |
| `benchmarks/run_matrix.sh` | Matrix runner: phases `verify\|resolve\|itemA\|itemC` | Committed |
| `benchmarks/run_cell.sh` | Single-cell harness (pkill, readyz poll, one fixture request, JSON line) | Committed |
| `benchmarks/prompts/essay-1024.txt` | Pinned fixture, SHA-256 `7ed683f8…` | Committed |
| `benchmarks/results/itemA.jsonl` | Authoritative Item A (clean re-run, 24 cells) | Committed |
| `benchmarks/results/itemC.jsonl` | Item C (12 cells, bit-exact) | Committed |
| `progress.md` | Task log incl. Item A provenance + engagement finding | Committed |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` | Load-time fusion engagement summary print | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35Kernels.swift` | QMV dispatch grid fix + E120 lane-decode fixes | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/MLXLMCommon/FusedQuantizedLinear.swift` | Env gates + global/interleaved layouts + fallback diagnostics | Committed (f730e87) |
| `../mlx-swift-lm/Libraries/QmvBench/main.swift` | Item B microbenchmark | Committed (f730e87) |

## Decisions

| Decision | Reason | Rejected alternative |
|---|---|---|
| Discard original Item A numbers | 02:41 binary provably ran fully eager (correct hash under buggy dispatch that would diverge) | Keeping them as "neutral" evidence |
| Discard first re-run numbers | Parallel test runs contaminated last cells (spikes up to 239.6 ms) | Outlier-trimming (arbitrary) |
| Clean re-run with zero parallel work | Only uncontaminated in-session Δs are valid | Reusing contaminated data |
| Prove engagement via load-time summary, not RSS | RSS not reliable on Metal shared heaps | RSS probing (inconclusive/inverted) |
| Commit engine before server | AGENTS.md commit-order dependency | Server-first commit |

## Commands and evidence

```text
$ swift test --filter HTTPServerTests            # server: 108 passed, 0 failed
$ cd ../mlx-swift-lm && swift test --filter Qwen35FusedSwiGLUProjectionTests   # 11/11 passed
$ swift test --filter Qwen35FusedQKVProjectionTests    # 8/8 passed
$ swift test --filter Qwen35FusedGDNProjectionTests    # 14/14 passed
$ swift test --filter Qwen38MTPDiagnosticTests  # 1 test passed (151 s)
$ bash benchmarks/run_matrix.sh itemA           # clean re-run: 24/24 bit-exact
$ bash benchmarks/run_matrix.sh itemC           # 12/12 bit-exact
$ ./scripts/agent-checkpoint.sh                 # fresh checkpoint — see result below
```

- Passing checks: all suites above; `git diff --check` clean in both repos.
- Failing checks: none.
- Checks not yet run: none outstanding.
- Useful logs / artifacts: `benchmarks/results/*.jsonl`; `.tmp/itemA-clean.log`, `.tmp/itemA-rerun.log` (contaminated, kept as evidence), `.tmp/probe-summary.log` (engagement summary line).

## Risks and blockers

- Known issue: cross-session absolute step latencies are not comparable (thermal/system state); only in-session Δs are valid.
- Assumption needing verification: none.
- External dependency or access constraint: none (weights local).
- Do not rerun / potentially stateful action: do not run builds/tests in parallel with `run_matrix.sh` timing cells; do not rebuild the server binary mid-matrix.

## Exact next step

1. Run:
   ```bash
   git log -1 --oneline   # both repos, confirm merged state
   ```
2. Expected result: engine `main` at f730e87; server `main` at the fusion-diagnosis commit; both worktrees clean (untracked `.tmp/` only).
3. If it fails: inspect `git status --short` / `git log --oneline -3` and repair the merge before any new work.
4. Then do: new task only.

## Fresh checkpoint

- `./scripts/agent-checkpoint.sh` result: see appended result below (recorded at session end).

</parameter>