# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a **resume-state checkpoint**: the next task, the task just finished, repo state, and the hard rules. History lives in `progress.md` — never duplicate a number here that progress.md already carries.
> **Rolling policy:** when a task completes, its detail moves to progress.md; this file keeps a one-paragraph outcome plus a pointer. Older tasks collapse to one line each.

## Goal (next task) — Item D: verify-pass QMV routing

Objective: route the verify pass through the candidate QMV kernel. The `ndim == 2` guard currently sends verify (3-D batched `x`, the dominant `tEvalAvg` share) to incumbent `quantizedMM`; a free reshape to `[M, K]` would dispatch it through the routed kernel at M ∈ 2…9, where `qmvbench` measures ~15–20% faster. Gate on an end-to-end A/B win, not the microbench alone (the A2 null result is the cautionary precedent).

Acceptance criteria: bit-exact greedy streams on both fixtures (essay 599/1086/426, hash `949b9423…`; specdec 645/1008/380, hash `139acb9d…`); in-session A/B (default vs Item D) on at least one fixture, 5 measured reps per cell, Phase 3 harness pattern (fresh server per cell, 6 reps interleaved, rep 1 warmup, per-rep `pmset -g therm` + determinism gate); keep only on an end-to-end win.

Non-goals: do not re-add the interleaved layout; do not change fusion defaults (`MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU` stay ON).

## Last completed — Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13)

18/18 reps bit-exact. Headline on current main: essay-1024 **17.55 tok/s** (136.559 ms steps), specdec-800 **18.92 tok/s** (141.623 ms steps); compiled-path ablation (B3 − B1): **+7.35 ms/step (+5.38%)** for the `MLX_COMPILED_DECODE`-gated family. B2 vs the 22.28 record: acceptance identical, gap entirely step time (+17.2%) — session thermal + M=1 routed-QMV dispatch; cross-session comparison remains invalid. Matrix ran on force-recompiled binary SHA-256 `88e27643…` (the 18:38 binary predated engine `901d2ca`; SwiftPM had not invalidated the changed modules). Server commit `4fb92ca`. Full tables: progress.md Phase 3 section; artifacts `benchmarks/results/rebaseline-*.jsonl`, `ablation-compiled-off.jsonl`.

## Earlier completed (pointers)

- Interleaved gate+up layout: implemented, measured (no gain, +6.5 GB), removed — engine `901d2ca`. Detail: progress.md.
- Fusion diagnosis (Items A–C): both packed fusions bit-exact and latency-neutral; QMV grid bug fixed; prompt provenance resolved. Engine `f730e87`, server `d071300`. Detail: `benchmarks/FUSION_REPORT.md`.

## Repository checkpoint

- Server `main` at `4fb92ca`; engine `../mlx-swift-lm` `main` at `901d2ca`; both worktrees clean (fresh checkpoint below).
- Environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/`; venv `/tmp/benchvenv` (HF tokenizer for stream hashes — wiped on reboot, recreate if missing).
- Workspace: `qwen38-mlx-server/` is a **non-repo DSH workspace** of symlinks — no git operations inside it; run everything in the real repos (`/Users/cwong/ai/qwen38-mtp-server`, `/Users/cwong/ai/mlx-swift-lm`). Keep both symlinks at the same level (engine referenced as `../mlx-swift-lm`).

## Hard rules (inline because breaking them invalidates runs)

- Only in-session deltas are valid; cross-session absolute latencies are never comparable.
- Force-recompile changed engine modules (or compare relink SHA-256) before benchmarking — the stale-binary trap.
- No parallel builds/tests during timing cells; no mid-matrix rebuild; log `pmset -g therm` per rep.
- Every timing cell must reproduce its fixture's stream hash — a mismatch is a correctness stop, not a performance result.
- Engine commits before server; `git merge --no-edit`; zsh-safe messages; no `print()` in hot paths; no python/sed source edits.
- Decisions on record and the full benchmark protocol: `progress.md`. Orientation, build/test commands, env knobs, file map: `docs/README.md`.

## Exact next step

1. **Land the outstanding `run_cell.sh` provenance logging** (agreed in Phase 4, not yet implemented): emit `binary_sha256` (shasum -a 256 of the release binary) plus engine/server HEADs and dirty-worktree flags into each cell's JSONL record. Smoke-test with one `run_matrix.sh verify` run.
2. Item D: relax the `ndim == 2` guard on the verify path via the free 2-D reshape; bit-exact tests first, then `run_matrix.sh verify` on the force-rebuilt binary.
3. In-session A/B per the Goal section; keep only on an end-to-end win.
4. Update progress.md with results; refresh this handoff per the rolling policy.

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

Both worktrees clean of tracked and untracked changes (`.tmp/` transient logs are gitignored); `feature/prompt-phase3` merged and deleted in the server repo; no engine changes this task. _(Regenerate via `scripts/agent-checkpoint.sh` — run inside each repo — at the start of the next session.)_
