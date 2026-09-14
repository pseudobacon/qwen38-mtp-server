# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a **resume-state checkpoint**: the next task, the task just finished, repo state, and the hard rules. History lives in `progress.md` — never duplicate a number here that progress.md already carries.
> **Rolling policy:** when a task completes, its detail moves to progress.md; this file keeps a one-paragraph outcome plus a pointer. Older tasks collapse to one line each.

## Goal (next task) — `qmvbench` throughput mode + M = 16/17 extension

Objective: close the Item D null. Add a throughput mode to `qmvbench` (batch N calls per sync instead of per-call sync) so per-kernel numbers reflect async pipeline conditions, and extend the M sweep to 16/17 (the width band above the current 2..9 gate). This establishes whether the candidate QMV kernel's micro win (~15–20% at M = 2..4 under per-call sync) persists under async conditions at all.

Acceptance criteria: throughput-mode numbers for routed vs incumbent `quantizedMM` at M ∈ {2,4,7,9,16,17} under identical async submission patterns; a verdict on whether an in-model Item D re-run is worth pursuing. Any such re-run additionally requires the guard fix first: replace the per-call `x.asData(access: .noCopy).strides` probe (which calls `self.eval()` on every routed dispatch — the root cause of the Item D null) with a metadata-only, no-eval contiguity decision (e.g. validate row-major layout once per shape at warm-up, where a flush is harmless, and cache the decision; public `MLXArray.strides` is deprecated, `asData` always evaluates).

Non-goals: do not flip the `MLX_QWEN_QMV_VERIFY` default (stays OFF); do not touch the GDN `in_proj` fused path; do not re-add the interleaved layout.

## Last completed — Item D: verify-pass QMV routing (2026-09-14)

Implemented engine gate `MLX_QWEN_QMV_VERIFY` (default OFF, loud load-time log): 3-D verify `[B, L, K]` with `B·L ∈ 2..9` reshaped to `[B·L, K]` and routed through the candidate QMV kernel; M = 1 flipped to incumbent `layer(x)` (kernel ~5% slower at M = 1 per qmvbench); per-dispatch routed/fallback/materialized counters in MTP-STEP-SUMMARY and at shutdown. All Phase 2 gates passed: 18/18 fusion projection tests in both knob states (incl. new 3-D routing tests), `Qwen38MTPDiagnosticTests` unchanged (93.46% / 16.25 / 14.0), OFF verify matrix bit-exact with zero routed dispatches, ON spot cell bit-exact with 99.1% routed share and zero materializations. 12-cell in-session A/B on essay-1024 (single binary, env-var-only difference, 5 measured reps per cell, no thermal warnings, all 12 reps bit-exact): mean Δ **−0.252 ms/step** (D1 marginally slower; 4/5 pairs favor D1), decode wall 58.644 vs 58.750 s, TTLT 17.475 vs 17.446 tok/s — **NULL, default stays OFF**. The tEvalAvg "improvement" (125.6 → 6.3 ms) is a **measurement artifact**, not a kernel win: the guard's per-call `asData` probe calls `self.eval()` on every routed dispatch, shifting ~119 ms of GPU wait from the tEval phase into the graphBuild phase (10.3 → 129.9 ms) and serializing host build + GPU execution; the qmvbench micro win does not transfer end-to-end. Engine `b900aad`; server `4ca9589` + `4af4e73`. Full numbers, per-rep table, and attribution: progress.md "Phase 3 — Item D A/B" section; records `benchmarks/results/itemd-D0-essay.jsonl` / `itemd-D1-essay.jsonl`.

## Earlier completed (pointers)

- Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13): essay-1024 17.55 tok/s (136.559 ms steps), specdec-800 18.92 tok/s (141.623 ms); compiled-path family ablated at +7.35 ms/step, kept default ON. Detail: progress.md.
- Interleaved gate+up layout: implemented, measured (no gain, +6.5 GB), removed — engine `901d2ca`. Detail: progress.md.
- Fusion diagnosis (Items A–C): both packed fusions bit-exact and latency-neutral; QMV grid bug fixed; prompt provenance resolved. Engine `f730e87`, server `d071300`. Detail: `benchmarks/FUSION_REPORT.md`.

## Repository checkpoint

- Server `main` at `4af4e73` plus one docs/results commit landing this handoff (progress.md + README counts + Item D result JSONL + this file); engine `../mlx-swift-lm` `main` at `b900aad` (no engine changes since Item D implementation); both worktrees clean (fresh checkpoint below).
- Environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/`; venv `/tmp/benchvenv` (HF tokenizer for stream hashes — wiped on reboot, recreate if missing).
- Workspace: `qwen38-mlx-server/` is a **non-repo DSH workspace** of symlinks — no git operations inside it; run everything in the real repos (`/Users/cwong/ai/qwen38-mtp-server`, `/Users/cwong/ai/mlx-swift-lm`). Keep both symlinks at the same level (engine referenced as `../mlx-swift-lm`).
- Provenance note: `run_cell.sh`'s `server_dirty` field reads `dirty` for any matrix run because the matrix's own result JSONL files are untracked at record time (source tree is clean; `.tmp/` is gitignored). Known false positive — do not "fix" it mid-matrix.

## Hard rules (inline because breaking them invalidates runs)

- Only in-session deltas are valid; cross-session absolute latencies are never comparable.
- Force-recompile changed engine modules (or compare relink SHA-256) before benchmarking — the stale-binary trap.
- No parallel builds/tests during timing cells; no mid-matrix rebuild; log `pmset -g therm` per rep.
- Every timing cell must reproduce its fixture's stream hash — a mismatch is a correctness stop, not a performance result.
- Engine commits before server; `git merge --no-edit`; zsh-safe messages; no `print()` in hot paths; no python/sed source edits.
- Decisions on record and the full benchmark protocol: `progress.md`. Orientation, build/test commands, env knobs, file map: `docs/README.md`.

## Exact next step

1. **`qmvbench` throughput mode** (N calls per sync) + M = 16/17 sweep — engine work in `mlx-swift-lm` (`Sources/QmvBench`); then decide whether an in-model Item D re-run is warranted (guard fix is the prerequisite, see Goal).
2. Thermal control for benchmarks (shorter run blocks / cooldowns) so absolute numbers become comparable across sessions.
3. Attention-layer kernels and acceptance-rate work — the remaining path toward the `tEvalMs` 24 ms / 30 tok/s target.

## Fresh checkpoint

_(filled in by the session ending this task, after the final commits — see the recorded checkpoint below)_.
