# HANDOFF — Head fusion exploration (COMPLETE, NEGATIVE 2026-09-14)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**: see the
> marker timestamps at the bottom of this file (written after the final edit
> batch of this task).

## Objective and acceptance criteria

Reduce the k = 2 eval window (80.4 ms) by ≥ 8 ms by fusing the MTP head
forward into the backbone's last-layer graph (Option A: one graph, one
blocking eval; Option B: 4-bit head — already in place). Success required
bit-exact stream hash `949b9423…` and a measured ≥ 8 ms/round win on a 6-rep
k = 2 cell. **Verdict: NO WIN — not implemented.** The task's own diagnostic
branch ("is the fusion removing work, or rearranging it?") is answered:
rearranging it.

## Result (one live number per fact)

- Head-family GPU work per round: **11.40 ms** (HeadBench, q4 head, d = 2,
  flush = 3: flush 1.95 + step 1.47 + 2 × proj1 7.98; proj1 = single-row
  4-bit backbone lm_head 635.7 MB + argmax = 3.99 ms, forced sequential by
  the autoregressive chain).
- 5-way trace (461-round essay cell, medians, binary `e448b2e2…`, hash
  `949b9423…`): d_head1 74 µs, d_chain 41 µs, verify_build 2,969 µs,
  eval_wall 79,870 µs, round 84,186 µs (stepAvg 84.18 ms).
- Head submitted 0.23 ms after round start → **3.13 ms hidden behind the
  verify build; 8.27 ms in the eval window; implied M = 3 tape 71.60 ms**
  (= serial 58.30 + 2 × 6.65 ms marginal row — the three independent numbers
  triangulate exactly).
- Per-round GPU util **98.6 %** (busy 83.00 / wall 84.19 ms): the host side is
  already fully hidden; the round is GPU-throughput-bound.
- **Option A prediction: +3.1 ms/round REGRESSION** (a fused single graph is
  submitted at t = 3.36 ms, losing the 3.13 ms the shipped asyncEval design
  hides; eval would be 11.40 + 71.60 = 83.00 ms vs 79.87 ms). No scheduling
  variant beats the shipped design (it already submits the head at the
  earliest possible instant).
- Even removing 100 % of the head work (impossible — draft ids are verify
  inputs) caps the saving at 8.27 ms, at the bar rather than through it.

## Why this closes the question

The verify input is `[primary] + draftIdArrays`: the head's draft-id work is
on the verify critical path. Fusion conserves every byte of weight streaming
(head 238.9 MB × 2 forwards + lm_head 635.7 MB × 2 single-row projections)
and only changes when the work is submitted. The only levers that actually
reduce GPU work are model-level (native 2-token head / draft-vocabulary
lm_head ≈ 12.1 ms/round for one step + proj1 + verify row), draft-depth
policy (k = 1: −12.1 ms/round at −1.22 tokens/round on this fixture), or the
verify tape itself (71.60 ms of backbone 4-bit weight streaming — the
separate workstream that owns the eval window).

## Files

- Server (`qwen38-mtp-server`): `benchmarks/PROFILE-K2.md` §8 (new addendum),
  `progress.md` (Done line + "Head fusion exploration" section), this file.
- Engine (`../mlx-swift-lm`): **no changes** (HeadBench pre-existed from the
  W2/W4 tasks and was used as-is).
- No source changes in either repo; no new binaries; no git branches needed
  beyond the docs commit in the server repo.

## Persistent facts (still live)

- Production default: pinned k = 2; rollback knob `QWEN_MTP_DRAFT_K=3`
  verified bit-exact; `--spec-draft-n-max` offer cap bounds effective k.
- Final headline (current main, single binary `e448b2e2…`): essay 21.89 /
  specdec 23.29 tok/s. Registered streams: essay `949b9423…` (k=2, q4),
  specdec `139acb9d…` (k=2, q4), specdec serial `c70882fc…`.
- Engine main `97a9d85` (FullBench); server main `0d8767c` (HANDOFF marker;
  the K2 decomposition commit is `6bc4bdc`). Verify with `git log` before
  relying on any of it.
- PROFILE-K2.md §6 verdict stands: no kernel-level win of order 10 ms is
  addressable from this checkout without changing draft depth, head design,
  or MLX C++.

## Commands / verification

- 5-way trace cell (reproduces the segment split): from the server repo,
  `MLX_QWEN_MTP_TRACE_PATH=/tmp/… benchmarks/run_cell.sh fusion5way "MLX_QWEN_MTP_TRACE=1" <port> benchmarks/prompts/essay-1024.txt q4`.
- HeadBench (isolated head cost): `cd ../mlx-swift-lm && .build/release/headbench --model /Users/cwong/ai/qwen38-mtp-server/weights --head /Users/cwong/ai/qwen38-mtp-server/mtp-head/q4 --drafts 2 --flush 3 --warmup 10 --timed 50`.
- No production source changed → no rebuild, no test rerun required (engine
  diagnostics and server suite last green on the current mains; the binary
  SHA `e448b2e2…` is unchanged).

## Do NOT repeat

- Do not implement Option A (fused head-into-backbone single graph): measured
  analysis predicts +3.1 ms/round; it is rearrangement, not removal.
- Never enable `MLX_QWEN_MTP_TRACE_SYNC_HEAD=1` (destroys head/verify overlap).
- Never cite per-rep FullBench M = 1 numbers at prime > 256 as serial-decode
  cost — use `--serial` mode or the in-pipeline cells.
- xctrace GPU `start-time` is trace-relative; always offset-calibrate.
- Never edit source files with python/sed/awk/shell scripts.
- Engine commits before server; plain commit messages (no parentheses/brackets
  under zsh); `git merge --no-edit`.
- `MLX_QWEN_MTP_TRACE=1` (5-way) adds ~2.19 ms/round of instrumentation — fine
  for segment splits, never for ranked absolute round times.

## Next step (exact)

None — the task is complete. Candidate follow-ups (recorded in
`benchmarks/PROFILE-K2.md` §8 and §6): (a) model-level head change (native
2-token head or draft-vocabulary lm_head, ≈ 12.1 ms/round), (b) K = 1 repair
path, (c) host tape-build flush in MLX C++ (3.4 ms tG), (d) verify-tape
workstream (71.60 ms backbone weight streaming).

## Repository state (verified at write time)

- `qwen38-mtp-server`: branch `main`, clean at the last HANDOFF marker;
  this task's docs commit lands on a feature branch and merges to `main`
  (see the commit line recorded in `progress.md` after the merge).
- `../mlx-swift-lm`: branch `main` at `97a9d85`, clean, untouched by this
  task.
- Fresh checkpoint markers: timestamps recorded below; fresh-checkpoint
  procedure completed in both repos.
