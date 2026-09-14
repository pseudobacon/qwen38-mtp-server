# HANDOFF — 2026-09-14 (end of W4 session)

## Objective

The five-work-item pre-fused-kernel queue is **complete** (W1, W2, W3, W4, W5 all
done 2026-09-14). W4 verdict: **KEEP — `MLX_QWEN_MTP_HEAD_QUANT` default flipped
ON** (essay +8.6 %, specdec +6.1 %, 24/24 reps bit-exact). **Next task: correctness
bug A — specdec even-k (k=2, k=4) committed-stream divergence** (root-cause in the
engine's batched verify path + regression test + re-run the specdec
k∈{1,2,3,4} determinism gate). Acceptance criteria for bug A: (i) root cause
identified and documented, (ii) fix makes forced k=2 and k=4 bit-exact on
specdec-800 (`139acb9d…`) without changing the essay streams or the k=1/k=3
streams, (iii) regression test added, (iv) full determinism gate re-run on both
fixtures at k∈{1,2,3,4}, (v) progress.md bug-A section + decision record.

## Branch / commit / uncommitted state (verify independently before editing)

- Engine `/Users/cwong/ai/mlx-swift-lm`: `main` @ `b79140b` (W4 diagnostic test
  changes + Qwen35 head-fusion comment), expected clean. `feature/w4-head-q4`
  deleted after merge.
- Server `/Users/cwong/ai/qwen38-mtp-server`: `feature/w4-head-q4` @ `330d432` with
  uncommitted W4 changes at the time of the fresh checkpoint below (they are
  committed + merged to main after the HANDOFF write — verify with
  `git status --short`, `git log -1 --oneline` that main contains the W4 commit
  and the feature branch is deleted; if not, complete that first).
- If this file's state disagrees with `git status` / `git log`, Git wins —
  update this file first, then proceed.

## Important files

- `benchmarks/make_q4_head.py` — 4-bit head generation (idempotent; refuses to
  overwrite). Artifacts: `mtp-head/q4/` (238.9 MB, disk-only, gitignored) from
  `mtp-head/pinned/model.safetensors` (849.3 MB BF16, untouched).
- `benchmarks/run_w4ab.sh` — 2-cell × 2-fixture × 6-rep interleaved A/B matrix
  (gates: stream hash, head_selected engagement, phaseSumOK, materialized=0,
  finish_reason, completion_tokens, thermal snapshot per rep).
- `benchmarks/run_cell.sh` — now records `head_selected` + `fusion_summary` per rep
  (scans both stdout and stderr logs; the load-time prints go to stdout).
- `Sources/HTTPServer/Generation/MLXGenerator.swift` — `MLX_QWEN_MTP_HEAD_QUANT`
  selection (unset = default ON with loud BF16 fallback; 1 force q4; 0 rollback;
  invalid value throws).
- `Sources/HTTPServer/ServerConfig.swift` — env documented in `--help`.
- `Tests/MLXLMTests/Qwen38MTPDiagnosticTests.swift` (engine) —
  `QWEN_MTP_HEAD_TEST_PATH` / `QWEN_MTP_TEST_DEPTH` overrides (depth default 4),
  per-prompt + overall committed-stream hashes (printed, never asserted).
- `progress.md` — W4 section (full numbers + verdict), refreshed headline, W4
  decision on record, open items re-ordered (bug A → bug B).
- Results: `benchmarks/results/w4-ab-{essay,specdec}-{bf16,q4}.jsonl` (6 reps
  each, rep 1 warmup per state), `w4-diag-{bf16,q4}-{d1,d4,depth8}.txt`.

## Decisions and evidence

- **W4 verdict: KEEP, default ON.** In-session A/B at k=3 (Phase 3 protocol,
  binary `c56ca6ea…`): essay BF16 19.63 → q4 21.32 tok/s (+8.6 %); specdec BF16
  22.11 → q4 23.45 tok/s (+6.1 %); **all 24 reps bit-exact** (essay
  `949b9423…`, specdec `139acb9d…`); head fusion engaged in q4 cells
  (`head swiGLU 1 qkv 1`); zero QMV materializations; phase-sums exact.
- The delta is carried by **tGraphBuild** (−7.2/−7.8 ms — smaller fused graph);
  tEval is within in-session noise (−4.0/−0.1 ms). headbench isolated: head body
  flush 16.38 → **1.52 ms**, step 16.26 → 1.48 ms; per-round d=2 45.94 → 16.52 ms.
  The isolated collapse does not transfer 1:1 in-pipeline — recorded as an open
  observation, not blocking (W5 serialized-vs-sustained pattern).
- **The diagnostic's short prompts are NOT a committed-stream gate** — within a
  fixed head state the streams already change with depth (d1 ≠ d4 ≠ d8), and at
  depth 1 (width 2) prompt 1 diverges between head states. Same bug A/B
  bit-exactness family; the boundary is content/fixture-sensitive, not a clean
  width threshold. The A/B matrix at k=3 is the valid W4 gate.
- Recorded diagnostic acceptance (printed, never asserted): BF16 93.46 % (d8,
  historical), q4 93.97 % (d8); BF16 93.97 % (d4), q4 94.68 % (d4); BF16 95.31 %
  (d1), q4 96.35 % (d1).
- Head fusion eligibility requires stock `QuantizedLinear`: BF16 head ineligible
  (`head swiGLU 0 qkv 0`), 4-bit head engages both fusions. The head's quantized
  linears also route through the Item D QMV verify kernel (recorded interaction).
- `lm_head` stays 4-bit report-only (already 4-bit, 635.7 MB; changing it changes
  committed tokens).
- Post-flip verification (default ON, binary `9753a41e…`): default cells load q4
  and reproduce both pinned streams bit-exact; `=0` rollback loads BF16
  bit-exact.

## Commands / tests and results

- Engine: `swift test --filter Qwen38MTPDiagnosticTests` — PASS (35.9 s; BF16
  default head, depth 4, overall hash `06e40dd4…` as recorded).
- Server: `swift test --filter HTTPServerTests` — PASS (121 tests, 2 suites).
- A/B matrices: 24/24 GATE PASS (both fixtures, both states).
- `git diff --check` clean in both repos before the engine commit.
- **Fresh checkpoint (server): 2026-09-14 10:36 BST** —
  `bash /Users/cwong/ai/qwen38-mlx-server/scripts/agent-checkpoint.sh` from
  `/Users/cwong/ai/qwen38-mtp-server` (feature branch, uncommitted W4 changes);
  the fresh-checkpoint procedure **completed successfully**. Re-run it after the
  final merge so the recorded state is clean main.

## Unresolved risks

- **Bug A (next task):** specdec k=2/k4 committed streams diverge from the
  pinned stream (deterministic per prompt, ~95 % position knife-edge flips). The
  W4 diagnostic evidence adds: width-sensitivity exists even at width 2 on some
  prompts → the fix must make batched verify bit-exact with serial at ALL widths
  that the engine claims to support, not just width ≤ 5.
- **Bug B:** width ≥ 6 (k ≥ 5) commits wrong tokens on essay; missing SDPA
  exactness chunk in `attentionWithCacheUpdate` (qL·gqa > 32 boundary). Likely
  the same fix family as bug A — investigate together, document separately.
- In-pipeline attribution gap: isolated head-family collapse (−29 ms/round at
  d=2) vs measured tEval delta (noise) — a new in-pipeline profile (xctrace with
  shader profiler) could resolve it, but is not required.

## Operations that must not be repeated

- Do not run any timing cell or draft-k sweep at `QWEN_MTP_DRAFT_K ≥ 5` /
  `--spec-draft-n-max > 4` (bug B — wrong streams).
- Do not use the diagnostic's committed-stream hashes as a gate for any future
  A/B (width-sensitive prompts; recorded values only).
- Do not present cross-session absolute tok/s as conclusions — only in-session
  paired deltas. Current in-session headlines: essay 21.32, specdec 23.45 tok/s
  (q4 default, W4 matrix session).
- Do not rebuild the server binary mid-matrix; no parallel builds/tests during
  timing cells; binary SHA must match across a matrix.
- Do not run `make_q4_head.py` twice (it refuses to overwrite; `mtp-head/q4/` is
  disk-only, gitignored — regenerate only after deleting the tree).
- `*.log` and `*.json` are gitignored in the server repo — use `.txt` /
  `.jsonl` extensions for provenance files.
- The qmvbench product is `qmvbench` (lowercase); `/tmp/benchvenv` must exist for
  `run_cell.sh` (wiped on reboot); `/tmp/q4venv` (mlx 0.32.2) for quantization.
- zsh: no parentheses/brackets in inline `git commit -m`; use `git merge --no-edit`.

## One exact next step

In the engine repo, root-cause bug A: with the specdec-800 prompt at k=2
(`QWEN_MTP_DRAFT_K=2`, `--spec-draft-n-max 2`), bisect the verify path to find
where the batched forward first diverges from serial greedy at the ~95 % stream
position (wrong-stream hash `06882d85…`): compare per-round committed token IDs
and target logits between serial decode and verify rounds (engine-level
diagnostic, no server needed), confirm/refute the SDPA-exactness-chunk
hypothesis for width 3 (qL·gqa = 18 ≤ 32 — if the boundary is not the chunk,
look at GDN tape/state handling and the QMV verify routing at M=3), then fix,
add a regression test, and re-run the determinism gate on both fixtures at
k∈{1,2,3,4}.

## Completion marker

Fresh checkpoints completed successfully: **2026-09-14 10:36 BST** (server
repo, feature-branch state recorded above, before the final commit) and
**2026-09-14 10:38 BST** on both repos after the merges — engine `main`
@ `b79140b` and server `main` @ `34e67b6`, both clean (`git status --short`
empty), feature branches deleted. The fresh-checkpoint procedure completed
successfully.
