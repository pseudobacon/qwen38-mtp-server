# HANDOFF — 2026-09-14 (post-W4 queue complete)

## Objective

The six-phase post-W4 queue is **complete** (Phases 0–5 all done 2026-09-14).

| phase | acceptance criterion | result |
|---|---|---|
| 0 | harness hardening | DONE — `EXPECT_HEAD` gate in `run_cell.sh`/`run_matrix.sh`; q4 default fail-loud; PROFILE.md §7 flush-contamination label |
| 1 | Bug A verdict | DONE — **precision family, not a logic bug**; per-(fixture, config) stream-hash gate policy now binding; no fix |
| 2+3 | headline re-measure + k=2 | DONE — 36-cell session, all deterministic; headline q4 default essay 19.63 / specdec 21.11 tok/s; k2 21.28 / 22.75; serial 16.26 / 16.22 |
| 4 | Bug B fix | DONE — SDPA exactness chunk in `attentionWithCacheUpdate`; unit + model regression tests green; deep-k gate: d5/d6/d8 essay serial-identical, headline hashes invariant |
| 5 | head-structure decision | DONE — **CLOSE the head-structure workstream**; q4 head stays as-is; k=2 recommended operating point |

## Branch / commit / uncommitted state (verify independently before editing)

- Engine `/Users/cwong/ai/mlx-swift-lm`: `main` @ `126591a` (Bug B fix + regression tests),
  clean. `feature/postw4-queue` merged (fast-forward) and deleted.
- Server `/Users/cwong/ai/qwen38-mtp-server`: `main` @ `4a5c9d9` (queue results, drivers,
  docs), clean. `feature/postw4-queue` merged (fast-forward) and deleted.
- If this file's state disagrees with `git status` / `git log`, Git wins — update this
  file first, then proceed.

## Important files

- Engine `Libraries/MLXLMCommon/AttentionUtils.swift` — the Bug B fix: SDPA exactness
  chunk (gate: L ∈ 6..9, `L·gqa > 32`, head_dim 256, `5·gqa ≤ 32`, `offset > 0`,
  symbolic `.causal`; 5-row + (L−5)-row vector-kernel calls with incremental KV scatter
  updates). Index-slice style: `arr[.ellipsis, s..<e, 0...]` (the fork's
  `arr[range, axis:]` is deprecated; double `.ellipsis` index lists are rejected).
- Engine `Tests/MLXLMTests/Qwen38SDPAExactnessTests.swift` — unit regression
  (bit-exact vs promoted-windows reference, L ∈ {6,7,8,9} × offset ∈ {13,29,41,57,250},
  live cache state equality; negative controls L=5 / offset=0 / `.array` mask stay on
  the legacy single call). Merge-safe, permanent.
- Engine `Tests/MLXLMTests/Qwen38MTPDiagnosticTests.swift` — added
  `testWideVerifyStaysInSerialFamily` (serial vs depth-5 width-6 streams, 256 tokens,
  greedy: 256/256 match, no divergence; asserts `firstDivergence >= 16` and
  `matchRate >= 0.85`).
- Engine `Tests/MLXLMTests/Qwen38BugADiscriminatorTests.swift` — **removed** (throwaway
  Phase 1 discriminator; it exists only in the feature-branch history commit `91ab1c8`,
  never on main). Do not resurrect it; the Phase 1 evidence is in progress.md.
- Server `benchmarks/run_postw4.sh` / `benchmarks/run_deepk.sh` — the two gate-session
  drivers (6 cells × 6 reps interleaved; deep-k = 3 cells × 6 reps + 2 regate cells).
- Server `benchmarks/results/postw4.jsonl` (36 records) and `deepk.jsonl` (20 records) —
  provenance for Phases 2/3/4/5.
- Server `progress.md` — Phase 0–5 sections + roadmap (all queue items closed).
- Server `Sources/HTTPServer/Generation/MLXGenerator.swift` — q4 head default is now
  **required** (fail-loud: unset env → q4 mandatory; missing q4 tree → startup throw).
  `MLX_QWEN_MTP_HEAD_QUANT=0` is the BF16 rollback.
- `mtp-head/q4/` (238.9 MB) — disk-only, gitignored; required at startup now.

## Decisions and evidence

- **Bug A (Phase 1): precision family.** The batched M-row verify forward is a different
  bf16 reduction order than M=1 serial (drift ≤ 2 ulp = 0.125 at logit ~21); flips
  argmax only at knife-edge positions (top-2 gap ≤ 2–4 ulp; ~9 per 1024 on specdec).
  The pinned `139acb9d…` reference is an MTP-path stream, not serial greedy
  (`c70882fc…`, head-invariant). W3's "even-k divergence" was a confound of the
  non-serial reference. No code fix; candidate mitigations need explicit approval.
- **Per-config stream-hash registry (1024 tokens, q4):**

  | config | essay-1024 | specdec-800 |
  |---|---|---|
  | serial | `949b9423…` | `c70882fc…` |
  | k=2 | `949b9423…` | `139acb9d…` |
  | default (cost model) | `949b9423…` | `139acb9d…` |
  | k=5/6/8 (post-fix) | `949b9423…` | unmeasured (essay-only sweep) |

  Never claim cross-config or against-serial bit-exactness; gate per (fixture, config).
- **Bug B (Phase 4): fixed.** Pre-fix, verify widths M = 6..9 fell to the reference
  matmul → fp32-softmax → matmul SDPA fallback (qL·gqa > 32, head_dim 256 not in the
  fused full-path set) → gross corruption (W3 `da7bb159…`, first divergence at the
  first drafted token). Post-fix: d5/d6/d8 essay all serial-identical (`949b9423…`),
  deterministic, gates PASS; headline (M ≤ 5) hashes invariant in the regate cells.
  The gate is geometry-specific — no M ≤ 5, prefill, other-model, or array-mask path
  is touched (negative-control tests).
- **Phases 2+3 (in-session, one 1 h session, binary `db39b916…`):** headline default
  essay 19.63 / specdec 21.11 tok/s; k2 21.28 / 22.75; serial 16.26 / 16.22. k2 beats
  default (+9.9 % / +7.8 %); default beats serial (+20.4 % / +30.4 %); k2 vs serial
  (+31.0 % / +40.2 %). Serial tEval floor 59–62 ms/step; k2 − serial tEval 38–40 ms
  (joint: 2 head steps + width-3 batch effect). Thermal drift present (later reps
  slower); medians used; only in-session deltas are conclusions.
- **Phase 5: CLOSE the head-structure workstream.** In-pipeline head cost is well
  under the flush-contaminated isolated upper bounds (24.66–26.02 ms/round); deep
  drafts are net-negative on essay (d5/d6/d8 = 14.53/12.85/9.80 tok/s vs 21.28 at k2;
  acc/step plateaus ~1.7–1.8 while stepAvg grows superlinearly 185→212→283 ms). The
  q4 head stays as-is (W4 KEEP, default ON, now required). Standing recommendation:
  `QWEN_MTP_DRAFT_K=2` for this fixture class; a default depth-policy change is a
  separate A/B decision.

## Commands / tests and results (all observed this session)

- Engine `swift test --filter Qwen38SDPAExactnessTests` — PASS (0.09 s, 4 tests).
- Engine `swift test --filter Qwen38MTPDiagnosticTests` — PASS (63.9 s; acceptance test
  + `testWideVerifyStaysInSerialFamily` 256/256 match).
- Server `swift test --filter HTTPServerTests` — PASS (107 XCTest + 121 Swift Testing
  tests, 0 failures), built against the fixed engine.
- `swift build --configuration release --product qwen38-mtp-server` — release binary
  rebuilt with the fix (`db39b916…` was pre-fix; the post-fix binary was built before
  the deepk session and used there — see `deepk.jsonl` `binary_sha256`).
- Gate sessions: `run_postw4.sh` 36/36 cells; `run_deepk.sh` 20/20 cells — all
  deterministic, `head_gate` PASS, phaseSumOK, `committed=1024`.
- `git diff --check` clean in both repos before each commit.

## Unresolved risks

- Deep-k sweep was **essay-only**; specdec (higher acceptance, 2.38 acc/step at k2)
  could shift the depth optimum — recorded caveat, not blocking.
- Thermal drift over long sessions: per-rep `pmset -g therm` logging exists but
  shorter run blocks / cooldowns are still the proper control (roadmap item).
- Cross-session absolute tok/s are labels, never conclusions (W4's 21.32/23.45 and
  this session's 19.63/21.11 differ ~8 % — thermal/ambient, not code).
- `/tmp/benchvenv`, `/tmp/q4venv`, and `/tmp/*.log` session logs are wiped on reboot;
  durable provenance is the `.jsonl` records in `benchmarks/results/`.
- The default adaptive depth policy is suboptimal vs k2 on both fixtures — left
  unchanged by design (separate decision).

## Operations that must not be repeated

- No parallel builds/tests during timing cells; no server-binary rebuild mid-matrix;
  binary SHA must match across a matrix.
- Do not run `make_q4_head.py` twice (refuses to overwrite; regenerate only after
  deleting the tree). The q4 tree is now **required** at startup (fail-loud).
- Do not use the diagnostic's short-prompt committed-stream hashes as an A/B gate
  (width/head-state sensitive; recorded values only).
- Do not present cross-session absolute tok/s as conclusions.
- `*.log` and `*.json` are gitignored in the server repo — provenance files use
  `.txt` / `.jsonl`.
- zsh: no parentheses/brackets in inline `git commit -m`; always `git merge --no-edit`.
- Engine commits before the server; matching feature branches in both repos at task
  start (none exist now — both on clean main).
- Do not resurrect the throwaway Phase 1 discriminator test.

## One exact next step

Nothing is pending from this queue. The next task is a new prompt; candidates already
recorded in `progress.md` roadmap: (1) thermal-controlled benchmark runs for
  cross-session comparability, (2) attention-layer kernel / acceptance-rate work
  toward the tEval 24 ms / 30 tok/s target, (3) an A/B session if `QWEN_MTP_DRAFT_K=2`
  is to become the default depth policy. Verify the branch/commit state above before
  editing.

## Completion marker

Fresh checkpoints completed successfully: **2026-09-14 13:04 BST** — server repo
`/Users/cwong/ai/qwen38-mtp-server` at `main` @ `4a5c9d9` clean
(`.dsh/last-agent-checkpoint` 2026-09-14T13:04:10+01:00) and engine repo
`/Users/cwong/ai/mlx-swift-lm` at `main` @ `126591a` clean
(`.dsh/last-agent-checkpoint` 2026-09-14T13:04:57+01:00). Both feature branches
merged and deleted; `git status --short` empty in both repos. The fresh-checkpoint
procedure completed.
