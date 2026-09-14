# HANDOFF — k=2 default flip, final headline, and stale-claim sweep (COMPLETE 2026-09-14)

> **Fresh checkpoint state (2026-09-14).** The fresh-checkpoint procedure
> **completed**: `./scripts/agent-checkpoint.sh` ran successfully and wrote
> `.dsh/last-agent-checkpoint` (gitignored in both repos) before this file was
> finalized.

## Objective and acceptance criteria (met)

Part A: flip the production default draft depth from the adaptive cost model to
pinned k = 2 (engine `Qwen38MTPBlockSession.draftPolicy`), prove the rollback
knob (`QWEN_MTP_DRAFT_K=3`) still works, run post-flip correctness gates
(default ≡ k=2 registered streams; k3 ≡ k3 registered streams; rollback
functional; load-time log line states the active depth), execute the final
headline benchmark run (6 reps × 2 fixtures, full Phase 3 protocol), and update
the headline table with one live row per fixture plus docs (README, progress.md,
HANDOFF). Part B: grep-driven stale-claim sweep over progress.md, PROFILE.md,
README, and this file.

All acceptance criteria met:

- Post-flip gate: 4/4 one-shot cells PASS (default k=2 and rollback k=3, both
  fixtures; registered hashes; `head_gate PASS (q4)`; `phaseSumOK` true;
  `draft_depth` field records the active depth).
- Final headline: 12/12 cells deterministic, single binary `e448b2e2…`, per-rep
  thermal snapshots clean (12/12, no warnings).
- Rollback knob verified bit-exact: `QWEN_MTP_DRAFT_K=3` reproduces the
  registered k=3 streams (`949b9423…` / `139acb9d…`) on both fixtures.
- Engine and server tests green; both repos merged to main; branches deleted.

## Final headline (one live row per fixture — current main)

| Fixture | tok/s (median, reps 2–6 of 6) | decode s (median) | stream hash | depthDist | acc/step |
|---|---|---|---|---|---|
| `essay-1024` | **21.89** | 46.77 | `949b9423…` | 2:461 | 1.2213 |
| `specdec-800` | **23.29** | 43.96 | `139acb9d…` | 2:431 | 1.3782 |

Config: default (QMV verify ON, fusions ON, compiled decode ON, 4-bit MTP head
ON, draft depth pinned k = 2, offer cap 3). Records:
`benchmarks/results/k2default-gate.jsonl` (gate) and
`benchmarks/results/k2default.jsonl` (headline); session log
`/Users/cwong/ai/qwen38-mtp-server/.tmp/k2default-run.log`; thermal log
`.tmp/k2default-thermal.log` (12 `pmset -g therm` snapshots, all clean).
All other headline numbers in `progress.md` are labeled
superseded-config/superseded-session historical provenance.

## Changes and evidence

**Engine (`../mlx-swift-lm`), commit `609e0d5` on main (branch
`feature/k2-default-flip` merged, deleted):**

- `Libraries/MLXLLM/Models/Qwen38MTPBlockSession.swift` — `draftPolicy` default
  branch now `Swift.min(offeredDepth, Self.defaultDraftDepth)` (k = 2) instead
  of the adaptive `costModelDepth`; `QWEN_MTP_DRAFT_K` override unchanged;
  the cost-model math moved to `costModelDepth`'s doc comment (retained as a
  documented research artifact); `defaultDraftDepth`'s doc documents the
  decision provenance and the rollback knob.
- `Tests/MLXLMTests/Qwen38MTPDiagnosticTests.swift` — both test sessions pin
  `session.draftPolicy = { offered, _ in offered }` so they exercise exactly
  their intended verify widths (4 → width 5; 5 → width 6) regardless of the
  production default. New measured values: acceptance 87.50 / 91.02 / 94.53,
  aggregate **91.02%** (width-5 offer); wide-verify stream 256/256
  serial-identical (genuinely width 6 now).

**Server (`qwen38-mtp-server`), commit on main (branch merged, deleted):**

- `Sources/HTTPServer/Generation/MLXGenerator.swift` — load-time log line
  `MLXLM: MTP draft depth: k=…` (default: `k=2 (default 2; offer cap 3;
  override QWEN_MTP_DRAFT_K)`; forced: `k=3 (forced via QWEN_MTP_DRAFT_K=3)`).
- `Sources/HTTPServer/ServerConfig.swift` — help text: `--spec-draft-n-max`
  default corrected 8 → 3; `QWEN_MTP_DRAFT_K` documented in the env-var block.
- `benchmarks/run_cell.sh` — parses the load-time draft-depth line into a
  `draft_depth` JSONL field.
- `benchmarks/run_k2default.sh` — new driver: `gate` phase (4 one-shot cells,
  EXPECT_HEAD=q4, hash + phaseSum + committed-1024 + draft-depth gates) and
  `headline` phase (6 reps × 2 fixtures interleaved, per-rep thermal snapshot,
  same protocol as `run_postw4.sh`).
- Docs: `progress.md` (headline table refreshed, registry consolidated with
  canonical labels, Phase 5 standing recommendation marked implemented, open
  item 5 re-scoped without the 24 ms target, W5 M=16/17 item closed-untriggered,
  W3 "current value" fixed, test counts updated, decision on record added),
  `benchmarks/PROFILE.md` (§1 scope note, §5(b) flush pointer, §5(c) item 2
  superseded note, §6 safety note corrected to the per-config registry
  semantics), `docs/README.md` (test counts, `QWEN_MTP_DRAFT_K` knob row,
  Recommended-configuration section with the final headline).

## Commands / verification

- `swift build --target MLXLLM` (engine) — clean.
- `swift test --filter Qwen38MTPDiagnosticTests` — 2/2 PASS (~68 s; acceptance
  aggregate 91.02%; wide-verify 256/256 serial-identical).
- `swift test --filter Qwen38SDPAExactnessTests` — 4/4 PASS.
- `swift build --configuration release --product qwen38-mtp-server` — clean;
  binary SHA `e448b2e2bbbfa7174c9d24e42108f5f721cb0a76b588f3fa3be7c1cbe1e167ab`.
- `swift test --filter HTTPServerTests` — 121/121 PASS.
- `bash benchmarks/run_k2default.sh gate` — 4/4 GATE PASS (hashes, phaseSumOK,
  committed 1024, head gate q4, draft_depth k=2 default / k=3 forced).
- `bash benchmarks/run_k2default.sh headline` — 12/12 cells deterministic;
  medians 21.89 / 23.29 tok/s (reps 2–6).

## Decisions on record

- **k = 2 is the production default draft depth.** Post-W4 queue evidence:
  post-W4 Phase 2+3 session (k2 21.28/22.75 vs adaptive default 19.63/21.11
  tok/s, in-session) + Phase 5 closure findings. The adaptive cost model is
  retired from the default path; it remains in the tree as a documented
  research artifact (`costModelDepth`).
- **Rollback:** `QWEN_MTP_DRAFT_K=3` (verified bit-exact against the registered
  k=3 streams); `MLX_QWEN_MTP_HEAD_QUANT=0` (BF16 head); `--spec-draft-n-max`
  offer cap bounds the effective k.
- **Diagnostic tests are pinned to the offered width** — they test the
  verify-width geometry, not the production depth policy. Do not let them
  silently track the production default.
- **The 24 ms `tEvalMs` / 30 tok/s target is formally retired** (v1.1-era
  target, different head state/binary/config). Open item 5 in progress.md is
  re-scoped as open work without a numeric target; any successor sets its own
  success criterion against the current build.
- **Stream-hash registry semantics (unchanged, now consistently stated):**
  per-(fixture, config) hashes; never claim bit-exactness across configs or
  against serial; the essay stream is config-invariant across the measured
  configs (no knife-edge flips landed); specdec serial (`c70882fc…`) differs
  from the MTP stream (`139acb9d…`) at the knife-edge family.

## Risks / unresolved

- Cross-session absolute tok/s remain non-comparable (thermal); only
  in-session deltas are valid. The final headline numbers are the live
  single-source-of-truth for current main.
- Specdec deep-k (k ≥ 5) was never measured (essay-only deep-k gate); deep
  drafts are net-negative on essay and the default no longer offers them, so
  this is low priority.
- The in-pipeline head cost remains unestablished (flush-free measurement
  pending); the head-structure workstream is closed — reopening requires a
  new task.

## Do NOT repeat

- Do not re-run the 6×2 headline matrix unless the binary or fixtures change —
  it is the final headline for this queue.
- Do not restore the adaptive cost model as the default without a fresh A/B
  session and explicit approval; do not delete `costModelDepth` (documented
  research artifact).
- Do not let the diagnostic tests track the production default policy — the
  width pins are intentional.
- Do not resurrect the 24 ms target; it is retired, not paused.
- Do not compare headline numbers across sessions (different binaries/
  thermal states); cite the records, not cross-session deltas.
- Engine commits before server; merge engine first. Checkpoint before ending
  work.

## Next step (exact)

None — the queue is complete. Both repos are on `main` with all work merged
and branches deleted. Any successor task starts from this handoff, the
`progress.md` roadmap (open items 1–6), and the current-main recommended
configuration in `docs/README.md`.

## Repository state (verified at write time)

- `qwen38-mtp-server`: branch `main` at `0973cf7` (the k=2 default flip task —
  server-side changes + docs, merged from `feature/k2-default-flip`), clean;
  this HANDOFF marker commit follows it on `main`; feature branch deleted.
- `../mlx-swift-lm`: branch `main` at `609e0d5` (the engine flip), clean;
  feature branch deleted. (Local main is ahead of `origin/main` — push is a
  separate, unrequested operation.)
- Fresh checkpoint: `.dsh/last-agent-checkpoint` written by
  `./scripts/agent-checkpoint.sh` (both repos) before this file was finalized.
