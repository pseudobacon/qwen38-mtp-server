# LEV-E — Draft-select Metal kernel ceiling (bound from the SCH1 ledger + confirmation)

**Run ID:** lev-e-draftselect (confirmation run2) + `sch1-20260918-0041` (ledger)
**Date:** 2026-09-18
**Decision:** **CLOSE the `qwen35DraftSelectKernel` lever.** The addressable
draft-select/accept-walk host cost is **~0.81 ms (fresh run2, mean) to ~1.60 ms
(SCH1, mean) per round — both below the 2 ms/round kill-switch threshold.**
Evidence: the Swift compact draft-walk negative result (Task 1, 2.05 % *slower*)
plus this bound.

---

## What the walk owns

The k=2 speculative round phases (from `MLX_QWEN_MTP_TRACE` per-round stamps):

| phase | window | walk-owned? |
|---|---|---|
| draft_build | tRound0 → tDraftBuilt | head-chain submit (GPU submit), not the accept walk |
| **snap** | tDraftBuilt → tSnapshotDone | **YES** — recurrent rollback snapshot |
| **tape** | tSnapshotDone → tTapeBuilt | **YES** — the M=3 verify-tape concat |
| verify_build (rest) | tTapeBuilt → tVerifyBuilt | NO — 64-layer verify graph encode + async-ladder GPU wait |
| eval_wall | tVerifyBuilt → tEvalDone | GPU (kernel exec + sync); top-2 kernels ride here |
| **readout** | tEvalDone → tReadDone | **YES** — top-2 `.item()`/`.asArray()` copies |
| **commit** | tReadDone → tCommitDone | **YES** — accept walk + rollback state update |
| **upkeep** | tCommitDone → tTailDone | **YES** — per-round state/EMA housekeeping |

A `qwen35DraftSelectKernel` would move the top-2 selection, the accept decision,
and the rollback state update off the host. The addressable **host** cost is
**(iv) commit+upkeep + snap + tape + readout**.

## Confirmation measurement (trace-only instrumentation)

Added `snap_us` and `tape_us` to the existing `MLX_QWEN_MTP_TRACE` round line
(engine `feature/prompt-lev-campaign`, gated on the pre-existing `traceRounds`
flag — no behavioral change). Fresh steady-state k2 decode, essay-1024,
441 sane rounds, MISS cache, greedy.

**Provenance:** binary `3d5bc118…` (mtime 2026-09-18T04:34:35), server
`306fece`, engine `cfd6df5`; metallib `b57de586…` (Cmlx `1f8e74e`, Xcode 26.6).
Run: `benchmarks/results/lev-e-draftselect/` (`stderr.log`, `run2.log`,
`provenance.txt`).

### Per-phase medians (µs → ms)

| quantity | SCH1 ledger | RUN2 (fresh) |
|---|----:|----:|
| snap (rollback snapshot) | 0.084 | **0.060** |
| tape (M=3 concat) | ~0.003 | **0.003** |
| verify_build total | 3.940 | 2.653 |
| — of which walk-owned (snap+tape) | ~0.087 | **0.063** |
| — of which NOT walk (graph+ladder) | 3.854 | 2.590 |
| readout (top-2 copies) | 0.020 | **0.016** |
| **BOUND = (iv)+snap+tape+readout (mean)** | **~1.60** | **~0.81** |

### `commit_us` is bimodal by acceptance (the rollback path)

`commit` is cheap when both drafts are accepted and expensive on rejection
(the rollback/repair state update) — exactly the work a Metal draft-select
kernel would absorb:

| acc | SCH1 commit (n) | RUN2 commit (n) |
|---|----:|----:|
| 2 (both accepted) | 0.158 ms (226) | 0.124 ms (228) |
| 1 | 2.097 ms (105) | 1.194 ms (117) |
| 0 | 1.943 ms (126) | 1.132 ms (96) |

Acceptance-weighted **mean** commit: SCH1 1.368 ms, RUN2 0.630 ms. The ~2×
cross-time difference (same machine, ~4 h apart) is thermal; both are used.

## Bound

`BOUND = (commit+upkeep) + snap + tape + readout` (per-round, mean):
- **SCH1 (conservative):** 1.491 + 0.084 + 0.003 + 0.020 = **1.598 ms**
- **RUN2:** 0.730 + 0.060 + 0.003 + 0.016 = **0.809 ms**

**Both < 2 ms/round → KILL SWITCH → CLOSE.** Even the conservative bound
(1.60 ms) is 20 % under the threshold. The walk's share of `verify_build`
(snap+tape ≈ 0.06–0.09 ms) is negligible — the 3.85 ms `verify_build` remainder
is the 64-layer verify graph encode + async-ladder GPU wait, **not** the
draft/accept walk, so it is not addressable by a draft-select kernel.

## Verdict

**CLOSE the `qwen35DraftSelectKernel` lever.**
- The addressable host cost is ~0.8–1.6 ms/round (< 2 ms), an order of
  magnitude below a round (84.6 ms); a perfect Metal draft-select kernel saves
  at most ~1–2 % of the round.
- The dominant round cost is (i) kernel exec = 76.43 ms (90.6 %), not the walk.
- Corroborating negative result: the Swift compact draft-vocab walk (Task 1)
  was 2.05 % *slower* than the ≥3 % bar.
- **Category-error note for the record:** the "56 % FFN bottleneck" framing was
  wrong — FFN share is a *prefill* property; draft-select is a *decode* cost,
  and decode is 90.6 % kernel execution (the walk is ~1–2 %).

## Reproduce
```
# fresh confirmation (trace-only instrumentation already in the binary):
bash /tmp/lev_e_run.sh     # starts server, 1024-token essay, parses stderr.log
python3 -  # per-phase medians from the mtp-trace round lines
# SCH1 ledger: benchmarks/results/sch1-20260918-0041/
```
