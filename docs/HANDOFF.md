# HANDOFF — Draft-depth policy sweep (COMPLETE, NEGATIVE 2026-09-15)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**:
> `./scripts/agent-checkpoint.sh` ran successfully in both repositories
> (exit 0) and wrote `.dsh/last-agent-checkpoint` in each before this file
> was finalized. Markers recorded in the Checkpoint markers section below.

## Objective and acceptance criteria

Decide the speculative draft-depth policy for the current server: keep fixed
k = 2, change the fixed default to another measured depth, use a bounded
adaptive policy, or split interactive/throughput defaults. Benchmark-and-
decision task only: no kernel, head-topology, quantization, weight, or MLX
source changes. Success includes a defensible negative result.
**Verdict: KEEP fixed k = 2. No policy change.** k = 2 is the best measured
depth in all four tested modes (two fixtures × 1024/128-token regimes);
no candidate clears the positive-change bar (≥ 3 % sustained beyond noise,
bit-exact stream, no interactive regression); no mode exists in which any
other depth wins, so no adaptive policy or mode split is justified.

## Result (one live number per fact)

Single binary `e448b2e2…` (server `800218e` clean at start, engine
`97a9d85` clean, q4 head gate in every cell, 76 timed cells, all gates
green), medians of reps 2–6 of 6 interleaved rotated reps:

| mode | s | k1 | k2 | k3 |
|---|---|---|---|---|
| essay-1024 tok/s | 16.74 | 19.37 | **22.08** | 20.00 |
| specdec-1024 tok/s | 15.36 | 18.61 | **21.27** | 20.70 |
| essay-128 tok/s | 15.13 | 19.23 | **24.70** | 21.90 |
| specdec-128 tok/s | 16.77 | 19.99 | **25.04** | 18.77 |

k4/k6/k8 (Stage A cold recon, essay): 19.47 / 14.04 / 10.37 tok/s —
diminishing returns begin at k3, collapse by k6. k2 wins decode and wall
time in 4/4 modes; k1's cheaper round never pays (k2 beats k1 by 12–22 %);
no meaningful TTFT difference (identical prefill).

**Flag mapping (audited).** `--spec-draft-n-max` = per-round offer (default
3, 0 = serial, max 8); `QWEN_MTP_DRAFT_K` = pin; actual d = min(offer,
pin ?? 2); verify width M = d + 1. Every tested cell ran at exactly its
requested d (constant `depthDist` gate).

**New finding — near-tie width-family streams (pre-existing).** Greedy
streams are per verify-width family on specdec-800: M=1 `c70882fc…`,
M=2 `a3dfa862…`, M=3/M=4 `139acb9d…` (registered k=2 stream). First M=1 vs
M=3 divergence at completion token 989/1024 (reproduced twice); essay
families all agree for 1024 tokens; specdec 128-token prefixes all agree
(`6eb4c26a…`). Cause class: per-width accumulation-order ulps flipping a
rare near-tie argmax. Not a regression from this task; the registered k=2
product stream is unaffected; cross-width greedy identity remains an open
item in kernel/numerics scope. Sweep cells were gated per family (essay
against the single registered hash; specdec/128 against per-(mode, cell)
learned family hashes, k2 family asserted equal to the registered stream);
times unaffected and fully comparable.

## Raw results and files

- `benchmarks/DRAFT-DEPTH-POLICY.md` — protocol, flag map, near-tie table,
  raw-result locations, result tables, noise caveats, recommendation,
  explicit keep-default conclusion.
- `benchmarks/results/dpsweep-A-essay1024-{s,k1,k2,k3,k4,k6,k8}.jsonl`,
  `dpsweep-B-essay1024-{s,k1,k2,k3}.jsonl`,
  `dpsweep-C-{specdec1024,essay128,specdec128}-{s,k1,k2,k3}.jsonl` —
  full provenance per line (binary sha, heads, dirty flags, stream hash,
  depthDist, phase sums, per-phase averages).
- `benchmarks/run_dpsweep.sh` — new sweep runner (cells s/k1/k2/k3/k4/k6/k8,
  modes essay1024/specdec1024/essay128/specdec128, rotated interleaved reps,
  per-family hash gates, thermal snapshots).
- `benchmarks/run_cell.sh` — optional 7th argument `max-tokens` (default
  1024; existing invocations unchanged) + `ttlt` computed against it.
- `.tmp/dpsweep-*.log`, `.tmp/dpsweep-*.thermal` — run and thermal logs.
- `/tmp/dpdiag-*.json` — the serial/k2 specdec diagnostic pair used to
  locate the near-tie divergence (outside the repo, diagnostic only).

## Git state

- `qwen38-mtp-server` (this repo): branch `feature/draft-depth-policy`,
  docs/benchmarks-only changes, no library source, no rebuild, no new
  binary.
- `../mlx-swift-lm`: `main`, clean, untouched by this task.

## Commands / verification

- Sweep driver (all stages):
  `bash benchmarks/run_dpsweep.sh <A|B|C> <mode> "s k1 k2 k3" 6`
  (Stage A used `"s k1 k2 k3 k4 k6 k8" 1`).
- Gates: per-rep stream hash (registered / per-family), completion_tokens,
  finish_reason, phaseSumOK, constant depthDist, q4 head; any failure stops
  the stage. All stages completed 100 % gated on one binary.
- No builds, tests, or engine work were required or run; no parallel GPU
  work during timing (sequential cells, thermal snapshot per cell).

## Unresolved risks / caveats

- specdec-1024 k2-vs-k3 margin (2.7 %) is inside the within-session spread
  (~9–13 %, specdec session ran hotter); k2 is never worse and wins every
  other comparison, so the keep decision is unaffected.
- 128-token cells have wide per-rep spreads (13–41 %; decode window ~5–8 s
  against ~0.25 s fixed overhead); medians still rank k2 first in all four
  short modes.
- k4/k6/k8 have cold-recon measurements only (Stage A); they are decisively
  dominated and were correctly cut, so this is not a gap.
- Cross-session absolutes remain non-comparable (thermal); this session's
  k2 medians (22.08 / 21.27) differ from the registered headline session
  (21.89 / 23.29) — labels only, never conclusions.
- Cross-width greedy near-tie identity (see New finding) is open; it is not
  a policy input and did not block this decision.

## Do-not-repeat

- Do not re-derive the depth flag semantics from names — the audited mapping
  is in DRAFT-DEPTH-POLICY.md; `--spec-draft-n-max` is the OFFER, not the
  depth.
- Do not gate specdec-800 cells against a single global stream hash —
  families differ (near-tie property); use per-family hashes.
- Do not use FullBench newCache + prime + one decode timings for policy
  (retracted reset-structure artifact, prior session).
- Do not run `head -N` on the checkpoint script output (SIGPIPE aborts
  before the marker write); redirect to a file.
- Do not cite cross-session absolute tok/s as a conclusion.

## Next step

None — task complete (negative result, documented). If the near-tie width-
family streams item is ever pursued, that is a kernel/numerics task in
`mlx-swift-lm` with its own plan; the k = 2 product stream is unaffected.

## Checkpoint markers

- engine: 2026-09-15T00:52:00+01:00 (placeholder — replaced after fresh run)
- server: 2026-09-15T00:52:00+01:00 (placeholder — replaced after fresh run)

The fresh-checkpoint procedure completed.