# HANDOFF — 2026-09-14 (end of W2/W3 session)

## Objective

The five-work-item queue is complete through W3 (W4 is now triggered and is the
next task). **Next task: W4 — 4-bit quantization of the MTP draft head** (BF16
849.4 MB → ~300 MB, group 64), A/B vs the BF16 head under the determinism gate.
Acceptance criteria for W4: (i) acceptance rate at or above the BF16 head's
within noise, (ii) tok/s improvement (expected ≈ 14 ms/round at d=2, ~13 % of
the step), (iii) committed-stream hash unchanged (`949b9423…` essay /
`139acb9d…` specdec), (iv) decision record + `benchmarks/results/W4-ab.jsonl`.

## Branch / commit / uncommitted state (verify independently before editing)

- Engine `/Users/cwong/ai/mlx-swift-lm`: `main`, expected clean. New this
  session: `headbench` tool (`Libraries/HeadBench/main.swift` + `Package.swift`
  product entry), committed on `feature/w2-headbench` and merged to main
  (branch deleted). Build: `swift build -c release --product headbench` →
  `.build/arm64-apple-macosx/release/headbench`.
- Server `/Users/cwong/ai/qwen38-mtp-server`: `main`, expected clean. New this
  session: `benchmarks/PROFILE.md` (W2 deliverable), `benchmarks/run_w2trace.sh`,
  `benchmarks/run_w3sweep.sh`, `benchmarks/results/w3-*.jsonl` (sweep + probes),
  `run_cell.sh` depth/acc collection, `progress.md` W2/W3 sections + refreshed
  headline, this HANDOFF. Committed on `feature/w2-w3-sweep` and merged to main
  (branch deleted).
- If this file's state disagrees with `git status` / `git log`, Git wins —
  update this file first, then proceed.

## Important files

- `benchmarks/PROFILE.md` — W2 profile: GPU 98.2 % busy; headbench measured
  per-round head-family cost (d=1 24.66 ms, d=2 45.94 ms, d=3 67.47 ms,
  d=4 88.89 ms); raw-matmul floor 7.36 ms/head forward @ ~115 GB/s; per-round
  tEval decomposition (backbone ~55–59 ms / head 40.6 ms / verify lm_head
  5.4 ms at d=2); **W4 trigger MET**; levers ranked.
- `progress.md` — W2 section, W3 section (incl. the depth-≥5 bug report),
  refreshed headline (essay 19.90 / specdec 25.44 tok/s, default k=3),
  decisions on record, open items (bug #1, W4 #2).
- `benchmarks/results/w3-draftk-essay-k{1,2,3,4}.jsonl` — sweep records (last
  5 records per file = measured reps r2–r6; rep 1 discarded as warmup).
- `benchmarks/results/w3-probe-*.jsonl` — bug provenance (k5, k6-qmvoff,
  k6-nofuse, specdec k2/k2-rerun/k3/k4).
- `benchmarks/run_cell.sh` — per-round `d`/`acc` collection added (5th arg
  `EXTRA_ARGS` for CLI flags); outputs `depthDist`, `depthAvg`, `accAvg`.
- Engine `Libraries/HeadBench/main.swift` — reusable for W4 (already loads the
  head and times per-round calls; re-point `--head` at a 4-bit head directory).
- MTP head: `mtp-head/pinned/model.safetensors` (849.4 MB BF16: `fc` + one
  full-attention `layers.0` + `mlp` + norms; final vocab projection uses the
  shared 4-bit `lm_head`, not in the head file).

## Decisions and evidence (this session)

- **W2 done.** GPU-bound (98.2 % busy); CPU not the bottleneck; `tGraphBuild`
  ~10.5 ms/round is the dominant CPU cost. No per-kernel GPU time (shader
  profiler not enabled in the Metal System Trace template — stated limitation).
  Head cost measured directly with `headbench` instead of in-graph timing
  (per the W2 no-instrumentation rule).
- **W3 done (valid subset k=1..4).** Essay: k1 19.25, **k2 21.26** (optimum,
  +6.8 % over default), k3 default 19.90, k4 16.26 tok/s; all bit-exact
  `949b9423…`, phaseSumOK, materialized=0. k=2 is **conditional**: it fails the
  specdec determinism gate.
- **Correctness bug (top open item, report-only):** verify width ≥ 6 commits
  wrong tokens from round 1. Boundary matches qL·gqa > 32 (width 5 = 30 clean,
  width 6 = 36 diverges). Root-cause hypothesis: the "exactness chunk" split
  (two ≤5-row SDPA calls for 6..9-row causal verify) specified in the engine's
  design comments (`Qwen38MTPBlockSession.swift` ~1428–1436, ~1996–1998;
  SDPA warm-up comment ~755–770) does not exist in
  `MLXLMCommon/AttentionUtils.swift`. QMV verify and fused QKV/SwiGLU
  exonerated by exclusion probes (all three probes give the identical
  deterministic wrong stream `da7bb159…` on essay). **Do not fix this bug in a
  W4 task** — separate task; `--spec-draft-n-max`/`QWEN_MTP_DRAFT_K` above 4
  is broken until then.
- **Specdec confirmation:** k1 PASS, k2 FAIL (deterministic wrong stream
  `06882d85…`, divergence ~94.5 %), k3 default PASS, k4 FAIL (`48728382…`,
  ~95.2 %). k=2 and k=4 specdec failures have a different signature (late
  knife-edge flips, deterministic per prompt, different wrong stream from the
  essay bug). Only k=1 and default k=3 are certifiably bit-exact on both
  fixtures; k=3 stays the recommended configuration.
- **lm_head size corrected:** 4-bit U32 [248320, 640] = **635.7 MB** payload +
  79.5 MB scales/biases (the earlier 317.8 MB note was wrong).
- **Headline refreshed:** essay 19.90 tok/s (k3, 5 reps), specdec 25.44 tok/s
  (k3, single probe rep) — both default config, binary `11a8e61e…`.
- **W4 trigger MET** (measured 24.66/45.94 ms/round at d=1/2 vs ~5 ms gate).
  W4 safety: the head emits only draft proposals; greedy target verify commits,
  so head precision affects acceptance, not the committed stream. `lm_head`
  remains report-only (already 4-bit; quantizing it changes committed tokens).

## Commands / tests and results

- `benchmarks/run_w2trace.sh` — trace capture: 102 rounds, 256 committed,
  acceptance 1.5294, avgStepMs 105.9, decodeSeconds 10.81; routed 8176
  dispatches, materialized 0. Trace `/tmp/w2-profile.trace` (~44 MB) +
  `/tmp/w2-*.xml` exports.
- `.build/arm64-apple-macosx/release/headbench --model <backbone> --head
  mtp-head/pinned --drafts 1,2,3,4 --warmup 10 --timed 50` — measured
  head-family cost table (PROFILE.md §2); `--raw` — raw matmul reference
  (PROFILE.md §3). Inputs are float32 (no bf16 initializer in this release);
  weights are the model's real bf16/4-bit — negligible GPU cost difference.
- `benchmarks/run_w3sweep.sh` — 4 cells × 6 reps, 24 GATE PASS, all
  phaseSumOK; plus specdec single-rep probes. Thermal logs in `.tmp/w3-*.log`.
- Determinism hashes (pinned): essay-1024 `949b9423bd85…`; specdec-800
  `139acb9d30fe…`. Wrong streams (bug evidence): essay k5/k6 `da7bb159…`;
  specdec k2 `06882d85…`, k4 `48728382…`.

## Unresolved risks

- The depth-≥5 bug means any `--spec-draft-n-max > 4` deployment would commit
  wrong tokens. The server default (adaptive, cap 7, offered 3) is safe; the
  knob surface above 4 is not.
- The ~9 ms head layer-structure cost is an estimate (no per-kernel profile);
  W4's expected 14 ms/round saving is a projection, to be confirmed by the A/B.
- specdec k=2/k=4 divergence mechanism is not yet root-caused (signature
  differs from the essay bug); do not conflate the two.
- `/tmp/benchvenv` must exist for `run_cell.sh` (recreate if missing after
  reboot: `python3 -m venv /tmp/benchvenv`); `/tmp/w2-profile.trace` may be
  gone after reboot — PROFILE.md is the durable record.

## Do not repeat

- Do not run the draft-k sweep or any timing cell with `QWEN_MTP_DRAFT_K ≥ 5`
or `--spec-draft-n-max` above 4 — wrong streams (bug).
- Do not fix the SDPA exactness chunk as part of W4 (separate task; greedy
  semantics — AGENTS stop condition).
- Do not claim per-kernel GPU breakdown from the existing trace (shader
  profiler not enabled); do not add in-graph timing instrumentation (W2 rule).
- Do not present the pre-Item D B1/B2 headline rows (17.55 / 18.92 tok/s) as
  current; they are superseded (stale rows kept as provenance only).
- Do not use the k=6/k8 sweep records as performance data (wrong stream).

## One exact next step

**W4:** quantize the pinned MTP head to 4-bit group 64 (write a quantized
`model.safetensors` to a new head directory, e.g. `mtp-head/q4/`, in the
backbone's exact U32 group-64 layout — `fc`, `q`, `k`, `v`, `o`, `gate`, `up`,
`down` + norms stay bf16 or follow the backbone convention; verify the head
loads via `Qwen38MTPHeadAttachment`), then A/B: BF16 head vs 4-bit head,
default config, essay-1024 + specdec-800, 6 reps each, gates: acceptance rate,
tok/s, committed-stream hash. Use `run_cell.sh` (5th arg for extra env) and
`headbench` for the per-round head-cost confirmation. Deliverable:
`benchmarks/results/W4-ab.jsonl` + decision record in `progress.md`.

## Fresh-checkpoint procedure

Checkpoint status: **pending — record below after running
`bash /Users/cwong/ai/qwen38-mlx-server/scripts/agent-checkpoint.sh` from each
repo.**
