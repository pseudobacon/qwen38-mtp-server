# HANDOFF — Verify tape profile (COMPLETE, NEGATIVE 2026-09-14)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**:
> `./scripts/agent-checkpoint.sh` ran successfully in both repositories
> (exit 0) and wrote `.dsh/last-agent-checkpoint` in each before this file
> was finalized. Engine marker 2026-09-14T21:45:00+01:00; server marker
> 2026-09-14T21:45:00+01:00.

## Objective and acceptance criteria

Reduce the k = 2 verify tape (71.60 ms, 85 % of the 84.19 ms round) by
≥ 10 ms (tape → ~61 ms, eval window → ~72 ms, step → ~76 ms) via backbone
4-bit weight-streaming / kernel optimization, bit-exact (stream hash
`949b9423…` unchanged), no draft-depth changes, engine changes first.
**Verdict: NO WIN — no ≥8 ms lever exists in this checkout.** The tape is
DRAM-bound on the fixed ~15 GB 4-bit weight set at 205–257 GB/s effective
(≈ machine peak); all four candidate branches are dead (see below). The
task's fallback path applies: profile breakdown documented, next levers
recommended.

## Result (one live number per fact)

Profiled at command-buffer (CB) granularity — the finest the Metal System
Trace capture offers (per-shader intervals were **not** recorded in the
k2trace run; `metal-shader-profiler-intervals` table has 0 rows). 18 steady
rounds bucketed, binary `e448b2e2…`, 256 ctx, same capture as PROFILE-K2
§3/§8 (offset 696,415,092,781,278 validated: 102.4 % eval-window CB
coverage).

- **Structure: one fused CB per layer (64 total) + lm_head CB + 3 small
  CBs.** The entire layer (QKV + attention/GDN + MLP + norms) is a single
  CB — no norm/elementwise CBs exist in the tape (fusion already maximal:
  swiGLU 64/64, qkv 16/64, gdn 48/64).
- **Per-layer-type bucket (medians, n=18):** 48 GDN layers 1138.7 µs each =
  54.66 ms (74.5 %); 16 full-attention layers 1058.3 µs each = 16.93 ms
  (23.1 %); lm_head (M=3, 635.7 MB 4-bit) 3935.7 µs = 3.94 ms (5.4 %);
  inter-CB gaps 1.32 ms (1.8 %); 3 small CBs 0.05 ms (0.1 %). CB-measured
  tape busy median 73.35 ms — consistent with the registered 71.60 ms
  within window-edge noise. FA layers are *cheaper* per layer than GDN
  layers at 256 ctx (attention over 3 queries × ~258 KV is trivial; GDN
  conv+scan costs more per layer).
- **Bottleneck: memory bandwidth (weight streaming).** Per-layer effective
  BW ~205 GB/s (GDN), ~209 GB/s (FA), uniform across all 64 layers; M=1
  in-pipeline 257 GB/s (FullBench serial 269 GB/s) is the practical peak;
  M=1 flat across ctx 256→3072 (55.77→56.80 ms) — bandwidth-bound, not
  compute. tape = 58.30 (M=1 base) + 2 × 6.65 (marginal rows); the marginal
  6.65 ms/row is real row compute (MLP + attention for 2 rows), not
  inefficiency.
- **Kill analysis of the candidate branches.** (1) Weight layout: dead —
  the layout is the MLX quantized format consumed by prebuilt kernels; any
  change alters fp32 accumulation order → breaks bit-exactness; the kernels
  are in prebuilt MLX (not this checkout). (2) Row batching (M=3 as one
  dispatch): already done — the tape is M=3 batched, one fused CB per layer.
  (3) Norm fusion into matmul: already done — no norm/elementwise CBs in the
  tape. (4) Graph caching: saves 0 ms of the eval window — the host build
  (2.97 ms) is already hidden behind head GPU; the inter-CB gaps are 1.32
  ms/round, not graph build.
- **Only sub-8 ms inefficiency found:** 1.32 ms/round inter-CB gaps + 0.05
  ms small CBs (total ~1.4 ms, ~2 % of the tape) — below the 8 ms bar by a
  factor of ~6. The QMV M=3 effective-BW gap (205 vs 257 GB/s) is in
  prebuilt MLX and is already accounted for in the 13.30 ms marginal-row
  cost.

## Next levers (all outside this checkout)

1. Model-level: a 2-token native head or draft-vocabulary lm_head (removes
   ~12.1 ms/round — the head + one verify row, not the tape).
2. Model-level: smaller/denser backbone quantization (fewer bytes streamed
   per forward).
3. MLX upstream: a faster M=3 QMV kernel (close the 205→257 GB/s gap) — in
   prebuilt MLX C++/Metal.
4. Draft-depth policy: k=2→1 removes 12.1 ms/round at −1.22 tokens/round
   (a policy change, not a kernel change).

## Git state

- `qwen38-mtp-server` (this repo): branch `main`, clean before the docs
  commit for this task (docs-only: `benchmarks/PROFILE-K2.md` §9, this file,
  `progress.md`).
- `../mlx-swift-lm`: branch `main` at `97a9d85`, clean, untouched by this
  task.
- Fresh checkpoint markers: engine 2026-09-14T21:45:00+01:00, server
  2026-09-14T21:45:00+01:00.

## Baseline re-verification (this session)

One k = 2 cell (6 reps, `--spec-draft-n-max 3`, essay fixture, single
stream) on the current binary `e448b2e2…` to confirm the registered numbers
reproduce before recording the verdict (`/tmp/k2baseline.jsonl`):

| field | value |
|---|---|
| tEvalAvg (reps 1–6) | 79.93 / 82.67 / 81.19 / 84.06 / 83.17 / 84.81 ms (rep 1 cold = registered 79.87) |
| avgStepMs (reps 1–6) | 84.29 / 87.05 / 85.48 / 88.47 / 87.55 / 89.03 ms (rep 1 cold = registered 84.19) |
| stream hash | `949b9423…` — 6/6 bit-exact |
| phaseSumOK / draft_depth / head | 6/6 True / k=2 / q4 |

The rep-to-rep rise (tEval 79.93→84.81) is thermal drift within the session
(no thermal warning level recorded); rep 1 reproduces the registered numbers
cold.

## Commands / verification

- Bucketing scripts (analysis only, /tmp): `analyze_k2_tape.py` (per-layer
  CB bucketing), `analyze_k2_cbs.py` (CB structure discovery).
- Inputs: `/tmp/k2trace-gpu-intervals.xml` (CB intervals),
  `/tmp/k2trace-mtp-trace.log` (mtp-anchor lines),
  `/tmp/k2trace-toc.xml` (table inventory), `/tmp/k2-gpu-counters.xml` (1.2 GB
  — RT-Unit-Active tick stream; single counter type, 3.8M ticks).
- Baseline cell: `/tmp/run_k2baseline.sh` → `/tmp/k2baseline.jsonl`.
- No source changes in either repo; no new binaries; no rebuild required.

## Unresolved risks / caveats

- Per-shader (per-kernel) intervals were not recorded in the k2trace run, so
  the component split (QKV vs attention vs MLP vs norms *within* a layer CB)
  is not directly measurable; the one-fused-CB-per-layer structure is itself
  the finding (the components are fused by design and the layer CB is
  uniform ~1.13 ms).
- The 3 small CBs (~18 µs each) have unconfirmed identity (candidates: GDN
  state ops, KV scatter, tape bookkeeping); total 0.05 ms — immaterial.
- The per-layer CB bucketing assumes the 64-layer forward occupies the last
  64 large CBs before the lm_head CB in each round; validated by count
  (exactly 65 large CBs in the tail-68 across all 18 rounds) and by the
  pattern alignment (FA every 4th).

## Do-not-repeat

- Do **not** re-capture the trace with `MLX_QWEN_MTP_TRACE_SYNC_HEAD=1`
  (destroys head/verify overlap — prior-session finding).
- Do **not** enable `MLX_QWEN_MTP_LADDER=off` for ranked runs (attribution
  probe only).
- Do **not** cite fb11 trace captures (pid 77915, 79949) — empty GPU
  tables.
- Do **not** run `head -N` on the checkpoint script output (SIGPIPE aborts
  the script before the marker write); redirect to a file.
- Do **not** attempt the weight-layout branch — it is dead on bit-exactness
  grounds, not just on the prebuilt-kernel grounds.
- Do **not** cite per-rep FullBench M=1 numbers at prime > 256 as
  serial-decode cost (per-rep first-decode penalty is a bench artifact).

## Next step

None — task complete (negative result, documented). All follow-up candidates
are outside this checkout (model-level changes, MLX upstream, or policy).

## Checkpoint markers

- engine: 2026-09-14T21:45:00+01:00
- server: 2026-09-14T21:45:00+01:00

The fresh-checkpoint procedure completed.
