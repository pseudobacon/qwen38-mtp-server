# tEval profile — W2 (2026-09-14)

Scope: bucket the ~105–125 ms/step `tEval` of the default-config greedy MTP
server (QMV verify ON, fusions ON, compiled decode ON, adaptive draft depth,
code default 3), answer (a) draft-head forward + draft-logits cost per round,
(b) verify-logits-head cost per round, (c) where the next fused-kernel work
should go, and provide the W4 trigger measurement. In-session evidence only;
cross-session absolutes are labels, not conclusions.

**Environment.** MacBook M5 Pro, 48 GB unified memory. Server release binary
SHA-256 `11a8e61e3e0ba82b76a8c0d37e82a10fa591c9ea1624d0063cc53ea8489a5f9e`
(engine `a5f102f` + server W1 evidence commits), essay-1024 fixture (SHA-256
`7ed683f8…`), greedy `temperature 0`, `enable_thinking: false`, `max_tokens
1024`, port 18099, fresh server per capture, no parallel builds/tests during
measurement.

**Evidence files.**
- `benchmarks/run_w2trace.sh` — trace-capture driver (server launch, readyz
  poll, `xctrace record --attach`, one request, SIGINT finalize, cleanup).
- `/tmp/w2-profile.trace` (~44 MB, Metal System Trace) + exports
  `/tmp/w2-*.xml` (TOC, GPU intervals, signposts, app events, execution
  points, shader-profiler tables, CPU time-profile).
- engine `Libraries/HeadBench` (product `headbench`) + `/tmp/headbench-raw.txt`
  (raw matmul reference) — per-call head-family measurements.
- W3 sweep records `benchmarks/results/w3-draftk-essay-k{1,2,3,4}.jsonl`
  (cross-check of the backbone share).

## 1. External Metal System Trace

One greedy essay-1024 request under `xctrace record --template 'Metal System
Trace' --attach <pid>`: 256 committed tokens, 102 rounds, acceptance 1.5294,
avgStepMs 105.9, decodeSeconds 10.81, wall 11.05 s. Routed dispatches
`2:192, 3:3552, 4:6432` (8 176 ≈ 80.2/round), fallback `1:96, 38:96, 512:192`,
materialized 0.

- **GPU utilization 98.2 %** (10 817.13 ms busy over a 11 010.09 ms span) —
  decode is GPU-bound; the GPU is saturated, not starved by dispatch.
- CPU time-profile: ~2.56 s CPU over ~11 s wall; top consumers
  qwen38-mtp-server 14.1 % (361 ms), malloc 4.8 %, AGX (Metal driver) 4.2 %,
  swiftCore 3.7 %. The CPU is **not** the bottleneck; the step-trace's
  `tGraphBuild` (~10.5 ms/round, ~10 %) is the dominant CPU-side cost.
- **Limitation (stated up front):** this trace template does not enable the
  Metal shader profiler, so it contains **no per-kernel GPU intervals** — the
  `metal-gpu-intervals` rows are command-buffer level only, the execution-point
  function IDs are object addresses (not shader IDs), and the
  `metal-shader-profiler-intervals` table is empty. Per-kernel bucketing below
  therefore comes from (i) direct micro-benchmark measurement of the head
  family (§2) and (ii) weight-bytes ÷ measured-kernel-bandwidth estimates for
  the backbone (§4), each labeled as measured or estimated.

## 2. `headbench` — measured per-round head-family cost

`headbench` loads the backbone + pinned MTP head exactly the way the server
does (`LLMModelFactory.loadContainer` + `Qwen38MTPHeadAttachment.
withHeadAttached`, no-op tokenizer — the bench never converts text) and times
the exact per-round session calls on synthetic hidden states: fresh head
cache per rep (one per round in the session), graph-build + sync-eval per rep,
10 warmup reps then 50 timed reps, median reported. Inputs are float32 (this
mlx-swift release has no bf16 array initializer); the head's `Linear` layers
cast to the bf16 weight dtype internally, so the measured GPU cost matches
production within noise. Release build, no parallel work.

Measured per call (median of 50, release build):

| call | median ms | min ms | max ms |
|---|---|---|---|
| `flush(F=1)` — `mtpHeadHiddenForward` over the committed-history flush row, fresh head cache | 16.38 | 16.06 | 17.72 |
| `step(F=1)` — draft step `mtpHeadHiddenForward` on the flushed cache (flush completed outside the timed window) | 16.26 | 16.04 | 16.59 |
| `proj1(M=1)` — `draftTokenID` (shared 4-bit `lm_head` over 1 row) | 3.95 | 3.90 | 4.11 |
| `verifyM(M=2)` — `applyFinalNorm` + `applyLMHead` over 2 verify rows | 4.33 | 4.29 | 4.45 |
| `verifyM(M=3)` | 5.41 | 5.37 | 8.81 |
| `verifyM(M=4)` | 6.73 | 6.67 | 6.86 |
| `verifyM(M=5)` | 7.94 | 7.89 | 8.29 |

Per-round head-family cost (flush F = 1; the session's per-round structure is
1 flush + (d−1) draft steps + d draft projections + 1 verify projection over
1+d rows):

| draft depth d | flush | (d−1)·step | d·proj1 | verify(1+d) | **total ms/round** |
|---|---|---|---|---|---|
| 1 | 16.38 | 0.00 | 3.95 | 4.33 | **24.66** |
| 2 | 16.38 | 16.26 | 7.89 | 5.41 | **45.94** |
| 3 | 16.38 | 32.52 | 11.84 | 6.73 | **67.47** |
| 4 | 16.38 | 48.78 | 15.78 | 7.94 | **88.89** |

**Answer to (a):** the draft-head forward + draft-logits cost is **24.7 ms/round
at d=1 and 45.9 ms/round at d=2** (measured) — the head family is the single
largest non-backbone cost in the step and scales ~16.3 ms per extra draft
token.

## 3. Raw matmul reference (`headbench --raw`, no model load)

Plain matmul at the head's exact weight shapes, M = 1 (GEMV regime), same
timing discipline:

| shape (x × wᵀ, bf16) | median ms | effective GB/s |
|---|---|---|
| fc [1,10240] × [5120,10240] (104.9 MB) | 0.94 | 111 |
| q [1,5120] × [12288,5120] (125.8 MB; gated attention, 2×24×256) | 1.08 | 116 |
| kv [1,5120] × [2048,5120] (21.0 MB) | 0.31 | 67 |
| o [1,6144] × [5120,6144] (62.9 MB) | 0.66 | 96 |
| gate [1,5120] × [17408,5120] (178.3 MB) | 1.46 | 122 |
| up [1,5120] × [17408,5120] (178.3 MB) | 1.46 | 122 |
| down [1,17408] × [5120,17408] (178.3 MB) | 1.44 | 124 |
| **SUM — one head forward's matmuls (849.4 MB)** | **7.36** | **115** |
| 4-bit ref [1,5120] × [17408,5120] group-32 (backbone path; 44.6 MB) | 0.42 | 105 |

Reading: the raw-matmul floor of one head forward is 7.36 ms (stock bf16 GEMV
at ~115 GB/s); the measured head forward is 16.38 ms. The **~9 ms gap is the
head's layer structure** — gated-attention split/mul, RoPE, q/k RMSNorms, the
eager (non-fused) SwiGLU path (the load-time fusion summary reports
`head swiGLU 0 qkv 0` for the BF16 head), the per-rep fresh KV-cache update,
and ~25 kernel launches. That gap is an **estimate** (no per-kernel profile was
taken — see §1 limitation); it is not a single identifiable kernel.

## 4. Per-round `tEval` decomposition (d = 2 class, tEval ≈ 105 ms)

Weight traffic computed from the safetensors headers (4-bit group-64 U32
layout): backbone body per forward **13.70 GB** (16 full-attention layers ×
209.4 MB + 48 GDN layers × 215.7 MB); `lm_head` 4-bit payload **635.7 MB**
(U32 [248320, 640]) + 79.5 MB scales/biases; `embed_tokens` likewise 635.7 MB
(row gather negligible); grand total 15.13 GB.

**Correction:** an earlier progress.md note listed `lm_head` as 317.8 MB —
wrong. U32 [248320, 640] is 635.7 MB (the 317.8 figure used a 2-byte
element size). The measured `verifyM` times below are unaffected (they are
time, not bytes).

Decomposition of one d=2 round (tEval ≈ 105 ms), each row labeled measured or
estimated:

| bucket | ms/round | share | basis |
|---|---|---|---|
| Backbone verify forward, 3 rows (4-bit QMV) | ~55–59 | ~55 % | **estimated**: 14.42 GB ÷ 200–275 GB/s; implied bandwidth back-computed per depth from the W3 sweep (stepAvg − measured head family): k1 215, k2 247, k3 273, k4 200 GB/s — all inside the measured 4-bit kernel band (qmvbench sustained 310–355 GB/s; serialized 142–186 GB/s) |
| MTP head family (flush + 1 step + 2 proj1) | 40.6 | ~39 % | **measured in isolation** (§2; per-call-sync → flush-contaminated upper bound, see §7) |
| Verify `lm_head` over 3 rows (4-bit) | 5.4 | ~5 % | **measured** (§2, `verifyM(M=3)`) |
| Gaps, launches, dispatch | remainder | ~1–2 % | trace: 98.2 % GPU busy leaves little idle |

The cross-check is tight at all four sweep depths (implied backbone bandwidth
200–273 GB/s, no depth falls outside the measured 4-bit band), which supports
the decomposition and the measured head numbers.

## 5. Answers to the W2 questions

**(a) Draft-head forward + draft-logits cost per round** — measured, §2:
24.66 ms at d=1, 45.94 ms at d=2, 67.47 ms at d=3, 88.89 ms at d=4. One head
forward is 16.38 ms (849 MB bf16; raw-matmul floor 7.36 ms at ~115 GB/s + ~9 ms
layer-structure estimate).

**(b) Verify-logits-head cost per round** — measured, §2: 4.33 ms (M=2) to
7.94 ms (M=5); grows ~1.2 ms per extra verify row. The `lm_head` is already
4-bit in the checkpoint (635.7 MB payload, group 64) — there is **no further
quantization headroom** on the logits path; report-only per the W4 scope.

**(c) Where the fused-kernel work should go next**, in lever size order:
1. **4-bit quantize the MTP head (W4).** The head's matmul floor is 7.36 ms
   per forward at bf16; at 4-bit (≈ half the bytes, same group-64 style as the
   backbone) the floor drops to ≈ 1.9–2.5 ms per forward, while the ~9 ms
   layer-structure cost is unchanged. Net: ≈ 4.5 ms saved per head forward →
   ≈ 9 ms/round at d=1, ≈ 14 ms/round at d=2 (~13 % of the d=2 step). Memory:
   849 MB → ≈ 300 MB. This is the largest remaining per-step lever outside the
   backbone's weight traffic. — **DONE (W4, 2026-09-14): KEEP, default ON.**
   Measured in-pipeline: essay 19.63 → 21.32 tok/s (+8.6 %), specdec 22.11 →
   23.45 tok/s (+6.1 %), 24/24 reps bit-exact; the win was carried by
   `tGraphBuild` (−7.2/−7.8 ms), and the isolated head-body collapse
   (16.38 → 1.52 ms) did **not** transfer 1:1 to `tEval` (within in-session
   noise) — see §7 for the flush-contamination explanation.
2. **Extend the SwiGLU/QKV fusions to the head.** The load-time fusion summary
   reports `head swiGLU 0 qkv 0`: the head's MLP runs the eager
gate/up/silu/mul/down chain (5 ops) and its QKV the unfused projections.
   Fusing them removes launches and one matmul; expected gain is small
   (launch-level, not bandwidth-level) but it is the same pattern already
   proven bit-exact on the backbone.
3. **Backbone 4-bit bandwidth headroom.** Implied in-pipeline bandwidth is
   200–273 GB/s vs qmvbench's sustained 310–355 GB/s at the same shapes — a
   possible ~10–20 % on the backbone share, but it requires kernel work on the
   incumbent 4-bit GEMV, not a new fused kernel.
4. **`tGraphBuild` ≈ 10.5 ms/round (~10 %)** — CPU-side graph construction;
   secondary, and it is the phase a mid-graph flush would poison (W1 lesson).

## 6. W4 trigger decision

Gate: proceed if the W2 profile shows ≥ ~5 ms/round on the draft-head path.
**Measured: 24.66 ms/round (d=1) and 45.94 ms/round (d=2) — TRIGGERED, by a
factor of 5–9×.** W4 (4-bit MTP head, group 64, A/B vs the BF16 head) is
cleared to proceed. `lm_head` remains report-only (already 4-bit; quantizing
it further would change committed tokens and requires explicit sign-off).

**Safety note for W4:** the MTP head emits only draft proposals; the target's
greedy verify commits, so head precision affects acceptance rate, not the
committed stream (the W3 sweep's k=1..4 bit-exactness under different draft
depths is the evidence that the committed stream is invariant to draft
quality). The W4 A/B gates on (i) acceptance rate, (ii) tok/s, and
(iii) the committed-stream hash staying `949b9423…` / `139acb9d…`.

## 7. Post-W4 amendment — head-family numbers are flush-contaminated upper bounds (2026-09-14)

The §2 head-forward numbers (16.38 / 16.26 ms) and the per-round head-family
bucket (24.66 / 45.94 / 67.47 / 88.89 ms; "40.6 ms ≈ 39 %" in §2's
decomposition) are **isolated measurements with per-rep graph build +
synchronous `eval`** — one GPU pipeline flush between every timed call. By
the Item D lesson (the `asData` flush artifact), a per-call-sync isolated
number is an **upper bound on in-pipeline cost, not the in-pipeline cost**:
in the real session the head forward is one of ~25 kernel groups inside a
continuously running pipeline with no per-call drain.

W4's in-pipeline A/B (2026-09-14, k=3, 6 reps per state, both fixtures)
provides the correction: quantizing the head body (849 → 238.9 MB; 8 linears
to 4-bit) collapsed the **isolated** head forward **16.38 → 1.52 ms
(10.8×)**, yet the **in-pipeline** `tEval` moved only within in-session noise
(−4.01 / −0.12 ms on essay/specdec) while the total step improved 8–11 ms/
round carried by `tGraphBuild` (−7.2 / −7.8 ms — the smaller fused graph).
If the 40.6 ms/round bucket were the true in-pipeline cost, the q4 state
would have dropped `tEval` by tens of ms; it did not.

Consequences (recorded, not re-measured here):

- The §2 "MTP head family 40.6 ms (~39 %)" bucket is an **upper bound**;
  the true in-pipeline head cost is **unestablished** and must be measured
  flush-free in-pipeline before the head-structure GPU kernel item is scoped.
- The "~14 ms/round at d=2" estimate in §5(c) was not realized in-pipeline
  on the GPU side; the measured W4 win was `tGraphBuild` plus q4
  acceptance/routing effects. The committed-stream gate held bit-exact
  (24/24 reps); q4 acceptance ran ~2–4 % lower per step (1.3759/1.6693 vs
  1.4061/1.6974).
- The §6 lever list's item (1) is **done** (W4, KEEP, default ON); the
  GPU-side head lever is **unproven** until a corrected in-pipeline
  measurement exists.

Nothing in §2–§6 is deleted — all numbers remain valid as labeled isolated
measurements; this section labels their in-pipeline applicability.
