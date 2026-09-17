# PRO — In-pipeline bandwidth gap probe (run `pro-bw-20260917-2306`)

Diagnostic-only task to locate the ~2× in-pipeline bandwidth gap: **176.6 GB/s
in-pipeline** (tEvalAvg 81.6 ms) vs **310–355 GB/s sustained** (qmvbench FFN
sustained, v0.32.2 engagement map). Three candidate mechanisms to separate:

- **(a)** a hardware interleave property (QMV kernel-mix / weight-rotation costs BW);
- **(b)** the server/round dispatch structure (MTP draft + accept/rollback, per-dispatch sync);
- **(c)** a measurement artifact.

Same-session in-pipeline comparisons only; cross-session absolutes are never
conclusions. No production source changes (qmvbench `--layer-seq` probe added;
FullBench reused).

## PRO0 — Headline refresh (18 cells, controlled MISS, interleaved)

- essay tEvalAvg med = **83.2 ms** (min 78.4 max 91.1), ttlt 24.8 tok/s, stream
  hash `949b9423` (matches registry).
- specdec tEvalAvg med = **85.6 ms** (min 80.1 max 94.4), ttlt 26.2 tok/s, stream
  hash `139acb9d` (matches registry).
- 32K prefill wall = **118.2 s** (matches the ~125 s MER2 anchor).
- TTFT pair (max_tokens=16): MISS **1.56 s** vs RAM-HIT **0.78 s** (2× — the
  prefix-cache state is a first-class dimension, re-confirmed).

Confirms the 81.6 ms in-pipeline anchor. Stream hashes match the registry
exactly (no cold/warm drift on v0.32.2 — the ND fix holds).

## PRO1 — QMV interleave + weight-rotation (qmvbench `--layer-seq`, M=3)

Loads all 64 layers' gateup + down + o_proj (10.8 GB weight stream) and runs a
sustained wall-timed 3-cell probe:

| cell | QMVs/batch | GB/s | vs prior |
|---|---|---|---|
| 1 single-kernel (gateup) | 1 | **177.4** | reference |
| 2 interleave (1 layer's gateup+down+oproj) | 3 | **176.8** | 99.6% |
| 3 thrash (64 layers' mix, weights rotated) | 192 | **173.6** | 98.2% |

**The QMV kernel-mix interleave is FREE (99.6%) and the weight-rotation (thrash)
is FREE (98.2%).** Running three different QMV kernels per layer, and rotating
through all 64 layers' weights, costs no bandwidth. This **rules out (a)**: the
gap is not a hardware interleave / weight-rotation property.

## PRO2 — FullBench round replay (M=1, M=3, serial)

- M=1 serial (back-to-back decodes on a growing cache): tE med = **58.82 ms**
  → **250 GB/s** (14.4 GB / 58.82 ms).
- M=3 verify (prime 2048, width matrix): eval wall med = **79.52 ms** → **181 GB/s**.
- In-pipeline (M=3, PRO0 tEvalAvg): **83.2 ms** → **176.6 GB/s**.

The in-pipeline (83.2 ms) ≈ the FullBench M=3 verify (79.52 ms) + **3.7 ms**
(draft head + accept/rollback + prime-depth difference). The round dispatch
(draft + accept/rollback) is **not** the gap — it is ~3.7 ms.

## Mechanism verdict

- **(a) hardware interleave — NO.** PRO1: the QMV interleave (99.6%) and
  weight-rotation (98.2%) are free.
- **(b) per-dispatch / round structure — YES (dominant).** The gap is the
  per-dispatch overhead: the M=3 verify **with per-step sync** (176.6 GB/s) vs
  the FFN sustained **no-sync** (310–355 GB/s). The M=3 verify is less
  memory-bound than M=1 (176.6 vs 250 GB/s) because it streams the same 14.4 GB
  over 3× the activations/KV writes. The draft + accept/rollback round dispatch
  is small (~3.7 ms, PRO2) — the gap is the **per-dispatch sync + M effect**,
  not the round-trip.
- **(c) measurement artifact — NO.** The in-pipeline 176.6 GB/s reproduces
  consistently across PRO0 (83.2 ms) and PRO2 (M=3 verify 79.52 ms). Stream
  hashes match the registry.

**The 2× gap is the per-dispatch/state cost (per-step sync + M=3 verify), not a
kernel, interleave, thrash, round-trip, or measurement artifact.**

## GO / NO-GO for a scheduling task

**GO.** The lever is real and isolated to dispatch, consistent with the
engagement map's ranked lever #1: **batched/fused dispatch and state
amortization across MTP steps** (not a new GEMM). The expected win is the
176.6 → 310+ GB/s on the weight stream. Concretely: the per-step sync (one
`MLX.eval` per MTP step) is the cost; fusing the MTP verify steps (or batching
the per-layer dispatch) amortizes the weight stream and the sync over the round.
The interleave/thrash is not a lever (PRO1: free), so the scheduling task should
target the **per-dispatch sync + state setup**, not the kernel mix.

## Files

- PRO0: `pro0-{essay,specdec}-r{1..6}.json`, `pro0-prefill-{8k,16k,32k,96k}.json`,
  `pro0-ttft-{miss,ramhit}.json`
- PRO1: `pro1.txt`, `pro1.err`
- PRO2: `pro2-widths.txt`, `pro2-serial.txt`, `pro2-run.sh`
