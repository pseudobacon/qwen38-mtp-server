# LEV-B — Fused GDN A/B at 32K (verify fusion + prefill-fusion engagement)

**Run ID:** lev-b-fusedgdn
**Date:** 2026-09-18
**Method:** `run_cell.sh`, 32K (`longctx-32k.txt`, 32 780 prompt tokens),
greedy (temp 0, top_k 1), max_tokens 128, q4 head, cache MISS, 6 reps per
cell with rep 1 discarded (5 measured), interleaved A/B with rotating start,
`pmset -g therm` per rep. Same binary both cells; env-var/CLI-only difference.

**Provenance:** binary `29434ccf…`; server `306fece` + engine `cfd6df5`
(instrumentation uncommitted at measurement time); metallib `b57de586…`.
Data: `benchmarks/results/lev-b-fusedgdn/{A,B}.jsonl`, `thermal.log`.

## Setup notes (measured constraints)

- **32K dense prefill overflows the buffer** (51.6 GB transient > 27.1 GB
  allocatable) and is rejected; **32K requires `MLX_CHUNKED_PREFILL=1`**.
  Both cells run chunked, so the A/B isolates the GDN-fusion knob only.
- 32K decode is ~221 ms/step (long-KV attention + GDN recurrent), acc/step
  ~1.43, 53 rounds/128 tok.

## Result 1 — verify-width fusion `MLX_QWEN_FUSED_GDN` (the literal task)

| | A (off) | B (on) | Δ |
|---|----:|----:|----:|
| stepAvg (ms, med meas) | 221.51 | 219.07 | **−1.10 %** |
| tGraphBuildAvg (ms) | 89.06 | 85.99 | −3.07 ms |
| wall (s) | 116.84 | 116.69 | −0.13 % |
| stream hash | 1 distinct | 1 distinct | **A == B (bit-exact)** |
| phaseSumOK | 5/5 | 5/5 | — |

- **Bit-exact**: A and B produce the identical greedy stream in every rep.
- **Marginal speed win**: B saves ~3 ms/step of host graph-build (5 GDN-metal
  launches → 1), a **−1.1 % step reduction — within the per-rep noise band
  (≤ ~7 %)** and never ahead by a margin exceeding it.

**Verdict (verify fusion):** bit-exact, safe to default-ON, but the 32K win is
**marginal (≤ ~1 %, within noise)**. No quality cost. GO-as-default is
justified only for the launch-overhead saving; it is not a meaningful
performance lever at 32K.

## Result 2 — prefill fusion `MLX_QWEN_FUSED_GDN_PREFILL` (engagement flag)

The code gate is `S ≤ 4096` for the prefill variant. Measured at 32K:

| config (chunked) | stream hash | self-consistent |
|---|---|---|
| no `_PREFILL` (rep1, rep2) | `6576c099…` | yes |
| `_PREFILL=1` (rep1, rep2) | `f666254f…` | yes |

**Finding:** `MLX_QWEN_FUSED_GDN_PREFILL=1` **deterministically changes the 32K
greedy stream** (both configs self-consistent, the two configs differ). The
startup banner only reflects `MLX_QWEN_FUSED_GDN` (not `_PREFILL`), so it
reads "off" even with `_PREFILL=1` — do not use the banner as the `_PREFILL`
indicator. This implies the prefill fusion **engages on the long-prefill path
and is not bit-exact with the default at 32K** (its bit-exactness was verified
at verify widths 3–9, not at prefill widths).

**Verdict (prefill fusion):** **flag, not close.** It is a *correctness/quality*
item: enabling `MLX_QWEN_FUSED_GDN_PREFILL` changes the stream at 32K. It must
not be made a default until its prefill-width bit-exactness is proven (or it is
documented as a non-bit-exact approximation with a quality gate). Hand to the
follow-up campaign as a **verify-prefill-bit-exactness** task.

## Reproduce
```
bash /tmp/levab2.sh   # both matrices (lev-b-fusedgdn, lev-a-kvquant)
python3 /tmp/levab_analyze.py
```
