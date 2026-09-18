# LEV-A — Quantized KV cache A/B: fp16 vs affine8 at 32K

**Run ID:** lev-a-kvquant
**Date:** 2026-09-18
**Method:** `run_cell.sh`, 32K (`longctx-32k.txt`, 32 780 prompt tokens),
greedy, max_tokens 128, q4 head, cache MISS, `--kv-tail-size 1024` (default),
6 reps per cell (rep 1 discarded, 5 measured), interleaved A/B with rotating
start, `pmset -g therm` per rep. Both cells chunked (required for 32K).

- **A (baseline):** `--kv-scheme fp16` (the current default; 16/16 bits).
- **B (lever):** `--kv-scheme affine8` (8/8 bits; last 1024 tokens kept fp16
  by the tail, the rest affine-quantized).

**Hard rule honored:** q4 KV (`affine4`) is **not** a default candidate
(documented catastrophic quality failures) — this A/B is fp16 vs affine8 only.

**Provenance:** binary `29434ccf…`; server `306fece` + engine `cfd6df5`;
metallib `b57de586…`. Data: `benchmarks/results/lev-a-kvquant/{A,B}.jsonl`,
`thermal.log`, `div/{fp16,affine8}.json`.

## Result

| | A (fp16) | B (affine8) | Δ |
|---|----:|----:|----:|
| stepAvg (ms, med meas) | 221.75 | 257.57 | **+16.15 %** |
| tGraphBuildAvg (ms) | 88.38 | 124.77 | +36.39 ms |
| wall (s) | 116.60 | 118.89 | +1.97 % |
| acc/step | 1.434 | 1.500 | +0.066 |
| stream hash | 1 distinct | 1 distinct | **A ≠ B (0 overlap)** |
| phaseSumOK | 5/5 | 5/5 | — |

**affine8 is slower, not faster.** The +36 ms/step of host graph-build is the
per-step KV quantize/dequantize traffic; the +16 % step penalty outweighs any
latency benefit at 32K. The only upside is memory: at 32K, affine8 stores
roughly half the KV bytes of fp16 (~1 GB vs ~2 GB) — a headroom/admission
benefit, not a speed one.

### Quality (first-divergence + coherence, mt=256)

- Both completions are **coherent, well-formed** (a structured analysis of the
  `mlx-swift-lm` library); no repetition loop or garbage in either
  (most-common word ≤ 5 %).
- **First divergence at char 165**: fp16 `…testing utilities.` vs affine8
  `…testing helpers.` — a **minor synonym swap**, not a catastrophic failure.
  The streams then follow the same structure. So affine8 is a *reasonable but
  different* greedy completion — a normal quantization-induced divergence,
  far milder than the q4 catastrophic failures the hard rule excludes.

## Verdict

**NO-GO as a default.** `affine8` is **+16 % slower per step** and **changes the
stream** (coherent but divergent from the fp16 reference). It trades speed and
exact-match reproducibility for KV memory headroom.
- **Keep `fp16` as the default** (fastest, reference-exact).
- `affine8` is a **legitimate memory-recovery option** for headroom-constrained
  operation (longer context / more concurrent sessions) where the operator
  accepts the +16 % step cost and a divergent-but-coherent stream — not a
  default.
- Consistent with the q4 hard rule: q4 (worse quality) is excluded; affine8
  (milder, coherent) is viable only as an explicit memory option.

## Reproduce
```
bash /tmp/levab2.sh    # leva matrix (A=fp16, B=affine8), then:
# first-divergence: two mt=256 cells -> div/{fp16,affine8}.json
```
