# 32K prefill per-phase GPU breakdown

**Method:** eval-synchronized wall-clock timing, `MLX_CHUNKED_PREFILL=1`,
`QWEN_PREFILL_CHUNK_SIZE=512`, one 32K request (32780 prompt tokens).

This is a **profiling report, not an optimization**. It answers "where does the
32K prefill time actually go?" with a GPU-time breakdown (not CPU enqueue time).

## Method

A plain `CFAbsoluteTimeGetCurrent()` bracket around a section measures CPU
*enqueue* time (MLX enqueues Metal commands asynchronously), **not** GPU time. To
measure GPU time, each section was followed by `eval(sectionOutput)`, which forces
the GPU to finish that section before the clock stops. This serializes the GPU, so
the **absolute** numbers are not representative of the async throughput — only the
**relative split** between cost centers is. The measurement is env-gated
(`MLX_TRACE_PREFILL=1`, default off) and was removed after profiling (engine left
byte-identical to main).

The accumulation was gated on `x.dim(1) > 100` so it captures the prefill chunks
(L=512) and excludes the MTP draft/verify passes (L=depth=3).

## Breakdown (eval-synchronized, prefill only)

| Cost center | time | share |
| --- | --- | --- |
| **FFN** (`mlp`, 64 layers) | 71.2 s | **~50%** |
| **GDN block** (`linearAttn`, 48 layers) | 38.7 s | **~27%** |
| **Full-attention block** (`selfAttn`, 16 layers) | 33.7 s | **~23%** |
| Norms / residuals / embedding / lm_head | ≈ 0 | ≈ 0% |
| **Total (eval-sync)** | **143.6 s** | 100% |

Cross-check (two runs, both agree within thermal variance):

| run | full | gdn | ffn | FFN % | GDN % | full-attn % |
| --- | --- | --- | --- | --- | --- | --- |
| gate L>1 | 29.3 s | 32.0 s | 65.3 s | 51.6% | 25.3% | 23.2% |
| gate L>100 | 33.7 s | 38.7 s | 71.2 s | 49.6% | 27.0% | 23.5% |

The relative split is stable; the absolute seconds drift with thermals
(request wall time varied 124 s ↔ 165 s across runs for the same 32K input).

## Top cost centers

1. **FFN (~50%)** — the largest cost center by far. The MLP
   (`gate_up` + SiLU + `down`), O(L) 4-bit GEMMs.
2. **GDN block (~27%)** — the 48 linear-attention (gated delta net) layers:
   `in_proj` + the recurrent scan + `out_proj`.
3. **Full-attention block (~23%)** — the 16 full-attention layers: QKV + SDPA + O.
   Because the `prefillChunkSize=512` default makes each chunk's attention
   `O(512²)`, this block is dominated by the QKV/O **projections** (O(L) GEMMs),
   not the SDPA.

## Caveats

- **Full-attention sub-breakdown (QKV / SDPA / O) was not captured.** The
  full-attention module dispatches to the compiled fast path
  (`MLXHardwareInfo.isCompiledDecodeSupported` → `forwardFastPath`), not the
  eager `projectPreRope` / `attentionWithCacheUpdate` / `mergeHeadsAndProject`
  chain, so sub-instrumenting the eager chain measured 0. The SDPA's share is
  therefore not directly measured here; it is `O(512²)` per chunk and small
  relative to the O(L) projections in the block.
- **GDN sub-breakdown (in_proj / recurrence / out_proj) not separated** — the
  block is reported as a whole.
- The eval-synchronized total (143.6 s) is a serialization artifact; the async
  prefill wall time is lower (≈ 117–120 s). Only the percentages are meaningful.

## Implication for optimization targeting

If prefill speedup is pursued, the **FFN (~50%)** is the single largest lever,
followed by the **GDN block (~27%)** and the **full-attention block (~23%)** —
all O(L) 4-bit GEMM / scan work, none of it the O(L²) attention. Flash attention
or any attention-only optimization cannot move the total meaningfully; the gain
would have to come from the FFN / GDN / projection GEMMs. This is a separate,
larger optimization surface and is **not** pursued here (profiling only).
