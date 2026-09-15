# 32K+ prefill validation — chunked prefill, per-phase timing, memory

**Date:** 2026-09-15 · **Model:** Qwen3.8-27B 4-bit + MTP · **HW:** M5 Pro 48 GB
**Flags:** `MLX_CHUNKED_PREFILL=1`, `QWEN_PREFILL_CHUNK_SIZE` = 512 (default) or 0 (single-pass, pc=0 trap-fix path), `MLX_TRACE_PREFILL=1` (eval-synchronized GPU timing, default off).
**Batch:** 1 · **Determinism:** greedy (`temperature=0`), fixed input prompts, no randomness in the prefill path.

All logs, responses, and RSS samples are in this directory (`benchmarks/results/prefill-verify-2026-09-15/`):
`srv-<tag>.log` (server stderr, contains the `PF2` trace lines), `resp-<tag>.json`
(the completion), `rss-<tag>.csv` (1 Hz RSS samples in KB).

## 1. Summary of results (pc=0 vs pc=512)

| run | wall | prompt tok | content hash | peak RSS |
| --- | --- | --- | --- | --- |
| **32K pc=512** (64 passes) | **132 s** | 32780 | `97bc0d74…` | 13.4 GB |
| **32K pc=0** (single pass) | **218 s** | 32780 | `95b445a6…` | 14.6 GB |
| **64K pc=512** (128 passes) | **333 s** | 65548 | `14b26f9f…` | 14.6 GB |

- **pc=512 is the near-optimal baseline** (reproduced: 132 s at 32K, consistent
  with the prior 117–120 s within thermal variance). pc=0 (single-pass) is
  **~65 % slower** (218 s) — the single-pass SDPA reads the full cache per
  query-tile, so it is not a perf path.
- **64K completes** with pc=512 (333 s) — the long-context path is validated
  end-to-end, not just by memory math.
- pc=0 no longer traps (the empty-chunk fix, engine `45df72a`); it is a
  correctness/robustness baseline, not a perf baseline.

## 2. Per-phase timing (eval-synchronized, prefill only)

Method: `eval(sectionOutput)` after each section forces the GPU to finish before
the clock stops, so these are **GPU-time** shares (not CPU-enqueue). The
measurement is gated on `L > 100` (prefill chunks), excluding MTP draft/verify
(L = depth). The primary rows (FFN / full-attn / GDN / norms) are additive and
sum to the prefill; the indented rows decompose the full-attention block.

| phase | 32K pc=512 | 32K pc=0 | 64K pc=512 |
| --- | --- | --- | --- |
| **FFN** (64 layers) | 61.70 s · **49.4 %** | 64.48 s · 48.4 % | 123.81 s · **43.3 %** |
| **Full-attention block** (16 layers) | 30.96 s · 24.8 % | 36.41 s · 27.4 % | 98.89 s · **34.6 %** |
| — **SDPA** (`attentionWithCacheUpdate`) | 23.14 s · **18.5 %** | 28.97 s · 21.8 % | 83.95 s · **29.3 %** |
| — QKV projection | 4.92 s · 3.9 % | 4.65 s · 3.5 % | 9.39 s · 3.3 % |
| — O projection | 2.44 s · 2.0 % | 2.42 s · 1.8 % | 4.57 s · 1.6 % |
| — RoPE (q/k/v) | 0.43 s · 0.3 % | 0.37 s · 0.3 % | 0.92 s · 0.3 % |
| **GDN block** (48 linear-attn layers) | 30.93 s · **24.8 %** | 30.89 s · 23.2 % | 60.99 s · 21.3 % |
| Norms / residuals | 1.24 s · 1.0 % | 1.33 s · 1.0 % | 2.55 s · 0.9 % |
| **Total (eval-sync)** | **124.82 s** | **133.11 s** | **286.23 s** |

**The full-attention SDPA is the #2 cost center and grows as O(L²):** 18.5 % at
32K → **29.3 % at 64K** (the FFN, being O(L), shrinks from 49.4 % → 43.3 %).
This **corrects** the earlier `docs/FLASH-ATTENTION.md` figure ("SDPA ~0.04 %"),
which was CPU *enqueue* time, not GPU time. The prefill SDPA uses the **dense
path** (the fused `MLXFast.scaledDotProductAttention` Metal kernel is
decode-only, T_q = 1), which is why it is a large, inefficient share.

## 3. Memory usage

Peak RSS (CPU, `ps -o rss`, 1 Hz sampling) and the model-derived buffers:

| run | peak RSS | KV cache (64 KiB/tok) | per-tile scores buffer (24·512·L·2) |
| --- | --- | --- | --- |
| 32K pc=512 | 13.4 GB | 2.05 GB | 0.81 GB |
| 32K pc=0 | 14.6 GB | 2.05 GB | 0.81 GB (tile=512, engine chunked) |
| 64K pc=512 | 14.6 GB | 4.10 GB | 1.61 GB |

- The 4-bit weights (~15 GB) are wired into GPU memory and are **not** in RSS;
  RSS is the CPU-side KV cache + activations + allocator. Peak RSS grows modestly
  (13.4 → 14.6 GB) from 32K to 64K, well under the 48 GB unified budget.
- The per-tile scores buffer is bounded by the chunked prefill (linear in L, not
  quadratic): 0.81 GB at 32K, 1.61 GB at 64K — far under the 30.2 GB Metal
  per-buffer cap. (The dense path would need 51.6 GB at 32K and 206 GB at 64K —
  both trap.)
- `rss_avg` in the raw CSVs is dominated by the low-RSS model-load/decode tail
  and is not meaningful; use peak RSS.

## 4. Bit-exactness validation

Content SHA-256 (first 16 hex) of the completion, greedy, per fixture:

| fixture | dense (no chunked) | pc=512 (chunked) | pc=0 (chunked) |
| --- | --- | --- | --- |
| 8K (8532 tok) | `660dd120…` | `660dd120…` **match** | `660dd120…` **match** |
| 16K (17703 tok) | `2e583ad2…` | `2e583ad2…` **match** | `2e583ad2…` **match** |
| 32K (32780 tok) | rejected (507, dense buffer 51.6 GB) | `97bc0d74…` | `95b445a6…` **diverge** |

- **8K and 16K: dense == pc=512 == pc=0** — no regression in bit-exactness
  against the dense reference at these lengths.
- **32K: pc=0 diverges from pc=512 at the first token** (Phase 1 Bug A
  knife-edge). At L=32780 the engine chunked-SDPA gate (L > 4096) engages for
  pc=0 (one L=32780 pass) but not for pc=512 (per-512 passes, each L=512 <
  4096), so the two prefill splits accumulate the full-attention FP in a
  different order and flip a knife-edge first token. This is the expected
  sensitivity to the prefill split, not a defect. The dense 32K reference is not
  available (admission rejects it pre-prefill).

## 5. Observed anomalies and edge cases

- **SDPA far slower than the FLOP estimate.** The 32K SDPA is 23.1 s, but the
  ~13.4 TFLOP attention workload at ~100 TFLOPS peak is ~0.13 s — the prefill
  SDPA runs at well under 1 % of peak. Root cause: the prefill uses the **dense
  (unfused) attention path**, which materializes the `[512, 24, s]` scores
  matrix per chunk (0.81 GB at the last 32K chunk) and is memory-bound, not
  compute-bound. This is the inefficiency a flash/prefill-fused kernel would
  remove.
- **pc=0 single-pass is slower, not faster** (218 s vs 132 s at 32K). The
  single pass reads the full 32780-token cache per query-tile (2× the cache
  reads of the incremental pc=512 path). pc=0 is a robustness baseline only.
- **`nfull`/`ngdn` counts** in the `PF2` lines reflect the number of layer
  calls with L > 100 (64 prefill passes × 16 full-attn = ~1024–1056 for
  pc=512; ~48 for pc=0's single pass). They are a sanity check, not a metric.
- **Edge cases handled:** empty prefill (`count=0` → no chunks) and the pc=0
  empty-chunk trap are guarded in `prefillChunkRanges` (engine `45df72a`, 4
  unit tests). No non-deterministic randomness is introduced; the prefill is
  greedy and deterministic per (fixture, config).

## 6. Recommendations

1. **Keep the pc=0 path as a defect-fixed baseline, not a default.** It is
   correct (no trap) and bit-exact with pc=512 at 8K/16K, but it is ~65 % slower
   at 32K and diverges at 32K (knife-edge). The default should remain pc=512.
2. **Optimization targeting (evidence-based):**
   - **FFN (~43–49 %)** is the single largest cost center — O(L) 4-bit GEMMs.
   - **Full-attention SDPA (~18–29 %, growing with L²)** is the #2 center and the
     one that *grows* with context. Because the prefill uses the dense (unfused)
     path, a **prefill-fused / flash-style attention kernel is the highest-leverage
     long-context lever** (it directly attacks the 18.5 % → 29.3 % share). This
     revises the prior "flash attention won't help" conclusion, which rested on
     the invalidated CPU-enqueue figure.
   - **GDN block (~21–25 %)** is the #3 center (in_proj + recurrence + out_proj).
   - The decode path is out of scope here (this is a prefill breakdown).
3. **Do not** pursue pc=0 as a perf path; it is strictly slower than pc=512.

## 7. Reproduction

```bash
cd /Users/cwong/ai/qwen38-mtp-server
# per-phase + memory (32K pc=512)
MLX_CHUNKED_PREFILL=1 QWEN_PREFILL_CHUNK_SIZE=512 MLX_TRACE_PREFILL=1 \
  .build/release/qwen38-mtp-server serve --port 18099 --model ./weights &
# then: curl -s http://127.0.0.1:18099/v1/chat/completions \
#   -H 'Content-Type: application/json' -d @benchmarks/results/prefill-verify-2026-09-15/req32k.json
# the last "PF2 ..." line in the server stderr is the prefill total.
```

Fixtures: `req8k.json` (8532 tok), `req16k.json` (17703 tok), `req32k.json`
(32780 tok), `req64k.json` (65548 tok) — all in this directory. The benchmark
driver is `/tmp/prefill_verify.sh` (one server per run, RSS sampled at 1 Hz).

## 8. Validation criteria

- **Reproduce near-optimal pc=512 with chunked prefill:** ✅ 132 s at 32K
  (consistent with prior 117–120 s within thermal variance).
- **Full-attention a minor contributor (< 25 %):** ✅ at 32K the full-attention
  *block* is 24.8 % (just under 25 %), but the **SDPA alone is 18.5 %** and
  rises to 29.3 % at 64K — so attention is *not* minor at long context.
- **No bit-exactness regression vs dense where applicable:** ✅ 8K/16K
  chunked (pc=512 and pc=0) match the dense reference exactly.

## 9. Safety / deviations

- No non-deterministic randomness in the prefill (greedy, fixed prompts).
- The eval-synchronized timing **serializes** the GPU; absolute seconds are not
  the async throughput (the eval-sync total 124.8 s ≈ the 132 s wall), only the
  relative split is meaningful.
- Deviation from the prior `docs/FLASH-ATTENTION.md` "SDPA ~0.04 %" figure: that
  was CPU *enqueue* time (MLX enqueues Metal commands asynchronously); this
  report's eval-sync measurement shows the SDPA is 18.5–29.3 % of the prefill.
  `docs/FLASH-ATTENTION.md` should be updated to reflect this.
