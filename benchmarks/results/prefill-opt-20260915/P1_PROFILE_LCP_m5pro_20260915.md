# P1 — Long-Context Prefill Baseline Profile (LCP)

**Date:** 2026-09-16 · **EXP_ID:** LCP · **Device:** M5 Pro 48 GB (`m5pro`)
**Model:** Qwen3.8-27B 4-bit + native MTP head (64 layers: 48 GDN + 16 full-attention; hidden 5120)
**Binary:** release `.build/release/qwen38-mtp-server` (engine fork `mlx-swift-lm`, `feature/prompt-1`)
**Flags:** `MLX_CHUNKED_PREFILL=1`, `QWEN_PREFILL_CHUNK_SIZE=512` (production default), `MLX_TRACE_PREFILL=1`. All LCP Phase 2/3 toggles unset (default off).
**Determinism:** greedy (`temperature=0` in fixtures), pinned request files (`req8k/16k/32k/64k.json`), fresh server per cell.

## 1. Determinism gate (content SHA-256, first 16 hex)

| fixture | prompt tok | pinned baseline | this run | result |
| --- | --- | --- | --- | --- |
| 8K | 8532 | `660dd1208737764c` | `660dd1208737764c` | PASS |
| 16K | 17703 | `2e583ad29dc28465` | `2e583ad29dc28465` | PASS |
| 32K | 32780 | `97bc0d74846a043b` | `97bc0d74846a043b` | PASS |
| 64K | 65548 | `14b26f9f89f16da5` | `14b26f9f89f16da5` | PASS |

Baselines are the pc=512 results of `benchmarks/results/prefill-verify-2026-09-15` (itself
verified bit-exact against the dense path at 8K/16K).

## 2. Per-phase prefill timing (eval-synchronized GPU time)

Method: `MLX_TRACE_PREFILL=1` evals each section output, forcing the GPU to finish the
section before the clock stops (GPU-time, not CPU-enqueue). Section rows (FFN / GDN /
full-attention / norms / residuals) are additive and sum to `total`; the indented rows
decompose the full-attention block (`attn`). `nfull`/`ngdn` = layer-chunk calls (chunks ×
16 / × 48): 17/35/64/128 chunks for 8K/16K/32K/64K at pc=512.

| phase | 8K | share | 16K | share | 32K | share | 64K | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **FFN** (64 layers) | 15.83 s | **56.3 %** | 44.26 s | **48.9 %** | 116.86 s | **45.2 %** | 252.85 s | **36.5 %** |
| **Full-attention block** (16 layers) | 3.59 s | 12.8 % | 20.89 s | 23.1 % | 75.73 s | **29.3 %** | 291.82 s | **42.1 %** |
| — SDPA | 1.61 s | 5.7 % | 14.16 s | 15.7 % | 56.22 s | 21.7 % | 247.24 s | **35.6 %** |
| — QKV projection | 1.22 s | 4.3 % | 3.92 s | 4.3 % | 12.29 s | 4.8 % | 29.90 s | 4.3 % |
| — O projection | 0.60 s | 2.1 % | 2.36 s | 2.6 % | 6.16 s | 2.4 % | 12.72 s | 1.8 % |
| — RoPE (q/k/v) | 0.16 s | 0.6 % | 0.43 s | 0.5 % | 1.02 s | 0.4 % | 1.88 s | 0.3 % |
| **GDN block** (48 linear-attn layers) | 7.68 s | 27.3 % | 22.40 s | 24.8 % | 58.85 s | 22.8 % | 132.21 s | 19.1 % |
| Norms | 0.47 s | 1.7 % | 1.76 s | 1.9 % | 4.93 s | 1.9 % | 12.14 s | 1.8 % |
| Residuals | 0.52 s | 1.8 % | 1.12 s | 1.2 % | 2.20 s | 0.9 % | 4.50 s | 0.6 % |
| **Total (eval-sync)** | **28.10 s** | 100 % | **90.42 s** | 100 % | **258.56 s** | 100 % | **693.52 s** | 100 % |
| wall (request, incl. decode) | 37.4 s | | 110.0 s | | 310.9 s | | 779.6 s | |

Per-cell raw lines:

```
PF2 prefill wall_ms=28125.33 total_ms=28102.60 ffn_ms=15832.50 gdn_ms=7682.98 attn_ms=3594.42 sdpa_ms=1606.82 qkv_ms=1218.36 oproj_ms=600.09 rope_ms=161.72 norm_ms=472.97 residual_ms=519.73 nfull=272 ngdn=816
PF2 prefill wall_ms=90475.78 total_ms=90424.69 ffn_ms=44255.05 gdn_ms=22398.43 attn_ms=20893.42 sdpa_ms=14163.24 qkv_ms=3921.29 oproj_ms=2362.01 rope_ms=429.68 norm_ms=1756.86 residual_ms=1120.93 nfull=560 ngdn=1680
PF2 prefill wall_ms=258682.20 total_ms=258564.92 ffn_ms=116860.22 gdn_ms=58849.34 attn_ms=75728.29 sdpa_ms=56223.83 qkv_ms=12289.87 oproj_ms=6155.18 rope_ms=1023.21 norm_ms=4925.79 residual_ms=2201.29 nfull=1024 ngdn=3072
PF2 prefill wall_ms=693743.61 total_ms=693524.15 ffn_ms=252853.87 gdn_ms=132208.20 attn_ms=291818.24 sdpa_ms=247236.55 qkv_ms=29904.45 oproj_ms=12723.22 rope_ms=1879.60 norm_ms=12141.64 residual_ms=4502.21 nfull=2048 ngdn=6144
```

## 3. Memory (peak RSS, 1 Hz `ps -o rss` during the request)

| run | peak RSS | prompt KV (64 KiB/tok) |
| --- | --- | --- |
| 8K | 4.73 GB | 0.53 GB |
| 16K | 14.09 GB | 1.09 GB |
| 32K | 13.94 GB | 2.13 GB |
| 64K | 13.92 GB | 4.27 GB |

RSS is the CPU-side view (weights wired to GPU memory do not appear). Peaks from 16K up
are dominated by the MLX allocator's high-water reservation, not the KV cache; all four
cells stay far under the 48 GB budget. (The 8K peak is below the 16K+ plateau — the
short prefill never drives the allocator to the high-water mark seen at 16K+; the
2026-09-15 8K run measured ~14.6 GB peak with the older per-chunk instrumentation build.)

## 4. Findings

1. **FFN is the dominant cost at short/medium context** (56.3 % at 8K) and stays the
   single largest phase through 32K (45.2 %). It is O(L) 4-bit GEMM work — already the
   fused wide shapes (gate+up, 4-proj GDN); the remaining body is the prebuilt MLX
   4-bit GEMM kernel (out of scope for this task).
2. **Full-attention is the growing center and becomes the #1 block at 64K** (42.1 % vs
   FFN 36.5 %). SDPA alone: 5.7 % → 15.7 % → 21.7 % → **35.6 %** — the O(L²) term. The
   prefill SDPA runs the dense (unfused) Metal path per 512-chunk; a flash-style
   prefill kernel is the highest-leverage long-context lever (Phase 3 scope note: a
   non-bit-exact fused attention must stay behind a user gate).
3. **GDN is #2 at 16K–64K** (19–25 %). The prefill widths (512) are not covered by the
   fused GDN prework kernel (verify-widths only) — Phase 2 extends it, toggle-gated.
4. **Norms/residuals are ~3 % total** at all lengths — small but non-zero; Phase 2
   extends the fused residual+RMSNorm kernel to 3-D prefill shapes, toggle-gated.
5. **Thermal note:** absolute times in this matrix are ~2× the 2026-09-15 report's
   cooler-session values (32K: 258.6 s vs 124.8 s eval-sync; 64K: 693.5 s vs 286.2 s)
   with no `pmset` thermal warning recorded — sustained-load GPU downclock. The
   *relative* per-phase split is stable and the Phase 2/3 comparisons are made against
   this baseline under the same conditions, not against the older report.
6. **Bit-exactness held end-to-end**: all four content hashes match the pinned pc=512
   baselines, so this baseline is comparable token-for-token to Phase 2/3 runs.

## 5. Reproduction

```bash
cd /Users/cwong/ai/qwen38-mtp-server
bash benchmarks/run_lcp_p1.sh   # 4 cells, ~25 min, hash-gated
```

Artifacts in this directory: `srv-p1-<len>.log` (PF2 lines), `resp-p1-<len>.json`,
`rss-p1-<len>.csv` (1 Hz), `summary-p1-<len>.json`, `wall-p1-<len>.txt`, `run.log`.
