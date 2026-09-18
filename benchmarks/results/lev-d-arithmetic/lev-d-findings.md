# LEV-D — Flash + large-pc compounding model (zero-code arithmetic)

**Run ID:** lev-d-arithmetic
**Date:** 2026-09-18
**Decision:** **CREDIBLE — hand the bar forward to LEV-J. Do NOT close flash.**
The GO bar (c_flash ≤ ~630 µs/tok @32K, worst-case FFN) is 3–35× *above* a realistic
Metal flash kernel's per-token cost. Large-pc (pc≥8192) + flash beats the pc=2048
dense incumbent at both 32K and 64K by a wide margin.

---

## Model

`total(pc) = FFN(pc) + GDN(pc) + SDPA(pc) + other`

Each term is a per-token cost × L. The only unknown is the flash SDPA per-token cost
`c_flash`. All other inputs are registered in-session measurements.

| term | source | pc-dependence |
|---|---|---|
| FFN | MCP2 32K per-phase (512/1024/2048) + MCP1/MER4A.2 M-curve (→8192) | O(L); per-chunk overhead ↓ with pc, GEMM per-token ↑ with pc (M-curve) |
| GDN | MCP2 32K per-phase | O(L); ~flat at large pc (per-chunk overhead ↓) |
| SDPA | dense: MCP2 32K measured + prefill-verify 64K (29.3% share); flash: **unknown `c_flash`** | dense O(L²); flash O(L) amortized (KV-bound) |
| other | MCP2 32K per-phase (QKV+O+RoPE+norm+resid) | O(L); ~flat |

## Inputs (in-session, single-session where stated)

**32K per-token (µs/tok), MCP2 single session (`mcp-20260917`):**

| pc | FFN | GDN | SDPA(dense) | other | total |
|----|----:|----:|----:|----:|----:|
| 512  | 2209.7 | 1090.6 | 841.5 | 420.2 | 4562.0 |
| 1024 | 2052.8 | 1056.3 | 801.5 | 344.8 | 4255.3 |
| 2048 | 1991.5 |  946.0 | 753.9 | 293.4 | 3984.8 |

**FFN @ pc=8192 (extrapolated, µs/tok):** two bounds bracket the answer —
- `A+B/pc` fit on the 3 end-to-end points (per-chunk overhead keeps dropping at 8192):
  **1915.4** (−3.8% vs 2048 — 8192 is *better*).
- M-curve ratio (isolated GEMM per-token rises 2048→8192, ×1.058 v0.31.1 / ×1.061 v0.32.2):
  **2107–2114** (+5.8% vs 2048 — 8192 is *worse*).

The end-to-end FFN at 8192 is uncertain between these (the per-chunk-overhead drop and
the GEMM per-token rise compete). The non-SDPA delta (8192 vs 2048) is therefore
**−76.1 µs/tok (optimistic) to +122.5 µs/tok (pessimistic)**; GDN and other are ~flat.

## GO bar (solve `total(8192)+flash = total(2048)+dense`)

`c_flash < c_dense(2048) − nonSDPA_delta`

| L | c_dense(2048) | nonSDPA delta (opt→pess) | **GO bar c_flash** |
|---|----:|----:|----:|
| 32K | 753.9 µs/tok | −76.1 → +122.5 | **631.5 – 830.0 µs/tok** |
| 64K | ~1328 µs/tok (O(L²) 2× + measured×pc-ratio) | −76.1 → +122.5 | **1205 – 1404 µs/tok** |

**Binding (strict) bar = 32K worst-case: `c_flash ≤ ~630 µs/tok`.** A flash kernel that
meets this at 32K automatically beats the incumbent at 64K too (the 64K bar is ~2× looser).

## Credibility of the bar

FLOP floor for 32K SDPA ≈ 130 ms = **4.09 µs/tok**. The flash KV-read floor (each K token
read once per Q-block, 64 KiB/token, ~200 GB/s in-pipeline BW):

| pc | KV read | time @200 GB/s | per-token |
|----|----:|----:|----:|
| 8192 | 3.0 GiB | 16.1 ms | 0.49 µs/tok |
| 2048 | 15.0 GiB | 80.5 ms | 2.46 µs/tok |

A credible Metal flash kernel runs within ~5–50× the FLOP/KV floor → **c_flash ≈ 20–204
µs/tok @32K** (≈ 40–400 @64K). The GO bar (~630 @32K / ~1300 @64K) is **3–35× above** a
credible kernel. Even a very inefficient flash (100× floor) would clear the 32K bar.

**Verdict:** the bar is comfortably within a credible Metal kernel for head_dim 256 /
GQA-6. **Do not close flash.** Hand the bar forward.

## Memory side (admission headroom)

Flash removes the per-chunk dense scores buffer (O(pc·L) → O(L)):

| config | scores buffer |
|---|----:|
| 64K, pc=2048 (incumbent) | **6.29 GB** (MCP2) |
| 64K, pc=8192 (would-be) | ~25.2 GB (4×) — near the 30 GB Metal cap, no headroom |
| 64K, flash (any pc) | O(L), ~GB-scale, **buffer removed** |

So flash is what *enables* pc≥8192 (and longer context toward the 262 K ceiling) without
blowing the transient-buffer budget. Large-pc+flash is also more KV-efficient than
small-pc+flash (3.0 vs 15.0 GiB KV read @32K).

## Notes / caveats
- 64K `c_dense(2048)` is estimated (no single-session 64K pc=2048 per-phase exists); the
  32K bar is the binding constraint and is measured, so the verdict rests on the 32K number.
- Cross-session absolutes (MCP1 vs MCP2 vs prefill-verify) are used only for *ratios*
  and *shapes*, never as in-session conclusions, per protocol.
- Zero code: pure arithmetic from registered data. `model.py` in this directory is the
  reproducible computation.

## Reproduce
```
python3 benchmarks/results/lev-d-arithmetic/model.py
```
