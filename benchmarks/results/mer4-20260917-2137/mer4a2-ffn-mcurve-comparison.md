# MER4A.2 — FFN M-curve (gateup + down_proj), v0.32.2 re-baseline

Sustained no-sync, DVFS-fair interleaved (`--ffn-pair`), 2 clean reps at the decision
region (M=512/1024), 1 rep each at M=2048/4096/8192. Binary `a4b56e2b…`.

| M | gateup µs/tok | down µs/tok | **FFN µs/tok** | (v0.31.1 FFN µs/tok) |
|---|---------------|-------------|----------------|----------------------|
| 512  | 3.440 | **9.818** | **13.258** | 11.261 |
| 1024 | 4.444 | **2.857** | **7.301** ← min | 6.426 |
| 2048 | 4.980 | 2.860 | 7.839 | 7.528 |
| 4096 | 5.120 | 2.812 | 7.932 | 8.190 |
| 8192 | 5.388 | 2.933 | 8.321 | 7.966 |

(FFN µs/tok at M=512/1024 = mean of the 2 clean reps; M=2048/4096/8192 = single rep.
v0.31.1 column = registered `mcp1-curve.md`, labels only.)

## Finding

The M=512 down_proj anomaly **persists** on v0.32.2: down is **9.818 µs/tok** at M=512 vs
**2.857** at M=1024 (3.4×). The upstream split-K quantized matmul did **NOT** close the
small-M/large-K tiling anomaly. The per-token FFN cost still has an interior minimum at
**M=1024** (7.301), rising to ~7.8–8.3 at M=2048–8192 — the same shape as v0.31.1.

**MER4C consequence:** the pc=2048 default is **NOT stale** (the M-curve did not flatten
materially). The pc sweep is NOT triggered.

Cross-session note: v0.32.2 per-token costs are slightly higher than the v0.31.1 labels at
matched M (e.g. M=1024 down 2.407→2.857); these are cross-session absolutes (never
conclusions) — the decision-relevant ratio (M=512/M=1024 ≈ 3.4×) is unchanged.
