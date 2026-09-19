# FB0 — Flash + pc compounding model (re-derived with CURRENT measured numbers)

**Run ID:** lev-j / fb0
**Date:** 2026-09-19
**Status:** PREDICTIONS PUBLISHED (FD must confirm; never reverse-engineer).

The LEV-D bar (c_dense=753.9 @32K) was derived from an older MLX and is STALE.
This model is rebuilt with the in-session current-MLX measurements, which are the
only numbers FD's AB will be judged against.

## Inputs (all in-session, current MLX)

**32K pc=2048 full prefill (PF2 trace, this run, L=32780 tok, greedy):**

| phase | wall | % of total |
|-------|-----:|-----------:|
| **total** | **110.4 s** | 100 % |
| FFN | 57.5 s | 52.1 % |
| GDN | 26.0 s | 23.6 % |
| SDPA (the term flash replaces) | 17.0 s | 15.4 % |
| other (QKV+O+RoPE+norm+resid) | 9.8 s | 8.9 % |

This matches the 2026-09-15 registered 18.5 % SDPA share (pc=512) within the
pc-difference + thermal variance — there is **no** hidden MLX speedup; the current
dense SDPA @32K is ~17 s, the same order as the registered 23.14 s.

**FA micro-bench (single-shot median, n=20) — flash vs incumbent dense SDPA,
per full-prefill token, per full-attention layer:**

| L | flash µs/tok | incumbent µs/tok | speedup |
|---|-------------:|-----------------:|--------:|
| 32K | 18.4 | 64.8 | 3.51× |
| 64K | 39.8 | 143.2 | 3.60× |

The flash kernel replaces **only the SDPA term**. FFN/GDN/other are flash-
independent; FFN per-token rises with pc (MCP1 M-curve v0.32.2, 8192/2048 = 1.061),
GDN/other are ~flat in pc.

## Predicted end-to-end prefill wall (s) — the FD matrix

| cell | 32K | 64K |
|------|----:|----:|
| **dense pc=2048 (baseline)** | **110.4** | **259.5** |
| **flash pc=2048** | **98.6** | **213.1** |
| dense pc=8192 | 113.9 | 259.5 |
| flash pc=8192 | 101.6 | 211.1 |

## KEEP gate: % saving vs the (dense, pc=2048) baseline

| cell | 32K | 64K |
|------|----:|----:|
| **flash pc=2048** | **+10.7 %** (PASS) | +17.9 % |
| flash pc=8192 | +8.0 % (FAIL) | +18.6 % |

## Prediction (what FD must confirm)

1. **The winning cell is (flash, pc=2048), not (flash, pc=8192).** At 32K the
   pc=8192 cell is *slower* than pc=2048: the FFN GEMM per-token penalty at pc=8192
   (M-curve, +3.3 s) eats the extra SDPA gain from the larger flash chunk. The
   original LEV-D premise — *large-pc + flash* — is **not** the winner; **fixed
   pc=2048 + flash** is.
2. **The 32K gate (≥10 %) is knife-edge: +10.7 % predicted.** This is the binding
   constraint. FD confirms or misses it by a hair. The pc-dimension does NOT
   rescue it at 32K (it makes 32K *worse*); the gain comes entirely from
   flash replacing the SDPA term at the incumbent pc.
3. **64K is comfortable (+17.9 %)** — SDPA is 29 % of the 64K prefill (vs 15 % at
   32K), so flash wins more there.
4. **The pc-dimension contributes negatively at 32K, positively at 64K.** The net
   flash gain (pc-invariant SDPA replacement) is ~10.7 % @32K / ~17.9 % @64K; the
   pc=8192 FFN penalty then subtracts ~2.7 % @32K / ~0.7 % @64K.

## What FD must NOT do

- Do not declare a 32K win from a single rep (thermal variance up to 1.6×);
  the gate is on the **mean** over ≥5 paired reps.
- Do not let the pc=8192 cell pass on 64K strength alone — the gate is 32K.
- If FD measures <10 % at 32K for (flash, pc=2048), the honest verdict is
  REJECT at 32K even if 64K is a clear win (the gate is 32K).

## Reproduce
```
python3 benchmarks/results/lev-j/fb0-predictions/model.py
```
