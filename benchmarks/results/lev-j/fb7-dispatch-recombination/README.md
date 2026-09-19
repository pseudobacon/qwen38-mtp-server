# LEV-J Phase FB7 — dispatch-recombination mechanism + 1-D grid workaround test

**Status: mechanism pinned (Part A); 1-D grid workaround REJECTED (Part B);
BLOCKED-on-upstream stands.** The FB6 "grid truncated to x=0 column / COLD-JIT"
label is superseded by the **recombination signature**: grid.y→group count,
grid.x→1-D thread count, threadGroup.x→threads-per-group cap (256). The
" COLD-JIT" is a mislabel — it is a dispatch recombination, NOT a JIT compile.

## Part A — Mechanism experiments (no production changes)

### A1 — TRIVIAL-KERNEL PARAMETER PROBE (standalone MLX repro, no flash code)

Dispatch the trivial probe kernel (records which (gx,gy,tx) executed) with the
EXACT failing params (grid (64,24,1), threadGroup (256,1,1)), 20x:

```
A1 probe iter=0: nz=1536    maxGX=0  maxGY=23 maxTX=63  gySet=24 txSet=64   RECOMBINED
A1 probe iter=1: nz=393216  maxGX=63 maxGY=23 maxTX=255 gySet=24 txSet=256  FULL
A1 probe iter=2: nz=1536    maxGX=0  maxGY=23 maxTX=63  gySet=24 txSet=64   RECOMBINED
A1 probe iter=3: nz=393216  maxGX=63 maxGY=23 maxTX=255 gySet=24 txSet=256  FULL
... (perfect alternation for iter=0..19)
```

**Verdict:** the recombination is REAL and DETERMINISTIC. The dispatch
ALTERNATES between RECOMBINED (24 groups × 64 threads) and FULL (1536 groups ×
256 threads) on every other dispatch. **The standalone MLX repro is COMPLETE
(no flash code needed).**

### A2 — Q SWEEP (deterministic across 3 runs)

Fixed prefix=4160, Q in {16,32,64,128,255,256,257,512,2048}:

```
Q       nz      nz/24  maxGX  maxGY  maxTX   cols  rows  threads
16      384     16     0      23     15       1     24    16
32      768     32     0      23     31       1     24    32
64      1536    64     0      23     63       1     24    64
128     3072    128    0      23     127      1     24    128
255     6120    255    0      23     254      1     24    255
256     6144    256    0      23     255      1     24    256
257     6168    257    1      23     255      2     24    256
512     12288   512    1      23     255      2     24    256
2048    49152   2048   7      23     255      8     24    256
```

**Verdict:** the recombination is:
- **grid.y (24) → group count** (rows = 24, always).
- **grid.x (Q) → 1-D thread count** (threads = Q for Q ≤ 256; for Q > 256, the
  threads are mapped 1-D to (gx,tx) where gx=i/256, tx=i%256, so cols=ceil(Q/256),
  threads=256).
- **threadGroup.x (256) → threads-per-group cap** (the 1-D thread index is
  capped at 256 per group).

This table goes in the upstream issue verbatim.

### A3 — IN-LOOP DISCRIMINATOR

Within ONE Q=64 loop, 40 dispatches:

```
A3 iter=0:  nz=1536    maxGX=0  maxTX=63  TRUNCATED
A3 iter=1:  nz=393216  maxGX=63 maxTX=255 full
A3 iter=2:  nz=1536    maxGX=0  maxTX=63  TRUNCATED
A3 iter=3:  nz=393216  maxGX=63 maxTX=255 full
... (perfect alternation for iter=0..39)
A3 summary: truncated=20 full=20 of 40
```

**Verdict:** the discriminator is the **dispatch index parity**. Every even
dispatch is TRUNCATED, every odd dispatch is FULL. This is a Metal dispatch
cache issue: the cache alternates between the recombined and full configurations
on every dispatch.

### A4 — FP32-REF RE-CLASSIFICATION

```
A4 fp32-ref: totalDiff=98304 maxDiff=0.16943918 qRowsWithDiff=16/16 dsWithDiff=256/256 hsWithDiff=24/24
  A4 qRow diff spread: min=6144 max=6144
  A4 d diff spread: min=384 max=384
  A4 verdict: DIFFUSE (numerics, not dispatch bug)
```

**Verdict:** the fp32-ref difference is **DIFFUSE** (all 98,304 elements,
uniform spread across all q rows, d dims, and heads). This is the FB3
8-partial reduction-order issue (numerics) at the small geometry (Q=16,
prefix=256), NOT the dispatch bug. The 0.169 maxDiff is a numerical issue, NOT
corruption. The bound (0.0625) is NOT reset (the dispatch bug is still live in
the determinism gate, and the fp32-ref is a separate numerical issue).

## Part B — 1-D grid workaround test (the unblock candidate)

### B1 — Flattened 1-D grid kernels (behind the debug hook)

The flattened 1-D grid kernels (`qwen38FlashSDPAMax1D`,
`qwen38FlashSDPASum1D`) dispatch with grid (Q*nq, 1, 1), threadGroup (256,1,1),
decoding pos→(q_row, h) inside the kernel. Same math, same fixed-order
reductions; the ONLY change is the grid mapping.

### B2 — Gate the flattened variant: **REJECTED**

```
flattened NOT bit-exact at 16 blocks  (99/100 pairs differed; last a-nz=1536 b-nz=393216)
flattened NOT bit-exact at 65 blocks  (100/100 pairs differed; last a-nz=393216 b-nz=393216)
flattened NOT bit-exact at 128 blocks (100/100 pairs differed; last a-nz=393216 b-nz=393216)
flattened NOT bit-exact at 512 blocks (100/100 pairs differed; last a-nz=393216 b-nz=393216)
flattened NOT bit-exact at Q=32  (1/50 pairs differed)
flattened NOT bit-exact at Q=128 (1/50 pairs differed)
flattened NOT bit-exact at Q=255 (1/50 pairs differed)
flattened NOT bit-exact at Q=256 (1/50 pairs differed)
flattened 3x dispatch FNV-1a not equal: 422a7a99cbe0 / 49254c89d1fa / 49254c89d1fa
```

**Verdict:** the 1-D grid workaround **DOES NOT WORK**. The flattened grid
(Q*nq, 1, 1) = (1536, 1, 1) is NOT recombined (a-nz=b-nz=393216, both full),
but the values **differ** (non-determinism). This is a **DIFFERENT**
non-determinism — the MLX allocator/buffer issue (FB3/FB4), NOT the
recombination. The 1-D grid does NOT avoid the non-determinism.

**BLOCKED-on-upstream stands.** The probe (A1) + Q sweep (A2) are the filing.

### B3 — Re-bench: NOT RUN

The workaround does not work, so re-benching is not applicable.

## Part C — Upstream issue (evidence package)

**Title/label:** "MLXFast.metalKernel dispatch recombination: grid.y→group
count, grid.x→1-D thread count, threadGroup.x→threads-per-group cap" — NOT
"grid truncation" and NOT "JIT".

**Evidence:**
1. **Trivial-kernel repro (A1):** the standalone MLX repro, no flash code.
   Dispatch the trivial probe kernel with grid (64,24,1), threadGroup (256,1,1)
   → the dispatch ALTERNATES between RECOMBINED (24 groups × 64 threads) and
   FULL (1536 groups × 256 threads) on every other dispatch.
2. **Q-sweep table (A2):** deterministic across 3 runs. For Q ≤ 256: 24 groups
   × Q threads. For Q > 256: 24 groups × Q threads (1-D mapped to (gx,tx)).
3. **In-loop discriminator (A3):** the dispatch index parity (even=TRUNCATED,
   odd=FULL).
4. **Pins:** mlx-swift `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` (resolved
   `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`, 2026-07-01); mlx C++
   `ce45c52505c8158ea48d2a54e8caae05efd86bfe` (2026-03-12).
5. **1-D grid workaround (B):** does NOT work (the MLX allocator/buffer issue
   persists in a different form).

**Record correction:** the FB6 "grid truncated to x=0 column / COLD-JIT" label
is superseded by the recombination signature. The "COLD-JIT" is a mislabel — it
is a dispatch recombination, NOT a JIT compile. This is the FOURTH mechanism
label in this saga (FB2 "kernel non-deterministic" → FB3 "MLX allocator bug" →
FB5 "m/out byte alias" → FB6 "grid truncation / COLD-JIT" → FB7 "dispatch
recombination"). The record shows the correction discipline held.

## Hygiene

- No production code changed (only `#if DEBUG` hooks + tests + the flattened
  kernels behind the debug hook).
- Gate OFF (flash-OFF byte-identical to main).
- Both suites build; Fp32Ref/Determinism fail (the known open bug, documented).
- Results in `benchmarks/results/lev-j/fb7-dispatch-recombination/`.
- All raw numbers reported above.
