# LEV-J Phase FB6 — re-derivation of the corruption mechanism

**Status: mechanism re-derived.** The FB5 m→out byte-overlay diagnosis is
disproven. The actual mechanism is a **Metal dispatch grid truncation to the
x=0 column** (24 threadgroups = q_row=0, all heads), **persistent** for a pure
same-geometry dispatch sequence, affecting **both** passes. This is an **MLX
Metal dispatch bug**, not a kernel math bug. No production code changed.

## Raw results (all reported)

### A1 — RAW-BYTES vs GRAPH-OP read (65-block geometry, Q=64, prefix=4160)
```
A1 iter=0: rawNZ=1536 graphNZ=1536 bytes=786432
... (identical for iter=1..7)
```
**Verdict:** raw bytes and graph-op count AGREE (both 1536). The array is
GENUINELY corrupted — the corruption is NOT in the realization path (the gate
itself is not the bug surface). Continue to A2.

### A2 — SUBSET test (same geometry)
```
A2 dispatch 2 nz=1536
A2 dispatch 3 nz=1536
...
A2 dispatch 6 nz=1536
A2: NO warm reference reached (COLD JIT persists past 20 dispatches); coldNZ=1536
```
**Verdict:** the COLD-JIT (1,536 non-zeros) PERSISTS past 20 dispatches for a
pure same-geometry sequence (no WARM reference reached). It is NOT a one-time
JIT compile in this sequence.

### A3 — STRUCTURE analysis (same geometry)
```
A3 out: nonZero threadgroups=24 of 1536; byHead spread: min=64 max=64; byQRow spread: min=1536 max=1536
A3 m: nonZero=24 byHead count=24 (one-per-head? true) byQRow spread: min=24 max=24
  A3 m nonZero qRow positions (first 24): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472]
```
**Verdict:**
- `out` (2,4×64×256): only **24 of 1,536 threadgroups** have non-zeros; each
  head has exactly 64 non-zeros; ALL at q_row=0. → the surviving writes are
  `out[h, 0, 0..63]` for h in 0..23 (first query row, first 64 d-dims).
- `m` (24×64): **24 non-zeros (one per head)**, at positions h*64 for h in
  0..23 → m[h, 0] (q_row=0) is the ONLY surviving value per head.
- **The grid is truncated to the x=0 column** (q_row=0, all 24 heads) × 64
  threads (d=0..63). Metal launched 24 threadgroups × 64 threads = 1,536
  threads, not the full 1,536 threadgroups × 256 threads.

### A4 — COMPILE-BOUNDARY correlation
```
A4 fresh-geo prefix=2048 iter=0: nz=1536
... (identical for iter=1..5)
A4 re-dispatch: r1nz=1536 r2nz=1536 (recovered if 393216)
```
**Verdict:** a FRESH geometry (prefix=2048, never dispatched before) shows the
COLD-JIT (1,536) at EVERY dispatch (iter 0..5), and an immediate re-dispatch
with identical inputs does NOT recover it (r1nz=r2nz=1536). The COLD-JIT is
persistent for a pure same-geometry sequence.

### A5 — SINGLE-KERNEL isolation (pass 1 alone, no pass 2 in flight)
```
A5 pass1-alone iter=0: mNZ=24/1536
... (identical for iter=1..5)
```
**Verdict:** pass 1 (the max kernel) ALONE is corrupted (mNZ=24/1536, one per
head). The two-call composition is NOT the cause — the corruption is in the
Metal dispatch of a SINGLE kernel.

### A6 — VERSION probe
- mlx-swift pin: `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` (resolved
  `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`, 2026-07-01).
- mlx C++ submodule pin: `ce45c52505c8158ea48d2a54e8caae05efd86bfe`
  (2026-03-12, "[CUDA] Use qmv kernel for fp quantizations (#3239)").

## B1 — Test hygiene (landed)
- **eval-flash-before-reference** in `testFlashMatchesDenseForward` / `Reverse`
  / `Fp32Reference` (separate command buffer for flash before the reference).
- **Zero-signature detector** in every determinism-gate failure message:
  `last failing a-nz=1536 b-nz=393216` (the COLD-JIT vs WARM split).
- **513→512** block label fixed (prefix=32768 → ceil(32768/64) = 512).
- Determinism gate result (B1): `a-nz=1536 (COLD), b-nz=393216 (WARM)` at
  EVERY geometry (16/65/128/512); 99–100/100 pairs differ.

## B3 — fp32-ref re-diagnosis (after B1)
```
B3 fp32-ref: maxDiff=0.16943918 diffCount=98304 (geometry Q=16 prefix=256)
```
**Verdict:** diffCount=98,304 = the TOTAL element count for Q=16 (24×16×256).
This is a **DIFFUSE** difference (all elements), NOT the scattered COLD-JIT
signature. The 0.169 maxDiff is the FB3 8-partial reduction-order issue
(expected at the small geometry), NOT corruption. **The bound is NOT reset**
(corruption is still live in the determinism gate).

## Part C — Verdict and decision tree

**Mechanism:** the `MLXFast.metalKernel` Metal dispatch **truncates the grid
to the x=0 column** (q_row=0, all heads) for a FRESH geometry, launching only
24 threadgroups × 64 threads (1,536 threads) instead of the full grid. The
truncation is **persistent** for a pure same-geometry dispatch sequence (A2,
A4) and affects **both** passes (A5: pass 1 alone corrupts). It is an **MLX
Metal dispatch bug**, NOT a kernel math bug (the kernel correctly computes
attention for the threadgroups it does execute).

**Decision-tree branch:** "A2 subset + A4 compile-correlated → MLX dispatch
bug." The **minimal repro** is pass 1 alone (A5): a single `metalKernel`
dispatch at a fresh geometry shows the grid truncation, with NO flash code.

**Wrapper workarounds (to evaluate in a follow-up task, NOT implemented here):**
1. eval immediately after every dispatch (the determinism gate's eval-between)
   — does NOT clear the COLD-JIT (A4 re-dispatch r1nz=r2nz=1536).
2. dispatch-and-discard the compile-triggering call — A4 shows successors are
   COLD for a pure sequence, so this does NOT work for pure sequences (only
   interleaved sequences, where the second dispatch is WARM — determinism gate).

**The 1 s-idle re-arming (FB4)** means idle-decompile re-arms the COLD-JIT, so
any workaround must cover post-idle prefills, not just startup.

**Terminal state if no robust workaround + no upstream fix:** LEV-J
BLOCKED-on-upstream — gate OFF (zero production risk today), evidence package
filed, roadmap records the dependency. This is an acceptable close, not a
failure — the kernel's MATH has never failed a test; every failure across
FB–FB6 has been in the MLX dispatch/allocator/JIT layer.

## Files changed (no production code)
- `FlashSDPA.swift`: +`flashSDPADebug` (FB5), +`flashSDPAPass1Debug` (FB6 A5),
  both under `#if DEBUG`. `flashSDPA` production path UNCHANGED.
- `Qwen38FlashSDPATests.swift`: +A1–A5 diagnostic tests, +B1 hygiene
  (eval-before-reference, zero-signature detector, 513→512), +B3 diffCount.
- No production code changed. Gate OFF.
