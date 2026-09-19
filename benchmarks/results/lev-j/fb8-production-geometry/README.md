# LEV-J Phase FB8 — gate at production geometry; re-try rejected experiments

**Status: TERMINAL — LEV-J BLOCKED-on-upstream.** The version probe (latest
mlx-swift `origin/main` 9019419, 2026-09-17) FAILS (recombination persists).
The guard experiment (Item 3) DIES: for a pure same-geometry sequence (the
production pattern), ALL 10 dispatches are TRUNCATED (nz=49152, same hash) —
re-dispatch does NOT clear the truncation. The FB7 A4 "DIFFUSE (numerics)"
classification was **WRONG** — the 0.169 is the dispatch bug (truncation),
confirmed by flashNZ=384 at Q=16. The flattened grid workaround does NOT work
(also recombined, a-nz=24·Q).

## Item 1 (TERMINAL) — version probe: FAILS (recombination persists)

Scratch-built against the latest mlx-swift (`origin/main` = 9019419,
2026-09-17, 5 commits ahead of the pinned 2bebe4e9). The 5 commits touch 100
files (integration tests, stream pooling #472, Cuda #478, logging #484,
distributed #482) but **NOT the Metal dispatch layer**. The FB8 Item 1 gate at
9019419:

```
Q=2048 prefix=8192  blocks=128 : 2/100 failed, a-nz=49152  (TRUNCATED), b-nz=12582912 (full)
Q=2048 prefix=32768 blocks=512 : 1/100 failed, a-nz=12582912, b-nz=12582909 (differ by 3)
Q=2048 prefix=65536 blocks=1024: 1/100 failed, a-nz=12582909, b-nz=12582912 (differ by 3)
Q=8192 prefix=8192  blocks=128 : 1/100 failed, a-nz=196608 (TRUNCATED), b-nz=50331644 (full)
safe-edge Q=257: 1/5 failed; Q=512: 1/5 failed
3x hash Q=2048: h1=658e3ea0c68c (TRUNCATED) / h2=h3=9d92e9b444f7 (stable)
```

**Verdict: the recombination PERSISTS at the latest mlx-swift.** Identical
pattern to the pinned version. **BLOCKED-on-upstream stands.** Pin reverted to
2bebe4e9 (no pin change on main); build restored.

## Item 3 — guard experiment: DIES (re-dispatch does NOT clear the truncation)

The guard premise: correct (full) outputs contain zero exact-zero elements, so
the nz count distinguishes full (12582912 at Q=2048) from truncated (24·Q =
49152). The guard = GPU-side nz count (1-int readback) after each dispatch,
re-dispatch on the truncation signature.

Dispatch 10x at Q=2048/prefix=8192 (pure same-geometry sequence = the
production pattern):

```
iter=0..9: nz=49152 full=false hash=c753c6eff7f1   (ALL 10 TRUNCATED, same hash)
full-dispatch unique hashes: 0
```

**Verdict: the guard DIES.** For a pure same-geometry sequence (the production
pattern — the same Q=2048 prefill geometry dispatched repeatedly), the kernel
is TRUNCATED for **ALL** dispatches (nz=49152, deterministic, same hash). The
re-dispatch does NOT clear the truncation (all 10 dispatches are truncated,
identical hash). The guard would loop forever (never gets a full result).
**BLOCKED final.**

Note: the truncated output is DETERMINISTIC (same hash for all 10 dispatches)
but WRONG (only 24·Q elements, not the full Q·256·nq). In an INTERLEAVED
dispatch sequence (FB8 Item 1 serial gate), the recombination ALTERNATES
(even=truncated, odd=full); in a PURE same-geometry sequence (the production
pattern), the COLD state PERSISTS (all truncated, FB6 A2). The guard cannot
work in the production pattern.

## Original FB8 findings (superseded by the terminal verdicts above)

The recombination bug was thought to fire only for Q ≤ 256 (FB7 A2). But
production engages at Q ≥ 2048. **The gate at production geometry FAILS** —
the recombination fires at ALL Q (the first dispatch of a fresh geometry is
recombined, a-nz = 24·Q).

## Item 1 — THE DECISIVE EXPERIMENT: strengthened gate at PRODUCTION geometry

The recombination bug was thought to fire only for Q ≤ 256 (FB7 A2). But
production engages at Q ≥ 2048. **The gate at production geometry FAILS** —
the recombination fires at ALL Q (the first dispatch of a fresh geometry is
recombined, a-nz = 24·Q).

### Item 1 serial gate (100 eval-between reps per cell)

```
Q=2048 prefix=8192  blocks=128 : 1/100 failed, a-nz=49152  (=24*2048, TRUNCATED), b-nz=12582912 (=24*2048*256, full)
Q=2048 prefix=32768 blocks=512 : 1/100 failed, a-nz=12582912 (full), b-nz=12582909 (full, differ by 3)
Q=2048 prefix=65536 blocks=1024: 1/100 failed, a-nz=12582909 (full), b-nz=12582912 (full, differ by 3)
Q=8192 prefix=8192  blocks=128 : 1/100 failed, a-nz=196608 (=24*8192, TRUNCATED), b-nz=50331644 (=24*8192*256, full)
safe-edge Q=257 prefix=4160    : 1/5   failed
safe-edge Q=512 prefix=4160    : 1/5   failed
3x hash Q=2048: h1=fc15e3170ddb (first dispatch, TRUNCATED) / h2=h3=a6609a17fcd6 (stable, full)
```

**Pattern (deterministic across 3 runs):**
- **Q=2048/prefix=8192:** a-nz=49152 (=24×2048, **truncated**), b-nz=12582912 (full). The first dispatch is recombined.
- **Q=2048/prefix=32768/65536:** both a and b are "full" (a-nz≈b-nz≈12582912), but they differ by 3 elements. A residual non-determinism.
- **Q=8192/prefix=8192:** a-nz=196608 (=24×8192, **truncated**), b-nz=50331644 (full). The first dispatch is recombined.
- **3x hash:** h1 (first dispatch) is truncated; h2=h3 (stable, full).

**Verdict: the recombination fires at ALL Q, not just Q≤256.** The first
dispatch of a fresh geometry is recombined (a-nz = 24·Q), then it's full.
**The production geometry (Q=2048) IS affected. BLOCKED-on-upstream stands.**

### Item 1 concurrent-pairs variant (20 pairs, NO eval-between)

```
Q=2048 prefix=8192  blocks=128 : 0/20 failed, totalDiff=0    (CLEAN)
Q=2048 prefix=32768 blocks=512 : 20/20 failed, totalDiff=250675180  (FAIL)
```

**Verdict:** the concurrent-pairs variant is **prefix-dependent**: prefix=8192
is clean, prefix=32768 fails. The non-determinism at production geometry is
complex (prefix-dependent, dispatch-sequence-dependent).

## Item 2 — Re-try Part B (the flattened 1-D grid workaround) with evidence

```
FB8-Item2 flattened Q=64:   1/50 failed, a-nz=1536  (=24*64, TRUNCATED),  b-nz=393216  (full), maxDiff=0.0625
FB8-Item2 flattened Q=2048: 1/50 failed, a-nz=49152 (=24*2048, TRUNCATED), b-nz=12582912 (full), maxDiff=0.0625
```

**Verdict:** the flattened grid is **ALSO recombined** (a-nz=24·Q) at both Q=64
and Q=2048. The first dispatch is truncated, then it's full. **The flattened
workaround does NOT work** — the recombination fires for the flattened grid too.
The FB7 rejection is CONFIRMED (the flattened grid has the same issue).

## Item 3 — Re-classify A4 (fp32-ref) with the truncation check

```
FB8-Item3 Q=16/prefix=256:   flashNZ=384 (TRUNCATED! =24*16) maxDiff=0.16796875
FB8-Item3 Q=2048/prefix=8192: flashNZ=12582912 (full) maxDiff=0.10888672
```

**Verdict:**
- **Q=16/prefix=256:** flashNZ=**384** (TRUNCATED, =24×16). The 0.16796875
  maxDiff is the **DISPATCH BUG** (truncation), NOT fp32 reduction order.
  **The FB7 A4 "DIFFUSE (numerics)" classification was WRONG.** The 0.169 is
  the dispatch bug hitting the test by dispatch parity (the first dispatch is
  truncated).
- **Q=2048/prefix=8192:** flashNZ=12582912 (full, NOT truncated). maxDiff vs
  dense = 0.10888672 (> 0.0625 bound). This is the **reduction-order issue**
  (the flash uses a different reduction order than the dense path), NOT the
  dispatch bug.

## Item 4 — Upstream filing

The repro is ready. The key evidence:
1. **Trivial probe (FB7 A1):** standalone MLX repro, no flash code. The dispatch
   ALTERNATES between RECOMBINED and FULL.
2. **Q sweep (FB7 A2):** deterministic across 3 runs. For Q ≤ 256: 24 groups ×
   Q threads. For Q > 256: 24 groups × Q threads (1-D mapped to (gx,tx)).
3. **In-loop discriminator (FB7 A3):** the dispatch index parity.
4. **Production geometry (FB8 Item 1):** the recombination fires at Q=2048
   (a-nz=49152=24×2048) and Q=8192 (a-nz=196608=24×8192). The first dispatch
   of a fresh geometry is recombined, for ALL Q.
5. **Pins:** mlx-swift `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` (resolved
   `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`, 2026-07-01); mlx C++
   `ce45c52505c8158ea48d2a54e8caae05efd86bfe` (2026-03-12).
6. **1-D grid workaround (FB7 B2, FB8 Item 2):** does NOT work (also recombined).
7. **FB3 concurrent-recycling finding:** the concurrent-pairs variant at
   Q=2048/prefix=32768 fails 20/20 (possibly-related).

**The upstream issue is filed against ml-explore/mlx (the MLXFast.metalKernel
dispatch layer).** See `benchmarks/results/lev-j/fb8-production-geometry/`
for the full evidence package.

## Item 5 — Verdict and unblock decision (TERMINAL)

**Item 1 FAILS at production geometry AND the version probe FAILS (recombination
persists at the latest mlx-swift) AND the guard DIES (re-dispatch does NOT clear
the truncation in the production pattern) → LEV-J BLOCKED-on-upstream is the
TERMINAL state, now with the complete filing.** Gate OFF; zero production risk.

The recombination fires at ALL Q (not just Q≤256). The production geometry
(Q=2048) is affected. The flattened grid workaround does NOT work. The FB7 A4
"DIFFUSE (numerics)" classification was WRONG (the 0.169 is the dispatch bug).

## Record corrections

1. **FB7 A4 "DIFFUSE (numerics)" → RECLASSIFIED:** the 0.169 at Q=16/prefix=256
   is the **DISPATCH BUG** (truncation, flashNZ=384), NOT fp32 reduction
   order. The FB7 A4 classification was quantitatively impossible (0.169 ≈
   max|ref|, ALL 98,304 elements differ — that is the truncated-flash
   signature).
2. **FB7 "Q≤256 boundary" → CORRECTED:** the recombination fires at ALL Q, not
   just Q≤256. The first dispatch of a fresh geometry is recombined (a-nz=24·Q).
3. **FB7 Part B rejection → CONFIRMED:** the flattened grid is also recombined
   (a-nz=24·Q at Q=64 and Q=2048). The workaround does NOT work.

## Hygiene

- No production code changed (gate OFF; flash-OFF byte-identical to main).
- Both suites build. The documented open-bug tests (Fp32Ref, Determinism,
  FB8-Item1) fail as expected.
- `git diff --check` clean (both repos). No server running. Fresh checkpoints.
- Results in `benchmarks/results/lev-j/fb8-production-geometry/` with all raw
  numbers.
