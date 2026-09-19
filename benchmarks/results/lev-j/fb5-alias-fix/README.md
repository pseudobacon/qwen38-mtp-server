# LEV-J Phase FB5 — Item 1a PROOF: the m/out byte-alias does NOT reproduce

**Status: STOP.** Per the FB5 task ("If the signature does NOT reproduce at the
failing iteration, STOP and report — the diagnosis needs revisiting before any
patch"), the Item 1a proof did **not** reproduce the diagnosed m/out byte alias.
No production patch (Item 1b) has been applied.

## What the diagnosis predicted

The FB5 byte-level diagnosis stated the failing output `a` (Q=64, [24,64,256]
bf16, 393,216 elements) is a zero-initialized buffer with **exactly the pass-1
`m` array's bytes at its head**: m [24,64] fp32 = 1,536 values = 6,144 bytes →
1,536 non-zeros + 390,144 untouched zero slots = 391,680 zeros / 1,536
non-zeros.

## What was measured (65-block geometry, prefix=4160, Q=64)

Via the `flashSDPADebug` hook (identical two-kernel dispatch sequence, no
eval-between), 8 iterations:

```
ALIAS-PROOF iter=0: headEq=false outNZ=1536 mNZ=24
ALIAS-PROOF iter=1: headEq=false outNZ=1536 mNZ=24
... (identical for iter=2..7)
```

- `headEq=false` at **every** iteration — the first 6,144 bytes of `out` are
  NOT equal to `m`'s bytes. **The m/out byte alias does not reproduce.**
- `outNZ=1536` — the non-zero count matches the diagnosis (1,536 = nq*Q).
- `mNZ` is **run-dependent**: 24 in the first proof run, 1536 (fully populated)
  in the gate run. So `m` is sometimes fully computed, sometimes not — an MLX
  dispatch/allocator timing artifact, NOT a stable byte alias.
- `out`'s 1,536 non-zeros are **normal small attention outputs** (≈0.001–0.03),
  **scattered** across the buffer (firstNZ=0, lastNZ=376895), NOT m's bytes
  and NOT at the head.

## What this means

- The high-level signature (out has exactly 1,536 non-zeros = nq*Q) **is** the
  COLD-JIT / partial-computation signature, and it is DETERMINISTIC for a fixed
  dispatch sequence.
- The specific mechanism in the diagnosis (m's bytes overlaid at out's head)
  is **not** what the bytes show. The non-zeros in `out` are computed attention
  values, not `m`'s fp32 bytes.
- The 1,536-non-zero COLD-JIT pattern is **sequence-sensitive** (a prior probe
  with an extra `flashSDPA` in the loop shifted the COLD/WARM boundary), i.e.
  an MLX dispatch/allocator timing artifact, consistent with FB3's
  allocator-recycling finding — not a one-time m/out buffer alias.

## Consequence

- Item 1a (proof) → the diagnosed byte alias does not reproduce. **STOP.**
- The 1,536-non-zero COLD-JIT residual is the real, deterministic signature to
  explain before the Item 1b `MLX.eval([mIn])` boundary fix can be validated
  against the gate.
- No production code was changed (FlashSDPA.swift `flashSDPA` is unmodified;
  only a `#if DEBUG` hook `flashSDPADebug` and this test were added).

## Open question for the next session

Re-derive the byte-level diagnosis from a fresh failing sample: dump the full
byte layout of `out` (the 1,536 non-zero positions AND values) and of `m`, and
compare against a warm (393,216-non-zero) reference at the SAME dispatch
sequence, to determine (a) whether `out`'s non-zeros are a correct subset of a
warm output, and (b) whether the 1,536-non-zero pattern is the partial-write
COLD JIT or an MLX output-buffer alias. Only then decide the correct
structural fix.
