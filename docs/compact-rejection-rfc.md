# RFC: Compact-space rejection walk for non-greedy MTP verification

**Status:** proposed (Task 1, `feature/compact-rejection-math`)
**Scope:** fork `Libraries/MLXLLM/Models/Qwen38MTPBlockSession.swift`, sampling path only
(`temperature > 0`). The greedy path (`temperature <= 0`) is untouched and must
stay byte-identical.

## 1. Problem

The MTP head has a **compact vocabulary** (`vCompact ≈ 98,330`): its lm_head is
coarser than the target's full vocabulary (`V = 248,320`), and
`mapDraftTokenIds` (`c2f: [vCompact] -> Int32`) gives the compact→full index
map. Prompt 16 fixed the `[broadcast_shapes]` crash by expanding the compact
draft distribution `q` to full vocabulary before storage:

```swift
let qDistFull = MLXArray.full([vocab], values: MLXArray(0.0)).at[c2f].add(qDist)
```

The rejection walk, fallback, and bookkeeping then operate on `[248,320]`
arrays every verification round. The expansion (allocate + fill-zero +
scatter-add) is pure overhead: the draft distribution has support **only** on
the compact vocabulary, and the out-of-support residual mass has a closed form.

## 2. Notation

- `V` = full vocabulary size (248,320); `C` = compact size (98,330).
- `p ∈ [0,1]^V` = target distribution (row i of the verify softmax, after
  penalties + filters, same as today).
- `q ∈ [0,1]^C` = draft distribution (row i of the draft softmax, after
  penalties + filters), support on compact indices only.
- `c2f: [C] → [0, V)` = compact→full map (bijection onto its image; identity
  when the head uses the full vocabulary, in which case `C == V` and the new
  walk degenerates to the old one with one extra gather).
- Full-vocab lift of `q`: `q_f[c2f(c)] = q[c]`, `q_f[f] = 0` for
  `f ∉ image(c2f)`.
- Leviathan residual: `r_f = max(0, p_f − q_f)`, `R = Σ_f r_f`.

## 3. The split (exact math)

Because `q_f = 0` off the compact support, the residual decomposes exactly:

- **In-support** (`f ∈ image(c2f)`, written as `c = c2f⁻¹(f)`):
  `r_f = max(0, p_f − q_c)`. Let
  `p_c = p[c2f(c)]` (one gather), `r_c = max(0, p_c − q_c)`,
  `R_in = Σ_c r_c`.
- **Out-of-support** (`f ∉ image(c2f)`): `r_f = p_f`. Let
  `S = Σ_c p_c = Σ_{f ∈ image(c2f)} p_f` (gather-then-sum), then
  `R_out = 1 − S` (both `p` and `q` are normalized; the `max(0, ·)` clamps
  away any fp rounding that makes `S` exceed 1).
- Total: `R = R_in + R_out`.

**Sampling.** The old walk draws `f ~ r_f / R` over all `V` entries. Draw
instead:

1. With probability `R_in / R` (i.e. `u · R < R_in` for `u ~ U(0,1)`):
   draw `c ~ r_c / R_in` over the `C` compact entries and emit `c2f(c)`.
2. With probability `R_out / R`: draw `f ~ p_f / R_out` over
   `f ∉ image(c2f)`.

**Lemma (distribution preservation).** For `f ∈ image(c2f)`:
`Pr(new = f) = (R_in/R) · r_c/R_in = r_c/R = r_f/R`. For
`f ∉ image(c2f)`: `Pr(new = f) = (R_out/R) · p_f/R_out = p_f/R = r_f/R`.
So the new walk emits exactly the same normalized residual `r / R` as the
old walk — the Leviathan/Chen construction is unchanged, and the emitted
token sequence remains distributed exactly as target-only sampling
(`p`), independent of the draft model. The acceptance test
(`α = min(1, p_f(q⁻¹)/q)`, `r ≤ α`) and the bonus-token step (sample from the
bonus-row `p`) are untouched, so the full-speculation invariance argument of
`docs/speculative-sampling-rfc.md` §3 carries over unchanged.

### Out-of-support tail sampling (step 2)

The tail is `p` restricted to the complement of `image(c2f)`. To sample it
exactly we need one full-vocab construction, **only on this branch**:
`logits_f = log(max(p_f, 1e-10))` for `f ∉ image(c2f)`,
`−∞` for `f ∈ image(c2f)` (a `−∞` entry gets exactly zero softmax mass;
the `1e-10` clamp on the kept entries preserves the Prompt 17 log-space
convention). One `where` over `[V]` + one `categorical` — and this branch
fires only when (a) a draft is rejected AND (b) the uniform draw lands in the
`R_out` slice. The per-round common path (storage + acceptance walk) and the
full-acceptance path (bonus sample) perform **no** new full-vocab work.

### Degenerate case

If `R == 0` then `p ≤ q` pointwise and, both being normalized, `p == q`
everywhere; the residual is empty. The old code's fallback (sample from the
full target `p`, `1e-10`-clamped log) is retained verbatim for `R ≤ 0`.

## 4. What changes in code

fork `Libraries/MLXLLM/Models/Qwen38MTPBlockSession.swift`, sampling path only:

1. **Draft loop:** `qDists` stores the **compact** `qDist` (drop the
   `qDistFull` scatter expansion). `qScalar = qDist[sampledCompact]` unchanged.
   The penalty frequency gather `freqFull_i[c2f]` is untouched.
2. **Rejection walk:** on rejection at index `i`:
   `p_c = pDist[c2f]` (gather), `r_c = max(0, p_c − qDist_i)`,
   `R_in = Σ r_c`, `R_out = max(0, 1 − Σ p_c)`, then the split of §3.
   The full-vocab `pDist` is still read for the tail branch and the
   degenerate fallback (it already exists as the verify softmax — no new
   materialization).
3. **Bonus sample:** unchanged (full-vocab target sample; the bonus token is
   a target-generated token and must come from full `p`).
4. **Trace:** the env-gated `MLX_QWEN_MTP_TRACE=1` round line gains
   `active_bytes=<Memory.activeMemory>` (per-round memory delta, default-off,
   zero overhead when the trace is off).

No other file changes behavior. Greedy path, Prompt 17 log-space clamps
(three `categorical` sites receive logits; zeros clamped to `1e-10`),
Prompt 19 offset-invariant `committed` bookkeeping, Prompt 20 stop-token
early-stop branch, and the stop-token break inside the walk are all
preserved.

## 5. Numerics

All arithmetic stays in fp32 `MLXArray` on-device, same dtype as today.
Rounding differences vs the old walk: (a) `R_in`/`R_out` are sums of the same
values the old walk summed inside one big `sum()`; (b) the split draw
introduces one extra `u ~ U(0,1)` per rejection. Both are below the chi-square
/KS tolerance of the test plan. The `1e-10` clamps are applied at the same
points (residual entries, fallback, bonus).

## 6. Failure modes and mitigations

| Mode | Behavior | Mitigation |
| --- | --- | --- |
| `R_in > 0`, `R_out == 0` (mass never leaves support) | `u·R < R_in` always true → compact branch only; no full-vocab work | — |
| `R_in == 0`, `R_out > 0` (residual entirely out-of-support) | `u·R < 0` false → tail branch; tail has `R_out > 0` mass, `categorical` well-defined | — |
| `R == 0` (p == q) | Legacy full-`p` fallback, bit-identical to old code | — |
| fp: `Σ p_c > 1` by rounding | `R_out = max(0, 1 − Σ p_c)` clamps; total `R = R_in + R_out` then slightly under-counts the true residual by < ulp — statistically invisible | clamped |
| fp: `R_in` tiny, `R` dominated by `R_out` | Branch probability `R_in/R` computed in fp64 on host from two fp32 reads — no underflow risk at these magnitudes | — |
| Head switches to full vocab (`C == V`) | `c2f` is identity, `R_out ≡ 0`; walk degenerates to the old residual walk plus one gather | degenerate-safe |
| Repeated rejections per session | Tail branch builds one `[V]` logit array per such rejection — same asymptotic cost as the old walk on rejection, no new steady-state work | measured in benchmark |

## 7. Test plan (per `speculative-sampling-rfc.md` §6)

1. **Greedy regression (no tolerance):** `mtpCorrectnessSerialMatchesMTPAtAllDepths`
   stays token-ID identical at all depths (`QWEN_RUN_WEIGHT_TESTS=1`).
2. **Pure distributional tests (no weights), seeded RNG:** reference
   implementations of the OLD walk (full-vocab residual) and the NEW walk
   (compact split) over synthetic `p`/`q` pairs; chi-square that the new walk
   matches the old walk and both match the analytic residual, across:
   well-overlapping, disjoint supports, `q`-mass-where-`p`=0,
   `p`-mass-where-`q`=0 (out-of-support), top-k=1 nucleus, and the `C == V`
   degenerate case.
3. **Weight-gated serial-vs-MTP distributional equivalence**
   (`QWEN_RUN_WEIGHT_TESTS=1`, `t = 0.7`, top-p 0.95): token-frequency
   chi-square between serial (depth 0) and MTP (depth 2) — distributional,
   not token-identical, per §6B.

## 8. Benchmark

Fixed non-greedy workload (`temperature 0.7`, `top_p 0.95`, fixed prompt,
`QWEN_MLX_SEED` fixed, `max_tokens 300`), interleaved A/B (baseline `main`
binary vs feature binary) ≥ 5 per side, ≥ 60 s cooldown, same session.
Report: median decode tok/s, per-round verify duration
(`MLX_QWEN_MTP_TRACE=1`: `eval_wall_us`/`round_us`), per-round
`Memory.activeMemory` delta (`active_bytes`).
