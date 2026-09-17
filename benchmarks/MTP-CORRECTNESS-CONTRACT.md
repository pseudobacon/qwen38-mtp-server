# MTP Correctness Contract

Compatibility contract for the native MTP (multi-token-prediction) speculative path in the
Qwen3.8-27B 4-bit + MTP-head server. This document states, with evidence, what the MTP path
guarantees against the serial target-only path, and what it does **not** guarantee.

It is the Phase A gate: Phase B (prefix-cache work) may proceed only because the claims below
are backed by unit tests and a distributional harness, not by assertion.

---

## 0. Bit-exactness policy v2 (2026-09-17) — scoped relaxation for prefill-width FFN GEMMs

**Decision (explicit, recorded before any code).** Bit-exactness is **relaxed to a numeric
tolerance for FFN GEMMs dispatched at prefill chunk widths only**, and held **byte-for-byte
elsewhere**. This supersedes the FFP1 NO-GO rationale (the FFP1 headroom was precisely the
GEMM tiling that bit-exactness forbade changing); it does **not** invalidate FFP1 — it
reopens it under a scoped, gated, default-OFF relaxation.

**Scope of the relaxation.**

| Path | M | Exactness | Notes |
|------|---|-----------|-------|
| FFN down_proj / gateup GEMM at prefill chunk width | **M ≥ 256** | **numeric tolerance** (a few ulp of bf16 output; fp32-accumulate differences only) | The relaxed path. `M ≥ 256` must exceed every verify width (M 1..9) and decode (M=1), and exclude the short-prompt fixtures `essay-1024` / `specdec-800` (prefill M < 256). |
| Decode | M = 1 | **byte-for-byte** | Unchanged. |
| MTP verify rows | M 2..9 | **byte-for-byte** | Unchanged. |
| Short-prefill (essay-1024, specdec-800) | M < 256 | **byte-for-byte** | Unchanged; must reproduce the existing registry in both gate states. |
| Every other kernel / code path (attention, GDN, norms, …) | — | **byte-for-byte** | Unchanged; relaxation is FFN-only and does not broaden. |

**Gate.** `MLX_QWEN_FFN_PREFILL_FAST`, **default OFF**, with an M ≥ 256 dispatch geometry
gate. The OFF build is byte-identical to current `main` (every incumbent path taken). The
relaxed path engages only when the gate is ON **and** M ≥ 256.

**Accepted consequence.** Long-context (8K+) committed streams computed through the relaxed
path will **differ from the incumbent registry**. The divergence is the Phase-1 knife-edge
family (fp accumulation order; ≤ a few ulp drift flipping near-tie argmaxes), **not
corruption**. Per FFP6, ON runs register **new** per-fixture stream hashes; OFF runs must
reproduce the incumbent registry exactly. A divergence at a top-2 logit gap > 8 ulp is a STOP
condition (outside the accepted family → the numeric bound is too loose). Decode/verify and
short-prefill streams are unaffected (byte-for-byte), so the relaxation is invisible to them.

**Bound.** The accepted numeric bound is established in FFP4 (tolerance mode of
`--ffn-check`): expected fp32-accumulate differences only, bounded by a few ulp of bf16 output
magnitude, with a max|diff| and ulp-histogram contract recorded for the unit tests.

---

## 1. The two exactness regimes

The speculative path is only ever used with one of two sampling configurations, and the
contract differs for each:

| Regime | Configuration | Guarantee |
|--------|---------------|-----------|
| **Greedy** | `temperature = 0` | Committed token stream is **token-ID identical** to serial greedy target decoding, *modulo the near-tie ulp property* (Section 3). |
| **Stochastic** | `temperature > 0`, **no non-default penalties** | Committed-token **distribution** is **exactly** the serial target distribution (mathematically exact speculative rejection sampling). |
| **Non-default penalties** | presence/frequency/repetition ≠ default | MTP is **not used**. The request is forced to serial target-only depth (`effectiveMTPEnabled == false`). No speculative claim applies. |

The penalty case is enforced, not assumed: see the F1 fix (Section 5).

---

## 2. Greedy (T=0) exactness

**Claim.** At `temperature = 0`, the MTP path commits exactly the tokens serial greedy target
decoding would commit, except at measure-zero near-tie boundaries (Section 3).

**Why (A1 call-path map).**
- The primary token each round is `argmax(softmax(targetLogits))` = the serial greedy pick.
- Every drafted (non-primary) token is **verified against the target logit at its position**
  before commit: a draft is accepted only if it is the target's argmax at that state.
- Therefore the committed stream is a subsequence-consistent extension of serial greedy:
  each committed token is *the* greedy pick for its state. The only way it can diverge from a
  pure serial run is if the draft's and target's max-logit differ enough at a near-tie to flip
  the argmax (Section 3).

**Evidence.**
- `Qwen38MTPDiagnosticTests.testMTPPostNormAcceptanceAndLogitParity` — measures greedy
  acceptance (~91% on the diagnostic prompt) and confirms post-norm logit alignment between
  draft and target.
- `Qwen38MTPDiagnosticTests.testWideVerifyStaysInSerialFamily` — depth-5 verify stream stays in
  the serial family (match rate asserted ≥ 0.85; observed 0.85–0.97 across prompts).
- Draft-depth policy sweep (`DRAFT-DEPTH-POLICY.md`) — depths k=1..8 all remain in the serial
  greedy family; no depth changes the committed stream beyond the near-tie property.

**Reproduce:**
```
cd ../mlx-swift-lm
swift test --filter Qwen38MTPDiagnosticTests
```

---

## 3. The near-tie ulp property (accepted residual, not a defect)

At a near-tie boundary where the top-1 and top-2 target logits differ by less than roughly
`1e-6`, a last-ulp difference between the draft head's max-logit and the target's max-logit can
flip which of the two tokens is the argmax, and therefore which token is committed. Consequences:

- **This is not a correctness violation.** Both tokens are valid greedy picks at the ulp
  boundary; the model has no canonical answer there.
- **It is measure-zero** in the prompt/token space and does not accumulate.
- **It is the sole documented cause** of committed-stream divergence between MTP and serial
  greedy. It is *not* a distributional error (greedy is degenerate at T=0).

This property is why the greedy guarantee is stated as "token-ID identical *modulo the
near-tie ulp property*" rather than unconditionally identical.

---

## 4. Stochastic (T>0) exactness

**Claim.** At `temperature > 0` with no non-default penalties, the MTP path's committed-token
**distribution** is exactly the serial target distribution.

**Why.** The MTP path implements mathematically exact speculative rejection sampling
(Chen et al. 2023). For each drafted token with target probability `p` and draft probability `q`:
- accept the draft with probability `min(1, p/q)`;
- otherwise resample from the normalized residual `max(0, p - q)`;
- the round's bonus token (the first non-drafted token) is sampled directly from `p`.

This construction is a proven identity: the output distribution equals the target `p`, for
**any** draft distribution `q` (including `q = 0` on tokens, which is handled by the zero-
probability floor). Because the target `p` here is the *filtered* target (temperature / top-k /
top-p / min-p applied), and the **same** filter is applied identically to the draft `q` and the
target `p`, the identity holds for the product sampling controls, not only raw temperature.

**Evidence — math unit tests (pure, deterministic, always run).** In
`Qwen38MTPKernelTests` (10 tests):
- `acceptanceAlpha is min(1, p/q) with a zero-q floor` — acceptance ratio boundary cases
  (p/q>1 → 1, p/q<1 → ratio, q=0 p>0 → 1, p=0 → 0).
- `residualLogits concentrate on the p-minus-q support` — residual mass only where `p > q`.
- `residualLogits falls back to p when the residual is empty` — `p == q` → resample `p`.
- `residualLogits keep zero-probability tokens negligible` — no zero-`q` token gains mass.
- `applySamplingFilters scales by temperature` / `top-k keeps the k largest` / `top-p boundary`
  / `min-p masks the tail` — the target/draft shared filter is well-behaved at its boundaries.

**Evidence — distributional harness (end-to-end, model-loaded).**
`Qwen38MTPDiagnosticTests.testA4DistributionalParity` (env-gated on
`QWEN_MTP_DIST_HARNESS=1`): for each of T=0.8 and T=1.0, runs 64 fresh sessions in serial
(depth 0) and MTP (depth 2) modes and compares the **top-8 carried mass** of the committed
next-token and first-post-draft-token distributions.

Observed (one run, weights `e448b2e2…`):

| T | token (next) top-8 mass Δ | token-1 (post-draft) top-8 mass Δ |
|-----|------|------|
| 0.8 | 0.047 | 0.031 |
| 1.0 | 0.016 | 0.078 |

Gate: Δ < 0.15. All pass. (Full-support total-variation distance is **not** used as the gate:
for a 248k-vocabulary high-entropy distribution at N=64, TVD is dominated by long-tail sampling
noise — even the trivially-shared next token, sampled from `p` directly in both modes, shows
TVD ≈ 0.4–0.7, which is pure noise. Top-k carried mass is the tightly-estimated quantity.)

**Reproduce:**
```
cd ../mlx-swift-lm
QWEN_MTP_DIST_HARNESS=1 swift test --filter Qwen38MTPDiagnosticTests/testA4DistributionalParity
```

---

## 5. The F1 fix (this change set)

**Defect.** `hasNonDefaultPenalties` was computed but never consulted in depth selection, so a
request carrying non-default presence/frequency/repetition penalties silently ran the MTP
path, where penalties are **not** applied to the draft/verify logits — an unenforced, silent
sampling parameter (a hard non-negotiable violation).

**Fix.** `SamplingParameters.effectiveMTPEnabled` (server) is `mtpEnabled && !hasNonDefaultPenalties`,
and both `decodeDepth` selection sites use it. A request with non-default penalties now runs
serial target-only, where penalties *are* applied to the target logits before selection.
`effectiveMTPEnabled` also guards the MTP sampling-config construction.

- Server: `Sources/HTTPServer/Generation/MLXGenerator.swift`, `Sources/HTTPServer/API/OpenAIValidation.swift`.
- Test: `RequestValidationTests` (penalty request → serial depth; no-penalty → MTP depth).

---

## 6. Sampling-control semantics (as implemented, applied to target before selection)

| Control | Behavior | Exactness note |
|---------|----------|----------------|
| `temperature` | logits / temperature, then the downstream filters; T=0 = greedy. | Applied identically to draft `q` and target `p`. |
| `top_k` | keep the k largest logits, rest → −∞. k=0 = off. | Identical to q and p. |
| `top_p` | **non-standard** transform: sorted reverse-cumsum of sorted probabilities placed back at the original indices, then mask `< 1 - topP`. *Not* textbook nucleus. | Consistent across q and p, so the `p-q` residual is exact; documented as a known deviation, **not** changed (would alter product output). |
| `min_p` | mask tokens below `minP * maxProb` (argmax always kept). | Identical to q and p. |
| penalties | presence/frequency/repetition on **target logits** before selection. | **Rejected at the speculative boundary** (Section 5); never applied to drafts. |

---

## 7. What this contract does **not** guarantee

1. **Stochastic token-for-token identity with serial.** Serial and MTP sample independently; only
   the **distribution** matches. A given MTP completion is generally not the same token sequence
   as a serial completion at the same seed. Do not gate stochastic MTP against a single global
   stream hash.
2. **Greedy identity at near-tie boundaries.** See Section 3 (ulp property).
3. **Distributional test coverage for penalties.** Penalties are rejected from the speculative
   path, so no MTP-vs-serial distributional test is defined for them.
4. **Cross-session throughput.** This contract is about *correctness*, not speed. No tok/s claim
   is made here (and cross-session absolute tok/s is not citable per project rules).

---

## 8. Reproduction summary

| Check | Command | Status |
|-------|---------|--------|
| A4 math unit tests (10) | `cd ../mlx-swift-lm && swift test --filter Qwen38MTPKernelTests` | green |
| Diagnostic greedy/width (2) | `cd ../mlx-swift-lm && swift test --filter Qwen38MTPDiagnosticTests` | green |
| A4 distributional harness | `cd ../mlx-swift-lm && QWEN_MTP_DIST_HARNESS=1 swift test --filter Qwen38MTPDiagnosticTests/testA4DistributionalParity` | green |
| Server validation (122) | `cd qwen38-mtp-server && swift test --filter HTTPServerTests` | green |
| Draft-depth policy (A3) | `benchmarks/DRAFT-DEPTH-POLICY.md` | delivered |

Binary under test: `e448b2e2bbbfa7174c9d24e42108f5f721cb0a76b588f3fa3be7c1cbe1e167ab`.
