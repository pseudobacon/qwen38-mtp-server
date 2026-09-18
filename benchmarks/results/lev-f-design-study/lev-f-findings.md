# LEV-F — Tree-drafting + conversation-resume design study (paper, zero code)

**Run ID:** lev-f-design-study
**Date:** 2026-09-18
**Decision:** **Ranking: Stage-3 conversation resume first; tree drafting as a
follow-on sharing its machinery.** The single dependency that unlocks both is a
**composable generated-state checkpoint** (KV + GDN recurrent `h` + MTP-head
state) at arbitrary sequence points, with **forward-only rollback** (restore to a
saved checkpoint and advance — never an in-place inverse). If that design is
sound, Stage-3 resume is the lower-risk sibling and goes first.

---

## 1. Tree-drafting headroom (from existing data)

### The measured chain baseline (in-session, q4 head, k pinned)

| config | fixture | acc/step | tok/round | step ms | wall tok/s |
|---|---|----:|----:|----:|----:|
| **k2 (default)** | essay | 1.221 | 2.22 | 100.5 | **22.08** |
| k2 | specdec | 1.378 | 2.38 | 111.5 | **21.27** |
| k3 | essay | 1.543 | 2.54 | 126.9 | 20.00 |
| k3 | specdec | 1.765 | 2.76 | 133.1 | 20.70 |
| d5 | essay | 1.710 | — | 184–192 | 14.53 |
| d6 | essay | 1.741 | — | 210–219 | 12.85 |
| d8 | essay | 1.771 | — | 277–283 | 9.80 |

**Acceptance plateau: ~1.7–1.8 acc/step** at deep chain depths (d5/d6/d8). k2
sits at 1.22–1.38. The headroom over k2 is therefore bounded at
**+0.35–0.58 tok/round** (1.8 − 1.22 essay; 1.8 − 1.38 specdec).

**Verify cost is superlinear in width.** The marginal verify row is
**6.65 ms** (tape, PROFILE-K2 §3) and ~**15.9 ms** end-to-end (k2→k3 eval delta,
DRAFT-DEPTH-POLICY). Step grows 100.5 → 126.9 → 185 → 212 → 283 ms as depth
rises, *faster than linear* — so even though acc/step rises toward the
plateau, **wall tok/s monotonically collapses** (22.08 → 20.0 → 14.53 → 12.85 →
9.80). k3 (already near the plateau) is *net-negative* vs k2 on wall tok/s.

### SDPA exactness chunk geometry (the hard constraint)

The fused vector-kernel SDPA handles verify widths via a **5-row split**
(`L ∈ 6..9` = 5-row + (L−5)-row pair of `.causal` SDPA calls, Phase 4 Bug B fix).
Widths **beyond 9 rows leave the fused vector kernel** entirely and fall to the
reference matmul→fp32-softmax→matmul path — the **Bug B** regime that grossly
corrupts wide-verify streams. So a tree's verify width is bounded to **≤ 9 rows
(ideally ≤ 5 for the single fused call)** before it pays the reference-path
penalty.

### Headroom model: when does a tree beat chain k2?

A tree of branching B / depth D verifies ~B·D nodes (one row each). To beat
k2 (22.08 tok/s essay) it must achieve
`(acc/step + bonus) / step_ms > 22.08`.

- **To reach the 1.8 plateau** the tree needs effective depth/width ≈ 3–4,
  i.e. **verify ≥ 4–5 rows** (just at the fused-call limit).
- At verify = 5 rows: step ≈ 100.5 + 2×6.65 (extra tape) + ~4 ms (extra head)
  ≈ **118 ms**; tok/round ≈ 2.8 → **~23.7 tok/s** — marginally above k2, *only if
  the tree fully captures the 1.22→1.8 acceptance headroom*.
- At verify = 6–9 rows (5-row split): step ≈ 125–140 ms; tok/round capped ~2.8
  → **~20–22 tok/s** — at or below k2.
- At verify > 9 rows (reference path / Bug B): net-negative.

**Conclusion:** the headroom is **marginal (< ~5 %)** and only realizable by a
*small, shallow tree* whose verify width stays **≤ 5 rows** (the fused
single-call limit). It is *uncertain* because it requires the tree to actually
capture the acceptance headroom (a property of the model's per-position draft
accuracy, not the chain structure) — and k2 already dominates every deeper
*chain* on wall tok/s. A tree does not raise the acceptance ceiling; it only
reduces the chance that an early divergence wastes the remaining draft budget.

## 2. Shared blocker with Stage-3 conversation resume

Both levers need the **same** primitive:

> **Checkpoint and restore the full *generated* state** — KV cache + GDN
> recurrent `h` (the `MambaCache` per-layer state) + MTP-head state — at an
> arbitrary sequence point, then advance by prefiling only the suffix.

- **Stage-3 conversation resume:** store that state at the *end of the
  assistant turn* (prompt + generated tokens). The next user turn resumes from
  it and prefills only the new user tokens — avoiding re-prefilling the whole
  conversation. (Today the radix stores per-node `state`/`metaState` for
  **prompt prefixes** — Tasks 3/4 + SSD; Stage-3 extends it to **generated**
  state.)
- **Tree drafting:** store that state at each **branch point**, branch, and
  roll back by restoring to the branch-point checkpoint.

**The rollback-composition question the RFC flags:** the GDN recurrent `h` is
a *linear-attention state* — the map tokens→`h` is **not invertible** (you
cannot "subtract" a token from `h`). Therefore:
1. **Rollback is forward-only:** restore a *previously saved* checkpoint and
   advance from it; there is no in-place undo. Every branch point / turn
   boundary must save `h` (and KV) *before* it is mutated.
2. **Composition = restore + suffix-prefill:** "composing" two states means
   loading checkpoint C and evaluating the token suffix `[C .. now]` on top of
   it — not merging two `h` matrices.
3. **Memory cost:** a full generated state is 64 KiB/token (KV) + 48 GDN-layer
   `h` matrices + head state. Storing this at many radix nodes (one per turn,
   or per branch point) is a large, unbounded-in-practice memory load that must
   be budgeted in `MemoryAdmissionPolicy` and evicted (LRU) like the existing
   cache.

The radix-ssd-persistence RFC already serializes *every stateful node*
(`state`/`metaState`) for prompt prefixes; the Stage-3 / tree-drafting work is
the **generated-state** analogue of that, plus the forward-only-rollback
discipline and the admission budget for many large checkpoints.

## 3. Ranking and recommendation

| lever | risk | benefit | dependency |
|---|---|---|---|
| **Stage-3 conversation resume** | **low** — reuses the existing radix state-store (prompt prefixes) + forward-only rollback | bounded, clear: multi-turn TTFT (avoids re-prefilling generated tokens) | generated-state checkpoint |
| **Tree drafting** | **high** — marginal headroom (<5 %), uncertain acceptance capture, superlinear verify cost, SDPA ≤9-row constraint, restructures the draft/verify path | speculative: decode tok/s if the tree captures the 1.22→1.8 headroom at ≤5 verify rows | generated-state checkpoint **+** the tree draft/verify restructure |

**Recommendation:**
1. **Build the composable generated-state checkpoint first** (forward-only
   rollback, admission-budgeted, SSD-evictable). This is *the* shared
   dependency.
2. **Ship Stage-3 conversation resume on top of it** — it is the lower-risk
   sibling: it needs no change to the draft/verify path, reuses the checkpoint
   directly, and has a bounded, measurable TTFT win.
3. **Treat tree drafting as a follow-on** that shares the same checkpoint
   (save state at branch points) and *only* pursues it if a cheap prototype
   shows a tree actually captures the 1.22→1.8 acceptance headroom at a verify
   width ≤ 5 rows. The LEV-E closure (draft-select walk is ~1–2 ms/round) and
   the superlinear verify cost mean the bar is high; do not open it without
   that evidence.

**Gate for the implementation campaign (LEV-L):** the generated-state checkpoint
design must be sound (forward-only rollback proven, memory bounded/evictable,
bit-exact restore). Stage-3 resume is gated on that design; tree drafting is
gated on that design **plus** a measured acceptance-capture prototype.

## Reproduce / sources
- Chain baseline + plateau: `benchmarks/DRAFT-DEPTH-POLICY.md` (sustained
  1024-table), `progress.md` Phase 4 deep-k gate (d5/d6/d8).
- Marginal verify row: `benchmarks/PROFILE-K2.md` §3 (6.65 ms/row).
- SDPA 5-row split / Bug B: `progress.md` Phase 4 (Bug B fix), engine
  `Qwen38MTPBlockSession.swift` (`attentionWithCacheUpdate` exactness chunk).
- State-store / rollback: `docs/radix-ssd-persistence-rfc.md`
  (`state`/`metaState` per node), `docs/SESSION-BOUNDARIES.md` §3
  (clone/restore KV/GDN/MTP/sampler/rollback state).
