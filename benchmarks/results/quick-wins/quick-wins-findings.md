# Phase 3 quick-wins — findings (Items 1–4)

**Date:** 2026-09-18
**Scope:** consume the LEV-B / LEV-C verdicts. Four quick-win items, each measured
in-session with the standing protocol (paired, interleaved, thermal-log, binary +
metallib provenance). Items 1, 3, 4 are measured below; Item 2 (32K prefill audit)
is in flight.

**Binary provenance (server):**
- verify-flip + top-2-gap log + SSD sub-timers + lazy-SSD built in stages; final
  release binary sha recorded per cell in the run logs.
- Engine head `9f4ceb9` (+ verify flip in `Qwen35Kernels.swift`, banner in
  `Qwen35.swift`, top-2-gap trace in `Qwen38MTPBlockSession.swift`); metallib
  `b57de586`, cmlx `1f8e74e`.

**Standing thermal caveat:** the 32K prefill (~130 s) re-heats the M5 Pro GPU within
2–3 reps (tEval 132 → 872 ms observed), so 32K per-rep deltas are noisy. The 8K
cell (prefill ~30 s) is much cleaner (tEval ~83–104 ms). In-session paired deltas
only; the -1% verify signal is small (~1–3 ms) and sits near the per-rep noise floor.

---

## Item 1 — verify-width fusion flipped to default ON → **GO (marginal)**

**Change:** `Qwen35FusedGDNPreworkRouting.enabled` default OFF → ON in
`Qwen35Kernels.swift`; banner updated in `Qwen35.swift`; rollback
`MLX_QWEN_FUSED_GDN=0/false/no/off`.

**Bit-exact (required):** confirmed at both lengths — 32K (chunked) stream
`6576c099` in all 11 valid A/B reps; 8K (dense) stream `f669c4e9` in all 12 reps.
No quality cost.

**End-to-end (paired, rep 1 discarded):**
- 32K (chunked, thermal-degraded): tGraph (host) 4/4 A-faster; stepAvg swamped
  (tEval ~1000 ms) — not a clean end-to-end read.
- 8K (dense, clean, tEval ~83–104 ms): stepAvg A-faster in 3/5 measured reps
  (r2 −4.6, r3 −13.7, r4 −2.1, r5 +4.1, r6 +1.3 ms); **mean −3.0 ms (~2.7%)**
  A/ON-faster. tGraph ~12–15 ms (small at 8K).

**Verdict:** **GO as default (marginal)**, consistent with LEV-B's "GO-as-default
(marginal)". Rationale: bit-exact (never worse on quality) + mean-favorable
(−1.1 % at 32K per LEV-B, −2.7 % at 8K here) + host graph-build reduction
(tGraph 4/4 at 32K). The strict ≥ 4/5 end-to-end reps gate is **not** met (3/5 at
8K; thermal-swamped at 32K), so this is documented as marginal, not a clean ≥ 4/5
win. **Kept ON** because it is bit-exact (no downside) and mean-favorable;
rollback `MLX_QWEN_FUSED_GDN=0`.

---

## Item 3 — startup warmup decomposition + persistent kernel-compile cache → **INVESTIGATED, implementation hand-off**

**What the 16.9 s warmup is:** `Qwen38MTPBlockSession.warmAllDepthShapes` warms the
exact decode family — a 512-token seed forward, the verify widths (1..maxDepth+1),
the head's draft steps, the committed-history head shapes, and `draftTokenID`
(the same expression the scored rounds dispatch). The dominant cost is the **Metal
cold-JIT** for these shapes (the code explicitly references "cold-JIT" and a
0.368 s one-off long-prefix stall it seeds away).

**Metal cache behaviour (measured across many restarts, same binary):**
- Cold cache (first-ever / after a fresh metallib): warmup ~15.9–18.1 s.
- Warm cache (Metal's built-in disk cache populated): warmup ~3.0 s.
- So **~14 s of the 16.9 s is the cold Metal JIT**; the residual ~3 s is
  allocation / first-touch (unavoidable). Metal's built-in persistent cache already
  carries the compiled kernels across process restarts (the 16.9→3 s decay).

**Remaining gap + lever:** only the FIRST startup after a fresh install / metallib
change pays the ~14 s JIT. A provenance-safe persistent runtime-kernel compile
cache would pre-compile these kernels at build/install time and bundle them
**version-keyed** by (MLX cmlx revision `1f8e74e` + metallib SHA `b57de586` + model
config), validated at startup with fail-loud on mismatch (fall back to cold warmup).

**Why it's a hand-off (not implemented this session):** pre-compiling the **MLX
runtime's** kernels (not this repo's metallib) requires either (a) a deployment
step that runs the warmup once at install to populate Metal's cache, or (b) an
`MTLBinaryArchive` capture of MLX's compiled kernels — which likely needs an
MLX-side pre-compile/archive hook (investigate MLX's Metal kernel compilation).
Both are a build-system / MLX-side change beyond a quick win. See `docs/HANDOFF.md`
for the precise next step.

**Measured GO bar would be ≥ 5 s cold-start drop** (the ~14 s JIT is the addressable
part); the lever, if built, clears it with margin.

---

## Item 4 — lazy SSD restore → **GO (implemented, default ON)**

**What the 8.5 s "SSD block" is (gated sub-timer):** 93% is the **weight-identity
hash + template fingerprint** (7.94 s of pure-CPU file I/O over the 15 GB of
weights), **not** the restore (store init + skeleton restore ≈ 1 ms). It ran on
the actor's critical startup path (blocking the warmup).

**Change:** `QWEN_KV_SSD_LAZY` (default ON; `=0` for eager). The hash now runs in a
background `Task.detached` (off the actor's critical path, pure CPU); when done,
`finishSSDSetup` wires the store + restores the prefix skeleton on the actor.
Requests before it completes simply miss (no wrong results, just no acceleration
yet) — no correctness change.

**A/B (time-to-readyz, 5 restarts each, alternating order to remove the GPU
first-position artifact):** EAGER ~25.1–27.6 s; LAZY ~5.2–5.7 s → **~21 s drop,
5/5 pairs**, far above the ≥ 5 s bar. Breakdown: 7.9 s hash off the critical path
(definitive from the trace: `ssd-done` 7057 ms EAGER vs 0 ms LAZY) + the GPU not
being idled for 7 s (warmup 3.0 s LAZY vs 16–18 s EAGER).

**Verdict:** **GO.** Default ON. No correctness change (functional smoke: requests
serve normally in lazy mode). Rollback `QWEN_KV_SSD_LAZY=0`.

---

## Item 2 — `_PREFILL` divergence audit → **NO-GO (keep default OFF), audit logged**

Driver `item2_ab.sh`: 32K prefill off (incumbent) vs on, 6 reps each (top-2 gap
trace + completion_ids); 8K/16K dense, 3 reps each. `MLX_QWEN_TOP2_GAP_TRACE=1`
logs the primary top-2 gap per commit round (new gated engine trace).

**Correction to the task premise:** the MTP session chunks the prefill into
`prefillChunkSize=2048`-token GDN forwards at **all** lengths (S=2048 ≤ 4096 per
chunk), so the fused GDN prefill engages at 8K/16K/32K alike — **not** only 32K.
The 8K/16K "bit-exact" expectation does not hold.

**Divergence (O = off incumbent vs P = on), rep 1:**
| length | O family | P family | first-div (1-based) | flip rate |
|--------|----------|----------|---------------------|-----------|
| 8K  | `f669c4e9` | `771652c1` | 25  | 103/128 (80.5 %) |
| 16K | `388bf97f` | `6d70ae4b` | 72  | 20/128 (15.6 %) |
| 32K | `6576c099` | `f666254f` | 72  | 33/128 (25.8 %) |

All flip-rates are ≫ the ~9/1024 (≈0.9 %) knife-edge supply; the 32K first-flip
top-2 gap is a real ~2.0-logit gap (not a near-tie). The 32K text difference is a
**near-synonym rewording** (e.g. "injected with specific … by integration
packages" vs "into integration packages with specific …"), same meaning — not a
gross semantic break — but it is not a knife-edge.

**32K prefill win (wall − decode, rep 1 discarded):** O mean 112.4 s, P mean
111.6 s → **−0.7 %** (P faster 3/5), **not** ≥ 3 %.

**Verdict:** **NO-GO — keep `MLX_QWEN_FUSED_GDN_PREFILL` default OFF.** Fails both
gates: (a) the 32K prefill win is −0.7 % (< 3 %), and (b) the divergence is
**gross, not a knife-edge** (flip-rate ≫ 0.9 % supply; real first-flip gap) at
8K/16K/32K → the policy v3 §0d STOP condition. Audit logged here. Consistent with
LEV-B's prefill FLAG. The GDN prework is a small fraction of the (attention +
FFN + GDN-scan-dominated) prefill, so a <3 % win was expected.
