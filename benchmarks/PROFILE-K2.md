# PROFILE-K2 — k=2 speculative round: non-backbone overhead decomposition

Measured 2026-09-14, engine `609e0d5` (feature branch `feature/k2-decomposition`),
server `f6ac7bc` (same branch), MacBook M5 Pro 48 GB, Qwen3.8-27B 4-bit + q4 native MTP head.
All cells single-stream, release binaries, no parallel builds/tests during measurement.

Question: the k=2 round is ~86–112 ms depending on context and thermal state, while
serial decode ("backbone") is ~56–60 ms. Where does the ~30–50 ms of non-backbone
cost go, and how much of it is addressable by kernel work?

## 1. Method and cross-validation

Three independent measurements, all agreeing within noise:

| Source | Context | Serial M=1 tE | k=2 round | Instrument |
|---|---|---|---|---|
| Phase 2 algebra (`run_k2decomp.sh`, 12 cells/config, 6 reps × 2 fixtures) | s: 38→1062, k2: 38→499 | s stepAvg 59.78 | k2 stepAvg 100.73 | host phase timers (`QWEN_MTP_STEP_TRACE=1`) |
| Phase 3 trace (`run_k2trace.sh`, 110 rounds, xctrace Metal System Trace, offset-calibrated) | 256 | FullBench serial@256: 55.77 | k2 round 86.0 (tE 80.4) | host timers + GPU interval union |
| FullBench serial mode (engine `FullBench --serial`) | 256 / 1062 / 3072 | 55.77 / 56.04 / 56.80 | — | wall clock around blocking `MLX.eval` |

FullBench serial mode (one cache, prime once, N back-to-back M=1 decodes) reproduces the
in-pipeline serial cell to 0.2 ms (56.04 vs 56.24 ms, ctx 38→1062). It is the clean
backbone reference. Per-rep FullBench mode (newCache + prime per rep) is **not** a valid
serial reference — see §5 (bench artifact).

A/B gate: `MLX_QWEN_MTP_TRACE=1` (the 5-way host timers + anchor line) estimated at
**+2.19 ms/round** in the 4-pair alternating A/B — below the phase resolution used here
and accounted for in the trace numbers as "instrumentation, measured, not assumed".

## 2. Round structure (boundary map, condensed)

One k=2 round in `Qwen38MTPBlockSession.generateRound` (16 host segments A–P):

```
A invariants/entry → B head flush graph (d+1=3 rows) → C draft id row 0 →
D asyncEval(flush)          [head GPU work starts, host continues]
E step chain graph (d-1=1) → F asyncEval(chain) →
G snapshot/cache-state bundle → H verifyTokens concat →
I backbone verify graph (M=3) → J top-2 build →
K blocking eval             [the ONLY device→host sync in the round]
L host readback (top-2 rows) → M accept walk / commit / rollback →
N head history upkeep → O trace output → P server tokenize/SSE
```

The head flush+chain is submitted via `asyncEval` **before** the blocking eval, so its
GPU work overlaps the host's verify-tape construction (segments G–J). The single blocking
eval drains: head flush tail + head chain tail + full M=3 verify tape.

Existing host-measurable boundaries (phase timers): tG = graph build (I+J, plus head
graphs in D–F), tE = blocking eval (K), tH = readback (L), tC = commit/rollback (M+N).

## 3. Decomposition table

### Same-context (256 tokens) — cleanest comparison

| Component | Serial M=1 | k=2 round | Δ |
|---|---|---|---|
| Host: graph build tG | 1.02 | 4.5 | +3.48 (verify tape M=3 + head graphs + top-2) |
| GPU: eval window (wall tE) | 58.3 | 80.4 | +22.1 |
| — GPU busy in window | 55.6 (util 96.0%) | 79.8 (util 98.4%) | +24.2 |
| — host gap in window | 2.7 | 0.6 | −2.1 |
| Host: commit/rollback tC | 0.006 | 1.1 | +1.09 |
| Host: readback tH | 0.012 | 0.02 | +0.01 |
| **Round total** | **59.4** | **86.0** | **+26.6** |

Reading of the Δ (26.6 ms non-backbone, same context):

* **~24.2 ms is real extra GPU work** inside the eval window: the M=3 verify tape
  (2 extra rows vs serial; per-row GPU is sub-linear because GDN verify uses a
  parallel scan and QMV verify routes M≥2) plus the head flush+chain tail that the
  blocking eval drains (head flush 3 rows + 1 chain step at depth 2). This is work,
  not overhead — it is what buys the 2 accepted draft tokens.
* **~3.5 ms is host-side**: verify-tape graph construction (MLX C++ graph traversal +
  Metal command encoding — not Swift-addressable in this checkout) + accept walk.
* **No reclaimable host gap**: at 256 context the eval window is 98.4% GPU-busy;
  the host keeps up with the GPU. The round is GPU-saturated, not host-starved.

### In-pipeline, growing context (algebra, 12 records per config)

| Config | rounds (ctx range) | stepAvg | tE | tG | tC | tH |
|---|---|---|---|---|---|---|
| s (serial) | 1024 (38→1062) | 59.78 | 58.75 | 1.02 | 0.006 | 0.012 |
| k1 | 582 (38→620) | 90.08 | 73.62 | 3.28 | **13.17** | 0.015 |
| k2 | 461 (38→499) | 100.73 | 96.61 | 3.41 | 0.69 | 0.014 |
| k3 | 403 (38→441) | 126.17 | 121.67 | 3.57 | 0.92 | 0.015 |

* k1 carries the **dead-path penalty**: `rollbackCheckpoints` is never written, so the
  K=1 reject path always falls to the generic repair (+13.17 ms tC). Without it,
  k1 ≈ 77 ms — between serial and k2, consistent with M=2 verify GPU.
* k2−s = 40.95 ms at different mean contexts (270 vs 550); same-context (256) the
  difference is 26.6 ms (§3 table). The context term (serial tE is flat: 55.8→56.8
  across ctx 256→3072) means k2's context growth, not serial's, explains the rest.
* Headline k2default run (6 reps, thermal drift within one session): k2-essay round
  89.5 → 102.5 ms across reps. The "107–112 ms" label is the warm end of that
  trajectory; the cold end is ~86 ms. One live number per fact; the spread is
  thermal, not structural.

## 4. QMV routing (kernel-path note)

M=1 verify inputs (2-D decode, B·L=1) fall back to the incumbent `quantizedMM`;
M≥2 (B·L=2..9) route through the QMV verify kernel. Both paths are fused
(sw-iGLU 64/64, qkv 16/64, gdn 48/64). The M=1 fallback is why serial and k=2
verify rows are not directly per-row comparable; the sub-linear per-row cost of the
M=3 tape is a property of the QMV/scan path, not a bug.

## 5. Findings

1. **The k=2 round is GPU-saturated at measured contexts.** Eval-window utilization
   98.4% (256 ctx, 110 rounds). The ~27 ms non-backbone cost is ~90% real GPU work
   (extra verify rows + head) and ~10% host graph construction. There is no hidden
   host gap to reclaim in-pipeline at these contexts.

2. **Kernel-addressable surface is small.** Fusion is already at its documented
   ceiling (swiGLU 64/64, gdn 48/64, qkv 16/64). The levers are: draft depth
   (k=1: −~16 ms GPU but −2 accepted tokens and the +13.2 ms dead-path repair —
   net worse), head flush size (fixed by the d+1/d−1 design), and the host tape
   build (3.4 ms, MLX C++ — outside this checkout). No single kernel in the window
   dominates: ~56–68 compute intervals of ~0.8–0.9 ms each, back-to-back.

3. **K=1 dead path confirmed and quantified** (tC 13.17 ms, 12 records). Evidence
   for a follow-up (write `rollbackCheckpoints` on the K=1 path); not fixed here —
   k=1 is not a production config.

4. **FullBench per-rep first-decode penalty is a bench artifact, not a model
   property.** Per-rep mode (newCache + prime + one measured decode):

   | prime | per-rep M=1 eval wall (med) | serial M=1 at same ctx |
   |---|---|---|
   | 1 | 58.4 | — |
   | 256 | 56.48 | 55.77 |
   | 512 | 66.9 | — |
   | 1024 | 116.0 | — |
   | 2048 | 117–156 | 56.8 |

   Serial mode (one cache, N back-to-back decodes) is **flat 55.8→56.8 ms across
   ctx 256→3072** and matches the in-pipeline serial cell (56.2 ms). The per-rep
   penalty is paid only on the *first* decode after each prime (`--widths 1,1`:
   first 122 ms, second 56 ms, same cache), persists after 10 warmup reps, and
   scales with prime size. xctrace of the slow first decode (pid 73984, offset
   calibrated): 68 compute intervals, union 60.5 ms, ~1.2 ms GPU-idle gaps, host
   cpu→gpu stride ~2.2 ms/kernel vs ~0.8 ms/kernel in the fast case — the host
   MLX flush/encode loop falls behind after a fresh large prime + newCache cycle.
   Mechanism is in MLX's C++ eval flush (not readable in this checkout — the
   `mlx-swift` checkout is Swift-only); isolated SDPA is 0.3–0.5 ms across
   K=256–4096 (not the gap), cache slice update is O(1) host-side, GDN S=1 update
   is a single kernel with no readback, and no fire-and-forget eval exists in the
   forward path. **Conclusion: prior "M=1 non-monotonic in context" readings from
   per-rep FullBench are retracted; serial decode is flat ~56 ms in context up to
   3072 tokens in this process configuration.** In-pipeline at real server contexts
   (2–8k) has no per-round newCache/prime, so the in-pipeline path is not expected
   to carry this penalty; the flat serial curve and the 98.4% util at 256 ctx are
   the evidence for that.

5. **Instrumentation is measured-zero-artifact-ish, not assumed**: +2.19 ms/round
   estimate from the 4-pair A/B (trace on/off), below phase resolution used here.

## 6. Closing verdict on kernel-addressability

At the measured contexts, the k=2 round (86 ms cold @256 ctx, 90–102 ms warm
in-session @~270 ctx) is ~93% GPU-busy with 98.4% eval-window utilization. The
non-backbone ~27–41 ms is dominated by *necessary* GPU work (2 extra verify rows +
3-row head flush + 1 chain step) plus ~3.5 ms host tape construction in MLX's C++
flush. **No kernel-level win of order 10 ms is addressable from this checkout**
without either (a) changing draft depth (rejected: acceptance cost + k=1 dead
path), (b) changing the head design, or (c) C++ work in MLX (tape-build flush).
The next-order target if this matters is the host tape build (3.4 ms tG) and the
K=1 repair path — both outside the current production config's hot path.

## 7. Reproduction

```bash
# algebra (48 cells + A/B): see benchmarks/run_k2decomp.sh
bash benchmarks/run_k2decomp.sh algebra
# trace run (110 rounds @256 ctx, xctrace Metal System Trace): run_k2trace.sh
# FullBench serial reference (engine repo):
cd ../mlx-swift-lm && swift build --configuration release --product FullBench
.build/arm64-apple-macosx/release/FullBench \
  --model /Users/cwong/ai/qwen38-mtp-server/weights --serial --prime 256 \
  --reps 64 --warmup 2
```

Trace artifacts (session-local, /tmp): k2trace-mtp-trace.log, k2trace-gpu-intervals.xml
(offset 696,415,092,781,278), fb1-profile.trace (per-rep M=1@2048, pid 73984),
fb256-profile.trace (per-rep M=1@256, pid 76137, offset 702,590,931,325,346).
