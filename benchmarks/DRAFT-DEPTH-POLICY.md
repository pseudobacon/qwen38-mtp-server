# Draft-Depth Policy Sweep (2026-09-15)

**Verdict: KEEP the fixed k = 2 default.** No tested depth beats k = 2 in any
mode; no mode exists in which a lower or higher depth wins; no adaptive policy
is justified. No default change, no engine or server source change.

## Provenance

| field | value |
|---|---|
| server repo | `qwen38-mtp-server` @ `800218e` (branch `feature/draft-depth-policy`), clean at task start; docs/bench-only changes since |
| engine repo | `mlx-swift-lm` @ `97a9d85` (main), clean, untouched |
| binary | `.build/release/qwen38-mtp-server` SHA-256 `e448b2e2bbbfa7174c9d24e42108f5f721cb0a76b588f3fa3be7c1cbe1e167ab`, mtime 2026-09-14T13:39:06 — identical across all 76 timed cells (checked per result file) |
| head | q4 (4-bit quantized, default ON) — `head_gate PASS (q4)` in every cell |
| fusion | `backbone swiGLU 64/64 qkv 16/64 gdn 48/64; head swiGLU 1 qkv 1` in every cell |
| protocol | greedy `temperature 0`, `enable_thinking: false`, `top_k 1`, fixed fixture, `finish_reason: length`, `QWEN_MTP_STEP_TRACE=1`, fresh server per cell (port 18099), sequential only, thermal snapshot per cell, rep 1 warmup, reps 2–6 measured, rotating cell order per rep |
| gates (per rep) | stream hash (registered / per-family — see Near-tie property), `completion_tokens` = mode length, `finish_reason = length`, `phaseSumOK = true`, constant `depthDist` equal to the cell's d, q4 head gate. Any failure stopped the stage. All stages completed 100 % gated. |

## Flag / actual-depth mapping (audited 2026-09-15)

Verified against `ServerConfig.swift`, `MLXGenerator.swift` (`decodeDepth`),
`Qwen38MTPBlockSession.draftPolicy` and the startup log line
`MLXLM: MTP draft depth:`.

```
server offer  = --spec-draft-n-max          (default 3; 0 = serial; max 8)
              = decodeDepth offered every round (mtp_enabled true)
actual depth  d = draftPolicy(offer) = min(offer, QWEN_MTP_DRAFT_K ?? 2)
verify width  M = d + 1
tokens/round  = 1 (primary) + accepted drafts (≤ d)
```

| label | flags | d | M | observed `depthDist` (all cells) |
|---|---|---|---|---|
| s  | `--spec-draft-n-max 0` | 0 | 1 | `0:…` (one per round, 1024/128 rounds) |
| k1 | `QWEN_MTP_DRAFT_K=1` | 1 | 2 | `1:582` essay, `1:566` specdec |
| k2 | `QWEN_MTP_DRAFT_K=2` | 2 | 3 | `2:461` essay, `2:431` specdec |
| k3 | `QWEN_MTP_DRAFT_K=3` | 3 | 4 | `3:403` essay, `3:371` specdec |
| k4 | `QWEN_MTP_DRAFT_K=4 --spec-draft-n-max 8` | 4 | 5 | `4:388` |
| k6 | `QWEN_MTP_DRAFT_K=6 --spec-draft-n-max 8` | 6 | 7 | `6:375` |
| k8 | `QWEN_MTP_DRAFT_K=8 --spec-draft-n-max 8` | 8 | 9 | `8:371` |

No requested depth was silently capped: every cell's `depthDist` is constant
at the requested d (runner depth gate). k8 is natively supported (verify width
9; deepk precedent) and was included in Stage A.

## Near-tie property (new finding, pre-existing)

Greedy streams are **per verify-width family**, not globally identical, on
`specdec-800`:

| family | stream hash (specdec-800, 1024) |
|---|---|
| M=1 (serial) | `c70882fc40a2…` |
| M=2 (k1) | `a3dfa8621b06…` |
| M=3 (k2) and M=4 (k3) | `139acb9d30fe…` (the registered k=2 stream) |

- First M=1 vs M=3 divergence at completion token **989/1024** (located by
detokenizing both streams; reproduced twice). M=2 diverges from M=3 as well.
- On `essay-1024` all families (M=1…M=9) agree for all 1024 tokens.
- Cause class: per-width QMV/SDPA accumulation order differs by a few ulps;
a rare near-tie argmax flips. Each family is deterministic per (binary, d) —
reproduced cell-for-cell across the whole matrix.
- Consequence for this task: streams were gated per family (essay cells
against the single registered hash; specdec/128 cells against per-(mode, cell)
family hashes learned on first encounter, k2 family asserted equal to the
registered `139acb9d…`). Times are unaffected and fully comparable.
- The first 128-token prefixes of all specdec families are identical
(`6eb4c26a…`), so interactive cells are cross-depth comparable.
- **This is a pre-existing numerical property of the width kernels, not a
regression from this task.** It means the "speculative output token-identical
to serial greedy" invariant holds in practice (no divergence on the essay
fixture, none within 128 tokens on specdec) but is not guaranteed at the
near-tie level for long specdec-style generations. Tracked as an open item;
fixing it is a kernel/numerics task, explicitly out of scope here.

## Stages executed

| stage | scope | cells | reps | result |
|---|---|---|---|---|
| A (recon) | essay1024 | s k1 k2 k3 k4 k6 k8 | 1 | all gated; k2 best; k4/k6/k8 steeply dominated → cut |
| B (primary) | essay1024 | s k1 k2 k3 | 6 (rotated) | k2 best |
| C (generalize) | specdec1024 | s k1 k2 k3 | 6 (rotated) | k2 best |
| C (interactive) | essay128, specdec128 | s k1 k2 k3 | 6 (rotated) | k2 best |

Raw results (one JSON line per rep, full provenance per line):

```
benchmarks/results/dpsweep-A-essay1024-{s,k1,k2,k3,k4,k6,k8}.jsonl   (7)
benchmarks/results/dpsweep-B-essay1024-{s,k1,k2,k3}.jsonl            (4)
benchmarks/results/dpsweep-C-specdec1024-{s,k1,k2,k3}.jsonl          (4)
benchmarks/results/dpsweep-C-essay128-{s,k1,k2,k3}.jsonl             (4)
benchmarks/results/dpsweep-C-specdec128-{s,k1,k2,k3}.jsonl           (4)
thermal logs: .tmp/dpsweep-{A,B,C}-*.thermal   run logs: .tmp/dpsweep-*.log
```

(Diagnostic cells outside the matrix: `/tmp/dpdiag-*.json` — the serial/k2
specdec pair used to locate the near-tie divergence.)

## Results (median of reps 2–6; same-session, same-binary)

### Sustained, 1024 tokens

| fixture / length | depth | step ms | tok/round | acc/step | decode tok/s | wall tok/s | Δ vs k2 | verdict |
|---|---|---|---|---|---|---|---|---|
| essay-1024 | s  | 59.7 | 1.00 | —      | 16.74 | 16.68 | −24.2 % | dominated |
| essay-1024 | k1 | 90.7 | 1.76 | 0.761  | 19.37 | 19.07 | −12.3 % | dominated |
| essay-1024 | **k2** | **100.5** | **2.22** | **1.221** | **22.08** | **21.95** | — | **best** |
| essay-1024 | k3 | 126.9 | 2.54 | 1.543  | 20.00 | 19.78 | −9.4 %  | dominated |
| specdec-800 | s  | 65.0 | 1.00 | —      | 15.36 | 15.31 | −27.8 % | dominated |
| specdec-800 | k1 | 97.1 | 1.81 | 0.809  | 18.61 | 18.52 | −12.5 % | dominated |
| specdec-800 | **k2** | **111.5** | **2.38** | **1.378** | **21.27** | **21.15** | — | **best** |
| specdec-800 | k3 | 133.1 | 2.76 | 1.765  | 20.70 | 20.57 | −2.7 %  | within noise of k2; never ahead |

Within-cell spread (min–max of reps 2–6, % of median): essay 3.2–7.2 %,
specdec 9.3–14.7 % (specdec session ran hotter/later — thermal; the
interleaved rotation keeps it shared). k2's essay margin (≥ 9.4 %) exceeds
noise; the specdec k2-vs-k3 margin (2.7 %) is inside noise, and k2 is never
worse — on per-rep data k2 takes the best rep in both sustained fixtures.

Stage A (single cold reps, recon only — not used for the headline): k4 19.47,
k6 14.04, k8 10.37 tok/s on essay — monotone collapse from k3 on; diminishing
returns begin at k3, severe by k6.

### Interactive, 128 tokens

| fixture / length | depth | step ms | tok/round | decode tok/s | Δ vs k2 | verdict |
|---|---|---|---|---|---|---|
| essay-1024/128 | s  | 65.2 | 1.00 | 15.13 | −38.8 % | dominated |
| essay-1024/128 | k1 | 93.6 | 1.80 | 19.23 | −22.2 % | dominated |
| essay-1024/128 | **k2** | **92.4** | **2.29** | **24.70** | — | **best** |
| essay-1024/128 | k3 | 114.4 | 2.51 | 21.90 | −11.3 % | dominated |
| specdec-800/128 | s  | 59.5 | 1.00 | 16.77 | −33.0 % | dominated |
| specdec-800/128 | k1 | 88.8 | 1.78 | 19.99 | −20.2 % | dominated |
| specdec-800/128 | **k2** | **94.5** | **2.37** | **25.04** | — | **best** |
| specdec-800/128 | k3 | 128.4 | 2.42 | 18.77 | −25.0 % | dominated |

128-token cells have wide per-rep spreads (13–41 %) because the decode window
is only ~5–8 s against a ~0.25 s fixed overhead plus system jitter (one
rep-wide jitter window is visible across all cells simultaneously, e.g.
essay128 r4); medians still rank k2 first in all four interactive/short
comparisons, and k2's best reps are best overall. The extra draft/verify cost
per round is constant regardless of response length, so there is no regime
where k1's cheaper round wins.

### TTFT

No meaningful TTFT differentiation: prefill is identical for all depths, and
the only depth-dependent first-token cost is the first round (≤ ~100 ms at
k3), far below per-cell server startup variance (wall − decode is flat at
0.23–0.37 s across all cells). Speculation does not harm time-to-first-token.

## Analysis questions

1. **Best median committed tok/s per mode?** k2 in all four modes (22.08 /
   21.27 / 24.70 / 25.04).
2. **Lowest total completion time?** k2 in all four modes (decode and wall).
3. **Lowest TTFT?** No meaningful difference (see above).
4. **Does k1's lower round cost outweigh k2's tokens/round?** No. k2 beats k1
   by 12.3 % (essay), 12.5 % (specdec), 22.2 % (essay128), 20.2 % (specdec128).
   The marginal verify row costs ~15.9 ms (k2→k3 eval delta, sustained) while
   the marginal accepted draft is worth ~0.4–0.6 tok/round — net positive to
   k2, net negative beyond.
5. **Diminishing returns?** Begin at k3 (−9.4 % essay, −2.7 % specdec),
collapse at k4/k6/k8 (−11 / −33 / −51 % cold recon).
6. **Does deeper help only on the speculation-friendly fixture?** No. k3 is
   not ahead of k2 on either fixture; the friendly fixture only narrows the
gap (2.7 % vs 9.4 %).
7. **Above noise?** essay1024 margins yes (≥ 9.4 % vs ≤ 7.2 % spread);
specdec1024 k2-vs-k3 no (2.7 % vs ~10 %), but k2 is never worse and wins
   every other comparison; interactive margins are large in median terms.
8. **Single fixed default vs adaptive?** A single fixed default is
   justified: k2 wins in every tested mode, so no signal (acceptance,
   fixture, length, entropy) predicts a different winner — there is no other
   winner. An adaptive policy would add machinery to choose between
   k2 and options that are never better.

## Recommendation

| use case | recommended policy | evidence | confidence | tradeoff |
|---|---|---|---|---|
| default (all) | keep fixed k = 2 | best median tok/s, decode and wall time in 4/4 modes; margins above noise on the primary sustained workload | high (24 interleaved sustained + 48 short cells, 1 binary, all gates green) | none identified |
| interactive / latency | k = 2 (no separate mode) | k2 best in both 128-token regimes; no TTFT penalty vs k1 | high | — |
| throughput / long | k = 2 (no separate mode) | k2 best in both 1024-token regimes | high | — |
| adaptive by acceptance/entropy | not recommended | no mode where another depth wins; nothing to adapt toward | high | machinery for zero gain |

The positive-change bar (≥ 3 % sustained improvement over k2 beyond noise,
bit-exact stream, no interactive regression) is met by **no** candidate.
**The existing k = 2 default is retained unchanged.**

## Open items (not addressed by this task)

- Near-tie width-family streams on specdec-style long generations (above):
decide whether cross-width greedy identity needs a kernel-level fix; the
registered product stream (k=2 family) is unaffected.
- k4/k6/k8 were not re-run in the Stage B/C protocol (cold recon only);
  they are decisively dominated and not candidates, so this is not a gap.
- Cross-session absolutes remain non-comparable (thermal); this session's
  k2 medians (22.08 / 21.27) are hotter than the registered headline session
  (21.89 / 23.29) — labels only.
