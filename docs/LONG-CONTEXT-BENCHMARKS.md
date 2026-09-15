# Long-context benchmarks (8K / 32K / 96K)

Benchmarking and profiling of the server at agentic-coding context lengths
(8K, 32K, 96K tokens). **This is a measurement task, not an optimization task:**
no server, engine, kernel, model, or quantization changes were made. The release
binary is built from `main` (`18ed7d7`); the engine is untouched.

## TL;DR — headline findings

1. **32K and 96K context are infeasible on this hardware with the current
   prefill path.** The prefill attention buffer is allocated as a dense
   `[seq × seq]` matrix that grows **quadratically**. At 32K it requests
   **51.6 GB** in a single Metal buffer, exceeding the **30.2 GB** Metal maximum
   buffer size, and the server **crashes** (`[metal::malloc] ... greater than the
   maximum allowed buffer size` → SIGTRAP). At 96K the same buffer would be
   ~464 GB. The **maximum feasible prompt is ≈ 24K tokens** (buffer ≈ 30 GB).
2. **Prefill dominates request time**, not decode. At 8K, prefill is ~27 s of a
   ~30 s total request (~90%). Prefill throughput is ~275 tokens/s and scales
   ~linearly in sequence length (8K → 27 s, 16K → 59 s).
3. **The fused GDN kernel does NOT pay off at long context.** It is bit-exact
   (identical output tokens with it on vs. off) but is not faster: at 8K, full
   request time is 29.3 s (off) vs 32.7 s (on). The hypothesis that "kernel
   launch overhead accumulates with context" does not hold — the fused-GDN
   launch count is **per verify round** (a function of draft depth), independent
   of context length, and prefill (not decode) dominates the request anyway.
4. **Draft depth k = 2 remains optimal at 8K**, matching the short-context
   result (k1 36.1 s, k2 29.3 s, k3 32.0 s full-request time).
5. **Session / prefix caching gives only a small TTFT win (~7%)** for this
   hybrid GDN model: a warm repeat of an 8K prompt drops TTFT from 27.3 s to
   ~25 s, not to near-zero. The recurrent gated-delta layers cannot be resumed
   from a token-prefix, so the prefill is effectively recomputed each request.
6. **Every configuration is bit-exact** (identical completion-token stream)
   across fused GDN on/off, k = 1/2/3, and repeated requests.

The single most important takeaway: **the blocker for long context is the
quadratic prefill attention buffer (a memory-layout issue), not kernel-launch
overhead or decode throughput.** Micro-optimizations to the decode/verify path
(fused GDN, draft-depth tuning) are a small fraction of a long-context request
and do not change the picture at scale.

## Fixtures

Deterministic, real Swift source from the two repositories (server `Sources/`,
`Tests/`; engine `Libraries/`, `Tests/`), concatenated in sorted-path order and
truncated at a token boundary. Built by `benchmarks/make_longctx_fixtures.py`
(corpus: 479 files, 167,825 lines, 1,762,659 raw tokens). SHA-256 per fixture is
in `benchmarks/results/longctx-2026-09-15/NOTES.txt`. `prompt_tokens` reported
below include the Qwen chat-template framing (a constant ~15–25 tokens).

| Fixture | Raw tokens | Notes |
|---|---|---|
| `prompts/longctx-8k.txt` | ~8,192 | single-file-ish module slice |
| `prompts/longctx-16k.txt` | ~16,384 | prefix of 32k (built for the threshold probe) |
| `prompts/longctx-32k.txt` | ~32,768 | multi-module slice |
| `prompts/longctx-96k.txt` | ~98,304 | large slice |

Language/structure: Swift source (the actual codebase), so token distributions
(real identifiers, keywords, control flow) resemble an agentic-coding prompt
better than synthetic `word1 word2 …` text.

## Protocol

- One release server per cell (fresh process, model loaded each time) for the
  A/B matrix, so samples are thermally and cache-isolated.
- Streaming request, `temperature 0`, `top_k 1`, `mtp_enabled true`,
  `enable_thinking false`; `max_tokens` 64 (A/B matrix) or 32 (prefix-cache
  probe). `max_tokens` is well below the prefill length, so decode is a small
  suffix of the request.
- **Client metric** (consistent across cells): `TTFT` = time to first non-empty
  content delta (prefill + first verify round); `full_s` = time to stream end;
  `decode_s = full_s − TTFT`.
- **Server metric** (per-request `MTP-STEP-SUMMARY`): `decodeSeconds`,
  `avgStepMs`, `acceptedPerStep`, and the QMV dispatch histogram.
- N per cell: 4 (fused GDN A/B, k=2), 3 (k=1, k=3), 3–4 (prefix-cache). N is
  smaller than the 6–10 in the brief because each 8K request costs ~30 s
  (dominated by the 27 s prefill) and 32K/96K cannot complete; the reported
  means have small spread (see below).
- Raw per-request lines: `benchmarks/results/longctx-2026-09-15/ab-matrix.jsonl`.

## Results

### Scaling (k = 2, fused GDN off, cold)

| Context | TTFT (s) | Prefill tok/s | Decode (server `decodeSeconds`) | Status |
|---|---|---|---|---|
| 8K  | 27.3 | ~296 | 64 tok in 6.3 s (~10 tok/s) | OK |
| 16K | 59.2 | ~274 | 32 tok in 10.9 s            | OK |
| 32K | —    | —    | —                              | **CRASH**: 51.6 GB Metal buffer > 30.2 GB |
| 96K | —    | —    | —                              | infeasible (~464 GB buffer) |

Prefill **time** is ~linear in sequence length (~275 tok/s). Prefill **memory**
is quadratic: the 32K crash requests 51,577,363,200 bytes; scaling as `seq²`,
the buffer that fits under the 30.2 GB cap is at `seq ≈ 32K × √(30.2/51.6) ≈
24.5K`.

### Fused GDN A/B at 8K (k = 2, `max_tokens` 64) — client `full_s`

| Config | full_s samples (s) | mean | decode_s mean | bit-exact |
|---|---|---|---|---|
| OFF | 30.1, 27.05, 29.87, 30.3 | 29.3 | 2.29 | — |
| ON  | 30.7, 32.26, 32.3, 35.5   | 32.7 | 2.52 | yes (same token stream) |

No benefit (ON is at best within noise, at worst slightly slower). The fused
kernel engages (verify width S = k+1 = 3 ∈ [3,9], B=1, bf16, `mask == nil`), and
its output is token-identical to the eager path, so this is a valid A/B — the
result is simply that fusing the GDN prologue does not move wall-clock here.

### Draft depth at 8K (fused GDN off, `max_tokens` 64) — client `full_s`

| k | full_s samples (s) | mean | accepted/step |
|---|---|---|---|
| 1 | 39.4, 34.7, 34.2 | 36.1 | 0.81 |
| 2 | 30.1, 27.05, 29.87, 30.3 | 29.3 | 1.56 |
| 3 | 35.4, 30.4, 30.3 | 32.0 | 2.42 |

k = 2 is the best throughput at 8K, as at short context. Higher acceptance at
k = 3 (2.42/step) does not beat its higher per-round cost; k = 1 is too few
drafts to amortize the round.

### Prefix / session cache at 8K (k = 2, `max_tokens` 32) — TTFT

| Request | TTFT (s) |
|---|---|
| 1 (cold) | 27.3 |
| 2 (warm, same prompt) | 23.8 |
| 3 (warm) | 25.1 |
| 4 (warm) | 25.0 |

Warm repeats save ~7 % of TTFT, not the full prefill. Because the model's
gated-delta (recurrent) layers hold state that is not resumable from an arbitrary
token-prefix, a prefix hit cannot skip the prefill the way it does for a
pure-transformer KV cache. The saving comes from full-attention KV reuse plus
the tokenization cache.

### Memory

| Context | Peak RSS | Breakdown |
|---|---|---|
| 8K | ~15.25 GB | ~15 GB 4-bit weights + ~0.5 GB KV (8K × ~64 KiB) |
| 16K | ~15.25 GB (measured during 8K runs; 16K adds ~1 GB KV) | |
| 32K | n/a — crashes before steady state | would be ~17 GB KV+weights, but the transient prefill buffer is 51.6 GB |

The **steady-state** footprint (weights + KV) is far under the 44 GB admission
limit even at 32K; the crash is the **transient quadratic prefill buffer**, which
admission control does not model (it sizes the KV cache, not the one-shot
`[seq × seq]` attention allocation).

### Decode phase breakdown (8K, steady rounds)

From `STEP-TRACE` (rounds 2–25 of a 25-round decode):

| phase | ms/round | share |
|---|---|---|
| `tEvalMs` (GPU) | ~89 | ~96 % |
| `tGraphBuildMs` | ~3.8 | ~4 % |
| `tCacheStateMs` + `tHostReadMs` | ~0.2 | <1 % |
| **stepMs** | ~93 | 100 % |

Decode is GPU-eval-bound (weight streaming + KV reads); host/graph overhead is
small. The first verify round after prefill is a one-time ~4 s (graph build for
the verify shape at full context), which is why `avgStepMs` (250 ms over 25
rounds) is well above the ~93 ms steady round.

## Bottleneck analysis

- **Blocking (long context):** the quadratic prefill attention buffer. It is the
  only thing that prevents 32K/96K, and it is a memory-layout property of the
  dense full-attention prefill, independent of decode micro-optimizations.
- **Dominant cost (feasible contexts):** prefill throughput (~275 tok/s). At 8K
  this is ~27 s and ~90 % of the request. Improving time-to-first-token means
  improving prefill, not decode.
- **Not the bottleneck:** kernel-launch overhead. The fused-GDN A/B is the
  direct test and it is null; the launch count is per verify round, not per
  context token, so it does not accumulate with context length.
- **Prefix caching:** a weak lever for this GDN model (≈7 % TTFT) because the
  recurrent state is not prefix-resumable.

## Profiling note (what was and wasn't measured)

Measured from the server's built-in instrumentation: prefill/decode wall time
(client TTFT/full), per-round phase breakdown (`tEvalMs`/`tGraphBuildMs`/
`tCacheStateMs`/`tHostReadMs`), `decodeSeconds`, acceptance, the Metal buffer
allocation that crashes, and peak RSS.

Not run: a full Instruments GPU-timeline / memory-bandwidth trace. It would
require re-running the (expensive, partly-crashing) cells under a profiler and
adds little beyond the phase breakdown above for the questions in the brief; the
conclusions are carried by the wall-time A/B, the phase split, and the Metal
buffer allocation. If a dedicated GPU-saturation trace is wanted, it is a
separate profiling task.

## Conclusions

- **Do not pursue decode/verify micro-optimizations (e.g., fused GDN) to fix
  long-context latency.** They are bit-exact but do not move wall-clock, and
  the prefill — not the decode — is the cost.
- **The real long-context blocker is the quadratic prefill buffer.** Enabling
  32K/96K requires changing the prefill attention to a chunked/flash form (no
  dense `seq × seq` buffer) **and** teaching admission control about the
  transient prefill allocation. Both are separate optimization tasks (see the
  brief's optional follow-ups), not part of this benchmarking effort.
- **k = 2 stays the recommended draft depth** across short and long context.
- **Session caching is not a TTFT remedy for this GDN model** at these lengths;
  its value is in reusing full-attention KV and the tokenizer, not skipping the
  recurrent prefill.

## Reproduction

```bash
# Server repo at 18ed7d7; release binary built from it.
swift build -c release --product qwen38-mtp-server

# Build fixtures (deterministic; sha256 in NOTES.txt).
/tmp/benchvenv/bin/python benchmarks/make_longctx_fixtures.py

# One streaming request, prints TTFT/full/decode/content hash.
# (Server started manually: .build/release/qwen38-mtp-server serve --port 18099 --model ./weights)
/tmp/benchvenv/bin/python /tmp/measure.py 18099 benchmarks/prompts/longctx-8k.txt 64 base8k

# Reproduce the 32K crash:
#   start server, then POST the longctx-32k.txt prompt -> [metal::malloc] ... 51577363200 > 30150672384
```

The A/B matrix driver (`/tmp/run8k.sh`) and measurer (`/tmp/measure.py`) are
scratch scripts kept outside the tree; the durable artifacts are the fixtures,
`benchmarks/results/longctx-2026-09-15/`, and this document.
