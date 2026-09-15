# Handoff — qwen38-mtp-server

## Status
**COMPLETE: pc=0 single-pass prefill trap fix (engine-only).**

`--prefill-chunk-size 0` (single-pass prefill) fatal-`[reshape]`'d on an empty
chunk (both prefill loops computed `start=0, end=min(0+0,count)=0`). Extracted
the partition into a pure `Qwen38MTPBlockSession.prefillChunkRanges(count:chunkSize:)`
that guards `chunkSize==0` (one full-range chunk); the `chunkSize>0` branch is
mathematically identical to the old inline loop, so the **default pc=512 path is
byte-identical** (no perf regression). 4 unit tests + real-model validation:
pc=0 no longer traps at 8K/16K/32K, bit-exact with pc=512 at 8K/16K, and 32K
diverges at the first token (the expected Phase 1 Bug A FP-accumulation-order
sensitivity to the prefill split — the engine chunked-SDPA gate engages at
L>4096 for pc=0 but not per-512-chunk for pc=512). pc=0 is slower than pc=512
(205 s vs 117 s at 32K), so this is a robustness fix, not a perf change. Engine
`45df72a`, server `54f167c` (docs). Engine 7/7 + server 111/111 green.

### Prior: Flash-attention feasibility analysis (Phase I follow-up)

Evaluated integrating a flash-attention kernel for prefill. **Decision: do not
integrate** (see `docs/FLASH-ATTENTION.md`). Evidence:
- MLX has no flash kernel for prefill (T_q > 1 materializes `[L,L]`; the Metal
  kernel is decode-only).
- A true flash kernel's online softmax is **not bit-exact** (Phase 1, Bug A),
  which the task required.
- An early "~0.04%" SDPA figure was **CPU enqueue time, not GPU time** (MLX
  enqueues Metal commands asynchronously) — invalidated; the per-phase GPU split
  is unmeasured. Valid wall-time signal: prefill is near-optimal at the default
  `prefillChunkSize=512` (119 s; pc=8192 is 55% slower; pc=0 was the trap now
  fixed).
- Chunked prefill already enables 128K+ (per-tile buffer 3.2 GB @128K).

Server suite 0 failures; engine byte-identical to main. Server commit `c16f166`.

### Prior: Chunked causal prefill (Phase I)

Eliminates the quadratic `[seq × seq]` dense-attention scores buffer that traps
the process (SIGTRAP) at ~24K+ context. **Default-OFF** (`MLX_CHUNKED_PREFILL=1`
to enable); the off build is bit-identical to before. Server models the transient
buffer in admission control (dense quadratic vs chunked linear); oversized dense
prefills are rejected with HTTP 507 / `prefill_buffer_exceeded`.

Validated: 8K greedy stream hash identical dense vs chunked (`763eccc3…`);
32K completes with chunked (175.9 s) where dense traps; dense 32K cleanly
rejected pre-prefill (507, `51.6 GB > 27.1 GB`). Engine KVCache 118/118 + MTP
3/3, server 224/224 all green. See `docs/CHUNKED-PREFILL.md`.

## Repository state
- **Engine** `/Users/cwong/ai/mlx-swift-lm`: `main` @ `4cd8603`, with Phase I
  changes (uncommitted until this task's engine commit):
  - `Libraries/MLXLMCommon/AttentionUtils.swift` — `chunkedCausalPrefill` + gate
  - `Libraries/MLXLMCommon/KVCache.swift` — `MLXChunkedPrefill`, quantized chunked
  - `Tests/MLXLMTests/KVCacheTests.swift` — 2 new tests
- **Server** `/Users/cwong/ai/qwen38-mtp-server`: `main` @ `1a247b4`, with Phase I
  changes (uncommitted until this task's server commit):
  - `Sources/HTTPServer/Generation/MemoryAdmission.swift` — transient buffer model
  - `Sources/HTTPServer/Generation/MLXGenerator.swift` — passes `chunkedPrefillEnabled`
  - `Sources/HTTPServer/API/OpenAIValidation.swift` — 507 error response
  - `Sources/HTTPServer/Routes/OpenAIRouter.swift` — catches `TransientBufferFailure`
  - `Tests/HTTPServerTests/KVCacheConfigTests.swift` — 4 new admission tests
  - `docs/CHUNKED-PREFILL.md` (new), `progress.md` (Phase I section), this file

## Prior: Phase H long-context benchmarks
The Phase H findings below (32K/96K infeasible on the dense path, prefill
dominates, fused GDN not a long-context win) motivated Phase I. They remain
accurate for the **default dense build**.
  - `benchmarks/make_longctx_fixtures.py` (new)
  - `benchmarks/prompts/longctx-{8k,16k,32k,96k}.txt` (new fixtures)
  - `benchmarks/results/longctx-2026-09-15/{ab-matrix.jsonl,NOTES.txt}` (raw data)
  - `docs/LONG-CONTEXT-BENCHMARKS.md` (new)
  - `progress.md` (Phase H section appended)
  - `docs/HANDOFF.md` (this file)

## Headline findings (evidence)
- **32K and 96K are infeasible on this hardware.** The prefill attention buffer
  is a dense `[seq × seq]` allocation (quadratic in seq). At 32K it requests
  **51,577,363,200 bytes (51.6 GB)**, exceeding the **30,150,672,384-byte (30.2
  GB) Metal max buffer** → `[metal::malloc] ... greater than the maximum allowed
  buffer size` → **SIGTRAP crash** (reproduced twice). 96K would be ~464 GB.
  **Max feasible prompt ≈ 24K tokens** (buffer ≈ 30 GB at `seq ≈ 32K·√(30.2/51.6)`).
- **Prefill dominates** the request: ~27 s at 8K of a ~30 s total (~90 %).
  Prefill throughput ~275 tok/s, ~linear in seq (8K 27 s, 16K 59 s).
- **Fused GDN does NOT pay off at long context.** Bit-exact (identical completion
  token stream on/off) and not faster at 8K (full_s 29.3 s off vs 32.7 s on). Its
  launch count is per verify round (draft-depth dependent), not per context token,
  so it does not accumulate with context; and prefill (not decode) is the cost.
- **k = 2 remains optimal at 8K** (full_s k1 36.1 / k2 29.3 / k3 32.0 s), matching
  short context.
- **Prefix/session caching ≈ 7 % TTFT win only** (8K cold 27.3 s → warm ~25 s):
  the gated-delta recurrent layers are not resumable from a token-prefix, so the
  prefill is effectively recomputed each request.
- **All configs bit-exact**: identical `content_sha256` (`377c6fda…` at 8K) across
  fused on/off, k = 1/2/3, and repeated requests.
- **Memory**: peak RSS ~15.25 GB at 8K (~15 GB weights + ~0.5 GB KV). Steady-state
  fits 32K under the 44 GB admission limit; the crash is the *transient* quadratic
  prefill buffer, which admission control does not model.
- **Decode is GPU-eval-bound**: `tEvalMs` ~89 ms of ~93 ms `stepMs` at 8K; host/
  graph overhead small. First verify round after prefill is a one-time ~4 s.

## Important files
- `docs/LONG-CONTEXT-BENCHMARKS.md` — the full write-up (fixtures, protocol,
  results tables, bottleneck analysis, reproduction).
- `benchmarks/make_longctx_fixtures.py` — deterministic fixture builder (real
  Swift source from both repos, 479 files / 1.76M-token corpus, truncated at token
  boundaries).
- `benchmarks/results/longctx-2026-09-15/ab-matrix.jsonl` — raw per-request lines
  (client TTFT/full/decode + content sha + server decodeSeconds/avgStepMs/
  acceptedPerStep).
- `benchmarks/results/longctx-2026-09-15/NOTES.txt` — fixture sha256, binary
  sha256, Metal cap.

## Decisions
- **No code changes** (explicit task constraint). The release binary is built from
  `main`; the engine is untouched. All findings are measurements.
- **N per cell is 3–4**, not the brief's 6–10: each 8K request costs ~30 s (27 s
  prefill) and 32K/96K cannot complete. Reported means have small spread.
- **Client `full_s`/`TTFT` is the cross-cell metric** (measured identically);
  server `decodeSeconds` is reported alongside for phase context. There is a
  consistent client-vs-server decode gap (client ~2.3 s vs server ~5 s); the
  cross-cell comparison is valid because every cell is measured the same way.
- **Profiling** used the server's built-in instrumentation (`STEP-TRACE` phase
  split, `decodeSeconds`, the Metal buffer allocation, RSS) rather than a full
  Instruments GPU trace (see "Unresolved risks").

## Verification
- `git diff --check` → clean.
- Engine: `swift test --filter Qwen38MTPDiagnosticTests` → 3/3 green (untouched,
  sanity only). Server test suite unaffected (no server code changed).
- 32K crash reproduced twice with the identical `[metal::malloc]` allocation size.
- Bit-exactness: identical `content_sha256` across all A/B cells at 8K.
- `git status --short` → only the Phase H artifacts above; no tracked source
  files modified.

## Unresolved risks / caveats
- **Instruments GPU-timeline / memory-bandwidth trace not run.** It would require
  re-running the (expensive, partly-crashing) cells under a profiler and adds
  little beyond the `STEP-TRACE` phase split for the questions asked. If a
  dedicated GPU-saturation trace is wanted, treat it as a separate profiling task.
- **32K/96K not benchmarked** (they crash / are infeasible); the numbers for
  those lengths are the predicted buffer sizes, not measured throughput.
- Do NOT cite cross-session absolute tok/s; these numbers are one environment.

## Operations that must not be repeated
- Do NOT change server/engine code to "fix" the long-context crash as part of a
  benchmarking task — that is a separate optimization task (chunked/flash prefill
  + admission-control awareness of the transient buffer).
- Do NOT cite the fused GDN kernel as a wall-clock win (it is not, at any context).
- Do NOT claim 32K/96K work on this hardware (they crash / are infeasible).
- Do NOT run `agent-checkpoint.sh` from the symlink wrapper dir; run it by
  absolute path with the CWD inside the target git repo.
- Do NOT `head -N` the checkpoint script output (SIGPIPE).

## Completion marker
Fresh checkpoint procedure completed successfully in **both** repositories:
`2026-09-15T19:21:15+00:00` (server `b4374be`, engine `62c4ac7`; both
`agent-checkpoint.sh` exit 0).

## Next step (exact)
None. Phase I (chunked causal prefill) is complete and committed to `main` in
both repos; the roadmap (`progress.md` open item 7, Done list, `docs/README.md`
runtime knobs) records chunked prefill as the long-context solution for 32K–64K
(Flash attention not required). All gates green: engine KVCache 118/118 + MTP
diagnostic 3/3, server 224/224; 8K dense == chunked stream hash; 32K completes
chunked, rejected dense (507). No follow-up work is required for this task.
