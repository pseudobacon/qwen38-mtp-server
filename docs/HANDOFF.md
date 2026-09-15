# Handoff — qwen38-mtp-server

## Status
**COMPLETE: Long-context benchmarking at 8K / 32K / 96K (Phase H) — measurement
only, no code changes.**

Benchmarked the server at agentic-coding context lengths, profiled the
bottleneck, and wrote `docs/LONG-CONTEXT-BENCHMARKS.md`. **No server, engine,
kernel, model, or quantization changes** — only fixtures, results, and docs were
added. Server `main` @ `18ed7d7`; engine `main` @ `4cd8603` (untouched).

## Repository state
- **Engine** `/Users/cwong/ai/mlx-swift-lm`: `main` @ `4cd8603`, clean. **No
  engine changes this task.**
- **Server** `/Users/cwong/ai/qwen38-mtp-server`: `main` @ `18ed7d7`, with the
  Phase H artifacts added (untracked until this task's commit):
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
`2026-09-15T17:31:33+01:00` (server `18ed7d7`, engine `4cd8603`; both
`agent-checkpoint.sh` exit 0).

## Next step (exact)
Commit the Phase H artifacts to the server repo (fixtures + results +
`docs/LONG-CONTEXT-BENCHMARKS.md` + `progress.md` + this `docs/HANDOFF.md`),
merge `main`, and stop. No engine commit is needed (engine untouched). No follow-
up work is required for this task.
