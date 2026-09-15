# HANDOFF — Draft-depth calibration (wall-clock tokens/s per depth) (COMPLETE, merged to main)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**:
> `agent-checkpoint.sh` ran successfully (exit 0) in both repositories and
> wrote `.dsh/last-agent-checkpoint` in each before this file was finalized.
> Markers in the Checkpoint markers section below.

## Objective and acceptance criteria

A **draft-depth calibration mode** for `qwen38-mtp-server` that measures
wall-clock decode throughput at several speculative draft depths and selects
the fastest one per model, storing the winner in a JSON config and serving at
it. Acceptance: `--spec-draft-calibrate` flag + startup benchmark loop (off by
default); JSON config storage (`spec-draft-calibration.json`, per model); the
stored depth is a **hint** overridable by `--spec-draft-n-max`; runtime
adaptation documented as future (not implemented); pure-Swift tests; docs +
README example commands; no kernel/head/quantization/weight/sampling changes;
full `HTTPServerTests` green; engine diagnostic suite green.

## Result

**Calibration (COMPLETE, merged to main).** Engine: one small init hook
(`draftDepth` pin). Server: the calibration surface. No kernel, head topology,
quantization, weight, or sampling-semantics changes. Off by default.

- **Engine** (`Qwen38MTPBlockSession`, `mlx-swift-lm`, commit `d01535d`): new
  optional `draftDepth: Int?` init param (pinned per-round depth). Pinned depth
  takes highest priority in `draftPolicy` (pinned → `QWEN_MTP_DRAFT_K` →
  default 2), above the offer cap. Lets the server sweep depths 0..3 in one
  process (the env var is a process-global static; the per-session pin is not).
  `nil` path unchanged (engine diagnostic 3/3 green).
- **`SpecDraftCalibration`** (new, server): `DepthBenchmark`, `ModelCalibration`,
  `SpecDraftCalibrationFile` (Codable); `parseDepths`, `selectOptimalDepth`
  (max tok/s, ties → lower depth), `load`/`save` (pretty JSON; missing/corrupt →
  nil), `report`, `iso8601Now`.
- **`ServerConfig`**: `--spec-draft-calibrate` (off by default),
  `--spec-draft-calibrate-depths` (default `0,1,2,3`),
  `--spec-draft-calibrate-tokens` (default 100),
  `--spec-draft-calibration-file` (default `./spec-draft-calibration.json`);
  `--spec-draft-n-max` now sets `specDraftNMaxExplicit`; `QWEN_MTP_DRAFT_K` read
  into `specDraftK`. `resolvedForcedDraftDepth(storedCalibratedDepth:)`:
  explicit `--spec-draft-n-max` > `QWEN_MTP_DRAFT_K` > stored > default 2.
- **`MLXGenerator`**: `forcedDraftK: Int?` init param (passed to both
  production sessions); `calibrateDraftDepths(depths:tokens:)` (sweeps pinned
  sessions, decode-only wall-clock, greedy, acceptance from accepted/rejected
  counters); `applyCalibratedDepth(_:)` (sets the pin for the running process).
- **`Qwen38Server`**: loads the store, resolves the forced k, passes it to the
  generator; on `--spec-draft-calibrate`, runs the sweep after warmup, logs the
  report, applies the winner, saves the store (keyed by canonical model id).
  Calibration failure is non-fatal (serves at the resolved k).
- **Docs**: `docs/DEPTH-CALIBRATION.md` (usage, config format, resolution
  precedence, caveats, future runtime-adaptation roadmap), `docs/README.md`
  (example commands + runtime-knobs note + test counts), `progress.md`.

### Tests (pure Swift, no model weights; 17 new)

- `DraftCalibrationTests` (17): `parseDepths` (basic/trim/dedupe/out-of-range/
  empty), `selectOptimalDepth` (max/empty/tie→lower), `report` format, store
  round-trip / missing→nil / corrupt→nil, `resolvedForcedDraftDepth` precedence
  (default/env/stored/explicit-n-max), `iso8601Now` format.
- **Full `HTTPServerTests`: 199 green** (182 → 199). Engine
  `Qwen38MTPDiagnosticTests` 3/3 green.

## Git state

- `qwen38-mtp-server` (this repo): branch `main`, HEAD `148fa47` before this
  task; the calibration commit is the fast-forward merge of
  `feature/prompt-draft-calibrate` (branch deleted). Working tree clean (except
  this `docs/HANDOFF.md` update).
- `../mlx-swift-lm`: branch `main`; HEAD `d01535d` (the `draftDepth` pin), the
  fast-forward merge of `feature/prompt-draft-calibrate` (branch deleted).

## Commands / verification

```
swift build --target HTTPServer          # clean, no warnings
swift test --filter DraftCalibrationTests # 17 green
swift test --filter HTTPServerTests      # 199 green
cd ../mlx-swift-lm && swift build --target MLXLLM   # clean
cd ../mlx-swift-lm && swift test --filter Qwen38MTPDiagnosticTests  # 3/3
# live (optional, needs weights):
#   qwen38-mtp-server serve --model ./weights --spec-draft-calibrate
```

## Unresolved risks / caveats

- Wall-clock throughput measurement, **not** a 1024-token benchmark cell. Do
  not cite calibration tok/s as a headline number; it is only for in-session
  relative depth ranking.
- Depth selection never changes correctness: greedy output is bit-identical
  across depths (speculative decoding changes tokens-per-round, never which
  tokens are committed).
- The stored depth is a **hint**: an explicit `--spec-draft-n-max` or
  `QWEN_MTP_DRAFT_K` overrides it. `--spec-draft-n-max 0` disables MTP.
- Online runtime adaptation (rolling acceptance → depth) is **documented as
  future, not implemented**.

## Do-not-repeat

- Do not cite the calibration tok/s as a headline number (wall-clock, 50–100
  tokens, not the benchmark protocol).
- Do not claim depth selection changes correctness (greedy is bit-identical
  across depths).
- Do not treat the stored depth as authoritative over an explicit
  `--spec-draft-n-max` or `QWEN_MTP_DRAFT_K`.
- Do not run `head -N` on checkpoint output (SIGPIPE aborts before the marker
  write); redirect to a file.
- The `agent-checkpoint.sh` script must be run **inside** each actual git repo
  (the `qwen38-mlx-server` symlink wrapper is not a worktree), invoked by its
  full path from the repo directory.
- Engine commit precedes server commit; both merged to `main`, feature branches
  deleted.

## Next step

None — the calibration mode is complete, tested, documented, merged to `main`,
and checkpointed. A successor session should independently verify (per the
resume procedure): `git status --short` (clean) in both repos,
`swift test --filter HTTPServerTests` (199 green), and
`swift test --filter Qwen38MTPDiagnosticTests` (3/3). If an **online** runtime
adaptive depth policy is requested, that is the documented-future work in
`docs/DEPTH-CALIBRATION.md` (not implemented).

## Checkpoint markers

- server: 2026-09-15T13:20:43+01:00
- engine: 2026-09-15T13:20:42+01:00

The fresh-checkpoint procedure completed.
