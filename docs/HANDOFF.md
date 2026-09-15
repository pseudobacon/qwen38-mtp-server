# Handoff — qwen38-mtp-server

## Status
**COMPLETE: Online adaptive draft depth (Phase G) implemented, tested, committed.**

A serve-time policy that moves the per-request speculative draft depth within
`[1, --spec-draft-n-max]` from observed acceptance rate and wall-clock
throughput. **Server-only** (no engine change). Off by default (`--spec-draft-
adaptive`). Server committed on `feature/prompt-adaptive-depth` at `0cee743`
(+ this HANDOFF update), about to merge to `main`. Engine repo `main` at
`4cd8603`, untouched (this task made no engine changes).

## Repository state
- **Engine** `/Users/cwong/ai/mlx-swift-lm`: branch `main` @ `4cd8603`, clean.
  **No engine changes this task.**
- **Server** `/Users/cwong/ai/qwen38-mtp-server`: branch
  `feature/prompt-adaptive-depth` @ `0cee743` (+ this HANDOFF update to be
  committed), then merged to `main`.

## What changed (server `0cee743`)
- `Sources/HTTPServer/Generation/AdaptiveDraftDepth.swift` (new): pure
  `AdaptiveDraftDepthPolicy` — bounded rolling window (default 50 completed
  requests), hysteresis (default 10), `[1, maxDepth]` clamp. Increase on
  acceptance `>= 0.7` sustained; decrease on acceptance `<= 0.5` sustained; a
  throughput safety signal (tps > 10% below rolling mean for 3 samples) reduces
  depth and vetoes an increase; dead band between thresholds moves no counter
  and resets hysteresis; `setDepth` (calibration sync) clamps + resets.
- `ServerConfig.swift`: `--spec-draft-adaptive` (off) + `-window`/`-threshold-
  high`/`-threshold-low`/`-hysteresis`; `adaptiveDraftDepthConfig(maxDraftDepth:)`
  returns nil when off or MTP disabled.
- `MLXGenerator.swift`: `adaptivePolicy` (nil = off) seeded at the resolved
  forced depth; `recordAdaptiveSample(acceptanceRate:tokensPerSecond:)` (actor
  method, pure mutation) fed per completed request in `generateStream`'s
  success path (gated on `proposedDraftTokens > 0`); `applyCalibratedDepth`
  also `setDepth`s the policy; `adaptiveDraftDepthSnapshot()` for `/metrics`.
- `RequestMetrics.swift`: `MetricsSummary` gains `adaptiveDraftDepth`,
  `adaptiveRollingAcceptanceRate`, `adaptiveDraftDepthAdjustments` (nil when off)
  + CodingKeys.
- `OpenAIRouter.swift`: merges the snapshot into `GET /metrics`.
- `Qwen38Server.swift`: wires `adaptiveDraftDepth:` into the generator init.
- Docs: `docs/ADAPTIVE-DRAFT-DEPTH.md` (new), `docs/README.md` example,
  `docs/DEPTH-CALIBRATION.md` "future" section now points to the implementation,
  `progress.md` Phase G entry.
- Tests: `Tests/HTTPServerTests/AdaptiveDraftDepthTests.swift` (25 tests, pure
  Swift, no weights).

## Key semantics (for a successor)
- **Granularity is per-request** (a sample = one completed request that proposed
  ≥ 1 draft), not per-round. A change takes effect for the *next* request
  (the new `forcedDraftK` is read at session creation).
- **Throughput is wall-clock** for the safety signal, not a benchmark cell.
- **The policy never changes correctness**: at any depth greedy output is
  token-identical to serial; the policy only changes how many drafts are
  proposed per round.
- **Off by default and backward compatible**: with the flag off the policy is
  never created, `recordAdaptiveSample` no-ops, and the three `MetricsSummary`
  fields encode as nil/absent.
- **No engine change**: the policy mutates the existing `forcedDraftK`, which is
  already read at session creation. `MLXLLM`/`MLXFastModel`/kernels untouched.

## Verification
- `swift test --filter AdaptiveDraftDepthTests` → 25/25 green.
- `swift test --filter HTTPServerTests` → 224/224 green (was 199; +25).
- `swift build --target HTTPServer` → clean (no new warnings).
- `git diff --check` → no whitespace errors.

## Open items
- None.
