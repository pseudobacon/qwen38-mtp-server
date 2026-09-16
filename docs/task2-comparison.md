# Task 2 comparison: canonical depth tuning vs recovered `DepthTuning`

Decision: **KEEP** — the canonical depth-tuning surface remains the single
depth-tuning system. Nothing is ported from recovered Task 2
(`DepthTuning.swift`, `--tune`, `POST /tune`, `tuned-depth.json`).
Exactly one tuner remains in the repo (satisfies the hard rule).

Two canonical safety gaps from the recovered design are recorded as
deferred work and ride the **Phase 3** branch (they depend on the same
`WeightTreeDigest`/`hardwareID` machinery the SSD index needs):

1. **Calibration-store invalidation by weight digest + hardware ID.**
   Canonical `spec-draft-calibration.json` is keyed by model-ID string
   with a free-form `hardware` label — an in-place weight update keeps a
   stale optimal depth. Recovered `TunedDepth` invalidates on
   `(weightDigest, hardwareID)` mismatch (`TunedDepthStore.effective`).
2. **≥5%-over-serial margin gate before persistence**
   (`DepthTuning.serialBaselineMargin = 0.05`). Canonical persists the
   sweep winner unconditionally (depth 0 is a candidate, so a losing
   depth is never *selected*, but a marginal winner is persisted).

Both are small, testable extensions to `SpecDraftCalibration` + the
calibration entry point, to be done with `WeightTreeDigest` (recovered
`DepthTuning.swift` L40–140: SHA-256 tree digest over model files +
`hw.model`/RAM hardware id) in Phase 3.

## Feature-by-feature table

| Dimension | Canonical (`--spec-draft-calibrate*` + `--spec-draft-adaptive*`) | Recovered (`DepthTuning.swift` / `--tune` / `POST /tune`) |
|---|---|---|
| Trigger | Startup flags `--spec-draft-calibrate` (+ `-depths`, `-tokens`, `-calibration-file`); online policy via `--spec-draft-adaptive` (+ window/thresholds/hysteresis) | `--tune` startup flag (runs before serving) and `POST /tune` HTTP endpoint |
| Persistence file + schema | `spec-draft-calibration.json` (cwd-relative, overridable): `models: [modelID → {optimalDepth, calibratedAt, acceptanceRate?, contextLength?, hardware? (free-form label), results: [per-depth rows]}]` (`SpecDraftCalibration.swift` L26–52) | `~/.qwen-mtp/tuned-depth.json`: `{weight_digest, hardware_id, depth, per_depth: [depth → {medianTps, meanAcceptance?, medianRoundMicros}], calibrated_at, margin_over_serial?}` (`DepthTuning.swift` `TunedDepth`) |
| Cache key / invalidation | Model-ID string key only; `hardware` stored but **not** checked; no weight digest → stale depth survives in-place weight updates | `(SHA-256 weight-tree digest, hardwareID)`; `TunedDepthStore.effective` returns nil on either mismatch (safe invalidation) |
| Selection rule | `selectOptimalDepth`: highest wall-clock tok/s; exact ties → lower depth (`SpecDraftCalibration.swift` L71–79). Greedy calibration (temp 0, topK 1) | `selectWinner`: highest median tok/s over reps; exact ties → lower depth. Non-greedy calibration (temp 0.7, topP 0.95, 300 tokens/rep) so acceptance is non-trivial |
| Precedence resolution | `--spec-draft-n-max` > `QWEN_MTP_DRAFT_K` > stored calibration > `defaultDraftDepth` (k=2); adaptive policy adjusts around the baseline at runtime | CLI > tuned file (if digest+hw match) > spec default; clamped to `[0, specDraftNMax]` |
| Metrics surface | Request-level acceptance/tps feed `recordAdaptiveSample`; `/metrics` adaptive fields (see `RequestMetrics.swift`) | `tunedDraftDepth` / `tuneCalibratedAt` / `isTuning` on the `/metrics` summary (`OpenAIRouter.swift` L48–55) |
| Test coverage | 17 `@Test` (`DraftCalibrationTests.swift`) + 25 `@Test` (`AdaptiveDraftDepthTests.swift`) = 42, all pure | 16 `@Test` (`DepthTuningTests.swift`) + 1 weight-gated (`tunedDepthCalibratesAndGreedyByteIdentity`) |

## Why KEEP

- The canonical surface is strictly broader: offline calibration **plus**
  an online adaptive policy (acceptance/throughput hysteresis) that the
  recovered lineage does not have at all, with 42 vs 17 tests.
- Selection rules are mathematically equivalent (max throughput,
  lower-depth tie-break).
- Precedence chains are compatible (explicit CLI wins in both).
- The only recovered-specific behaviors worth keeping (digest/hardware
  invalidation, 5% serial margin) are isolated, small, and deferred to
  Phase 3 as extensions of the canonical store — not a parallel system.
- Porting `DepthTuning.swift` as-is would create a second tuner
  (`tuned-depth.json` + `/tune`) alongside the canonical one → automatic
  fail of this phase's hard rule.
