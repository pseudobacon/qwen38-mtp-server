# Online adaptive draft depth

`qwen38-mtp-server` can adjust the per-request speculative draft depth **online**,
at serve time, from the acceptance rate and wall-clock throughput it observes in
completed requests. This is the runtime complement to [`DEPTH-CALIBRATION.md`](DEPTH-CALIBRATION.md):
calibration picks a good *fixed* starting depth for the machine; the adaptive
policy then *moves* the depth in response to the workload (acceptance rate varies
with prompt entropy, context length, and model behavior).

It is **off by default** and is a pure server-side policy — no kernel, model,
head, quantization, or sampling-semantics change. The emitted token stream is
bit-identical across depths for a greedy request (speculative decoding changes
*how many* tokens are verified per round, never *which* are committed).

## How it works

The server owns a small state machine (`AdaptiveDraftDepthPolicy`, a pure
model-free struct in `Generation/AdaptiveDraftDepth.swift`). At the end of each
completed, non-cancelled request that proposed at least one draft, the generator
feeds the policy one sample:

- **Acceptance rate** = `acceptedDraftTokens / proposedDraftTokens` for the request.
- **Throughput** = `completionTokens / decodeSeconds` for the request.

The policy keeps a bounded rolling window (default 50 requests) and hysteresis
counters, and moves the depth by **at most one step per sample**:

- **Increase** — if the acceptance rate is at/above `--spec-draft-adaptive-threshold-high`
  (default `0.7`) for `--spec-draft-adaptive-hysteresis` consecutive samples
  (default `10`), *and* the depth is below `--spec-draft-n-max`, *and* the
  throughput safety signal is not firing, increase the depth by 1.
- **Decrease** — if the acceptance rate is at/below `--spec-draft-adaptive-threshold-low`
  (default `0.5`) for `hysteresis` consecutive samples, *and* the depth is above `1`,
  decrease the depth by 1.
- **Throughput safety signal** (on by default, internal) — if the request
  throughput is more than 10% below the rolling mean for 3 consecutive samples,
  *and* the depth is above `1`, decrease the depth by 1. This also vetoes an
  acceptance-driven increase: it catches the case where a high acceptance rate
  does not translate to throughput (e.g. deeper rounds whose per-round overhead
  eats the acceptance gains).
- **Dead band** — an acceptance rate strictly between the two thresholds moves
  neither counter and causes no change.

The depth is bounded to `[1, --spec-draft-n-max]`. Depth `0` (serial) is a
startup/operator setting, not a runtime target. The **hysteresis** counter is
the oscillation guard: a direction must be *sustained* for `hysteresis` samples
before the depth moves, and the counters reset on a depth change and in the dead
band, so noisy acceptance cannot flap the depth.

### Granularity: per-request

Each request is an independent `Qwen38MTPBlockSession`, so the policy adapts
**across requests**, not within one. A sample is one completed request (its full
decode), not one round. The window is therefore "the last N completed requests."
A change takes effect for the *next* request (the new depth is read at session
creation). With `hysteresis = 10` and `window = 50`, the policy moves at most
every 10 requests per direction — far slower than any single request's duration.

### Starting depth

The policy is seeded at the resolved forced depth (the same resolution as
calibration: explicit `--spec-draft-n-max` → `QWEN_MTP_DRAFT_K` → stored
calibrated depth → default `2`). If `--spec-draft-calibrate` also runs, its
winner is applied to the policy as the new baseline (hysteresis reset).

## Usage

```bash
# Serve with online adaptive draft depth (off by default):
qwen38-mtp-server serve --model ./weights \
  --spec-draft-n-max 5 \
  --spec-draft-adaptive \
  --spec-draft-adaptive-window 50 \
  --spec-draft-adaptive-threshold-high 0.7 \
  --spec-draft-adaptive-threshold-low 0.5 \
  --spec-draft-adaptive-hysteresis 10
```

The depth can only move within `[1, --spec-draft-n-max]`, so set the offer cap to
the deepest you are willing to run. Combine with a startup calibration to start
from a good baseline:

```bash
qwen38-mtp-server serve --model ./weights \
  --spec-draft-n-max 5 \
  --spec-draft-calibrate \
  --spec-draft-adaptive
```

Flags (all optional; adaptation is **off by default**):

| Flag | Default | Meaning |
|---|---|---|
| `--spec-draft-adaptive` | off | Enable online adaptive draft depth |
| `--spec-draft-adaptive-window` | `50` | Rolling window of completed requests |
| `--spec-draft-adaptive-threshold-high` | `0.7` | Acceptance rate at/above which depth may increase |
| `--spec-draft-adaptive-threshold-low` | `0.5` | Acceptance rate at/below which depth may decrease |
| `--spec-draft-adaptive-hysteresis` | `10` | Consecutive same-direction samples required before a depth change |

The throughput safety signal's drop fraction (10%), hysteresis (3), and minimum
history (3) are internal constants, not flags, matching the documented flag
surface.

## Observability

- **Logs** — an info line on enable (with the initial depth and thresholds) and
  on every depth change, e.g.
  `Adaptive draft depth: 2 -> 3 (acceptance high; rolling acc 0.712, rolling tps 21.4)`.
- **`GET /metrics`** — when adaptation is on, the summary includes:
  - `adaptive_draft_depth` — the depth the policy currently wants to run.
  - `adaptive_rolling_acceptance_rate` — the rolling mean acceptance rate.
  - `adaptive_draft_depth_adjustments` — lifetime count of depth changes.

All three are `null` when adaptation is off (a `MetricsSummary` built without a
generator, or with the feature off, encodes them as absent/nil).

## Caveats

- **Throughput is wall-clock, not a benchmark cell.** The per-request tokens/s
  is a measurement for the safety signal, not the 1024-token benchmark protocol.
  Do not cite it as a headline number.
- **Acceptance and throughput are per-request.** A request with no drafts
  (serial, `mtpEnabled` false) is not fed to the policy; only requests that
  proposed at least one draft count as samples.
- **Bounded and stable.** The depth moves by at most one step per sample and is
  clamped to `[1, --spec-draft-n-max]`; hysteresis prevents oscillation on noisy
  acceptance. There is no unbounded feedback loop.
- **Never changes correctness.** At any depth, greedy output is token-identical
  to serial; the policy only changes how many drafts are proposed per round.
- **No persistence.** The runtime depth is process-local; a restart re-seeds at
  the resolved forced depth (see `DEPTH-CALIBRATION.md`).

## Tests

`Tests/HTTPServerTests/AdaptiveDraftDepthTests.swift` (25 tests, pure Swift, no
weights) covers: increase on sustained high acceptance, decrease on sustained
low acceptance, the dead band, hysteresis (single samples don't move the depth;
a dead-band sample resets the counter), bounds (`[1, maxDepth]`), no-oscillation
under alternating signals, bounded adjustments under a sustained shift, the
throughput safety signal (drop decreases depth; the A/B that a sustained drop
reduces depth where acceptance alone would increase it), `setDepth` calibration
sync, initial-depth clamping, rolling stats, the window bound, config
validation, and the `ServerConfig` → policy mapping (off by default, nil when
MTP disabled, builder fields).
