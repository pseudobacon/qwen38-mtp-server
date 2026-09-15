# Draft-depth calibration

`qwen38-mtp-server` can measure wall-clock decode throughput at several
speculative draft depths and select the fastest one per model, then serve at
that depth. This is a **wall-clock calibration**, not a static cost model: it
measures actual throughput on this hardware/workload and stores the winner.

It complements the fixed `k = 2` default (post-W4). The pinned default is a
good *static* answer; calibration finds the best answer *for this machine
right now* (thermals, memory pressure, and workload shift the optimum).

## What it does

When run with `--spec-draft-calibrate`, the server (at startup, **before** it
serves — it never blocks a live request):

1. Loads the model and warms the common shapes (normal startup).
2. For each depth in `--spec-draft-calibrate-depths` (default `0,1,2,3`), runs a
   short greedy decode (default 100 tokens) from a fixed continuation prompt,
   pinned to exactly that per-round draft depth.
3. Records wall-clock tokens/s and the acceptance rate per depth.
4. Prints a table and selects the depth with the highest throughput
   (ties → lower depth, cheaper).
5. Saves the result to the calibration file and **applies the winner** to the
   running process (subsequent generations use it).

Example:

```text
Depth calibration results (100 tokens each):
  Depth 0: 18.5 tok/s
  Depth 1: 21.3 tok/s
  Depth 2: 22.1 tok/s  <-- optimal
  Depth 3: 20.8 tok/s
Selected depth: 2
```

## Usage

```bash
# Full sweep (0,1,2,3) at 100 tokens each, then serve at the winner.
qwen38-mtp-server serve \
  --model ./weights \
  --spec-draft-calibrate \
  --spec-draft-calibrate-depths 0,1,2,3 \
  --spec-draft-calibrate-tokens 100
```

Flags (all optional; calibration is **off by default**):

| Flag | Default | Meaning |
|---|---|---|
| `--spec-draft-calibrate` | off | Run the sweep at startup, then serve at the winner |
| `--spec-draft-calibrate-depths` | `0,1,2,3` | Comma-separated depths to sweep (each in `0..8`) |
| `--spec-draft-calibrate-tokens` | `100` | Tokens to decode per depth (keep 50–100; it's a measurement, not a benchmark cell) |
| `--spec-draft-calibration-file` | `./spec-draft-calibration.json` | JSON store of per-model optimal depths |

Calibration reuses the normal startup (model load + warmup) and then runs the
sweep; a full default sweep is on the order of tens of seconds. It is a manual
operator action, so it is acceptable at startup. To just *read* a previously
calibrated depth (no sweep), start without `--spec-draft-calibrate`.

## Config file

`spec-draft-calibration.json` (one entry per model ID):

```json
{
  "models": {
    "qwen3.8-27b-mtp": {
      "optimal_depth": 2,
      "calibrated_at": "2026-09-15T12:00:00Z",
      "acceptance_rate": 0.638,
      "context_length": 262144,
      "hardware": "macOS/Apple-Silicon",
      "results": [
        { "depth": 0, "tokens": 100, "seconds": 5.4, "tokensPerSecond": 18.5, "acceptanceRate": null },
        { "depth": 1, "tokens": 100, "seconds": 4.7, "tokensPerSecond": 21.3, "acceptanceRate": 0.55 },
        { "depth": 2, "tokens": 100, "seconds": 4.5, "tokensPerSecond": 22.1, "acceptanceRate": 0.638 },
        { "depth": 3, "tokens": 100, "seconds": 4.8, "tokensPerSecond": 20.8, "acceptanceRate": 0.61 }
      ]
    }
  }
}
```

`acceptance_rate` at the optimal depth and the per-depth `results` are stored
for diagnosis; the load path only needs `optimal_depth`. The file is created/
updated by a calibration run and read at every startup. A missing file is not
an error (a fresh checkout has none); a corrupt file is logged and treated as
missing.

## How the depth is resolved at startup

The effective per-round depth `k` is `min(offer cap, forced k)`, where the
**offer cap** is `--spec-draft-n-max` (default 3) and the **forced k** resolves
as:

1. `--spec-draft-n-max`, **when set explicitly** (an operator override wins);
2. else `QWEN_MTP_DRAFT_K` (an explicit pin);
3. else the stored `optimal_depth` (the calibration hint);
4. else the engine default (`k = 2`).

So the stored depth is a **hint**: it only wins when neither an explicit
`--spec-draft-n-max` nor `QWEN_MTP_DRAFT_K` is present. An explicit
`--spec-draft-n-max 1` therefore forces depth 1 even if the file says 2, and
`--spec-draft-n-max 0` disables MTP.

## Caveats

- **Wall-clock, not exactness.** Calibration measures throughput. The emitted
  token stream is bit-identical across depths for a greedy request (speculative
  decoding only changes *how many* tokens are verified per round, never *which*
  tokens are committed). Selecting a depth never changes correctness.
- **Not a benchmark cell.** 50–100 tokens is a measurement, not the 1024-token
  benchmark protocol. Do not cite the calibration tok/s as a headline number;
  it is only for relative depth ranking in-session.
- **Thermal / machine-dependent.** The optimum can drift with thermals and load.
  Re-run calibration when the machine or workload changes materially. The
  stored `calibrated_at` timestamp is there for exactly this.
- **Decode-only.** The sweep times the decode phase (the one-time prefill is
  excluded); the draft depth affects per-round draft proposals, not prefill.
- **Greedy only.** The sweep runs `temperature: 0.0`; it measures the greedy
  decode path (the production default).

## Runtime adaptation (future, not implemented)

Documented for roadmap only — **not implemented**:

- A rolling acceptance-rate monitor that nudges depth online (e.g. raise depth
  when acceptance stays above ~0.7, lower it below ~0.5), analogous to a
  simplified llama.cpp `draft-mtp-adaptive`.
- An A/B harness comparing the adaptive policy against the fixed calibrated
  depth over the benchmark fixtures.

The current design deliberately ships a *calibrate-once, serve-fixed* model:
the stored depth is a hint, `--spec-draft-n-max` is the override, and no
online state feeds back into the running process.
