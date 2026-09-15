# README — qwen38-mtp-server docs

Orientation for agents and future sessions. **This is a map, not a status report.**
Status lives in `progress.md`; the active task-state checkpoint lives in `docs/HANDOFF.md`.

## Where truth lives

| Document | Contents |
|---|---|
| `progress.md` (repo root) | Current status, checkpoint log, authoritative benchmark tables, decisions on record, open items (roadmap) |
| `docs/HANDOFF.md` | Active task-state checkpoint — updated before every context reset; describes the current task only |
| `benchmarks/FUSION_REPORT.md` | Final fusion-diagnosis study: prompt-fixture provenance, 2×2 matrix, `qmvbench` microbenchmark, interleaved-layout verdict |
| `benchmarks/prompts/` | Pinned prompt fixtures with recorded SHA-256 / accepted counts / stream hashes |
| `benchmarks/results/` | Raw benchmark run artifacts (`.jsonl`) |

**Update discipline:** when a checkpoint or benchmark completes, update `progress.md` and retract any table it supersedes (with the reason). Never leave two live copies of the same number.

## Repository layout

- `qwen38-mtp-server/` — this repo. Server layer (Vapor, OpenAI-compatible API, SSE, observability) and all project docs + benchmark harness. `qwen38-mlx-server` is the DSH working root that symlinks this repo; docs stay here so they version with the work.
- `mlx-swift-lm/` — engine fork. Custom code only in `Qwen35-FastPath.swift`, `Qwen35Kernels.swift`, `FusedQuantizedLinear.swift` (MLXLMCommon), `QmvBench/`, plus 1-line hooks in `Qwen35.swift`. Keep the fork diff minimal; no prose docs in this repo.

## Build and test

```bash
# Engine (fork)
cd mlx-swift-lm && swift build --target MLXLLM

# Server release binary — NOTE: `swift build --target HTTPServer` does NOT
# relink the product; always build the executable via:
swift build --configuration release --product qwen38-mtp-server

# Tests (current counts)
swift test --filter Qwen38MTPDiagnosticTests         # 3/3 (~70 s) — logit alignment + acceptance (91.02% at the pinned width-5 offer) + wide-verify serial family
swift test --filter Qwen38SDPAExactnessTests          # 4/4 — SDPA exactness chunk unit suite (Phase 4, Bug B)
swift test --filter Qwen35FusedSwiGLUProjectionTests  # 9/9
swift test --filter Qwen35FusedQKVProjectionTests     # 9/9
swift test --filter Qwen35FusedGDNProjectionTests    # 15/15
swift test --filter HTTPServerTests                  # 199 passed, 0 failed

# Benchmark matrix
bash benchmarks/run_matrix.sh verify   # §0 fixture provenance check (both pinned fixtures)
bash benchmarks/run_matrix.sh itemA    # 2×2 fusion matrix
```

### MTP head asset (fresh-checkout bootstrap)

The pinned BF16 head tree (`mtp-head/pinned/`) and the generated 4-bit tree
(`mtp-head/q4/`) are both gitignored; the `mtp-head/` entries are cache-backed.
The server **requires** the q4 tree by default — a missing tree is a loud
startup failure, never a silent BF16 run. Once, after the model assets are in
place:

```bash
python3 -m venv /tmp/q4venv && /tmp/q4venv/bin/pip install -q mlx==0.32.2 safetensors
/tmp/q4venv/bin/python benchmarks/make_q4_head.py   # → mtp-head/q4/ (238.9 MB)
```

`make_q4_head.py` is idempotent (refuses to overwrite an existing tree). The
benchmark harness additionally gates every cell on the actually-loaded head
state (`run_cell.sh` `expect-head` argument) so a fallback can never silently
produce a headline number.

## Runtime knobs

| Env var | Default | Effect |
|---|---|---|
| `MLX_QWEN_FUSED_QKV` | ON | Fused W_qkv packed projection (rollback: `0`) |
| `MLX_QWEN_FUSED_SWIGLU` | ON | Fused W_gate+up packed projection (rollback: `0`) |
| `MLX_QWEN_QMV_VERIFY` | ON | Routed QMV kernel on the verify pass (`B·L ∈ 2..9`, 3-D verify reshaped to `[B·L, K]`); M = 1 falls back to the incumbent (rollback: `0`) |
| `MLX_QWEN_FOUR_GDN` | ON | GDN 4-projection input fusion |
| `MLX_COMPILED_DECODE` | ON | `compile(shapeless: true)` activation micro-fusions (opt-out for the Tahoe Metal JIT bug) |
| `MLX_QWEN_MTP_HEAD_QUANT` | ON | 4-bit draft head `<QWEN_MTP_HEAD>/q4`; the tree is **required** by default (missing → loud startup failure); `0`/`false`/`off` = explicit BF16 rollback |
| `QWEN_MTP_DRAFT_K` | unset (pinned k = 2) | Pins the per-round draft depth to `min(offer, k)`; `3` is the rollback knob for the pre-flip adaptive default (reproduces the registered k = 3 streams bit-exact). The stored calibration depth (below) is a hint that loses to an explicit `--spec-draft-n-max` or this var |
| `QWEN35_QMV_ARM` | `liveSums` | `table` selects the xsums sum-table QMV arm |
| `QWEN_MTP_STEP_TRACE` | off | Per-request MTP-STEP-SUMMARY on stderr (rounds / proposed / accepted / avgStepMs / component timings) |
| `MLX_CHUNKED_PREFILL` | OFF | **Long-context support (32K–64K).** `1` enables chunked causal prefill, bounding the quadratic `[L,L]` scores buffer to `tile × L` (linear) so long prompts complete where the dense path traps at ~24K. Default OFF (bit-identical to the dense path). The server's admission control models the transient buffer and rejects oversized *dense* prefills with HTTP 507 / `prefill_buffer_exceeded`. See `docs/CHUNKED-PREFILL.md` |

### Draft-depth calibration (measure tokens/s, pick the best depth)

Wall-clock calibration is **off by default**. Run it once to find the fastest
draft depth on this machine and store the winner:

```bash
# Sweep depths 0..3 at 100 tokens each, print the table, save the winner, serve at it.
qwen38-mtp-server serve --model ./weights \
  --spec-draft-calibrate \
  --spec-draft-calibrate-depths 0,1,2,3 \
  --spec-draft-calibrate-tokens 100
```

The winner is written to `./spec-draft-calibration.json` (per model) and applied
immediately. On later startups (no `--spec-draft-calibrate`) the stored depth is
used as a **hint** — an explicit `--spec-draft-n-max` or `QWEN_MTP_DRAFT_K`
overrides it. Full semantics and caveats: `docs/DEPTH-CALIBRATION.md`.

### Online adaptive draft depth (adjust depth at serve time)

Adaptive depth is **off by default** and moves the per-request depth within
`[1, --spec-draft-n-max]` from observed acceptance rate and throughput:

```bash
qwen38-mtp-server serve --model ./weights \
  --spec-draft-n-max 5 \
  --spec-draft-adaptive \
  --spec-draft-adaptive-threshold-high 0.7 \
  --spec-draft-adaptive-threshold-low 0.5 \
  --spec-draft-adaptive-hysteresis 10
```

Current depth, rolling acceptance, and the adjustment count are exposed on
`/metrics`. Full semantics, the throughput safety signal, and caveats:
`docs/ADAPTIVE-DRAFT-DEPTH.md`.

## Recommended configuration (current main, 2026-09-14)

Production defaults: QMV verify ON, fusions ON, compiled decode ON, **4-bit MTP head ON** (`MLX_QWEN_MTP_HEAD_QUANT`), **draft depth pinned k = 2** (`--spec-draft-n-max 3` offer cap, `QWEN_MTP_DRAFT_K` unset). Final headline on this default (12/12 cells deterministic, single binary, per-rep thermal snapshots clean):

| Fixture | tok/s (median, reps 2–6 of 6) | stream hash | depthDist |
|---|---|---|---|
| `essay-1024` | **21.89** (1024 / 46.77 s) | `949b9423…` | 2:461 |
| `specdec-800` | **23.29** (1024 / 43.96 s) | `139acb9d…` | 2:431 |

Rollback: `QWEN_MTP_DRAFT_K=3` (the adaptive-default-era config; bit-exact against the registered k = 3 streams) or `MLX_QWEN_MTP_HEAD_QUANT=0` (BF16 head). The `24 ms tEval` figure in the v1.1 planning docs is retired — see `progress.md`, open item 5.

## Benchmark protocol (short form — full rules in progress.md)

Release build, port 18099, greedy (`temperature: 0.0`, `enable_thinking: false`,
`max_tokens: 1024`, `finish_reason: length`), prompt read from a pinned fixture file,
`QWEN_MTP_STEP_TRACE=1`, rep 1 discarded, 5 measured reps, interleaved cell order,
no parallel builds/tests during timing. **Every cell must reproduce the fixture's
accepted counts and stream hash** — a mismatch is a correctness/protocol failure,
never a performance result. Cross-session absolute latencies are not comparable;
only in-session deltas are valid.

## Conventions for agents

- New Metal kernels go in `Qwen35Kernels.swift`; new Swift fast-path logic in `Qwen35-FastPath.swift`; `Qwen35.swift` stays vendor-shaped with 1-line hooks only.
- Any new fast path ships with a bit-exact eager fallback and a test proving bit-identity; verify engagement via load-time logs, not RSS.
- `MLXFast.metalKernel` `grid` is **total thread count**, not threadgroup count — the 2b-fix grid bug came from this; don't reintroduce it.
- Benchmark before/after any kernel change, in-session, with the determinism gate; record prompt-fixture hashes with results.
- No `print()` in hot paths (load-time prints allowed); no python/sed source edits; engine commits before server.