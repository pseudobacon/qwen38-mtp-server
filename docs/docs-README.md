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
swift test --filter Qwen38MTPDiagnosticTests         # 1/1 (~151 s) — bit-exactness + acceptance (93.46%)
swift test --filter Qwen35FusedSwiGLUProjectionTests  # 11/11
swift test --filter Qwen35FusedQKVProjectionTests     # 8/8
swift test --filter Qwen35FusedGDNProjectionTests    # 14/14
swift test --filter HTTPServerTests                  # 108 passed, 0 failed

# Benchmark matrix
bash benchmarks/run_matrix.sh verify   # §0 fixture provenance check
bash benchmarks/run_matrix.sh itemA    # 2×2 fusion matrix
bash benchmarks/run_matrix.sh itemC    # layout comparison
```

## Runtime knobs

| Env var | Default | Effect |
|---|---|---|
| `MLX_QWEN_FUSED_QKV` | ON | Fused W_qkv packed projection (rollback: `0`) |
| `MLX_QWEN_FUSED_SWIGLU` | ON | Fused W_gate+up packed projection (rollback: `0`) |
| `MLX_QWEN_SWIGLU_LAYOUT` | `global` | `interleaved` variant implemented but REJECTED — do not enable |
| `MLX_QWEN_FOUR_GDN` | ON | GDN 4-projection input fusion |
| `MLX_COMPILED_DECODE` | ON | `compile(shapeless: true)` activation micro-fusions (opt-out for the Tahoe Metal JIT bug) |
| `QWEN35_QMV_ARM` | `liveSums` | `table` selects the xsums sum-table QMV arm |
| `QWEN_MTP_STEP_TRACE` | off | Per-request MTP-STEP-SUMMARY on stderr (rounds / proposed / accepted / avgStepMs / component timings) |

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