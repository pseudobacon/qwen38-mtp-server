# HANDOFF — qmm_nax integration (2026-10-01, committed)

## Current state
- **Engine fork** `/Users/cwong/ai/mlx-swift-lm`, branch
  `feature/prefill-ffn-gemm`, 2 new commits on `b1f74b4`:
  - `0771a6b` — `scripts/prefill-gemm/qmm-tile-knob.patch`: the re-cut repair
    (pair-M descriptor fix; TN==1 zero-padded pair-M branch superseding the
    rejected single-fragment mma; `MLX_QMM_*` knobs with stock defaults;
    WN>2 clamp; BM/WM/BN guards). `git apply --check` clean on stock 346eff75.
  - `b47c064` — validation infra: `build-metallib.sh` (SIGPIPE fix,
    content-aware cache key, provenance digest + staleness check),
    `metallib-provenance.json`, `apply-qmm-tile-knob.sh` (idempotent,
    `--skip-build`), `PrefillGemmBench` protocol (`--no-override`,
    effective-config print, gateup in `--checkall`), and the full evidence
    set `benchmarks/results/prefill-gemm/64k-pair-20261001/` (README +
    45 result files) + sweep-doc final status.
- **Server repo**, branch `feature/lev-j`, 1 new commit on `861c11a`:
  - `Package.swift` + `Package.resolved`: `mlx-swift` now pinned to the same
    fork + exact revision as the engine (pseudobacon `472c262`). The old
    `ml-explore` URL + fork-only revision hybrid failed to check out
    ("unable to read tree"); same-identity packages must share a location.
  - `docs/HANDOFF.md` + `progress.md` (this file; permanent experiment record).
- **Decision preserved**: stock qmm_nax tile policy is the global default;
  bm128/bn64/bk64/wm4/wn2 is an experimental `MLX_QMM_*` override only.
  64K evidence (5 paired 48-batch runs, `64k-pair-20261001/`): gateup mean
  −17.8% (high thermal variance), down a tie, full layer mean −2.45%
  (range −8.1…+2.0) — within noise of flat; the lever's wm=4 default breaks
  all M≤32 shapes (TM=0 → Metal compile failure), which is decisive.
  The historical "~18% (190453 µs)" claim remains unreproducible — do not cite.
- Engine working tree: clean except an empty untracked `Source/` directory
  (a stale artifact; git ignores empty dirs — nothing to stage).
- The SwiftPM checkout (`.build/checkouts/mlx-swift`) remains in the repaired
  state (patched submodule + current generated tree + metallib
  `3a91039f…`); that tree is `.build` — deliberately not committed.

## Test results (this session, post-change)
- Engine bench gates (repaired kernel, real Qwen3.8-27B-4bit weights):
  M=512 4-config `--checkall` (pair-N / candidate / pair-M / TN==1-odd):
  10/10 PASS; M=16 `--no-override` (compiled-in stock default): PASS.
- Server full suite: `swift test` → **237 tests, 7 suites, all pass**
  (against the aligned fork pin; `swift build --target HTTPServer` clean).

## Remaining work / known gaps
- No push/merge done (per instruction). Feature branches are local in both
  repos; integration to `main` is a separate, explicit step.
- The kernel repair reaches the SERVER build only if the mlx submodule itself
  carries it (the server's Cmlx comes from the `mlx-swift` checkout's
  submodule at stock 346eff75). The stock default path (bn64 → pair-N) is
  behavior-neutral w.r.t. the repair, so the server suite is green either
  way; making the repair part of the shipped submodule is the next task
  (commit in an `mlx` fork → bump the submodule pin in the `mlx-swift` fork
  → bump the pin in the engine/server).
- `--check` (single-projection) is down-only; `--checkall` covers both.
- Non-transpose (`qmm_n_nax`) TN==1 tiles share the repaired branch but were
  not bench-exercised (model FFN path is transpose; metallib covers it).
- `MLXLLM` binary absent from the engine `.build/release` (swift build clean);
  rebuild with `swift build -c release --product MLXLLM` if a model-level
  probe is wanted.
- The engine AGENTS' `Qwen38MTPDiagnosticTests` filter matches no test class
  in the server package (0 tests); the full unfiltered suite is the
  authoritative gate (used above).

## Commands (verified)
```
cd /Users/cwong/ai/mlx-swift-lm
bash scripts/prefill-gemm/apply-qmm-tile-knob.sh      # idempotent
bash scripts/build-metallib.sh check .build/release   # 5 OK, 0 FAIL
cd /Users/cwong/ai/qwen38-mtp-server
swift test                                           # 237/237
```
