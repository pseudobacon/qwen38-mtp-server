# HANDOFF — qmm_nax repair delivered through the dependency-pin chain (2026-10-01)

## The chain (all local; nothing pushed)
| Level | Repo | Branch | Commit | Content |
|---|---|---|---|---|
| 1 | mlx (local clone `/tmp/qmm-chain/mlx`) | `feature/qmm-nax-repair` | `8c4c79aa` | on 346eff75: nax.h pair-M descriptor fix + TN==1 zero-padded branch; quantized.cpp `MLX_QMM_*` knobs (stock defaults) + WN clamp + guards |
| 2 | mlx-swift (local clone `/tmp/qmm-chain/mlx-swift`) | `feature/qmm-nax-repair` | `00f401ca` | on 472c262: submodule → `8c4c79aa`, `.gitmodules` URL → `pseudobacon/mlx` fork, regenerated Cmlx tree |
| 3 | mlx-swift-lm | `feature/prefill-ffn-gemm-deliver` | `a16c7530` | on `b47c064`: Package.swift pin → `00f401ca` |
| 4 | qwen38-mtp-server | `feature/lev-j-deliver` | `eed62835` | on `58c5995`: Package.swift + Package.resolved pin → `00f401ca` |

Both working trees are checked out on their deliver branches (clean). The
`/tmp/qmm-chain/{mlx,mlx-swift}` clones hold levels 1–2; the `bare/` mirrors and
validation sandboxes under `/tmp/qmm-chain/` are disposable.

## Chain is pushed (2026-10-01)
All four levels are on github as new branches (no force push, no existing ref
touched):
- `pseudobacon/mlx:feature/qmm-nax-repair` @ `8c4c79aa` (carries `346eff75`)
- `pseudobacon/mlx-swift:feature/qmm-nax-repair` @ `00f401ca` (carries `472c262`)
- `pseudobacon/mlx-swift-lm:feature/prefill-ffn-gemm-deliver` @ `a16c7530`
- `pseudobacon/qwen38-mtp-server:feature/lev-j-deliver` @ `d614609`
Post-push proof: a fresh engine worktree at `a16c7530` resolved purely from the
github URLs — submodule auto-checked-out at `8c4c79aa`; bench built; metallib
bit-identical (`3a91039f…`); M=512 4-config `--checkall` 8/8 PASS. Fresh
machines can now resolve the whole chain.

## Fresh-state validation (from clean worktrees + clean dependency state)
- Engine worktree @ `a16c7530`, dependency via the level-2 clone (path dep,
  submodule at `8c4c79aa`, generated tree bit-identical to the validated
  tree), fresh `.build`:
  - `prefill-gemm-bench` built; metallib rebuilt from the fresh tree =
    sha256 `3a91039f…` (bit-identical to the validated artifact).
  - M=512 4-config `--checkall` (pair-N / candidate / pair-M / TN==1-odd):
    10/10 PASS; M=16 `--no-override` PASS; M=65536 spot PASS.
- Server worktree @ `eed62835`, same dependency state, fresh `.build`:
  - `swift build --target HTTPServer` clean; full `swift test`:
    **237 tests, 7 suites, all pass** (metallib placed in the test bundle
    MacOS dir, as the runtime requires).

## Decision (preserved)
Stock qmm_nax tile policy is the global default. BM=128/BN=64/BK=64/WM=4/WN=2
remains an experimental `MLX_QMM_*` override only. The 5 paired 48-batch 64K
runs (engine `benchmarks/results/prefill-gemm/64k-pair-20261001/`) show full
layer mean −2.45% (range −8.1…+2.0), gateup −17.8% (noisy), down a tie — no
default change; the ~18% historical claim is unreproducible and retired.

## Remaining work
1. ~~Push decision~~ — DONE: all four branches pushed (see above).
2. Feature-branch → main merges (separate explicit step, per-repo).
3. The engine's patch/apply scripts (`qmm-tile-knob.patch`,
   `apply-qmm-tile-knob.sh`) remain as a manual mechanism for stock
   checkouts; delivery is now the pin chain.
4. Known gaps (unchanged): `--check` is down-only; non-transpose TN==1 tiles
   unexercised; `MLXLLM` binary absent from engine `.build/release`.

## Commands (verified)
```
cd /Users/cwong/ai/mlx-swift-lm   # on feature/prefill-ffn-gemm-deliver
bash scripts/build-metallib.sh check .build/release   # 5 OK, 0 FAIL (engine checkout state)
cd /Users/cwong/ai/qwen38-mtp-server  # on feature/lev-j-deliver
swift test                                # 237/237
```
