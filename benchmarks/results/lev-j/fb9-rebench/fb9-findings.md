# FB9 — Flash SDPA re-bench with the fixed MLX dispatch: PERFORMANCE NO-GO (terminal)

**Date:** 2026-09-20
**Verdict:** LEV-J Flash SDPA **CLOSED — PERFORMANCE NO-GO** (true full-work).
The FA micro-kill-switch GO (13.9–18.4 µs/tok @32K) is **RETRACTED** as a
performance conclusion: those timings were measured through the broken MLX
dispatch layer and therefore represent **truncated work**, not full work.

## Provenance

| Item | Value |
|------|-------|
| Engine repo | `mlx-swift-lm`, branch `feature/lev-j`, commit `3ed1d3ec15e85faa0ee2400c9954b6fdbd2d6002` ("bench: restore FlashBench executable target") |
| Harness | `flashbench` executable (`Libraries/FlashBench/main.swift`, single-shot, n=1 per cell, per-rep nz fullness check, post-run 120 s wall-clock guard per rep) |
| MLX Swift binding | ml-explore/mlx-swift pinned revision `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` (unpinned change; pin unchanged by this closure) |
| MLX C++ base | `1f8e74e3f12f31365464a6867c6579f0e9b29d85` ("Hold GIL in AttachedData destructor (#4391)", 2026-08-25) — the submodule revision of the pinned mlx-swift |
| Local dispatch fix | 2-line change in `mlx/backend/metal/custom_kernel.cpp` `CustomKernel::eval_gpu`: `group_dims = MTL::Size(std::min(tx,gx), std::min(ty,gy), std::min(tz,gz))` + `dispatch_threads(...)` → `group_dims = MTL::Size(tx, ty, tz)` + `dispatch_threadgroups(grid_dims, group_dims)` (plus a `MLXFD_INSTRUMENT` env-gated stderr trace in the working tree only). **This fix was applied to the nested SwiftPM checkout; it is being landed as a dedicated MLX commit/PR (see Phase 2 of the closure task).** |
| metallib | `mlx-swift-lm/default.metallib`, SHA-256 `886247dfd773ca073bc9d941fcda5157d7ed6dc8` (built 2026-09-11 from the pinned MLX sources; the dispatch bug lives in C++ `eval_gpu`, not in compiled Metal kernels, so the metallib is not a variable here) |
| Machine | MacBook (Apple M5 Pro, 48 GiB unified), macOS 26.5.2 (build 25F84) |

## Raw output

`flashbench-fixed-dispatch-raw.txt` (exact file, unmodified):

```
flashbench: geometry nq=24 nkv=4 D=256 scale=0.0625
flashbench: determinism (Q=2048, prefix=8192) bit-exact = true
flashbench: correctness (Q=2048, prefix=8192) max|diff| flash-ref=6.104e-05 inc-ref=6.104e-05 flash-inc=6.104e-05 (bf16 ulp ~0.0078)
  rep=0 flash=1862.016 ms nz=12582912 full=yes
flashbench: chunk Q=2048 prefix=8192  flash=1862.016 ms (909.188 us/tok)  incumbent=32.268 ms (15.756 us/tok)  speedup=0.02x  (n=1, single-shot)  fullness: allFull=yes minNZ=12582912/12582912
  rep=0 flash=70561.706 ms nz=201326592 full=yes
flashbench: full Q=32768 prefix=32768  flash=70561.706 ms (2153.372 us/tok)  incumbent=OOM  (n=1, single-shot, flash-only)  fullness: allFull=yes minNZ=201326592/201326592
  rep=0 flash=293370.452 ms nz=402653184 full=yes
  GUARD TRIPPED: rep 0 did not complete in 120 s (elapsed 293370.452 ms); stopping cell
flashbench: full Q=65536 prefix=65536  flash=293370.452 ms (4476.478 us/tok)  incumbent=OOM  (n=1, single-shot, flash-only)  fullness: allFull=yes minNZ=402653184/402653184 GUARD
```

## Fullness proof

Every reported rep is **full** by exact non-zero count (per-rep `nz` equals
the exact expected element count `Q × 24 heads × 256 dim`):

- Q=2048/prefix=8192: nz=12,582,912 = 2048·24·256 → `allFull=yes`
- Q=32768/prefix=32768: nz=201,326,592 = 32768·24·256 → `allFull=yes`
- Q=65536/prefix=65536: nz=402,653,184 = 65536·24·256 → `allFull=yes`

This is the exact non-zero-count detector required: no timing below is
credited without per-rep fullness evidence.

## Correctness and determinism (with the fixed dispatch)

- Determinism at Q=2048/prefix=8192: **bit-exact = true**.
- Correctness at Q=2048/prefix=8192: max|diff| flash-ref = inc-ref =
  flash-inc = **6.104e-05** (bf16 ulp ≈ 0.0078). The kernel is numerically
  correct; that is not what disqualifies it.

## Performance (true full-work)

| Cell | Flash (full) | Incumbent (dense SDPA) | Ratio |
|------|--------------|------------------------|-------|
| Q=2048, prefix=8192 (the only directly comparable production chunk cell) | 1862.016 ms = 909.188 µs/tok | 32.268 ms = 15.756 µs/tok | **1862.016 / 32.268 = 57.7× slower** |
| Q=32768, prefix=32768 | 70.562 s = 2153.372 µs/tok | OOM (dense scores > RAM) | incumbent OOM |
| Q=65536, prefix=65536 | 293.370 s = 4476.478 µs/tok | OOM (dense scores > RAM) | incumbent OOM |

### The 64K "guard" is a post-run threshold, not an interrupt

The `GUARD TRIPPED` line on the 65536 cell is **post-run detection**: the rep
**completed** in 293.370 s and only then was compared against the 120 s guard,
which failed ("rep 0 did not complete in 120 s (elapsed 293370.452 ms)"). The
guard did not terminate work at 120 s; 293.370 s is the true full-work
duration for that cell.

## Why the original FA performance result is invalid

The FA micro-kill-switch measured 13.9–18.4 µs/tok @32K (3.5–4.8× faster than
the incumbent) and declared GO. That measurement was taken **before** the MLX
dispatch bug was understood and fixed. `CustomKernel::eval_gpu` clamped the
requested threadgroup dimensions against the grid dimensions and dispatched
through `dispatch_threads`, so the flash kernel's dispatch was silently
truncated: only a small prefix of the requested threadgroups executed. The FA
timings therefore measure **incomplete work** (a fraction of the queries'
prefixes were never computed), not the full flash-attention computation. The
FB7/FB8 dispatch-recombination investigations traced the symptom; FB9 proves
the full-work cost. A timing measured through a truncated dispatch is not a
valid performance measurement of the kernel.

## Scope of the verdict

This is **not** a flash-attention category failure. It is a failure of **this
specific kernel design**: one threadgroup per (query row, head), a full prefix
rescan per row (O(Q·prefix) work with no KV tiling), repeated
threadgroup barriers, and a two-pass score recomputation. A different design
(e.g. tiled KV blocks with online softmax in a single pass) may or may not
reach the incumbent; nothing in this closure adjudicates it.

## Closure statement

> **This specific per-query-row, two-pass flash SDPA design is closed as a
> true-full-work performance NO-GO.** A future attention kernel requires a new
> design, a new standalone target, and a new micro kill-switch; it must prove
> full dispatch (exact per-rep non-zero coverage) before any timing is
> credited.

## Consequences

- Do NOT proceed to FC (model-level audit) or FD (end-to-end A/B with pc
  re-sweep): their prerequisite (a credible full-work flash kernel) failed.
- Do NOT run more FlashBench reps, 32K/64K chunked A/Bs, pc re-sweeps, or the
  old 100-rep production-geometry FlashSDPA test (hours-long under corrected
  full-work dispatch; the answer is already 57.7×).
- The MLX dispatch fix is an independent, real correctness win and is landed
  separately (dedicated MLX commit + regression test + upstream issue/PR).

## MLX dispatch fix — landing provenance

| Item | Value |
|------|-------|
| Upstream issue | https://github.com/ml-explore/mlx/issues/4534 |
| Upstream PR | https://github.com/ml-explore/mlx/pull/4535 |
| Fork | `pseudobacon/mlx`, branch `fix/metal-custom-kernel-dispatch` |
| Fix commit | `346eff7503a22cfe2e020cf595260ce347961af4` |
| Base (unpinned) | MLX C++ `1f8e74e3f12f31365464a6867c6579f0e9b29d85` (2026-08-25); mlx-swift `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` |
| Files | `mlx/backend/metal/custom_kernel.cpp` (the 2-line fix), `tests/gpu_tests.cpp` (full-coverage regression test) |
| Fix | `group_dims = MTL::Size(tx, ty, tz)` (unclamped) + `dispatch_threadgroups(grid_dims, group_dims)`, replacing the `std::min(tx,gx)/...` clamp + `dispatch_threads` |

### Regression test

`tests/gpu_tests.cpp` → `TEST_CASE("fast metal kernel dispatches the full requested grid")`.
A probe kernel writes each executing thread's unique index; the test checks
**exact coverage** of the requested `gx*ty*tz` threads (output == `1..N`, not a
non-zero count) on the **first** dispatch at a fresh geometry and on a **repeated
same-geometry** dispatch, across three (grid, threadgroup) pairs including the
production per-row geometry with `grid.x (24) < threadgroup width (256)`.

**Validation status (this machine, Xcode 26.6.0 / macOS 26.5.2):** the test
compiles in a local CMake build and is the standard MLX-CI deliverable, but it
cannot be *executed* on this machine: every Metal custom kernel (including a
trivial one, and the identical source via pip `mlx` 0.32.1) fails to JIT-compile
with `utils.h:expected expression` — a pre-existing Xcode 26.6.0 Metal toolchain
incompatibility with this MLX version's `utils()` preamble, independent of this
change. The fix itself is independently validated by the FB9 full-work re-bench
(exact per-rep non-zero coverage after the fix), and the probe/coverage
methodology is the FB7 trivial-probe design. The test will run in MLX CI.
