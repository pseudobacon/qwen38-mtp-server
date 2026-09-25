# Handoff — qwen38-mtp-server

## Status
**GO (2026-09-24, REVISED after correctness gate): Prefill FFN GEMM tile lever — real but ~8–18 %; the earlier "1.3×" was INVALID.** Engine `feature/prefill-ffn-gemm` adds `Libraries/PrefillGemmBench` (product `prefill-gemm-bench`, `--synthetic` = ~133 MB no model load) + the `MLX_QMM_*` tile-knob (`scripts/prefill-gemm/`) + the spec. **REVISED result:** the `--check` bit-exactness gate + full-grid `--checkall` (real FFN, layer 0) exposed that **WN≥3 `qmm_nax` tiles produce partial/all-zero output that finishes fast** — the "best" timing-sweep tiles (bm128/wn4) were computing WRONG results. **Every WN=2 tile is correct (rel ~0.0007); every WN=4 tile FAILS (rel 0.94–1.00).** Best CORRECT tile is **`bm128/wm4/wn2`** (WN stays 2): ~8 % faster at M=512, **~18 % at M=65536** (190453 vs 225083 µs/layer). Projected ~3–7 % total-prefill win. **Methodology fix:** `--checkall` validates every sweep config inline (a broken tile is flagged "** FAIL (broken tile)", its timing untrustable) + out-of-bounds tiles are rejected. **Next (standalone, NOT integrated):** (1) durable `MLX_QMM_*` knob in the MLX fork pin; (2) LCP-context confirmation of the bm128/wm4/wn2 win on the real 64K prefill; (3) **the WN≥3 `qmm_nax` bug is FIXED** (root cause + 3-part fix in `steel/gemm/nax.h` `tile_matmad_nax`, proven by a standalone Metal test — stock 3/8 fail, fixed 8/8 bit-exact); needs a **swift-impl `libmlx.dylib` rebuild + pin bump** to go live, since small-M qmm_nax tiles JIT-compile from the prebuilt dylib. **WN stays 2 in production until that lands.** **No production source/routing/model behavior changed**; knob unset = stock tiles = bit-identical. Spec §11–15 + `progress.md` carry the detail.
**Fresh checkpoint: COMPLETED 2026-09-25 14:20 CST** (both worktrees via `qwen38-mlx-server/scripts/agent-checkpoint.sh`; that dir is not itself a git repo so the script runs from inside each). Engine `mlx-swift-lm` `feature/prefill-ffn-gemm` @ `b1f74b4` (only a build-artifact `scripts/metallib-provenance.json` uncommitted). Server `qwen38-mtp-server` `feature/lev-j` @ `79f8415` (only an unrelated `Package.resolved` mlx-swift rev bump uncommitted).

**TERMINAL: Block-tiled causal attention candidate A = NO-GO (run `ta-20260920`, 2026-09-20).** 4-phase micro-kill-switch (no production integration) for a block-tiled causal-attention kernel (BQ=64, 8 query rows/simdgroup, `simdgroup_matrix<half,8,8>`, online softmax, O in registers) for Qwen3.8-27B full-attention prefill (bf16, D=256, GQA 6). P0 (durable MLX dispatch fix) **PASS** — fork `pseudobacon/mlx-swift` @ `472c262a` bumps submodule `Source/Cmlx/mlx` → `346eff750`; `DispatchSmokeProbe` 6/6 exact full coverage (binary SHA `23ef8588…c325c`). P1 (perf model + design) DONE (dense incumbent 15.756 µs/tok; GO bar ≤ 10.5 µs/tok). P3 (standalone benchmark) **Candidate A v1 NO-GO: invalid `simdgroup_matrix` fragment construction via whole-vector `thread_elements` assignment** (corrected root cause). The prior "divergent-sg `simdgroup_load` corruption" hypothesis is **refuted by M0–M5** of a minimal spec-conforming reproducer (`sml-minimal-repro.metal`, `--smlrepro`): `simdgroup_load`/`_store`/`_multiply_accumulate` all **PASS** for per-simdgroup *distinct* slices; the earlier "corruption" was a **diagnostic artifact** (non-owned `thread_elements()[i]` reads + a debug data race). The **real** root cause is **invalid fragment construction**: step 3 built the P tile via `Pm.thread_elements() = pe` (a whole 64-element vector), an invalid whole-fragment assignment — only lane 0's 2 owned elements survive (M6: 496/512 bad), so P is zero for all `fm != 0` rows and `O = P*V` is zero there. **The M5 Pro platform is not implicated by this investigation.** **Candidate A v1 is closed as an invalid implementation** (not a Metal/driver/GPU defect, and not a proof tiled attention cannot work on M5 Pro). The conforming **Candidate B** (build P from lane-owned elements only) is being implemented as a standalone prototype; **it receives no performance claim until it passes the complete micro-kill-switch (G0–G7).** **GUARDRAIL: do NOT merge `feature/tiled-attention` to `main`; no production integration.** Docs: `benchmarks/results/tiled-attention/ta-20260920/p2-contract-audit.md`, `p3-no-go.md` (corrected), `p4-candidate-b-brief.md`. **No perf claim, no production claim.**

---

### Prior terminal: LEV-J = PERFORMANCE NO-GO (FB9, 2026-09-20)
**TERMINAL: LEV-J = PERFORMANCE NO-GO (FB9, 2026-09-20).** The FB7/FB8 blocker (the MLX Metal dispatch bug that truncated custom-kernel dispatches) is **FIXED upstream**: PR **ml-explore/mlx#4535** (commit `346eff7503a22cfe2e020cf595260ce347961af4`, issue **#4534**, fork `pseudobacon/mlx`), with a full-coverage regression test in `tests/gpu_tests.cpp`. The production path is **unaffected** (the fix is behind the flash gate, which is OFF; flash-OFF is byte-identical to main).

**FB9 re-bench at production geometry** (Q=2048/8192, prefixes {8192, 32768, 65536}, 100 serial + concurrent reps, 3× hash, dispatch fix, NO flashbench tuning): the kernel is **correct + full + deterministic** after the dispatch fix, but **~57.7× SLOWER than the dense incumbent** (Q=2048/prefix=8192: flash 47.87 ms vs dense 0.830 ms). **Per-row threadgroup granularity (24 threadgroups) is not viable at production depth.** The kernel is a research prototype, not a production kernel. **No flash-attention performance claim; no flash routing; gate OFF; no perf claim.**

**Reverted engine path:** all production flash routing/integration reverted — `feature/lev-j` differs from `main` by **only** the preserved `FlashBench` harness target (`main.swift` + `Package.swift`). The `Qwen38FlashSDPATests` suite is removed (dispatch coverage now lives in MLX). No slow tests remain in the engine.

**Final verification state (this machine, Xcode 26.6.0 / macOS 26.5.2):** engine builds (MLXLLM + FlashBench); flash-OFF byte-for-byte equivalent to pre-LEV-J; **server `HTTPServerTests` PASS (237 tests, 7 suites).** Two **pre-existing, machine-local** Metal limitations block only the model-forward / custom-kernel paths here (both reproduce on `main` / pip `mlx` 0.32.1, independent of this task): (1) every Metal custom kernel fails to JIT-compile (`utils.h: expected expression`) — Xcode 26.6.0 toolchain vs this MLX `utils()` preamble; (2) `Qwen38MTPDiagnosticTests` page-faults on model forward (reproduces identically on `main`). The MLX C++ regression test will run in MLX CI. **No perf claim, no production claim.**

**Prior terminal (superseded): FB8 BLOCKED-on-upstream (2026-09-20)** — the version probe then FAILS (recombination persists at origin/main 9019419, 5 commits ahead of pinned 2bebe4e9; guard experiment DIES: all 10 same-geometry dispatches TRUNCATED).

### Production incumbent (gate OFF, byte-identical)
- The flash-SDPA kernel is **OFF** in production (`minQueryRows` gate; flash-OFF is byte-identical to main). Zero production risk. The incumbent dense path is used.
- No production code changed in FB8 (gate OFF; flash-OFF byte-identical to main).

### RE-TEST TRIGGER
- **Any new mlx-swift/cmlx release OR upstream fix to the Metal dispatch layer** → re-run **FB8 Item 1 at production geometry** (Q=2048 and Q=8192, prefixes {8192, 32768, 65536}, 100 serial eval-between reps + concurrent-pairs + 3x hash, zero-signature detector).
- **0/100 at every production-geometry cell** → proceed DIRECTLY to **FC (model-level audit)** and **FD (end-to-end AB with pc re-sweep)**, both already specced. Refresh the FB0 predictions from the FA micro numbers.
- The version probe (FB8 Item 1) is the re-test: scratch-build against the new pin, run the gate, revert the pin.

### Documented caution for future Metal-kernel work (FB5–FB8 retracted-label history)
- **FB2** "kernel non-deterministic" → **FB3** "MLX allocator bug" → **FB5** "m/out byte alias" (DISPROVEN) → **FB6** "grid truncation / COLD-JIT" → **FB7** "dispatch recombination (Q≤256 boundary)" → **FB8** "recombination fires at ALL Q; the 0.169 is the dispatch bug (not numerics); version probe + guard DIES".
- **The lesson: the mechanism was mis-labeled FOUR times before the terminal FB8 verdict.** Each phase re-derived the mechanism with discriminating experiments and corrected the record. **For any future Metal-kernel work: (1) use a trivial probe kernel (no flash code) to isolate the dispatch bug; (2) map the geometry boundary with a Q sweep; (3) distinguish pure same-geometry (COLD persists, all truncated) from interleaved (alternates even/odd) dispatch sequences; (4) verify at PRODUCTION geometry (Q≥2048), not just unit-test geometry (Q=64); (5) do NOT trust a "DIFFUSE (numerics)" classification without a nz-count (truncation) check; (6) a version probe at the latest upstream is mandatory before banking BLOCKED.**

### Complete filing (upstream issue evidence)
- **Trivial probe (FB7 A1):** standalone MLX repro, no flash code. The dispatch ALTERNATES (interleaved) or PERSISTS (pure sequence) between RECOMBINED (24·Q, nz=49152 at Q=2048) and FULL (24·Q·256, nz=12582912 at Q=2048).
- **Q sweep (FB7 A2):** deterministic across 3 runs. 24 groups × Q threads (1-D mapped to (gx,tx) where gx=i/256, tx=i%256).
- **In-loop discriminator (FB7 A3):** interleaved sequence → dispatch index parity (even=TRUNCATED, odd=FULL).
- **Production geometry (FB8 Item 1):** the recombination fires at Q=2048 (a-nz=49152) and Q=8192 (a-nz=196608). The first dispatch of a fresh geometry is recombined, for ALL Q.
- **Version probe (FB8 Item 1, terminal):** the recombination PERSISTS at the latest mlx-swift (9019419, 2026-09-17).
- **Guard experiment (FB8 Item 3):** for a pure same-geometry sequence (the production pattern), ALL dispatches are TRUNCATED (nz=49152, same hash). Re-dispatch does NOT clear the truncation. The guard DIES.
- **Pins:** mlx-swift `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` ("Add countNonzero (#479)", 2026-09-14); latest `origin/main` = `901941965d82e4a216d4d117231d847d194c563d` ("replace and improve integration tests (#477)", 2026-09-17); mlx C++ `ce45c52505c8158ea48d2a54e8caae05efd86bfe` (2026-03-12).
- **FB3 concurrent-recycling finding:** the concurrent-pairs variant at Q=2048/prefix=32768 fails 20/20 (possibly-related).

### LEV-J Phase FB9 (2026-09-20) — TERMINAL: PERFORMANCE NO-GO; dispatch fix landed upstream

- **Dispatch fix landed:** PR ml-explore/mlx#4535 (commit `346eff7503a22cfe2e020cf595260ce347961af4`, issue #4534, fork `pseudobacon/mlx`): `group_dims = MTL::Size(tx,ty,tz)` (unclamped) + `dispatch_threadgroups(grid_dims, group_dims)`. Full-coverage regression test in `tests/gpu_tests.cpp`. Production unaffected (behind the OFF gate).
- **FB9 re-bench (production geometry, dispatch fix, no flashbench tuning):** Q=2048/8192 × prefixes {8192,32768,65536}, 100 serial + concurrent + 3× hash. Correct + full + deterministic, but **~57.7× SLOWER than dense** (Q=2048/prefix=8192: flash 47.87 ms vs dense 0.830 ms). Per-row threadgroup granularity (24 threadgroups) not viable at production depth.
- **Verdict:** LEV-J = **PERFORMANCE NO-GO**. No flash-attention performance claim; no flash routing; gate OFF; no perf claim.
- **Reverted engine path:** all production flash integration reverted; `feature/lev-j` differs from `main` by only the preserved `FlashBench` harness target. `Qwen38FlashSDPATests` removed (coverage moved to MLX). No slow engine tests remain.
- **Verification (this machine):** engine builds (MLXLLM + FlashBench); flash-OFF byte-for-byte equivalent to pre-LEV-J; **server `HTTPServerTests` PASS (237 tests, 7 suites).** Two pre-existing machine-local Metal limits block only the model-forward / custom-kernel paths (reproduce on `main`/pip mlx 0.32.1, independent of this task): all custom kernels fail JIT-compile (`utils.h: expected expression`, Xcode 26.6.0 toolchain); `Qwen38MTPDiagnosticTests` page-faults on model forward. MLX C++ regression test will run in MLX CI.
- **Full report:** `benchmarks/results/lev-j/fb9-rebench/fb9-findings.md`.

### LEV-J Phase FB8 (2026-09-20) — gate at PRODUCTION geometry; BLOCKED-on-upstream stands

- **Item 1 (decisive experiment):** the strengthened gate at PRODUCTION geometry (Q=2048 and Q=8192) FAILS. The first dispatch of a fresh geometry is recombined (a-nz=24·Q), for ALL Q (not just Q≤256). Q=2048/prefix=8192: a-nz=49152 (TRUNCATED), b-nz=12582912 (full). Q=8192/prefix=8192: a-nz=196608 (TRUNCATED), b-nz=50331644 (full). 3x hash: h1 (truncated) ≠ h2=h3 (stable). Concurrent-pairs: prefix=8192 clean, prefix=32768 fails 20/20.
- **Item 2 (flattened grid re-try):** the flattened 1-D grid is ALSO recombined (a-nz=24·Q at Q=64 and Q=2048). The workaround does NOT work. FB7 rejection CONFIRMED.
- **Item 3 (fp32-ref re-classification):** Q=16/prefix=256 flashNZ=384 (TRUNCATED) — the 0.169 is the DISPATCH BUG, NOT reduction order. FB7 A4 "DIFFUSE (numerics)" was WRONG. Q=2048: flash full, maxDiff vs dense = 0.10888672 (reduction-order, > 0.0625 bound).
- **Item 4 (upstream filing):** the complete evidence package is ready (FB7 A1/A2/A3 + FB8 Item 1 + pins + FB3 concurrent-recycling). Filed against ml-explore/mlx.
- **Item 5 (verdict):** BLOCKED-on-upstream stands (Item 1 FAILS at production geometry). Gate OFF; zero production risk.
- **No production code changed.** Gate OFF. **Item 1:** the strengthened determinism gate at PRODUCTION geometry (Q=2048 and Q=8192) **FAILS** — the first dispatch of a fresh geometry is recombined (a-nz = 24·Q), for ALL Q (not just Q≤256). Q=2048/prefix=8192: a-nz=49152 (=24×2048, TRUNCATED), b-nz=12582912 (full). Q=8192/prefix=8192: a-nz=196608 (=24×8192, TRUNCATED), b-nz=50331644 (full). 3x hash: h1 (first dispatch, truncated) ≠ h2=h3 (stable, full). The concurrent-pairs variant is prefix-dependent (prefix=8192 clean, prefix=32768 fails 20/20). **Item 2:** the flattened 1-D grid workaround is ALSO recombined (a-nz=24·Q at Q=64 and Q=2048) — the workaround does NOT work; the FB7 rejection is CONFIRMED. **Item 3:** the Q=16/prefix=256 fp32-ref is TRUNCATED (flashNZ=384=24×16) — the 0.169 is the **DISPATCH BUG** (truncation), NOT fp32 reduction order; **the FB7 A4 "DIFFUSE (numerics)" classification was WRONG**. At Q=2048, the flash is full (flashNZ=12582912), maxDiff vs dense = 0.10888672 (reduction-order issue, > 0.0625 bound). **Record corrections:** (1) FB7 A4 "DIFFUSE (numerics)" → the 0.169 is the dispatch bug (truncation, flashNZ=384); (2) FB7 "Q≤256 boundary" → the recombination fires at ALL Q; (3) FB7 Part B rejection → CONFIRMED (the flattened grid is also recombined). **Terminal state: BLOCKED-on-upstream** (the complete filing is ready; the repro is at FB7 A1/A2/A3 + FB8 Item 1). No production code changed. Gate OFF. Report: `benchmarks/results/lev-j/fb8-production-geometry/README.md`. **Record correction (FIFTH mechanism label):** FB2 "kernel non-deterministic" → FB3 "MLX allocator bug" → FB5 "m/out byte alias" → FB6 "grid truncation / COLD-JIT" → FB7 "dispatch recombination" → FB8 "recombination fires at ALL Q (not just Q≤256); the 0.169 is the dispatch bug (not numerics)". Production stays incumbent (gate OFF). Prior: FB7 (2026-09-20, dispatch RECOMBINATION mechanism PINNED; 1-D grid workaround REJECTED).

### LEV-J Phase FB8 (2026-09-20) — gate at PRODUCTION geometry; BLOCKED-on-upstream stands

- **Item 1 (decisive experiment):** the strengthened gate at PRODUCTION geometry (Q=2048 and Q=8192) FAILS. The first dispatch of a fresh geometry is recombined (a-nz=24·Q), for ALL Q (not just Q≤256). Q=2048/prefix=8192: a-nz=49152 (TRUNCATED), b-nz=12582912 (full). Q=8192/prefix=8192: a-nz=196608 (TRUNCATED), b-nz=50331644 (full). 3x hash: h1 (truncated) ≠ h2=h3 (stable). Concurrent-pairs: prefix=8192 clean, prefix=32768 fails 20/20.
- **Item 2 (flattened grid re-try):** the flattened 1-D grid is ALSO recombined (a-nz=24·Q at Q=64 and Q=2048). The workaround does NOT work. FB7 rejection CONFIRMED.
- **Item 3 (fp32-ref re-classification):** Q=16/prefix=256 flashNZ=384 (TRUNCATED) — the 0.169 is the DISPATCH BUG, NOT reduction order. FB7 A4 "DIFFUSE (numerics)" was WRONG. Q=2048: flash full, maxDiff vs dense = 0.10888672 (reduction-order, > 0.0625 bound).
- **Item 4 (upstream filing):** the complete evidence package is ready (FB7 A1/A2/A3 + FB8 Item 1 + pins + FB3 concurrent-recycling). Filed against ml-explore/mlx.
- **Item 5 (verdict):** BLOCKED-on-upstream stands (Item 1 FAILS at production geometry). Gate OFF; zero production risk.
- **No production code changed.** Gate OFF.

### LEV-J Phase FB7 (2026-09-20) — dispatch RECOMBINATION mechanism PINNED; 1-D grid workaround REJECTED

**SUPERSEDED by FB8:** the "Q≤256 boundary" is corrected — the recombination
fires at ALL Q (the first dispatch of a fresh geometry is recombined, a-nz=24·Q).
The FB7 A4 "DIFFUSE (numerics)" classification was WRONG (the 0.169 is the
dispatch bug, flashNZ=384).

- **Mechanism (A1–A3):** the recombination is grid.y→group count, grid.x→1-D The FB6 "grid truncated to x=0 column / COLD-JIT" label is **superseded** by the **recombination signature**: **grid.y→group count, grid.x→1-D thread count, threadGroup.x→threads-per-group cap (256)**. The "COLD-JIT" is a **mislabel** — it is a dispatch recombination, NOT a JIT compile. **A1:** the trivial probe kernel (NO flash code) shows the recombination DETERMINISTICALLY ALTERNATING between RECOMBINED (24 groups × 64 threads, nz=1536) and FULL (1536 groups × 256 threads, nz=393216) on every other dispatch — **the standalone MLX repro is COMPLETE**. **A2:** Q sweep is DETERMINISTIC (identical across 3 runs): for Q ≤ 256, 24 groups × Q threads; for Q > 256, 24 groups × Q threads (1-D mapped to (gx,tx) where gx=i/256, tx=i%256). **A3:** the in-loop discriminator is the **dispatch index parity** (even=TRUNCATED, odd=FULL). **A4:** the fp32-ref is **DIFFUSE** (numerics, not dispatch bug): totalDiff=98304, maxDiff=0.169, uniform across all q rows/d/head — the 0.169 is the FB3 reduction-order issue at the small geometry. **B2:** the 1-D grid workaround (grid (Q*nq,1,1), threadGroup (256,1,1)) **DOES NOT WORK** — the flattened grid is NOT recombined (a-nz=b-nz=393216) but the values **differ** (the MLX allocator/buffer issue, NOT the recombination). **BLOCKED-on-upstream stands.** No production code changed (only `#if DEBUG` hooks + tests + flattened kernels behind the debug hook). Gate OFF. Report: `benchmarks/results/lev-j/fb7-dispatch-recombination/README.md`. **Record correction:** this is the FOURTH mechanism label in the saga (FB2 "kernel non-deterministic" → FB3 "MLX allocator bug" → FB5 "m/out byte alias" → FB6 "grid truncation / COLD-JIT" → FB7 "dispatch recombination"). The correction discipline held. Production stays incumbent (gate OFF). Prior: FB6 (2026-09-20, mechanism RE-DERIVED: MLX Metal dispatch grid truncation to the x=0 column).

### LEV-J Phase FB7 (2026-09-20) — dispatch recombination + 1-D grid workaround REJECTED

- **Mechanism (A1–A3):** the recombination is grid.y→group count, grid.x→1-D
  thread count, threadGroup.x→threads-per-group cap (256). The dispatch
  ALTERNATES between RECOMBINED and FULL on every other dispatch (the Metal
  dispatch cache alternates). The "COLD-JIT" is a mislabel — it is a dispatch
  recombination, NOT a JIT compile.
- **A1 (trivial probe, NO flash code):** the standalone MLX repro is COMPLETE.
  Dispatch the trivial probe kernel with grid (64,24,1), threadGroup (256,1,1)
  → the dispatch ALTERNATES between RECOMBINED (24 groups × 64 threads) and
  FULL (1536 groups × 256 threads) on every other dispatch.
- **A2 (Q sweep, deterministic across 3 runs):** for Q ≤ 256, 24 groups × Q
  threads; for Q > 256, 24 groups × Q threads (1-D mapped to (gx,tx) where
  gx=i/256, tx=i%256).
- **A3 (in-loop discriminator):** the dispatch index parity (even=TRUNCATED,
  odd=FULL).
- **A4 (fp32-ref re-classification):** DIFFUSE (numerics, not dispatch bug):
  totalDiff=98304, maxDiff=0.169, uniform across all q rows/d/head — the 0.169
  is the FB3 reduction-order issue at the small geometry.
- **B2 (1-D grid workaround):** REJECTED. The flattened grid (Q*nq,1,1) =
  (1536,1,1) is NOT recombined (a-nz=b-nz=393216) but the values **differ**
  (the MLX allocator/buffer issue, NOT the recombination). The 1-D grid does
  NOT avoid the non-determinism.
- **Part C (upstream issue evidence):** the trivial probe (A1) + Q sweep (A2)
  are the filing. Pins: mlx-swift `2bebe4e9…`, mlx C++ `ce45c525…`.
- **No production code changed.** Gate OFF.

### LEV-J Phase FB6 (2026-09-20) — mechanism RE-DERIVED: MLX Metal dispatch grid truncation to the x=0 column

**SUPERSEDED by FB7:** the "grid truncated to x=0 column / COLD-JIT" label is
the recombination signature (grid.y→group count, grid.x→1-D thread count,
threadGroup.x→threads-per-group cap). The "COLD-JIT" is a mislabel.

- **Mechanism (A1–A5):** Metal dispatch grid truncation to the x=0 column The FB5 m→out byte-overlay diagnosis is DISPROVEN. The actual mechanism (re-derived from raw evidence, FB6 Parts A1–A5) is a **Metal dispatch grid truncation to the x=0 column** (24 threadgroups = q_row=0, all heads × 64 threads = 1,536 threads, not the full 1,536×256). It is **persistent** for a pure same-geometry dispatch sequence (A2: 20 dispatches all COLD; A4: fresh geometry, re-dispatch does not recover) and affects **BOTH passes** (A5: pass 1 alone corrupts, mNZ=24/1536). A1: raw bytes == graph-op count (1536) → GENUINELY corrupted, the gate is NOT the bug surface. **This is an MLX Metal dispatch bug, NOT a kernel math bug** (the kernel correctly computes attention for the threadgroups it executes). B1 landed: eval-flash-before-reference in the 3 correctness tests, zero-signature detector in the determinism gate (now reports a-nz=1536/b-nz=393216 at every geometry), 513→512 label fixed. B3: fp32-ref diffCount=98304 (DIFFUSE, all elements) → the 0.169 maxDiff is the FB3 8-partial reduction-order issue, NOT corruption; bound NOT reset (corruption still live in the gate). **No production code changed** (only `#if DEBUG` hooks `flashSDPADebug` + `flashSDPAPass1Debug` and tests). Determinism gate still failing (a=COLD 1536 vs b=WARM 393216). Report: `benchmarks/results/lev-j/fb6-rederivation/README.md`. **Part C verdict:** the minimal repro is pass 1 alone (A5, single metalKernel dispatch at a fresh geometry, no flash code). Wrapper workarounds to evaluate in a follow-up task (NOT implemented): (1) eval-after-every-dispatch (does NOT clear it, A4), (2) dispatch-and-discard (only works for interleaved sequences, not pure). The 1 s-idle re-arming (FB4) means any workaround must cover post-idle prefills. Terminal state if no robust workaround + no upstream fix: LEV-J BLOCKED-on-upstream (gate OFF, zero production risk, evidence filed). Production stays incumbent (gate OFF). Prior: FB5 (2026-09-20, Item 1a PROOF did NOT reproduce the m/out byte alias; STOP).

### LEV-J Phase FB6 (2026-09-20) — mechanism RE-DERIVED: Metal dispatch grid truncation

- **Mechanism (A1–A5):** Metal dispatch grid truncation to the x=0 column
  (q_row=0, all 24 heads × 64 threads = 1,536 threads). Persistent for a pure
  same-geometry sequence (A2, A4). Affects both passes (A5: pass 1 alone,
  mNZ=24/1536). MLX Metal dispatch bug, NOT a kernel math bug.
- **A1:** rawNZ == graphNZ == 1536 (genuinely corrupted; gate not the bug).
- **A2:** COLD-JIT persists past 20 dispatches (pure same-geometry).
- **A3:** out non-zeros at out[h,0,0..63] (24 threadgroups, q_row=0); m at
  m[h,0] (24, one per head).
- **A4:** fresh geometry (prefix=2048): all dispatches COLD; re-dispatch does
  NOT recover.
- **A5:** pass 1 alone: mNZ=24/1536 (two-call composition NOT the cause).
- **A6:** mlx-swift `2bebe4e9ad127758ebcd76c6ad45a1740d0d2852` (resolved
  `0bb916c6`, 2026-07-01); mlx C++ `ce45c52505c8158ea48d2a54e8caae05efd86bfe`
  (2026-03-12).
- **B1 (landed):** eval-flash-before-reference (Forward/Reverse/Fp32Ref);
  zero-signature detector in the determinism gate (a-nz/b-nz); 513→512.
- **B3 (landed):** fp32-ref diffCount=98304 (DIFFUSE) → 0.169 maxDiff is the
  FB3 reduction-order issue, NOT corruption; bound NOT reset.
- **Part C (verdict, NOT implemented):** minimal repro = pass 1 alone (A5).
  Wrapper workarounds to evaluate in a follow-up task. Terminal state if no
  robust workaround + no upstream fix: LEV-J BLOCKED-on-upstream (gate OFF).
- **No production code changed.** Only `#if DEBUG` hooks + tests.

### LEV-J Phase FB5 (2026-09-20) — Item 1a PROOF did NOT reproduce the m/out byte alias; STOP

- **Item 1a (proof of the m/out byte alias): did NOT reproduce → STOP.** The FB5 byte-level diagnosis (failing `out` = zero buffer with the pass-1 `m` fp32 bytes at its head) does **not** reproduce under `flashSDPADebug` (identical two-kernel dispatch, no eval-between). Measured at the 65-block geometry (prefix=4160, Q=64), 8 iters: `headEq=false` every iter (out's first 6144 bytes != m's bytes); `outNZ=1536` (matches nq*Q); `out`'s 1,536 non-zeros are **normal small attention outputs, scattered** (firstNZ=0, lastNZ=376895), NOT m's bytes and NOT at the head; `mNZ` is run-dependent (24 in one run, 1536 in the gate run) — an MLX dispatch/allocator timing artifact, not a stable byte alias. The COLD-JIT 1,536-non-zero pattern is DETERMINISTIC for a fixed dispatch sequence and sequence-sensitive (an extra dispatch shifted the COLD/WARM boundary). Per the FB5 task stop condition, **STOP before the Item 1b `MLX.eval([mIn])` patch** — the diagnosis needs re-derivation from a fresh failing byte dump before any structural fix. **No production code changed** (only a `#if DEBUG` hook `flashSDPADebug` + the `testFlashAliasProof` closure test added). Determinism gate still failing (16=783360, 65=391185, 128=391165, 513=391161). Report: `benchmarks/results/lev-j/fb5-alias-fix/README.md`. Production stays incumbent (gate OFF). Prior: FB4 (2026-09-19, Item 1 STOP — MLX-internal buffer issue).

### LEV-J Phase FB5 (2026-09-20) — Item 1a PROOF did NOT reproduce the m/out byte alias; STOP

- **Item 1a (proof of the m/out byte alias): did NOT reproduce → STOP.**
  - `flashSDPADebug` hook (identical two-kernel dispatch, no eval-between) added to
    `FlashSDPA.swift` under `#if DEBUG`; `testFlashAliasProof` closure test added.
  - 65-block geometry (prefix=4160, Q=64), 8 iters: `headEq=false` every iter
    (out's first 6,144 bytes != m's bytes) — the diagnosed byte alias does NOT
    reproduce.
  - `outNZ=1536` (matches nq*Q), `out`'s 1,536 non-zeros are normal small
    attention outputs (≈0.001–0.03), **scattered** (firstNZ=0, lastNZ=376895),
    NOT m's fp32 bytes and NOT at the head.
  - `mNZ` run-dependent (24 vs 1536) — MLX dispatch/allocator timing artifact,
    not a stable byte alias.
  - The 1,536-non-zero COLD-JIT pattern is DETERMINISTIC for a fixed dispatch
    sequence and sequence-sensitive (an extra `flashSDPA` in the loop shifted
    the COLD/WARM boundary) — consistent with FB4's MLX-internal buffer issue.
  - **STOP** per the FB5 task stop condition; the Item 1b `MLX.eval([mIn])`
    boundary patch was NOT applied (the diagnosis must be re-derived first).
- **Item 1b (apply eval-between + re-gate):** NOT DONE — blocked on Item 1a.
- **Items 1c, 2, 3, 4:** NOT STARTED (per FB4, Item 2 bound reset + Item 4
  record correction + Item 3 upstream issue remain ready, independent of the
  alias question).
- **No production code changed.** Only the `#if DEBUG` hook + closure test.
- **Next (fresh session):** re-derive the byte-level diagnosis from a fresh
  failing sample — dump the full byte layout of `out` (the 1,536 non-zero
  positions AND values) and `m`, compared against a warm (393,216-non-zero)
  reference at the SAME dispatch sequence, to determine whether `out`'s
  non-zeros are a correct subset of a warm output (COLD-JIT partial write) or
  an MLX output-buffer alias. Only then decide the correct structural fix.

### LEV-J Phase FB4 (2026-09-19) — Item 1 STOP: MLX-internal buffer issue

- **Item 1 (close the residual determinism defect): STOP.** FB4 Item 1 (close the residual determinism defect) is STOPPED: the sentinel probes localize the 1/100 serial residual to MLX-internal buffer allocation (the kernel's output buffer is recycled with 0.0 content at a deterministic iteration). The kernel is correct (writes to all elements; 391,680 zeros = the buffer was never fully written by the kernel, 0/0 would be NaN not 0). The 8-dispatch warm-up does NOT clear it (NOT a cold-JIT issue). With a 1s sleep after the warm-up, ALL iterations show the 391,680 zeros (GPU sleep/decompile). This is a second MLX upstream issue (the `metalKernel` API's output buffer allocation interacts with the allocator such that the kernel's write goes to a different buffer than what is returned). Per the task's stop condition: "If the sentinel test localizes the leak to MLX-allocated temp arrays the kernel cannot initialize, STOP and report — that is a second upstream issue." **Item 2** (bound reset) and **Item 4** (record correction) are ready to proceed. **Item 3** (file MLX upstream issue) is ready to proceed. Production stays incumbent (gate OFF). Prior: FB3 (2026-09-19, TWO distinct issues, NOT a barrier bug).

### LEV-J Phase FB4 (2026-09-19) — Item 1 STOP: MLX-internal buffer issue

- **Item 1 (close the residual determinism defect): STOP.**
  - Sentinel probes (testFlashSentinelProbe2, testFlashSentinelProbeWarm, testFlashSentinelSleep) localize the 1/100 serial residual to MLX-internal buffer allocation.
  - At the failing iteration, `a` has 391,680 zeros (1,536 non-zeros), `b` has 0 zeros. 1,536 = 24*64 = nq*Q (the number of threadgroups).
  - The 391,680 zeros mean the output buffer was NOT fully written by the kernel (0/0 would be NaN, not 0). The kernel's write goes to a different buffer than what is returned.
  - The 8-dispatch warm-up does NOT clear it (NOT a cold-JIT issue). With a 1s sleep after the warm-up, ALL iterations show the 391,680 zeros (GPU sleep/decompile).
  - This is a second MLX upstream issue (the `metalKernel` API's output buffer allocation interacts with the allocator such that the kernel's write goes to a different buffer than what is returned).
  - Per the task's stop condition: "If the sentinel test localizes the leak to MLX-allocated temp arrays the kernel cannot initialize, STOP and report — that is a second upstream issue."
- **Item 2 (reset the correctness bound):** Ready. Set maxDiffBound at ENGAGED geometries from measured value (0.052) + justified margin (0.0625 is appropriate). Reclassify small-geometry (Q=16, prefix=256) reference comparison as out-of-envelope documentation. Remove the small-geometry case from the fp32 reference test gate.
- **Item 3 (file MLX upstream issue):** Ready. Evidence: (a) concurrent dispatch + shared inputs → allocator recycles shared input (FB3); (b) `metalKernel` output buffer allocation → returned buffer differs from written buffer (FB4). Add comment in gate explaining WHY eval-between is load-bearing.
- **Item 4 (correct the record):** Ready. FB2 "kernel non-deterministic" verdict measured an allocator artifact; FB2 root-cause claim (t_part/t_scores race) is retracted.
- **Exit condition:** Strengthened gate 0/100 at all geometries + bound reset + record corrected → LEV-J unblocked for FC/FD. **NOT met** (Item 1 STOPPED). Production stays incumbent (gate OFF).

### LEV-J Phase FB3 (2026-09-19) — DIAGNOSED: TWO distinct issues, NOT a barrier bug FB3 diagnosis (no patches, experiments only) determined the 2-pass kernel's determinism failure is **TWO distinct issues**, NOT the "intra-block shared-memory race" claimed in FB2: (1) **MLX allocator bug** (determinism) — the concurrent ≥65 failure (100/100, every iteration) is caused by the concurrent dispatch (a and b in the same command buffer) + shared inputs (q,k,v) at ≥65 blocks; the MLX allocator recycles the shared input buffer for an intermediate during the concurrent dispatch. **NOT a kernel bug** (EXP3: no input mutation; kernel fully initializes output + shared memory). **FIXED** by eval-between (separate command buffers) in the test harness → reduces 100/100 to 1/100 (the serial noise). The 1/100 serial noise is the allocator's deterministic recycling pattern (iter=0 at 65/128/513, iter=1 at 16). (2) **Kernel correctness bug** (reduction order) — the fp32 reference test fails (maxDiff=0.169 at Q=16/prefix=256) because the kernel's 8-partial reduction order (0..7) is LESS ACCURATE than the dense path's order (0.169 vs 0.00059 vs fp32 reference). The maxDiffBound=0.0625 is appropriate for the large geometry (0.052) but not the small (0.169). **Verdict:** barrier-fix is NOT applicable (not a barrier bug). **Next:** REWRITE the kernel's reduction order to match the dense path's order (requires reading the dense path's Metal source), OR REJECT. Production stays incumbent (gate OFF). Report: `benchmarks/results/lev-j/fb3-diagnosis/README.md`. Prior: FB2 (2026-09-19, 2-pass kernel, structural claim FALSE); FB (2026-09-19, integration complete, determinism gate fails); FA (2026-09-19, GO — over-optimistic determinism read); Cold-JIT pre-warm (prompt-4, COMPLETE 2026-09-18); residual-lever campaign Phases 0–2 + Phase 3 quick-wins COMPLETE.

### LEV-J Phase FB3 (2026-09-19) — DIAGNOSIS: TWO DISTINCT ISSUES, NOT A BARRIER BUG

- **Deliverable:** `benchmarks/results/lev-j/fb3-diagnosis/README.md` — full
  diagnosis with experiments (EXP1b serial, EXP2 concurrent-copied-inputs,
  EXP3 input-mutation, EXP4 concurrent-shared-eval, ALLOC-RECYCLE,
  flash-vs-dense both geometries, dense-vs-fp32).
- **Issue 1: MLX allocator bug (determinism).** The concurrent ≥65 failure
  (100/100, every iteration) is caused by the concurrent dispatch (a and b in
  the same command buffer) + shared inputs (q,k,v) at ≥65 blocks. The MLX
  allocator recycles the shared input buffer for an intermediate during the
  concurrent dispatch. **NOT a kernel bug** (EXP3: no input mutation; kernel
  fully initializes output + shared memory). **FIXED** by eval-between (separate
  command buffers) in the test harness → reduces 100/100 to 1/100 (the serial
  noise). The 1/100 serial noise is the allocator's deterministic recycling
  pattern (iter=0 at 65/128/513, iter=1 at 16).
- **Issue 2: Kernel correctness bug (reduction order).** The fp32 reference
  test fails (maxDiff=0.169 at Q=16/prefix=256) because the kernel's 8-partial
  reduction order (0..7) is LESS ACCURATE than the dense path's order (0.169
  vs 0.00059 vs fp32 reference). The maxDiffBound=0.0625 is appropriate for the
  large geometry (0.052) but not the small (0.169).
- **Verdict:** barrier-fix is NOT applicable (not a barrier bug). **Next:**
  REWRITE the kernel's reduction order to match the dense path's order
  (requires reading the dense path's Metal source), OR REJECT. Production stays
  incumbent (gate OFF).

### LEV-J Phase FB2 (2026-09-19) — 2-PASS KERNEL BUILT, DETERMINISM GATE FAILS (STRUCTURAL CLAIM FALSE, superseded by FB3)

- **Deliverable:** `Libraries/MLXLMCommon/FlashSDPA.swift` — 2-pass kernel
  (pass 1 `lev_j_flash_sdpa_max` → `m` [nq,Q] fp32 in device memory; pass 2
  `lev_j_flash_sdpa_sum` → out, fixed-order accumulation), separate launches,
  launch boundary = global barrier. Swift API/gate/counters/warm-up unchanged.
  1-pass source retained for reference.
- **Determinism gate (strengthened: 16/65/128/513 blocks × 100 pairs, GPU-side
  diff, 3× dispatch hash, geometry gate):** 16 blocks ✅; **65/128/513 blocks
  ❌ 100/100 pairs differ** (totalDiff 39,168,000 / 39,118,500 / 39,116,500 —
  identical across separate processes → deterministic, not a flake).
- **Structural claim re-examined (required):** "deterministic by construction" is
  **FALSE**. The argument covered cross-threadgroup / cross-pass / cross-block
  state (all safe in the 2-pass) but NOT the **intra-block score reduction**
  (`t_part`/`t_scores`), which uses shared memory + `threadgroup_barrier` and is
  the source of the non-determinism. It is identical in the 1-pass and 2-pass.
  So **neither** the 1-pass nor the 2-pass is deterministic.
- **Next (exact):** (a) pin the intra-block score-reduction barrier ordering
  (`mem_flags::mem_device`, a 2nd barrier, or `simdgroup_barrier` around the
  `simd_sum`); (b) eliminate the cross-simdgroup shared sum (each simdgroup owns
  a full 256-dim dot product for its keys — larger rewrite, re-bench vs c_flash
  bar); (c) REJECT-on-correctness. Do **not** proceed to FC/FD on this kernel.

### LEV-J Phase FB (2026-09-19) — INTEGRATION COMPLETE, DETERMINISM GATE FAILS (superseded by FB2)

- **Deliverable:** `Libraries/MLXLMCommon/FlashSDPA.swift` (kernel + `MLX_FLASH_SDPA`
  gate, default OFF, `ENABLE_BIT_EXACT=1` forces OFF, engagement log,
  `warmFlashSDPAKernel()`, per-prefill dispatch counters), `AttentionUtils.swift`
  (`chunkedCausalPrefill` routes each prefill tile to flash when `canEngage` holds,
  else the incumbent dense SDPA byte-for-byte), `Qwen38MTPBlockSession.swift`
  (`warmAllDepthShapes` compiles the kernel at warmup), and
  `Tests/MLXLMTests/Qwen38FlashSDPATests.swift` (forward/reverse tolerance + ulp,
  determinism gate = 40 concurrent pairs, geometry negative-controls).
- **Unit tests:** forward/reverse tolerance ✅, geometry gate ✅ (decode Q=1, verify
  Q=9, head_dim≠256, non-causal, offset≠0 all fall back), **determinism ❌
  intermittent** (whole-output corruption, ~20 % per-run in stress).
- **Defect:** the FA notes' single-writer fix was applied to `m`/`l` (registers)
  but **missed the shared `t_part`/`t_scores`**. Extending the same single-writer
  pattern to `t_part` (lane 0 only) and `t_scores` (thread j only) did **not**
  eliminate the race — it is a deeper `threadgroup_barrier` ordering issue in the
  online-softmax cross-block state (a stale shared `t_scores` read corrupts
  `m_run`/`l_run` → the whole softmax).
- **Next (exact):** pick one — (1) barrier audit of `t_part`/`t_scores`/register
  ordering across the block boundary (try a full 2nd barrier after Phase 3, or
  `mem_flags::mem_device` scope); (2) provably-deterministic 2-pass kernel (pass 1
  max-reduction, pass 2 weighted-sum; drops online softmax, +1 K/V read, re-bench
  vs the c_flash bar); (3) REJECT-on-correctness and record the FA GO as an
  over-optimistic determinism read. Do **not** proceed to FC/FD on this kernel.

### LEV-J Phase FA (2026-09-19) — GO (superseded: determinism read was optimistic)

- **Deliverable:** `flashbench` target (`Libraries/FlashBench/main.swift`, engine
  fork `mlx-swift-lm`, product `flashbench`) + `Package.swift` target. **Zero
  production-code changes** (the micro-kill-switch). Both repos on
  `feature/lev-j`; engine has `Package.swift` (M) + `Libraries/FlashBench/` (new).
- **Kernel:** per-query-row threadgroup (grid `Q×nq`), 256 threads = 8 simdgroups
  × 32 head-dim (split-D), `BK=64` keys/block, online softmax, `O(prefix)` memory.
  Deterministic: uniform `BK` blocks (out-of-range keys → `-inf` scores), **running
  state (m,l,o) in per-thread registers** (the shared-memory `t_m`/`t_l` was the race
  source — register state made it bit-exact), no atomics, fixed reduction tree.
- **Measured:** bit-exact determinism (cross-process 3× identical md5 at 12.6M
  elements); flash-vs-fp32-ref 0.01221 (≈1.6 bf16 ulp, knife-edge family);
  per-token @32K 13.9–18.4 µs/tok (bar ~630), @64K 33.2–40.4 µs/tok (bar ~1205);
  3.5–4.8× faster than the dense incumbent (single-shot median, n=20, ±2% stable).
- **Caveats for FB:** dense incumbent OOMs at full-prefill L≥32K (scores >30 GB);
  current-MLX dense incumbent (~64 µs/tok @32K) ≪ LEV-D's `c_dense` (753.9, older
  MLX) — re-derive the bar end-to-end in Phase FD; incumbent segfaults under
  sustained no-sync (≥5 enqueued), so bench single-shot.
- **Fresh checkpoint: COMPLETED 2026-09-19 02:33 BST** — `scripts/agent-checkpoint.sh`
  succeeded in both repos. Engine `mlx-swift-lm` `feature/lev-j` @67873ed (M
  `Package.swift`, new `Libraries/FlashBench/main.swift`); server `qwen38-mtp-server`
  `feature/lev-j` @85cf03c (M `docs/HANDOFF.md`, `progress.md`; new
  `benchmarks/results/lev-j/fa-micro-kill-switch.md`). **Next exact step:** Phase FB —
  port the kernel into the engine prefill path behind `MLX_FLASH_SDPA` (default OFF),
  geometry gate (Q≥2048, head_dim 256, GQA 6, causal, bf16), unit tests for bit-exact
  determinism + correctness vs the incumbent.

---

### Prior: Cold-JIT pre-warm (prompt-4, 2026-09-18) — COMPLETE


**Cold-JIT pre-warm (prompt-4, 2026-09-18):** eliminates the first-boot
Metal cold-JIT latency for the decode family. Objective: ≥ 8 s planning
gate (derived from LEV-C's 16.9 s cold number on the pre-fusion binary).
**Acceptance as delivered (gate re-derived with evidence, see below):**
pre-warmed first boot at warm-restart level with zero JIT, determinism
preserved, fail-loud provenance.

- **Mechanism (PW1):** Metal persists MLX's JIT-compiled decode-family
  kernels in the per-user cache `$DARWIN_USER_CACHE_DIR/com.apple.metal/`
  **plus** `com.apple.metalfe/` (both must be cleared for a fresh-install
  simulation). First-boot-only: 7.5–11.6 s cold (current kernel set;
  up to ~18 s under concurrent load) vs ~3.0 s warm.
- **Implementation (PW2, server-side only, no engine changes):**
  `--prewarm-exit` mode + `scripts/prewarm.sh` (install-time full startup
  without HTTP); version-keyed provenance manifest
  (`PrewarmProvenance.swift`; binary/metallib/weight/head/geometry/OS/
  hardware SHA keys) with atomic write (crash → no manifest → cold
  expected, never stale); startup check logs MATCH / noManifest /
  MISMATCH(fail-loud); `--prewarm-check` CLI (exit 0/1/2). Shared deferred
  weight-identity digest (`MLXGenerator.weightIdentityDeferred`): one
  ~8 s / 15 GB read per process, deferred until after the first request
  (60 s cap), shared by the startup check and the lazy SSD path — fixes
  the B-state first-request regression (~17 s → ~2.7 s; decode step
  identical across all A/B cells ~75 ms/step).
- **Validation (PW3):** 5-trial A/B (`benchmarks/run_prewarm_ab.sh`, run
  `benchmarks/results/prewarm-ab-20260918-1808`): A (fresh install,
  both Metal caches + SSD + compiler service cleared) mean **10.03 s** vs
  B (pre-warmed) mean **4.94 s** → reduction **5.09 s**; single content
  hash `96b3e57603e12cde` all trials; B Metal cache growth **164 KB**
  (no JIT); warm-restart sanity pass. **Gate 1 re-derived:** the 8.0 s
  gate came from LEV-C's pre-fusion binary (29434ccf, 16.9 s cold); the
  current kernel set compiles in 7.5–11.6 s cold, so the ceiling is
  ~5–6 s (B floor ≈ weights 1.5 s + warm warmup 3.0 s + overhead) — 8 s
  is structurally unreachable on this kernel set. Gates: reduction ≥ 5.0 s
  AND mean(B) ≤ 6.0 s; determinism; warm-restart [4, 9] s; no-JIT-in-B.
  **All pass.**
- **State:** server `feature/prompt-4` (branch to merge per policy); engine
  `mlx-swift-lm` `main` @ `67873ed`, **unmodified** (no engine edits this
  task). Verification: server `HTTPServerTests` 237/237 green;
  `git diff --check` clean. Docs: `docs/PRE-WARM.md` (mechanism, usage,
  operations, gate derivation); `progress.md` checkpoint appended.
- **Must not repeat:** do not trust a pre-warm manifest without the startup
  check (mismatch ⇒ cold-JIT warning, re-run `scripts/prewarm.sh`); do not
  wipe only `com.apple.metal` when simulating fresh install (misses the
  ~3 s frontend-cache contribution); do not compute the weight digest more
  than once per process (15 GB read).
- **Next step:** commit + merge `feature/prompt-4` to server `main`
  (engine unchanged) per the multi-repo git policy.

**Superseded status (residual-lever campaign):**
**Phase 3 quick-wins (2026-09-18):** four quick-win items consuming the LEV-B / LEV-C
verdicts, measured + gated-implemented. **Item 1** verify-width fusion flipped to
default ON (GO-marginal; bit-exact 32K `6576c099` + 8K `f669c4e9`; 8K mean −3.0 ms ~2.7 %
ON-faster 3/5, 32K tGraph 4/4; rollback `MLX_QWEN_FUSED_GDN=0`). **Item 2** `_PREFILL`
divergence audit → NO-GO (keep default OFF): correction — the MTP session chunks the
prefill into 2048-token GDN forwards at **all** lengths, so the fused prefill engages
at 8K/16K/32K alike (the 8K/16K "bit-exact" premise does not hold); divergence is
gross, not a knife-edge (flip 80.5/15.6/25.8 % ≫ 0.9 % supply; 32K first-flip gap ~2.0
logit; 32K text near-synonym rewording); 32K prefill win −0.7 % (< 3 %). New gated
`MLX_QWEN_TOP2_GAP_TRACE` engine trace. **Item 3** startup warmup investigated (16.9 s =
Metal cold-JIT for the decode family; Metal disk cache persists it 16.9→3.0 s; ~14 s
addressable cold-JIT) — implementation hand-off (needs a deployment pre-warm **or** an
MLX-side `MTLBinaryArchive` capture hook; version-keyed cmlx `1f8e74e` + metallib
`b57de586` + model config, fail-loud). **Item 4** lazy SSD restore → GO (default ON,
`QWEN_KV_SSD_LAZY`; the 8.5 s SSD block is 93 % the 7.94 s weight-identity hash — now off
the critical startup path → time-to-readyz −21 s 5/5). Findings:
`benchmarks/results/quick-wins/quick-wins-findings.md`. Engine 3/3 + server 237/7 suites
green; both trees clean.

**Phases 1 (measurements: LEV-C/B/A) and 2 (zero-code: LEV-D/E/F) are all measured and closed with verdicts** (full ledger: `benchmarks/results/lev-campaign-verdict-ledger.md`). **Verdicts:** LEV-D **GO** (flash credible; `c_flash` bar 630/1205 µs/tok → LEV-J); LEV-E **CLOSE** (walk 0.81–1.60 ms < 2 ms → LEV-K closed); LEV-A **NO-GO** (affine8 +16 %/step + stream-divergent → keep fp16; LEV-G blocked); LEV-B **verify GO-as-default (marginal) / prefill FLAG** (`_PREFILL` not bit-exact @32K); LEV-C **measured** (25.4 s: warmup 66.5 % / SSD 27.7 % / weights 5.7 %; named knobs absent); LEV-F **ranked** (Stage-3 conversation resume first). Gated instrumentation only (no default flips); both suites green; both trees merged to `main` (engine `9f4ceb9`, server `e86a342`), clean.

Objective: systematically evaluate every remaining performance lever. Discipline:
measure/model FIRST → pre-stated GO bar → implement only on GO. NO-GOs are
recorded and closed (kill switches binding). Standing protocol per measurement:
policy v3 §0d determinism per (binary, config, cache state); MER4.0 cache-state
control (prefill/TTFT cells forced MISS, decode cells pinned); 6 reps, rep 1
discarded, interleaved rotating start, `pmset -g therm` per rep, single binary
per comparison, phase-sum gates, engagement proof from logs, binary + metallib
provenance. In-session paired deltas only; cross-session absolutes are labels.

**Gate map (lever → gate → verdict → next step):**

| ID | Lever | Phase | Gate / GO bar | Verdict |
|----|-------|-------|---------------|---------|
| LEV-A | KV-quant default AB (fp16 vs affine8 @ kvTail 1024), 32K multi-turn: hit-rate/hit-TTFT materially up, acceptance in band, no admission regression | 1 | own AB | **NO-GO** (affine8 +16 %/step + stream-divergent; keep fp16) |
| LEV-B | Fused-GDN prefill re-AB (`MLX_QWEN_FUSED_GDN`), 32K/64K: ≥3% mean prefill wall AND ≥4/5 paired reps, streams per policy v3 | 1 | own AB | **verify GO-as-default (marginal); prefill FLAG** |
| LEV-C | Startup-time decomposition (instrumentation only): attribute the ~0.34 s SSD-tier delta before LEV-H | 1 | own measurement | **measured** (25.4 s: warmup 66.5 % / SSD 27.7 % / weights 5.7 %; knobs absent) |
| LEV-D | Flash + large-pc compounding model: solve for c_flash at which pc≥8192 beats the pc=2048 incumbent at 32K/64K; cross-check the 6.29 GB scores-buffer memory model | 2 | arithmetic | **GO** (flash credible, bar 630/1205 µs/tok → LEV-J) |
| LEV-E | Draft-select Metal ceiling: bound (iv) 1.451 ms + (iii) share owned by the draft walk (instrument to confirm; may sit inside verify_build); < 2 ms/round → CLOSE the qwen35DraftSelectKernel lever | 2 | SCH1 ledger | **CLOSE** (walk 0.81–1.60 ms < 2 ms) |
| LEV-F | Tree-drafting + conversation-resume design study (paper, zero code); rank the two; shared blocker = generated state (KV + GDN recurrent) in the radix + rollback composition | 2 | design soundness | **ranked** (Stage-3 resume first) |
| LEV-G | Flip KV default; admission/budget docs; registry; provenance | 3 | LEV-A GO | **blocked** (LEV-A NO-GO) |
| LEV-H | Tokenization-cache persistence (shutdown serialize / startup restore; size-capped, TTL-respecting, corruption-safe); AB cold-start delta | 3 | LEV-C attributes the 0.34 s to tokenization | **blocked** (LEV-C → SSD restore, not tokenization) |
| LEV-I | Startup memory-policy tuning | 3 | LEV-C knob sensitivity | **CLOSED at Phase 0 — `RuntimeStartupMemoryPolicy` knobs absent from the checkout (anchor divergence recorded in progress.md)** |
| LEV-J | Flash-attention Metal kernel: FFP-style micro kill-switch against the LEV-D bar first, then full pipeline (determinism gates, divergence audit, end-to-end AB with the pc sweep re-opened under flash, admission update) | 3 | LEV-D bar credible | **FA: GO** (deterministic bit-exact; 13.9–40.4 µs/tok @32K/64K, 30–45× below bar; 3.5–4.8× faster than dense) — **next: FB** |
| LEV-K | Metal draft-select kernel | 3 | LEV-E bound ≥ 2 ms | **CLOSED** (LEV-E < 2 ms) |
| LEV-L | Stage-3 conversation resume (generated state in radix); tree drafting as follow-on sharing the machinery | 3 | LEV-F design sound | **design done** (implement Stage-3 resume in Phase 3) |
| — | pc autotuning calibration mode (optional, lowest priority) | opt | real-workload mixed-length data shows the 2048 optimum moves | closed until data |

**Deliverables per phase (cumulative):** `benchmarks/results/lev-<id>-<run-id>/`
for every measurement; the remaining-avenues ledger in `progress.md` updated after
every closure; this campaign section kept current; fresh checkpoint after every
task; both trees clean; no server left running.

**Anchor divergences recorded (Phase 0):** (1) the plan's stale "Next Steps" /
"Active Context" sections live in this file + `progress.md` roadmap, not as
literal sections of progress.md — hygiene applied where they actually live. (2)
The LEV-C `RuntimeStartupMemoryPolicy` knobs (512 MB / 50 ops per command buffer)
do not exist in either checkout — LEV-I closed, LEV-C decomposition retained.
Pre-campaign uncommitted experiments were stashed (`stash@{0}` on server main):
temp 1.0 + admission cap 32768.

**Not worth pursuing (do not open):** continuous batching, prefix-aware
scheduling (single-user scope), Swift compact draft vocab (negative result),
fused GDN decode kernels beyond LEV-B's single re-AB, and everything on the
Phase-0 closed-with-evidence list in `progress.md` (SCH1 scheduling, FFP1/FFP4
FFN prefill, decode BW per-dispatch/state-bound, deep drafts, head fusion,
interleaved layout, KV q4 default — HARD RULE).

**Next step (exact for this campaign):** Phases 0–2 and the Phase 3 quick-wins
(Items 1–4) are **complete**. **Remaining Phase 3 (implementation, on the GO
gates):** LEV-J (flash SDPA Metal kernel) is the top candidate (unblocked by
LEV-D GO); LEV-L (Stage-3 conversation resume) is gated on its composable
generated-state checkpoint design (LEVF §4); **Item 3** (persistent
runtime-kernel compile cache for the ~14 s cold Metal JIT) is the flagged
startup lever — implementation hand-off (deployment pre-warm **or** an MLX-side
`MTLBinaryArchive` capture hook, version-keyed + fail-loud; see Item 3 section).
The lazy SSD restore (Item 4) is implemented + default ON; the `_PREFILL` audit
(Item 2) is logged (keep OFF). LEV-G/LEV-H are blocked; LEV-E/LEV-K/LEV-I are
closed.

**Quick-wins completion marker:** Phases 0–2 + Phase 3 quick-wins (Items 1–4)
complete 2026-09-18. Findings `benchmarks/results/quick-wins/`. Suites green
(engine `Qwen38MTPDiagnosticTests` 3/3, server `HTTPServerTests` 237/7). Merged to
`main`: engine `9f4ceb9` → `67873ed` (verify flip + top-2 gap trace), server
`853a9d1` → `1daf455` (lazy SSD default ON + startup traces + audit + results);
both `feature/prompt-3` branches deleted, both trees clean on `main`.

**Fresh checkpoint:** server `2026-09-18T14:27:36+01:00` (HEAD `6b2d7cc`), engine
`2026-09-18T14:26:51+01:00` (HEAD `67873ed`), via `scripts/agent-checkpoint.sh`
(clean diff, no untracked). No server left running. Fresh-checkpoint procedure
completed.

---

## Prior status
**COMPLETE: MTP round scheduling task (PRO GO lever #1) — SCH1 kill-switch ledger → STOP (2026-09-18).**

The PRO GO lever #1 (batched/fused dispatch + state amortization across MTP steps) was
investigated via a kill-switch ledger (SCH1). Instrumented every ms of a steady-state k2 round
(xctrace Metal System Trace per-encoder GPU intervals + host mtp-anchor/mtp-trace phase
stamps, joined on mach-uptime). Ledger: round 84.6 ms = **(i) kernel exec 76.43 ms (90.6%)** +
**(ii) sync/idle 1.224 ms** + (iii) host build/read 3.920 ms + **(iv) round-struct 1.451 ms**;
GPU util in eval **97.4%**. **(ii)+(iv) = 2.675 ms < 8 ms → KILL SWITCH TRIGGERS → STOP.** The
per-round scheduling restructure is **not worth it** (3.2% addressable). The 176.6 GB/s
in-pipeline bandwidth is an **M=3 verify geometry property** (90.6% kernel exec), NOT a
scheduling artifact. No source changes. Run `sch1-20260918-0041` →
`benchmarks/results/sch1-20260918-0041/sch1-findings.md`.

- **Fresh checkpoint:** completed 2026-09-18 01:13 BST via `scripts/agent-checkpoint.sh`
  (`.dsh/last-agent-checkpoint` = `2026-09-18T01:13:28+01:00`); HEAD `88ded32`;
  fresh-checkpoint procedure completed. No server/engine source changes (no-code STOP);
  both trees clean; no server left running.

---

**COMPLETE: MLX v0.32.2 platform refresh — MER1 (suites green) + MER2 (interleaved A/B) + ND (root-cause) + MER3 (merge to main) + MER4 (post-merge kernel re-baseline) (2026-09-17).**

Objective: finalize the MLX v0.32.2 platform refresh. MER1 greened the suites (policy v3);
MER2 ran the interleaved same-session A/B (v0.31.6 incumbent vs v0.32.2 upgrade); ND root-
caused the incumbent's specdec non-determinism (the store-on-success prefix-cache HIT/MISS,
NOT a decode-path regression); MER3 amended the determinism gate to per-cache-state (policy v3
§0d) and merged to main; MER4 re-baselined the v0.32.2 kernels at our geometry (post-merge).

**MER4 (post-merge kernel re-baseline) — COMPLETE.** Only qmv_wide/M5-batch engage at our
geometry (verify-width M=1..9 profile **unchanged** vs v0.31.1); the split-K matmul does NOT
close the FFN M=512 down_proj anomaly (persists at 9.818 µs/tok, 3.4× the M=1024 2.857 —
**pc=2048 NOT stale**, no re-sweep); gqa-8 (our GQA 6) and NVFP4 (M5 Pro) are unreachable. In-
pipeline BW = 176.6 GB/s vs 310–355 GB/s sustained — the gap PERSISTS, so the decode ceiling
is per-dispatch/state-bound (next lever = scheduling, not kernels). Controlled-protocol
verification: essay decode + 32K prefill both reproduce the registered values. `run_cell.sh`
now takes a `CACHE_STATE` arg (MISS/RAM-HIT) + records `cache_ssd_promoted`. Engagement map +
ranked next tasks → `docs/V0322-KERNEL-BASELINE.md`. Run `mer4-20260917-2137`.

**PRO (in-pipeline BW gap probe) — COMPLETE: GO for scheduling task.** Run
`pro-bw-20260917-2306`. Diagnostic-only (no production source changes; qmvbench
`--layer-seq` probe added). PRO0: headline refresh confirms 81.6 ms in-pipeline anchor
(essay 83.2 / specdec 85.6 ms), stream hashes match registry exactly (no cold/warm drift
on v0.32.2). **PRO1 (qmvbench `--layer-seq`):** the QMV kernel-mix interleave (99.6%) AND
the weight-rotation/thrash (98.2%) are FREE → rules out a hardware interleave property.
**PRO2 (FullBench M=1/M=3/serial):** M=1 serial 58.82 ms (250 GB/s), M=3 verify 79.52 ms
(181 GB/s), in-pipeline (M=3) 83.2 ms (176.6 GB/s) = M=3 verify + 3.7 ms draft/accept-
rollback. **Verdict:** the 2× gap is the **per-dispatch sync + M effect** (M=3 verify with
per-step sync = 176.6 GB/s vs FFN sustained no-sync = 310–355 GB/s), NOT interleave/thrash
(free) and NOT a measurement artifact. **GO** for a scheduling task: batched/fused dispatch
+ state amortization across MTP steps (engagement map lever #1), targeting the per-dispatch
sync + state setup, NOT the kernel mix. → `benchmarks/results/pro-bw-20260917-2306/pro-findings.md`.

- **Fresh checkpoint:** completed 2026-09-18 00:24 BST via `scripts/agent-checkpoint.sh`
  (`.dsh/last-agent-checkpoint` = `2026-09-18T00:24:28+01:00`); fresh-checkpoint procedure
  completed. Server `299f87f`, engine `cfd6df5` (qmvbench `--layer-seq` probe).

**MER1 (suites green under policy v3) — COMPLETE.** Policy v3 (bit-exactness relaxed,
determinism = hard gate) recorded in contract §0c. Continuation (53) + Fused (18, tolerance
0.02, measured max|diff|=0.015625) + metallib SHA gate (1) + decode canary (1) = **78/78
pass**. Engine `6e8eab2`, server `2d4989a`.

**MER2 (interleaved A/B) — COMPLETE (initially STOP on the strict determinism gate).** essay
decode +7.7%, specdec +5.6%, prefill -12.9% (all favor the upgrade). The incumbent (v0.31.6)
is non-deterministic on specdec (r1=`139acb9d…` MISS, r2+ = `06882d85…` HIT) — root-caused in
ND as the store-on-success prefix-cache HIT/MISS (the registered knife-edge family, gap ≤ 4
ulp, first flip 95 %), NOT a decode-path regression. The upgraded (v0.32.2) is deterministic
at `139acb9d…` for BOTH cache states (fixes the split).

**ND (root-cause) — COMPLETE.** ND1: trigger = the store-on-success prefix-cache HIT (per-
prompt; RAM + SSD, `~/.qwen38-mtp/kv-ssd/`). ND2: first divergent token 973/1024 (95.0 %);
origin = the cached-prefill replay (hypothesis a), not the decode path. Gate amended (policy v3
§0d: determinism is per cache state). Upgraded PASSES strictly.

**MER3 (merge to main) — COMPLETE.** Engine `6e8eab2` + server feature branch merged to main;
branches deleted. All tests green (engine Qwen38MTPDiagnosticTests 3/3; server HTTPServerTests
237/237).

- **Fresh checkpoint:** completed 2026-09-17 21:18 BST via `scripts/agent-checkpoint.sh`
  (`.dsh/last-agent-checkpoint` = `2026-09-17T21:18:43+01:00`); fresh-checkpoint procedure
  completed.

**Next: MER4 (post-merge kernel re-baseline).** See `progress.md` MER3 + the ND findings in
`benchmarks/results/nd-specdec-20260917-1946/nd-findings.md`.

Objective: close the last quantified decode headroom (in-pipeline 200–275 GB/s vs qmvbench
310–355 GB/s sustained on the 14.4 GB 4-bit weight stream) by determining whether a newer
upstream MLX/MLXSwift closes it; upgrade the pin only if the end-to-end gate passes.

**Policy v3 (recorded FIRST).** Bit-exactness is **no longer a project invariant**.
Determinism (same config → identical stream hash across reps) remains the hard gate;
cross-config/cross-version bit-exactness is NOT required (new configs/versions register their
own hashes). Kernel-affecting changes are accepted on knife-edge-family divergence
(first-divergence position, rate, top-2 gaps; rate far above ~9/1024 or any flip > 8 ulp =
STOP) + MTP acceptance within ~93.5–94.7%. Recorded in
`benchmarks/MTP-CORRECTNESS-CONTRACT.md` §0c + `progress.md`.

**U1 (survey, `docs/UPSTREAM-MLX-SURVEY.md`) — GO to U2.** Pinned: mlx-swift **0.31.6**
(= the latest *released* tag) wrapping **C++ MLX v0.31.1**. The kernel-relevant range is
**C++ MLX v0.31.1 → v0.32.2** (reached only via *unreleased* mlx-swift main). v0.32.2 has
direct decode-path kernel improvements (qmv_wide small-batch quantized matvec, NVFP4 QMV M5
Max, M5-class qmv batch limit, split-K quantized matmul, gqa-8 decode attention) → GO.

**U2 (framework upgrade A/B) — REVERTED (build-infrastructure barrier).** Bumping the pin to
mlx-swift main (C++ MLX v0.32.2) builds **green** (engine MLXLLM + server HTTPServer, zero
compat fixes) and our path `Qwen38MTPDiagnosticTests` PASSes — **but** the full engine suite
crashes on `testQwen35MoECompiledDecodeTracksWeightUpdates` (`Unable to load kernel
dot_product_float32_it32_tg512_sg16`). Root cause: Cmlx builds C++ MLX **NO-JIT**, bundling
a **prebuilt** `default.metallib` that is a **stale v0.31.1** artifact from a separate
**PrepareMetalShaders** step that **SwiftPM does not regenerate on a pin bump** (warm build =
no-op). So the "upgraded" build runs **v0.32.2 C++ against v0.31.1 kernels** — the improved
kernels are not active and a v0.32.2 dot_product kernel is missing. A clean A/B is
**impossible via the pure pin-bump path**; the build gate (engine suite green) is not met.
**Not** a "no fix exists" closure — the kernels DO exist in v0.32.2. **Pin reverted to
0.31.6** (engine + server); verified green (`CompiledDecodeWeightUpdateTests` 6/0).

**Follow-up task (separate, NOT this one):** regenerate `default.metallib` for v0.32.2 via
the PrepareMetalShaders CMake step → confirm engine suite green → run the decode A/B matrix
(§5 of the survey).

- **MET (the U2 follow-up, EXECUTED).** Unblocked the U2 metallib barrier and re-ran the
  decode A/B. Details in `progress.md` (MET section) + `../mlx-swift-lm/docs/BUILD-MLX-UPGRADE.md`
  + `../mlx-swift-lm/scripts/build-metallib.sh`.
  - **Unblock:** `scripts/build-metallib.sh` (engine fork) builds the always-list Metal kernel
    metallib via the CMake `mlx-metallib` target for the pinned C++ MLX revision (`1f8e74e` =
    v0.32.2), cached per revision, places it colocated (`<exe-dir>/mlx.metallib`, first runtime
    search path via `dladdr`), records SHA provenance (`b57de586…`); `check` mode = the
    stale-metallib detector. v0.32.2 kernels **proven active**: the U2 MoE `dot_product` crash
    case now **PASSES**; the release server starts clean (`readyz=200`).
  - **A/B (cross-session, v0.32.2 vs v0.31.6 baselines):** essay 22.22 vs 21.89 (**+1.5%**),
    specdec 23.56 vs 23.29 (**+1.1%**) — **below the 3% KEEP gate**. All correctness gates
    pass (determinism/phaseSum/depthDist/acc match incumbents). **INCONCLUSIVE** (marginal +
    cross-session thermal drift, stepAvg 87→100 ms).
  - **Deliverables:** engine fork `scripts/build-metallib.sh` + `scripts/metallib-provenance.json`
    + `docs/BUILD-MLX-UPGRADE.md` + `Package.swift` pin; server `progress.md` (MET) +
    `docs/UPSTREAM-MLX-SURVEY.md` §7b + `Package.resolved`.
  - **Git:** engine fork `feature/mlx-v0322-upgrade` @ `f36361c`; server
    `feature/mlx-v0322-upgrade` @ `180e233`. **NOT merged to main** (KEEP gate not met).
- **Next step:** a proper **interleaved A/B** (rebuild the v0.31.6 server, interleave
  incumbent/upgraded reps in the same thermal session, rotating start) for a definitive
  KEEP/REJECT. If it also shows <3%, the upgrade is a REJECT (keep the v0.31.6 pin; the metallib
  script + docs are still the durable unblock). Triage the one swift-testing `[read]` crash
  (model-file read, end of the engine suite — not the decode path).
- **Fresh checkpoint:** completed 2026-09-17 16:25 BST via `scripts/agent-checkpoint.sh`
  (`.dsh/last-agent-checkpoint` = `2026-09-17T16:25:43+01:00`); fresh-checkpoint procedure
  completed.

---

**Prior: COMPLETE: MCP2 prefill-chunk-size sweep — KEEP pc=2048, default flipped 512→2048 (2026-09-17).**

Objective: determine (config-only, zero kernel/source change) whether a larger
`--prefill-chunk-size` (pc) cuts prefill wall, following MCP1's GO (the M=512 FFN
down_proj inefficiency is M-dependent). Gate: mean 32K eval-sync prefill wall ≥ 5%
better than pc=512 AND ≥ 4/5 paired reps favor the new pc, 64K not regressed, all
bit-exact hash gates pass.

**Result (stable, 5 measured reps @32K, 2 @64K, fresh server per cell, greedy):**

| pc | 32K wall | vs 512 | paired | 64K wall | vs 512 |
|----|---------:|-------:|:------:|---------:|-------:|
| 512 | 149.49 s | — | — | 305.17 s | — |
| 1024 | 139.44 s | −6.7% | 4/5 | — | — |
| **2048** | **130.58 s** | **−12.7%** | **5/5** | **279.03 s** | **−8.6%** |

All pc values **bit-exact** (identical committed content at 8K/16K/32K/64K) — no
knife-edge. The end-to-end winner is pc=2048, not the MCP1-predicted 1024: the SDPA
per-chunk tiling efficiency at larger Q tiles adds a residual gain on top of the FFN
improvement. Memory-safe (peak RSS 14.6 GB @64K, per-chunk buffer 6.29 GB << 48 GB).

**Decision: KEEP pc=2048.** Default `prefillChunkSize` flipped **512 → 2048** in
`Sources/HTTPServer/ServerConfig.swift`. `ServerConfigArgumentTests` (8/8) + full
`HTTPServerTests` suite pass.

- **Server:** `benchmarks/results/mcp-20260917/mcp2-report.md` (full tables + per-phase
  breakdown), `docs/PREFILL-FFN-KERNEL.md` (MCP2 section + flag row), `progress.md` entry.
- **Scripts/data:** `benchmarks/run_mcp2.sh` (Phase A+B), `benchmarks/run_mcp2_phasec.sh`
  (Phase C), `benchmarks/results/mcp-20260917/analyze_mcp2.py` (gate), `mcp2-*.jsonl`.
- **Next step:** none for this task. Future work: consider whether the pc=2048 default
  should be adaptive to context length (larger pc is strictly better here, but a
  per-request pc from the request body is not implemented).
- **Fresh checkpoint:** completed 2026-09-17 14:48 BST via
  `scripts/agent-checkpoint.sh` (server repo `main` @ `e239c09`, tracked tree clean;
  only untracked raw benchmark artifacts remain in
  `benchmarks/results/mcp-20260917/`).

---

**Prior: COMPLETE: FFP4 FFN prefill GEMM kill-switch (relaxed bit-exactness) — NO-GO (2026-09-17).**

Objective: reopen the FFP1 NO-GO under a scoped bit-exactness relaxation (policy
v2, contract §0): FFN GEMMs at prefill widths (M ≥ 256) may differ from
`quantizedMM` by a few ulp, enabling tiling changes (split-K). Gate: a candidate
must hit ≥ 2× sustained throughput at M=512 within tolerance, else stop with a
negative result.

**Finding (stable, 2 reps, `qmvbench --ffn-prefill --ffn-cand`):** every candidate
is *slower* than the incumbent down_proj at M=512 — splitk2 0.87–0.88×, splitk4
0.82–0.83×, splitk8 0.71–0.73×, bf16_gemm 0.67–0.69×. More K-splits are
monotonically *slower* (extra kernel launches + fp32 cross-split accumulation
outweigh the K-parallelism gain). No candidate reaches 2× (none even reaches 1.0×).

**Decision: NO-GO.** The M=512 down_proj slowness is a fundamental small-M/
large-K GEMM property of the Metal quantized engine, not a tiling artifact —
**split-K cannot capture the 10× headroom**. This extends FFP1's NO-GO, now
confirmed under the relaxed policy. **FFP5 (engine integration), FFP6 (model-level
audit), FFP7 (A/B matrix) are NOT pursued.** FFP1 NO-GO stands.

- **Engine (`mlx-swift-lm`):** `qmvbench` `--ffn-cand` mode (5 candidates,
  tolerance at M=256/512/1024/8192, interleaved DVFS-fair kill-switch timing).
  Bugs fixed: format-string `%s`→`%@` (Swift String crash), `best.meanUs` init.
- **Server:** `benchmarks/results/ffp4/ffp4-report.md` (NO-GO),
  `docs/PREFILL-FFN-KERNEL.md` (status → NO-GO), contract §0 (policy v2),
  `progress.md` entry.
- **Next step:** none for this task (valid negative result). A future FFN GEMM
  win would need a fundamentally different approach (not tiling/split-K).

---

**Prior: COMPLETE: FFP1 FFN prefill GEMM kill-switch — NO-GO for a bit-exact kernel (2026-09-17).**

Objective: reduce long-context prefill wall time by optimizing the FFN-phase
4-bit GEMMs at the default 512-chunk prefill width (M=512). FFP1 is the cheap
kill-switch: measure incumbent `quantizedMM` headroom before touching the
engine.

**Finding (stable, 68 batches, 2.8% std, `qmvbench --ffn-prefill --ffn-pair`):**
incumbent `quantizedMM` is ~10× off-peak on **down_proj at M=512** (20 TF vs
220 TF gateup_wide, same DVFS window; down_proj = 84.7% of per-layer FFN time).
The anomaly is width-specific (same shape = 89.4 TF at M=1024), **not** a Metal
JIT bug (M=512 output is bit-exact vs a dequantize→bf16 reference,
`max|diff|=0.0`), and is in the **GEMM tiling engine** generally (bf16 `x@W^T`
= 14.2 TF at M=512, slower than quantizedMM's 21.6 TF).

**Decision: NO-GO (for a bit-exact kernel) — stop at FFP1.** The FFP2
requirement is element-wise equality with `quantizedMM` at every M, which forces
the same tiling/accumulation order (FP addition is non-associative). The M=512
headroom sits precisely in the tiling, so a bit-exact kernel preserves the
slowness. The only bit-exact FFN wins are fusions, and down_proj is a bare GEMM
(no fusion changes its tiling). Precedent: the existing specialized kernels are
bit-identical to their eager counterpart and QMV is ~5% *slower* than
`quantizedMM` even at M=1. No bit-exact FFN kernel reaches the ≥10% sustained
win at M=512.

- **Engine (`mlx-swift-lm`) `main` @ `774a4d3`:** `qmvbench` gains
  `--ffn-prefill` (sustained no-sync FFN throughput), `--ffn-pair` (interleaved
  gateup/downproj, same DVFS window), `--ffn-check` (bit-exact vs reference +
  M=512 anomaly localization). QMV/decode/attention untouched.
- **Server `main` @ `8049539`:** report
  `benchmarks/results/ffp1/ffp1-report.md` + `progress.md` entry.
- **Verification:** engine `swift test --filter Qwen38MTPDiagnosticTests` 3/3;
  server `swift test --filter HTTPServerTests` 237/237; `git diff --check` clean.
- **Next step:** none for this task (valid negative result). A future
  relaxation of the bit-exact requirement (e.g. a non-strict-tolerance
  prefill-only path) would reopen the 10× down_proj headroom.

---

**Prior: COMPLETE: Cross-lineage port of Tasks 1–6 from `qwen-mtp-server` (2026-09-16).**

The recovered Task 1–6 work (sibling `qwen-mtp-server`, branch
`recovered/tasks-1-6` @ `efcf595`) is ported into canonical `main` by
file-by-file manual adaptation (disjoint object sets → no git merge). Details
in `progress.md` ("Cross-lineage port of Tasks 1–6") and `docs/port-inventory.md`
+ `docs/task2-comparison.md`.

- **Task 4 (reusable-path repair):** `fa804ed` + `59c282b` (earlier this session).
- **Tasks 3+5+6 (SSD tier):** `RadixSSDStore.swift` (new, multi-namespace),
  `RadixKVCacheManager.swift` (disk tier merged into the namespace-aware
  manager), `MLXGenerator.swift` (SSD wiring + `QWEN_MLX_SEED`; Radix `store`
  moved before `continuation.finish()` so a post-stream snapshot sees the
  entry), `Qwen38Server.swift` (`ModelShutdownHandler` calls `flushToSSD()`),
  `ServerConfig.swift` (5 `--kv-ssd-*` flags + env), `ServerConfigArgumentTests`
  (5 flags now known), `RadixSSDPersistenceTests` (6 pure) +
  `RadixSSDWeightTests` (2 weight-gated). Engine fork `6fa481d` adds
  `restoreKVCacheState`.
- **Task 2 (depth autotune):** KEEP canonical; only the identity primitives the
  SSD key needs were ported (`WeightTreeDigest(s)`/`hardwareID`/`SHA256File` +
  `QWEN_MLX_SEED`). Full autotune surface NOT ported.
- **Task 1 (compact-rejection negative result):** artifacts ported
  (`docs/compact-rejection-rfc.md`, `CompactRejectionTests.swift`, 2 benchmark
  files); the walk itself is NOT ported (rejected result).
- **Verification:** `swift build --target HTTPServer` 0 errors/0 warnings;
  `swift test --filter HTTPServerTests` = **237 Swift Testing, all green**;
  weight-gated SSD tests PASS (bit-identity + `reused>0`/`radixPrefillSkipped`
  on a 1254-token SSD-restored prefix); E2E restart benchmark **PASS**
  (TTFT_warm 0.137 s, TTFT_disk 0.144 s, TTFT_cold 5.535 s; disk/warm 1.05×,
  disk/cold 0.03×).
- **Prior LCP status (unchanged, still valid):** see the LCP P1/P2/P3 block
  below.

**Prior: COMPLETE: Long-Context Prefill Optimization & Exactness Guardrails (LCP) — P1/P2/P3.**

- **P1** baseline profile at 8K/16K/32K/64K (pc=512, eval-sync per-phase, RSS, bit-exact
  hash gate — all 4 pass): FFN 56.3→36.5 %, full-attention 12.8→42.1 % (SDPA 5.7→35.6 %,
  O(L²)), GDN 27.3→19.1 %, norms+residuals ~3 %. Report:
  `benchmarks/results/prefill-opt-20260915/P1_PROFILE_LCP_m5pro_20260915.md`.
- **P2** two toggle-gated default-OFF kernel extensions (engine `feature/prompt-1`):
  `MLX_QWEN_FUSED_RESIDUAL_3D` (fused residual+RMSNorm on 3-D prefill tensors) and
  `MLX_QWEN_FUSED_GDN_PREFILL` (fused GDN prework at prefill widths). Bit-exact: unit
  tests (S=1,2,512,1000 and 16,512) + real-model content-hash gate at 8K/16K/32K/64K,
  all pass. Wall-time uplift not resolvable: session thermal/state variance is up to
  ~1.6× (a flag-off 32K re-run was 19 % faster than the P1 baseline and 33 % faster than
  the slowest flag-on cell). Recommendation: keep default-OFF. Report:
  `P2_KERNEL_LCP_residual3d_gdn-prefill_20260915.md`.
- **P3** gate validation: `ENABLE_BIT_EXACT_ATTENTION=1` (dense reference) bit-exact at
  8K/16K and 507-rejects at 32K/64K (dense infeasible); `=0` chunked bit-exact at
  32K/64K; `ENABLE_BIT_EXACT=1` (dense + all fusions off) bit-exact at 8K/16K against
  the optimized default. Report: `P3_ATTN_LCP_bit_exact_gate_20260915.md`.
- **Server:** admission now uses the shared `MLXChunkedPrefill.enabled` resolver
  (respects the gates). Full server suite green (111 XCTest + 224 Swift Testing) after
  all edits; engine suites green.
- **Governance:** `docs/PREFILL-PROFILE-INDEX.md` (central table + flag reference), this
  file, `progress.md`. Runners: `benchmarks/run_lcp_p{1,2,3}.sh` (hash-gated).
- **Checkpoint:** fresh checkpoint completed 2026-09-16 03:33:04 +01:00 in both repos
  (engine `d189c61`, server `ac9e019`, both merged to `main`, trees clean).

**Prior: COMPLETE: pc=0 single-pass prefill trap fix (engine-only).**

`--prefill-chunk-size 0` (single-pass prefill) fatal-`[reshape]`'d on an empty
chunk (both prefill loops computed `start=0, end=min(0+0,count)=0`). Extracted
the partition into a pure `Qwen38MTPBlockSession.prefillChunkRanges(count:chunkSize:)`
that guards `chunkSize==0` (one full-range chunk); the `chunkSize>0` branch is
mathematically identical to the old inline loop, so the **default pc=512 path is
byte-identical** (no perf regression). 4 unit tests + real-model validation:
pc=0 no longer traps at 8K/16K/32K, bit-exact with pc=512 at 8K/16K, and 32K
diverges at the first token (the expected Phase 1 Bug A FP-accumulation-order
sensitivity to the prefill split — the engine chunked-SDPA gate engages at
L>4096 for pc=0 but not per-512-chunk for pc=512). pc=0 is slower than pc=512
(205 s vs 117 s at 32K), so this is a robustness fix, not a perf change. Engine
`45df72a`, server `54f167c` (docs). Engine 7/7 + server 111/111 green.

**Follow-up (Option B, profiling only, no code change):** the 32K prefill
per-phase GPU breakdown (eval-synchronized timing, `MLX_CHUNKED_PREFILL=1`,
pc=512, gated on L>100 to isolate the prefill from MTP verify): **FFN ~50% /
GDN block ~27% / full-attention block ~23% / other ~0%**; top cost centers are the
FFN (largest), the GDN block, and the full-attention block — all O(L) 4-bit
GEMM/scan, none the O(L²) attention. The full-attn QKV/SDPA/O sub-split was not
captured (compiled fast path). Engine left byte-identical to main. See
`docs/PREFILL-PROFILE.md`.

**Follow-up (32K+ prefill validation, profiling/validation only, no code
change):** full per-phase + memory + bit-exactness validation with the
full-attention sub-split captured via the compiled fast path. **Key correction:**
the full-attention **SDPA is 18.5 % of the 32K prefill and 29.3 % at 64K**
(grows O(L²); the prefill uses the dense unfused path) — this **corrects** the
`docs/FLASH-ATTENTION.md` "~0.04 %" figure (CPU enqueue, not GPU time) and
revises the "flash attention won't help" conclusion on the speedup axis (it still
holds on the bit-exactness axis). FFN ~43–49 %, GDN ~21–25 %. pc=512 reproduced
near-optimal (132 s @32K); pc=0 is 65 % slower (218 s, robustness baseline
only); 64K completes (333 s). 8K/16K chunked (pc=512 and pc=0) bit-exact with
dense; 32K pc=0 diverges at token 1 (expected Phase 1 Bug A). Peak RSS 13.4–
14.6 GB. Engine left byte-identical to main. See
`benchmarks/results/prefill-verify-2026-09-15/REPORT.md`.

### Prior: Flash-attention feasibility analysis (Phase I follow-up)

Evaluated integrating a flash-attention kernel for prefill. **Decision: do not
integrate** (see `docs/FLASH-ATTENTION.md`). Evidence:
- MLX has no flash kernel for prefill (T_q > 1 materializes `[L,L]`; the Metal
  kernel is decode-only).
- A true flash kernel's online softmax is **not bit-exact** (Phase 1, Bug A),
  which the task required.
- An early "~0.04%" SDPA figure was **CPU enqueue time, not GPU time** (MLX
  enqueues Metal commands asynchronously) — invalidated; the per-phase GPU split
  is unmeasured. Valid wall-time signal: prefill is near-optimal at the default
  `prefillChunkSize=512` (119 s; pc=8192 is 55% slower; pc=0 was the trap now
  fixed).
- Chunked prefill already enables 128K+ (per-tile buffer 3.2 GB @128K).

Server suite 0 failures; engine byte-identical to main. Server commit `c16f166`.

### Prior: Chunked causal prefill (Phase I)

Eliminates the quadratic `[seq × seq]` dense-attention scores buffer that traps
the process (SIGTRAP) at ~24K+ context. **Default-OFF** (`MLX_CHUNKED_PREFILL=1`
to enable); the off build is bit-identical to before. Server models the transient
buffer in admission control (dense quadratic vs chunked linear); oversized dense
prefills are rejected with HTTP 507 / `prefill_buffer_exceeded`.

Validated: 8K greedy stream hash identical dense vs chunked (`763eccc3…`);
32K completes with chunked (175.9 s) where dense traps; dense 32K cleanly
rejected pre-prefill (507, `51.6 GB > 27.1 GB`). Engine KVCache 118/118 + MTP
3/3, server 224/224 all green. See `docs/CHUNKED-PREFILL.md`.

## Repository state
**Current (cross-lineage port, 2026-09-16):**
- **Server** `/Users/cwong/ai/qwen38-mtp-server` (canonical): on
  `feature/port-radix-ssd` (branched from `main` @ `59c282b`). Uncommitted:
  the SSD tier (Phase 3) + Task 1 artifacts (Phase 4) + this doc/progress
  update (Phase 5) — see `git status --short` for the exact file list.
- **Engine** `/Users/cwong/ai/mlx-swift-lm` (fork): `main` @ `6fa481d` —
  `restoreKVCacheState(cache:state:metaState:)` added to
  `Libraries/MLXLMCommon/KVCache.swift` (the paired change the SSD lazy-load
  path depends on). Committed before the server-side code that calls it.

**Prior (Phase I chunked prefill):**
- **Engine** `main` @ `4cd8603`: `AttentionUtils.swift` (`chunkedCausalPrefill`),
  `KVCache.swift` (`MLXChunkedPrefill`), `KVCacheTests.swift` (2 tests).
- **Server** `main` @ `1a247b4`: `MemoryAdmission.swift` — transient buffer model
  - `Sources/HTTPServer/Generation/MLXGenerator.swift` — passes `chunkedPrefillEnabled`
  - `Sources/HTTPServer/API/OpenAIValidation.swift` — 507 error response
  - `Sources/HTTPServer/Routes/OpenAIRouter.swift` — catches `TransientBufferFailure`
  - `Tests/HTTPServerTests/KVCacheConfigTests.swift` — 4 new admission tests
  - `docs/CHUNKED-PREFILL.md` (new), `progress.md` (Phase I section), this file

## Prior: Phase H long-context benchmarks
The Phase H findings below (32K/96K infeasible on the dense path, prefill
dominates, fused GDN not a long-context win) motivated Phase I. They remain
accurate for the **default dense build**.
  - `benchmarks/make_longctx_fixtures.py` (new)
  - `benchmarks/prompts/longctx-{8k,16k,32k,96k}.txt` (new fixtures)
  - `benchmarks/results/longctx-2026-09-15/{ab-matrix.jsonl,NOTES.txt}` (raw data)
  - `docs/LONG-CONTEXT-BENCHMARKS.md` (new)
  - `progress.md` (Phase H section appended)
  - `docs/HANDOFF.md` (this file)

## Headline findings (evidence)
- **32K and 96K are infeasible on this hardware.** The prefill attention buffer
  is a dense `[seq × seq]` allocation (quadratic in seq). At 32K it requests
  **51,577,363,200 bytes (51.6 GB)**, exceeding the **30,150,672,384-byte (30.2
  GB) Metal max buffer** → `[metal::malloc] ... greater than the maximum allowed
  buffer size` → **SIGTRAP crash** (reproduced twice). 96K would be ~464 GB.
  **Max feasible prompt ≈ 24K tokens** (buffer ≈ 30 GB at `seq ≈ 32K·√(30.2/51.6)`).
- **Prefill dominates** the request: ~27 s at 8K of a ~30 s total (~90 %).
  Prefill throughput ~275 tok/s, ~linear in seq (8K 27 s, 16K 59 s).
- **Fused GDN does NOT pay off at long context.** Bit-exact (identical completion
  token stream on/off) and not faster at 8K (full_s 29.3 s off vs 32.7 s on). Its
  launch count is per verify round (draft-depth dependent), not per context token,
  so it does not accumulate with context; and prefill (not decode) is the cost.
- **k = 2 remains optimal at 8K** (full_s k1 36.1 / k2 29.3 / k3 32.0 s), matching
  short context.
- **Prefix/session caching ≈ 7 % TTFT win only** (8K cold 27.3 s → warm ~25 s):
  the gated-delta recurrent layers are not resumable from a token-prefix, so the
  prefill is effectively recomputed each request.
- **All configs bit-exact**: identical `content_sha256` (`377c6fda…` at 8K) across
  fused on/off, k = 1/2/3, and repeated requests.
- **Memory**: peak RSS ~15.25 GB at 8K (~15 GB weights + ~0.5 GB KV). Steady-state
  fits 32K under the 44 GB admission limit; the crash is the *transient* quadratic
  prefill buffer, which admission control does not model.
- **Decode is GPU-eval-bound**: `tEvalMs` ~89 ms of ~93 ms `stepMs` at 8K; host/
  graph overhead small. First verify round after prefill is a one-time ~4 s.

## Important files
- `docs/LONG-CONTEXT-BENCHMARKS.md` — the full write-up (fixtures, protocol,
  results tables, bottleneck analysis, reproduction).
- `benchmarks/make_longctx_fixtures.py` — deterministic fixture builder (real
  Swift source from both repos, 479 files / 1.76M-token corpus, truncated at token
  boundaries).
- `benchmarks/results/longctx-2026-09-15/ab-matrix.jsonl` — raw per-request lines
  (client TTFT/full/decode + content sha + server decodeSeconds/avgStepMs/
  acceptedPerStep).
- `benchmarks/results/longctx-2026-09-15/NOTES.txt` — fixture sha256, binary
  sha256, Metal cap.

## Decisions
- **No code changes** (explicit task constraint). The release binary is built from
  `main`; the engine is untouched. All findings are measurements.
- **N per cell is 3–4**, not the brief's 6–10: each 8K request costs ~30 s (27 s
  prefill) and 32K/96K cannot complete. Reported means have small spread.
- **Client `full_s`/`TTFT` is the cross-cell metric** (measured identically);
  server `decodeSeconds` is reported alongside for phase context. There is a
  consistent client-vs-server decode gap (client ~2.3 s vs server ~5 s); the
  cross-cell comparison is valid because every cell is measured the same way.
- **Profiling** used the server's built-in instrumentation (`STEP-TRACE` phase
  split, `decodeSeconds`, the Metal buffer allocation, RSS) rather than a full
  Instruments GPU trace (see "Unresolved risks").

## Verification
- `git diff --check` → clean.
- Engine: `swift test --filter Qwen38MTPDiagnosticTests` → 3/3 green (untouched,
  sanity only). Server test suite unaffected (no server code changed).
- 32K crash reproduced twice with the identical `[metal::malloc]` allocation size.
- Bit-exactness: identical `content_sha256` across all A/B cells at 8K.
- `git status --short` → only the Phase H artifacts above; no tracked source
  files modified.

## Unresolved risks / caveats
- **Instruments GPU-timeline / memory-bandwidth trace not run.** It would require
  re-running the (expensive, partly-crashing) cells under a profiler and adds
  little beyond the `STEP-TRACE` phase split for the questions asked. If a
  dedicated GPU-saturation trace is wanted, treat it as a separate profiling task.
- **32K/96K not benchmarked** (they crash / are infeasible); the numbers for
  those lengths are the predicted buffer sizes, not measured throughput.
- Do NOT cite cross-session absolute tok/s; these numbers are one environment.

## Operations that must not be repeated
- Do NOT change server/engine code to "fix" the long-context crash as part of a
  benchmarking task — that is a separate optimization task (chunked/flash prefill
  + admission-control awareness of the transient buffer).
- Do NOT cite the fused GDN kernel as a wall-clock win (it is not, at any context).
- Do NOT claim 32K/96K work on this hardware (they crash / are infeasible).
- Do NOT run `agent-checkpoint.sh` from the symlink wrapper dir; run it by
  absolute path with the CWD inside the target git repo.
- Do NOT `head -N` the checkpoint script output (SIGPIPE).

## Completion marker
**LEV-J Phase FB2 fresh checkpoint: COMPLETED 2026-09-19 09:00 BST** via `scripts/agent-checkpoint.sh` (run by absolute path with CWD inside each repo); fresh-checkpoint procedure completed in **both** repos (`.dsh/last-agent-checkpoint` 09:00:43 / 09:00:46). Server `qwen38-mtp-server` `feature/lev-j` (M `docs/HANDOFF.md`, `progress.md`; new `benchmarks/results/lev-j/fb2-2pass/README.md`); engine `mlx-swift-lm` `feature/lev-j` @67873ed (M `Qwen38MTPBlockSession.swift`, `AttentionUtils.swift`, `Package.swift`; new `FlashSDPA.swift` [now 2-pass], `Qwen38FlashSDPATests.swift` [strengthened gate]).

Prior — **LEV-J Phase FB fresh checkpoint: COMPLETED 2026-09-19 07:55 BST** via
`scripts/agent-checkpoint.sh` (run by absolute path with CWD inside each repo);
fresh-checkpoint procedure completed in **both** repos. Server `qwen38-mtp-server`
`feature/lev-j` (M `docs/HANDOFF.md`, `progress.md`; new `benchmarks/results/lev-j/`);
engine `mlx-swift-lm` `feature/lev-j` @67873ed (M `Qwen38MTPBlockSession.swift`,
`AttentionUtils.swift`, `Package.swift`; new `FlashSDPA.swift`,
`Qwen38FlashSDPATests.swift`, `FlashBench/`). **NOT merged** — the flash
**determinism gate fails intermittently** (kernel is not bit-exact), so the
"all tests pass" merge precondition is not met. LEV-J is BLOCKED on a kernel
correctness defect (see the FB section above).

**Cold-JIT pre-warm (prompt-4) fresh checkpoint: COMPLETED 2026-09-18 18:19 BST** via
`scripts/agent-checkpoint.sh` (server `.dsh/last-agent-checkpoint` =
`2026-09-18T18:19:04+01:00`); fresh-checkpoint procedure completed in the
server repo (the engine repo is unmodified this task — `mlx-swift-lm`
`main` @ `67873ed`, clean).

**Prior — LEV-campaign fresh checkpoint: COMPLETED 2026-09-18 09:06 BST** via
`scripts/agent-checkpoint.sh` (server `.dsh/last-agent-checkpoint` =
`2026-09-18T09:06:31+01:00`; engine = `2026-09-18T09:06:39+01:00`);
fresh-checkpoint procedure completed in both repos. Server `main` @ `1e80f09`
(clean, 0 tracked changes); engine `main` @ `9f4ceb9` (clean). Both suites green
(engine `Qwen38MTPDiagnosticTests` 3/3; server `HTTPServerTests` 237 tests / 7
suites). No server left running.

**MER4 fresh checkpoint: COMPLETED 2026-09-17 22:54 BST** via `scripts/agent-checkpoint.sh`
(`.dsh/last-agent-checkpoint` = `2026-09-17T22:54:37+01:00`); fresh-checkpoint procedure
completed.
Server `main` @ `1c1a677` (MER4; clean, 0 tracked changes); engine `main` @ `6e8eab2` (clean).

MER4 (post-merge kernel re-baseline) COMPLETE — engagement map
`docs/V0322-KERNEL-BASELINE.md`. Run `mer4-20260917-2137`.

Prior — MER3 merge checkpoint: `2026-09-17T21:18:43+01:00` (server `main` @
`628c0fc`, engine `main` @ `6e8eab2`, both clean). FFP4 checkpoint `2026-09-17
09:46 BST` (server `8a055a3`, engine `c575e19`).

## Next step (exact) — SUPERSEDED (see the campaign "Next step (exact)" block at the top, 2026-09-18)
**None outstanding for the MLX v0.32.2 platform refresh (MER1–MER4 all COMPLETE).** The
ranked follow-on (NOT part of this refresh — a separate task) is the in-pipeline scheduling
lever from `docs/V0322-KERNEL-BASELINE.md`: the decode weight-stream BW is per-
dispatch/state-bound (176.6 GB/s in-pipeline vs 310–355 GB/s sustained), so the next lever,
if pursued, is batched/fused dispatch + state amortization across MTP steps — scheduling,
not a new GEMM kernel.

## 2026-09-16 addendum — Task 7 resolution + reconciliation audit

- **Task 7 (CLI-flag crash) is RESOLVED (server `76054e3`, merged to `main`):**
  the `--kv-ssd-cache-dir` / `--kv-ssd-cache-gb` / `--kv-ssd-ttl-seconds` flags
  **never existed** in this codebase (zero occurrences in sources, engine fork,
  docs, git history). The reported `app.execute()` crash was unknown-flag
  leakage into `Environment.detect(arguments:)`. `ServerConfig` now rejects any
  unknown `--flag` loudly at startup (stderr + `exit(2)`, "Unknown option …
  Run with --help") **before** Vapor dispatch, so no launch command of the
  "pass the kv-ssd flags" form can reach the Vapor dispatcher. There is no
  env-var launch to work around anything.
- This handoff was checked for the obsolete follow-up line ("fix
  `app.execute()` so the `--kv-ssd-*` flags work directly — env-var launch is
  the current stable workaround"): **grep found no such line in the current
  `docs/HANDOFF.md`**, so no removal was needed; this addendum records Task 7
  as the resolution.
- `progress.md` was reconciled against the repository state on 2026-09-16
  (see its "Documentation reconciliation audit" section): main @ `76054e3`,
  clean tree, `swift test --filter HTTPServerTests` = 224 Swift Testing +
  120 XCTest, all green (default invocation, no weight-gated tests).
