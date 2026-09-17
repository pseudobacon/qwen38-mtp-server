# Upstream MLX survey — decode-bandwidth headroom (U1)

**Task.** The M3 quantized GEMV/GEMM decode path runs at **200–275 GB/s** effective
bandwidth in-pipeline vs **310–355 GB/s** sustained in `qmvbench` (per-layer 205 GDN / 209 FA
GB/s; GPU util 98.2–98.6%). The gap lives in the **prebuilt MLX Metal kernels**, not this
checkout. This survey determines whether a newer upstream MLX / MLXSwift (or a tuning knob, or
an upstream fix) closes it — and drives the U2 pin-upgrade gate.

**Date:** 2026-09-17. **Method:** evidence only, no code changes. All facts verified against
`git ls-remote` / a blobless clone of both upstream repos.

---

## 1. Pinned vs latest

| Layer | Pinned (engine fork) | Latest released | Unreleased main |
|-------|----------------------|-----------------|-----------------|
| **mlx-swift** (Swift binding) | **0.31.6** (tag `0bb916c`) | **0.31.6** (= our pin) | `2bebe4e` (17 commits past 0.31.6) |
| **C++ MLX** (core, has the Metal kernels) | **v0.31.1** (submodule `ce45c52`) | v0.32.2 (reached via main) | v0.32.2 |
| **mlx-c** | v0.6.0 | — | — |

- The latest **released** `mlx-swift` tag is **0.31.6 — exactly our pin**. There is **no 0.32.x
  release and no pre-release tag** of mlx-swift (verified via `git ls-remote --tags`).
- The Metal kernels are **not** in the Swift binding; they are in the **C++ MLX** submodule. Our
  pin wraps **C++ MLX v0.31.1**. The only way to move the kernels is the C++ MLX bump, which
  upstream landed in **unreleased** mlx-swift main:
  - mlx-swift `ab924c8 update for mlx v0.32.2 (#450)` →
    https://github.com/ml-explore/mlx-swift/commit/ab924c8 — bumps the C++ MLX submodule
    **v0.31.1 → v0.32.2**.
- **Therefore the kernel-relevant range is C++ MLX v0.31.1 → v0.32.2, reachable only by pinning
  mlx-swift to an *unreleased* commit (main).** No tagged mlx-swift release carries the new
  kernels.

---

## 2. Relevant kernel changes (C++ MLX v0.31.1 → v0.32.2)

All links: `https://github.com/ml-explore/mlx/commit/<sha>`.

| SHA | Change | One-line relevance to the decode-bandwidth gap |
|-----|--------|----------------------------------------------|
| `548dd80e` | Add small-batch quantized matvec kernel (**qmv_wide**) #3764 | **Direct.** New quantized GEMV for M=1..9; dispatch `use_qmv_wide` engages affine (4-bit) mode on gen-15+ (our M5 Pro qualifies). This *is* the decode GEMV path. |
| `5a1e44c3` | Optimize large NVFP4 QMV on **M5 Max** #3961 | **Direct.** QMV tuning explicitly for M5-class GPUs (we are M5 Pro). |
| `e7838d5e` | Raise qmv batch limit for large matrices on **M5-class** GPUs #3791 | **Direct.** M5-class qmv batching bound. |
| `38ad2570` | [Metal][Perf] Add **split-K for quantized matmul (small M)** #3120 | **Direct.** Small-M quantized GEMM (decode) split-K path. |
| `fa0d4463` | Read each K/V byte once in **gqa-8 decode attention** #4077 | **Direct.** Decode attention for GQA (our 4 KV heads); cuts redundant K/V reads. |
| `8056817b` | Derive the qmv fast path K alignment from bits #3965 | Supporting. qmv fast-path alignment (quantized decode). |
| `1700b39a` | Fix fp quantized matvec for output dim < 8 #3804 | Supporting. Quantized matvec edge fix. |
| `714a7efc` | Add fused full-attention path for **head_dim 256 on NAX** #3842 | **Conditional (NAX).** Our head_dim is 256; applies only if the M5 Pro is an NAX device (runtime-detected, see §3). Also the prefill SDPA lever for the follow-up file. |
| `09ebe730` | Break monolithic MTLResidencySet into smaller sets #4211 | Indirect. Metal residency/step-time. |
| `291e909f` | Reuse Metal WAR tracking hash tables #3882 | Indirect. Metal dispatch overhead. |
| `f599c020` | Fix concurrent Metal kernel cache lookup #4043 | Indirect. Kernel-cache contention. |

The **direct** group (qmv_wide, NVFP4 QMV M5 Max, M5-class qmv batch limit, split-K quantized
matmul, gqa-8 decode attention) is squarely in the decode-GEMV/attention path where our
200–275 GB/s effective bandwidth lives. This is not a mixed/unrelated full-framework bump —
there are concrete kernel improvements in the exact path under investigation.

---

## 3. NAX applicability (runtime fact, not static)

`is_nax_available()` (C++ MLX `mlx/backend/metal/device.cpp`) requires **macOS 26.2+ AND GPU
generation ≥ 17** (`gen >= (arch=='p' ? 18 : 17)`). NAX is a property of the *physical GPU at
runtime*, not statically knowable from the model. Whether the M5 Pro is NAX:
- **Not determinable from source.** Confirmed empirically at U2 (the step-trace engagement
  summary shows which kernel family — NAX vs standard Metal — is engaged).
- The **standard** Metal-path improvements (qmv_wide, NVFP4 QMV M5 Max, M5-class qmv batch
  limit, split-K quantized matmul, gqa-8 decode attention) apply **regardless** of NAX. So the
  survey GO decision does not depend on the NAX question.
- The **NAX-only** improvements (head_dim-256 fused full attention, NAX qmm) are upside if NAX
  is available, and are the prefill-SDPA lever for the follow-up file.

---

## 4. Follow-up-file note (NOT this task): prefill SDPA / flash kernel

For the reopened "prebuilt flash/SDPA prefill kernel" task:
- `714a7efc` (fused full-attention, head_dim 256, **NAX**) is a prefill-width fused attention
  path, usable at head_dim 256 / GQA 6 **if** the M5 Pro is NAX.
- `fa0d4463` (gqa-8 decode attention) is the *decode* attention path (K/V-read efficiency), not
  a prefill-width flash kernel.
- **Caveat:** these arrive together with the v0.32.2 kernel set (unreleased main); there is no
  way to take the prefill SDPA lever without the full pin bump. Reassess under that task after
  U2 characterizes the v0.32.2 build.

---

## 5. Decision: **GO to U2**

Relevant, path-specific kernel improvements exist in C++ MLX v0.31.1 → v0.32.2. Per the task
decision rules, a relevant kernel change ⇒ **U2** (framework upgrade A/B matrix).

**Risk carried into U2 (the divergence audit + phase-level checks bear it):**
1. **Unreleased pin.** The only carrier is mlx-swift **main** (`2bebe4e`), not a tagged release.
   U2 records the exact SHA of both binaries (incumbent vs upgraded).
2. **The bump is not clean.** `ab924c8` notes: (a) a **carried C++ MLX patch** for
   `Device::operator<` / `Stream::operator<` (not strict weak orderings, abort under the Xcode 27
   SDK — "still broken upstream, no issue filed"); (b) **streams became thread-affine in
   v0.31.2**, adopting `new_thread_unsafe_stream` via a C shim. These are binding-internal, but
   they mean the upgrade is a real change surface, not a no-op recompile.
3. **Compat-fix list must stay mechanical.** Per the task, U2 allows only the pin + mechanical
   API-compat fixes in the engine fork; anything beyond rename/adaptation is a **STOP**.

**U2 plan (from the task):** bump the pin to the recorded main SHA; build engine + server (both
suites green, compat list mechanical); then the A/B matrix — decode toks + stepAvg + tEval /
tGraphBuild on **essay-1024** and **specdec-800** (6 reps, rep 1 discarded, interleaved rotating
start, `pmset -g therm` per rep), one **32K prefill** regression cell (pc=2048), qmvbench
M=1..9 sustained attribution, and one in-pipeline **step-trace** cell (`QWEN_MTP_STEP_TRACE=1`).
KEEP gate: ≥3% mean decode-toks improvement on **both** fixtures AND ≥4/5 paired reps favor the
upgrade, all correctness gates pass (policy v3), prefill within noise, phase-sums exact,
engagement summaries unchanged, compat list mechanical.

---

## 6. Open question (for U2, not blocking the survey)

- Is the M5 Pro an **NAX** device? (Empirically confirmed by the U2 step-trace engagement
  summary; the standard-Metal kernel improvements apply regardless.)

---

## 7. U2 result — **REVERT (build-infrastructure barrier); negative result**

**Attempt.** Bumped the engine pin `mlx-swift 0.31.6 → main 2bebe4e` (C++ MLX **v0.31.1 →
v0.32.2**), `swift package resolve` (Package.resolved → `2bebe4e`).

**Build gates (Swift/C++) — green, zero compat fixes.**
- Engine `swift build --target MLXLLM`: **complete** (C++ MLX v0.32.2 compiled).
- Server `swift build --target HTTPServer`: **complete** (63 s).
- Compat-fix list: **just the pin** (`.upToNextMinor(from: "0.31.6")` → `.revision("2bebe4e")`).
  No engine/server source changes.
- Our model path: `Qwen38MTPDiagnosticTests` **PASS** (3/3, incl. "Wide Verify depth-5 in
  serial family").

**Barrier — the Metal kernel library is a stale prebuilt artifact.**
- The full engine test suite **crashed** on `testQwen35MoECompiledDecodeTracksWeightUpdates`:
  `Fatal error: [metal::Device] Unable to load kernel dot_product_float32_it32_tg512_sg16`.
- **Root cause:** the Cmlx target builds C++ MLX in **NO-JIT** mode (`nojit_kernels.cpp`;
  `METAL_PATH="default.metallib"`), bundling a **prebuilt** Metal kernel library. That
  `default.metallib` (158 MB) is a **stale v0.31.1** artifact (dated Sep 14) produced by a
  separate **PrepareMetalShaders** step. A pin bump recompiles the C++ code **but SwiftPM does
  not regenerate the metallib** (a warm `swift build` is a 1.4 s no-op; deleting the metallib
  does not make SwiftPM rebuild it). The missing `dot_product_…` kernel is a v0.32.2 kernel
  absent from the v0.31.1 metallib; it is **not** in the fresh v0.32.2 JIT sources either (it
  is a prebuilt kernel). So the "upgraded" build runs **v0.32.2 C++ against v0.31.1 kernels**.
- **Consequence:** the improved kernels (`qmv_wide`, NVFP4 QMV M5 Max, M5-class qmv batch
  limit, split-K quantized matmul, gqa-8 decode attention) are **not active**; a clean A/B
  matrix is **impossible via the pure pin-bump path** (the "upgraded" binary would not run the
  v0.32.2 kernels, making any measured delta meaningless). The build gate (engine suite fully
  green) is **not met** (MoE crash).
- **Not a "no fix exists" (U4) closure.** The kernel improvements DO exist in C++ MLX v0.32.2
  (Section 2). The blocker is the metallib build-infrastructure gap, not the absence of an
  upstream fix.

**Action taken.** Reverted the pin to `0.31.6` (engine + server `Package.resolved`). Verified
**green**: `CompiledDecodeWeightUpdateTests` 6/0 (the MoE dot_product kernel loads again from
the v0.31.1 metallib, matching the v0.31.1 C++).

**Follow-up task (NOT this one).** To run the A/B: (1) re-run the **PrepareMetalShaders**
CMake step to regenerate `default.metallib` for C++ MLX v0.32.2; (2) confirm the full engine
suite is green (MoE kernel loads); (3) then run the decode A/B matrix (essay-1024 +
specdec-800, 6 reps, both binary SHAs recorded) + 32K prefill regression cell + qmvbench
M=1..9 + one step-trace cell, per the U2 plan in §5.
