# Handoff — qwen38-mtp-server

## Status
**ACTIVE: Residual-lever triage campaign (measure-first, kill-switched) — Phase 0 done (2026-09-18).**

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
| LEV-A | KV-quant default AB (fp16 vs affine8 @ kvTail 1024), 32K multi-turn: hit-rate/hit-TTFT materially up, acceptance in band, no admission regression | 1 | own AB | open |
| LEV-B | Fused-GDN prefill re-AB (`MLX_QWEN_FUSED_GDN`), 32K/64K: ≥3% mean prefill wall AND ≥4/5 paired reps, streams per policy v3 | 1 | own AB | open |
| LEV-C | Startup-time decomposition (instrumentation only): attribute the ~0.34 s SSD-tier delta before LEV-H | 1 | own measurement | open |
| LEV-D | Flash + large-pc compounding model: solve for c_flash at which pc≥8192 beats the pc=2048 incumbent at 32K/64K; cross-check the 6.29 GB scores-buffer memory model | 2 | arithmetic | open |
| LEV-E | Draft-select Metal ceiling: bound (iv) 1.451 ms + (iii) share owned by the draft walk (instrument to confirm; may sit inside verify_build); < 2 ms/round → CLOSE the qwen35DraftSelectKernel lever | 2 | SCH1 ledger | open |
| LEV-F | Tree-drafting + conversation-resume design study (paper, zero code); rank the two; shared blocker = generated state (KV + GDN recurrent) in the radix + rollback composition | 2 | design soundness | open |
| LEV-G | Flip KV default; admission/budget docs; registry; provenance | 3 | LEV-A GO | gated |
| LEV-H | Tokenization-cache persistence (shutdown serialize / startup restore; size-capped, TTL-respecting, corruption-safe); AB cold-start delta | 3 | LEV-C attributes the 0.34 s to tokenization | gated |
| LEV-I | Startup memory-policy tuning | 3 | LEV-C knob sensitivity | **CLOSED at Phase 0 — `RuntimeStartupMemoryPolicy` knobs absent from the checkout (anchor divergence recorded in progress.md)** |
| LEV-J | Flash-attention Metal kernel: FFP-style micro kill-switch against the LEV-D bar first, then full pipeline (determinism gates, divergence audit, end-to-end AB with the pc sweep re-opened under flash, admission update) | 3 | LEV-D bar credible | gated |
| LEV-K | Metal draft-select kernel | 3 | LEV-E bound ≥ 2 ms | gated |
| LEV-L | Stage-3 conversation resume (generated state in radix); tree drafting as follow-on sharing the machinery | 3 | LEV-F design sound | gated |
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

**Next step (exact for this campaign):** Phase 2 zero-code levers — LEV-D
(flash arithmetic from existing pc-sweep + M-curve + 64K SDPA-share data),
LEV-E (draft-select bound from the SCH1 ledger), LEV-F (design study) — then
Phase 1 measurements (LEV-C, LEV-B, LEV-A) under the standing protocol.

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
