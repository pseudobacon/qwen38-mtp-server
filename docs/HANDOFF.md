# HANDOFF — K=2 round decomposition (IN PROGRESS at write time; measurement complete, commits pending)

> **Checkpoint status (2026-09-14 20:02:33+01:00).** The fresh-checkpoint
> procedure **completed**: `./scripts/agent-checkpoint.sh` ran successfully in
> both repositories and wrote `.dsh/last-agent-checkpoint` (gitignored) in
> each before this file was finalized.

## Objective and acceptance criteria

Decompose the ~50 ms/round of non-backbone overhead in the pinned k = 2
speculative round and close with a kernel-addressability verdict in
`benchmarks/PROFILE-K2.md`.

- [x] Phase 1: boundary map of the round (16 host segments A–P; the single
      blocking `eval` is the only device→host sync; head flush+chain is
      `asyncEval`'d before it).
- [x] Phase 2: algebraic cells (4 configs × 2 fixtures × 6 reps, hash-gated)
      + 4-pair trace A/B (+2.19 ms/round artifact estimate).
- [x] Phase 3: xctrace Metal System Trace of 110 k=2 rounds at 256 context,
      offset-calibrated (offset 696,415,092,781,278; GPU `start-time` is
      trace-relative, never ns-since-boot).
- [x] Phase 4: FullBench diagnostic tool (engine) + root-cause of the M=1
      context anomaly → bench artifact (per-rep first-decode penalty), not a
      model property.
- [x] `benchmarks/PROFILE-K2.md` written (server repo).
- [x] progress.md updated (Done line + K=2 decomposition section).
- [ ] Commit engine first (FullBench + Package.swift), then server
      (PROFILE-K2.md + drivers + results + progress.md + this file); merge
      both to main; delete feature branches.

## Result (one live number per fact)

Same context (256 tokens): serial M=1 round 59.4 ms (GPU busy 55.6, util
96.0%); k=2 round 86.0 ms (tG 4.5 + tE 80.4 (GPU busy 79.8, util 98.4%) + tC
1.1 + tH 0.02). Δ = 26.6 ms ≈ 24.2 ms real extra GPU work (2 verify rows +
head flush/chain tail) + 3.5 ms host (tape build in MLX C++ flush + accept
walk); **no reclaimable host gap** — the round is GPU-saturated at measured
contexts. In-pipeline algebra (12 records/config): s 59.78 / k1 90.08 /
k2 100.73 / k3 126.17 ms stepAvg (k1 carries the +13.17 ms dead-path repair).
The "107–112 ms" label is the warm end of the in-session thermal trajectory
(k2default 6-rep essay: 89.5 → 102.5 ms); cold end ~86 ms.

**Verdict: no kernel-level win of order 10 ms is addressable from this
checkout** without changing draft depth (rejected), head design, or MLX C++
(tape build, 3.4 ms tG). Fusion is at its documented ceiling (swiGLU 64/64,
gdn 48/64, qkv 16/64).

## Key findings

1. **FullBench per-rep first-decode penalty is a bench artifact.**
   newCache + prime + one measured decode: 117–156 ms at prime = 2048; serial
   mode (one cache, N back-to-back M=1 decodes) is flat 55.8 → 56.8 ms across
   ctx 256 → 3072 and matches the in-pipeline serial cell (56.2 ms). Penalty
   is paid only on the first decode after each prime (`--widths 1,1`: 122 ms
   then 56 ms, same cache), persists after 10 warmup reps, scales with prime
   size. xctrace of the slow first decode: 68 compute intervals, union
   60.5 ms, ~1.2 ms GPU-idle gaps, host cpu→gpu stride ~2.2 ms/kernel vs
   ~0.8 ms/kernel — host MLX flush/encode falls behind after a fresh large
   prime + newCache cycle. Mechanism is in MLX's C++ eval flush (not readable
   in this checkout — the `mlx-swift` checkout is Swift-only; isolated SDPA
   0.3–0.5 ms, cache slice update O(1), GDN S=1 single kernel, no
   fire-and-forget eval — all ruled out). **Prior "M=1 non-monotonic in
   context" readings from per-rep FullBench are retracted.** In-pipeline at
   real server contexts (2–8k) has no per-round newCache/prime and is not
   expected to carry the penalty.
2. **K=1 dead path quantified** (tC 13.17 ms; `rollbackCheckpoints` never
   written). Evidence for a follow-up; not fixed; k=1 is not a production
   config.
3. **QMV routing**: M=1 verify falls back to `quantizedMM`; M≥2 routes the
   QMV verify kernel. The M=3 tape's sub-linear per-row cost is a property of
   that path.
4. **Instrumentation measured, not assumed**: trace timers +2.19 ms/round
   (4-pair alternating A/B).

## Files

- Server (`qwen38-mtp-server`, branch `feature/k2-decomposition`):
  `benchmarks/PROFILE-K2.md` (new, deliverable), `benchmarks/run_k2decomp.sh`
  (new), `benchmarks/run_k2trace.sh` (new), `benchmarks/results/k2decomp.jsonl`
  + `k2decomp-ab.jsonl` (new), `progress.md` (updated), this file.
- Engine (`../mlx-swift-lm`, branch `feature/k2-decomposition`):
  `Libraries/FullBench/main.swift` (new diagnostic tool: per-rep width matrix,
  `--serial` mode, `--sdpa` micro-bench, `FULLBENCH_ANCHORS=1` trace anchors,
  `--hold`), `Package.swift` (FullBench product).
- No production source changes in either repo for this task.
- Session-local trace artifacts: `/tmp/k2trace-*` (server k2 trace),
  `/tmp/fb1-profile.trace` (per-rep M=1@2048, pid 73984),
  `/tmp/fb256-profile.trace` (per-rep M=1@256, pid 76137, offset
  702,590,931,325,346), `/tmp/fb11-*` (corrupt GPU table — do not cite).

## Persistent facts (from the prior completed task — still live)

- Production default: pinned k = 2 (engine `Qwen38MTPBlockSession.draftPolicy`,
  engine main `609e0d5`); rollback knob `QWEN_MTP_DRAFT_K=3` verified
  bit-exact; `--spec-draft-n-max` offer cap bounds effective k.
- Final headline (current main, single binary `e448b2e2…`): essay 21.89 /
  specdec 23.29 tok/s. Registered streams: essay `949b9423…` (k=2, q4),
  specdec `139acb9d…` (k=2, q4), specdec serial `c70882fc…`.
- Engine main `609e0d5`; server main `f6ac7bc` (plus the HANDOFF marker commit
  `0973cf7` era — verify with `git log` before relying on it).

## Commands / verification

- Engine build: `cd ../mlx-swift-lm && swift build --configuration release
  --product FullBench` — clean (built this session).
- Engine tests: `swift test --filter Qwen38MTPDiagnosticTests` (2/2 last run
  on main `609e0d5`; FullBench does not touch production code, so no rerun
  required unless production source changed — it did not).
- Server: `swift build --configuration release --product qwen38-mtp-server`
  (no server source changed this task; binary SHA `e448b2e2…` unchanged).
- Reproduction of the decomposition: `benchmarks/PROFILE-K2.md` §7.

## Do NOT repeat

- Never enable `MLX_QWEN_MTP_TRACE_SYNC_HEAD=1` (destroys head/verify overlap).
- Never cite per-rep FullBench M=1 numbers at prime > 256 as serial-decode
  cost — use `--serial` mode or the in-pipeline cells.
- xctrace GPU `start-time` is trace-relative; always offset-calibrate against
  `mtp-anchor` / `fb-anchor` uptime-ns probes (CLOCK_UPTIME_RAW ==
  DispatchTime uptime ns, excludes sleep; macOS CLOCK_MONOTONIC includes
  sleep — the naming is reversed from Linux).
- Finalize xctrace with `kill -INT` while the target is alive; wait for
  "Attaching to" before proceeding; validate the exported GPU table has rows
  before analyzing (the fb11 capture came back empty twice — do not cite it).
- Never edit source files with python/sed/awk/shell scripts.
- Engine commits before server; merge engine first; plain commit messages
  (no parentheses/brackets under zsh); `git merge --no-edit`.
- Single-stream measurement; no parallel builds/tests during headline cells.

## Next step (exact)

1. Run `./scripts/agent-checkpoint.sh` (both repos) and confirm success.
2. Engine first: commit `Libraries/FullBench/` + `Package.swift` on
   `feature/k2-decomposition` (message: `Add FullBench diagnostic tool for the
   k2 round decomposition`), then `git checkout main && git merge
   --no-edit feature/k2-decomposition && git branch -d feature/k2-decomposition`.
3. Server: commit `benchmarks/PROFILE-K2.md`, `benchmarks/run_k2decomp.sh`,
   `benchmarks/run_k2trace.sh`, `benchmarks/results/k2decomp*.jsonl`,
   `progress.md`, `docs/HANDOFF.md` (message: `K2 round decomposition profile
   and drivers`), merge to main, delete branch.
4. Checkpoint marker already updated in this file (completed 2026-09-14
   20:02:33+01:00, both repos).

## Repository state (verified at write time)

- `qwen38-mtp-server`: branch `feature/k2-decomposition` (main at `f6ac7bc`
  + HANDOFF marker), untracked: `benchmarks/run_k2decomp.sh`,
  `benchmarks/run_k2trace.sh`, `benchmarks/results/k2decomp.jsonl`,
  `benchmarks/results/k2decomp-ab.jsonl`; modified: `progress.md`,
  `docs/HANDOFF.md`; new: `benchmarks/PROFILE-K2.md` (verify with
  `git status --short` before editing).
- `../mlx-swift-lm`: branch `feature/k2-decomposition` (main at `609e0d5`),
  modified: `Package.swift`; untracked: `Libraries/FullBench/`.
- Fresh checkpoint marker: written 2026-09-14T20:02:33+01:00 in both
  repositories (`.dsh/last-agent-checkpoint`); fresh-checkpoint procedure
  completed.
