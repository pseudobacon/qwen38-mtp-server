# Active handoff

> Update this file before a context reset, compaction, session switch, or stopping work.
> This is a **resume-state checkpoint**: the next task, the task just finished, repo state, and the hard rules. History lives in `progress.md` — never duplicate a number here that progress.md already carries.
> **Rolling policy:** when a task completes, its detail moves to progress.md; this file keeps a one-paragraph outcome plus a pointer. Older tasks collapse to one line each.

## Goal (next task) — W2: profile `tEval` via external Metal System Trace

Objective: bucket the ~107–125 ms/step `tEval` (≈14.5 GB/step of weight traffic) to find where the next fused-kernel work goes. Answer: (a) MTP draft-head forward + per-round cost, (b) logits-head (`lm_head`) cost per round, (c) recommended next kernel target.

Acceptance criteria: an external `xctrace` Metal System Trace of one greedy request (essay fixture, max_tokens ~256, MTP on, default config = QMV verify ON); a bucketed kernel-time table; numeric answers to (a) and (b) in ms/round. Deliverable: `benchmarks/PROFILE.md`. If (a) ≥ ~5 ms/round, proceed to W4 (quantize the BF16 MTP head to 4-bit group 64; Python for weight conversion is allowed; A/B vs the BF16 head under the determinism gate).

Non-goals: no in-graph timing instrumentation (re-introduces the W1 flush artifact — W2 is external-only); no fusion-default changes (`MLX_QWEN_FUSED_QKV` / `MLX_QWEN_FUSED_SWIGLU` stay ON, `MLX_QWEN_QMV_VERIFY` default stays ON); no re-litigating W1; `lm_head` quantization is report-only (it changes committed tokens).

## Last completed — W1: flush-free Item D rerun → default ON; W5: qmvbench throughput mode (2026-09-14)

W1: the guard's per-call `asData` contiguity probe (which called `self.eval()` on every routed dispatch — the root cause of the original Item D null) was replaced by `Qwen35RowMajorCache`, a lock-protected shape-keyed cache paying the probe once per shape at warm-up (engine `c87fc6b`). All hot-path `asData` sites audited and classified (one hidden flush — fixed; two dead-code probes — fixed with the same pattern; the rest non-hot-path or intentional post-`eval` host reads). The harness gained phase-sum validation (`tEval + tGraphBuild + tCacheState + tHostRead ≈ stepAvg`) and `materialized=0` gates. The flush-free 12-cell A/B (same binary both cells, essay-1024, 5 measured reps): **D1 wins +16.659 ms/step mean (12.2%), 5/5 paired reps**, bit-exact (`949b9423…`), phase-sums exact in all 12 reps, zero materializations, 99.1% routed. Decision rule met → **`MLX_QWEN_QMV_VERIFY` default is now ON** (engine default-flip commit; `=0` selects the pre-Item D baseline). Phase 2 gates re-passed on the default-ON binary (fusion tests both states, diagnostic 93.46% both states, verify matrix bit-exact on both fixtures with engaged routing). Detail: progress.md W1 section.

W5: `qmvbench --throughput N` implemented (N back-to-back submissions, one sync per batch) and measured at M ∈ {1,2,4,8,9}, narrow/wide × routed/fallback: the serialized per-call-sync protocol carried a fixed ~170–235 µs/call overhead; the routed kernel's M = 2..9 win **grows to ~30% under sustained conditions** (confirming the W1 keep); M = 1 is a wash (justifies the M = 1 flip). Raw log: `benchmarks/results/w5-throughput.txt`. Detail: progress.md QMV microbenchmark section.

## Earlier completed (pointers)

- Item D A/B (2026-09-14): originally NULL (−0.252 ms/step) — the `asData` flush artifact; superseded by the W1 flush-free rerun (keep, default ON). Engine `b900aad`; server `4ca9589` + `4af4e73`. Detail: progress.md Phase 3 Item D section.
- Phase 3 dual-fixture re-baseline + compiled-path ablation (2026-09-13): essay-1024 17.55 tok/s (136.559 ms steps), specdec-800 18.92 tok/s (141.623 ms); compiled-path family ablated at +7.35 ms/step, kept default ON. Detail: progress.md.
- Interleaved gate+up layout: implemented, measured (no gain, +6.5 GB), removed — engine `901d2ca`. Detail: progress.md.
- Fusion diagnosis (Items A–C): both packed fusions bit-exact and latency-neutral; QMV grid bug fixed; prompt provenance resolved. Engine `f730e87`, server `d071300`. Detail: `benchmarks/FUSION_REPORT.md`.

## Repository checkpoint

- Engine `../mlx-swift-lm` `main`: `c87fc6b` (W1 cache fix) + default-flip commit (`MLX_QWEN_QMV_VERIFY` unset → ON; `qmvbench --throughput N`); both merged from `feature/w1-default-flip` and earlier; worktree clean.
- Server `/Users/cwong/ai/qwen38-mtp-server` `main`: `31032ec` (harness phase-sum + rerun gates) + W1/W5 evidence commit (progress.md W1 + W5 sections, `benchmarks/results/itemd-rerun-D0/D1.jsonl`, `verify.jsonl`, `w5-throughput.log`, this file); worktree clean.
- Release server binary: `.build/release/qwen38-mtp-server` (SHA `11a8e61e3e0ba82b76a8c0d37e82a10fa591c9ea1624d0063cc53ea8489a5f9e`), built from the merged engine. The qmvbench release binary needs `Resources/default.metallib` next to it (copied from the repo-root `default.metallib`) — recreate if the build dir is wiped.
- Environment: port 18099; `QWEN_MTP_STEP_TRACE=1`; weights in `weights/` + `mtp-head/` (head is **BF16**, 849 MB — W4's live target); venv `/tmp/benchvenv` (HF tokenizer for stream hashes — wiped on reboot, recreate if missing).
- Workspace: `qwen38-mlx-server/` is a **non-repo DSH workspace** of symlinks — no git operations inside it; run everything in the real repos. Keep both symlinks at the same level (engine referenced as `../mlx-swift-lm`).
- Provenance note: `run_cell.sh`'s `server_dirty` field reads `dirty` for matrix runs because result JSONL files are untracked at record time (source tree is clean; `.tmp/` is gitignored). Known false positive — do not "fix" it mid-matrix.

## W3 knob mapping (verified against source 2026-09-14)

- `QWEN_MTP_DRAFT_K=<k>` pins the session's `maxDraftDepth`; `--spec-draft-n-max` server CLI (code default **3**; the help text's "8" is stale; env `QWEN_MTP_MAX_DRAFT_DEPTH` also maps to it) sets the offered depth; offered = min(specDraftNMax, depth). The default `draftPolicy` is the adaptive `costModelDepth` (cap 7 via `segmentedVerifyDepthCap`) — the "current value" cell is adaptive with no env override and no flag.
- k ∈ {1,2}: `QWEN_MTP_DRAFT_K=k` only. k = 3: no env, no flag (offered default 3). k ∈ {4,6,8}: `--spec-draft-n-max 8` AND `QWEN_MTP_DRAFT_K=k`.
- Determinism: greedy committed stream must be bit-identical for all k (draft depth changes proposals/acceptance only, never the target stream); essay-1024 target hash `949b9423…`.
- `run_cell.sh` has no extra-CLI-args hook yet — add it in the W3 task (keep `run_matrix.sh`'s baseline untouched). Do not edit `run_cell.sh`/`run_itemd.sh` while a matrix is in flight.
- Dispatch-count sanity reference: two full backbone forwards per verify round at 16 full-attention layers; default-ON fusions = 3 routed dispatches/layer (96/round), fusion-OFF = 6 (192/round); GDN in_proj is a plain `Linear`, never counted; warmup adds width-1/38/512 fallback buckets.

## Hard rules (inline because breaking them invalidates runs)

- Only in-session deltas are valid; cross-session absolute latencies are never comparable.
- Force-recompile changed engine modules (or compare relink SHA-256) before benchmarking — the stale-binary trap.
- No parallel builds/tests during timing cells; no mid-matrix rebuild; log `pmset -g therm` per rep.
- Every timing cell must reproduce its fixture's stream hash — a mismatch is a correctness stop, not a performance result.
- No in-graph timing/eval-forcing in hot paths (`asData`, stream syncs, host reads inside forward); W2 profiling is external-only.
- Engine commits before server; `git merge --no-edit`; zsh-safe messages (no unescaped parens in `-m`); no `print()` in hot paths; no python/sed source edits (Python OK for W4 weight quantization).
- Decisions on record and the full benchmark protocol: `progress.md`. Orientation, build/test commands, env knobs, file map: `docs/README.md`.

## Exact next step

**W2:** from `/Users/cwong/ai/qwen38-mtp-server`, capture an external Metal System Trace of one greedy request (essay fixture, max_tokens ~256, MTP on, default config): start the release server, `xctrace record --attach <pid> --template 'Metal System Trace'` (or launch under `xctrace record`), export, and bucket kernel time (per-layer backbone matmuls vs MTP-head forward vs `lm_head` logits vs attention/GDN recurrence). Write `benchmarks/PROFILE.md` with the bucketed table and the ms/round answers to (a)/(b). Then W3 (draft-k sweep, knob mapping above) and W4 (if W2 triggers) follow; W5's M = 16/17 extension is gated on W3's optimum sitting at the sweep ceiling.

## Fresh checkpoint

Fresh-checkpoint procedure completed successfully (2026-09-14 03:11 BST):

```
# Repository checkpoint (server)
- Repository: /Users/cwong/ai/qwen38-mtp-server
- Branch: main
- HEAD: a603c33
- Git status: clean
- Latest commit: a603c33 W1: record flush-free Item D rerun (keep, default ON) and W5 throughput results

# Repository checkpoint (engine)
- Repository: /Users/cwong/ai/mlx-swift-lm
- Branch: main
- HEAD: a5f102f
- Git status: clean
- Latest commit: a5f102f W1: default MLX_QWEN_QMV_VERIFY to ON after flush-free rerun keeps D1; add qmvbench throughput mode
```

Both worktrees clean (run via `scripts/agent-checkpoint.sh` inside each repo). _(Regenerate at the start of the next session.)_
