# LEV Campaign — Verdict Ledger (measurement campaign, no implementation)

**Date:** 2026-09-18
**Scope:** Phases 0–2. Phase 0 = record hygiene (closed, `306fece`). Phase 2 =
zero-code feasibility (LEV-D/E/F). Phase 1 = measurements (LEV-A/B/C).
Every lever below has a verdict with evidence. No source behavior was changed
except **gated, trace-only instrumentation** (LEV-C `QWEN_STARTUP_TRACE`,
LEV-E `snap_us`/`tape_us` on `traceRounds`).

| lever | verdict | one-line rationale | evidence dir |
|---|---|---|---|
| **LEV-A** KV-quant default | **NO-GO (as default)** | affine8 is +16 %/step slower and changes the stream (coherent but divergent); keep fp16 default, affine8 = explicit memory option | `lev-a-kvquant/` |
| **LEV-B** fused GDN | **verify: GO-as-default (marginal); prefill: FLAG** | `MLX_QWEN_FUSED_GDN` bit-exact, −1 % step (within noise); `MLX_QWEN_FUSED_GDN_PREFILL` **not bit-exact at 32K** (changes stream) → verify prefill-width exactness before defaulting | `lev-b-fusedgdn/` |
| **LEV-C** startup decomp | **measured; 2 levers flagged** | 25.4 s to warmup: warmup/kernel-JIT 66.5 %, SSD restore 27.7 %, weights 5.7 %; `RuntimeStartupMemoryPolicy` knobs absent (reported actual admission values) | `lev-c-startup/` |
| **LEV-D** flash+large-pc | **GO (credible → hand to LEV-J)** | flash removes the 6.3–25 GB scores buffer; `c_flash` bar ~630 µs/tok@32K, ~1205 @64K; a credible Metal flash kernel is 3–35× below the bar | `lev-d-arithmetic/` |
| **LEV-E** draft-select kernel | **CLOSE** | addressable walk host cost 0.81–1.60 ms/round < 2 ms kill-switch; commit is bimodal (rollback path); corroborates the Swift-walk negative (2.05 % slower) | `lev-e-draftselect/` |
| **LEV-F** tree-drafting + resume | **ranked: Stage-3 resume first, tree-drafting follow-on** | shared blocker = composable generated-state checkpoint (KV+GDN `h`+head) with forward-only rollback; tree headroom marginal (<5 %, acceptance ceiling ~1.7–1.8, SDPA ≤9-row) | `lev-f-design-study/` |

## Key cross-cutting findings

1. **32K/64K prefill requires chunked prefill.** Dense 32K prefill overflows the
   transient buffer (51.6 GB > 27.1 GB allocatable) and is rejected; both
   LEV-A/B 32K cells run `MLX_CHUNKED_PREFILL=1`. This is a standing constraint
   on any long-context measurement.
2. **The startup banner is not a reliable `_PREFILL` indicator.** It only
   reflects `MLX_QWEN_FUSED_GDN`; `MLX_QWEN_FUSED_GDN_PREFILL` reads "off" in
   the banner even when set.
3. **Determinism held everywhere.** Every A/B cell was self-consistent
   (1 distinct stream hash per config, rep 1 discarded), and every rep passed
   `phaseSumOK`. In-session paired deltas only (cross-session absolutes are
   labels, per protocol).
4. **The `qwen35DraftSelectKernel` and the deep-draft avenues are closed**
   (LEV-E); the remaining decode cost is (i) kernel exec = 90.6 % of the round.

## Follow-up campaign hand-offs (implementation, not this campaign)
- **LEV-J:** Metal flash SDPA (from LEV-D GO).
- **Verify prefill-width bit-exactness** for `MLX_QWEN_FUSED_GDN_PREFILL`
  (from LEV-B flag) before it can be a default.
- **Stage-3 conversation resume** on a composable generated-state checkpoint
  (from LEV-F); tree-drafting as a gated follow-on.
- **Startup levers** (from LEV-C): persistent runtime-kernel compilation cache
  (cuts the 16.9 s warmup); lazy/async SSD restore (moves the 7.0 s off the
  critical path).
- **`affine8` KV** as an explicit memory-recovery option (from LEV-A), not a
  default.

## Reproduce
```
# measurements (fresh binary with gated instrumentation):
QWEN_STARTUP_TRACE=1 .build/release/qwen38-mtp-server serve --port 18099
bash /tmp/levab2.sh                # lev-b + lev-a matrices (32K, 6 reps)
# zero-code:
python3 benchmarks/results/lev-d-arithmetic/model.py
```
