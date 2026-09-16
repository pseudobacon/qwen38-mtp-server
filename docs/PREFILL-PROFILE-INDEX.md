# Prefill Profile Index

Central index of long-context prefill profiling for this project. Newest first.

| date | run | scope | key result | report |
| --- | --- | --- | --- | --- |
| 2026-09-16 | LCP P1 | 8K/16K/32K/64K pc=512, eval-sync per-phase + RSS + bit-exact hashes | FFN 56.3→36.5 %, full-attn 12.8→42.1 % (SDPA 5.7→35.6 %, O(L²)); GDN 27.3→19.1 %; norms+residuals ~3 % | [`P1_PROFILE_LCP_m5pro_20260915.md`](../benchmarks/results/prefill-opt-20260915/P1_PROFILE_LCP_m5pro_20260915.md) |
| 2026-09-16 | LCP P2 | `MLX_QWEN_FUSED_RESIDUAL_3D`, `MLX_QWEN_FUSED_GDN_PREFILL` (default off) | bit-exact at 8K/16K/32K/64K (unit + real-model hash gate); per-kernel wall-time not resolvable (±1.6× session thermal/state variance); keep default-OFF | [`P2_KERNEL_LCP_residual3d_gdn-prefill_20260915.md`](../benchmarks/results/prefill-opt-20260915/P2_KERNEL_LCP_residual3d_gdn-prefill_20260915.md) |
| 2026-09-16 | LCP P3 | `ENABLE_BIT_EXACT_ATTENTION`, `ENABLE_BIT_EXACT` gate validation | see [`P3_ATTN_LCP_bit_exact_gate_20260915.md`](../benchmarks/results/prefill-opt-20260915/P3_ATTN_LCP_bit_exact_gate_20260915.md) | [`P3_ATTN_LCP_bit_exact_gate_20260915.md`](../benchmarks/results/prefill-opt-20260915/P3_ATTN_LCP_bit_exact_gate_20260915.md) |
| 2026-09-15 | prefill-verify | 8K/16K/32K/64K pc=512 vs pc=0 vs dense; per-phase + memory + bit-exactness | pc=512 near-optimal baseline; 64K completes (333 s); pc=0 65 % slower (robustness only); 8K/16K dense==pc=512==pc=0; 32K pc=0 knife-edge diverge; SDPA 18.5 % @32K → 29.3 % @64K | [`REPORT.md`](../benchmarks/results/prefill-verify-2026-09-15/REPORT.md) |
| 2026-09-15 | prefill-profile (32K) | 32K pc=512 eval-sync per-phase | FFN ~50 % / GDN ~27 % / full-attn ~23 % | [`docs/PREFILL-PROFILE.md`](PREFILL-PROFILE.md) |
| 2026-09-14 | fusion | SwiGLU/QKV/4-GDN packed fusion benchmark | fusion report | [`benchmarks/FUSION_REPORT.md`](../benchmarks/FUSION_REPORT.md) |

## Standing conclusions

1. **SDPA is the growing center** (5.7 % @8K → 35.6 % @64K, O(L²), dense unfused
   prefill path). A flash-style prefill kernel is the highest-leverage long-context
   optimization — but a fused/online-softmax attention is **not bit-exact** (Phase 1
   Bug A) and therefore stays behind the user gate `ENABLE_BIT_EXACT_ATTENTION=0`
   (chunked/fused path); `=1` is the reference dense path.
2. **FFN is the dominant cost at short/medium context** (56.3 % @8K); the body is the
   prebuilt MLX 4-bit GEMM (out of scope for Swift-level work).
3. **Bit-exactness is the default**: all production paths (pc=512 chunked prefill,
   fused fusions) are bit-exact; the strict unoptimized fallback is
   `ENABLE_BIT_EXACT=1`; per-path gates see the LCP reports.
4. **Thermal/state variance on this M5 Pro is up to ~1.6× within a session** (no
   `pmset` warning recorded); single-shot wall-time cells cannot resolve <3 % deltas.

## Environment flags (prefill/attention/fusion surface)

| flag | default | effect |
| --- | --- | --- |
| `ENABLE_BIT_EXACT` | off | strict unoptimized fallback: dense attention + all fusions disabled (master switch, takes precedence) |
| `ENABLE_BIT_EXACT_ATTENTION` | (unset = legacy) | `1` = reference dense attention; `0` = chunked/fused attention path |
| `MLX_CHUNKED_PREFILL` | off | chunked prefill (pc=512); superseded by the two flags above when set |
| `QWEN_PREFILL_CHUNK_SIZE` | 512 | prefill chunk size |
| `MLX_QWEN_FUSED_RESIDUAL_3D` | off | fused residual+RMSNorm on 3-D prefill tensors (LCP P2) |
| `MLX_QWEN_FUSED_GDN_PREFILL` | off | fused GDN prework at prefill widths (LCP P2) |
| `MLX_QWEN_FUSED_SWIGLU` / `_QKV` / `_FOUR_GDN` / `_GDN` | off | existing packed fusions |
| `MLX_TRACE_PREFILL` | off | eval-synchronized per-section prefill timing (`PF2` line) |
