# P3 — Gated Attention Path (LCP)

**Date:** 2026-09-16 · **EXP_ID:** LCP · **Variant:** `bit_exact_gate`
**Device:** M5 Pro 48 GB · **Model:** Qwen3.8-27B 4-bit + MTP

## 1. Gate semantics

| flag | value | attention path | other fusions |
| --- | --- | --- | --- |
| `ENABLE_BIT_EXACT_ATTENTION` | `1` | reference **dense** attention (single-pass prefill, no chunking) | unchanged (existing env) |
| `ENABLE_BIT_EXACT_ATTENTION` | `0` | **chunked** attention path (pc=512) | unchanged |
| `ENABLE_BIT_EXACT_ATTENTION` | unset | legacy `MLX_CHUNKED_PREFILL` behavior | unchanged |
| `ENABLE_BIT_EXACT` | `1` | dense (master switch, takes precedence) | **all disabled** (SwiGLU/QKV/4-GDN/residual-3D/GDN-prework) |

Single resolver in the engine (`MLXBitExact` / `MLXChunkedPrefill.enabled`,
`MLXLMCommon`); the server admission policy uses the same resolver, so
`ENABLE_BIT_EXACT_ATTENTION=1` makes admission model the dense (quadratic) buffer and
reject infeasible requests with the OpenAI-shaped **507** before any model execution.

## 2. Validation matrix (greedy, pinned fixtures, pc=512 default)

| cell | flag | expected | observed | result |
| --- | --- | --- | --- | --- |
| p3-8k-dense | `ENABLE_BIT_EXACT_ATTENTION=1` | hash `660dd1208737764c` | `660dd1208737764c` | PASS |
| p3-16k-dense | `ENABLE_BIT_EXACT_ATTENTION=1` | hash `2e583ad29dc28465` | `2e583ad29dc28465` | PASS |
| p3-8k-strict | `ENABLE_BIT_EXACT=1` | hash `660dd1208737764c` | `660dd1208737764c` | PASS |
| p3-16k-strict | `ENABLE_BIT_EXACT=1` | hash `2e583ad29dc28465` | `2e583ad29dc28465` | PASS |
| p3-32k-dense | `ENABLE_BIT_EXACT_ATTENTION=1` | HTTP 507 | 507 | PASS |
| p3-64k-dense | `ENABLE_BIT_EXACT_ATTENTION=1` | HTTP 507 | 507 | PASS |
| p3-32k-chunk | `ENABLE_BIT_EXACT_ATTENTION=0` | hash `97bc0d74846a043b` | `97bc0d74846a043b` | PASS |
| p3-64k-chunk | `ENABLE_BIT_EXACT_ATTENTION=0` | hash `14b26f9f89f16da5` | `14b26f9f89f16da5` | PASS |

## 3. Results

1. **Reference path is bit-exact at 8K/16K**: `ENABLE_BIT_EXACT_ATTENTION=1`
   (dense, single-pass) reproduces the pinned pc=512 chunked-baseline hashes exactly —
   consistent with the 2026-09-15 validation (dense == pc=512 == pc=0 at 8K/16K; the
   chunked SDPA query-tile gate engages only above L > 4096 per attention call, and
   both paths reduce to the same FP accumulation order at these lengths).
2. **The strict fallback is bit-exact against the optimized default**:
   `ENABLE_BIT_EXACT=1` (dense attention + every fusion disabled) reproduces the same
   hashes at 8K/16K. This is the end-to-end proof that the production fusions (SwiGLU,
   QKV, 4-GDN, and the LCP P2 extensions when enabled) change nothing bit-for-bit.
3. **Infeasible dense requests are rejected, not silently degraded**: 32K/64K under
   `ENABLE_BIT_EXACT_ATTENTION=1` return the OpenAI-shaped 507 pre-prefill (dense
   scores buffer would be 51.6 GB / 206 GB — above the Metal per-buffer limit and the
   admission budget). The client can then retry with the chunked path.
4. **Chunked path under the gate** (`=0`) is the pc=512 baseline, bit-exact at
   32K/64K.
5. **Performance framing**: the gate does not itself change performance — it selects
   between the reference dense path (feasible only ≤16K; single-pass, larger
   per-pass scores buffer) and the chunked path (the production default, the one the
   P1 profile measures). The optimization lever remains the SDPA kernel (35.6 % of
   64K prefill, O(L²)); any fused/online-softmax replacement is **not** bit-exact
   (Phase 1 Bug A) and would therefore only ever ship behind this gate as the
   non-reference path.

## 4. Reproduction

```bash
cd /Users/cwong/ai/qwen38-mtp-server
bash benchmarks/run_lcp_p3.sh   # 8 cells, ~35 min, hash/507-gated
```

Artifacts in `benchmarks/results/prefill-opt-20260915/`:
`srv-p3-<cell>.log`, `resp-p3-<cell>.json`, `summary-p3-<cell>.json`,
`status-p3-<cell>.txt`, `run-p3.log`.
