# Flash Attention Integration — Analysis

**Status: NOT integrated — fundamental conflict with the bit-exactness requirement.**
This document records the investigation and why a flash-attention kernel cannot
satisfy the task as specified, and what the current chunked-prefill path already
provides. No engine or server code was changed by this analysis.

## Task as specified

Integrate a flash-attention kernel to reduce prefill memory and cut prefill time
>20% vs. chunked prefill at 32K, enabling 64K+ context, **with bit-exactness
required** and chunked prefill retained as a fallback.

## Investigation (evidence)

### 1. MLX has no flash-attention kernel for the prefill (T_q > 1) case

`MLXFast.scaledDotProductAttention` (`.build/checkouts/mlx-swift/Source/MLX/MLXFast.swift:203`)
documents its dispatch:

> "This function will dispatch to an optimized Metal kernel **when the query
> sequence length is 1**. It handles other cases with regular MLX operations."

So:
- **Decode (T_q = 1):** optimized Metal kernel (this is the fused decode path the
  model already uses).
- **Prefill (T_q > 1):** "regular MLX operations" = `scores = (q·scale) @ kᵀ`,
  `softmax(scores)`, `@ v` — i.e. it **materializes the full `[L, L]` scores
  matrix**. This is the source of the 51.6 GB buffer at 32K (24 heads × L² × 2 B).

There is no separate flash-attention entry point in the Swift API, the C header
(`mlx/c/fast.h`), or the prebuilt MLX framework. The `memoryEfficientThreshold`
parameter in one Swift overload is not forwarded to the C function (dead).

### 2. Flash attention (online softmax) is not bit-exact

Flash attention replaces the single-pass softmax with an **online** form: process
keys in blocks, maintaining a running max `m` and running sum `l`, and rescale
partial outputs by `exp(m_old − m_new)`. The result is *mathematically* equal to
the exact softmax but is computed in a different accumulation order, giving a
~1–2 ulp difference.

This project's Phase 1 (Bug A) finding quantifies exactly this hazard: a batched
forward that differs from the serial forward by a **bf16 reduction order of
≤ 2 ulp flips the greedy argmax at ~9 of 1024 positions** (top-2 gap ≤ 2–4 ulp).
A flash-attention kernel changes the accumulation order, so it **cannot be
bit-exact (token-stream identical)** with the dense/chunked path. The
"bit-exactness required" constraint is therefore not satisfiable by a true flash
kernel.

### 3. The current chunked prefill already achieves the bit-exact memory bound

The Phase I chunked prefill tiles the **query axis** (tile 512) and computes each
row's softmax **exactly** (MLX's single-pass softmax over keys `[0, i]`), updating
the KV cache incrementally. This is the query-tiled half of flash attention with
**exact** per-row softmax — the maximum bit-exact memory optimization:

| Context | per-tile scores buffer | KV cache | status |
| --- | --- | --- | --- |
| 32K | 0.81 GB | 2.15 GB | works (measured, 175.9 s) |
| 64K | 1.61 GB | 4.29 GB | works (memory) |
| 128K | 3.22 GB | 8.59 GB | works (memory) |

The per-tile buffer hits the 30.2 GB Metal cap only at L ≈ 1.2M tokens; the KV
cache hits the 44 GB unified-memory budget at ≈ 671K tokens. **The "64K+ works"
criterion is already met by chunked prefill** — no flash kernel is required to
raise the context ceiling.

What true flash attention would add over chunked prefill is only (a) a lower
memory constant (`O(L)` vs `O(tile·L)` per tile — 805 MB → a few MB at 32K, both
far under the cap) and (b) a *potential* speedup from keeping partials in on-chip
memory. Neither is required for the context goal, and (b) is not bit-exact.

### 4. Measured: the full-attention SDPA is ~0% of the 32K prefill

To test whether the >20% prefill-speedup target was even reachable, I added a
temporary, env-gated (`MLX_TRACE_ATTENTION=1`) wall-clock timer around the
full-attention SDPA (L > 1) in the engine, ran one 32K chunked request, and
removed it (engine left byte-identical to main). Result:

| | |
| --- | --- |
| 32K wall | 120.7 s (TTFT / prefill 114.9 s) |
| full-attention SDPA total | **~0.05 s** |
| full-attention fraction of prefill | **~0.04%** |

The server's own `--prefill-chunk-size 512` (default) splits the prefill into
L=512 forward passes, so each chunk's attention is `O(512²)` and the **O(L) parts
(QKV/output projections, 48 GDN recurrent layers, FFN) dominate the prefill**.
The O(L²) full-attention SDPA — the *only* thing a flash kernel would optimize —
is ~0.04% of the prefill. **No attention optimization (including flash attention)
can approach the >20% target**; the prefill cost is in the O(L) GEMMs and GDN
layers, a separate and much larger optimization surface.

## The conflict

The task requires **both** a flash kernel **and** bit-exactness. These are
incompatible: flash attention's online softmax changes the accumulation order and
therefore the greedy token stream at knife-edge positions (Phase 1, Bug A).
Integrating it would relax the token-stream bit-exactness invariant for
long-context prompts — a change to greedy semantics that this project gates
explicitly.

## Options

1. **Keep chunked prefill (recommended).** Bit-exact, already enables 128K+.
   No work; the context-ceiling goal is met.
2. **Integrate a non-bit-exact flash kernel.** Would need an *explicit* decision
   to relax token-stream bit-exactness for long-context (a new documented
   invariant change), plus a from-scratch Metal kernel for this geometry
   (head_dim 256, GQA 6) — a large, high-effort, high-risk change that the
   project's "no kernel changes" discipline discourages. Not bit-exact.
3. **Bit-exact performance optimization of chunked prefill.** The prefill FLOPs
   are `O(L²)` in the 16 full-attention layers and are unchanged by any bit-exact
   tiling; the only bit-exact levers are tile-size/launch-overhead and GEMM
   efficiency (kernel tuning, discouraged). Upside is bounded and must be
   measured against the attention fraction of the 32K prefill before it is worth
   pursuing.

## Recommendation

Do **not** integrate a flash-attention kernel under the current constraints. The
evidence is now definitive on both counts:

1. **Bit-exactness:** a true flash kernel's online softmax is not bit-exact
   (Phase 1, Bug A); the task requires bit-exactness.
2. **Speedup is not reachable:** the measured full-attention SDPA is **~0.04%**
   of the 32K prefill (§4). Flash attention only touches that part, so it cannot
   approach the >20% target. The prefill cost is in the O(L) projections, GDN
   layers, and FFN — a separate optimization surface, not attention.

The context goal (64K+, in fact 128K+) is already met by chunked prefill (§3).
**Keep chunked prefill.** If prefill speedup is later pursued, target the O(L)
GEMMs / GDN layers, not attention.
