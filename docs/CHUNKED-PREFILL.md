# Chunked Causal Prefill

Bounded-memory prefill for long prompts. Eliminates the quadratic
`[seq × seq]` dense-attention scores buffer that overflows the Metal single-buffer
cap at ~27K context tokens.

**Status: complete.** Bit-exact (greedy stream hash) at 8K; 32K completes where
dense crashes.

## Why

The dense causal prefill materializes the full scores matrix
`[B, nQHeads, L, L]` (bf16) in one Metal buffer. At `L = 32768`,
`nQHeads = 24`: `24 × 32768² × 2 = 51.6 GB`, exceeding the Metal hard cap
(`MTL_MAX_BUFFER_SIZE`, ~30.2 GB). The allocation fails with
`[metal::malloc] ... greater than the maximum allowed buffer size` and the process
traps (SIGTRAP). The maximum feasible dense context on this hardware is therefore
~24K tokens (see `LONG-CONTEXT-BENCHMARKS.md`).

The scores matrix is only ever read by the causal softmax (each row `i` reads
columns `0…i`), so it need not materialize all at once.

## Design

When `MLX_CHUNKED_PREFILL=1` (default **off**), a fresh causal prefill with
`L > 4096` is processed **sequentially in query tiles** of 512. For each tile
`[s, e)` the layer updates the KV cache with that tile's keys/values (the cache
grows to `e` keys) and runs the fused SDPA over the grown cache with the
bottom-right-aligned `.causal` mask, so query `s+i` attends to keys `[0, s+i]`.

The per-tile buffer is `nQHeads × tile × L × 2` bytes — **linear** in `L`
(`24 × 512 × 32768 × 2 ≈ 0.8 GB` at 32K) vs the dense `L²` (`51.6 GB`). Each
tile's buffer is released before the next tile.

### Gate

The chunked path engages only for a fresh causal prefill (cache offset `0`,
`L > 4096`, `.causal` or materialized causal `.array` mask). Decode (`L = 1`),
short prefills, and incremental prefills (offset > 0) keep the dense path
byte-for-byte, so the default (off) build is bit-identical to the pre-change
engine. The gate lives in `attentionWithCacheUpdate` (the KVCacheSimple branch
that runs the fused SDPA), which is the path the model's full-attention layers
actually use during prefill.

### Correctness

The `.causal` mask is **bottom-right aligned**: a tile of `(e-s)` rows against a
cache of `e` keys makes query `s+i` attend to keys `[0, s+i]` — exactly the key
set and softmax reduction of the un-chunked prefill. This is an exact partition,
not an approximation; the only difference from dense is FP accumulation order
across tiles, within FP tolerance.

Verified:
- Unit: `testChunkedCausalPrefillMatchesDense` (KVCacheSimple, L = 33/64/128/129)
  — chunked vs dense within 1e-2 (a 1-row last tile routes through a different
  fused-SDPA accumulation path). `testChunkedCausalMatchesDense` covers the
  quantized-cache variant of the same tiling.
- (legacy note) `testChunkedCausalMatchesDense` (L = 64/600/1024/1025, nRepeats 1/2) —
  chunked vs dense within 1e-3.
- End-to-end: 8K greedy stream hash identical dense vs chunked
  (`763eccc3…`), 32K completes with chunked (dense traps).

## Configuration

| Knob | Default | Meaning |
| --- | --- | --- |
| `MLX_CHUNKED_PREFILL` | `0` (off) | `1` enables chunked causal prefill for `L > 4096`. |

Tile size (512) and threshold (4096) are fixed constants (`MLXChunkedPrefill`), not
exposed as knobs.

## Admission control

The server's memory admission now models the transient prefill buffer in addition
to the steady-state KV cache:

- Dense: `nQHeads × L² × 2` bytes (quadratic).
- Chunked: `nQHeads × tile × L × 2` bytes (linear).

A request whose transient buffer exceeds the Metal single-buffer cap (×0.9) is
rejected with HTTP 507 / code `prefill_buffer_exceeded`. With dense prefill, 32K+
requests are rejected cleanly (the buffer cannot fit); with `MLX_CHUNKED_PREFILL=1`,
the same requests are admitted (buffer bounded to ~0.8 GB). See
`Sources/HTTPServer/Generation/MemoryAdmission.swift`.

## Scope

- No kernel, head-topology, quantization, or weight changes.
- The GDN (gated-delta-net) recurrent layers are unaffected (they are not
  attention; their prefill is already linear in `L`).
- No continuous batching.
- Off by default; the default build is bit-identical to the pre-change engine.
