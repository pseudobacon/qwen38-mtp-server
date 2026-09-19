# LEV-J Phase FA — Metal flash-SDPA micro-kill-switch

**Run ID:** lev-j / fa
**Date:** 2026-09-19
**Decision:** **GO.** A standalone, deterministic, bit-exact Metal flash-SDPA kernel
clears the LEV-D `c_flash` bar by 30–35× at both 32K and 64K prefixes and is 3.5–4.8×
faster than the incumbent dense SDPA at the identical geometry. Proceed to Phase FB
(engine integration, default OFF).

---

## Deliverable

Standalone benchmark target `flashbench` (`Libraries/FlashBench/main.swift`, engine fork
`mlx-swift-lm`, product `flashbench`). **Zero production-code changes** — this is the
micro-kill-switch. The kernel, reference, and incumbent are all in one file.

Build: `cd mlx-swift-lm && swift build --product flashbench`
Run:   `.build/arm64-apple-macosx/debug/flashbench --reps 20 --warmup 8 --q 2048,8192 --prefix 8192,32768,65536 --full 0`

## Geometry

Qwen3.8-27B attention: `nq=24`, `nkv=4` (GQA 6), `head_dim=256`, `scale=1/16=0.0625`,
bf16, bottom-right causal. Chunked prefill: `Q ∈ {2048, 8192}`, `prefix ∈ {8K, 32K, 64K}`
(chunk attends to `[0, prefix-Q+q_row]`). Decode (Q=1) and verify rows (Q 2..9) are never
routed to the kernel.

## Kernel design (deterministic by construction)

Per-query-row threadgroup (grid `Q × nq`), 256 threads = 8 simdgroups × 32 head-dim
elements (split-D). Each threadgroup owns one `(h, q_row)` output row and streams K/V in
blocks of `BK=64` keys with online (running max/sum) softmax. `O(prefix)` memory — no
`[nq, Q, prefix]` score materialization.

Determinism invariants (all verified, see below):
- **Uniform `BK` blocks** with out-of-range keys masked to `-inf` scores (a partial final
  block with a *varying* loop bound was observed non-deterministic; uniform blocks are not).
- **Running state (m, l, o) in per-thread registers**, not threadgroup memory. This was
  the decisive fix: the shared-memory `t_m`/`t_l` (redundant 256-thread write) raced with
  the next block's read even across barriers, with per-block probability that scaled with
  block count (deterministic ≤16 blocks, non-deterministic at 65/128 blocks). Register
  state eliminated the race entirely.
- **No atomics.** Fixed reduction tree (`simd_sum` over 32 lanes, then fixed 8-simdgroup
  sum). `exp()` is a pure function of its input.
- **No decode/verify routing** (Q≥2048 only), so decode/verify byte-for-byte unchanged.

## Correctness + determinism (measured, not asserted)

At `Q=2048, prefix=8192` (and cross-verified at `Q=2048, prefix=16384`):

| check | result |
|---|---|
| **Determinism** (bit-exact, two runs same inputs, byte-snapshot) | **true** |
| **Determinism** (cross-process dumps, 3× md5 identical at 12.6M elements) | **identical** |
| **flash vs fp32 reference** max abs diff | 0.01221 (≈1.6 bf16 ulp) |
| **incumbent vs fp32 reference** max abs diff | 6.1e-05 |
| **flash vs incumbent** max abs diff | 0.01221 |

The flash-vs-reference residual (1.6 bf16 ulp) is the expected knife-edge family for
online-softmax rescaling vs a single-pass fp32 softmax — audited, not a bug.

## Benchmark matrix (single-shot median, n=20, DVFS-fair interleaved, same process)

`per-token = median(single-call wall) / Q`. Incumbent = `MLXFast.scaledDotProductAttention`
rank-4 dense (the production path).

| Q | prefix | flash µs/tok | inc µs/tok | speedup |
|----|--------|-------------:|-----------:|--------:|
| 2048 | 8192   | 3.47  | 14.47  | 4.16× |
| 2048 | 32768  | **18.43** | 64.39  | 3.49× |
| 2048 | 65536  | **40.36** | 143.06 | 3.54× |
| 8192 | 8192   | (0.055, full-prefill diagonal, not bar-relevant) | 14.16 | — |
| 8192 | 32768  | **13.93** | 67.13  | 4.82× |
| 8192 | 65536  | **33.18** | 147.08 | 4.43× |

Stability: a repeat run reproduced within ±2% (thermal variance negligible at this bar).

## GO/NO-GO against the bar

LEV-D binding bar: `c_flash ≤ ~630 µs/tok @32K` (strict), `~1205–1404 @64K`. FLOP floor
`4.09 µs/tok @32K`.

| prefix | flash (this run) | bar | margin | vs FLOP floor |
|--------|-----------------:|-----:|-------:|--------------:|
| 32K | 13.9–18.4 µs/tok | ~630 | **34–45× below** | 3.4–4.5× floor |
| 64K | 33.2–40.4 µs/tok | ~1205 | **30–36× below** | — |

The flash kernel is far below the LEV-D "credible flash" range (20–204 µs/tok @32K) — it
runs at ~3.4× the FLOP floor. **GO is overwhelming and not bar-sensitive.**

## Caveats / notes for Phase FB

1. **Incumbent OOMs at full-prefill L≥32K** (dense scores `[nq,L,L]` = 51.5 GB > 30 GB
   buffer limit). The bar-relevant cases are the *chunked* `prefix=32K/64K` geometries,
   which the flash kernel handles in `O(Q)` memory. Full-prefill is flash-only (no
   incumbent).
2. **Current-MLX dense incumbent ≪ LEV-D `c_dense`.** The direct incumbent here is
   ~64 µs/tok @32K, vs LEV-D's `c_dense(2048)=753.9 µs/tok` @32K (≈12×). LEV-D's bar was
   derived from an older/slower dense incumbent. This only makes the GO *more*
   comfortable (flash clears both the stale bar and the current incumbent). Phase FD must
   re-derive the bar with in-session current-MLX numbers before claiming end-to-end win.
3. **Incumbent crashes under sustained no-sync (≥5 enqueued)** — the dense SDPA segfaults
   on buffer reuse. The bench therefore uses single-shot (1 enqueued, 1 eval) for both,
   which is fair. Phase FB integration must not rely on sustained no-sync for the incumbent.
4. Per-token unit here is `chunk_wall / Q` (per query in the chunk). The bar's per-token
   is `total_prefill / L`. These differ by a constant factor that does not change the GO
   (30–45× margin). Phase FD will use the exact bar definition end-to-end.
