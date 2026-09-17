# Prefill FFN kernel — fast down_proj GEMM at prefill widths

Central document for the prefill-FFN GEMM optimization: the FFP1 finding, the
scoped bit-exactness relaxation that reopens it (policy v2), and the FFP4–FFP7
plan.

**Status:** FFP4 kill-switch **NO-GO** (best candidate 0.87–0.88× at M=512, gate
requires ≥ 2×; all candidates *slower* than the incumbent). FFP1 NO-GO stands, now
confirmed under the relaxed policy. FFP5/FFP6/FFP7 not pursued. See
`benchmarks/results/ffp4/ffp4-report.md`.

**MCP probe (2026-09-17):** the M=512 down_proj inefficiency is **M-dependent, not a
fixed shape property** — the per-token FFN cost has an interior minimum at **M=1024**
(6.43 µs/tok, −42.9% vs M=512; down_proj alone −73.7%). This reopens prefill wall via
the **config-only** axis `--prefill-chunk-size` (zero kernel/source change). MCP1 GO;
predicted optimal pc=1024. See `benchmarks/results/mcp-20260917/mcp1-curve.md`.

## Background (established)

- **FFP1 finding** (`benchmarks/results/ffp1/ffp1-report.md`): incumbent
  `quantizedMM` runs `down_proj` (N=5120, K=17408, 4-bit group-64) at ~21.6 TF
  at M=512, vs ~220 TF on `gateup` (N=34816, K=5120) in the same sustained
  no-sync protocol. `down_proj` is 84.7% of per-layer FFN time; FFN is ~43–49%
  of prefill across 8K–64K.
- **Ruled out:** Metal JIT bug (M=512 output is bit-exact vs a dequant→bf16
  reference, `max|diff|=0.0`) and quantizedMM-specific slowness (the same-shape
  bf16 GEMM is 14.2 TF, *slower*). The headroom is in the MLX GEMM **tiling
  engine** for small-N / large-K at M=512 — exactly what bit-exactness forbade
  changing.
- **Tooling:** `qmvbench --ffn-prefill`, `--ffn-pair` (interleaved, DVFS-fair),
  `--ffn-check` (bit-exactness + anomaly localization).

## The decision: bit-exactness policy v2 (2026-09-17) — supersedes the FFP1 NO-GO

**Recorded before any code.** The FFP1 NO-GO is **superseded-by-decision, not
invalidated**: the headroom was un-reachable *only* under bit-exactness. We now
grant a **scoped relaxation** of bit-exactness, gated and default-OFF:

- **RELAXED (numeric tolerance):** FFN GEMMs dispatched at prefill chunk widths,
  **M ≥ 256**. The threshold exceeds every verify width (M 1..9) and decode
  (M=1), and excludes the short-prompt fixtures `essay-1024` / `specdec-800`
  (prefill M < 256).
- **UNCHANGED (byte-for-byte):** decode M=1, MTP verify M 2..9, all short-prefill
  M < 256, every other kernel and code path (attention, GDN, norms, …).
- **Gate:** `MLX_QWEN_FFN_PREFILL_FAST`, **default OFF**, with an M ≥ 256
  dispatch geometry gate. The OFF build is byte-identical to current `main`.
- **Accepted consequence:** long-context (8K+) committed streams through the
  relaxed path differ from the incumbent registry (knife-edge family: fp
  accumulation order, ≤ a few ulp flipping near-tie argmaxes), **not
  corruption**. ON runs register **new** per-fixture stream hashes; OFF runs
  reproduce the incumbent registry exactly. A divergence at a top-2 logit gap >
  8 ulp is a STOP condition.

The full policy is in `benchmarks/MTP-CORRECTNESS-CONTRACT.md` §0.

## Plan

| Phase | Scope | Status |
|-------|-------|--------|
| FFP1 | M=512 incumbent micro-bench + anomaly localization | DONE (NO-GO, superseded) |
| FFP4 | Candidate down_proj kernels, micro-benched (kill switch): ≥2× sustained at M=512 within the numeric bound, else stop | IN PROGRESS |
| FFP5 | Engine integration, gated default-OFF, M ≥ 256 dispatch, counters, unit tests | pending FFP4 GO |
| FFP6 | Model-level audit: 8K/16K/32K/64K ON/OFF, new registry entries, divergence audit | pending FFP5 |
| FFP7 | End-to-end AB matrix + decision (KEEP → merge + default ON, else REJECT) | pending FFP6 |

## MCP probe — FFN M-curve (2026-09-17)

The FFP1/FFP4 kernel axis is closed (incumbent is the best-known kernel at M=512 for this
shape class). The orthogonal **config-only** axis is the prefill chunk size `pc`, which
sets the FFN GEMM width M. MCP1 measured the FFN per-token cost vs M (sustained, DVFS-
interleaved, `qmvbench --ffn-pair`, 2 reps at the decision region):

| M | gateup µs/tok | down µs/tok | **FFN µs/tok** |
|---|---------------|-------------|----------------|
| 512  | 2.108 | **9.153** | 11.261 |
| 1024 | 4.019 | **2.407** | **6.426** ← min |
| 2048 | 4.920 | 2.608 | 7.528 |
| 4096 | 5.298 | 2.892 | 8.190 |
| 8192 | 5.105 | 2.861 | 7.966 |

The down_proj per-token cost drops 73.7% from M=512 (the tiling anomaly) to M=1024, then
the gateup per-token cost rises and the FFN total bottoms at **M=1024**. Total causal
attention work is O(L²) and pc-invariant, so FFN is the only pc-dependent prefill cost →
**predicted optimal pc = 1024**. Full tables + decision: `benchmarks/results/mcp-
20260917/mcp1-curve.md`.

### MCP2 — end-to-end pc sweep: **KEEP pc=2048**, default flipped 512→2048

The 32K/64K end-to-end sweep (fresh server per cell, greedy, bit-exact hash gates)
resolved the prediction: **pc=2048 is the winner**, not the MCP1-predicted 1024 — the
SDPA per-chunk tiling efficiency at larger Q tiles adds a residual gain on top of the FFN
improvement. All three pc values are **bit-exact** (identical committed content at
8K/16K/32K/64K), so there is no knife-edge divergence. Eval-sync prefill wall:

| pc | 32K | vs 512 | 64K | vs 512 |
|----|----:|-------:|----:|-------:|
| 512 | 149.49 s | — | 305.17 s | — |
| 1024 | 139.44 s | −6.7% | — | — |
| 2048 | **130.58 s** | **−12.7%** | **279.03 s** | **−8.6%** |

Memory-safe (peak RSS 14.6 GB @64K, per-chunk buffer 6.29 GB). Default
`prefillChunkSize` flipped **512 → 2048**. Full tables + per-phase breakdown:
`benchmarks/results/mcp-20260917/mcp2-report.md`.

## Flag reference

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_QWEN_FFN_PREFILL_FAST` | `0` (off) | Engage the fast down_proj GEMM at prefill widths M ≥ 256 (relaxed numeric tolerance). OFF = byte-identical incumbent. |
| `--prefill-chunk-size` | `2048` | Prefill chunk width M. MCP2 (2026-09-17): **2048** is bit-exact and −12.7% (32K) / −8.6% (64K) vs 512 → default raised 512→2048. |
