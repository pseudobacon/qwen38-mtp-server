# Prefill FFN kernel — fast down_proj GEMM at prefill widths

Central document for the prefill-FFN GEMM optimization: the FFP1 finding, the
scoped bit-exactness relaxation that reopens it (policy v2), and the FFP4–FFP7
plan.

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

## Flag reference

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_QWEN_FFN_PREFILL_FAST` | `0` (off) | Engage the fast down_proj GEMM at prefill widths M ≥ 256 (relaxed numeric tolerance). OFF = byte-identical incumbent. |
