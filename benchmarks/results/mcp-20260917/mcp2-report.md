# MCP2 — End-to-end `--prefill-chunk-size` sweep (32K/64K)

**Date:** 2026-09-17
**Binary (pinned):** `.build/release/qwen38-mtp-server`
SHA-256 `606ed2cfb27ce41e12ccdfd545c80a5b603042413d5f2322bfde8325d885eb08`
(16 Sep; server source unchanged since)
**Port:** 18099 · **Env:** `MLX_CHUNKED_PREFILL=1` · **Sampling:** greedy (temperature 0, top_k 1)
**Fixtures:** `benchmarks/results/prefill-verify-2026-09-15/req{8k,16k,32k,64k}.json`,
`benchmarks/prompts/essay-1024.txt`
**Decision: KEEP `pc=2048`. Default `prefillChunkSize` flipped 512 → 2048.**

---

## Objective

MCP1 established the FFN per-token cost curve is M-dependent (minimum at M=1024,
−42.9% vs M=512) and predicted a larger `--prefill-chunk-size` (pc) is a zero-kernel
lever to cut prefill wall. MCP2 is the designated end-to-end gate: does a larger pc
actually reduce the eval-sync prefill wall at 32K (≥5% mean + ≥4/5 paired), stay
bit-exact, and not regress 64K — with no kernel/source change (config only)?

## Protocol

* One pinned release binary; a **fresh server per cell** (port 18099, `MLX_CHUNKED_PREFILL=1`).
* **Phase A** (hash gates): for each pc ∈ {512,1024,2048}, run 8K/16K/32K once and check the
  committed content hash is bit-exact vs the incumbent (8K `660dd120`, 16K `2e583ad2`,
  32K `97bc0d74`). 32K is per-pc self-consistent + divergence-audited vs pc=512.
* **Phase B** (32K sweep): 6 reps × 3 pc, rotating start cell, rep 1 discarded (warmup),
  5 measured. `pmset -g therm` per rep.
* **Phase C** (generalization): 64K at pc ∈ {512,2048}, 3 reps (rep 1 discarded); essay
  (short) at pc=2048, 2 reps.
* Per rep: `pmset -g therm`, curl wall, PF2 per-phase parse, peak RSS (1 Hz), computed
  per-chunk buffer, committed content hash, phase-sum check.
* Gate metric: **eval-sync prefill wall = PF2 `total_ms`** (the "prefill wall (eval-sync)"
  per the task). Curl wall is the secondary, noisier total.

## Results

### Phase A — hash gates (all bit-exact, no knife-edge)

| pc | 8K | 16K | 32K |
|----|----|----|-----|
| 512 | `660dd120` PASS | `2e583ad2` PASS | `97bc0d74` (incumbent) |
| 1024 | `660dd120` PASS | `2e583ad2` PASS | `97bc0d74` PASS |
| 2048 | `660dd120` PASS | `2e583ad2` PASS | `97bc0d74` PASS |

All three pc values produce **identical** committed content at 8K/16K/32K — speculative
decoding is token-ID identical to the pc=512 incumbent at every chunk size. No knife-edge
divergence (no STOP required). Phase-sum check true on every cell.

### Phase B — 32K eval-sync prefill wall (measured reps 2–6)

| pc | mean total_ms | vs pc=512 | paired (favors vs 512) | gate |
|----|--------------:|----------:|:----------------------:|------|
| 512 | 149.49 s | — | — | base |
| 1024 | 139.44 s | **−6.7%** | 4/5 | **PASS** |
| 2048 | 130.58 s | **−12.7%** | 5/5 | **PASS** |

Both candidates clear the gate (≥5% mean AND ≥4/5 paired). **pc=2048 is the winner.**

Per-phase (mean of measured reps, ms):

| pc | FFN | GDN | attn | (sdpa) | norm | resid | total |
|----|----:|----:|-----:|-------:|-----:|------:|------:|
| 512 | 72408 | 35737 | 37176 | (27574) | 2128 | 2039 | 149488 |
| 1024 | 67265 | 34614 | 34863 | (26262) | 1363 | 1333 | 139438 |
| 2048 | 65259 | 30998 | 32416 | (24704) | 978 | 925 | 130575 |

The win is **not just FFN** — every phase improves with pc: FFN −9.9%, GDN −13.3%,
attn −12.9% (sdpa −10.5%), norm −54%. Fewer, larger chunks cut per-chunk overhead across
the board. (MCP1's SDPA O(L²) pc-invariance argument held for the *total* attention work;
the per-chunk tiling efficiency at larger Q tiles is the residual gain.)

### Phase C — generalization

**64K (measured reps 2–3, eval-sync total_ms):**

| pc | mean total_ms | vs pc=512 | paired |
|----|--------------:|----------:|:------:|
| 512 | 305.17 s | — | — |
| 2048 | 279.03 s | **−8.6%** | 2/2 |

Not regressed — pc=2048 is *faster* at 64K than at 32K's relative gain would suggest is
at risk. All 64K cells bit-exact (`14b26f9f`, the incumbent). Peak RSS at 64K pc=2048
= 14.6 GB (per-chunk buffer 6.29 GB) — far below the 48 GB unified-memory limit.

**Essay (short, ~25-token prompt) at pc=2048:** works correctly. `finish_reason=length`
(max_tokens 128), consistent content hash `00282563` across reps, wall ~5 s. For
prompt < pc the prefill is a single dense chunk (M = prompt len), identical to the
pc=512 behavior — no regression on short prompts.

### Memory (peak RSS, GB)

| L | pc=512 buffer | pc=2048 buffer | pc=2048 peak RSS |
|----|----:|----:|----:|
| 32K | 0.79 | 3.15 | 14.4 |
| 64K | 1.57 | 6.29 | 14.6 |

All well under the 30.15 GB Metal cap and the 48 GB unified-memory limit. The max-context
(256K) request is rejected by admission control before the per-chunk buffer (25.8 GB at
pc=2048) becomes a constraint.

## Decision

**KEEP `pc=2048`.** It is bit-exact at every length, −12.7% at 32K (5/5 paired) and
−8.6% at 64K (2/2 paired), and memory-safe. The default `prefillChunkSize` is flipped
**512 → 2048** (`Sources/HTTPServer/ServerConfig.swift`). `ServerConfigArgumentTests`
(8/8) and the full `HTTPServerTests` suite pass with the new default.

## Reproduce

```bash
# Phase A+B (32K sweep + hash gates):
bash benchmarks/run_mcp2.sh
# Phase C (64K + essay):
bash benchmarks/run_mcp2_phasec.sh
# Gate analysis:
/tmp/benchvenv/bin/python benchmarks/results/mcp-20260917/analyze_mcp2.py
```

Raw data: `mcp2-reps.jsonl` (Phase B), `mcp2-hashgates.jsonl` (Phase A),
`mcp2-phasec.jsonl` (Phase C); per-cell `srv-*.log`, `resp-*.json`, `rss-*.csv`,
`wall-*.txt` in this directory.
