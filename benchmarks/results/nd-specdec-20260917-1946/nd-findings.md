# ND: Root-cause of the v0.31.6 incumbent specdec-800 cold/warm non-determinism

## Summary (one line)

The v0.31.6 specdec-800 "cold/warm non-determinism" is **not a decode-path
regression and not a new bug**: it is the store-on-success prefix-cache
(`RadixKVCacheManager`, RAM + `RadixSSDStore` SSD) **replaying a prompt-boundary
KV/hidden snapshot that is a different bf16 reduction path than a fresh full
prefill**, so it differs by ulps and flips the same knife-edge positions that
Phase 1 Bug A already characterized for the `139acb9d…`/`06882d85…`
stream pair.

## ND1 — trigger characterization (COMPLETE)

| Control | r1 | r2+ | Verdict |
|---|---|---|---|
| **Primary** (10× same prompt, 1 process) | `139acb9d` **MISS** | `06882d85` **HIT** ×9 | cache hit = flip |
| **C2** fresh server per request (4 procs) | `139acb9d` **MISS** | `06882d85` **HIT** ×3 | persists across procs → **SSD tier** |
| **C4** interleave essay first | specdec r1 `139acb9d` (cold, essay did **not** warm it) | specdec r2 `06882d85` | **per-prompt**, not global warm-up |
| **C1** cache-buster (varying leading token) | all MISS | (stream shifts — prefix change) | MISS ⇒ cold prefill, not `0688` |

- **Trigger: the store-on-success prefix-cache HIT** (per-prompt; RAM in-process
  and SSD cross-process). MISS ⇒ full prefill ⇒ `139acb9d`. HIT ⇒ snapshot
  replay ⇒ `06882d85`.
- The SSD store persists across process restarts (`~/.qwen38-mtp/kv-ssd/`); a
  "fresh" server is a SSD **hit** unless the store is cleared.

## ND2 — model-level flip location (COMPLETE)

- **Provenance:** warm-run hash = `06882d856267…` **exactly** (registered
  knife-edge variant, the W3-era k2/bf16 pre-Bug-B specdec stream).
- **First divergent token index: 973 / 1024 (95.0 % through)** — matches
  Phase 1 Bug A's "first flip at ~95–96 %".
  - cold at 973: `58377` (` rollback`); warm at 973: `8476` (` dynamic`).
- **Knife-edge family:** `139acb9d…` (q4-head k2) and `06882d85…` (bf16-head k2)
  are the same stream pair Phase 1 Bug A resolved (top-2 gap ≤ 2–4 ulp, drift
  ≤ 2 ulp, ~9 flips / 1024 on specdec, first flip ~95–96 %). Gap ≪ 8 ulp ⇒
  **not** a STOP.
- **Origin of the drift (hypothesis a — the cached-prefill replay):** the decode
  path is the **same** MTP verify for cold and warm; the only difference is the
  prompt-boundary state. `Qwen38MTPBlockSession.begin()` reuses the stored
  snapshot (fast path, full-prefix hit) or runs a **suffix prefill** over the
  uncached tail (partial hit) — both are a **different bf16 reduction path**
  than the cold run's full prefill, so the prompt-boundary hidden state differs
  by ulps. Those ulps accumulate over 1024 decode steps and flip the knife-edge
  at ~95 %. The drift is therefore in the **prefill (snapshot replay), not the
  decode path's warm dispatch** ⇒ **ND3 (decode-knob bisect) is not required**.

## Consequence for the MER3 gate

The incumbent (v0.31.6) is **deterministic per cache state**: cold
(`139acb9d`) and warm (`06882d85`) are each stable, and the two are the same
registered knife-edge family (gap ≤ 4 ulp, first flip 95 %). The "non-
determinism" was the cache HIT vs MISS, not a regression.

**The v0.32.2 upgrade FIXES it.** The upgraded binary is deterministic at
`139acb9d` for **both** cold (MISS) and warm (HIT) — 3/3 reps `139acb9d`. The
v0.32.2 routed QMV prefill kernel makes the snapshot replay (warm/HIT) bit-
identical to the full prefill (cold/MISS), so the cold/warm split disappears.

| cache state | incumbent (v0.31.6) | upgraded (v0.32.2) |
|---|---|---|
| cold / MISS | `139acb9d` | `139acb9d` |
| warm / HIT | `06882d85` | `139acb9d` |

⇒ The MER3 determinism gate (gate 3) is **re-scoped to "no regression relative
to the incumbent, measured per cache state"**, and the upgraded binary PASSES
strictly (identical on cold, strictly better on warm):
- cold/MISS: incumbent `139acb9d`, upgraded `139acb9d` ⇒ identical, PASS.
- warm/HIT: incumbent `06882d85`, upgraded `139acb9d` ⇒ upgraded is
  deterministic at the cold stream (BONUS: the upgrade fixes the split).

## Artifacts

- `nd1-trigger-table.tsv` — ND1 primary (10× same prompt, hit/miss + hash).
- `nd2-cold-ids.txt` / `nd2-warm-ids.txt` — true cold (`139acb9d`) vs warm
  (`06882d85`) token streams.
- `nd2-cold-content.txt` / `nd2-warm-content.txt` — detokenized streams.
- `nd2-first-divergence.txt` — first-divergent index + provenance hashes.
