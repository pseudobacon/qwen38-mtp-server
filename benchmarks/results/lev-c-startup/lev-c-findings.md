# LEV-C — Startup-time decomposition (cold-start measurement)

**Run ID:** lev-c-startup
**Date:** 2026-09-18
**Method:** gated startup-timing instrumentation (`QWEN_STARTUP_TRACE=1`,
startup-only, no behavior change) added to `MLXGenerator.init`
(server `feature/prompt-lev-campaign`). One default-config start, readyz
polled, thermal snapshot, server killed.

**Provenance:** binary `29434ccf…` (mtime 2026-09-18T05:54:20), server
`306fece`, engine `cfd6df5`; metallib `b57de586…` (Cmlx `1f8e74e`, Xcode
26.6). Trace: `benchmarks/results/lev-c-startup/startup.log`.

---

## Startup stage decomposition

Default config (44 GiB limit, q4 head, k pinned 2, SSD **enabled by default**).
Stage = delta between consecutive `[startup]` marks; % is of time-to-
`warmup-done` (25.35 s). Total to readyz ≈ 27 s (the extra ~1.6 s is Vapor
app/route setup after warmup).

| # | stage | ms | % | what it covers |
|---|---|---:|---:|---|
| 1 | pre-load host setup | 0.1 | ~0 % | config, caches, head-tree selection |
| 2 | **model + head + tokenizer load** | **1 454.7** | **5.7 %** | Metal device init + metallib load + 15.13 GB weights (3 shards) → GPU + tokenizer + 4-bit head attach |
| 3 | host-setup (eval + admission) | 4.8 | ~0 % | `eval(model)`, memory baseline, admission-policy construction |
| 4 | **SSD restore** | **7 029.5** | **27.7 %** | `restoreFromSSD()` — restored **1 radix prefix** from the SSD tier (default ON) |
| 5 | **warmup (all depths)** | **16 862.5** | **66.5 %** | `warmAllDepths(maxDepth=3)` — 512-token seed prefill + per-width verify + MTP-head draft steps + committed-history shapes |
| | **total → warmup-done** | **25 351.6** | 100 % | |

### Weight load (stage 2)

- 3 shards, 15.13 GB total → **~505 ms/shard**, ~10.4 GB/s effective
  (disk read + safetensors deserialization + GPU copy).
- **Caveat: warm-disk.** The weights were in the macOS page cache from prior
  runs; `sudo purge` was unavailable (password-gated), so this is a
  warm-disk number. A cold-disk load would be slower in stage 2 only.

### The dominant cost is warmup (kernel JIT), not weight load

The precompiled metallib (from `scripts/build-metallib.sh`) is loaded during
**stage 2**; but the **per-kernel Metal JIT compilation** (first dispatch of
every `.metal` function/shape) happens in **stage 5 (warmup)** and is the
dominant startup cost: **16.9 s = 66.5 % of startup**. `warmAllDepths`
intentionally compiles every verify width (1..maxDepth+1) at an 8/512-token
long-prefix family plus the head draft steps and committed-history shapes —
so first-run kernel compilation is front-loaded at startup by design.

**Startup-lever implications (measurement only — no implementation):**
- **Warmup (16.9 s, 66.5 %)** is the single biggest lever. It is intrinsic
  (kernel JIT) unless a persistent compiled-kernel cache (Metal's
  `MTLGPUCompiler` / metallib pre-compilation of the *runtime* kernels, not
  just the shipped ones) were introduced — out of scope here, flagged for the
  follow-up campaign.
- **SSD restore (7.0 s, 27.7 %)** is the second lever and is **operator
  control**: `--kv-ssd-disabled` removes it entirely (faster cold start, no
  prefix reuse). It is default-ON, so it is a deliberate latency-vs-reuse
  tradeoff, not a defect. Restoring 1 large prefix from disk in 7 s is
  plausible I/O; a lazy/async restore (populate after readyz) would remove it
  from the critical path — flagged for the follow-up campaign.
- **Weight load (1.45 s, 5.7 %)** is small and disk-bound; not a meaningful
  lever at 15.13 GB / 3 shards.

## Memory-policy knobs (current values)

The named `RuntimeStartupMemoryPolicy` type (knobs `initialPool`, `minChunk`,
`maxChunk`, `maxChunks`) **does not exist** in the current mlx-swift checkout
(`.build/checkouts/mlx-swift`) nor in the engine fork — verified by source
search. The runtime memory system instead uses `MLX.Memory.cacheLimit` plus a
buffer-pool policy based on Metal's `recommendedMaxWorkingSetSize`
(`mlx-swift/Source/MLX/Memory.swift`), and the server's `MemoryAdmissionPolicy`.

Effective values at startup (from the gated trace):

| knob | value |
|---|---|
| `MLX.Memory.cacheLimit` (= admission limit) | **47.24 GB** (44 GiB) |
| system safety reserve | 4.29 GB |
| model baseline (post-load `Memory.activeMemory`) | 15.37 GB |
| KV budget | 27.58 GB |
| KV bits (k/v) | 16 / 16 (fp16) |
| tail size | 1024 |
| chunked prefill | false |

These are the real "current memory policy" values; the named `initialPool/
minChunk/maxChunk/maxChunks` knobs are not present in this toolchain revision.

## Verdict (measurement, no implementation)

- **Startup = 25.4 s to warmup-done, ~27 s to readyz** (default config, SSD on).
- **Warmup/kernel-JIT dominates (66.5 %)**; **SSD restore is 27.7 %** and is
  operator-toggleable; **weight load is 5.7 %**.
- The two actionable startup levers for the follow-up campaign are (1) a
  persistent runtime-kernel compilation cache (to cut the 16.9 s warmup) and
  (2) lazy/async SSD restore (to move the 7.0 s off the critical path).
  Neither is implemented here (no-implementation campaign).

## Reproduce
```
QWEN_STARTUP_TRACE=1 .build/release/qwen38-mtp-server serve --port 18099
# poll /readyz, capture [startup] lines + `pmset -g therm`, then kill
# instrumentation: MLXGenerator.init `StartupTimer` (gated by QWEN_STARTUP_TRACE)
```
