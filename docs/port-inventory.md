# Lineage Port Inventory (Phase 0)

Sources: canonical `main` (`bbb1061`) vs recovered lineage
`recovered/tasks-1-6` @ `efcf595` (fetched from sibling
`/Users/cwong/ai/qwen-mtp-server`), plus
`recovered-qwenmtp/feature/compact-rejection-math` @ `e03339f`
(Task 1 artifacts) and `recovered-qwenmtp/recovered/ssd-artifact`
@ `ab3456c`.

Baseline test counts: **224 Swift Testing + 120 XCTest**, all green.

## 1. What the recovered lineage changed (per task)

| Task | Commits | Files (server-side, deduped) |
|---|---|---|
| 1 (negative result) | `0e6fdb1`, `59cc211`, `6bdd909`, `42c364b`, `3eaa53c`, `245d367`, `e03339f` | `docs/compact-rejection-rfc.md`, `Tests/HTTPServerTests/CompactRejectionTests.swift` (400 ln, pure distributional), `Tests/HTTPServerTests/WeightTestLock.swift` (42 ln), `benchmarks/compact-rejection-prompt.json`, `benchmarks/run_compact_rejection_ab.sh`, `MLXGenerator.swift` (+5: `QWEN_MLX_SEED` hook), session compact-walk (REJECTED — do not port) |
| 2 (depth autotune) | `61e8b22` | `Sources/HTTPServer/Generation/DepthTuning.swift` (472 ln), `DepthTuningTests.swift` (16 `@Test` + 1 weight-gated), `ServerConfig.swift` (`--tune`), `OpenAIRouter.swift` (`POST /tune`), `MLXGenerator.swift` (tune driver + `weightIdentity()`) |
| 3 (SSD tier) | `ab3456c`/`6501d24` | `RadixSSDStore.swift` (409 ln, new), `RadixKVCacheManager.swift` (+253), `MLXGenerator.swift` (+203), `QwenServer.swift` (+22), `ServerConfig.swift` (+28), `RadixSSDPersistenceTests.swift` (128 ln, 6 pure), `RadixSSDWeightTests.swift`, `Vendor/.../MLXLMCommon/KVCache.swift` (+25: `restoreKVCacheState`), `benchmarks/run_radix_ssd_restart.sh` (147 ln), `docs/radix-ssd-persistence-rfc.md` (261 ln) |
| 4 (reusable-path repair) | `820c9f4` | `MLXGenerator.swift` (+55: prompt-boundary freeze), `RequestMetrics.swift` (+33: per-request + 3 summary keys), `OpenAIRouter.swift` (+10: `QWEN_STREAM_DIAG`), `Qwen36MTPBlockSession.swift` (+36: `reusedPrefixTokens` prop + diag log), `ObservabilityTests.swift` (+5), `RadixReusablePathWeightTests.swift` (80 ln, weight-gated), `benchmarks/run_reusable_path_e2e.sh` (149 ln), `benchmarks/reusable-path-prompt.json`, `docs/metrics.md` (+21) |
| 5 (gate re-spec) | `a7d057d`, `1d7ce2c`, `40b2c36`, `8d62949` | `MLXGenerator.swift` (+8), `ObservabilityTests.swift` (stale-key fix), `RadixSSDWeightTests.swift` (+77: `radixSSDRestoreReportsRealReuse`), `benchmarks/run_radix_ssd_equilibrate.sh` (209 ln), `benchmarks/results/radix-ssd/20260916/GATE-RESPEC.md`, `NEGATIVE.md` |
| 6 (gate PASS + merge) | `f51631f`, `6f9722d`, `efcf595` | `REPORT-EQ.md` (G1 0.117×, G2 0.609 s, median of 7), `run_radix_ssd_equilibrate.sh` (+13), docs |

## 2. Canonical repo current state

- `RadixKVCacheManager.swift` — **present, in-RAM only**: `matchPrefix`
  (LCP partial matching), `store` (insert with split topology),
  `purgeIfMemoryPressure`, `remove`, `clear`, `metrics()`. No
  `diskState`/`setOnEvict`/`snapshot`/`restoreSkeleton`/
  `matchPrefixForGeneration`/`promoteToRAM` (all SSD-tier, to port).
- `TokenizationCache.swift` — present (both lineages; no conflict).
- `RequestMetrics.swift` — **no** `matchedPrefixTokens` /
  `reusedPrefixTokens` / `radixPrefillSkipped` / `prefix_reuse_*`
  (to port in Phase 2).
- `ServerConfig.swift` — known-flag sets `valueTakingFlags`
  (L113–147), `valuelessFlags` (L148–152), `vaporArguments(from:)`
  (L169), `unknownFlagError(in:)` (L206, Task 7). **No** `--kv-ssd-*`
  entries (to add in Phase 3, §5 below).
- `MLXGenerator.swift` — radix wiring present: `matchPrefix` (L1075),
  `begin(prefixCount:...)` (L1081), `exportState()` + `store` **after**
  the decode loop (L1271/L1290) — i.e. exactly the pre-Task-4 store path
  (prompt + generated tokens stored; MambaCache not trimmable → repeat
  never reuses). `QWEN_MLX_SEED` absent. No SSD members.
- `Qwen38Server.swift` — `ModelShutdownHandler(runtimeState:scheduler:)`
  at L210 (no `ssdFlush` param yet).
- `ObservabilityTests.testMetricsSummaryEncodesSnakeCaseKeys` (L250) —
  expected set = 14 keys (no `prefix_reuse_*` yet).
- Session type instantiated: `Qwen38MTPBlockSession`
  (`MLXGenerator.swift` L529/626/1051/1365). Defined in the **fork**:
  `../mlx-swift-lm/Libraries/MLXLLM/Models/Qwen38MTPBlockSession.swift`.

## 3. Session-type rename surface & API verification

Recovered files referencing `Qwen36MTPBlockSession` (must become
`Qwen38MTPBlockSession` on port):
- `MLXGenerator.swift` — 4 refs
- `RadixKVCacheManager.swift` — 1 ref (comment)
- `Qwen36MTPBlockSession.swift` — the file itself (NOT ported; canonical
  session already exists in the fork)

API check (canonical `Qwen38MTPBlockSession`, fork):
- `exportState() -> (tokens: [Int], cache: [any KVCache], hidden: MLXArray, primary: Int, top2: ([Int], [Double]))?` — **present** (fork L379, identical signature).
- `begin(seedTokens:prefixCount:reusableCache:reusableHidden:reusablePrimary:reusableTop2:)` — **present** (fork L859, identical signature, identical desync guard `trimmableOffset == prefixCount` with full-prefill fallback).
- `trimmableOffset(_ cache: [any KVCache]) -> Int` — **present** (fork L2597, `public static`).
- `reusedPrefixTokens` property — **absent** in canonical session. NOT in
  the Phase 0.3 required list; not stubbed. The canonical generator
  computes the equivalent at the call site using the session's own
  adoption rule: `reused = (prefixCount > 0 && cachedEntry != nil && Qwen38MTPBlockSession.trimmableOffset(cachedEntry.cache) == prefixCount) ? prefixCount : 0`.

No stop condition triggered.

## 4. Vendored fork wiring

- Canonical consumes the fork via **local path dependency**
  `.package(path: "../mlx-swift-lm")` (`Package.swift` L23), physical
  `/Users/cwong/ai/mlx-swift-lm` @ `d189c61`.
- `restoreKVCacheState` — **absent** in the fork. The authorized paired
  change: add the 25-line public free function to
  `Libraries/MLXLMCommon/KVCache.swift` (recovered diff verified:
  dispatches `ArraysCache.restoreFromMetaState` (internal, present at
  L1609) / `KVCacheSimple` / `BaseKVCache` direct assignment; all
  prerequisite types present in the fork).

## 5. SSD flag collision check (Task 7 interaction)

Recovered `ServerConfig` **does** have CLI flags (confirmed, not
env-only):
- value-taking: `--kv-ssd-cache-dir`, `--kv-ssd-cache-gb`, `--kv-ssd-ttl-seconds`
- valueless: `--kv-ssd-enabled`, `--kv-ssd-disabled`
- env overrides: `QWEN_KV_SSD_ENABLED`, `QWEN_KV_SSD_CACHE_DIR`,
  `QWEN_KV_SSD_CACHE_GB`, `QWEN_KV_SSD_TTL_SECONDS`

Port requirement: add the 3 value-taking flags to `valueTakingFlags`,
the 2 valueless flags to `valuelessFlags`, AND one case each in
`ServerConfigArgumentTests` — otherwise Task 7's `unknownFlagError`
rejects them (the original crash class).

## 6. Port plan mapping

| Phase | Branch | Inputs (recovered refs) | Target files (canonical) |
|---|---|---|---|
| 2 (Task 4) | `feature/port-reusable-path` | `820c9f4` | `MLXGenerator.swift`, `RequestMetrics.swift`, `OpenAIRouter.swift` (diag), `ObservabilityTests.swift`, `RadixReusablePathWeightTests.swift` (new), `benchmarks/run_reusable_path_e2e.sh` + prompt (new), `docs/metrics.md` |
| 3 (Tasks 3+5+6) | `feature/port-radix-ssd` | `ab3456c`+`1d7ce2c`+`a7d057d`+`f51631f` | `RadixSSDStore.swift` (new), `RadixKVCacheManager.swift`, `MLXGenerator.swift`, `Qwen38Server.swift`, `ServerConfig.swift` (+5 flags, §5), `ObservabilityTests.swift`, `RadixSSDPersistenceTests.swift` (new), `RadixSSDWeightTests.swift` (new), **fork** `KVCache.swift` (paired change), `benchmarks/run_radix_ssd_restart.sh` + `run_radix_ssd_equilibrate.sh` (new), `docs/radix-ssd-persistence-rfc.md` (new), Task 2 calibration digest-invalidation (per `task2-comparison.md` decision) |
| 4 (Task 1) | `feature/port-task1-artifacts` | `e03339f` branch | `docs/compact-rejection-rfc.md`, `CompactRejectionTests.swift`, `WeightTestLock.swift`, `benchmarks/compact-rejection-prompt.json` + `run_compact_rejection_ab.sh`, `MLXGenerator.swift` (`QWEN_MLX_SEED` + `active_bytes` trace field if absent) |

Weight-gated tests (separate `swift test` invocations, never concurrent):
`RadixReusablePathWeightTests` (Phase 2), `RadixSSDWeightTests` (Phase 3),
`CompactRejectionTests` distributional subset (Phase 4, if weight-gated).
