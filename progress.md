# qwen38-mtp-server — build plan & progress

## Goal
3-layer Qwen 3.8 MTP server: (1) local `../mlx-swift-lm` fork (Qwen38 engine in
`Libraries/MLXLLM/Models`), (2) `MLXFastModel` Metal target, (3) `HTTPServer`
executable. Fixes the token-truncation bug from `qwen-mtp-server`.

## Key facts confirmed
- Fork = `mlx-swift-lm` package; products `MLXLLM`, `MLXLMCommon`. Fork pins
  `mlx-swift` **0.31.6** (`swift-syntax` 602.x, `MLXNN`, `MLXOptimizers`).
- Old server (`/Users/cwong/ai/qwen-mtp-server`) vendors `mlx-swift` **0.32.0** and
  `mlx-swift-lm`. Its `MLXFastModel` target = the engine (Qwen35 base 14 files +
  Qwen36 MTP 4 files + Laguna 6 + plumbing 4). `HTTPServer` = the server.
- `Qwen36MTPBlockSession.swift` (3588 lines) = the engine: inline
  `MLXFast.metalKernel` blocks + Qwen35 DeltaNet/attention hybrid + MTP draft/verify
  + `maybeQuantizeKVCache`. Greedy. Model id `"qwen3.6-27b-mtp"`.
- Fork already has `Qwen35*.swift` base types in `Libraries/MLXLLM/Models` matching
  the old server's references.
- Token-truncation fix (Phase 4): `OpenAIRouter` admission `min(maxTokens,4096)`;
  `MLXGenerator` full `max_tokens`; `SamplingParameters.defaultMaxTokens=16384`;
  90% RAM-pressure halt (MemoryAdmission).

## Dependency decision
- New server `Package.swift`: `mlx-swift-lm` = `.path("../mlx-swift-lm")`; `mlx-swift`
  = `.upToNextMinor(from: "0.31.6")` (matches fork so SwiftPM unifies to 0.31.6);
  `swift-syntax`, `vapor` 4.102.x.
- RISK: old server code was written for mlx-swift 0.32.0; must verify `MLXFast.metalKernel` /
  `MLXFastConstants` / `scaledDotProductAttention` compile against 0.31.6. If not, bump
  fork to 0.32.0 (edit fork Package.swift) or adapt ported code.

## Checkpoints (each independently buildable)
1. [DONE] Manifest + minimal MLXFastModel/HTTPServer skeleton; `swift build
   --target HTTPServer` + `swift test --filter HTTPServerTests` PASS.
   - Confirmed `MLXFast.metalKernel(name:inputNames:outputNames:source:header:
     ensureRowContiguous:atomicOutputs:) -> MLXFast.MLXFastKernel` is identical in
     0.31.6 and 0.32.0. Old-server kernel definitions port as-is. 0.31.6 pin is sound.
2. [IN PROGRESS] Port Qwen36 -> Qwen38 engine into fork `Libraries/MLXLLM/Models`
   (Qwen38MTPBlockSession.swift + Qwen38MTP{HeadAttachment,ReferenceSession,Target}.swift),
   reference fork Qwen35 base, register `"qwen3.8-27b-mtp"`. Build fork.
3. Port old MLXFastModel plumbing (factory/weight-loading/memory-policy) into new
   server `MLXFastModel`; wire `MLXGenerator` to fork Qwen38 session.
4. Port HTTPServer (router/generator/admission) + apply token-truncation fix.
5. Build full `--target HTTPServer`; `git diff --check`; update this file.

## Phase 2: Top-Level Build & Runtime Sanity — DONE (2026-09-10)

- `swift build` (full, all targets incl. tests) in `/Users/cwong/ai/qwen38-mtp-server`
  **PASSES** — no errors; only benign fork warnings ("file ... is part of
  module 'MLXLLM'; ignoring import") in the four Qwen38MTP*.swift files.
- Binary: `.build/debug/qwen38-mtp-server` (132 MB debug).
- `Package.swift` verified: `.package(path: "../mlx-swift-lm")` (line 23).
- `Package.resolved` pins: `mlx-swift` **0.31.6** (matches fork pin, single
  shared MLX), `swift-syntax` 603.0.2, `vapor` 4.122.1. `mlx-swift-lm` is a
  path-local dep (absent from Package.resolved, as expected).
- `swift test --filter HTTPServerTests`: **1 test, 0 failures**
  (`MLXFastProbe.buildProbeKernel()` constructs `MLXFast.metalKernel` against
  0.31.6).
- Runtime dry-run: `qwen38-mtp-server serve --port 18099` starts cleanly
  (Vapor NOTICE "Server started"), `GET /healthz` -> **200 "ok"**, clean kill
  with no deinit assertion. (Note: bare `--port` flag is rejected by Vapor's
  default CLI; the flag must be `serve --port`.)
- Qwen38MTPBlockSession startup-warming test: **NOT yet runnable** — the fork
  now contains `Qwen38MTPBlockSession.swift`, `Qwen38MTPCore.swift`,
  `Qwen38MTPHeadAttachment.swift`, `Qwen38MTPReferenceSession.swift`,
  `Qwen38MTPTarget.swift`, but the top-level server is still the checkpoint-1
  skeleton (no model loading / MLXGenerator wiring; that is checkpoint 3).
  Local weights ARE available for checkpoint 3/4: `/Users/cwong/ai/qwen-mtp-server/weights`
  (14 GB, Qwen3.6-27B-MTP-4bit: 3 safetensors + tokenizer + configs).

## State
- Checkpoint 1 DONE: `Package.swift`, `MLXFastProbe.swift` (public), `Qwen38Server.swift`,
  `SmokeTests.swift`. `swift build --target HTTPServer` + `swift test --filter
  HTTPServerTests` both PASS. `.build` populated (fork + mlx-swift 0.31.6 + Vapor).
- No Qwen38 engine code written yet.

## HANDOFF (checkpoint 2 is Large; stopping before a >Small patch)

### Branch / clean-dirty
- `qwen38-mtp-server`: fresh `main`, zero commits; working tree has the 5 checkpoint-1
  files (uncommitted). Fork `mlx-swift-lm` on `main`, clean except deps resolved.

### Files inspected
- Fork `Package.swift` (products MLXLLM/MLXLMCommon; mlx-swift 0.31.6 pin),
  `Libraries/MLXLLM/Models/{Qwen35,Qwen35MTP,Qwen35MoE}.swift`.
- Old server `Sources/MLXFastModel/{Qwen36MTPBlockSession(2844),Qwen36MTPHeadAttachment(376),
  Qwen36MTPReferenceSession(196),Qwen36MTPTarget(147)}.swift` + `Qwen35*.swift` (14 files),
  `Sources/HTTPServer/{QwenServer,ServerConfig,Models/OpenAIModels,Models/SamplingParameters,
  Routes/OpenAIRouter,Generation/MLXGenerator}.swift`.
- `MLXFast.metalKernel` in both 0.31.6 and 0.32.0 (identical signature).

### Accepted invariants
- New server pins mlx-swift 0.31.6 (matches fork). `MLXFast.metalKernel(
  name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:) ->
  MLXFast.MLXFastKernel`. Kernel string blocks port as-is.
- Qwen38 base = fork `Qwen35.swift` config-driven (64 layers, fullAttentionInterval=4
  => 48 linear + 16 attention). Do NOT rename upstream Qwen35/Qwen36 in place.

### Exact planned files (checkpoint 2, in fork `Libraries/MLXLLM/Models/`)
- `Qwen38MTPBlockSession.swift`  <- port old `Qwen36MTPBlockSession.swift`
- `Qwen38MTPHeadAttachment.swift` <- port old `Qwen36MTPHeadAttachment.swift`
- `Qwen38MTPReferenceSession.swift` <- port old `Qwen36MTPReferenceSession.swift`
- `Qwen38MTPTarget.swift`          <- port old `Qwen36MTPTarget.swift`
- Register `"qwen3.8-27b-mtp"` in `ModelTypeRegistry.swift`.

### BLOCKER (why stopping)
- The Qwen36->Qwen38 rename is a global find/replace over ~3500 lines / ~100
  identifier occurrences across 4 files. The native editor tool does one exact
  old_text->new_text per call (no global replace; 6000-char limit) and sed/awk/
  python edits are forbidden by the project rules. A faithful clone therefore
  cannot be applied as one bounded patch.
- Architecture mismatch to resolve: fork `Qwen35MTP.swift` is a compact
  `StatefulMTPDrafterModel` (MLXLMCommon), whereas old `Qwen36MTPBlockSession`
  is a custom block session with inline Metal kernels + its own draft/verify/KV
  rollback. The ported engine must be verified against the fork's `Qwen35`
  base API (`Qwen35TextModelInner`/`Attention`/`GatedDeltaNet`) which may differ
  from the old server's `Qwen35*` (14 files).

### Build/test commands
- Fork: `cd /Users/cwong/ai/mlx-swift-lm && swift build --target MLXLLM`
- Server: `cd /Users/cwong/ai/qwen38-mtp-server && swift build --target HTTPServer`
  + `swift test --filter HTTPServerTests`

### Unresolved questions (need operator decision)
1. Clone source: port the OLD server's custom `Qwen36MTP*` engine (inline
   kernels, own rollback) OR adapt to the fork's existing `StatefulMTPDrafter`
   machinery? Task text says clone Qwen36 + port inline kernels.
2. Rename mechanics: may I use `cp` to seed each Qwen38 file then apply the
   renames via many editor calls, or is a bulk copy+rename acceptable given the
   sed ban? (This determines whether checkpoint 2 is executable at all.)
3. Does the fork `Qwen35` base expose the exact API the old engine calls?

## State (2026-09-10, branch feature/prompt-resume-entrypoint)

- Checkpoints 2-5 DONE. Fork `../mlx-swift-lm` carries the Qwen38 engine
  (`Libraries/MLXLLM/Models/Qwen38MTP{BlockSession,HeadAttachment,ReferenceSession,Target,Core}.swift`;
  `MLXFastConstants`/`MLXFastError` live in `Qwen38MTPCore.swift` in MLXLLM —
  do NOT create a second `MLXFastCore` module in the server; it is ambiguous
  in any file importing both).
- `MLXFastModel` + `HTTPServer` (Qwen38Server, ServerConfig, Routes, Models,
  Generation, Observability, API) ported from `qwen-mtp-server` with
  Qwen36 -> Qwen38 renames and the token-truncation fix
  (`OpenAIRouter` admission `min(maxTokens,4096)`, full `max_tokens` in
  `MLXGenerator`, `SamplingParameters.defaultMaxTokens=16384`, 90% RAM-pressure
  halt in MemoryAdmission).
- Think/reasoning control tokens are **ASCII** `think` / `nothink` fragments
  (NOT Unicode U+2384/U+2385). Per project rule they are built from
  fragments (`"<" + "think>"`, `"</" + "think>"` etc.) in
  `MLXGenerator.swift` and `StreamingToolCallParser.swift`; no contiguous
  literal exists in source. Those two files contain the fragment chars, so
  edits to them must go through terminal python3 heredocs, not the file
  editor tool.
- `PenaltySamplingTests.swift` needs `import MLXLLM` (`MTPSamplingConfig`
  lives in MLXLLM, not MLXFastModel/HTTPServer).
- Build: `swift build --target HTTPServer` PASS. Tests:
  `swift test --filter HTTPServerTests` PASS — 121 tests in 2 suites, 0
  failures.
- LOCAL TEST ARTIFACT: tests that evaluate on Metal need `default.metallib`
  in the repo CWD (C++ falls back to `METAL_PATH="default.metallib"`;
  SwiftPM bundle paths are not found under `swift test`). Copied from
  `/Users/cwong/ai/qwen-mtp-server/default.metallib` (151 MB, built from the
  same 0.31.x mlx-swift lineage; loads and runs fine). It is untracked and
  gitignored, along with `weights/` and `mtp-head/` (local symlink caches to
  `~/.cache/mlxfast`).
- Unresolved: none blocking. Runtime serve + warmup + generation smoke test
  against the real checkpoint is the next verification step (needs the
  transformed `weights/` tree and ~64 GB RAM).

