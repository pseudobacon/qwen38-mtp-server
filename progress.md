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

## Task: Tied-Embedding Weight Sanitization (2026-09-11)

- Added the tied-embedding `lm_head.weight` alias to both sanitizers in the
  fork `Libraries/MLXLLM/Models/Qwen35.swift`:
  - `Qwen35TextModel.sanitize`: after `filterLMHeadWeights`, when
    `lmHead != nil` and `weights["lm_head.weight"] == nil`, alias
    `weights["model.embed_tokens.weight"]` into `lm_head.weight`.
  - `Qwen35Model.sanitize` (LLM wrapper): after the `language_model.` key
    rewrite, when `languageModel.lmHead != nil` and
    `sanitized["language_model.lm_head.weight"] == nil`, alias
    `language_model.model.embed_tokens.weight` into `language_model.lm_head.weight`.
  - The `lmHead != nil` guard matters: with `tie_word_embeddings == true` no
    head module is instantiated, and an aliased `lm_head.weight` would be an
    unhandled key under `model.update(verify: [.all])` (`noUnusedKeys`). An
    unconditional alias (the raw task snippet) would also add a stray bare
    `lm_head.weight` in the wrapper's prefixed namespace.
- Tests: 4 new cases in `Tests/MLXLMTests/Qwen35SanitizeTests.swift` —
  untied text model missing head is aliased; present head preserved; tied
  embeddings produce no alias key; wrapped model aliases in the
  `language_model` namespace (and no bare key). Needed
  `import MLXLLM` + qualified `MLXVLM.Qwen35Configuration` /
  `MLXLLM.Qwen35Configuration` because both modules define `Qwen35Configuration`.
- Fork also needed `default.metallib` in its CWD to run Metal-evaluating tests
  (same loader fallback as the server repo); copied from the server repo,
  gitignored via fork `.gitignore` (`default.metallib`).
- Verification: fork `swift build --target MLXLLM --force-resolved-versions`
  PASS; fork `swift test --filter Qwen35SanitizeTests` PASS — 7 tests, 0
  failures; server `swift build --target HTTPServer` PASS; server
  `swift test --filter HTTPServerTests` PASS (107 XCTest + 121 Swift Testing,
  0 failures).
- Fork state: Qwen35.swift / KVCache.swift / Qwen38MTP* changes remain
  uncommitted (checkpoint-2 port + this task); fork `git diff --check` clean.
- Notes for the smoke test: the pinned backbone checkpoint declares
  `tie_word_embeddings: false` and DOES carry `language_model.lm_head.weight`
  (1,847 tensors), so the alias is a defensive no-op for it. Separate known
  fork gap (not this task): `_additionalWeightSources` /
  `_primaryWeightKeyPrefixStrip` globals are set by
  `Qwen38MTPHeadAttachment.withHeadAttached` but not yet read by the fork's
  upstream `loadWeights` path — the MTP-head merge and `language_model.`
  prefix strip will not actually happen at load until that hook lands.

## Task: Wire MTP weight hooking & run server startup smoke test (2026-09-11)
- Load seam verified: `LLMModelFactory.load` (MLXLLM) → async `loadWeights`
  (`Libraries/MLXLMCommon/Load.swift`) → `loadWeightArrays` → `sanitize` →
  `quantize` → `update(parameters:verify:[.all])`.
- Module-direction fix: the two weight-hook globals were declared in MLXLLM's
  `Qwen38MTPCore.swift`, but the load path lives in MLXLMCommon, which cannot
  import MLXLLM. Moved `AdditionalWeightSource`, `_additionalWeightSources`,
  `_primaryWeightKeyPrefixStrip` (and added `strippingWeightKeyPrefix`, which
  throws on a post-rename key collision) into MLXLMCommon's `Load.swift`;
  `Qwen38MTPCore.swift` keeps only `_qwen35MTPEnabled` (read by
  `Qwen35TextModel.init`). `withHeadAttached` needs no change — MLXLLM imports
  MLXLMCommon, so it sets the globals that `loadWeights` now reads.
- Hook in the sync `loadWeights`, after shard load and BEFORE `sanitize` and
  before the quantization walk (both orderings load-bearing): strip the primary
  `language_model.` prefix first, then merge each additional tree with its
  `keyPrefix` (`mtp.`) via `safetensorWeightURLs` + `loadWeightArrays`. The
  async overload delegates to the sync one, so the server path is covered.
- Build gotcha (important): `swift build --target HTTPServer` on this
  toolchain recompiles module objects but does NOT relink the executable
  product — the 00:27 binary predated the hook (verified with `nm`: no
  `strippingWeightKeyPrefix` symbol) while the fork's object file in the
  server's `.build` was fresh. `swift build --product qwen38-mtp-server`
  relinks; use that (or plain `swift build`) whenever the executable must
  carry new fork code.
- First smoke run (stale binary) failed with
  `Model startup failed: keyNotFound(path: ["mtp", "pre_fc_norm_embedding", "weight"],
  modules: ["Qwen35TextModel", "Qwen35MTPPredictor", "RMSNorm"])` — consistent
  with the merge never having run (head tree verified to carry
  `pre_fc_norm_embedding.weight`).
- Smoke test (fresh binary), real checkpoint via `weights/` + `mtp-head/`
  symlinks: model loaded with MTP head attached, warmup completed,
  `[INFO] Model runtime is ready.`, `/healthz` 200, `/readyz` 200.
  Non-streaming chat completion: 16 completion tokens, `finish_reason: "length"`,
  usage prompt 57 / completion 16. Streaming SSE: chunks emitted correctly.
  Clean SIGTERM shutdown (process gone).
- Verification: fork `swift build --target MLXLLM` PASS; server `swift build
  --target HTTPServer` PASS + `swift build --product qwen38-mtp-server` PASS;
  `git diff --check` clean in both repos.
- Caveats: token-fidelity vs the serial trajectory is NOT verified by this
  smoke test (it proves load + merge + warmup + generation work, not
  correctness). Fork changes remain uncommitted alongside the checkpoint-2
  port.

## Task: Test audit — Qwen35MTPMetalTests GDN checkpoint failures (2026-09-11)
- Failing tests: `testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections`
  and `testQwen35VLMGDNCheckpointMatchesPrefix` (fork
  `Tests/MLXLMTests/Qwen35MTPTests.swift`, suite `Qwen35MTPMetalTests`).
- Root cause (measured, not assumed): the test config uses
  `linear_key_head_dim = 8` → `gatedDeltaUpdate` takes the pure-ops fallback
  (`gatedDeltaOps`), and all weights/inputs are fp32. The failure is NOT a
  tape/restore bug: instrumenting the test showed the restored conv state
  (a slice of the M=2 `in_proj_qkv` matmul) differs from the prefix-only
  cache's conv state (the M=1 matmul) by ~9e-4, while the rec state differs by
  ~2e-5. Direct comparison against an exact CPU fp32 dot product: the M=1
  (GEMV) result is accurate to 1.2e-7, the M=2 (GEMM) result deviates by
  ~8.7e-4 — a Metal GEMM-vs-GEMV accumulation-order artifact on O(1) fp32
  values. The conv-slice math and checkpoint/restore logic are exact.
- Fix: widened ONLY the restored-state-vs-prefix-state comparison tolerance
  in both GDN checkpoint tests from `rtol/atol 1e-5` to `rtol/atol 1e-3` (the
  output-vs-output `1e-5` expectations are untouched and still pass), with a
  comment explaining the GEMM/GEMV caveat. Matches repo convention
  (`Gemma3EncoderAccessTests` uses `atol: 1e-3` for state comparisons).
- Verification: fork `swift test --filter 'Qwen35MTPTests|Qwen35SanitizeTests'`
  PASS (7 XCTest + 18 Swift Testing, 0 failures); server
  `swift test --filter HTTPServerTests` PASS (107 XCTest + 121 Swift Testing,
  0 failures). `git diff --check` clean in both repos.
- Recorded metrics from the post-norm A/B + release runs: A/B acceptance
  ~66.0% vs ~63.2%; release binary TTFT ~0.446 s, ~66% MTP acceptance.
## Task: MTP acceptance divergence audit — live server vs unit test (2026-11-09)

Question: why does the live server show ~49-55% greedy MTP acceptance while
the unit test shows 93.46%?

- Prior-session change (now recorded here): `ServerConfig.temp` default
  changed 1.0 -> 0.0 (greedy default). Live run of the short benchmark prompt
  (T=0, `mtp_enabled: true`, `enable_thinking` left at default, depth 3)
  measured **55.1%** acceptance — confirming the old 1.0->0.0 default theory
  does NOT explain the gap (the client already sends T=0).
- Unit-test baseline re-run on the CURRENT fork
  (`mlx-swift-lm` `Tests/MLXLMTests/Qwen38MTPDiagnosticTests.swift`,
  `swift test --filter Qwen38MTPDiagnosticTests`): aggregate **93.46%**
  (94.35 / 93.07 / 93.00 per prompt). The baseline is still valid on the
  current code.
- Live A/B matrix (release binary, port 18123, per-scenario fresh server
  unless noted; `mean_mtp_acceptance_rate` from `/metrics`):
  - short prompt, thinking ON, depth 3, 128 tok: **55.1%**
  - short prompt, thinking OFF, depth 3, 128 tok: **74.1%**
  - short prompt, thinking OFF, depth 8, 128 tok: **61.9%**
  - short prompt, thinking OFF, depth 8, 64 tok: **70.8%**
  - ~900-token technical prose, thinking OFF, depth 8, 128 tok: **55.6%**
  - same prose, thinking ON, depth 8, 128 tok: **56.6%**
- Root cause (measured, not assumed): the 93.46% is specific to the unit
  test's RAW token-ID context (13-15 tokens starting at `<|im_start|>`, NO
  chat-template role lines, 64 rounds). Server requests carry the full Qwen3
  chat template, whose context makes draft acceptance lower (55-74%).
  `enable_thinking: true` (server default, `MLXGenerator.swift:287`) is a
  second, smaller factor on short prompts (~-19 pts) but not the primary
  one. Draft depth is NOT the gap: deeper offers (8 vs 3) LOWER pooled
  acceptance (deeper draft positions are harder), so the unit test's depth 8
  vs server depth 3 moves the wrong direction. No metric-accounting, KV,
  or sampling bug: same engine, same greedy config, same acceptance
  arithmetic; the contexts differ.
- `draftPolicy`/`specDraftNMax` interaction verified (fork
  `Qwen38MTPBlockSession.swift`): server passes `decodeDepth = mtpEnabled ?
  maxDraftDepth : 0` (`MLXGenerator.swift:639,992`); `generateRound` throws
  `.invalidDepth` outside 0...8; `draftCount = draftPolicy(depth, round)` is
  capped at `min(offered, 8, segmentedVerifyDepthCap=7)` by
  `costModelDepth` and re-enforced by the runtime `precondition`
  (`draftCount <= depth`). `specDraftNMax=3` therefore cannot cause
  out-of-bounds draft evaluations; the fork's policy is adaptive (EMA
  marginal rule), not the constant 2 of the pinned track.
- Task 4 action: the default is now explicitly documented in a comment at
  `MLXGenerator.swift:287` (thinking defaults ON; acceptance metric is
  context/thinking-mode-dependent). The request payload already applies
  `enable_thinking` cleanly; the default was NOT flipped, because thinking-on
  is the reasoning model's normal generation behavior, not a misconfiguration.
- No regression test added: a >=90% acceptance assertion is not a valid
  server-level invariant (acceptance is prompt/context-dependent, measured
  55-74% on server contexts); the unit test already pins the engine-side
  93.46% baseline.
- Verification: server `swift build --target HTTPServer` + full
  `swift test --filter HTTPServerTests` PASS (121 tests, 0 failures);
  `git diff --check` clean.

