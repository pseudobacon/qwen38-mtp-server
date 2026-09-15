# HANDOFF — OpenAI-compatible tool calling (Phase C) (COMPLETE, merged to main)

> **Checkpoint status.** The fresh-checkpoint procedure **completed**:
> `agent-checkpoint.sh` ran successfully in both repositories (exit 0) and
> wrote `.dsh/last-agent-checkpoint` in each before this file was finalized.
> Markers in the Checkpoint markers section below.

## Objective and acceptance criteria

Server-side OpenAI-compatible **function calling** for the Qwen 3.8 MTP server.
The server must **transport, render, and parse** tool calls and **NEVER execute**
them. Acceptance: request validation before model execution; `tool_choice`
`required`/named and `parallel_tool_calls: true` rejected with 400 (not
silently treated as `auto`); `finish_reason: "tool_calls"` on normal stop;
malformed tool blocks degrade to content (never throw/crash); tokenization
and KV caches keyed on tool content; pure-Swift tests for render/parse/
serialize/validate; docs; full `HTTPServerTests` green.

## Result

**Phase C (COMPLETE, merged).** Phases A (MTP exactness) and B (RAM prefix
cache) were already complete and merged to `main` before this task (server
`ea89536`). This task adds Phase C; all changes are in the server repo only —
the engine (`mlx-swift-lm`) is **untouched** (no engine commit needed).

- **`finish_reason: "tool_calls"`** via a shared `toolCallFinishReason` helper in
  `OpenAIRouter.swift`, used by both streaming and non-streaming paths.
  Streaming now tracks emitted `toolCalls` (the `.toolCall` case previously
  dropped them). Truncation reasons (`length`, `memory_pressure`) are
  preserved.
- **Rejections (400 `unsupported_parameter`):** `tool_choice: "required"` and
  named `{function:{name}}` (the Qwen template cannot force a tool call);
  `parallel_tool_calls: true` (no server-controlled parallel mode). `auto`/
  `none` and `false`/omit accepted.
- **Validation before model execution:** `validateTools` (type `function`, name
  `[a-zA-Z0-9_-]` 1–64, unique names, `parameters` a JSON object, ≤ 128 tools),
  `validateToolChoice`, `validateParallelToolCalls`, `validateToolConversation`
  (every `tool`/`function` result references a prior assistant `tool_call` id —
  no orphans, no cross-turn mismatch; assistant ids unique).
- **Tokenization cache key** (`MLXGenerator.tokenizationCacheKey`, extracted
  static) now includes assistant `tool_calls` (name|args) and tool-result
  `tool_call_id`/`name` — closes a cross-conversation contamination gap.
- **XML parameter newline fix** (`StreamingToolCallParser.parseXMLBlock`): the
  template places each value on its own line; strip exactly one formatting
  newline per side, preserving internal newlines.
- **Docs:** `benchmarks/TOOL-CALLING.md` (internal contract),
  `docs/TOOL-PROTOCOL.md` (public client contract).

### Tests (pure Swift, no model weights; 43 new)

`StreamingToolCallParserTests` (11), `ToolCallingValidationTests` (17),
`ToolCallingRenderTests` (7), `ToolCallingHTTPTests` (3), `ToolCallingCacheTests`
(4), plus 3 `RequestValidationTests` updated for the new orphaned-result rule.
Full **`HTTPServerTests`: 168 green** (baseline 125 + 43).

## Git state

- `qwen38-mtp-server` (this repo): branch `main`, HEAD `a781bad` (Phase C,
  fast-forward merge of `feature/prompt-tool-calling`, branch deleted). Working
  tree clean.
- `../mlx-swift-lm`: branch `main`, HEAD `38f2bd2`. Untouched by this task.

## Commands / verification

```
swift build --target HTTPServer          # clean
swift test --filter HTTPServerTests      # 168 green
# focused:
swift test --filter StreamingToolCallParserTests   # 11
swift test --filter ToolCallingValidationTests     # 17
swift test --filter ToolCallingRenderTests         # 7
swift test --filter ToolCallingHTTPTests           # 3
swift test --filter ToolCallingCacheTests          # 4
```

## Unresolved risks / caveats

- No tool execution (server parses/returns only); execution is the client's job.
- `tool_choice` `required`/named are rejected, not emulated (no template
  mechanism to force a tool call).
- Parse correctness is guaranteed; model adherence to the tool format is not
  (malformed output degrades to visible content, request still succeeds).
- Reasoning-tag marker mismatch (parser `think`/`/think` vs the template
  marker) is pre-existing and unchanged; it affects the reasoning path
  identically with and without tools.

## Do-not-repeat

- Never execute a tool from the server; transport/render/parse only.
- Never silently treat an unrepresentable `tool_choice`/`parallel_tool_calls`
  as `auto` — reject with 400.
- Never swallow malformed tool blocks — emit them verbatim as content.
- Keep tool content in both the tokenization cache key and the KV/session cache
  key so conversations never cross-contaminate.
- Do not run `head -N` on checkpoint output (SIGPIPE aborts before the marker
  write); redirect to a file.

## Next step

None — Phase C is complete, tested, documented, merged to `main`, and
checkpointed. A successor session should independently verify (per the resume
procedure): `git status --short` (clean), `swift test --filter HTTPServerTests`
(168 green), and confirm `docs/TOOL-PROTOCOL.md` + `benchmarks/TOOL-CALLING.md`
exist. If new tool-calling scope is requested (e.g. an end-to-end model-in-
the-loop tool-call harness), that is a new task.

## Checkpoint markers

- server: 2026-09-15T09:40:18+01:00
- engine: 2026-09-15T09:40:18+01:00

The fresh-checkpoint procedure completed.