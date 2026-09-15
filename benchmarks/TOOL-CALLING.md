# Tool Calling (OpenAI-compatible)

Server-side function calling for the Qwen 3.8 MTP server. The server
**transports, renders, parses, and serializes** tool calls in an
OpenAI-compatible shape. It **never executes** a tool. The model emits a
Qwen-native XML/JSON tool-call block in its completion; the server parses it
out of the token stream and re-emits it as `tool_calls` on the response.

This document is the internal contract: what is implemented, the exactness /
boundary behavior, the test matrix, the limits, and what is **not** claimed.

---

## 1. Wire surface

- `POST /v1/chat/completions`
  - Request: `tools` (array of `{type: "function", function: {name, description?, parameters?}}`),
    `tool_choice` (`"auto"` | `"none"`), `parallel_tool_calls` (`false` | omit).
  - Response (non-streaming): `choices[0].message.tool_calls[]` and
    `choices[0].finish_reason: "tool_calls"`.
  - Response (streaming): SSE chunks with `choices[0].delta.tool_calls[]`
    (each completed call is one delta with the full call) and a final chunk
    with `finish_reason: "tool_calls"`.
- History: assistant `tool_calls` are rendered back into the prompt as the
  Qwen XML block; `role: "tool"` results are rendered from `content`.

The public client-facing contract is in `docs/TOOL-PROTOCOL.md`.

## 2. Rendering (request → prompt tokens)

`MLXGenerator.applyChatTemplate` builds HuggingFace-style message dicts and
passes `tools` (via `toJSONString` → the template's `<tools>` block) and
`tool_choice` resolution to the tokenizer's `applyChatTemplate`. The Qwen
3.8 chat template (`weights/chat_template.jinja`) does the actual rendering:
- tool schemas are injected as a system-prompt `<tools>` JSON block;
- assistant `tool_calls` are rendered as the `工具调用 <function=NAME> <parameter=KEY>…</parameter> </function> 工具调用` block;
- `role: "tool"` results are rendered from `content` (the template ignores
  `tool_call_id`/`name` when rendering results).

`tool_choice: "none"` drops the tool block entirely (the no-tools path).

**Cache identity.** `MLXGenerator.tokenizationCacheKey` captures everything
the template renders: the tool schemas, the `tool_choice` resolution, assistant
`tool_calls`, tool-result `tool_call_id`/`name`, reasoning, and
`enable_thinking`. Two conversations differing in any rendered field produce
different tokenization-cache keys, so a no-tools conversation can never be
served a tool conversation's tokenized prompt (and vice versa). The token-
prefix KV/session cache (`RadixKVCacheManager`) then reuses exact token
prefixes; the rendered tool block is part of that prefix, so tool and no-tool
conversations have disjoint prefixes.

## 3. Parsing (model output → tool calls)

`StreamingToolCallParser` runs on **committed** tokens (after MTP
verification), so speculative decoding does not affect parsing. It is a strict
superset of the reasoning parser: `toolCallsEnabled: false` delegates to
`SimpleStreamingReasoningParser` and is byte-identical (the no-tool pipeline
is unchanged).

- Dialects (dispatched on the block's first non-whitespace char):
  - **XML** (primary): `<function=NAME>` then zero or more `<parameter=KEY>VALUE</parameter>`.
  - **JSON** (secondary): `{"name": ..., "arguments": {...}}`.
- A parameter value is placed on its own line by the template
  (`<parameter=K>\nVALUE\n</parameter>`); the parser strips exactly one
  formatting newline on each side, preserving internal newlines (multi-line
  values).
- `serializeArguments` preserves key order and emits values that are already
  valid JSON literals (number/bool/null/array/object) as-is; other values are
  JSON-encoded as strings.
- **Malformed / unterminated block** → emitted verbatim as visible `content`
  (open tag + block + close tag, or the partial block at `finish()`). Never
  throws, never crashes. A defensive iteration cap flushes the buffer as
  content on pathological input.
- Multiple blocks get an incrementing `index` from a per-generation counter.
- A retained suffix of `max(tool-tag, think-tag).count - 1` chars means a tag
  split across token boundaries is never emitted prematurely.

## 4. `finish_reason`

`toolCallFinishReason(toolCallCount:finishedReason:)` (shared by streaming and
non-streaming): when ≥ 1 tool call was produced **and** the model terminated
normally (`"stop"`), the reason is `"tool_calls"`; truncation reasons
(`"length"`, `"memory_pressure"`) are preserved. A truncated tool call is
flushed as content, so a well-formed tool response always ends in a stop token
→ `"tool_calls"`.

## 5. Validation (before model execution)

`ChatCompletionRequestValidator` rejects, with an OpenAI-shaped 400
(`{"error": {message, type, param, code}}`), before any model execution:
- `tools`: `type == "function"`; name matches `[a-zA-Z0-9_-]` (1–64); names
  unique; `parameters`, if present, a JSON object; ≤ 128 tools.
- `tool_choice`: only `"auto"` (omit) and `"none"` are accepted. `"required"`
  and named selection are **rejected** (`unsupported_parameter`) because the
  Qwen 3.8 template has no mechanism to force a tool call — they are not
  silently treated as `auto`.
- `parallel_tool_calls`: only `false`/omit accepted; `true` is rejected
  (`unsupported_parameter`) because the template has no parallel-call mode.
- Conversation sequencing: every `role: "tool"`/`"function"` message has a
  non-empty `tool_call_id` matching a prior assistant `tool_calls` entry
  (no orphaned tool results, no cross-turn mismatch); assistant `tool_calls`
  IDs are unique within and across messages.

## 6. Test matrix (all pure-Swift, no model weights)

| File | Coverage |
|---|---|
| `StreamingToolCallParserTests` | well-formed XML/JSON, empty args, multi-call index, reasoning separation, malformed→content, unterminated→content, tag-split safety, disabled passthrough, JSON-literal arg preservation |
| `ToolCallingValidationTests` | tool schema (type/name/unique/parameters/size), `tool_choice` required/named rejection, `parallel_tool_calls` true rejection, orphaned tool result, missing tool_call_id, duplicate IDs, cross-turn mismatch |
| `ToolCallingRenderTests` | `tokenizationCacheKey` identity: tools vs no-tools, tool schemas, tool_choice, assistant tool_calls, tool_call_id, reasoning, enable_thinking, stability |
| `ToolCallingHTTPTests` | `toolCallFinishReason`: tool_calls on stop, preserved length/memory_pressure, no-tool stop stays stop |
| `ToolCallingCacheTests` | tools vs no-tools disjoint prefixes, multi-turn reuse, branch fork, byte-cap eviction |
| `ToolCallingSerializationTests` | (pre-existing) `delta.tool_calls` and `message.tool_calls` Codable shapes, empty omission, `toJSONString` stability, `ToolChoice` round-trip |

## 7. Limits

- ≤ 128 tool definitions per request; tool name ≤ 64 chars, `[a-zA-Z0-9_-]`.
- `n` must be 1 (server-wide, pre-existing).
- The parser's defensive iteration cap is 100,000 per `parse` call (pathological
  input flushes as content; it is not a user-facing limit).
- No streaming of *partial* tool calls: only fully-parsed calls are emitted.

## 8. What this contract does **not** guarantee

- **No tool execution.** The server never runs a tool; it only parses and
  returns the call. Execution is the client's responsibility.
- **No forced tool selection.** `tool_choice: "required"`/named are rejected,
  not emulated.
- **No guaranteed parallelism.** Multiple tool calls in one assistant message
  are parsed when the model emits them, but `parallel_tool_calls` is not a
  server toggle; `true` is rejected.
- **No claim that the model will always emit well-formed calls.** Malformed
  output degrades to content; correctness of the *parse* is guaranteed, not
  the model's adherence to the format.
- **Reasoning-tag note (pre-existing, out of scope).** The parser's reasoning
  tags are `think`/`/think`; the template uses a different marker. This affects
  the reasoning path identically with and without tools and is not changed
  here.

## 9. Reproduction

```bash
# Parser, validation, render-identity, finish-reason, and cache tests (no model).
swift test --filter "StreamingToolCallParserTests"
swift test --filter "ToolCallingValidationTests"
swift test --filter "ToolCallingRenderTests"
swift test --filter "ToolCallingHTTPTests"
swift test --filter "ToolCallingCacheTests"
swift test --filter "ToolCallingSerializationTests"

# Full server suite.
swift test --filter HTTPServerTests
```
