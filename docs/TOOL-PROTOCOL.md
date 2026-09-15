# Tool Calling Protocol (client contract)

OpenAI-compatible function calling for `POST /v1/chat/completions`. The server
parses tool calls out of the model's output and returns them in the
OpenAI `tool_calls` shape. **The server never executes a tool** — it only
transports, renders, and parses the call. Execution is the client's job.

Internal contract, limits, and test matrix: `benchmarks/TOOL-CALLING.md`.

## Request fields

| Field | Values | Notes |
|---|---|---|
| `tools` | array of `{type: "function", function: {name, description?, parameters?}}` | `name` must match `[a-zA-Z0-9_-]` (1–64); names must be unique; `parameters`, if present, must be a JSON object; ≤ 128 tools. |
| `tool_choice` | `"auto"` (default) \| `"none"` | `"required"` and named `{"type": "function", "function": {"name": ...}}` are **rejected** with 400 (`unsupported_parameter`): the Qwen 3.8 template cannot force a tool call. |
| `parallel_tool_calls` | `false` \| omit | `true` is **rejected** with 400 (`unsupported_parameter`): the template has no parallel-call mode. |

### Messages

- Assistant tool calls in history: `role: "assistant"` with `tool_calls[]`.
- Tool results: `role: "tool"` (legacy `role: "function"` also accepted) with
  `tool_call_id` (required) and `content`. The `tool_call_id` must reference a
  prior assistant `tool_calls` entry in the same request. Assistant `tool_calls`
  IDs must be unique.

## Response

### Non-streaming

```json
{
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_…",
        "type": "function",
        "function": { "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}" }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

`finish_reason` is `"tool_calls"` when ≥ 1 tool call was produced and the model
stopped normally. If generation is truncated (`"length"`) or hits memory
pressure (`"memory_pressure"`), that reason is preserved instead. `arguments`
is a JSON **string** (OpenAI shape).

### Streaming (SSE)

Each completed tool call is emitted as a chunk whose `choices[0].delta` carries
`tool_calls: [ <the full call> ]`. A final chunk carries
`finish_reason: "tool_calls"` and an empty `delta`.

```json
{ "choices": [{ "index": 0, "delta": { "tool_calls": [ { "id": "call_…", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}" } } ] } } ] }
```

## Errors

All rejections are OpenAI-shaped 400s:

```json
{ "error": { "message": "...", "type": "invalid_request_error", "param": "tool_choice", "code": "unsupported_parameter" } }
```

Rejected conditions (all 400): non-`function` tool type; bad/duplicate tool
names; non-object `parameters`; > 128 tools; `tool_choice` of `required`/named;
`parallel_tool_calls: true`; `role: "tool"` with a missing/empty
`tool_call_id`, an orphaned `tool_call_id` (no matching prior assistant call),
a cross-turn mismatch, or duplicate `tool_call_id`s.

## Malformed model output

If the model emits a tool-call block that is not well-formed XML/JSON, the
server does **not** fabricate a `tool_calls` entry: the block is returned
verbatim as visible `content` (the request still succeeds with
`finish_reason: "stop"`). You should treat `tool_calls` as authoritative only
when present and `finish_reason` is `"tool_calls"`.

## Cancellation

On client disconnect, SSE write failure, explicit cancel, timeout, or shutdown,
generation stops. A tool call only appears once it is fully parsed; a partially
parsed block at cancellation is flushed as content (never a half call).
