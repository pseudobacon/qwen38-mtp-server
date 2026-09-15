# Tool Calling Protocol (client contract)

OpenAI-compatible function calling for `POST /v1/chat/completions`. The server
parses tool calls out of the model's output and returns them in the
OpenAI `tool_calls` shape. **The server never executes a tool** — it only
transports, renders, and parses the call. Execution is the client's job.

Internal contract, limits, and test matrix: `benchmarks/TOOL-CALLING.md`.

Tool calls work identically within the server-owned [session API](./SESSION-API.md)
(`POST /v1/sessions/{id}/completions`): assistant tool calls and `role: "tool"
results` are stored in the session history and re-rendered on the next turn, and
the server still never executes a tool.

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

## Manual smoke test (curl)

Copy-pasteable end-to-end sequence against a local server. Re-run it after any
change to chat-template rendering, SSE framing, `StreamingToolCallParser`,
cache/session restore, OpenAI request validation, or a model/tokenizer upgrade.
The server **does not execute** `get_weather`; the client supplies the mock
result.

### 1. Server and identifiers

```bash
BASE=http://127.0.0.1:18199/v1
MODEL=qwen3.8-27b-mtp
```

### 2. Non-streaming tool-call proposal

```bash
cat >/tmp/tool-request.json <<'JSON'
{
  "model": "qwen3.8-27b-mtp",
  "stream": false,
  "temperature": 0,
  "messages": [
    { "role": "system", "content": "You are a concise assistant. When asked for a fictional weather report, call the get_weather tool. Do not invent tool results." },
    { "role": "user", "content": "What is the weather in London?" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Return the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string", "description": "City name" },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
JSON

curl -sS "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/tool-request.json | tee /tmp/tool-response.json | jq .
```

Expected:

- `choices[0].finish_reason == "tool_calls"`
- `choices[0].message.tool_calls` is nonempty
- each call has `id` with a `call_` prefix, `type: "function"`, `function.name`
  matching a declared tool, and `function.arguments` containing valid JSON (a
  JSON string)
- `message.content` may be null or framing whitespace — it is not the primary
  signal
- `reasoning_content` may be present if thinking is enabled
- the server does **not** execute the tool

### 3. Tool-result continuation

```bash
CALL_ID=$(jq -r '.choices[0].message.tool_calls[0].id' /tmp/tool-response.json)
TOOL_NAME=$(jq -r '.choices[0].message.tool_calls[0].function.name' /tmp/tool-response.json)
TOOL_ARGS=$(jq -c '.choices[0].message.tool_calls[0].function.arguments' /tmp/tool-response.json)

cat >/tmp/tool-result-request.json <<JSON
{
  "model": "qwen3.8-27b-mtp",
  "stream": false,
  "temperature": 0,
  "messages": [
    { "role": "system", "content": "You are a concise assistant. When asked for a fictional weather report, call the get_weather tool. Do not invent tool results." },
    { "role": "user", "content": "What is the weather in London?" },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "$CALL_ID",
          "type": "function",
          "function": { "name": "$TOOL_NAME", "arguments": $TOOL_ARGS }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "$CALL_ID",
      "content": "{\"city\":\"London\",\"temperature_c\":17,\"condition\":\"light rain\"}"
    }
  ]
}
JSON

curl -sS "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/tool-result-request.json \
  | tee /tmp/tool-final-response.json | jq .
```

Expected:

- `choices[0].finish_reason == "stop"`
- no new `tool_calls`
- `message.content` summarizes the supplied tool result (e.g. "17°C with light
  rain in London") and does **not** claim the server fetched weather itself

### 4. Streaming tool-call deltas

```bash
jq '.stream = true' /tmp/tool-request.json >/tmp/tool-stream-request.json

curl -N -sS "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @/tmp/tool-stream-request.json \
  | tee /tmp/tool-stream.sse

grep -E '"tool_calls"|"finish_reason"|\[DONE\]' /tmp/tool-stream.sse
```

Expected SSE sequence:

- an initial `role: "assistant"` delta
- one or more `delta.tool_calls` events; the first carries the complete call
  — `id`, `type`, `function.name`, and `function.arguments` as complete JSON
- a terminal event with `"finish_reason":"tool_calls"`
- a final `data: [DONE]`

> **Streaming behavior.** This server emits the complete tool call in a single
> `delta.tool_calls` event. Client integrations should not assume
> `function.arguments` arrive as one-character fragments; for broad OpenAI
> compatibility they should accept both a single complete delta and accumulated
> argument fragments.

### 5. Negative cases (expected HTTP 400)

The Qwen-native XML tool format cannot faithfully represent forced or parallel
tool selection, so these are intentionally rejected with 400
(`unsupported_parameter`):

- `tool_choice: "required"`
- named forced tool selection (`tool_choice: {"type":"function","function":{"name": ...}}`)
- `parallel_tool_calls: true`

```bash
jq '.tool_choice = "required"' /tmp/tool-request.json \
  | curl -sS "$BASE/chat/completions" -H 'Content-Type: application/json' -d @- | jq .

jq '.parallel_tool_calls = true' /tmp/tool-request.json \
  | curl -sS "$BASE/chat/completions" -H 'Content-Type: application/json' -d @- | jq .
```
