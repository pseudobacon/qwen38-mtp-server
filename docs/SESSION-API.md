# Session API

Server-owned conversation sessions for `qwen38-mtp-server`. Sessions give a
client **server-side ownership of the conversation history** so multi-turn
clients do not re-send prior turns, and pair with the `include_token_ids`
diagnostic so a client that *does* manage its own prefix can reuse the exact
token IDs.

> **Read this first.** A session guarantees **conversation ownership and
> history continuity**. It does **not** guarantee a KV-cache hit. Cache reuse
> is opportunistic and mode-dependent — see
> [`SESSION-BOUNDARIES.md`](./SESSION-BOUNDARIES.md) for the load-bearing
> token-boundary analysis and the integer-token evidence. Do not treat a
> session as a TTFT win; treat it as a clean multi-turn primitive.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST`   | `/v1/sessions` | Create an empty session |
| `GET`    | `/v1/sessions/{id}` | Inspect a session (metadata + token count) |
| `DELETE` | `/v1/sessions/{id}` | Delete a session and release its cached prefix |
| `POST`   | `/v1/sessions/{id}/completions` | Generate one turn within a session |

All session endpoints are **process-local and in-memory**: no disk persistence,
no cross-process sharing, no authentication. Sessions are bounded by an LRU cap
(`--max-sessions`, default 128) and an idle TTL (`--session-ttl`, default 1800 s)
and are lost on process restart.

### `POST /v1/sessions`

Create an empty session. Returns `201 Created` and a `SessionObject`:

```json
{
  "id": "0F8...A3C",
  "object": "session",
  "created": 1726291200,
  "model": "",
  "token_count": 0
}
```

### `GET /v1/sessions/{id}`

Return the `SessionObject` for a live session, or `404`:

```json
{ "error": { "message": "Session not found", "type": "invalid_request_error", "param": null, "code": "session_not_found" } }
```

`token_count` is `prompt_tokens + completion_tokens` of the last committed turn.
The full message history is **not** exposed (it is re-rendered server-side and
is not token-ID-stable to round-trip); the session is a history *owner*, not a
history *reader*.

### `DELETE /v1/sessions/{id}`

Delete the session and best-effort release its radix-cache prefix (a leaf
removal that never frees a node shared by another entry). Returns `200`:

```json
{ "deleted": true }
```

### `POST /v1/sessions/{id}/completions`

Generate one turn. The request body is a normal chat-completion request whose
`messages` are the **new turn's messages** (the user turn, and any tool-result
turns); they are appended to the stored history, the full conversation is
rendered, and the assistant reply is committed back to the session on success.

```json
{
  "model": "qwen3.8-27b-mtp",
  "messages": [ { "role": "user", "content": "Should I bring an umbrella?" } ],
  "include_token_ids": true
}
```

The response is the standard chat-completion response with two session
extensions:

- `session_id` — echoes the session this completion belongs to.
- `choices[0].message.token_ids` — present only when `include_token_ids` is
  `true`; the exact committed target token IDs for the turn (diagnostic).

Streaming is supported (`"stream": true`) with the same SSE framing as the
stateless endpoint; the final `delta` carries `token_ids` when requested.

**Concurrency:** one in-flight completion per session. A second completion on a
busy session returns `409`:

```json
{ "error": { "message": "Session already has an in-flight completion", "type": "invalid_request_error", "param": null, "code": "session_busy" } }
```

On completion failure (SSE write failure, cancellation, validation after
admission), the in-flight flag is cleared and **nothing** is committed to the
session history.

## `include_token_ids` (stateless endpoint too)

The stateless `POST /v1/chat/completions` also accepts `include_token_ids:
true` and returns `choices[0].message.token_ids` (non-streaming) or the final
delta's `token_ids` (streaming). It is a **diagnostic** of the completion's
token IDs — the same committed IDs the session would store. It is **not** a
request-side splice input: the request message schema is unchanged and the
server does not accept `token_ids` in request messages.

## Example (curl)

```bash
# 1. Create a session.
ID=$(curl -s -X POST http://localhost:18099/v1/sessions | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

# 2. First turn.
curl -s -X POST "http://localhost:18099/v1/sessions/$ID/completions" \
  -H 'Content-Type: application/json' \
  -d '{ "model": "qwen3.8-27b-mtp",
        "messages": [ { "role": "user", "content": "What is the weather in London?" } ],
        "include_token_ids": true }' \
  | python3 -m json.tool

# 3. Second turn — the prior turn is owned by the session; send only the new user turn.
curl -s -X POST "http://localhost:18099/v1/sessions/$ID/completions" \
  -H 'Content-Type: application/json' \
  -d '{ "model": "qwen3.8-27b-mtp",
        "messages": [ { "role": "user", "content": "Should I bring an umbrella?" } ],
        "include_token_ids": true }' \
  | python3 -m json.tool

# 4. Inspect and delete.
curl -s "http://localhost:18099/v1/sessions/$ID" | python3 -m json.tool
curl -s -X DELETE "http://localhost:18099/v1/sessions/$ID"
```

## Example (Python)

```python
import requests

BASE = "http://localhost:18099"
MODEL = "qwen3.8-27b-mtp"

sess = requests.post(f"{BASE}/v1/sessions").json()
sid = sess["id"]

def turn(text):
    r = requests.post(f"{BASE}/v1/sessions/{sid}/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "include_token_ids": True,
    })
    body = r.json()
    return body["choices"][0]["message"]["content"], body["choices"][0]["message"].get("token_ids")

print(turn("What is the weather in London?"))
print(turn("Should I bring an umbrella?"))

requests.delete(f"{BASE}/v1/sessions/{sid}")
```

## What a session does and does not do

- **Does:** own the conversation history; let a client send only the new turn;
  commit the assistant reply on success; report the honest cache-hit signal the
  stateless path produces; expose `include_token_ids`.
- **Does not:** guarantee a cache hit (reuse is opportunistic and mode-dependent,
  see `SESSION-BOUNDARIES.md`); accept request-side `token_ids`; persist to
  disk; batch across requests; share across processes.
