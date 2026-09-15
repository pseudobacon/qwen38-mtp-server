# Session token/state boundaries (design note)

Status: **pre-implementation decision note.** This resolves the load-bearing
question for the session API: *when a session re-renders the next-turn prompt,
how far does it share a token-ID prefix with the stored state from the previous
turn?* The answer determines what a session can honestly promise about cache
reuse.

## TL;DR (go/no-go)

- **NO-GO on "exact session-state reuse" as a guaranteed mechanism.** The engine
  *does* support resuming from an exact evaluated state (KV reuse via
  `Qwen38MTPBlockSession.begin(seedTokens:prefixCount:reusableCache:...)`), and the
  radix cache *does* reuse it — but **only as far as the re-rendered next-turn
  prompt shares a token-ID prefix with the stored state.**
- That prefix is **full in `enable_thinking == false`** mode and **partial in
  `enable_thinking == true`** (the default) mode. It is **mode-dependent** and is
  therefore **not** a property the session can guarantee unconditionally.
- **No token splicing.** We will NOT hand-construct user/assistant/tool suffix
tokens in Swift, and we will NOT duplicate the Jinja chat template, to force a
  full hit in thinking mode. That is forbidden and fragile (see §5).
- **Session semantics:** a session guarantees **conversation ownership and
  history continuity**. It does **not** guarantee a cache hit. Cache reuse is
  opportunistic and identical in kind to what a token-faithful stateless client
  already gets by re-sending the full history. `include_token_ids` is a
  **diagnostic** of the completion's token IDs; it is **not** a request-side
  splice input.

## 1. The three objects (and why they differ)

For a turn `N` with message history `H_N = [system, user1, asst1, …, userN]`:

1. **Rendered prompt** `P_N = render(H_N, add_generation_prompt=true)` =
   `M_N + OPEN`, where `M_N` is the re-rendered history through `userN` and
   `OPEN` is the assistant generation-prompt opener. This is the `seedTokens`
   fed to `begin()`.
2. **Generated tokens** `A_N` = the model's committed output for turn `N`.
3. **Stored state** `S_N = exportState().tokens = P_N + A_N`. Confirmed from the
   engine: `begin()` calls `recordTokenHistory(seedTokens)` (the full prompt) and
   `generateRound` appends committed tokens, so `tokenHistory` — and therefore
   `exportState().tokens` — is **prompt + generated**.

On the **next** turn the session re-renders the *completed* history
`H_N + [asstN] + [userN+1]`:

```
P_{N+1} = render(H_N + [asstN] + [userN+1], add_generation_prompt=true)
        = M_N + WRAP(asstN) + USER(N+1) + OPEN
```

where `WRAP(asstN)` is the assistant *message* rendering (role header, optional
`think`/`/think`, content, optional tool calls, closing tag). The question is
whether `P_{N+1}` starts with `S_N = P_N + A_N = M_N + OPEN + A_N`.

## 2. Integer-token evidence (real tokenizer, no model)

Captured by `TokenBoundaryTests` (`QWEN_RUN_WEIGHTS=1`), which loads the real
`weights/tokenizer.json` and renders through the same `applyChatTemplate` path
the server uses. Fixture: system + "What is the weather in London?" +
assistant "The weather in London is rainy." + "Should I bring an umbrella?".
All comparisons are on integer token IDs.

```
non-thinking: P1=30 S1=37 P2=57 commonPrefix=37   -> S1 is an EXACT prefix of P2
thinking:     P1=66 S73  P2=93 commonPrefix=65   -> FIRST MISMATCH at 65
```

### Why non-thinking is a full prefix
The non-thinking generation prompt is:

```

</think>





```

The assistant *message* re-render (Qwen template, `preserve_thinking` undefined
so the `think` form is taken with empty reasoning) produces the **identical**
leading tokens `\n</think>





` before the content. So `OPEN + A_N` (stored) equals the
re-render's `\n</think>





 + A_N` prefix, and `S_N` is a full prefix of
`P_{N+1}`.

### Why thinking diverges (default mode)
- Stored: `…\n</think>
\n + A_N` (the thinking `OPEN` is just `\n</think>
\n`,
  immediately followed by the raw generated content `A_N`).
- Re-render: `…\n</think>





 + A_N` (the assistant message wraps its
  content in `\n





`).

They share `…\n</think>
\n` and then **diverge**: the stored state has the raw
content token next, the re-render has the `\n

` closer/opener. First mismatch
at the assistant-content boundary (index 65 in the fixture).

### The residual (both modes)
Even in non-thinking mode the full prefix holds **only if the generated tokens
re-tokenize identically** to their decoded text. `A_N` is model output; the
server only ever sees `decode(A_N)` and re-encodes it. SentencePiece/BPE
decoding is not always the exact inverse of greedy encoding, so a non-thinking
hit is *full when `A_N` re-tokenizes identically* (the common case for clean
text) and *partial at the first re-tokenization divergence* otherwise.

## 3. What the engine supports (required action #4)

- **Resume from an exact evaluated state:** YES. `begin(seedTokens:prefixCount:
  reusableCache:reusableHidden:reusablePrimary:reusableTop2:)` reuses the KV
  state for `[0..<prefixCount]` and prefills only the suffix. This is exactly
  what the radix cache drives today.
- **Append a newly rendered suffix:** only in the sense that the suffix is the
  *tail of the full re-rendered prompt*. There is no "take this exact evaluated
  state and append these raw token IDs" primitive that is decoupled from the
  chat-template rendering — the resume key is a *token prefix of a rendered
  prompt*.
- **Evaluate a token-ID prefix + token-ID suffix (hand-built):** the model
  forward would accept arbitrary token IDs, but producing a *correct* suffix
  requires the template's per-message structure. Hand-building it in Swift
  duplicates the Jinja template and is the fragility we refuse to take.
- **Clone/restore KV/GDN/MTP/sampler/rollback state:** out of scope and not
  required; KV reuse via the radix cache is the supported resume path.

**Consequence:** the only *supported, non-fragile* reuse is the existing
radix-cache path driven by the re-rendered prompt. A session must use ordinary
full rendering and let the radix cache do as much as it can.

## 4. Session API semantics (the redesign)

- `session_id` **guarantees conversation ownership and history continuity**:
  the server stores the exact message history and the client need not re-send
  prior turns. This removes client-side drift and the requirement that the
  client reproduce the assistant turn.
- `session_id` **does NOT guarantee a cache hit.** Cache reuse is opportunistic
  and mode-dependent (§2). The session reports the same honest cache-hit signal
  the stateless path produces. A session's stored history stays valid even if
  the radix entry is evicted; the next completion simply recomputes the prefix.
- `include_token_ids` returns the **completion** token IDs for the turn — a
  diagnostic/optional extension for clients that want exact bookkeeping.
- **Token IDs are not accepted as a request-side splice mechanism.** We do not
  change the request message schema to accept `token_ids`, and we do not splice
  stored assistant tokens into the next prompt. (Future work may add this only
  with a separately proven token-equivalent renderer, and it is explicitly
  excluded here.)

### Honest expectation-setting
A session provides the *same* radix-cache reuse a token-faithful stateless
client already gets (full in non-thinking, partial in thinking). Its practical
value is **server-side ownership of the exact history** (no client drift, no
re-sending, one clean `session_id` per conversation) plus **`include_token_ids`**
— not a guaranteed cache-hit rate improvement. We document this plainly and do
not market a TTFT win.

## 5. What we will NOT do

- No token splicing of stored assistant tokens into the next prompt.
- No duplication of the external Jinja chat template in Swift; no hand-built
  user/assistant/tool suffix tokens.
- No request-side `token_ids` splice input.
- No claim that a session guarantees a cache hit.

## 6. Regression coverage

`Tests/HTTPServerTests/TokenBoundaryTests.swift` (`QWEN_RUN_WEIGHTS=1`) pins the
boundary on integer token IDs:
- non-thinking: stored state is an exact prefix of the re-render;
- thinking: the re-render diverges at the assistant boundary (strictly partial,
  matching within one token of the prompt end).
