#!/usr/bin/env python3
"""B4 TTFT benchmark: multi-turn exact-prefix reuse vs cold-miss control.

Measures time-to-first-streamed-byte (TTFT) for:
  - turn 1 (cold): full prefill of the shared system prompt;
  - turn 2 (HIT): the seed is turn-1's *actual* history + a short user message.
    Because the radix cache's recurrent (gated-delta) layers are not
    trimmable, a hit requires the stored history to be an EXACT token-prefix of
    the new seed; same-thread continuation satisfies that (turn N+1's seed =
    turn N's history + new user message), so only the short tail is prefilled;
  - cold control: the same total length with a DIFFERENT system prompt (no
    reuse), isolating the cache benefit from the longer seed.

The primary TTFT win quantified is turn-2 (hit) vs the cold control of equal
length. This is the mechanism the Phase B hardening (namespace, metrics, byte
cap) protects; it is a benchmark, not a correctness gate.

usage: prefix_ttft.py <port>
"""
import json
import sys
import time
import http.client

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18099


def ttft_stream(messages, max_tokens=8):
    """POST a streaming chat completion; return (ttft_seconds, full_text)."""
    body = {
        "model": "qwen3.8-27b-mtp",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "mtp_enabled": True,
        "stream": True,
    }
    data = json.dumps(body).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=data,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    first = None
    content = ""
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
            delta = j["choices"][0].get("delta", {})
            c = delta.get("content", "") or ""
            # TTFT = time to the first NON-EMPTY content delta (the role-only
            # framing line arrives when the stream opens, before prefill).
            if c and first is None:
                first = time.perf_counter() - t0
            content += c
        except (ValueError, KeyError, IndexError):
            pass
    full = time.perf_counter() - t0
    conn.close()
    return first, full, content


def long_prompt(n_words):
    return " ".join(f"word{i}" for i in range(n_words))


def main():
    system = long_prompt(3000)  # ~4K tokens: makes prefill the dominant cost
    user1 = "Summarize the above in one sentence."
    user2 = "Now add a title."
    user3 = "Give a one-line conclusion."

    # Turn 1 (cold): full prefill of [system, user1].
    ttft1, full1, asst1 = ttft_stream(
        [{"role": "system", "content": system}, {"role": "user", "content": user1}])

    # Turn 2 (HIT if the cache works): reuse turn 1's ACTUAL history as a token
    # prefix, then a short user2 tail. NOTE: the cache only hits if the model's
    # generated tokens re-tokenize bit-exactly when fed back as text; a text-
    # based round-trip that is not bit-exact (special-token / merge boundaries)
    # will miss, so this harness's turn-2 is a *necessary-not-sufficient*
    # probe, not proof. The match-level proof is RadixCacheBenchmarkTests.
    ttft2, full2, _ = ttft_stream([
        {"role": "system", "content": system},
        {"role": "user", "content": user1},
        {"role": "assistant", "content": asst1},
        {"role": "user", "content": user2},
    ])

    # Cold control: same total length, DIFFERENT system prompt -> no reuse.
    ttft_cold, full_cold, _ = ttft_stream([
        {"role": "system", "content": long_prompt(3000) + " CTRL-NO-REUSE"},
        {"role": "user", "content": user3},
    ])

    out = {
        "ttft_turn1_cold_s": round(ttft1, 4),
        "ttft_turn2_hit_s": round(ttft2, 4),
        "ttft_cold_control_s": round(ttft_cold, 4),
        "full_turn1_cold_s": round(full1, 4),
        "full_turn2_hit_s": round(full2, 4),
        "full_cold_control_s": round(full_cold, 4),
        "assistant_reused_chars": len(asst1),
    }
    out["ttft_speedup_vs_cold_control"] = round(ttft_cold / ttft2, 3) if ttft2 else None
    out["ttft_speedup_vs_turn1"] = round(ttft1 / ttft2, 3) if ttft2 else None
    print(json.dumps(out))


if __name__ == "__main__":
    main()
