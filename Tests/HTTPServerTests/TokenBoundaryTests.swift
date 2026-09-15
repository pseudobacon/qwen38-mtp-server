// TokenBoundaryTests.swift
//
// Pins the exact token boundary between a completed turn's exported state
// (`exportState().tokens` = rendered prompt + generated tokens) and the
// re-rendered next-turn prompt (the full message history run through the
// chat template again). This is the load-bearing fact for the session API:
// a session reuses the radix prefix cache only as far as the re-rendered
// prompt shares a token-ID prefix with the stored state.
//
// The two thinking modes behave differently, so both are asserted:
//   * enable_thinking == false: the non-thinking generation prompt is
//     byte-identical to the assistant re-render prefix, so the stored state
//     is an EXACT prefix of the next-turn prompt (full reuse, subject to
//     generated tokens re-tokenizing identically).
//   * enable_thinking == true (default): the re-render wraps the assistant
//     content in `think`/`/think`, so the re-rendered prompt DIVERGES from
//     the stored state at the assistant-content boundary (partial reuse).
//
// Gated behind `QWEN_RUN_WEIGHTS=1`: it loads the tokenizer (no model
// weights) from the weights directory. All comparisons are on integer token
// IDs, never decoded text.

import Testing
import Foundation
import Tokenizers

@Test
func sessionReRenderTokenBoundary() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHTS"] == "1" else {
        print("[TokenBoundary] Skipped: set QWEN_RUN_WEIGHTS=1 to run (needs tokenizer).")
        return
    }
    let weightsDir = URL(fileURLWithPath: env["QWEN_MODEL_PATH"] ?? "./weights")
    let tok = try await Tokenizers.AutoTokenizer.from(modelFolder: weightsDir)

    let systemText = "You are a helpful assistant."
    let user1Text = "What is the weather in London?"
    let asst1Text = "The weather in London is rainy."
    let user2Text = "Should I bring an umbrella?"

    let sys: [String: any Sendable] = ["role": "system", "content": systemText]
    let u1: [String: any Sendable] = ["role": "user", "content": user1Text]
    let a1: [String: any Sendable] = ["role": "assistant", "content": asst1Text]
    let u2: [String: any Sendable] = ["role": "user", "content": user2Text]

    func render(_ messages: [[String: any Sendable]], enableThinking: Bool) throws -> [Int] {
        try tok.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": enableThinking, "add_generation_prompt": true]
        )
    }

    func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var i = 0
        while i < limit, a[i] == b[i] { i += 1 }
        return i
    }

    // The generated tokens are the model's output; we stand in the
    // re-tokenization of the assistant text, which is the worst case the
    // server can observe on the next turn (it only ever sees text -> tokens).
    let A1 = tok.encode(text: asst1Text, addSpecialTokens: false)

    // Non-thinking: stored state must be an exact prefix of the re-render.
    do {
        let P1 = try render([sys, u1], enableThinking: false)
        let S1 = P1 + A1
        let P2 = try render([sys, u1, a1, u2], enableThinking: false)
        let hit = commonPrefix(S1, P2)
        print("[TokenBoundary] non-thinking: P1=\(P1.count) S1=\(S1.count) P2=\(P2.count) commonPrefix=\(hit)")
        #expect(hit == S1.count && P2.count > S1.count,
                "non-thinking: stored state must be an exact prefix of the re-render")
    }

    // Thinking (default): re-render must diverge at the assistant boundary,
    // so reuse is strictly partial — never a full stored-state prefix.
    do {
        let P1 = try render([sys, u1], enableThinking: true)
        let S1 = P1 + A1
        let P2 = try render([sys, u1, a1, u2], enableThinking: true)
        let hit = commonPrefix(S1, P2)
        print("[TokenBoundary] thinking: P1=\(P1.count) S1=\(S1.count) P2=\(P2.count) commonPrefix=\(hit)")
        #expect(hit < S1.count, "thinking: re-render must NOT be a full stored-state prefix")
        // The divergence is at the assistant-content boundary: the re-render
        // matches essentially the whole generation prompt and diverges at the
        // first generated token (within one token of the prompt end).
        #expect(hit >= P1.count - 1, "thinking: divergence must be at the assistant boundary")
    }
}