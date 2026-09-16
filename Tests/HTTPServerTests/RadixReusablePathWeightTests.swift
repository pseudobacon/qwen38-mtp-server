// RadixReusablePathWeightTests.swift
//
// Weight-gated correctness test for the in-RAM Radix prefix-reuse repair.
//
// A same-prompt repeat must produce token-identical output whether it is
// served cold (full prefill) or warm (prefix reused, prefill skipped). This
// is the load-bearing invariant of the reusable path: the cached prefix must
// be a bit-exact substitute for re-prefilling. Skipped unless
// QWEN_RUN_WEIGHT_TESTS=1 (loads the model; must run in its own `swift test`
// invocation so the two 14 GB model loads never overlap).

import Foundation
import Testing
@testable import HTTPServer

@Test("same-prompt repeat is token-identical with prefix reuse (weight-gated)")
func reusablePathProducesIdenticalOutput() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[reusable-path] Skipped: set QWEN_RUN_WEIGHT_TESTS=1.")
        return
    }
    guard acquireWeightTestLock() else {
        print("[reusable-path] Skipped: another weight-gated test is running.")
        return
    }

    let modelPath = env["QWEN_MODEL_PATH"] ?? "./weights"
    let mtpHeadPath = env["QWEN_MTP_HEAD_PATH"] ?? "./mtp-head"
    let generator = try await MLXGenerator(
        modelPath: modelPath, mtpHeadPath: mtpHeadPath, maxDraftDepth: 2
    )

    // Deterministic ~2K-token prompt. Greedy (temperature=0) so the same-prompt
    // repeat is well-defined and the warm path must match the cold path exactly.
    var promptText = "Explain the history and design of distributed systems in detail. "
    for i in 0..<200 {
        promptText += "Section \(i): discuss the tradeoffs of consistency, availability, and partition tolerance at depth. "
    }

    let request = ChatCompletionRequest(
        model: "qwen3.8-27b",
        messages: [ChatMessage(role: "user", content: promptText, reasoning: nil, reasoning_content: nil)],
        max_tokens: 64
    )

    func params() -> SamplingParameters {
        SamplingParameters(
            temperature: 0, topP: 1.0, topK: 0, minP: 0,
            repetitionPenalty: 1.0, presencePenalty: 0.0, frequencyPenalty: 0.0,
            maxTokens: 64, contextWindow: 262_144, enableThinking: false,
            mtpEnabled: true, prefillChunkSize: 512, stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default, ttlSeconds: nil
        )
    }

    // Cold: no cache yet -> full prefill. Stores the prompt boundary.
    let coldStream = await generator.generateStream(request: request, samplingParams: params())
    var cold = ""
    for await fragment in coldStream {
        switch fragment {
        case .content(let s): cold += s
        case .reasoning(let s): cold += s
        case .toolCall, .finished, .metrics: break
        }
    }
    // Warm: same prompt -> the stored prefix is reused, prefill skipped.
    let warmStream = await generator.generateStream(request: request, samplingParams: params())
    var warm = ""
    for await fragment in warmStream {
        switch fragment {
        case .content(let s): warm += s
        case .reasoning(let s): warm += s
        case .toolCall, .finished, .metrics: break
        }
    }
    print("[reusable-path] cold=\(cold.count) warm=\(warm.count) chars")
    #expect(!cold.isEmpty, "cold output empty")
    #expect(cold == warm, "warm (reused-prefix) output differs from cold: cold='\(cold.prefix(80))' warm='\(warm.prefix(80))'")

    // Keep the generator (and its model) alive until process exit. Deinit of
    // the model's `Qwen35DecoderLayer` at process-exit time runs after MLX's
    // global Metal compiler cache has been torn down and SIGSEGVs in
    // `CompiledFunction.deinit` (MLX runtime global-teardown ordering, not a
    // server bug). Leaking the generator at test-process exit is harmless
    // and keeps the exit code clean.
    RetainedGenerator.shared.generator = generator
}

/// Test-process lifetime holder. `MLXGenerator` is an actor and therefore
/// `Sendable`.
private final class RetainedGenerator: @unchecked Sendable {
    static let shared = RetainedGenerator()
    var generator: MLXGenerator?
}
