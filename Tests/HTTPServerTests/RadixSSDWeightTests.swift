// RadixSSDWeightTests.swift
//
// Weight-gated test for the Radix SSD persistence tier. Runs only with
// QWEN_RUN_WEIGHT_TESTS=1 and takes the weight-test lock (so it must run in a
// separate `swift test` invocation from the other weight-gated tests — two
// ~14 GB model loads in parallel would exceed 48 GB unified memory).
//
// Verifies the full tensor round-trip on a REAL in-RAM CacheEntry: persist to
// disk, restore, and confirm the reconstructed KV state is bit-identical.

import Foundation
import Testing
import MLX
@testable import HTTPServer

@Test
func radixSSDPersistRestoreIsBitIdentical() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[RadixSSD] Skipped: set QWEN_RUN_WEIGHT_TESTS=1 to run.")
        return
    }
    guard acquireWeightTestLock() else {
        print("[RadixSSD] Skipped: another weight-gated test is running.")
        return
    }

    let modelPath = env["QWEN_MODEL_PATH"] ?? "./weights"
    let mtpHeadPath = env["QWEN_MTP_HEAD_PATH"] ?? "./mtp-head"
    // A temp dir so the test doesn't clobber the real cache.
    let cacheDir = "/tmp/radix-ssd-weight-test-\(UUID().uuidString)"
    let generator = try await MLXGenerator(
        modelPath: modelPath,
        mtpHeadPath: mtpHeadPath,
        maxDraftDepth: 3,
        kvSSDEnabled: true,
        kvSSDCacheDir: cacheDir,
        kvSSDCacheGB: 1,
        kvSSDTTLSeconds: 86400
    )

    let ok = await generator.testSSDRoundTrip()
    #expect(
        ok,
        "SSD persist/restore produced a non-bit-identical KV state (or a step failed)")
    print("[RadixSSD] PASS: persist/restore bit-identity on a real CacheEntry")
}

// Phase 2.5 integration assertion: a request served from an SSD-restored prefix
// must report REAL reuse (`reusedPrefixTokens > 0`, `radixPrefillSkipped ==
// true`), proving the restored entry drives the in-RAM reusable path rather
// than silently falling back to a full prefill. gen1 is released (scope ends) before
// gen2 is created so the two ~14 GB model loads never overlap.
@Test
func radixSSDRestoreReportsRealReuse() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[RadixSSD] Skipped: set QWEN_RUN_WEIGHT_TESTS=1 to run.")
        return
    }
    guard acquireWeightTestLock() else {
        print("[RadixSSD] Skipped: another weight-gated test is running.")
        return
    }

    let modelPath = env["QWEN_MODEL_PATH"] ?? "./weights"
    let mtpHeadPath = env["QWEN_MTP_HEAD_PATH"] ?? "./mtp-head"
    let cacheDir = "/tmp/radix-ssd-integration-\(UUID().uuidString)"

    var promptText = "Explain the history and design of distributed systems in detail. "
    for i in 0..<60 {
        promptText += "Section \(i): discuss the tradeoffs of consistency, availability, and partition tolerance at depth. "
    }
    let request = ChatCompletionRequest(
        model: "qwen3.8-27b",
        messages: [ChatMessage(role: "user", content: promptText, reasoning: nil, reasoning_content: nil)],
        max_tokens: 4
    )
    func params() -> SamplingParameters {
        SamplingParameters(
            temperature: 0, topP: 1.0, topK: 0, minP: 0,
            repetitionPenalty: 1.0, presencePenalty: 0.0, frequencyPenalty: 0.0,
            maxTokens: 4, contextWindow: 262_144, enableThinking: true,
            mtpEnabled: true, prefillChunkSize: 512, stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default, ttlSeconds: nil
        )
    }

    // Phase 1: cold generation on gen1 stores the prompt boundary in RAM; flush
    // to SSD. gen1 is released at the end of this scope (model freed) so gen2's
    // model load never overlaps it.
    do {
        let gen1 = try await MLXGenerator(
            modelPath: modelPath, mtpHeadPath: mtpHeadPath, maxDraftDepth: 3,
            kvSSDEnabled: true, kvSSDCacheDir: cacheDir, kvSSDCacheGB: 1, kvSSDTTLSeconds: 86400
        )
        let coldStream = await gen1.generateStream(request: request, samplingParams: params())
        for await _ in coldStream {}
        await gen1.flushToSSD()
    }

    // Phase 2: gen2 restores the skeleton from SSD in init. The same prompt must
    // be served from the SSD-restored prefix and report real reuse.
    let gen2 = try await MLXGenerator(
        modelPath: modelPath, mtpHeadPath: mtpHeadPath, maxDraftDepth: 3,
        kvSSDEnabled: true, kvSSDCacheDir: cacheDir, kvSSDCacheGB: 1, kvSSDTTLSeconds: 86400
    )
    let restStream = await gen2.generateStream(request: request, samplingParams: params())
    var matched = 0
    var reused = 0
    var skipped = false
    for await fragment in restStream {
        if case .metrics(let m) = fragment {
            matched = m.matchedPrefixTokens
            reused = m.reusedPrefixTokens
            skipped = m.radixPrefillSkipped
        }
    }
    print("[RadixSSD-integration] SSD-restored: matched=\(matched) reused=\(reused) skipped=\(skipped)")
    #expect(
        reused > 0,
        "SSD-restored request must report reusedPrefixTokens > 0 (got \(reused))")
    #expect(skipped, "SSD-restored request must report radixPrefillSkipped == true")
}
