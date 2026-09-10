// MTPCorrectnessHarness.swift
//
// Deterministic MTP correctness harness: captures raw token IDs from serial
// vs. MTP decode, compares them, and gates weight-dependent tests behind
// an env var. Pure unit tests (token comparison logic) run without model
// weights during standard `swift test`.

import Foundation
import Testing
@testable import HTTPServer

// MARK: - Prompt Fixtures

/// Standard test prompt fixtures covering the six required categories.
struct MTPPromptFixture: Sendable {
    let name: String
    let messages: [ChatMessage]
    let maxTokens: Int

    static let all: [MTPPromptFixture] = [
        .ordinaryChat,
        .shortResponse,
        .codeWhitespace,
        .eosStopToken,
        .contextLimit,
        .lowAcceptance,
    ]

    /// Ordinary chat prompt: a general knowledge question.
    static let ordinaryChat = MTPPromptFixture(
        name: "ordinary-chat",
        messages: [
            ChatMessage(role: "user", content: "Explain what a speculative decoding system does in one paragraph.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 128
    )

    /// Short response prompt: expects a very brief answer.
    static let shortResponse = MTPPromptFixture(
        name: "short-response",
        messages: [
            ChatMessage(role: "user", content: "What is 2 + 2? Answer with just the number.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 16
    )

    /// Code/whitespace-heavy prompt: generates code with indentation.
    static let codeWhitespace = MTPPromptFixture(
        name: "code-whitespace",
        messages: [
            ChatMessage(role: "user", content: "Write a Swift function that reverses a string. Include the function signature and body.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 128
    )

    /// EOS/stop-token triggering prompt: should hit a stop token early.
    static let eosStopToken = MTPPromptFixture(
        name: "eos-stop-token",
        messages: [
            ChatMessage(role: "user", content: "Say exactly 'done' and stop.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 32
    )

    /// Context limit prompt: near context window boundary.
    static let contextLimit = MTPPromptFixture(
        name: "context-limit",
        messages: [
            ChatMessage(role: "user", content: String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 40) + "Now summarize the above in one sentence.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 64
    )

    /// Low acceptance / high rollback prompt: repetitive, hard-to-predict text.
    static let lowAcceptance = MTPPromptFixture(
        name: "low-acceptance",
        messages: [
            ChatMessage(role: "user", content: "List 20 random four-digit numbers separated by commas. Do not repeat any number.", reasoning: nil, reasoning_content: nil),
        ],
        maxTokens: 128
    )
}

// MARK: - Mismatch Diagnostic Reporter

/// Detailed diagnostic for a token-ID mismatch between serial and MTP decode.
struct MTPMismatchDiagnostic: Sendable {
    let fixtureName: String
    let draftDepth: Int
    let mismatchPosition: Int
    let expectedToken: Int
    let actualToken: Int
    let nearbyExpected: [Int]
    let nearbyActual: [Int]
    let acceptanceRate: Double
    let rollbackCount: Int

    var description: String {
        var lines: [String] = []
        lines.append("MTP MISMATCH: \(fixtureName) at depth \(draftDepth)")
        lines.append("  Position: \(mismatchPosition) (zero-indexed)")
        lines.append("  Expected (serial): \(expectedToken)")
        lines.append("  Actual (MTP):     \(actualToken)")
        lines.append("  Nearby expected:  \(nearbyExpected)")
        lines.append("  Nearby actual:    \(nearbyActual)")
        lines.append("  Acceptance rate:  \(String(format: "%.4f", acceptanceRate))")
        lines.append("  Rollback count:   \(rollbackCount)")
        return lines.joined(separator: "\n")
    }
}

/// Compares two token-ID arrays and produces a mismatch diagnostic on the
/// first divergence. Returns `nil` if the arrays match exactly.
func compareTokenIDs(
    serial: [Int],
    mtp: [Int],
    fixtureName: String,
    draftDepth: Int,
    acceptanceRate: Double,
    rollbackCount: Int
) -> MTPMismatchDiagnostic? {
    let count = min(serial.count, mtp.count)

    // Length mismatch is itself a failure.
    if serial.count != mtp.count {
        let pos = count
        let nearby = { (arr: [Int], center: Int) -> [Int] in
            let lo = max(0, center - 5)
            let hi = min(arr.count, center + 5)
            return Array(arr[lo..<hi])
        }
        return MTPMismatchDiagnostic(
            fixtureName: fixtureName,
            draftDepth: draftDepth,
            mismatchPosition: pos,
            expectedToken: pos < serial.count ? serial[pos] : -1,
            actualToken: pos < mtp.count ? mtp[pos] : -1,
            nearbyExpected: nearby(serial, pos),
            nearbyActual: nearby(mtp, pos),
            acceptanceRate: acceptanceRate,
            rollbackCount: rollbackCount
        )
    }

    for i in 0..<count {
        if serial[i] != mtp[i] {
            let lo = max(0, i - 5)
            let hi = min(count, i + 5)
            return MTPMismatchDiagnostic(
                fixtureName: fixtureName,
                draftDepth: draftDepth,
                mismatchPosition: i,
                expectedToken: serial[i],
                actualToken: mtp[i],
                nearbyExpected: Array(serial[lo..<hi]),
                nearbyActual: Array(mtp[lo..<hi]),
                acceptanceRate: acceptanceRate,
                rollbackCount: rollbackCount
            )
        }
    }

    return nil
}

// MARK: - Pure Unit Tests (no model weights required)

@Test
func tokenComparisonMatchesIdenticalArrays() {
    let tokens = [1, 2, 3, 4, 5]
    let diagnostic = compareTokenIDs(
        serial: tokens,
        mtp: tokens,
        fixtureName: "test",
        draftDepth: 2,
        acceptanceRate: 1.0,
        rollbackCount: 0
    )
    #expect(diagnostic == nil)
}

@Test
func tokenComparisonDetectsFirstMismatch() {
    let serial = [1, 2, 3, 4, 5]
    let mtp = [1, 9, 3, 4, 5]
    let diagnostic = compareTokenIDs(
        serial: serial,
        mtp: mtp,
        fixtureName: "test",
        draftDepth: 2,
        acceptanceRate: 0.5,
        rollbackCount: 1
    )
    #expect(diagnostic != nil)
    #expect(diagnostic?.mismatchPosition == 1)
    #expect(diagnostic?.expectedToken == 2)
    #expect(diagnostic?.actualToken == 9)
    #expect(diagnostic?.fixtureName == "test")
    #expect(diagnostic?.draftDepth == 2)
}

@Test
func tokenComparisonDetectsLengthMismatch() {
    let serial = [1, 2, 3, 4, 5]
    let mtp = [1, 2, 3]
    let diagnostic = compareTokenIDs(
        serial: serial,
        mtp: mtp,
        fixtureName: "test",
        draftDepth: 2,
        acceptanceRate: 0.8,
        rollbackCount: 0
    )
    #expect(diagnostic != nil)
    #expect(diagnostic?.mismatchPosition == 3)
}

@Test
func tokenComparisonNearbyWindow() {
    let serial = Array(1...20)
    let mtp = Array(1...20).map { $0 == 10 ? 99 : $0 }
    let diagnostic = compareTokenIDs(
        serial: serial,
        mtp: mtp,
        fixtureName: "test",
        draftDepth: 2,
        acceptanceRate: 0.9,
        rollbackCount: 1
    )
    #expect(diagnostic?.mismatchPosition == 9)
    // Nearby window: positions 4..14 (5 before, 5 after)
    #expect(diagnostic?.nearbyExpected == Array(5...14))
    #expect(diagnostic?.nearbyActual == [5, 6, 7, 8, 9, 99, 11, 12, 13, 14])
}

@Test
func promptFixturesCoverAllSixCategories() {
    let names = MTPPromptFixture.all.map(\.name)
    #expect(names.contains("ordinary-chat"))
    #expect(names.contains("short-response"))
    #expect(names.contains("code-whitespace"))
    #expect(names.contains("eos-stop-token"))
    #expect(names.contains("context-limit"))
    #expect(names.contains("low-acceptance"))
    #expect(names.count == 6)
}

// MARK: - Weight-Dependent Tests (gated behind QWEN_RUN_WEIGHT_TESTS=1)

/// Runs the full MTP correctness harness: for each prompt fixture, compares
/// serial decode (depth 0) against MTP decode at each allowed draft depth.
/// Gated behind `QWEN_RUN_WEIGHT_TESTS=1` because it requires model weights.
@Test
func mtpCorrectnessSerialMatchesMTPAtAllDepths() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[MTP Correctness] Skipped: set QWEN_RUN_WEIGHT_TESTS=1 to run weight-dependent tests.")
        return
    }

    let modelPath = env["QWEN_MODEL_PATH"] ?? "./weights"
    let mtpHeadPath = env["QWEN_MTP_HEAD_PATH"] ?? "./mtp-head"
    let maxDraftDepth = env["QWEN_SPEC_DRAFT_N_MAX"].flatMap(Int.init) ?? 2

    let generator = try await MLXGenerator(
        modelPath: modelPath,
        mtpHeadPath: mtpHeadPath,
        maxDraftDepth: maxDraftDepth
    )

    var failures: [String] = []

    for fixture in MTPPromptFixture.all {
        let request = ChatCompletionRequest(
            model: "qwen3.8-27b",
            messages: fixture.messages,
            max_tokens: fixture.maxTokens
        )

        // Serial baseline: MTP disabled, depth 0
        let serialParams = SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: 1.0,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            maxTokens: fixture.maxTokens,
            contextWindow: 262_144,
            enableThinking: false,
            mtpEnabled: false,
            prefillChunkSize: 512,
            stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default,
            ttlSeconds: nil
        )

        let serialTokens = try await generator.generateTokenIDs(
            request: request,
            samplingParams: serialParams
        )

        // MTP at each depth 1...maxDraftDepth
        for depth in 1...maxDraftDepth {
            let mtpParams = SamplingParameters(
                temperature: 0,
                topP: 1,
                topK: 0,
                minP: 0,
                repetitionPenalty: 1.0,
                presencePenalty: 0.0,
                frequencyPenalty: 0.0,
                maxTokens: fixture.maxTokens,
                contextWindow: 262_144,
                enableThinking: false,
                mtpEnabled: true,
                prefillChunkSize: 512,
                stopSequences: [],
                kvCacheConfig: ResolvedKVCacheConfig.default,
                ttlSeconds: nil
            )

            let mtpTokens = try await generator.generateTokenIDs(
                request: request,
                samplingParams: mtpParams
            )

            let acceptanceRate = Double(mtpTokens.count) / Double(max(1, mtpTokens.count + depth))
            let rollbackCount = max(0, mtpTokens.count - serialTokens.count)

            if let diagnostic = compareTokenIDs(
                serial: serialTokens,
                mtp: mtpTokens,
                fixtureName: fixture.name,
                draftDepth: depth,
                acceptanceRate: acceptanceRate,
                rollbackCount: rollbackCount
            ) {
                failures.append(diagnostic.description)
            } else {
                print("[MTP Correctness] PASS: \(fixture.name) at depth \(depth) (\(serialTokens.count) tokens)")
            }
        }
    }

    if !failures.isEmpty {
        let report = failures.joined(separator: "\n\n")
        print("[MTP Correctness] FAILURES:\n\(report)")
        #expect(failures.isEmpty, "MTP correctness harness found \(failures.count) mismatch(es)")
    } else {
        print("[MTP Correctness] All fixtures passed at all depths.")
    }
}