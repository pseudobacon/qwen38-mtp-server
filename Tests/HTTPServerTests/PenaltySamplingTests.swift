// PenaltySamplingTests.swift
//
// Deterministic tests for the target-logit penalty application
// (repetition / presence / frequency) in the serial sampling path.
//
// The unit tests exercise `MTPSamplingConfig.applyPenalties` on synthetic
// logits and token histories — pure MLX array math, no model weights.
// The integration tests are gated behind `QWEN_RUN_WEIGHT_TESTS=1` and verify
// the documented serial fallback (non-default penalties force the serial
// control depth even with `mtp_enabled: true`) and greedy identity under
// default penalties.

import Foundation
import MLX
import MLXFastModel
import MLXLLM
import Testing
@testable import HTTPServer

// MARK: - Helpers

/// Reads an MLXArray row into `[Double]` for exact comparison.
private func rowValues(_ row: MLXArray) -> [Double] {
    row.asArray(Float.self).map { Double($0) }
}

// MARK: - Pure unit tests (no model weights required)

@Test
func defaultPenaltiesLeaveLogitsBitIdentical() {
    let logits = MLXArray([1.0, -2.0, 3.5, 0.0, -0.25] as [Float])
    let config = MTPSamplingConfig()
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [3, 1, 1, 4])
    #expect(rowValues(result) == rowValues(logits))
}

@Test
func emptyHistoryIsNoOpWithNonDefaultPenalties() {
    let logits = MLXArray([1.0, -2.0, 3.5] as [Float])
    let config = MTPSamplingConfig(
        repetitionPenalty: 1.5, presencePenalty: 0.5, frequencyPenalty: 0.25)
    let result = MTPSamplingConfig.applyPenalties(logits, config, history: [])
    #expect(rowValues(result) == rowValues(logits))
}

@Test
func historyTokensOutsideVocabAreIgnored() {
    let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
    let config = MTPSamplingConfig(
        repetitionPenalty: 2.0, presencePenalty: 1.0, frequencyPenalty: 1.0)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [99, -1, 1000])
    #expect(rowValues(result) == rowValues(logits))
}

@Test
func repetitionPenaltyDividesPositiveAndMultipliesNegative() {
    // history: token 0 once, token 2 twice, token 3 once.
    let logits = MLXArray([2.0, -4.0, 1.0, 0.5, 3.0] as [Float])
    let config = MTPSamplingConfig(repetitionPenalty: 2.0)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [0, 2, 2, 3])
    // token 0: 2.0 / 2 = 1.0; token 1: absent -> -4.0; token 2: 1.0 / 2 = 0.5;
    // token 3: 0.5 / 2 = 0.25; token 4: absent -> 3.0
    #expect(rowValues(result) == [1.0, -4.0, 0.5, 0.25, 3.0])
}

@Test
func repetitionPenaltyMultipliesNegativeLogits() {
    let logits = MLXArray([-2.0, 2.0] as [Float])
    let config = MTPSamplingConfig(repetitionPenalty: 2.0)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [0, 1])
    // token 0: -2.0 * 2 = -4.0; token 1: 2.0 / 2 = 1.0
    #expect(rowValues(result)[0] == -4.0)
    #expect(abs(rowValues(result)[1] - 1.0) < 1e-6)
}

@Test
func presencePenaltySubtractsOncePerPresentToken() {
    let logits = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
    let config = MTPSamplingConfig(presencePenalty: 0.5)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [0, 0, 2])
    // token 0 present (twice, but subtracted once): 0.5; token 1 absent: 2.0;
    // token 2 present: 2.5; token 3 absent: 4.0
    #expect(rowValues(result) == [0.5, 2.0, 2.5, 4.0])
}

@Test
func frequencyPenaltySubtractsPenaltyTimesCount() {
    let logits = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
    let config = MTPSamplingConfig(frequencyPenalty: 0.25)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [0, 0, 2])
    // token 0: 1.0 - 0.25*2 = 0.5; token 1: 2.0; token 2: 3.0 - 0.25 = 2.75;
    // token 3: 4.0
    #expect(rowValues(result) == [0.5, 2.0, 2.75, 4.0])
}

@Test
func combinedPenaltiesComposeInDocumentedOrder() {
    let logits = MLXArray([2.0, -2.0, 1.0] as [Float])
    let config = MTPSamplingConfig(
        repetitionPenalty: 2.0, presencePenalty: 0.5, frequencyPenalty: 0.25)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [0, 0, 2])
    // token 0: rep 2.0/2 = 1.0, presence 1.0-0.5 = 0.5, freq 0.5-0.5 = 0.0
    // token 1: absent -> -2.0
    // token 2: rep 1.0/2 = 0.5, presence 0.5-0.5 = 0.0, freq 0.0-0.25 = -0.25
    #expect(rowValues(result) == [0.0, -2.0, -0.25])
}

@Test
func greedyArgmaxShiftsUnderRepetitionPenalty() {
    // Raw argmax is token 1 (logit 3.5); with repetition penalty 2.0 and
    // token 1 in the history, token 1 drops to 1.75 and token 0 (logit 2.0)
    // wins.
    let logits = MLXArray([2.0, 3.5, 1.0] as [Float])
    let config = MTPSamplingConfig(repetitionPenalty: 2.0)
    let penalized = MTPSamplingConfig.applyPenalties(
        logits, config, history: [1])
    #expect(argMax(penalized, axis: 0).item(Int.self) == 0)
    #expect(argMax(logits, axis: 0).item(Int.self) == 1)
}

@Test
func negativePresencePenaltyBoostsPresentTokens() {
    // A negative presence penalty ADDS to present tokens (encourages
    // repetition); absent tokens are untouched.
    let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
    let config = MTPSamplingConfig(presencePenalty: -1.0)
    let result = MTPSamplingConfig.applyPenalties(
        logits, config, history: [2])
    #expect(rowValues(result) == [1.0, 2.0, 4.0])
}

// MARK: - Integration tests (gated behind QWEN_RUN_WEIGHT_TESTS=1)

/// Verifies the documented serial fallback end to end: a greedy request with
/// a non-default repetition penalty is served at the serial control depth (0)
/// even with `mtp_enabled: true`, so its committed token IDs are identical to
/// the same request with MTP disabled. Gated behind `QWEN_RUN_WEIGHT_TESTS=1`
/// because it requires model weights.
@Test
func nonDefaultPenaltiesForceSerialFallbackWithMTPEnabled() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[Penalty Sampling] Skipped: set QWEN_RUN_WEIGHT_TESTS=1 to run weight-dependent tests.")
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

    let request = ChatCompletionRequest(
        model: "qwen3.8-27b",
        messages: [
            ChatMessage(
                role: "user",
                content: "What is 2 + 2? Answer with just the number.",
                reasoning: nil,
                reasoning_content: nil
            ),
        ],
        max_tokens: 16
    )

    func params(mtpEnabled: Bool) -> SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: 1.2,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            maxTokens: 16,
            contextWindow: 262_144,
            enableThinking: false,
            mtpEnabled: mtpEnabled,
            prefillChunkSize: 512,
            stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default,
            ttlSeconds: nil
        )
    }

    let mtpTokens = try await generator.generateTokenIDs(
        request: request,
        samplingParams: params(mtpEnabled: true)
    )
    let serialTokens = try await generator.generateTokenIDs(
        request: request,
        samplingParams: params(mtpEnabled: false)
    )

    if mtpTokens != serialTokens {
        print("[Penalty Sampling] FAIL: serial fallback mismatch, "
            + "\(mtpTokens.count) vs \(serialTokens.count) tokens")
    }
    #expect(mtpTokens == serialTokens)
    print("[Penalty Sampling] PASS: serial fallback with MTP enabled (\(serialTokens.count) tokens)")
}

/// Verifies greedy identity under default penalties: with all-default
/// penalties and temperature 0, MTP-enabled and MTP-disabled greedy decoding
/// commit identical token IDs (the penalties are no-ops and the default
/// path is byte-identical to the pre-penalty code). Gated behind
/// `QWEN_RUN_WEIGHT_TESTS=1` because it requires model weights.
@Test
func defaultPenaltiesGreedyIsIdenticalWithAndWithoutMTP() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[Penalty Sampling] Skipped: set QWEN_RUN_WEIGHT_TESTS=1 to run weight-dependent tests.")
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

    let request = ChatCompletionRequest(
        model: "qwen3.8-27b",
        messages: [
            ChatMessage(
                role: "user",
                content: "What is 2 + 2? Answer with just the number.",
                reasoning: nil,
                reasoning_content: nil
            ),
        ],
        max_tokens: 16
    )

    func params(mtpEnabled: Bool) -> SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: 1.0,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            maxTokens: 16,
            contextWindow: 262_144,
            enableThinking: false,
            mtpEnabled: mtpEnabled,
            prefillChunkSize: 512,
            stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default,
            ttlSeconds: nil
        )
    }

    let mtpTokens = try await generator.generateTokenIDs(
        request: request,
        samplingParams: params(mtpEnabled: true)
    )
    let serialTokens = try await generator.generateTokenIDs(
        request: request,
        samplingParams: params(mtpEnabled: false)
    )

    if mtpTokens != serialTokens {
        print("[Penalty Sampling] FAIL: default-penalty greedy identity mismatch, "
            + "\(mtpTokens.count) vs \(serialTokens.count) tokens")
    }
    #expect(mtpTokens == serialTokens)
    print("[Penalty Sampling] PASS: default-penalty greedy identity (\(serialTokens.count) tokens)")
}
