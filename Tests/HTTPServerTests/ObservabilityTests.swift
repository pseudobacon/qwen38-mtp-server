import XCTest
@testable import HTTPServer

/// Pure-Swift tests for Prompt 8: per-request metric arithmetic, the
/// nearest-rank p95 helper, and the bounded `MetricsCollector` aggregate.
/// No model weights are loaded.
final class ObservabilityTests: XCTestCase {

    // MARK: - Test helpers

    /// Builds a synthetic `RequestMetrics` with the given overrides.
    private func makeMetrics(
        ttftSeconds: Double? = 0.7,
        prefillSeconds: Double = 0.5,
        decodeSeconds: Double = 2.0,
        promptTokens: Int = 100,
        completionTokens: Int = 50,
        proposedDraftTokens: Int = 8,
        acceptedDraftTokens: Int = 6,
        cancellationCause: String? = nil,
        matchedPrefixTokens: Int = 0,
        reusedPrefixTokens: Int = 0,
        radixPrefillSkipped: Bool = false
    ) -> RequestMetrics {
        RequestMetrics(
            requestID: "gen-1",
            modelAlias: "qwen3.8-27b",
            isStream: true,
            queueWaitSeconds: 0.1,
            ttftSeconds: ttftSeconds,
            generationSeconds: 2.5,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            mtpEnabled: true,
            effectiveDraftDepth: 2,
            mtpRounds: 4,
            proposedDraftTokens: proposedDraftTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            finishReason: "stop",
            cancellationCause: cancellationCause,
            memoryAdmission: .admitted,
            tokenizationCacheHit: false,
            matchedPrefixTokens: matchedPrefixTokens,
            reusedPrefixTokens: reusedPrefixTokens,
            radixPrefillSkipped: radixPrefillSkipped
        )
    }

    // MARK: - RequestMetrics throughput arithmetic

    func testPromptTokensPerSecond() {
        let metrics = makeMetrics(prefillSeconds: 0.5, promptTokens: 100)
        XCTAssertEqual(metrics.promptTokensPerSecond ?? 0, 200)
    }

    func testPromptTokensPerSecondFractionalPrefill() {
        let metrics = makeMetrics(prefillSeconds: 0.25, promptTokens: 100)
        XCTAssertEqual(metrics.promptTokensPerSecond ?? 0, 400)
    }

    func testPromptTokensPerSecondNilWhenNoPrefillTime() {
        let metrics = makeMetrics(prefillSeconds: 0)
        XCTAssertNil(metrics.promptTokensPerSecond)
    }

    func testOutputTokensPerSecond() {
        let metrics = makeMetrics(decodeSeconds: 2.5, completionTokens: 50)
        XCTAssertEqual(metrics.outputTokensPerSecond ?? 0, 20)
    }

    func testOutputTokensPerSecondZeroTokens() {
        let metrics = makeMetrics(decodeSeconds: 1.0, completionTokens: 0)
        XCTAssertEqual(metrics.outputTokensPerSecond ?? -1, 0)
    }

    func testOutputTokensPerSecondNilWhenNoDecodeTime() {
        let metrics = makeMetrics(decodeSeconds: 0)
        XCTAssertNil(metrics.outputTokensPerSecond)
    }

    // MARK: - RequestMetrics MTP acceptance rate

    func testAcceptanceRate() {
        let metrics = makeMetrics(
            proposedDraftTokens: 8,
            acceptedDraftTokens: 6
        )
        XCTAssertEqual(metrics.acceptanceRate ?? 0, 0.75)
    }

    func testAcceptanceRateFullAcceptance() {
        let metrics = makeMetrics(
            proposedDraftTokens: 4,
            acceptedDraftTokens: 4
        )
        XCTAssertEqual(metrics.acceptanceRate ?? 0, 1.0)
    }

    func testAcceptanceRateZeroWhenNothingAccepted() {
        let metrics = makeMetrics(
            proposedDraftTokens: 4,
            acceptedDraftTokens: 0
        )
        XCTAssertEqual(metrics.acceptanceRate ?? -1, 0)
    }

    func testAcceptanceRateNilWhenNoDraftsProposed() {
        let metrics = makeMetrics(
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0
        )
        XCTAssertNil(metrics.acceptanceRate)
    }

    // MARK: - MetricsCollector.p95

    func testP95EmptyIsNil() {
        XCTAssertNil(MetricsCollector.p95(of: []))
    }

    func testP95SingleValue() {
        XCTAssertEqual(MetricsCollector.p95(of: [3.5]) ?? 0, 3.5)
    }

    func testP95TwentyValues() {
        // sorted 1...20; index = ceil(0.95 * 20) - 1 = 18 -> value 19
        let values = (1...20).map(Double.init)
        XCTAssertEqual(MetricsCollector.p95(of: values) ?? 0, 19)
    }

    func testP95HundredValues() {
        // sorted 1...100; index = ceil(0.95 * 100) - 1 = 94 -> value 95
        let values = (1...100).map(Double.init)
        XCTAssertEqual(MetricsCollector.p95(of: values) ?? 0, 95)
    }

    func testP95UnsortedInput() {
        // sorted: 1, 7, 42, 55, 99; index = ceil(0.95 * 5) - 1 = 4 -> 99
        let values = [42.0, 1.0, 99.0, 7.0, 55.0]
        XCTAssertEqual(MetricsCollector.p95(of: values) ?? 0, 99)
    }

    func testP95AllEqual() {
        let values = [2.0, 2.0, 2.0, 2.0, 2.0]
        XCTAssertEqual(MetricsCollector.p95(of: values) ?? 0, 2.0)
    }

    // MARK: - MetricsCollector.summary aggregation

    func testSummaryEmptyWindow() async {
        let collector = MetricsCollector()
        let summary = await collector.summary()
        XCTAssertEqual(summary.totalRequests, 0)
        XCTAssertEqual(summary.totalPromptTokens, 0)
        XCTAssertEqual(summary.totalCompletionTokens, 0)
        XCTAssertEqual(summary.totalTokensGenerated, 0)
        XCTAssertNil(summary.averageTTFTSeconds)
        XCTAssertNil(summary.p95TTFTSeconds)
        XCTAssertNil(summary.meanMTPAcceptanceRate)
        XCTAssertEqual(summary.cancelledRequests, 0)
        XCTAssertEqual(summary.windowSize, 0)
    }

    func testSummaryAggregatesTotalsAndAverages() async {
        let collector = MetricsCollector()
        await collector.record(
            makeMetrics(
                ttftSeconds: 0.4,
                promptTokens: 100,
                completionTokens: 50,
                proposedDraftTokens: 10,
                acceptedDraftTokens: 8
            )
        )
        await collector.record(
            makeMetrics(
                ttftSeconds: 0.8,
                promptTokens: 200,
                completionTokens: 100,
                proposedDraftTokens: 10,
                acceptedDraftTokens: 5,
                cancellationCause: "sseWriteFailure"
            )
        )

        let summary = await collector.summary()
        XCTAssertEqual(summary.totalRequests, 2)
        XCTAssertEqual(summary.totalPromptTokens, 300)
        XCTAssertEqual(summary.totalCompletionTokens, 150)
        XCTAssertEqual(summary.totalTokensGenerated, 150)
        XCTAssertEqual(summary.averageTTFTSeconds ?? 0, 0.6, accuracy: 1e-9)
        XCTAssertEqual(summary.p95TTFTSeconds ?? 0, 0.8, accuracy: 1e-9)
        // (8/10 + 5/10) / 2 = 0.65
        XCTAssertEqual(summary.meanMTPAcceptanceRate ?? 0, 0.65, accuracy: 1e-9)
        XCTAssertEqual(summary.cancelledRequests, 1)
        XCTAssertEqual(summary.windowSize, 2)
    }

    func testSummarySkipsNilTTFTAndAcceptance() async {
        let collector = MetricsCollector()
        await collector.record(
            makeMetrics(
                ttftSeconds: nil,
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0
            )
        )
        await collector.record(
            makeMetrics(
                ttftSeconds: 1.0,
                proposedDraftTokens: 2,
                acceptedDraftTokens: 1
            )
        )

        let summary = await collector.summary()
        // Only the second sample has a measurable TTFT.
        XCTAssertEqual(summary.averageTTFTSeconds ?? 0, 1.0)
        XCTAssertEqual(summary.p95TTFTSeconds ?? 0, 1.0)
        // Only the second sample has a measurable acceptance rate.
        XCTAssertEqual(summary.meanMTPAcceptanceRate ?? 0, 0.5)
    }

    // MARK: - Bounded window

    func testWindowDropsOldestSamples() async {
        let collector = MetricsCollector(windowSize: 3)
        for index in 1...5 {
            await collector.record(
                makeMetrics(promptTokens: index, completionTokens: index)
            )
        }

        let summary = await collector.summary()
        // Samples 1 and 2 were dropped; only 3, 4, 5 remain.
        XCTAssertEqual(summary.totalRequests, 3)
        XCTAssertEqual(summary.totalPromptTokens, 12)
        XCTAssertEqual(summary.totalCompletionTokens, 12)
        XCTAssertEqual(summary.windowSize, 3)
    }

    func testWindowSizeClampedToAtLeastOne() async {
        let collector = MetricsCollector(windowSize: 0)
        await collector.record(makeMetrics())
        await collector.record(makeMetrics())
        let summary = await collector.summary()
        // windowSize 0 is clamped to 1, so only the newest sample survives.
        XCTAssertEqual(summary.totalRequests, 1)
        XCTAssertEqual(summary.windowSize, 1)
    }

    // MARK: - MetricsSummary JSON shape

func testMetricsSummaryEncodesSnakeCaseKeys() async throws {
        let collector = MetricsCollector()
        await collector.record(makeMetrics())
        let summary = await collector.summary()

        let json = try JSONEncoder().encode(summary)
        let object = try JSONSerialization.jsonObject(with: json)
            as? [String: Any]
        XCTAssertNotNil(object)

        let keys = object.map { Set($0.keys) } ?? Set()
        XCTAssertEqual(
            keys,
            [
                "total_requests",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_tokens_generated",
                "average_ttft_seconds",
                "p95_ttft_seconds",
                "mean_mtp_acceptance_rate",
                "cancelled_requests",
                "window_size",
                "tokenization_cache_hits",
                "tokenization_cache_misses",
                "tokenization_cache_evictions",
                "tokenization_cache_entries",
                "tokenization_cache_bytes",
                "prefix_reuse_hits",
                "prefix_reuse_fallbacks",
                "prefix_reuse_tokens_saved"
            ]
        )
        XCTAssertEqual(object?["total_requests"] as? Int, 1)
        XCTAssertEqual(object?["window_size"] as? Int, 1)
    }

    func testMetricsSummaryAggregatesPrefixReuse() async {
        let collector = MetricsCollector()
        // Hit: adopted 128 prefix tokens.
        await collector.record(makeMetrics(
            matchedPrefixTokens: 128, reusedPrefixTokens: 128, radixPrefillSkipped: true
        ))
        // Fallback: matched 64, reused 0.
        await collector.record(makeMetrics(matchedPrefixTokens: 64, reusedPrefixTokens: 0))
        // Miss: no prior cache.
        await collector.record(makeMetrics())

        let summary = await collector.summary()
        XCTAssertEqual(summary.prefixReuseHits, 1)
        XCTAssertEqual(summary.prefixReuseFallbacks, 1)
        XCTAssertEqual(summary.prefixReuseTokensSaved, 128)
    }

    func testMetricsSummaryNilFieldsEncodeAsNull() async throws {
        let collector = MetricsCollector()
        let summary = await collector.summary()

        let json = try JSONEncoder().encode(summary)
        let object = try JSONSerialization.jsonObject(with: json)
            as? [String: Any]
        XCTAssertNil(object?["average_ttft_seconds"])
        XCTAssertNil(object?["p95_ttft_seconds"])
        XCTAssertNil(object?["mean_mtp_acceptance_rate"])
    }

    // MARK: - Memory admission decision

    func testMemoryAdmissionDecisionDescriptions() {
        XCTAssertEqual(MemoryAdmissionDecision.admitted.description, "admitted")
        XCTAssertEqual(MemoryAdmissionDecision.rejected.description, "rejected")
    }
}
