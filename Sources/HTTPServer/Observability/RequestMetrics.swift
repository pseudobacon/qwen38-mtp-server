// RequestMetrics.swift
//
// Per-request observability types for the Qwen 3.8 MTP server.
//
// `GenerationMetrics` carries the generator-side measurements (prefill and
// decode durations, MTP draft statistics) across the stream boundary as a
// `GenerationFragment.metrics` yield. The router merges it with its own
// measurements (queue wait, TTFT, generation duration) into a `RequestMetrics`,
// which is recorded in the bounded `MetricsCollector` actor and served by
// `GET /metrics`.
//
// Invariant: no raw prompt text or generated content is ever stored in these
// types; only identifiers, counts, and durations.

import Foundation

/// The memory-admission decision for a request.
enum MemoryAdmissionDecision: Sendable, Codable {
    /// The request fit in the memory budget and was admitted to the lane.
    case admitted
    /// The request was rejected before model execution.
    case rejected

    var description: String {
        switch self {
        case .admitted: return "admitted"
        case .rejected: return "rejected"
        }
    }
}

/// Generator-side measurements for one generation, yielded as a
/// `GenerationFragment.metrics` at stream end. Never written to SSE.
struct GenerationMetrics: Sendable {
    /// Prefill duration in seconds.
    let prefillSeconds: Double
    /// Decode duration in seconds (first round start to loop end).
    let decodeSeconds: Double
    /// Number of MTP rounds executed.
    let rounds: Int
    /// Draft tokens proposed by the MTP head across all rounds.
    let proposedDraftTokens: Int
    /// Draft tokens accepted by target verification across all rounds.
    let acceptedDraftTokens: Int
    /// The draft depth actually used for this request (0 = serial decode).
    let effectiveDraftDepth: Int
    /// `stop`, `length`, or the cancellation cause when aborted.
    let finishReason: String
    /// Prompt (seed) token count.
    let promptTokens: Int
    /// Committed completion token count.
    let completionTokens: Int
    /// Cancellation/disconnect cause if the generation was aborted.
    let cancellationCause: String?
    /// Whether the prompt tokenization was served from the bounded
    /// tokenization cache (Stage 0). Never logs prompt content.
    let tokenizationCacheHit: Bool
}

/// Per-request metrics: identifiers, timing, throughput, MTP performance,
/// and outcome/safety. Never contains raw prompt text or generated content.
struct RequestMetrics: Sendable {
    // Identifiers
    let requestID: String
    let modelAlias: String
    let isStream: Bool

    // Timing (seconds)
    let queueWaitSeconds: Double?
    let ttftSeconds: Double?
    let generationSeconds: Double?
    let prefillSeconds: Double
    let decodeSeconds: Double

    // Throughput
    let promptTokens: Int
    let completionTokens: Int

    // MTP performance
    let mtpEnabled: Bool
    let effectiveDraftDepth: Int
    let mtpRounds: Int
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int

    // Outcome & safety
    let finishReason: String
    let cancellationCause: String?
    let memoryAdmission: MemoryAdmissionDecision

    // Tokenization cache (Stage 0)
    let tokenizationCacheHit: Bool

    /// Prompt tokens / prefill time. Nil when prefill time is not measurable.
    var promptTokensPerSecond: Double? {
        guard prefillSeconds > 0 else { return nil }
        return Double(promptTokens) / prefillSeconds
    }

    /// Committed tokens / decode time. Nil when decode time is not measurable.
    var outputTokensPerSecond: Double? {
        guard decodeSeconds > 0 else { return nil }
        return Double(completionTokens) / decodeSeconds
    }

    /// Accepted draft tokens / proposed draft tokens. Nil when no drafts were
    /// proposed (MTP disabled or serial decode).
    var acceptanceRate: Double? {
        guard proposedDraftTokens > 0 else { return nil }
        return Double(acceptedDraftTokens) / Double(proposedDraftTokens)
    }
}


/// Aggregate server statistics served by `GET /metrics`.
struct MetricsSummary: Codable, Sendable {
    let totalRequests: Int
    let totalPromptTokens: Int
    let totalCompletionTokens: Int
    let totalTokensGenerated: Int
    let averageTTFTSeconds: Double?
    let p95TTFTSeconds: Double?
    let meanMTPAcceptanceRate: Double?
    let cancelledRequests: Int
    let windowSize: Int

    /// Cumulative tokenization-cache counters (Stage 0). These are server-wide
    /// and are merged in from the generator's cache at serve time; they are
    /// not derived from the rolling request window. Defaults to zero so a
    /// summary built without a generator still encodes cleanly.
    var tokenizationCacheHits: Int = 0
    var tokenizationCacheMisses: Int = 0
    var tokenizationCacheHitRate: Double? = nil
    var tokenizationCacheEvictions: Int = 0
    var tokenizationCacheEntries: Int = 0
    var tokenizationCacheBytes: Int = 0

    enum CodingKeys: String, CodingKey {
        case totalRequests = "total_requests"
        case totalPromptTokens = "total_prompt_tokens"
        case totalCompletionTokens = "total_completion_tokens"
        case totalTokensGenerated = "total_tokens_generated"
        case averageTTFTSeconds = "average_ttft_seconds"
        case p95TTFTSeconds = "p95_ttft_seconds"
        case meanMTPAcceptanceRate = "mean_mtp_acceptance_rate"
        case cancelledRequests = "cancelled_requests"
        case windowSize = "window_size"
        case tokenizationCacheHits = "tokenization_cache_hits"
        case tokenizationCacheMisses = "tokenization_cache_misses"
        case tokenizationCacheHitRate = "tokenization_cache_hit_rate"
        case tokenizationCacheEvictions = "tokenization_cache_evictions"
        case tokenizationCacheEntries = "tokenization_cache_entries"
        case tokenizationCacheBytes = "tokenization_cache_bytes"
    }
}

/// Bounded, actor-isolated metric store.
///
/// Records are appended to a rolling window so memory stays bounded; no
/// request content is retained. Recording is a single actor hop and never
/// blocks the model executor or the SSE yield stream.
actor MetricsCollector {
    private let windowSize: Int
    private var samples: [RequestMetrics] = []

    init(windowSize: Int = 1024) {
        self.windowSize = max(1, windowSize)
    }

    /// Records one completed request. Oldest samples are dropped once the
    /// window is full.
    func record(_ metrics: RequestMetrics) {
        samples.append(metrics)
        if samples.count > windowSize {
            samples.removeFirst(samples.count - windowSize)
        }
    }

    /// Current aggregate statistics over the rolling window.
    func summary() -> MetricsSummary {
        let ttfts = samples.compactMap(\.ttftSeconds)
        let rates = samples.compactMap(\.acceptanceRate)
        return MetricsSummary(
            totalRequests: samples.count,
            totalPromptTokens: samples.reduce(0) { $0 + $1.promptTokens },
            totalCompletionTokens: samples.reduce(0) { $0 + $1.completionTokens },
            totalTokensGenerated: samples.reduce(0) { $0 + $1.completionTokens },
            averageTTFTSeconds: ttfts.isEmpty
                ? nil
                : ttfts.reduce(0, +) / Double(ttfts.count),
            p95TTFTSeconds: MetricsCollector.p95(of: ttfts),
            meanMTPAcceptanceRate: rates.isEmpty
                ? nil
                : rates.reduce(0, +) / Double(rates.count),
            cancelledRequests: samples.filter { $0.cancellationCause != nil }.count,
            windowSize: samples.count
        )
    }

    /// Nearest-rank p95: `sorted[ceil(0.95 * n) - 1]`. Nil for empty input.
    static func p95(of values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let index = max(0, Int((0.95 * Double(sorted.count)).rounded(.up)) - 1)
        return sorted[index]
    }
}
