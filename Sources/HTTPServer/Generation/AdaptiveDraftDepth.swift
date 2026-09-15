// AdaptiveDraftDepth.swift
//
// Online adaptive draft-depth policy for the Qwen 3.8 MTP server.
//
// This is a pure, model-free state machine (no MLX, no weights): it consumes a
// per-request acceptance rate and (optionally) a per-request wall-clock
// throughput, maintains a bounded rolling window + hysteresis counters, and
// decides — one step at a time — whether to move the draft depth up or down
// by one. The `MLXGenerator` actor owns an instance of this policy, feeds it a
// sample at the end of each completed request, and applies the resulting
// decision by updating its `forcedDraftK` pin (which is read at the next
// session's creation time).
//
// Invariant: the policy only ever moves the depth by one step per sample and
// keeps it within `[minDepth, maxDepth]`. It changes how many drafts a round
// proposes, never which tokens are committed (speculative decoding is
// token-identical to serial at any depth).

import Foundation

/// The configuration for an `AdaptiveDraftDepthPolicy`. All fields are bounded
/// and validated by `validate()`; the server maps `--spec-draft-adaptive*`
/// flags onto this struct.
struct AdaptiveDraftDepthConfig: Sendable, Equatable {
    /// Upper bound on the depth (inclusive). Must be >= `minDepth`. This is the
    /// offer cap (`--spec-draft-n-max`).
    var maxDepth: Int
    /// Number of completed requests kept in the rolling window.
    var window: Int
    /// Acceptance rate at/above which the policy is eligible to increase depth.
    var thresholdHigh: Double
    /// Acceptance rate at/below which the policy is eligible to decrease depth.
    var thresholdLow: Double
    /// Consecutive same-direction samples required before a depth change (the
    /// oscillation guard).
    var hysteresis: Int
    /// Lower bound on the depth (inclusive). Fixed at 1 (serial is not allowed;
    /// depth 0 is a startup/operator setting, not a runtime target).
    var minDepth: Int = 1
    /// Whether the wall-clock throughput safety signal is active.
    var throughputEnabled: Bool = true
    /// Fraction below the rolling mean throughput at which a sample counts as a
    /// "drop" (e.g. 0.10 = 10% below the mean).
    var throughputDropFraction: Double = 0.10
    /// Consecutive throughput drops required before the safety signal may
    /// reduce depth.
    var throughputHysteresis: Int = 3
    /// Minimum number of prior throughput samples required before the safety
    /// signal is eligible to fire (avoids reacting to a cold start).
    var minThroughputSamples: Int = 3

    /// Returns an error message if the configuration is out of range, else
    /// `nil`. Used by the server to refuse to enable adaptation with a bad
    /// configuration, and directly by tests.
    func validate() -> String? {
        if maxDepth < 1 {
            return "maxDepth must be >= 1 (got \(maxDepth))"
        }
        if minDepth < 1 || minDepth > maxDepth {
            return "minDepth must be in [1, maxDepth] (got \(minDepth))"
        }
        if window < 1 {
            return "window must be >= 1 (got \(window))"
        }
        if hysteresis < 1 {
            return "hysteresis must be >= 1 (got \(hysteresis))"
        }
        if !(thresholdLow < thresholdHigh) {
            return "thresholdLow must be < thresholdHigh (got \(thresholdLow), \(thresholdHigh))"
        }
        if !(0 < thresholdHigh && thresholdHigh <= 1) {
            return "thresholdHigh must be in (0, 1] (got \(thresholdHigh))"
        }
        if !(0 <= thresholdLow && thresholdLow < 1) {
            return "thresholdLow must be in [0, 1) (got \(thresholdLow))"
        }
        if throughputEnabled {
            if !(0 < throughputDropFraction && throughputDropFraction < 1) {
                return "throughputDropFraction must be in (0, 1) (got \(throughputDropFraction))"
            }
            if throughputHysteresis < 1 {
                return "throughputHysteresis must be >= 1 (got \(throughputHysteresis))"
            }
            if minThroughputSamples < 1 {
                return "minThroughputSamples must be >= 1 (got \(minThroughputSamples))"
            }
        }
        return nil
    }
}

/// A pure, model-free online adaptive draft-depth policy. See the file header
/// for the invariants it maintains.
struct AdaptiveDraftDepthPolicy: Sendable {
    /// The outcome of feeding one sample: either no change, or a single one-step
    /// depth change with a short machine-readable reason.
    enum Decision: Sendable, Equatable {
        case noChange
        case adjusted(from: Int, to: Int, reason: Reason)

        enum Reason: Sendable, Equatable {
            case acceptanceHigh
            case acceptanceLow
            case throughputDrop
        }

        var toDepth: Int? {
            if case .adjusted(_, let to, _) = self { return to }
            return nil
        }
    }

    /// A bounded, recent history of depth adjustments (for observability).
    struct Adjustment: Sendable, Equatable {
        let from: Int
        let to: Int
        let reason: Decision.Reason
    }

    /// A snapshot of the policy's observable state, for `/metrics` and logs.
    struct Snapshot: Sendable, Equatable {
        let currentDepth: Int
        let maxDepth: Int
        let rollingAcceptanceRate: Double?
        let rollingTokensPerSecond: Double?
        let sampleCount: Int
        let totalAdjustments: Int
        let recentAdjustments: [Adjustment]
    }

    let config: AdaptiveDraftDepthConfig

    /// The depth the policy currently wants to run at. Updated by `record` and
    /// `setDepth`; always within `[config.minDepth, config.maxDepth]`.
    private(set) var currentDepth: Int
    /// Rolling window of (acceptance, tps) samples, most-recent last. Bounded to
    /// `config.window`.
    private(set) var samples: [(acceptance: Double?, tps: Double?)] = []
    /// Consecutive samples at/above `thresholdHigh` (for increasing).
    private(set) var consecutiveHigh = 0
    /// Consecutive samples at/below `thresholdLow` (for decreasing).
    private(set) var consecutiveLow = 0
    /// Consecutive throughput drops (for the safety signal).
    private(set) var consecutiveTpsDrop = 0
    /// Total number of depth adjustments made since init (lifetime counter).
    private(set) var totalAdjustments = 0
    /// Recent depth adjustments (bounded), most-recent last.
    private(set) var recentAdjustments: [Adjustment] = []

    init(config: AdaptiveDraftDepthConfig, initialDepth: Int) {
        self.config = config
        self.currentDepth = Swift.max(config.minDepth, Swift.min(config.maxDepth, initialDepth))
    }

    /// Feeds one completed request's measurements and returns the decision.
    ///
    /// - Parameters:
    ///   - acceptanceRate: `accepted / proposed` for the request, or `nil` when
    ///     no drafts were proposed (serial). A `nil` sample does not move the
    ///     acceptance hysteresis counters.
    ///   - tokensPerSecond: committed tokens / decode seconds for the request,
    ///     or `nil` when decode time is not measurable. A `nil` sample does not
    ///     move the throughput hysteresis counter.
    ///
    /// The depth moves by at most one step per call, and only within
    /// `[config.minDepth, config.maxDepth]`.
    mutating func record(acceptanceRate: Double?, tokensPerSecond: Double?) -> Decision {
        // Throughput baseline: the mean tps over the prior (non-empty) window,
        // computed before appending the current sample so the current sample is
        // compared against history, not itself.
        let priorTps = samples.compactMap { $0.tps }
        let priorMeanTps: Double? = priorTps.count >= config.minThroughputSamples
            ? priorTps.reduce(0, +) / Double(priorTps.count)
            : nil

        samples.append((acceptance: acceptanceRate, tps: tokensPerSecond))
        if samples.count > config.window {
            samples.removeFirst(samples.count - config.window)
        }

        // Acceptance hysteresis: the high and low counters are mutually
        // exclusive and both reset in the dead band (thresholdLow, thresholdHigh).
        if let acc = acceptanceRate {
            if acc >= config.thresholdHigh {
                consecutiveHigh += 1
                consecutiveLow = 0
            } else if acc <= config.thresholdLow {
                consecutiveLow += 1
                consecutiveHigh = 0
            } else {
                consecutiveHigh = 0
                consecutiveLow = 0
            }
        }

        // Throughput safety signal: a sample counts as a drop when it is below
        // the rolling mean by more than `throughputDropFraction`.
        if config.throughputEnabled,
           let tps = tokensPerSecond,
           let mean = priorMeanTps, mean > 0 {
            if tps < mean * (1.0 - config.throughputDropFraction) {
                consecutiveTpsDrop += 1
            } else {
                consecutiveTpsDrop = 0
            }
        }

        let throughputDropping =
            config.throughputEnabled && consecutiveTpsDrop >= config.throughputHysteresis

        // Decide. Acceptance is the primary signal; the throughput safety signal
        // vetoes an increase (high acceptance that does not translate to
        // throughput) and independently reduces depth on a sustained drop.
        if consecutiveHigh >= config.hysteresis, currentDepth < config.maxDepth, !throughputDropping {
            let old = currentDepth
            currentDepth += 1
            consecutiveHigh = 0
            return adjust(from: old, to: currentDepth, reason: .acceptanceHigh)
        }
        if consecutiveLow >= config.hysteresis, currentDepth > config.minDepth {
            let old = currentDepth
            currentDepth -= 1
            consecutiveLow = 0
            return adjust(from: old, to: currentDepth, reason: .acceptanceLow)
        }
        if throughputDropping, currentDepth > config.minDepth {
            let old = currentDepth
            currentDepth -= 1
            consecutiveTpsDrop = 0
            return adjust(from: old, to: currentDepth, reason: .throughputDrop)
        }
        return .noChange
    }

    /// Force the depth (e.g. after a calibration sweep sets a new baseline),
    /// clamped to `[config.minDepth, config.maxDepth]`, and reset the hysteresis
    /// counters so a stale run does not immediately trigger a change.
    mutating func setDepth(_ depth: Int) {
        currentDepth = Swift.max(config.minDepth, Swift.min(config.maxDepth, depth))
        consecutiveHigh = 0
        consecutiveLow = 0
        consecutiveTpsDrop = 0
    }

    /// The rolling mean acceptance rate over the window (nil if no sample has a
    /// rate).
    var rollingAcceptanceRate: Double? {
        let accs = samples.compactMap { $0.acceptance }
        guard !accs.isEmpty else { return nil }
        return accs.reduce(0, +) / Double(accs.count)
    }

    /// The rolling mean throughput over the window (nil if no sample has a rate).
    var rollingTokensPerSecond: Double? {
        let tps = samples.compactMap { $0.tps }
        guard !tps.isEmpty else { return nil }
        return tps.reduce(0, +) / Double(tps.count)
    }

    /// A snapshot of the observable state (for `/metrics` and logs).
    var snapshot: Snapshot {
        Snapshot(
            currentDepth: currentDepth,
            maxDepth: config.maxDepth,
            rollingAcceptanceRate: rollingAcceptanceRate,
            rollingTokensPerSecond: rollingTokensPerSecond,
            sampleCount: samples.count,
            totalAdjustments: totalAdjustments,
            recentAdjustments: recentAdjustments
        )
    }

    @inline(__always)
    private mutating func adjust(from: Int, to: Int, reason: Decision.Reason) -> Decision {
        totalAdjustments += 1
        recentAdjustments.append(Adjustment(from: from, to: to, reason: reason))
        if recentAdjustments.count > 10 {
            recentAdjustments.removeFirst(recentAdjustments.count - 10)
        }
        return .adjusted(from: from, to: to, reason: reason)
    }
}
