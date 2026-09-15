// AdaptiveDraftDepthTests.swift
//
// Pure-Swift tests for the online adaptive draft-depth policy and its
// configuration mapping. No model, no weights, no network. The model-in-the-loop
// feeding (`MLXGenerator.recordAdaptiveSample`) is exercised by the live server
// only.

import Testing
import Foundation
@testable import HTTPServer

/// A config with the task's documented defaults (window 50, high 0.7, low 0.5,
/// hysteresis 10); `maxDepth` is small so the tests need few samples.
private func config(maxDepth: Int = 5,
                    window: Int = 50,
                    high: Double = 0.7,
                    low: Double = 0.5,
                    hysteresis: Int = 10) -> AdaptiveDraftDepthConfig {
    AdaptiveDraftDepthConfig(
        maxDepth: maxDepth,
        window: window,
        thresholdHigh: high,
        thresholdLow: low,
        hysteresis: hysteresis
    )
}

@Suite struct AdaptiveDraftDepthTests {

    /// Feeds `n` identical acceptance samples (with `tps`, when set) into a
    /// policy seeded at `initial`, using the explicit `config`, and returns the
    /// policy plus the decision of the last sample.
    private func feed(_ n: Int,
                      acc: Double?,
                      tps: Double? = nil,
                      _ config: AdaptiveDraftDepthConfig = config(),
                      initial: Int = 2) -> (policy: AdaptiveDraftDepthPolicy, last: AdaptiveDraftDepthPolicy.Decision) {
        var p = AdaptiveDraftDepthPolicy(config: config, initialDepth: initial)
        var last = AdaptiveDraftDepthPolicy.Decision.noChange
        for _ in 0..<n {
            last = p.record(acceptanceRate: acc, tokensPerSecond: tps)
        }
        return (p, last)
    }

    // MARK: Config validation

    @Test func configValidDefaults() {
        #expect(config().validate() == nil)
    }

    @Test func configRejectsMaxDepthBelowOne() {
        var c = config(); c.maxDepth = 0
        #expect(c.validate() != nil)
    }

    @Test func configRejectsLowNotBelowHigh() {
        var c = config(); c.thresholdLow = 0.8; c.thresholdHigh = 0.7
        #expect(c.validate() != nil)
    }

    @Test func configRejectsThresholdHighAboveOne() {
        var c = config(); c.thresholdHigh = 1.5
        #expect(c.validate() != nil)
    }

    @Test func configRejectsZeroHysteresis() {
        var c = config(); c.hysteresis = 0
        #expect(c.validate() != nil)
    }

    // MARK: Increase on sustained high acceptance

    @Test func increaseOnSustainedHighAcceptance() {
        // hysteresis = 10: 9 highs do nothing, the 10th increases 2 -> 3.
        let (_, last9) = feed(9, acc: 0.8)
        #expect(last9 == .noChange)
        let (p, last10) = feed(10, acc: 0.8)
        #expect(last10 == .adjusted(from: 2, to: 3, reason: .acceptanceHigh))
        #expect(p.currentDepth == 3)
        #expect(p.totalAdjustments == 1)
    }

    @Test func increaseRespectsMaxDepth() {
        // maxDepth = 3; start at 3. High acceptance cannot push it above 3.
        let (p, last) = feed(20, acc: 0.9, config(maxDepth: 3), initial: 3)
        #expect(last == .noChange)
        #expect(p.currentDepth == 3)
        #expect(p.totalAdjustments == 0)
    }

    // MARK: Decrease on sustained low acceptance

    @Test func decreaseOnSustainedLowAcceptance() {
        let (_, last) = feed(10, acc: 0.3)
        #expect(last == .adjusted(from: 2, to: 1, reason: .acceptanceLow))
        // minDepth = 1; cannot go below 1.
        let (p, lastMore) = feed(30, acc: 0.3)
        #expect(lastMore == .noChange)
        #expect(p.currentDepth == 1)
        #expect(p.totalAdjustments == 1)
    }

    // MARK: Dead band

    @Test func deadBandNoChange() {
        // 0.6 is in (0.5, 0.7): neither counter moves.
        let (p, last) = feed(50, acc: 0.6)
        #expect(last == .noChange)
        #expect(p.currentDepth == 2)
        #expect(p.totalAdjustments == 0)
    }

    @Test func hysteresisResetInDeadBand() {
        // 5 highs, then a dead-band sample resets the high counter, so a later
        // 5 highs do not reach the threshold of 10.
        var p = AdaptiveDraftDepthPolicy(config: config(), initialDepth: 2)
        for _ in 0..<5 { _ = p.record(acceptanceRate: 0.8, tokensPerSecond: nil) }
        _ = p.record(acceptanceRate: 0.6, tokensPerSecond: nil)
        for _ in 0..<5 { _ = p.record(acceptanceRate: 0.8, tokensPerSecond: nil) }
        #expect(p.currentDepth == 2)
        #expect(p.totalAdjustments == 0)
    }

    // MARK: Nil acceptance does not move counters

    @Test func nilAcceptanceDoesNotMoveCounters() {
        var p = AdaptiveDraftDepthPolicy(config: config(), initialDepth: 2)
        for _ in 0..<20 { _ = p.record(acceptanceRate: nil, tokensPerSecond: nil) }
        #expect(p.currentDepth == 2)
        #expect(p.totalAdjustments == 0)
    }

    // MARK: No oscillation

    @Test func noOscillationOnAlternatingSignals() {
        // Alternate high and low each round: neither direction ever accumulates
        // `hysteresis` consecutive samples, so the depth never changes.
        var p = AdaptiveDraftDepthPolicy(config: config(), initialDepth: 2)
        for i in 0..<200 {
            _ = p.record(acceptanceRate: i.isMultiple(of: 2) ? 0.9 : 0.2, tokensPerSecond: nil)
        }
        #expect(p.currentDepth == 2)
        #expect(p.totalAdjustments == 0)
    }

    @Test func boundedAdjustmentsUnderSustainedShift() {
        // A sustained high phase, then a sustained low phase: the policy moves
        // up then back down by exactly the sustained steps (one per threshold
        // crossing), not more. hysteresis = 10.
        var p = AdaptiveDraftDepthPolicy(config: config(maxDepth: 5), initialDepth: 2)
        // 10 highs -> 3; 10 highs -> 4; 10 highs -> 5 (max).
        for _ in 0..<30 { _ = p.record(acceptanceRate: 0.9, tokensPerSecond: nil) }
        #expect(p.currentDepth == 5)
        #expect(p.totalAdjustments == 3)
        // 10 lows -> 4; 10 lows -> 3.
        for _ in 0..<20 { _ = p.record(acceptanceRate: 0.3, tokensPerSecond: nil) }
        #expect(p.currentDepth == 3)
        #expect(p.totalAdjustments == 5)
    }

    // MARK: Throughput safety signal

    @Test func throughputDropDecreasesDepth() {
        // Establish a stable tps baseline, then a sustained drop.
        var p = AdaptiveDraftDepthPolicy(config: config(), initialDepth: 3)
        for _ in 0..<5 { _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 100.0) }
        // 0.6 is dead band (no acceptance move). Now drop tps well below mean.
        _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 50.0)
        _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 50.0)
        #expect(p.currentDepth == 3) // needs throughputHysteresis = 3
        let last = p.record(acceptanceRate: 0.6, tokensPerSecond: 50.0)
        #expect(last == .adjusted(from: 3, to: 2, reason: .throughputDrop))
        #expect(p.currentDepth == 2)
    }

    @Test func throughputVetoesIncrease() {
        // With hysteresis = 3, high acceptance alone reaches the increase
        // threshold on the 3rd high sample. The throughput safety signal (also
        // reached: throughputHysteresis = 3) vetoes the increase when enabled,
        // and does not when disabled. Same samples, two policies.
        func run(throughputEnabled: Bool) -> Int {
            var c = config(hysteresis: 3)
            c.throughputEnabled = throughputEnabled
            var p = AdaptiveDraftDepthPolicy(config: c, initialDepth: 2)
            // Build a stable tps baseline in the dead band (no acceptance move).
            for _ in 0..<3 { _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 100.0) }
            // High acceptance + sustained tps drop: acceptance reaches
            // hysteresis AND the tps drop reaches throughputHysteresis.
            for _ in 0..<3 { _ = p.record(acceptanceRate: 0.9, tokensPerSecond: 50.0) }
            return p.currentDepth
        }
        // Acceptance alone increases (2 -> 3).
        #expect(run(throughputEnabled: false) == 3)
        // With the throughput safety signal on, the sustained tps drop does not
        // merely veto the increase — it independently reduces depth (2 -> 1).
        #expect(run(throughputEnabled: true) == 1)
    }

    @Test func throughputDisabledHasNoEffect() {
        var c = config(); c.throughputEnabled = false
        var p = AdaptiveDraftDepthPolicy(config: c, initialDepth: 3)
        for _ in 0..<5 { _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 100.0) }
        for _ in 0..<10 { _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 5.0) }
        #expect(p.currentDepth == 3)
        #expect(p.totalAdjustments == 0)
    }

    // MARK: setDepth (calibration sync)

    @Test func setDepthClampsAndResetsHysteresis() {
        var p = AdaptiveDraftDepthPolicy(config: config(maxDepth: 5), initialDepth: 2)
        for _ in 0..<9 { _ = p.record(acceptanceRate: 0.9, tokensPerSecond: nil) }
        // 9 highs accumulated. setDepth resets them.
        p.setDepth(4)
        #expect(p.currentDepth == 4)
        // One more high is not enough (counter was reset).
        _ = p.record(acceptanceRate: 0.9, tokensPerSecond: nil)
        #expect(p.currentDepth == 4)
        // setDepth clamps to maxDepth.
        p.setDepth(99)
        #expect(p.currentDepth == 5)
        // and to minDepth.
        p.setDepth(0)
        #expect(p.currentDepth == 1)
    }

    // MARK: Initial depth clamping

    @Test func initialDepthClampedToMax() {
        var p = AdaptiveDraftDepthPolicy(config: config(maxDepth: 3), initialDepth: 9)
        #expect(p.currentDepth == 3)
    }

    @Test func initialDepthClampedToMin() {
        var p = AdaptiveDraftDepthPolicy(config: config(maxDepth: 5), initialDepth: 0)
        #expect(p.currentDepth == 1)
    }

    // MARK: Rolling stats + snapshot

    @Test func rollingAcceptanceRate() {
        var p = AdaptiveDraftDepthPolicy(config: config(), initialDepth: 2)
        _ = p.record(acceptanceRate: 0.8, tokensPerSecond: 100.0)
        _ = p.record(acceptanceRate: 0.6, tokensPerSecond: 200.0)
        #expect(abs(p.rollingAcceptanceRate! - 0.7) < 1e-9)
        #expect(abs(p.rollingTokensPerSecond! - 150.0) < 1e-9)
        let snap = p.snapshot
        #expect(snap.currentDepth == 2)
        #expect(snap.maxDepth == 5)
        #expect(snap.sampleCount == 2)
        #expect(snap.totalAdjustments == 0)
        #expect(snap.recentAdjustments.isEmpty)
    }

    @Test func rollingStatsRespectWindow() {
        let c = config(window: 2)
        var p = AdaptiveDraftDepthPolicy(config: c, initialDepth: 2)
        for _ in 0..<3 { _ = p.record(acceptanceRate: 0.9, tokensPerSecond: nil) }
        #expect(p.samples.count == 2)
    }

    // MARK: ServerConfig mapping

    @Test func adaptiveConfigOffByDefault() {
        let c = ServerConfig()
        #expect(c.specDraftAdaptive == false)
        #expect(c.adaptiveDraftDepthConfig(maxDraftDepth: 3) == nil)
    }

    @Test func adaptiveConfigBuilderWhenOn() {
        var c = ServerConfig()
        c.specDraftAdaptive = true
        c.specDraftAdaptiveWindow = 30
        c.specDraftAdaptiveThresholdHigh = 0.8
        c.specDraftAdaptiveThresholdLow = 0.4
        c.specDraftAdaptiveHysteresis = 7
        let built = c.adaptiveDraftDepthConfig(maxDraftDepth: 4)
        #expect(built != nil)
        #expect(built!.maxDepth == 4)
        #expect(built!.window == 30)
        #expect(built!.thresholdHigh == 0.8)
        #expect(built!.thresholdLow == 0.4)
        #expect(built!.hysteresis == 7)
        #expect(built!.validate() == nil)
    }

    @Test func adaptiveConfigNilWhenMtpDisabled() {
        var c = ServerConfig()
        c.specDraftAdaptive = true
        #expect(c.adaptiveDraftDepthConfig(maxDraftDepth: 0) == nil)
    }

    @Test func adaptiveConfigDefaults() {
        var c = ServerConfig()
        c.specDraftAdaptive = true
        let built = c.adaptiveDraftDepthConfig(maxDraftDepth: 3)
        #expect(built!.window == 50)
        #expect(built!.thresholdHigh == 0.7)
        #expect(built!.thresholdLow == 0.5)
        #expect(built!.hysteresis == 10)
        #expect(built!.maxDepth == 3)
    }
}
