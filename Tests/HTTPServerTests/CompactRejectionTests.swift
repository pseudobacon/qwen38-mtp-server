// CompactRejectionTests.swift
//
// Distributional tests for the compact-space rejection walk
// (docs/compact-rejection-rfc.md). Pure Swift + MLX array math, no model
// weights: reference implementations of the OLD full-vocab residual walk and
// the NEW compact-space split walk, compared against each other and against
// the analytic residual distribution with a seeded RNG (chi-square + total
// variation), per docs/speculative-sampling-rfc.md §6.
//
// The weight-gated serial-vs-MTP distributional test lives below, gated
// behind QWEN_RUN_WEIGHT_TESTS=1.

import Foundation
import MLX
import Testing
@testable import HTTPServer

// MARK: - Reference walks

/// OLD walk (pre-change): full-vocab residual `max(0, p - q)`, renormalised,
/// 1e-10-clamped log-space categorical; degenerate fallback samples `p`.
func oldResidualSample(p: MLXArray, qFull: MLXArray) -> Int {
    let diff = p - qFull
    let positive = diff .> 0
    let masked = which(positive, diff, MLXArray.full([p.size], values: MLXArray(0.0)))
    let total = masked.sum().item(Float.self)
    if total > 0 {
        let safeDiff = maximum(masked, MLXArray(1e-10))
        let diffLogits = safeDiff.log()
        let resampled = MLXRandom.categorical(diffLogits, axis: 0)
        return Int(resampled.item(Int32.self))
    } else {
        let safePDist = maximum(p, MLXArray(1e-10))
        let pLogits = safePDist.log()
        let resampled = MLXRandom.categorical(pLogits, axis: 0)
        return Int(resampled.item(Int32.self))
    }
}

/// NEW walk (this change): compact-space split, mirroring the production
/// rejection walk in Qwen36MTPBlockSession.swift line-for-line. `qCompact`
/// has support only on the compact indices; `c2f` is the compact→full map.
func newResidualSample(p: MLXArray, qCompact: MLXArray, c2f: MLXArray) -> Int {
    let pCompact = p[c2f]
    let diff = pCompact - qCompact
    let positive = diff .> 0
    let masked = which(
        positive, diff,
        MLXArray.full([pCompact.size], values: MLXArray(0.0)))
    let totalIn = masked.sum().item(Float.self)
    let totalOut = max(0.0, 1.0 - pCompact.sum().item(Float.self))
    let total = totalIn + totalOut
    if total <= 0 {
        let safePDist = maximum(p, MLXArray(1e-10))
        let pLogits = safePDist.log()
        let resampled = MLXRandom.categorical(pLogits, axis: 0)
        return Int(resampled.item(Int32.self))
    } else if totalOut <= 0 {
        let safeDiff = maximum(masked, MLXArray(1e-10))
        let diffLogits = safeDiff.log()
        let resampled = MLXRandom.categorical(diffLogits, axis: 0)
        return Int(c2f[resampled].item(Int32.self))
    } else {
        let u = MLXRandom.uniform(0.0 ..< 1.0, [1]).item(Float.self)
        if u * total < totalIn {
            let safeDiff = maximum(masked, MLXArray(1e-10))
            let diffLogits = safeDiff.log()
            let resampled = MLXRandom.categorical(diffLogits, axis: 0)
            return Int(c2f[resampled].item(Int32.self))
        } else {
            let compactMask = MLXArray.full([p.size], values: MLXArray(0.0))
                .at[c2f].add(1.0)
            let pLogits = which(
                compactMask .> 0,
                MLXArray(-Float.infinity),
                maximum(p, MLXArray(1e-10)).log())
            let resampled = MLXRandom.categorical(pLogits, axis: 0)
            return Int(resampled.item(Int32.self))
        }
    }
}

// MARK: - Distribution helpers

/// Run a walk `n` times, returning the empirical count per full-vocab id.
func empiricalCounts(
    n: Int, sample: () -> Int
) -> [Int: Int] {
    var counts: [Int: Int] = [:]
    for _ in 0 ..< n {
        counts[sample(), default: 0] += 1
    }
    return counts
}

/// Chi-square goodness-of-fit of `counts` (total `n`) against the analytic
/// residual `expected` (full-vocab, normalized). Rare cells (expected
/// mass < 0.02) are pooled into one tail cell, the standard fix for sparse
/// expected counts. Returns (statistic, degreesOfFreedom).
func chiSquare(
    counts: [Int: Int],
    n: Int,
    expected: [Float]
) -> (stat: Float, dof: Int) {
    let vocab = expected.count
    var observed = [Float](repeating: 0, count: vocab)
    for (id, c) in counts where id >= 0 && id < vocab {
        observed[id] = Float(c)
    }
    // Pool: cells with expected mass < 0.02 (plus all unseen support) merge.
    var cells: [Int: [Float]] = [:]  // cellIndex -> [observed, expected]
    var pooledObs: Float = 0
    var pooledExp: Float = 0
    var cellCount = 0
    for f in 0 ..< vocab {
        let e = expected[f] * Float(n)
        if e < 0.02 * Float(n) {
            pooledObs += observed[f]
            pooledExp += e
        } else {
            cells[cellCount] = [observed[f], e]
            cellCount += 1
        }
    }
    if pooledExp > 0 || pooledObs > 0 {
        cells[cellCount] = [pooledObs, pooledExp]
        cellCount += 1
    }
    var stat: Float = 0
    for (_, cell) in cells where cell[1] > 0 {
        let d = cell[0] - cell[1]
        stat += d * d / cell[1]
    }
    return (stat, max(1, cellCount - 1))
}

/// Total variation distance between two count maps (same total).
func totalVariation(_ a: [Int: Int], _ b: [Int: Int]) -> Double {
    let keys = Set(a.keys).union(b.keys)
    let totalA = Double(a.values.reduce(0, +))
    let totalB = Double(b.values.reduce(0, +))
    var tv: Double = 0
    for k in keys {
        tv += abs(Double(a[k] ?? 0) / totalA - Double(b[k] ?? 0) / totalB)
    }
    return tv / 2
}

// MARK: - Synthetic cases

let testV = 64        // full vocab
let testC = 32        // compact vocab
/// c2f(c) = 2c: compact rows map to the even full ids; the odd full ids are
/// the out-of-support complement (R_out mass lives there).
let testC2F: [Int32] = (0 ..< testC).map { Int32($0) * 2 }

struct RejectionCase: Sendable {
    let name: String
    let p: [Float]       // full vocab, normalized
    let q: [Float]       // compact, normalized
}

func normalized(_ x: [Float]) -> [Float] {
    let s = x.reduce(0, +)
    return x.map { $0 / s }
}

func rejectionCases() -> [RejectionCase] {
    // Explicitly typed intermediates (keeps type-checking cheap).
    let p1: [Float] = normalized((0 ..< testV).map { Float(1) + Float($0) * 0.01 })
    let q1: [Float] = normalized((0 ..< testC).map { Float(1) + Float($0) * 0.02 })
    let p2: [Float] = normalized((0 ..< testV).map { $0 % 2 == 1 ? Float(1) : Float(0) })
    let q2: [Float] = normalized((0 ..< testC).map { Float($0) + Float(1) })
    let p3: [Float] = normalized((0 ..< testV).map { $0 == 4 ? Float(1) : Float(0) })
    let p4: [Float] = normalized((0 ..< testV).map { Float(1) + Float($0) * 0.005 })
    let q4: [Float] = normalized((0 ..< testC).map { $0 == 7 ? Float(1) : Float(0) })
    let p5: [Float] = normalized((0 ..< testV).map { $0 == 13 ? Float(1) : Float(0) })
    return [
        // 1. Well-overlapping: both spread, similar shape.
        RejectionCase(name: "overlapping", p: p1, q: q1),
        // 2. Disjoint supports: p entirely OUT of the compact support (odd
        // ids), q spread in-support. R_in = 0, R_out = 1: pure tail sampling.
        RejectionCase(name: "disjoint-out-of-support", p: p2, q: q2),
        // 3. q has mass where p = 0: p is a point mass in-support, q spread.
        RejectionCase(name: "q-mass-where-p-zero", p: p3, q: q2),
        // 4. p has mass where q = 0 (out-of-support): p spread over ALL ids,
        // q point-mass in-support. Exercises the split draw heavily.
        RejectionCase(name: "p-mass-out-of-support", p: p4, q: q4),
        // 5. Top-k = 1 nucleus on p, out-of-support target.
        RejectionCase(name: "topk1-out-of-support", p: p5, q: q2),
    ]
}

/// Full-vocab lift of a compact distribution under testC2F.
func lift(_ q: [Float]) -> [Float] {
    var full = [Float](repeating: 0, count: testV)
    for c in 0 ..< testC {
        full[Int(testC2F[c])] = q[c]
    }
    return full
}

/// Analytic residual r_f / R for a case (full vocab).
func analyticResidual(_ c: RejectionCase) -> [Float] {
    let qFull = lift(c.q)
    var r = [Float](repeating: 0, count: testV)
    var R: Float = 0
    for f in 0 ..< testV {
        r[f] = max(0, c.p[f] - qFull[f])
        R += r[f]
    }
    if R > 0 { r = r.map { $0 / R } }
    return r
}

// MARK: - Distributional tests

@Suite(.timeLimit(.minutes(10)))
struct CompactRejectionDistributionalTests {

    @Test("new walk matches old walk and the analytic residual")
    func newWalkMatchesOldWalkAndAnalytic() {
        let n = 20_000
        for c in rejectionCases() {
            MLXRandom.seed(0xC0FFEE)
            let pArr = MLXArray(c.p)
            let qFullArr = MLXArray(lift(c.q))
            let qCompactArr = MLXArray(c.q)
            let c2fArr = MLXArray(testC2F)
            let expected = analyticResidual(c)

            let oldCounts = empiricalCounts(n: n) { oldResidualSample(p: pArr, qFull: qFullArr) }
            let newCounts = empiricalCounts(n: n) { newResidualSample(p: pArr, qCompact: qCompactArr, c2f: c2fArr) }

            let (statNew, dofNew) = chiSquare(counts: newCounts, n: n, expected: expected)
            let tv = totalVariation(newCounts, oldCounts)
            print("[compact-rejection] \(c.name): chi2/new-vs-analytic=\(statNew) dof=\(dofNew) tv(new,old)=\(tv)")

            // Chi-square per dof: mean is 1; 5 is ~28 sigma for these sizes.
            #expect(statNew / Float(dofNew) < 5.0,
                "\(c.name): new walk deviates from the analytic residual")
            // Total variation between the two walks' empirical distributions:
            // sampling noise ~ sqrt(V/(2n)) ≈ 0.04; 0.15 is 3.75 sigma.
            #expect(tv < 0.15,
                "\(c.name): new walk diverges from the old walk (TV=\(tv))")
        }
    }

    @Test("full-vocab degenerate case C == V reduces to the old walk")
    func fullVocabDegenerate() {
        let n = 20_000
        let identity: [Int32] = (0 ..< testV).map { Int32($0) }
        let p: [Float] = normalized((0 ..< testV).map { Float(1) + Float($0) * 0.01 })
        let q: [Float] = normalized((0 ..< testV).map { Float(1) + Float($0) * 0.02 })
        MLXRandom.seed(0xF00D)
        let pArr = MLXArray(p)
        let qArr = MLXArray(q)
        let c2fArr = MLXArray(identity)
        let expected = analyticResidualFull(p: p, q: q)

        let oldCounts = empiricalCounts(n: n) { oldResidualSample(p: pArr, qFull: qArr) }
        let newCounts = empiricalCounts(n: n) { newResidualSample(p: pArr, qCompact: qArr, c2f: c2fArr) }

        let (statNew, dofNew) = chiSquare(counts: newCounts, n: n, expected: expected)
        let tv = totalVariation(newCounts, oldCounts)
        print("[compact-rejection] full-vocab: chi2=\(statNew) dof=\(dofNew) tv(new,old)=\(tv)")
        #expect(statNew / Float(dofNew) < 5.0)
        #expect(tv < 0.15)
    }

    @Test("point-mass targets are sampled deterministically")
    func pointMassDeterministic() {
        // p point mass OUT of support (odd id 13), q point mass in-support
        // (compact 7 -> full 14): R_in = 0, R_out = 1, tail has exactly one
        // nonzero entry -> the new walk must always return 13.
        var p = [Float](repeating: 0, count: testV); p[13] = 1
        var q = [Float](repeating: 0, count: testC); q[7] = 1
        MLXRandom.seed(1)
        let pArr = MLXArray(p)
        let qArr = MLXArray(q)
        let c2fArr = MLXArray(testC2F)
        for _ in 0 ..< 25 {
            #expect(newResidualSample(p: pArr, qCompact: qArr, c2f: c2fArr) == 13)
        }

        // p == q point mass in-support (compact 3 -> full 6): R = 0, the
        // degenerate fallback samples the point mass of p deterministically
        // (log(1) = 0 beats log(1e-10) everywhere).
        var p2 = [Float](repeating: 0, count: testV); p2[6] = 1
        var q2 = [Float](repeating: 0, count: testC); q2[3] = 1
        let p2Arr = MLXArray(p2)
        let q2Arr = MLXArray(q2)
        for _ in 0 ..< 25 {
            #expect(newResidualSample(p: p2Arr, qCompact: q2Arr, c2f: c2fArr) == 6)
        }
    }
}

func analyticResidualFull(p: [Float], q: [Float]) -> [Float] {
    var r = [Float](repeating: 0, count: p.count)
    var R: Float = 0
    for f in 0 ..< p.count {
        r[f] = max(0, p[f] - q[f])
        R += r[f]
    }
    if R > 0 { r = r.map { $0 / R } }
    return r
}

// MARK: - Weight-gated serial-vs-MTP distributional equivalence

/// Gated behind `QWEN_RUN_WEIGHT_TESTS=1`: for a non-greedy configuration,
/// serial (depth 0) and MTP speculative sampling must emit the same token
/// DISTRIBUTION (not the same tokens — independent RNG streams by design).
/// Chi-square over the top-40 observed tokens, the long tail pooled, per
/// docs/speculative-sampling-rfc.md §6B.
@Test("serial and MTP sampling match distributionally (weight-gated)")
func serialAndMTPMatchDistributionally() async throws {
    let env = ProcessInfo.processInfo.environment
    guard env["QWEN_RUN_WEIGHT_TESTS"] == "1" else {
        print("[compact-rejection] Skipped: set QWEN_RUN_WEIGHT_TESTS=1.")
        return
    }
    guard acquireWeightTestLock() else {
        print("[compact-rejection] Skipped: another weight-gated test is running (run weight-gated tests in separate invocations).")
        return
    }

    let modelPath = env["QWEN_MODEL_PATH"] ?? "./weights"
    let mtpHeadPath = env["QWEN_MTP_HEAD_PATH"] ?? "./mtp-head"
    let generator = try await MLXGenerator(
        modelPath: modelPath, mtpHeadPath: mtpHeadPath, maxDraftDepth: 2
    )

    let request = ChatCompletionRequest(
        model: "qwen3.8-27b",
        messages: [ChatMessage(role: "user", content: "Write a short story about a lighthouse keeper. Keep it to two paragraphs.", reasoning: nil, reasoning_content: nil)],
        max_tokens: 400
    )

    func params(mtpEnabled: Bool) -> SamplingParameters {
        SamplingParameters(
            temperature: 0.7,
            topP: 0.95,
            topK: 0,
            minP: 0,
            repetitionPenalty: 1.0,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            maxTokens: 400,
            contextWindow: 262_144,
            enableThinking: false,
            mtpEnabled: mtpEnabled,
            prefillChunkSize: 512,
            stopSequences: [],
            kvCacheConfig: ResolvedKVCacheConfig.default,
            ttlSeconds: nil
        )
    }

    let serialTokens = try await generator.generateTokenIDs(
        request: request, samplingParams: params(mtpEnabled: false))
    let mtpTokens = try await generator.generateTokenIDs(
        request: request, samplingParams: params(mtpEnabled: true))
    print("[compact-rejection] serial=\(serialTokens.count) mtp=\(mtpTokens.count) tokens")
    #expect(serialTokens.count >= 200, "serial sample too short for a chi-square")
    #expect(mtpTokens.count >= 200, "mtp sample too short for a chi-square")

    // Pool: top-40 tokens by serial frequency stay separate; everything else
    // (the long tail) merges into one cell.
    var freq: [Int: Int] = [:]
    for t in serialTokens { freq[t, default: 0] += 1 }
    let top40 = freq.sorted { $0.value > $1.value }.prefix(40).map { $0.key }
    let topSet = Set(top40)
    func cell(_ t: Int) -> Int { topSet.contains(t) ? t : -1 }

    var obsS = [Int: Int](); var obsM = [Int: Int]()
    for t in serialTokens { obsS[cell(t), default: 0] += 1 }
    for t in mtpTokens { obsM[cell(t), default: 0] += 1 }
    let n = obsS.values.reduce(0, +) + obsM.values.reduce(0, +)
    var expected: [Float] = []
    for c in Set(obsS.keys).union(obsM.keys) {
        expected.append(Float((obsS[c] ?? 0) + (obsM[c] ?? 0)) / Float(n))
    }
    // Renormalize expected over the pooled cells.
    let es = expected.reduce(0, +)
    expected = expected.map { $0 / es }
    // expected[] is indexed by Set order — rebuild chiSquare over aligned
    // per-cell counts instead (chiSquare indexes by full-vocab id, so do the
    // arithmetic directly here).
    var stat: Float = 0
    let cells = Array(Set(obsS.keys).union(obsM.keys))
    var pooledExp: [Int: Float] = [:]
    for (i, c) in cells.enumerated() { pooledExp[c] = expected[i] * Float(n) }
    for c in cells {
        let e = pooledExp[c]!
        let o = Float((obsS[c] ?? 0) + (obsM[c] ?? 0))
        if e > 0 { stat += (o - e) * (o - e) / e }
    }
    let dof = max(1, cells.count - 1)
    print("[compact-rejection] chi2=\(stat) dof=\(dof)")
    #expect(stat / Float(dof) < 5.0,
        "MTP sampling distribution deviates from serial (chi2/dof=\(stat / Float(dof)))")
}
