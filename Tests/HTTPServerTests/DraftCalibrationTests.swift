// DraftCalibrationTests.swift
//
// Pure-Swift tests for the draft-depth calibration surface: depth-list
// parsing, optimal-depth selection, the on-disk store round-trip, the report
// format, the startup depth resolution, and ISO-8601 stamping. No model, no
// weights, no network. The model-in-the-loop sweep itself
// (`MLXGenerator.calibrateDraftDepths`) is exercised by the live server only.

import Testing
import Foundation
@testable import HTTPServer

private func bench(_ depth: Int, _ tps: Double, acceptance: Double? = nil) -> DepthBenchmark {
    DepthBenchmark(
        depth: depth,
        tokens: 100,
        seconds: tps > 0 ? 100.0 / tps : 0,
        tokensPerSecond: tps,
        acceptanceRate: acceptance
    )
}

private func tempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("draft-cal-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

@Suite struct DraftCalibrationTests {

    // MARK: parseDepths

    @Test func parseDepthsBasic() {
        #expect(SpecDraftCalibration.parseDepths("0,1,2,3") == [0, 1, 2, 3])
    }

    @Test func parseDepthsTrimsAndDedupes() {
        #expect(SpecDraftCalibration.parseDepths(" 2 , 3, 2 ") == [2, 3])
    }

    @Test func parseDepthsDropsOutOfRange() {
        #expect(SpecDraftCalibration.parseDepths("99,-1,2,3") == [2, 3])
    }

    @Test func parseDepthsEmpty() {
        #expect(SpecDraftCalibration.parseDepths("") == [])
    }

    // MARK: selectOptimalDepth

    @Test func selectOptimalDepthMax() {
        let results = [bench(0, 18.5), bench(1, 21.3), bench(2, 22.1), bench(3, 20.8)]
        #expect(SpecDraftCalibration.selectOptimalDepth(results) == 2)
    }

    @Test func selectOptimalDepthEmpty() {
        #expect(SpecDraftCalibration.selectOptimalDepth([]) == nil)
    }

    @Test func selectOptimalDepthTieBreaksToLowerDepth() {
        let tie = [bench(1, 20.0), bench(2, 20.0)]
        #expect(SpecDraftCalibration.selectOptimalDepth(tie) == 1)
    }

    // MARK: report

    @Test func reportFormat() {
        let results = [bench(0, 18.5), bench(1, 21.3), bench(2, 22.1), bench(3, 20.8)]
        let report = SpecDraftCalibration.report(optimal: 2, results: results, tokens: 100)
        #expect(report.contains("Depth calibration results (100 tokens each):"))
        #expect(report.contains("Depth 2: 22.1 tok/s  <-- optimal"))
        #expect(report.contains("Selected depth: 2"))
        // Non-optimal rows carry no marker.
        #expect(!report.contains("Depth 3: 20.8 tok/s  <-- optimal"))
    }

    @Test func reportNoOptimal() {
        let report = SpecDraftCalibration.report(optimal: nil, results: [], tokens: 100)
        #expect(report.contains("Selected depth: none (no results)"))
    }

    // MARK: store round-trip

    @Test func calibrationFileRoundTrip() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("cal.json").path

        var file = SpecDraftCalibrationFile()
        file.models["qwen3.8-27b-mtp"] = ModelCalibration(
            optimalDepth: 2,
            calibratedAt: "2026-09-15T12:00:00Z",
            acceptanceRate: 0.638,
            contextLength: 256,
            hardware: "macOS",
            results: [bench(0, 18.5), bench(1, 21.3), bench(2, 22.1), bench(3, 20.8)]
        )
        try SpecDraftCalibration.save(file, to: path)

        let loaded = SpecDraftCalibration.load(path: path)
        #expect(loaded != nil)
        #expect(loaded?.models["qwen3.8-27b-mtp"]?.optimalDepth == 2)
        #expect(loaded?.models["qwen3.8-27b-mtp"]?.results.count == 4)
        #expect(loaded?.models["qwen3.8-27b-mtp"]?.acceptanceRate == 0.638)
        #expect(loaded?.models["qwen3.8-27b-mtp"]?.contextLength == 256)
    }

    @Test func loadMissingFileReturnsNil() {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json").path
        #expect(SpecDraftCalibration.load(path: path) == nil)
    }

    @Test func loadCorruptFileReturnsNil() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("corrupt.json").path
        try "{ this is not valid json ".write(toFile: path, atomically: true, encoding: .utf8)
        #expect(SpecDraftCalibration.load(path: path) == nil)
    }

    // MARK: startup depth resolution

    @Test func resolvedForcedDraftDepthDefaults() {
        var c = ServerConfig()
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: nil) == ServerConfig.defaultDraftDepth)
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: 3) == 3)
    }

    @Test func resolvedForcedDraftDepthEnvBeatsStored() {
        var c = ServerConfig()
        c.specDraftK = 4
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: 3) == 4)
    }

    @Test func resolvedForcedDraftDepthExplicitNMaxWins() {
        var c = ServerConfig()
        c.specDraftNMaxExplicit = true
        c.specDraftNMax = 1
        c.specDraftK = 4
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: 3) == 1)
    }

    @Test func resolvedForcedDraftDepthStoredBeatsDefault() {
        var c = ServerConfig()
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: 3) == 3)
        #expect(c.resolvedForcedDraftDepth(storedCalibratedDepth: nil) == 2)
    }

    // MARK: timestamps

    @Test func iso8601NowFormat() {
        let stamp = SpecDraftCalibration.iso8601Now()
        #expect(stamp.contains("T"))
        #expect(stamp.hasSuffix("Z"))
    }
}
