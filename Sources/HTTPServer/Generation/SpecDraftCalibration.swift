// SpecDraftCalibration.swift
//
// Draft-depth calibration: the on-disk store of per-model optimal draft
// depths, plus the pure selection/parsing/formatting logic. The pure parts
// (depth selection, list parsing, load/save, report) are model-free and
// unit-tested without weights; only `MLXGenerator.calibrateDraftDepths`
// (which drives the real model) requires a loaded model.
//
// This is a diagnostic/optimization surface. It never touches the committed
// token stream: the stored depth only seeds the per-round draft-depth pin
// (see `Qwen38MTPBlockSession(draftDepth:)`), which changes how many drafts a
// round proposes, never which tokens are emitted.

import Foundation
import MLXLLM

/// One row of the calibration table: a depth benchmarked over a fixed token
/// budget, with the wall-clock throughput and (for depths that draft) the
/// acceptance rate observed.
struct DepthBenchmark: Codable, Sendable, Equatable {
    let depth: Int
    let tokens: Int
    let seconds: Double
    let tokensPerSecond: Double
    /// accepted / (accepted + rejected) at this depth. `nil` for depth 0
    /// (serial, no drafting) or when no drafts were proposed.
    var acceptanceRate: Double?
}

/// The stored calibration record for a single model (a `models` value in the
/// on-disk store).
struct ModelCalibration: Codable, Sendable, Equatable {
    let optimalDepth: Int
    let calibratedAt: String        // ISO-8601 UTC
    var acceptanceRate: Double?     // acceptance at the optimal depth
    var contextLength: Int?         // context budget used during calibration
    var hardware: String?           // free-form hardware label
    var results: [DepthBenchmark]   // per-depth rows, in requested order
}

/// The on-disk calibration store: one entry per model ID (keyed by the
/// canonical model ID).
struct SpecDraftCalibrationFile: Codable, Sendable, Equatable {
    var models: [String: ModelCalibration] = [:]
}

enum SpecDraftCalibration {
    /// Default on-disk path for the calibration store (relative to the working
    /// directory; overridable with `--spec-draft-calibration-file`).
    static let defaultPath = "spec-draft-calibration.json"

    /// Parse a comma-separated depth list (e.g. `"0,1,2,3"`). Whitespace is
    /// trimmed; out-of-range values (negative or above the engine max) and
    /// duplicates are dropped; input order is preserved.
    static func parseDepths(_ raw: String) -> [Int] {
        var seen = Set<Int>()
        var out: [Int] = []
        for part in raw.split(separator: ",") {
            let trimmed = part.trimmingCharacters(in: .whitespaces)
            guard let d = Int(trimmed),
                  d >= 0, d <= MLXFastConstants.qwenMTPMaxDepth,
                  !seen.contains(d) else { continue }
            seen.insert(d)
            out.append(d)
        }
        return out
    }

    /// Select the depth with the highest wall-clock throughput. Ties are
    /// broken toward the lower depth (cheaper). Returns `nil` for an empty
    /// list.
    static func selectOptimalDepth(_ results: [DepthBenchmark]) -> Int? {
        let sorted = results.sorted { $0.depth < $1.depth }
        var best: DepthBenchmark? = nil
        for r in sorted where best == nil || r.tokensPerSecond > best!.tokensPerSecond {
            best = r
        }
        return best?.depth
    }

    /// Load the calibration store from disk. Returns `nil` when the file is
    /// missing, unreadable, or not valid JSON. A missing file is not an error
    /// (a fresh checkout has no store); a corrupt file is reported by the
    /// caller via `loadResult`.
    static func load(path: String) -> SpecDraftCalibrationFile? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            return nil
        }
        return try? JSONDecoder().decode(SpecDraftCalibrationFile.self, from: data)
    }

    /// Save the calibration store to disk (pretty-printed, keys sorted).
    /// Creates the containing directory if it does not exist.
    static func save(_ file: SpecDraftCalibrationFile, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(file)
        let url = URL(fileURLWithPath: path)
        let dir = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        try data.write(to: url, options: .atomic)
    }

    /// The human-readable report printed at the end of a calibration run.
    /// Matches the canonical table in `docs/DEPTH-CALIBRATION.md`.
    static func report(optimal: Int?, results: [DepthBenchmark], tokens: Int) -> String {
        var lines: [String] = []
        lines.append("Depth calibration results (\(tokens) tokens each):")
        for r in results.sorted(by: { $0.depth < $1.depth }) {
            let marker = r.depth == optimal ? "  <-- optimal" : ""
            lines.append(String(format: "  Depth %d: %.1f tok/s%@", r.depth, r.tokensPerSecond, marker))
        }
        lines.append("Selected depth: \(optimal.map { "\($0)" } ?? "none (no results)")")
        return lines.joined(separator: "\n")
    }

    /// Current time as an ISO-8601 UTC timestamp (e.g. `2026-09-15T12:00:00Z`).
    static func iso8601Now() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: Date())
    }
}
