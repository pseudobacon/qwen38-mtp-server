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
import CryptoKit
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

// MARK: - Weight-tree identity (ported from qwen-mtp-server Task 2, used by
// the Radix SSD prefix-set key: weightDigest + hardwareID).

public struct WeightTreeDigest: Equatable, Sendable {
    public let fileCount: Int
    public let byteCount: Int
    public let sha256: String
}

public enum WeightTreeDigests {
    /// SHA-256 tree digest over the regular files under `rootURL`, sorted by
    /// relative path. Matches `directoryDigest` (Sources/MLXFastHarness):
    ///   per-file:  sha256(file bytes)
    ///   tree:      sha256( relpath \0 fileSha \0 ... )  in sorted order.
    ///
    /// Returns `nil` if the directory does not exist or is not a directory.
    public static func compute(rootURL: URL) -> WeightTreeDigest? {
        let fm = FileManager.default
        let root = rootURL.resolvingSymlinksInPath()
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        // The enumerator yields fully-resolved paths (macOS: /var -> /private/
        // var), while `root.path` keeps the un-resolved form. Canonicalize the
        // base so the relative-path prefix check below matches.
        let basePath: String
        if root.path == "/var" {
            basePath = "/private/var"
        } else if root.path.hasPrefix("/var/") {
            basePath = "/private/var" + root.path.dropFirst("/var".count)
        } else {
            basePath = root.path
        }
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .totalFileSizeKey],
            options: []
        ) else {
            return nil
        }
        var fileCount = 0
        var byteCount = 0
        // Collect (relpath, sha) pairs, then fold in sorted order so the digest
        // is deterministic regardless of enumeration order.
        var pairs: [(rel: String, sha: SHA256.Digest)] = []
        while let entry = enumerator.nextObject() as? URL {
            let values = try? entry.resourceValues(forKeys: [.isRegularFileKey, .totalFileSizeKey])
            guard values?.isRegularFile == true else { continue }
            guard entry.path.hasPrefix(basePath + "/") else { continue }
            let rel = String(entry.path.dropFirst(basePath.count + 1))
            fileCount += 1
            byteCount += values?.totalFileSize ?? 0
            pairs.append((rel, SHA256File.hash(entry)))
        }
        var tree = SHA256()
        for pair in pairs.sorted(by: { $0.rel < $1.rel }) {
            tree.update(data: Data(pair.rel.utf8))
            tree.update(data: Data([0x00]))
            tree.update(data: Data(pair.sha))
            tree.update(data: Data([0x00]))
        }
        let final = tree.finalize()
        return WeightTreeDigest(
            fileCount: fileCount,
            byteCount: byteCount,
            sha256: final.compactMap { String(format: "%02x", $0) }.joined()
        )
    }

    /// Stable hardware identifier: `hw.model` (e.g. "Mac16,6") + physical RAM.
    public static func hardwareID() -> String {
        let model = sysctlString("hw.model") ?? "unknown"
        let memBytes = Int(sysctlInt64("hw.memsize") ?? 0)
        let gib = Double(memBytes) / (1024.0 * 1024.0 * 1024.0)
        return "\(model) /\(Int(gib.rounded()))GiB"
    }

    // MARK: sysctl helpers

    private static func sysctlInt64(_ name: String) -> Int64? {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        let r = sysctlbyname(name, &value, &size, nil, 0)
        guard r == 0 else { return nil }
        return value
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        var value = size
        guard sysctlbyname(name, &buf, &value, nil, 0) == 0 else { return nil }
        guard let cString = buf.withUnsafeBufferPointer({ $0.baseAddress.flatMap(String.init(cString:)) })
        else { return nil }
        return cString
    }
}

/// Streaming SHA-256 over a file (no full-file load into memory).
enum SHA256File {
    static func hash(_ url: URL) -> SHA256.Digest {
        var hasher = SHA256()
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return hasher.finalize()
        }
        defer { try? handle.close() }
        while let chunk = try? handle.read(upToCount: 1 << 20) {
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize()
    }
}
