import Foundation
import Crypto
import Logging

/// Provenance-safe pre-warm of Metal's built-in JIT cache (quick-wins Item 3,
/// LEV-C flag 1). See `docs/PRE-WARM.md` for the mechanism survey.
///
/// Background
/// ----------
/// The startup warmup (`Qwen38MTPBlockSession.warmAllDepths`) compiles the
/// decode-family Metal kernels on first use (~14 s of cold JIT). The results
/// persist in Metal's built-in per-user disk cache
/// (`$DARWIN_USER_CACHE_DIR/com.apple.metal/<framework-version>/`), so
/// restarts on the same machine pay only ~3 s. The `--prewarm-exit` mode runs
/// the full startup once at install time (populating that cache on the
/// deployment machine) and writes a version-keyed provenance manifest.
///
/// Provenance safety (the stale-metallib lesson)
/// ----------------------------------------------
/// We capture NO kernel state of our own. Metal's cache is content-keyed
/// (hash over kernel source + compile options, where the JIT kernel sources
/// are embedded in the runtime binary and the prebuilt kernels live in the
/// metallib), so ANY change to (binary, metallib, model geometry, OS)
/// produces a cache MISS — i.e. a cold JIT — never a stale hit. The manifest
/// exists so every normal startup can verify and REPORT whether the cache
/// can be trusted, and fail loudly (WARNING) when it cannot.
struct PrewarmManifest: Codable, Equatable {
    var schema: Int = 1
    var created_at: String
    var prewarm_wall_seconds: Double
    /// SHA-256 of the exact server executable that ran the pre-warm (its
    /// embedded C++ MLX JIT kernel sources are part of the cache key).
    var binary_sha256: String
    /// The prebuilt metallib the runtime loads (search order replicated from
    /// `mlx/backend/metal/device.cpp`); "" when unresolvable (fail-loud).
    var metallib_path: String
    var metallib_sha256: String
    var model_path: String
    /// `WeightTreeDigests` tree digest over the weight directory.
    var model_weight_digest: String
    var mtp_head_path: String
    var mtp_head_digest: String
    /// Draft geometry: the verify widths warmed are 1...specDraftNMax+1 and
    /// the per-round k selects the same live expressions a request dispatches.
    var spec_draft_n_max: Int
    var forced_draft_k: Int
    /// Metal framework version is pinned by the OS build; a change moves the
    /// cache into a new (empty) framework-version directory.
    var macos_build: String
    /// `hw.model` + RAM: kernels are device-specialized.
    var hardware_id: String
    /// Diagnostic only (not compared): the per-user Metal cache directory.
    var metal_cache_dir: String
}

/// Outcome of a provenance check.
enum PrewarmVerdict: Equatable {
    /// No manifest at the expected path: first boot, cold JIT expected.
    case noManifest
    /// All compared fields agree: the warm cache is trustworthy.
    case match
    /// At least one field differs: the cache cannot be trusted (it will miss
    /// by Metal's own keying — cold JIT, never a stale hit).
    case mismatch(fields: [String])
}

enum PrewarmError: Error, CustomStringConvertible {
    case metallibUnresolvable(candidates: [String])
    case manifestWrite(String)

    var description: String {
        switch self {
        case .metallibUnresolvable(let candidates):
            return "could not resolve the metallib the runtime loads; searched: "
                + candidates.joined(separator: ", ")
        case .manifestWrite(let detail):
            return "manifest write failed: \(detail)"
        }
    }
}

/// Computes, writes, and validates the pre-warm provenance manifest.
///
/// The compared fields are exactly the inputs that select the compiled
/// kernel set: runtime binary (JIT sources) + metallib (prebuilt kernels)
/// + model/head geometry (function specializations) + draft geometry
/// (verify widths) + OS build (Metal framework version) + hardware (device
/// specialization).
enum PrewarmProvenance {

    // MARK: - Provenance inputs

    /// Ordered search paths for the metallib the runtime will load,
    /// replicating `load_default_library` in `mlx/backend/metal/device.cpp`
    /// (the SwiftPM-bundle `default` lookup and the `METAL_PATH` CWD-relative
    /// fallback; the framework-identifier path is unreachable for a SwiftPM
    /// CLI executable).
    static func metallibSearchPaths() -> [URL] {
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        var exeDir = cwd
        if let exe = Bundle.main.executableURL {
            exeDir = exe.deletingLastPathComponent()
        }
        return [
            exeDir.appendingPathComponent("mlx.metallib"),
            exeDir.appendingPathComponent("Resources").appendingPathComponent("mlx.metallib"),
            // SWIFTPM_BUNDLE == "mlx-swift_Cmlx" (Cmlx cxxSetting define).
            exeDir.appendingPathComponent("mlx-swift_Cmlx.bundle")
                .appendingPathComponent("default.metallib"),
            cwd.appendingPathComponent("default.metallib"),
        ]
    }

    /// First existing candidate, or nil (fail-loud upstream).
    static func resolveMetallib() -> URL? {
        let fm = FileManager.default
        return metallibSearchPaths().first { fm.fileExists(atPath: $0.path) }
    }

    /// Streaming SHA-256 of a file ("", never throws — callers gate on the
    /// path existing first).
    static func sha256Hex(_ url: URL) -> String {
        let digest = SHA256File.hash(url)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }

    /// `kern.osproductversion` (e.g. "25F84") — pins the Metal framework
    /// version directory.
    static func macosBuild() -> String {
        var size = 0
        guard sysctlbyname("kern.osproductversion", nil, &size, nil, 0) == 0, size > 0 else {
            return "unknown"
        }
        var buf = [CChar](repeating: 0, count: size)
        var value = size
        guard sysctlbyname("kern.osproductversion", &buf, &value, nil, 0) == 0 else {
            return "unknown"
        }
        return buf.withUnsafeBufferPointer {
            $0.baseAddress.flatMap { String(cString: $0) } ?? "unknown"
        }
    }

    /// The per-user Metal cache directory (diagnostic only; not compared).
    /// `getconf` is not exposed to Swift; the env var is set by launchd for
    /// GUI/login-session processes and may be empty for shell launches — the
    /// prewarm script computes the authoritative path via `getconf`.
    static func metalCacheDir() -> String {
        guard let dir = ProcessInfo.processInfo.environment["DARWIN_USER_CACHE_DIR"]
        else { return "" }
        return (dir.hasSuffix("/") ? dir : dir + "/") + "com.apple.metal"
    }

    /// Builds the manifest for the CURRENT process state (the prewarm-exit
    /// path; no requests are in flight, so the weight digest is computed
    /// directly rather than deferred).
    static func build(
        modelPath: String,
        mtpHeadPath: String,
        specDraftNMax: Int,
        forcedDraftK: Int,
        processStart: DispatchTime
    ) async throws -> PrewarmManifest {
        let (weightDigest, hardwareID) = await Task.detached(priority: .utility) {
            let id = MLXGenerator.computeWeightIdentity(modelPath: modelPath)
            return (id.digest, id.hardware)
        }.value
        return try Self.currentProvenance(
            modelPath: modelPath,
            mtpHeadPath: mtpHeadPath,
            specDraftNMax: specDraftNMax,
            forcedDraftK: forcedDraftK,
            weightDigest: weightDigest,
            hardwareID: hardwareID,
            prewarmWallSeconds: Double(
                DispatchTime.now().uptimeNanoseconds - processStart.uptimeNanoseconds
            ) / 1e9
        )
    }

    /// The manifest describing the current process state, given the weight
    /// identity. Throws only when the metallib cannot be resolved (an
    /// unkeyed manifest / check would be worthless — fail loud).
    private static func currentProvenance(
        modelPath: String,
        mtpHeadPath: String,
        specDraftNMax: Int,
        forcedDraftK: Int,
        weightDigest: String,
        hardwareID: String,
        prewarmWallSeconds: Double
    ) throws -> PrewarmManifest {
        guard let _ = Self.resolveMetallib() else {
            throw PrewarmError.metallibUnresolvable(
                candidates: metallibSearchPaths().map { $0.path })
        }
        let fm = FileManager.default
        let exeURL = Bundle.main.executableURL
        let binarySha: String
        if let exeURL, fm.fileExists(atPath: exeURL.path) {
            binarySha = sha256Hex(exeURL)
        } else {
            binarySha = ""
        }
        let headURL = URL(fileURLWithPath: mtpHeadPath).resolvingSymlinksInPath()
        let headDigest = WeightTreeDigests.compute(rootURL: headURL)?.sha256 ?? ""
        var manifest = PrewarmManifest(
            created_at: SpecDraftCalibration.iso8601Now(),
            prewarm_wall_seconds: prewarmWallSeconds,
            binary_sha256: binarySha,
            metallib_path: "",
            metallib_sha256: "",
            model_path: URL(fileURLWithPath: modelPath).resolvingSymlinksInPath().path,
            model_weight_digest: weightDigest,
            mtp_head_path: headURL.path,
            mtp_head_digest: headDigest,
            spec_draft_n_max: specDraftNMax,
            forced_draft_k: forcedDraftK,
            macos_build: Self.macosBuild(),
            hardware_id: hardwareID,
            metal_cache_dir: Self.metalCacheDir()
        )
        if let metallib = Self.resolveMetallib() {
            manifest.metallib_path = metallib.path
            manifest.metallib_sha256 = sha256Hex(metallib)
        }
        return manifest
    }

    /// `--prewarm-exit`: write the manifest (atomic), then the caller exits.
    /// Fails loudly if the metallib cannot be resolved (an unkeyed manifest
    /// would be worthless) or the write fails.
    static func writeManifest(
        logger: Logger,
        manifestPath: String,
        modelPath: String,
        mtpHeadPath: String,
        specDraftNMax: Int,
        forcedDraftK: Int,
        processStart: DispatchTime
    ) async throws {
        guard Self.resolveMetallib() != nil else {
            throw PrewarmError.metallibUnresolvable(
                candidates: metallibSearchPaths().map { $0.path })
        }
        let manifest = try await Self.build(
            modelPath: modelPath,
            mtpHeadPath: mtpHeadPath,
            specDraftNMax: specDraftNMax,
            forcedDraftK: forcedDraftK,
            processStart: processStart
        )
        let url = URL(fileURLWithPath: manifestPath)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(manifest)
        let tmp = url.deletingLastPathComponent()
            .appendingPathComponent(".prewarm-manifest.json.tmp-\(ProcessInfo.processInfo.processIdentifier)")
        do {
            // A re-pre-warm (after a build/model/OS change) replaces the old
            // manifest: remove the destination first, then move into place.
            // If a crash lands between the two, the failure mode is "no
            // manifest" — the startup check reports cold-JIT-expected, never
            // a stale hit.
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            try data.write(to: tmp, options: .atomic)
            try FileManager.default.moveItem(at: tmp, to: url)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            throw PrewarmError.manifestWrite("\(error)")
        }
        let wall = String(format: "%.1f", manifest.prewarm_wall_seconds)
        let binary = Self.prefix(manifest.binary_sha256)
        let metallib = Self.prefix(manifest.metallib_sha256)
        let weights = Self.prefix(manifest.model_weight_digest)
        let msg = "Prewarm: manifest written to \(manifestPath) (wall \(wall) s, "
            + "binary \(binary), metallib \(metallib), weights \(weights))"
        logger.info("\(msg)")
    }

    /// Compares the manifest at `manifestPath` against the current process
    /// state. The weight identity comes from `weightIdentity` — in normal
    /// server mode this is `MLXGenerator.weightIdentityDeferred()` (shared
    /// with the lazy SSD path, deferred until after the first request), so
    /// the process pays for the ~8 s / 15 GB read at most once.
    static func check(
        manifestPath: String,
        modelPath: String,
        mtpHeadPath: String,
        specDraftNMax: Int,
        forcedDraftK: Int,
        weightIdentity: @Sendable () async -> (digest: String, hardware: String)
    ) async -> PrewarmVerdict {
        let url = URL(fileURLWithPath: manifestPath)
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(PrewarmManifest.self, from: data)
        else {
            return .noManifest
        }
        let (weightDigest, hardwareID) = await weightIdentity()
        guard let current = try? Self.currentProvenance(
            modelPath: modelPath,
            mtpHeadPath: mtpHeadPath,
            specDraftNMax: specDraftNMax,
            forcedDraftK: forcedDraftK,
            weightDigest: weightDigest,
            hardwareID: hardwareID,
            prewarmWallSeconds: 0
        ) else {
            // currentProvenance only throws when the metallib is
            // unresolvable — an unverifiable state must fail loud, not pass
            // silently.
            return .mismatch(fields: ["metallib(unresolvable)"])
        }
        let fields: [(name: String, recorded: String, current: String)] = [
            ("binary_sha256", manifest.binary_sha256, current.binary_sha256),
            ("metallib_sha256", manifest.metallib_sha256, current.metallib_sha256),
            ("model_weight_digest", manifest.model_weight_digest, current.model_weight_digest),
            ("mtp_head_digest", manifest.mtp_head_digest, current.mtp_head_digest),
            ("spec_draft_n_max", "\(manifest.spec_draft_n_max)", "\(current.spec_draft_n_max)"),
            ("forced_draft_k", "\(manifest.forced_draft_k)", "\(current.forced_draft_k)"),
            ("macos_build", manifest.macos_build, current.macos_build),
            ("hardware_id", manifest.hardware_id, current.hardware_id),
        ]
        let diffs = fields
            .filter { $0.recorded != $0.current }
            .map { $0.name }
        return diffs.isEmpty ? .match : .mismatch(fields: diffs)
    }

    /// Runs the check and logs the verdict. In normal server mode the caller
    /// supplies `MLXGenerator.weightIdentityDeferred` (shared + deferred;
    /// this runs on a background Task, never the request path); the
    /// `--prewarm-check` CLI path supplies a direct computation (no requests
    /// exist in that process). Startup-only.
    static func runStartupCheck(
        logger: Logger,
        manifestPath: String,
        modelPath: String,
        mtpHeadPath: String,
        specDraftNMax: Int,
        forcedDraftK: Int,
        weightIdentity: @Sendable () async -> (digest: String, hardware: String)
    ) async -> PrewarmVerdict {
        let verdict = await Self.check(
            manifestPath: manifestPath,
            modelPath: modelPath,
            mtpHeadPath: mtpHeadPath,
            specDraftNMax: specDraftNMax,
            forcedDraftK: forcedDraftK,
            weightIdentity: weightIdentity
        )
        switch verdict {
        case .match:
            let msg = "Prewarm check: provenance MATCH (binary/metallib/model/head/"
                + "draft-geometry/os/hardware) — Metal JIT cache expected WARM."
            logger.info("\(msg)")
        case .noManifest:
            let msg = "Prewarm check: no manifest at \(manifestPath) — first-boot cold "
                + "Metal JIT expected (~15 s). Run scripts/prewarm.sh once after "
                + "install to seed the cache."
            logger.info("\(msg)")
        case .mismatch(let fields):
            let fieldList = fields.joined(separator: ", ")
            let msg = "Prewarm check: provenance MISMATCH (fields: \(fieldList)) "
                + "— the Metal JIT cache cannot be trusted; expect a cold JIT on "
                + "first use. Re-run scripts/prewarm.sh with the current build."
            logger.warning("\(msg)")
        }
        return verdict
    }

    private static func prefix(_ hex: String) -> String {
        hex.count >= 12 ? String(hex.prefix(12)) : hex
    }
}
