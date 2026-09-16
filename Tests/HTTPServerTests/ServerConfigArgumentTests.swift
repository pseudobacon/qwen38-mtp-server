import XCTest
@testable import HTTPServer

/// Pure-Swift tests for the CLI argument contract:
/// - `vaporArguments(from:)` must strip EVERY implemented `ServerConfig`
///   flag (and its value) so Vapor's command dispatcher never sees them.
/// - `unknownFlagError(in:)` must reject any `--flag` the server does not
///   implement, so unknown flags (e.g. `--kv-ssd-cache-gbbs`) fail loudly at
///   startup instead of leaking through to `Environment.detect` /
///   `app.execute()`.
final class ServerConfigArgumentTests: XCTestCase {

    /// Every flag `ServerConfig.fromCommandLine` recognizes, paired with a
    /// dummy value for the flags that take one. Kept in sync with
    /// `valueTakingFlags` / `valuelessFlags` / the `fromCommandLine` switch.
    private static let valueTakingFlags: [String] = [
        "--host", "-H",
        "--port", "-p",
        "--model", "-m",
        "--mtp-head",
        "--model-aliases",
        "--ctx-size", "-c",
        "--n-predict", "--predict", "-n",
        "--temp", "--temperature",
        "--top-k",
        "--top-p",
        "--min-p",
        "--repeat-penalty",
        "--presence-penalty",
        "--frequency-penalty",
        "--spec-draft-n-max",
        "--spec-draft-calibrate-depths",
        "--spec-draft-calibrate-tokens",
        "--spec-draft-calibration-file",
        "--spec-draft-adaptive-window",
        "--spec-draft-adaptive-threshold-high",
        "--spec-draft-adaptive-threshold-low",
        "--spec-draft-adaptive-hysteresis",
        "--prefill-chunk-size",
        "--cache-type-k", "-ctk",
        "--cache-type-v", "-ctv",
        "--kv-scheme",
        "--kv-group-size",
        "--kv-bits",
        "--kv-tail-size", "--cache-type-v-tail",
        "--kv-ssd-cache-dir",
        "--kv-ssd-cache-gb",
        "--kv-ssd-ttl-seconds",
        "--memory-limit",
        "--max-queue-depth",
    ]

    /// Every flag that takes no value.
    private static let valuelessFlags: [String] = [
        "--help",
        "--tools-enabled",
        "--tools-disabled",
        "--spec-draft-calibrate",
        "--spec-draft-adaptive",
        "--kv-ssd-enabled",
        "--kv-ssd-disabled",
    ]

    private func fullFlagCommandLine(exe: String = "/usr/local/bin/HTTPServer") -> [String] {
        var args = [exe]
        for flag in Self.valueTakingFlags { args.append(contentsOf: [flag, "dummy-value"]) }
        args.append(contentsOf: Self.valuelessFlags)
        return args
    }

    func testVaporArgumentsStripsAllKnownFlags() {
        let args = fullFlagCommandLine()
        let kept = ServerConfig.vaporArguments(from: args)
        XCTAssertEqual(kept, [args[0]],
                       "vaporArguments must strip every implemented ServerConfig flag and its value")
    }

    func testVaporArgumentsKeepsVaporNativeArgsAndPositionals() {
        let kept = ServerConfig.vaporArguments(from: [
            "/usr/local/bin/HTTPServer",
            "serve",
            "--env", "production",
            "--port", "8080",
            "--kv-scheme", "turbo8v4",
            "--tools-enabled",
        ])
        XCTAssertEqual(kept, ["/usr/local/bin/HTTPServer", "serve", "--env", "production"])
    }

    func testVaporArgumentsStripsValuelessFlagsWithoutConsumingTheirValue() {
        // `--tools-enabled` is valueless; the following flag must survive.
        let kept = ServerConfig.vaporArguments(from: [
            "/usr/local/bin/HTTPServer",
            "--tools-enabled",
            "--spec-draft-calibrate",
            "--prefill-chunk-size", "1024",
        ])
        XCTAssertEqual(kept, ["/usr/local/bin/HTTPServer"])
    }

    func testUnknownFlagErrorDetectsUnknownValueTakingFlag() {
        // `--kv-ssd-cache-dir` is implemented (Task: Radix SSD persistence);
        // a typo of it must still fail loudly.
        let args = fullFlagCommandLine() + ["--kv-ssd-cache-dire", "/tmp/kv-ssd"]
        let error = ServerConfig.unknownFlagError(in: args)
        XCTAssertNotNil(error)
        XCTAssertTrue(error!.contains("--kv-ssd-cache-dire"))
    }

    func testUnknownFlagErrorDetectsTypoedFlag() {
        let error = ServerConfig.unknownFlagError(in: ["/usr/local/bin/HTTPServer", "--prefill-chunk", "512"])
        XCTAssertNotNil(error)
    }

    func testUnknownFlagErrorAcceptsFullKnownFlagSet() {
        let error = ServerConfig.unknownFlagError(in: fullFlagCommandLine())
        XCTAssertNil(error, "every implemented flag must be accepted")
    }

    func testUnknownFlagErrorIgnoresPositionalArguments() {
        // Positional args (Vapor command names) are not flags and are not
        // our concern.
        let error = ServerConfig.unknownFlagError(in: ["/usr/local/bin/HTTPServer", "serve"])
        XCTAssertNil(error)
    }

    func testUnknownFlagErrorDoesNotConsumeValuesOfUnknownFlags() {
        // A value-looking token after an unknown flag must not mask a
        // second unknown flag.
        let args = ["/usr/local/bin/HTTPServer", "--bogus", "value", "--also-bogus"]
        let error = ServerConfig.unknownFlagError(in: args)
        XCTAssertNotNil(error)
        XCTAssertTrue(error!.contains("--bogus"))
    }
}
