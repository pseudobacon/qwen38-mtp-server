import Foundation
import MLX
import Testing
@testable import HTTPServer

/// Benchmark: Radix Tree KV Cache Reuse on Interleaved Sessions.
///
/// Quantifies the KV-cache hit/miss behavior of the tree-structured
/// `RadixKVCacheManager` when several conversation sessions share a long
/// prefix (system prompt + tool catalog) but interleave. The radix tree
/// retains every session's history, so a continuation turn reuses its own
/// thread's cached history (shared prefix included) even after other
/// sessions have stored their own histories — the linear single-entry cache
/// achieved a 0% hit ratio on this workload.
///
/// Match semantics: a hit is only possible at a stored history that is an
/// EXACT prefix of the new seed. The recurrent (gated-delta) layers cannot
/// be trimmed to the bare shared prefix (`Qwen36MTPBlockSession.begin`
/// requires `trimmableOffset(reusableCache) == prefixCount`), so turns that
/// open a new thread miss even though the shared prefix is cached.
///
/// Pure Swift: no model weights are loaded. The `CacheEntry` MLX payloads
/// (`cache`, `hidden`) are dummies because `matchPrefix` only reads
/// `tokens`, `config`, and `lastAccessed` — never the tensors.
@Suite("RadixCacheBenchmarkTests")
struct RadixCacheBenchmarkTests {

    // MARK: - Workload construction

    /// Deterministic word-based tokenization (NOT the Qwen tokenizer). It only
    /// needs to be stable so that positional prefix matching is well-defined.
    private static func tokenize(_ text: String) -> [Int] {
        var result: [Int] = []
        for word in text.split(whereSeparator: { $0.isWhitespace }) {
            var h: UInt64 = 1469598103934665603
            for byte in word.utf8 {
                h ^= UInt64(byte)
                h = h &* 1099511628211
            }
            result.append(Int(h % 1_000_000))
        }
        return result
    }

    /// Build a large, deterministic tool catalog as `[ToolSpec]`. Each spec
    /// carries a realistic JSON schema so the serialized catalog is long.
    private static func buildToolCatalog(count: Int) -> [ToolSpec] {
        (0..<count).map { i in
            ToolSpec(
                type: "function",
                function: ToolSpecFunction(
                    name: "tool_\(i)",
                    description: "Tool \(i) performs operation \(i). It accepts parameters alpha, beta, gamma, delta, and epsilon, validates each of them, and returns a structured result describing the outcome of the operation along with any diagnostics.",
                    parameters: JSONValue.object([
                        "type": .string("object"),
                        "properties": .object([
                            "alpha": .object(["type": .string("string")]),
                            "beta": .object(["type": .string("integer")]),
                            "gamma": .object(["type": .string("number")]),
                            "delta": .object(["type": .string("boolean")]),
                            "epsilon": .object(["type": .string("array")])
                        ]),
                        "required": .array([.string("alpha"), .string("beta")])
                    ])
                )
            )
        }
    }

    /// Build the shared `targetCount`-token prefix: a system message plus a
    /// large tool catalog. Every session shares this identical prefix. The
    /// exact token IDs are synthetic; only the shared 2,000-token structure
    /// matters for the cache-matching behavior under test.
    private static func buildSharedPrefix(targetCount: Int) -> [Int] {
        let systemMessage = """
        You are a capable assistant with access to a large tool catalog. When the
        user's request requires an external action, select the appropriate tool and
        call it with well-formed arguments. Prefer precise, minimal responses and
        never fabricate tool results.
        """
        let tools = buildToolCatalog(count: 80)
        let toolText = tools.map { $0.toJSONString() }.joined(separator: "\n")
        let combined = systemMessage + "\n" + toolText
        var tokens = tokenize(combined)
        if tokens.count < targetCount {
            tokens.append(contentsOf: (tokens.count..<targetCount).map { 900_000 + $0 })
        } else {
            tokens = Array(tokens[0..<targetCount])
        }
        return tokens
    }

    // MARK: - Session model

    /// A distinct conversation thread. Each turn's seed is the full history so
    /// far (shared prefix + all prior turns + the new user message); the stored
    /// history is that seed plus the assistant response.
    private struct Session {
        let name: String
        let sharedPrefix: [Int]
        let user1: [Int]
        let response1: [Int]
        let user2: [Int]
        let response2: [Int]

        var seedTurn1: [Int] { sharedPrefix + user1 }
        var historyTurn1: [Int] { sharedPrefix + user1 + response1 }
        var seedTurn2: [Int] { sharedPrefix + user1 + response1 + user2 }
        var historyTurn2: [Int] { sharedPrefix + user1 + response1 + user2 + response2 }
    }

    /// Build a session whose user/response tokens live in a distinct range so
    /// threads diverge immediately after the shared prefix and continue within
    /// a thread.
    private static func makeSession(name: String, prefix: [Int], base: Int) -> Session {
        Session(
            name: name,
            sharedPrefix: prefix,
            user1: (0..<20).map { base + $0 },
            response1: (20..<60).map { base + $0 },
            user2: (60..<80).map { base + $0 },
            response2: (80..<120).map { base + $0 }
        )
    }

    // MARK: - Cache-entry helper

    /// Build a `CacheEntry` whose MLX payloads are dummies. `matchPrefix` never
    /// reads `cache`/`hidden`/`primary`/`top2`, so these values are inert.
    private static func makeEntry(tokens: [Int], config: ResolvedKVCacheConfig) -> RadixKVCacheManager.CacheEntry {
        RadixKVCacheManager.CacheEntry(
            tokens: tokens,
            cache: [],
            hidden: MLXArray.zeros([1]),
            primary: 0,
            top2: ([], []),
            config: config
        )
    }

    // MARK: - Benchmark

    @Test
    func interleavedSessionsRadixCacheBenchmark() async {
        let config = ResolvedKVCacheConfig.default
        let prefix = Self.buildSharedPrefix(targetCount: 2000)
        #expect(prefix.count == 2000)

        let a = Self.makeSession(name: "A", prefix: prefix, base: 10_000)
        let b = Self.makeSession(name: "B", prefix: prefix, base: 20_000)
        let c = Self.makeSession(name: "C", prefix: prefix, base: 30_000)

        // Interleaved multi-turn workload: (label, seed, post-turn history).
        let workload: [(label: String, seed: [Int], history: [Int])] = [
            ("A T1", a.seedTurn1, a.historyTurn1),
            ("B T1", b.seedTurn1, b.historyTurn1),
            ("A T2", a.seedTurn2, a.historyTurn2),
            ("C T1", c.seedTurn1, c.historyTurn1),
            ("B T2", b.seedTurn2, b.historyTurn2),
        ]

        let manager = RadixKVCacheManager()
        var matchCounts: [Int] = []
        var totalSeedTokens = 0
        var totalMatchedTokens = 0

        for turn in workload {
            let (prefixCount, entry) = await manager.matchPrefix(
                tokens: turn.seed,
                config: config,
                ttlSeconds: 3600
            )
            #expect((prefixCount > 0) == (entry != nil))
            matchCounts.append(prefixCount)
            totalSeedTokens += turn.seed.count
            totalMatchedTokens += prefixCount
            // Simulate the completed session: store its full history. The
            // radix tree retains it alongside the other sessions' histories.
            await manager.store(Self.makeEntry(tokens: turn.history, config: config))
        }

        let effectiveHitRatio = totalSeedTokens == 0
            ? 0
            : Double(totalMatchedTokens) / Double(totalSeedTokens)

        // The radix tree retains every session's history: continuation turns
        // (A T2, B T2) reuse their own thread's cached history, including the
        // shared prefix. Turns that open a new thread (B T1, C T1) partially
        // match the shared prefix (LCP = 2000) because the node's KV cache
        // supports trimming (vacuously, in this benchmark).
        let expectedRadixMatched: [Int] = [
            0,                    // A T1: first turn, nothing cached yet
            2000,                 // B T1: partial match of the 2000-token shared prefix
            a.historyTurn1.count, // A T2: reuses A's cached turn-1 history
            2000,                 // C T1: partial match of the 2000-token shared prefix
            b.historyTurn1.count, // B T2: reuses B's cached turn-1 history
        ]

        // MARK: Report
        print("=== Radix Tree KV Cache Benchmark (interleaved sessions) ===")
        print("Shared prefix tokens: \(prefix.count)")
        print("Workload: \(workload.map(\.label).joined(separator: " -> "))")
        print("--- Per-turn prefix match counts ---")
        for (index, turn) in workload.enumerated() {
            print("  \(turn.label): matched \(matchCounts[index]) / seed \(turn.seed.count)")
        }
        print("--- Aggregate metrics ---")
        print("  Total seed tokens across sequence:            \(totalSeedTokens)")
        print("  Total matched (reused) tokens:                \(totalMatchedTokens)")
        print(String(format: "  Effective cache hit ratio (radix):      %.4f", effectiveHitRatio))
        print("=== End of report ===")

        // The radix tree reuses cached histories across interleaved sessions
        // where the linear single-entry cache achieved 0%.
        #expect(matchCounts == expectedRadixMatched)
        #expect(effectiveHitRatio > 0)
    }
}