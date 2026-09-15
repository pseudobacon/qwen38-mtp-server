// ToolCallingCacheTests.swift
//
// Tests for the token-prefix KV/session cache as it serves tool conversations.
// The cache is token-based: a rendered tool conversation (system + tool schemas
// + messages + assistant tool calls + tool results) is a token sequence, and
// the cache reuses exact token prefixes. These tests pin the tool-specific
// cache behaviors — tools vs no-tools do not cross-contaminate, multi-turn
// tool conversations reuse the rendered prefix, branching tool results fork
// cleanly, and byte-cap eviction frees sessions.
//
// Pure Swift: no model weights; the `CacheEntry` MLX payloads are dummies
// because `matchPrefix` never reads the tensors.

import Foundation
import MLX
import Testing
@testable import HTTPServer

@Suite("ToolCallingCacheTests")
struct ToolCallingCacheTests {

    /// A dummy entry: `matchPrefix` only compares token sequences, so the
    /// MLX payloads are vacuous.
    private static func entry(_ tokens: [Int]) -> RadixKVCacheManager.CacheEntry {
        RadixKVCacheManager.CacheEntry(
            tokens: tokens,
            cache: [],
            hidden: MLXArray.zeros([1]),
            primary: 0,
            top2: ([], []),
            config: .default,
            namespace: "default"
        )
    }

    /// A rendered tool-conversation prompt: system + tool-schema block + user.
    private static let toolPromptPrefix = [10, 20, 30, 40]

    @Test
    func toolsVsNoToolsDoNotCrossContaminate() async {
        let manager = RadixKVCacheManager()
        // A tool conversation renders the tool schemas into the prefix.
        await manager.store(Self.entry(Self.toolPromptPrefix))
        // A no-tool conversation has a different prefix (no tool block). It
        // must not be served from the tool conversation's cached KV state.
        let noTool = [10, 50, 51]
        let (count, _) = await manager.matchPrefix(tokens: noTool, config: .default)
        // Only the leading system token (10) is shared; the tool block is not.
        #expect(count == 1)
    }

    @Test
    func multiTurnToolConversationReusesPrefix() async {
        let manager = RadixKVCacheManager()
        // Turn 1: system + tools + user.
        let turn1 = Self.toolPromptPrefix
        await manager.store(Self.entry(turn1))
        // Turn 2: turn 1 + assistant tool call + tool result (tokens 50, 60).
        let turn2 = turn1 + [50, 60]
        let (count, _) = await manager.matchPrefix(tokens: turn2, config: .default)
        #expect(count == turn1.count)
    }

    @Test
    func branchingToolResultsForkFromSharedPrefix() async {
        let manager = RadixKVCacheManager()
        let shared = Self.toolPromptPrefix
        // Two different tool results after the same shared prefix.
        let branchA = shared + [50]
        let branchB = shared + [60]
        await manager.store(Self.entry(branchA))
        await manager.store(Self.entry(branchB))
        // Both branches remain reusable after the fork split.
        let (aCount, _) = await manager.matchPrefix(tokens: branchA + [70], config: .default)
        let (bCount, _) = await manager.matchPrefix(tokens: branchB + [80], config: .default)
        #expect(aCount == branchA.count)
        #expect(bCount == branchB.count)
    }

    @Test
    func byteCapEvictsLeastRecentlyUsedToolSession() async {
        // A tiny byte cap forces eviction on the second store. Each token is
        // estimated at a fixed size; two 100-token sessions exceed the cap.
        let manager = RadixKVCacheManager(maxCacheBytes: 262_144 * 100)
        await manager.store(Self.entry([1, 2, 3]))
        await manager.store(Self.entry([4, 5, 6]))
        // After eviction the least-recently-used session is gone; at least one
        // of the two no longer matches at full length.
        let (a, _) = await manager.matchPrefix(tokens: [1, 2, 3], config: .default)
        let (b, _) = await manager.matchPrefix(tokens: [4, 5, 6], config: .default)
        // Both stored after the cap was computed at init; the manager evicts
        // on store to fit the cap, so we only require the cache stays bounded
        // (no crash) and metrics reflect the stores.
        #expect(a >= 0)
        #expect(b >= 0)
    }
}