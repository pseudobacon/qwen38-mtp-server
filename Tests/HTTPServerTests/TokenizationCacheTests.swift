// TokenizationCacheTests.swift
//
// Unit tests for the bounded, actor-isolated prompt tokenization cache
// (Stage 0 of the prefix-cache RFC). The cache is pure Swift (no MLX, no model
// weights), so these tests run on any machine without loading the model.
//
// Coverage: key isolation, TTL expiry/refresh, LRU eviction, entry/byte
// bounds, failure-safe behavior, no cross-config hits, and a benchmark of the
// cache's own per-operation overhead.

import Testing
import Foundation
@testable import HTTPServer

/// A controllable clock for deterministic TTL tests.
private final class TestClock: @unchecked Sendable {
    var now: Date
    init(now: Date) { self.now = now }
    func advance(_ seconds: TimeInterval) { now += seconds }
}

private func makeKey(
    namespace: String = "ns",
    enableThinking: Bool = false,
    formattedPrompt: String
) -> TokenizationCache.Key {
    TokenizationCache.Key(
        namespace: namespace,
        enableThinking: enableThinking,
        formattedPrompt: formattedPrompt
    )
}

// MARK: - Basic hit / miss

@Test func testInsertThenLookupIsHit() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let key = makeKey(formattedPrompt: "hello")
    cache.insert(key: key, tokenIDs: [1, 2, 3])

    let (ids, wasHit) = cache.lookup(key: key)
    #expect(wasHit)
    #expect(ids == [1, 2, 3])
}

@Test func testLookupOfUnknownKeyIsMiss() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let (ids, wasHit) = cache.lookup(key: makeKey(formattedPrompt: "absent"))
    #expect(!wasHit)
    #expect(ids == nil)
}

@Test func testInsertReplacesExistingKey() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let key = makeKey(formattedPrompt: "hello")
    cache.insert(key: key, tokenIDs: [1])
    cache.insert(key: key, tokenIDs: [9, 8])

    let (ids, wasHit) = cache.lookup(key: key)
    #expect(wasHit)
    #expect(ids == [9, 8])
    #expect(cache.stats().entryCount == 1)
}

// MARK: - Key isolation / no cross-config hits

@Test func testDifferentEnableThinkingIsDistinct() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let a = makeKey(enableThinking: false, formattedPrompt: "same")
    let b = makeKey(enableThinking: true, formattedPrompt: "same")
    cache.insert(key: a, tokenIDs: [1])

    let (ids, wasHit) = cache.lookup(key: b)
    #expect(!wasHit)
    #expect(ids == nil)
    #expect(cache.stats().entryCount == 1)
}

@Test func testDifferentNamespaceIsDistinct() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let a = makeKey(namespace: "model-a", formattedPrompt: "same")
    let b = makeKey(namespace: "model-b", formattedPrompt: "same")
    cache.insert(key: a, tokenIDs: [1])

    let (ids, wasHit) = cache.lookup(key: b)
    #expect(!wasHit)
    #expect(ids == nil)
}

@Test func testDifferentFormattedPromptIsDistinct() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let a = makeKey(formattedPrompt: "one")
    let b = makeKey(formattedPrompt: "two")
    cache.insert(key: a, tokenIDs: [1])

    let (ids, wasHit) = cache.lookup(key: b)
    #expect(!wasHit)
    #expect(ids == nil)
}

// MARK: - TTL

@Test func testExpiredEntryIsEvictedAndCountedAsMiss() {
    let clock = TestClock(now: Date(timeIntervalSince1970: 1_000_000))
    var cache = TokenizationCache(
        maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 10, clock: { clock.now }
    )
    let key = makeKey(formattedPrompt: "ttl")
    cache.insert(key: key, tokenIDs: [1])

    clock.advance(11) // past the 10s TTL
    let (ids, wasHit) = cache.lookup(key: key)
    #expect(!wasHit)
    #expect(ids == nil)
    #expect(cache.stats().entryCount == 0)
    #expect(cache.stats().misses == 1)
}

@Test func testHitRefreshesTTL() {
    let clock = TestClock(now: Date(timeIntervalSince1970: 1_000_000))
    var cache = TokenizationCache(
        maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 10, clock: { clock.now }
    )
    let key = makeKey(formattedPrompt: "ttl")
    cache.insert(key: key, tokenIDs: [1])

    clock.advance(8) // within the original TTL
    let (_, hit1) = cache.lookup(key: key) // hit; refreshes TTL to now+10
    #expect(hit1)

    clock.advance(9) // 17s after insert, but 9s after the refresh
    let (_, hit2) = cache.lookup(key: key) // still alive thanks to the refresh
    #expect(hit2)
}

// MARK: - LRU eviction

@Test func testLRUEvictsLeastRecentlyUsed() {
    var cache = TokenizationCache(maxEntries: 2, maxBytes: 1 << 20, defaultTTL: 300)
    let a = makeKey(formattedPrompt: "a")
    let b = makeKey(formattedPrompt: "b")
    let c = makeKey(formattedPrompt: "c")

    cache.insert(key: a, tokenIDs: [1])
    cache.insert(key: b, tokenIDs: [2])
    cache.insert(key: c, tokenIDs: [3]) // evicts a (least recently used)

    #expect(cache.stats().entryCount == 2)
    #expect(cache.stats().evictions == 1)
    #expect(cache.lookup(key: a).1 == false)
    #expect(cache.lookup(key: b).1 == true)
    #expect(cache.lookup(key: c).1 == true)
}

@Test func testHitMovesEntryToFront() {
    var cache = TokenizationCache(maxEntries: 2, maxBytes: 1 << 20, defaultTTL: 300)
    let a = makeKey(formattedPrompt: "a")
    let b = makeKey(formattedPrompt: "b")
    let c = makeKey(formattedPrompt: "c")

    cache.insert(key: a, tokenIDs: [1])
    cache.insert(key: b, tokenIDs: [2])
    _ = cache.lookup(key: a) // a becomes most recently used
    cache.insert(key: c, tokenIDs: [3]) // evicts b (now least recently used)

    #expect(cache.lookup(key: a).1 == true)
    #expect(cache.lookup(key: b).1 == false)
    #expect(cache.lookup(key: c).1 == true)
}

// MARK: - Bounds

@Test func testEntryBoundIsEnforced() {
    var cache = TokenizationCache(maxEntries: 3, maxBytes: 1 << 20, defaultTTL: 300)
    for i in 0..<10 {
        cache.insert(key: makeKey(formattedPrompt: "p\(i)"), tokenIDs: [i])
    }
    #expect(cache.stats().entryCount <= 3)
}

@Test func testByteBoundIsEnforced() {
    var cache = TokenizationCache(maxEntries: 1024, maxBytes: 64, defaultTTL: 300)
    // Each entry's byte size is prompt.utf8.count + tokenIDs.count * 8.
    // "p" (1 byte) + 1 token (8 bytes) = 9 bytes per entry, so 64 bytes
    // holds at most 7 entries.
    for i in 0..<20 {
        cache.insert(key: makeKey(formattedPrompt: "p"), tokenIDs: [i])
        // Same key each time, so this replaces rather than grows. Use a
        // distinct prompt to force growth.
        cache.insert(
            key: makeKey(formattedPrompt: "p\(i)"),
            tokenIDs: [i]
        )
    }
    #expect(cache.stats().byteCount <= 64)
    #expect(cache.stats().entryCount >= 1)
}

// MARK: - Failure-safe behavior

@Test func testEmptyTokenIDsAreStoredAndCounted() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 1 << 20, defaultTTL: 300)
    let key = makeKey(formattedPrompt: "empty")
    cache.insert(key: key, tokenIDs: [])

    let (ids, wasHit) = cache.lookup(key: key)
    #expect(wasHit)
    #expect(ids == [])
}

@Test func testLargeInsertStaysWithinByteBound() {
    var cache = TokenizationCache(maxEntries: 16, maxBytes: 100, defaultTTL: 300)
    // A single huge entry exceeds the byte bound; it must be evicted so the
    // cache never reports a byteCount above the bound.
    cache.insert(key: makeKey(formattedPrompt: "big"), tokenIDs: Array(0..<1000))
    #expect(cache.stats().byteCount <= 100)
}

@Test func testStatsAreConsistentAfterEvictions() {
    var cache = TokenizationCache(maxEntries: 2, maxBytes: 1 << 20, defaultTTL: 300)
    for i in 0..<5 {
        cache.insert(key: makeKey(formattedPrompt: "p\(i)"), tokenIDs: [i])
    }
    let stats = cache.stats()
    #expect(stats.entryCount == 2)
    // 5 inserts, 3 evictions, 0 hits, 0 misses.
    #expect(stats.evictions == 3)
    #expect(stats.hits == 0)
    #expect(stats.misses == 0)
}

// MARK: - Benchmark (cache's own per-operation overhead)

@Test func benchmarkCacheHitOverhead() {
    let clock = TestClock(now: Date(timeIntervalSince1970: 1_000_000))
    var cache = TokenizationCache(
        maxEntries: 1024, maxBytes: 1 << 20, defaultTTL: 300, clock: { clock.now }
    )
    let key = makeKey(formattedPrompt: "hello")
    cache.insert(key: key, tokenIDs: [1, 2, 3])

    let n = 100_000
    let start = ContinuousClock.now
    for _ in 0..<n {
        _ = cache.lookup(key: key)
    }
    let duration = start.duration(to: .now)
    let seconds = Double(duration.components.seconds)
        + Double(duration.components.attoseconds) / 1e18
    let perOpMicros = seconds / Double(n) * 1e6

    // The cache hit overhead must be negligible: well under 100 microseconds
    // per operation. The real-world benefit of a hit is that it skips the
    // CPU cost of `tokenizer.encode` entirely.
    #expect(perOpMicros < 100)
}
