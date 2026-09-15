import Foundation
import MLX
import Testing
@testable import HTTPServer

/// Focused unit tests for the tree-structured `RadixKVCacheManager`:
/// radix splitting, exact-prefix matching, TTL expiration, LRU eviction,
/// and clear. Pure Swift: no model weights are loaded; the `CacheEntry` MLX
/// payloads are dummies because `matchPrefix` never reads the tensors.
@Suite("RadixKVCacheManagerTests")
struct RadixKVCacheManagerTests {

    private static func makeEntry(
        tokens: [Int],
        config: ResolvedKVCacheConfig = .default,
        namespace: String = "default"
    ) -> RadixKVCacheManager.CacheEntry {
        RadixKVCacheManager.CacheEntry(
            tokens: tokens,
            cache: [],
            hidden: MLXArray.zeros([1]),
            primary: 0,
            top2: ([], []),
            config: config,
            namespace: namespace
        )
    }

    @Test
    func storeThenMatchExactHistory() async {
        let manager = RadixKVCacheManager()
        let history = [1, 2, 3, 4, 5]
        await manager.store(Self.makeEntry(tokens: history))
        let (count, entry) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4, 5, 6, 7],
            config: .default
        )
        #expect(count == 5)
        #expect(entry?.tokens == history)
    }

    @Test
    func matchRequiresExactPrefix() async {
        let manager = RadixKVCacheManager()
        await manager.store(Self.makeEntry(tokens: [1, 2, 3, 4, 5]))
        // Diverges at token 3: LCP = 2. The cache is `[]` (vacuously
        // trimmable), so the partial match at 2 is valid.
        let (count, _) = await manager.matchPrefix(
            tokens: [1, 2, 9, 4, 5],
            config: .default
        )
        #expect(count == 2)
        // Shorter than the stored history: LCP = 3. The cache is `[]`
        // (vacuously trimmable), so the partial match at 3 is valid.
        let (count2, _) = await manager.matchPrefix(
            tokens: [1, 2, 3],
            config: .default
        )
        #expect(count2 == 3)
    }

    @Test
    func branchingSplitRetainsBothHistories() async {
        let manager = RadixKVCacheManager()
        let aHistory = [1, 2, 3, 4, 10, 11]
        let bHistory = [1, 2, 3, 4, 20, 21]
        await manager.store(Self.makeEntry(tokens: aHistory))
        await manager.store(Self.makeEntry(tokens: bHistory))
        // Both histories remain reusable after the branch split.
        let (aCount, _) = await manager.matchPrefix(
            tokens: aHistory + [12],
            config: .default
        )
        #expect(aCount == aHistory.count)
        let (bCount, _) = await manager.matchPrefix(
            tokens: bHistory + [22],
            config: .default
        )
        #expect(bCount == bHistory.count)
    }

    @Test
    func shorterHistorySplitsLonger() async {
        let manager = RadixKVCacheManager()
        let long = [1, 2, 3, 4, 5, 6]
        let short = [1, 2, 3]
        await manager.store(Self.makeEntry(tokens: long))
        await manager.store(Self.makeEntry(tokens: short))
        // The shorter history now matches at its own (split) node.
        let (shortCount, _) = await manager.matchPrefix(
            tokens: short,
            config: .default
        )
        #expect(shortCount == short.count)
        // The longer history still matches at its (suffix) node.
        let (longCount, _) = await manager.matchPrefix(
            tokens: long + [7],
            config: .default
        )
        #expect(longCount == long.count)
    }

    @Test
    func ttlExpirationIgnoresStaleNodes() async {
        let manager = RadixKVCacheManager()
        await manager.store(Self.makeEntry(tokens: [1, 2, 3]))
        // A zero TTL expires immediately: the node's age is strictly
        // positive by the time the match runs.
        let (count, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4],
            config: .default,
            ttlSeconds: 0
        )
        #expect(count == 0)
        // A generous TTL still matches.
        let (count2, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4],
            config: .default,
            ttlSeconds: 3600
        )
        #expect(count2 == 3)
    }

    @Test
    func configMismatchMisses() async {
        let manager = RadixKVCacheManager()
        let other = ResolvedKVCacheConfig(
            scheme: .fp16,
            kFormat: KVCacheFormat(kind: .f16),
            vFormat: KVCacheFormat(kind: .f16),
            groupSize: 128,
            quantizedKVStart: 0
        )
        await manager.store(Self.makeEntry(tokens: [1, 2, 3]))
        let (count, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4],
            config: other
        )
        #expect(count == 0)
    }

    @Test
    func purgeEvictsLRULeafFirst() async {
        let manager = RadixKVCacheManager()
        let aHistory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let bHistory = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
        await manager.store(Self.makeEntry(tokens: aHistory))
        await manager.store(Self.makeEntry(tokens: bHistory))

        // Both leaves are 10 tokens; the conservative per-token KV bound is
        // 262,144 bytes, so each leaf accounts for 2,621,440 bytes.
        let purged = await manager.purgeIfMemoryPressure(
            activeBytes: 6_000_000,
            limitBytes: 10_000_000,
            thresholdFraction: 0.5
        )
        #expect(purged)
        // The LRU leaf (A, stored first) is evicted. The shared prefix node
        // (5 tokens) retains a trimmed cache (vacuously trimmable in this
        // benchmark), so A's continuation partially matches at 5 tokens.
        let (aCount, _) = await manager.matchPrefix(
            tokens: aHistory + [16],
            config: .default
        )
        #expect(aCount == 5)
        // B's full history remains reusable.
        let (bCount, _) = await manager.matchPrefix(
            tokens: bHistory + [16],
            config: .default
        )
        #expect(bCount == bHistory.count)
    }

    @Test
    func clearFlushesAllState() async {
        let manager = RadixKVCacheManager()
        await manager.store(Self.makeEntry(tokens: [1, 2, 3]))
        await manager.clear()
        let (count, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4],
            config: .default
        )
        #expect(count == 0)
    }

    @Test
    func namespaceMismatchMisses() async {
        let manager = RadixKVCacheManager()
        // Store under model-A's namespace; a model-B lookup must not see it
        // even though the token prefix and KV config are identical.
        await manager.store(Self.makeEntry(tokens: [1, 2, 3], namespace: "modelA"))
        let (same, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4], config: .default, namespace: "modelA"
        )
        #expect(same == 3)
        let (cross, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4], config: .default, namespace: "modelB"
        )
        #expect(cross == 0)
        // The original namespace is unaffected by the cross-namespace miss.
        let (same2, _) = await manager.matchPrefix(
            tokens: [1, 2, 3, 4], config: .default, namespace: "modelA"
        )
        #expect(same2 == 3)
    }

    @Test
    func metricsCountHitsMissesEvictionsAndBytes() async {
        let manager = RadixKVCacheManager()
        // No lookups yet: zero lifetime counters.
        #expect(await manager.metrics().hits == 0)
        #expect(await manager.metrics().misses == 0)
        await manager.store(Self.makeEntry(tokens: [1, 2, 3, 4, 5]))
        // Hit on the stored prefix.
        let (hit, _) = await manager.matchPrefix(tokens: [1, 2, 3, 4, 5], config: .default)
        #expect(hit == 5)
        // Miss on an absent prefix (fresh root, no such branch).
        let (miss, _) = await manager.matchPrefix(tokens: [9, 9, 9], config: .default)
        #expect(miss == 0)
        let m = await manager.metrics()
        #expect(m.entries == 1)
        #expect(m.hits == 1)
        #expect(m.misses == 1)
        #expect(m.stores == 1)
        #expect(m.estimatedBytes > 0)
        #expect(abs(m.hitRate - 0.5) < 1e-9)
    }

    @Test
    func byteCapEvictsLRUOnStore() async {
        // ~2.62e6 bytes per 10-token leaf; a 3e6 cap fits exactly one leaf.
        let manager = RadixKVCacheManager(maxCacheBytes: 3_000_000)
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let b = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        await manager.store(Self.makeEntry(tokens: a))
        let (aCount, _) = await manager.matchPrefix(tokens: a, config: .default)
        #expect(aCount == a.count) // A fits under the cap
        await manager.store(Self.makeEntry(tokens: b))
        // B's insertion pushes the total over the cap, evicting LRU leaf A.
        let (aAfter, _) = await manager.matchPrefix(tokens: a, config: .default)
        #expect(aAfter == 0)
        let (bAfter, _) = await manager.matchPrefix(tokens: b, config: .default)
        #expect(bAfter == b.count)
        #expect(await manager.metrics().evictions == 1)
    }
}
