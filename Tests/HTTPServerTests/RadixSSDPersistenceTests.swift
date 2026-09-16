// RadixSSDStoreTests.swift
//
// Pure (no-weights, no-model) tests for the Radix SSD persistence tier. These
// exercise the flat prefix-set skeleton semantics on the in-RAM manager
// (restore + disk-aware match), the file-base key stability, the index
// Codable round-trip, disk TTL expiry, and KV-config mismatch.
//
// The weight-gated round-trip (real tensor save/load + token-identical
// generation) is in RadixSSDWeightTests.swift.

import Foundation
import Testing
import MLX
@testable import HTTPServer

private func makeStore() throws -> RadixSSDStore {
    let dir = URL(fileURLWithPath: "/tmp/radix-ssd-test-\(UUID().uuidString)")
    let config = KVSSDConfig(
        enabled: true, directory: dir, budgetGB: 1, ttlSeconds: 86400)
    let dims = KVSSDKeyDimensions(weightDigest: "digest-A", templateHash: "tmpl-A")
    return RadixSSDStore(config: config, keyDims: dims)
}

private func skeletonNode(_ prefix: [Int], fileBase: String? = nil) -> KVSSDIndexNode {
    KVSSDIndexNode(
        fileBase: fileBase ?? prefix.map(String.init).joined(separator: "-"),
        fullPrefix: prefix,
        createdAt: Date(),
        bytes: 100,
        primary: 0,
        top2IDs: [1, 2],
        top2Values: [0.5, 0.25],
        config: nil,
        cacheCounts: nil,
        cacheMetaStates: nil)
}

struct RadixSSDPersistenceTests {

    @Test func fileBaseStableAndDistinct() throws {
        let store = try makeStore()
        let a1 = store.fileBase(fullPrefix: [1, 2, 3])
        let a2 = store.fileBase(fullPrefix: [1, 2, 3])
        let b = store.fileBase(fullPrefix: [1, 2, 4])
        #expect(a1 == a2, "same prefix must map to the same file base")
        #expect(a1 != b, "different prefixes must map to different file bases")
        #expect(!a1.isEmpty)
    }

    @Test func indexCodableRoundTrip() throws {
        let dims = KVSSDKeyDimensions(weightDigest: "w", templateHash: "t")
        let nodes: [KVSSDIndexNode] = [
            skeletonNode([1, 2, 3]), skeletonNode([1, 2, 4]),
        ]
        let index = KVSSDIndex(version: KVSSDIndex.version, keyDims: dims, nodes: nodes)
        let data = try JSONEncoder().encode(index)
        let decoded = try JSONDecoder().decode(KVSSDIndex.self, from: data)
        #expect(decoded.keyDims == dims)
        #expect(decoded.nodes.count == nodes.count)
        #expect(decoded.nodes[0].fullPrefix == [1, 2, 3])
        #expect(decoded.nodes[1].fullPrefix == [1, 2, 4])
    }

    @Test func skeletonRestoreDiskMatchFullPrefix() async throws {
        let manager = RadixKVCacheManager()
        await manager.restoreSkeleton(nodes: [skeletonNode([1, 2, 3, 4, 5, 6])])
        // Query that fully matches the restored node.
        let (prefixCount, entry, diskHit, _) = await manager.matchPrefixForGeneration(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8],
            config: .default)
        #expect(prefixCount == 6)
        #expect(entry == nil, "a disk hit has no in-RAM entry yet")
        #expect(diskHit != nil, "a disk hit must carry the diskState")
        #expect(diskHit?.fullPrefix == [1, 2, 3, 4, 5, 6])
    }

    /// The flat prefix-set model: internal shared-prefix nodes are themselves
    /// set entries, so a query that only reaches the shared prefix (not a full
    /// leaf) matches that internal node.
    @Test func skeletonRestoreDiskMatchInternalNode() async throws {
        let manager = RadixKVCacheManager()
        await manager.restoreSkeleton(nodes: [
            skeletonNode([1, 2, 3, 4, 5, 6]),
            skeletonNode([1, 2, 3, 7, 8, 9]),
            skeletonNode([1, 2, 3]),  // the shared prefix, its own set entry
        ])
        // Query matches the shared prefix [1,2,3] but diverges before any leaf.
        let (prefixCount, _, diskHit, _) = await manager.matchPrefixForGeneration(
            tokens: [1, 2, 3, 9, 9, 9],
            config: .default)
        #expect(prefixCount == 3)
        #expect(diskHit?.fullPrefix == [1, 2, 3])
    }

    @Test func skeletonTTLExpiryIsMiss() async throws {
        let manager = RadixKVCacheManager()
        // A node created in the far past (beyond the disk TTL).
        let oldNode = KVSSDIndexNode(
            fileBase: "old", fullPrefix: [1, 2, 3],
            createdAt: Date().addingTimeInterval(-999_999),
            bytes: 100, primary: 0, top2IDs: [1], top2Values: [0.5],
            config: nil, cacheCounts: nil, cacheMetaStates: nil)
        await manager.restoreSkeleton(nodes: [oldNode])
        let (prefixCount, _, diskHit, _) = await manager.matchPrefixForGeneration(
            tokens: [1, 2, 3, 4],
            config: .default,
            diskTTLSeconds: 60)
        #expect(prefixCount == 0)
        #expect(diskHit == nil, "an expired disk node must be a miss")
    }

    @Test func skeletonConfigMismatchIsMiss() async throws {
        let manager = RadixKVCacheManager()
        await manager.restoreSkeleton(nodes: [skeletonNode([1, 2, 3])])
        // A different KV config must not match (KV format / scheme changed).
        let other = ResolvedKVCacheConfig(
            scheme: .affine8,
            kFormat: KVCacheFormat(kind: .q8_0),
            vFormat: KVCacheFormat(kind: .q8_0),
            groupSize: 64,
            quantizedKVStart: 0)
        let (prefixCount, _, diskHit, _) = await manager.matchPrefixForGeneration(
            tokens: [1, 2, 3, 4],
            config: other)
        #expect(prefixCount == 0)
        #expect(diskHit == nil)
    }
}
