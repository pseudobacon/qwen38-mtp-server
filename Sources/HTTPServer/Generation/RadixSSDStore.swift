// RadixSSDStore.swift
//
// Cold disk tier beneath the in-RAM radix KV cache. See
// docs/radix-ssd-persistence-rfc.md for the full design.
//
// Model: the disk tier is a flat *prefix set* — one entry per stateful radix
// node, keyed by its full token prefix. Every stateful node (complete-session
// leaves AND internal shared-prefix nodes created by splits) is a distinct
// entry, so the radix tree's LCP match semantics (including partial matches at
// internal shared-prefix nodes) survive restore. On startup the in-RAM tree is
// rebuilt as a skeleton (one node per set entry, no tensors); on a `matchPrefix`
// hit the node's tensors are lazy-loaded on the model lane.
//
// Responsibilities (all invoked on the model lane only):
//   * Serialize eligible radix nodes to `nodes/<fileBase>.safetensors` and
//     record them in a single prefix-set index (`index.json`) that is the
//     visibility commit point (temp + rename).
//   * Load the index on startup (skeleton, no tensors) and lazy-load a node's
//     tensors on a `matchPrefix` hit.
//   * Enforce a byte budget by sweeping orphaned / LRU-oldest files.
//
// Failure discipline: any I/O, parse, or key-dimension problem yields a miss
// (the caller falls back to full prefill). The generation path never sees a
// thrown error from this store.
//
// Access is `internal` because the API exposes `RadixKVCacheManager.CacheEntry`
// (a type nested in the internal manager actor).

import Foundation
import CryptoKit
import MLX
import MLXLMCommon

// MARK: - Cross-restart key dimensions

/// The identity dimensions that must match for a disk entry to be valid. A
/// disk hit is a miss if ANY of these differs from the current process.
struct KVSSDKeyDimensions: Codable, Equatable, Sendable {
    /// SHA-256 tree over the weights dir (see `MLXGenerator.weightIdentity`).
    let weightDigest: String
    /// SHA-256 of the chat-template string (redundant with the token-id prefix
    /// but cheap insurance against tokenizer drift).
    let templateHash: String

    init(weightDigest: String, templateHash: String) {
        self.weightDigest = weightDigest
        self.templateHash = templateHash
    }
}

// MARK: - Configuration

struct KVSSDConfig: Sendable {
    var enabled: Bool
    var directory: URL
    var budgetBytes: Int
    var ttlSeconds: Int

    init(enabled: Bool, directory: URL, budgetGB: Int, ttlSeconds: Int) {
        self.enabled = enabled
        self.directory = directory
        self.budgetBytes = budgetGB * 1024 * 1024 * 1024
        self.ttlSeconds = ttlSeconds
    }

    var nodesDir: URL { directory.appendingPathComponent("nodes", isDirectory: true) }
    var indexURL: URL { directory.appendingPathComponent("index.json") }
}

// MARK: - Codable KV-config projection

/// `ResolvedKVCacheConfig` is `Sendable`/`Equatable` but not `Codable`. We
/// project it to its defining fields (which also fully encode the KV format:
/// f16 vs quantized + group size + quantizedKVStart).
struct KVSSDConfigCode: Codable, Equatable, Sendable {
    let scheme: String
    let kKind: String
    let vKind: String
    let groupSize: Int
    let quantizedKVStart: Int

    init(_ config: ResolvedKVCacheConfig) {
        self.scheme = config.scheme.rawValue
        self.kKind = config.kFormat.kind.rawValue
        self.vKind = config.vFormat.kind.rawValue
        self.groupSize = config.groupSize
        self.quantizedKVStart = config.quantizedKVStart
    }

    var resolved: ResolvedKVCacheConfig? {
        guard let scheme = KVCacheScheme(rawValue: scheme),
              let kKind = KVCacheFormat.Kind(rawValue: kKind),
              let vKind = KVCacheFormat.Kind(rawValue: vKind)
        else { return nil }
        return ResolvedKVCacheConfig(
            scheme: scheme,
            kFormat: KVCacheFormat(kind: kKind),
            vFormat: KVCacheFormat(kind: vKind),
            groupSize: groupSize,
            quantizedKVStart: quantizedKVStart
        )
    }
}

// MARK: - Prefix-set index (the visibility commit point)

/// One prefix-set entry: a stateful radix node keyed by its full token prefix.
/// `fileBase` names the safetensors file; the state metadata (primary/top2/
/// config/cacheCounts/cacheMetaStates) is everything needed to lazy-load
/// without re-reading the index.
struct KVSSDIndexNode: Codable, Equatable, Sendable {
    let fileBase: String
    let fullPrefix: [Int]
    let createdAt: Date
    let bytes: Int
    let primary: Int?
    let top2IDs: [Int]?
    let top2Values: [Double]?
    let config: KVSSDConfigCode?
    let cacheCounts: [Int]?
    let cacheMetaStates: [[String]]?
    /// The radix-cache namespace this entry belongs to (the canonical
    /// manager is multi-namespace; the recovered lineage was single).
    let namespace: String

    init(
        fileBase: String, fullPrefix: [Int], createdAt: Date, bytes: Int,
        primary: Int?, top2IDs: [Int]?, top2Values: [Double]?,
        config: KVSSDConfigCode?, cacheCounts: [Int]?, cacheMetaStates: [[String]]?,
        namespace: String = "default"
    ) {
        self.fileBase = fileBase
        self.fullPrefix = fullPrefix
        self.createdAt = createdAt
        self.bytes = bytes
        self.primary = primary
        self.top2IDs = top2IDs
        self.top2Values = top2Values
        self.config = config
        self.cacheCounts = cacheCounts
        self.cacheMetaStates = cacheMetaStates
        self.namespace = namespace
    }
}

struct KVSSDIndex: Codable, Sendable {
    static let version = 1
    let version: Int
    let keyDims: KVSSDKeyDimensions
    var nodes: [KVSSDIndexNode]

    init(version: Int, keyDims: KVSSDKeyDimensions, nodes: [KVSSDIndexNode]) {
        self.version = version
        self.keyDims = keyDims
        self.nodes = nodes
    }
}

// MARK: - Store

/// Stateless persistence engine (all mutable state lives on disk; the index
/// is the source of truth, re-read per call). A shared reference type so the
/// manager's eviction callback and the generator can both use it safely. All
/// methods run on the model lane; `@unchecked Sendable` because the `let`
/// fields are immutable and every method is pure over the on-disk state.
final class RadixSSDStore: @unchecked Sendable {
    let config: KVSSDConfig
    let keyDims: KVSSDKeyDimensions

    init(config: KVSSDConfig, keyDims: KVSSDKeyDimensions) {
        self.config = config
        self.keyDims = keyDims
    }

    // MARK: Directory layout

    func nodeURL(_ fileBase: String) -> URL {
        config.nodesDir.appendingPathComponent(fileBase + ".safetensors")
    }

    func ensureDirectories() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: config.directory, withIntermediateDirectories: true)
        try fm.createDirectory(at: config.nodesDir, withIntermediateDirectories: true)
    }

    // MARK: Index I/O (atomic)

    func loadIndex() -> KVSSDIndex? {
        guard let data = try? Data(contentsOf: config.indexURL) else { return nil }
        let decoder = Foundation.JSONDecoder()
        return try? decoder.decode(KVSSDIndex.self, from: data)
    }

    @discardableResult
    func writeIndex(_ index: KVSSDIndex) -> Bool {
        do {
            try ensureDirectories()
            let encoder = Foundation.JSONEncoder()
            let data = try encoder.encode(index)
            let tmp = config.indexURL
                .deletingLastPathComponent()
                .appendingPathComponent("index.json.tmp")
            try data.write(to: tmp, options: .atomic)
            try FileManager.default.replaceItem(
                at: config.indexURL, withItemAt: tmp,
                backupItemName: nil, resultingItemURL: nil)
            return true
        } catch {
            return false
        }
    }

    // MARK: Per-node safetensors I/O

    @discardableResult
    func writeNodeArrays(_ fileBase: String, _ arrays: [String: MLXArray]) -> Int? {
        do {
            try ensureDirectories()
            let url = nodeURL(fileBase)
            // MLX 0.31.6 exposes safetensors I/O as free functions in the
            // MLX module (not statics): `save(arrays:metadata:url:stream:)`
            // and `loadArrays(url:stream:)`.
            try save(arrays: arrays, url: url)
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            return (attrs?[.size] as? NSNumber)?.intValue ?? 0
        } catch {
            return nil
        }
    }

    func loadNodeArrays(_ fileBase: String) -> [String: MLXArray]? {
        try? loadArrays(url: nodeURL(fileBase))
    }

    // MARK: CacheEntry encode / decode

    /// Flatten a `CacheEntry`'s MLX state into a named-array dictionary plus
    /// the per-cache counts / metaState needed to rebuild it.
    func encodeEntry(_ entry: RadixKVCacheManager.CacheEntry)
        -> (arrays: [String: MLXArray], counts: [Int], metaStates: [[String]])
    {
        var arrays: [String: MLXArray] = [:]
        arrays["hidden"] = entry.hidden
        var counts: [Int] = []
        var metaStates: [[String]] = []
        for (i, cache) in entry.cache.enumerated() {
            let state = cache.state
            for (j, arr) in state.enumerated() {
                arrays["cache.\(i).\(j)"] = arr
            }
            counts.append(state.count)
            metaStates.append(cache.metaState)
        }
        return (arrays, counts, metaStates)
    }

    /// Rebuild a `CacheEntry` from loaded arrays + a fresh cache structure
    /// (from `model.newCache(parameters:)`). Returns nil on any mismatch.
    func decodeEntry(
        node: KVSSDIndexNode,
        arrays: [String: MLXArray],
        freshCache: [any KVCache]
    ) -> RadixKVCacheManager.CacheEntry? {
        guard node.fileBase != "",
              let counts = node.cacheCounts,
              let metaStates = node.cacheMetaStates,
              let config = node.config?.resolved,
              let primary = node.primary,
              let top2IDs = node.top2IDs,
              let top2Values = node.top2Values,
              let hidden = arrays["hidden"],
              freshCache.count == counts.count,
              freshCache.count == metaStates.count
        else { return nil }
        let caches = freshCache
        for i in 0..<counts.count {
            var state: [MLXArray] = []
            var ok = true
            for j in 0..<counts[i] {
                guard let arr = arrays["cache.\(i).\(j)"] else { ok = false; break }
                state.append(arr)
            }
            guard ok else { return nil }
            restoreKVCacheState(cache: caches[i], state: state, metaState: metaStates[i])
        }
        return RadixKVCacheManager.CacheEntry(
            tokens: node.fullPrefix,
            cache: caches,
            hidden: hidden,
            primary: primary,
            top2: (top2IDs, top2Values),
            config: config,
            namespace: node.namespace
        )
    }

    // MARK: Tree snapshot / persist

    /// A manager-provided snapshot of one stateful tree node. `fullPrefix` is
    /// the concatenation of token segments from the root to the node.
    struct SnapshotNode: Sendable {
        let fullPrefix: [Int]
        let entry: RadixKVCacheManager.CacheEntry
        let createdAt: Date
    }

    /// Stable file base name for a node: SHA-256 over the canonical JSON of
    /// (keyDims, fullTokenPrefix). Re-writing the same node is idempotent.
    func fileBase(fullPrefix: [Int]) -> String {
        Data(fullPrefix.canonicalKeyJSON(keyDims: keyDims).utf8).sha256Hex
    }

    /// Serialize every stateful node and atomically commit the index. This is
    /// the graceful-shutdown write point.
    @discardableResult
    func persistTree(_ nodes: [SnapshotNode]) -> Bool {
        var indexNodes: [KVSSDIndexNode] = []
        indexNodes.reserveCapacity(nodes.count)
        for node in nodes {
            let fileBase = self.fileBase(fullPrefix: node.fullPrefix)
            guard let bytes = writeNodeArrays(
                fileBase, encodeEntry(node.entry).arrays
            ) else { continue } // half-written node: skip (never visible)
            let entry = node.entry
            indexNodes.append(KVSSDIndexNode(
                fileBase: fileBase, fullPrefix: node.fullPrefix,
                createdAt: node.createdAt, bytes: bytes,
                primary: entry.primary, top2IDs: entry.top2.0,
                top2Values: entry.top2.1,
                config: KVSSDConfigCode(entry.config),
                cacheCounts: encodeEntry(node.entry).counts,
                cacheMetaStates: encodeEntry(node.entry).metaStates,
                namespace: entry.namespace))
        }
        return writeIndex(
            KVSSDIndex(version: KVSSDIndex.version, keyDims: keyDims, nodes: indexNodes))
    }

    /// Append a single evicted node to the prefix set (the LRU-eviction write
    /// point). Idempotent: re-persisting the same fullPrefix overwrites the
    /// same file and replaces the index entry.
    @discardableResult
    func persistEvicted(_ node: SnapshotNode) -> Bool {
        let entry = node.entry
        let fileBase = self.fileBase(fullPrefix: node.fullPrefix)
        guard let bytes = writeNodeArrays(
            fileBase, encodeEntry(entry).arrays
        ) else { return false }
        var index = loadIndex() ?? KVSSDIndex(
            version: KVSSDIndex.version, keyDims: keyDims, nodes: [])
        index.nodes.removeAll { $0.fileBase == fileBase }
        index.nodes.append(KVSSDIndexNode(
            fileBase: fileBase, fullPrefix: node.fullPrefix,
            createdAt: node.createdAt, bytes: bytes,
            primary: entry.primary, top2IDs: entry.top2.0,
            top2Values: entry.top2.1,
            config: KVSSDConfigCode(entry.config),
            cacheCounts: encodeEntry(entry).counts,
            cacheMetaStates: encodeEntry(entry).metaStates,
            namespace: entry.namespace))
        return writeIndex(index)
    }

    // MARK: Byte-budget sweep

    /// Reclaim space: remove orphaned files (on disk, absent from the index),
    /// then LRU-oldest indexed files, until total bytes ≤ budget.
    func sweep() {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: config.nodesDir,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]
        ) else { return }
        let indexedBases = Set(((loadIndex()?.nodes) ?? []).map(\.fileBase))
        var files: [(url: URL, base: String, size: Int, mod: Date)] = []
        for url in entries where url.pathExtension == "safetensors" {
            let base = url.deletingPathExtension().lastPathComponent
            let attrs = try? fm.attributesOfItem(atPath: url.path)
            files.append((
                url, base,
                (attrs?[.size] as? NSNumber)?.intValue ?? 0,
                (attrs?[.modificationDate] as? Date) ?? .distantPast))
        }
        var total = files.reduce(0) { $0 + $1.size }
        for f in files where !indexedBases.contains(f.base) {
            try? fm.removeItem(at: f.url)
            total -= f.size
        }
        if total > config.budgetBytes {
            for f in files.filter({ indexedBases.contains($0.base) })
                .sorted(by: { $0.mod < $1.mod })
            {
                guard total > config.budgetBytes else { break }
                try? fm.removeItem(at: f.url)
                total -= f.size
            }
        }
    }
}

// MARK: - Helpers

extension [Int] {
    func canonicalKeyJSON(keyDims: KVSSDKeyDimensions) -> String {
        let obj: [String: Any] = [
            "weight_digest": keyDims.weightDigest,
            "template_hash": keyDims.templateHash,
            "prefix": self,
        ]
        let data = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        return String(data: data, encoding: .utf8)!
    }
}

extension Data {
    var sha256Hex: String {
        SHA256.hash(data: self).map { String(format: "%02x", $0) }.joined()
    }
}
