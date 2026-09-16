import Foundation
import MLX
import MLXLMCommon

/// Actor-isolated radix-tree store of completed-session KV state, keyed by
/// token prefix. Replaces the linear single-entry `LinearKVCacheManager`:
/// every completed session's history is retained as a state-carrying node,
/// so interleaved multi-session workloads can reuse any previously cached
/// history that is an exact prefix of the new seed — not only the most
/// recent one.
///
/// Match semantics: a match is only valid at a node whose stored state
/// covers the node's full prefix exactly. The consumer
/// (`Qwen36MTPBlockSession.begin`) requires
/// `trimmableOffset(reusableCache) == prefixCount`, and the recurrent
/// (gated-delta) layers cannot be trimmed to a shorter prefix, so a
/// stateless internal node (created by a split) can never be matched.
///
/// Ownership invariant: entries are created by the `MLXGenerator` actor's
/// generation task and consumed by the next session's `begin`; the tensors
/// are immutable after `store` and no other code path holds or mutates them.
public actor RadixKVCacheManager {
    /// The cached state of one completed session: its full token history
    /// plus the KV/hidden/primary/top2 state at that history's end.
    ///
    /// `@unchecked Sendable`: `MLXArray` and `[any KVCache]` are not
    /// `Sendable`, but the state is immutable after creation — the session
    /// that produced it has already completed, and the consumer (the next
    /// session's `begin`) only reads the tensors, never mutates them.
    /// Ownership invariant: the `CacheEntry` is created inside the
    /// `RadixKVCacheManager` actor and consumed by the `MLXGenerator`
    /// actor's generation task; no other code path holds or mutates the
    /// underlying MLX state.
    public struct CacheEntry: @unchecked Sendable {
        public let tokens: [Int]
        public let cache: [any KVCache]
        public let hidden: MLXArray
        public let primary: Int
        public let top2: ([Int], [Double])
        public let config: ResolvedKVCacheConfig
        public let namespace: String
        public let createdAt: Date

        public init(
            tokens: [Int],
            cache: [any KVCache],
            hidden: MLXArray,
            primary: Int,
            top2: ([Int], [Double]),
            config: ResolvedKVCacheConfig,
            namespace: String = "default",
            createdAt: Date = Date()
        ) {
            self.tokens = tokens
            self.cache = cache
            self.hidden = hidden
            self.primary = primary
            self.top2 = top2
            self.config = config
            self.namespace = namespace
            self.createdAt = createdAt
        }
    }

    /// One edge/node of the radix tree.
    ///
    /// `tokens` is the segment of token IDs along this edge; the node's full
    /// prefix is the concatenation of the segments from the root to this
    /// node. `cache`/`hidden`/`primary`/`top2` are non-nil only when a
    /// completed session stored its full-prefix state at this node; internal
    /// nodes created by splits carry no state and can never be matched.
    struct RadixNode {
        var tokens: [Int]
        var cache: [any KVCache]?
        var hidden: MLXArray?
        var primary: Int?
        var top2: ([Int], [Double])?
        var config: ResolvedKVCacheConfig
        var namespace: String
        var lastAccessed: ContinuousClock.Instant
        var children: [RadixNode]
        /// Disk-resident state reference (set by `restoreSkeleton`, cleared
        /// when the node is loaded into RAM). Non-nil when this node's state
        /// lives on the SSD tier and has not been loaded. A node is matchable
        /// if `cache != nil` (RAM) OR `diskState != nil` (disk).
        var diskState: KVSSDIndexNode?
    }

    /// Conservative KV footprint bound per token, mirroring
    /// `MemoryAdmissionPolicy`: 2 × 64 layers × 4 KV heads × 256 head_dim ×
    /// 2 bytes, counting every layer as KV-carrying.
    private static let estimatedBytesPerToken =
        2 * Qwen38KVGeometry.numLayers * Qwen38KVGeometry.numKeyValueHeads
        * Qwen38KVGeometry.headDim * 2

    private var root: RadixNode
    private let maxCacheBytes: Int
    private let defaultTTLSeconds: TimeInterval = 300 // 5 minutes default

    /// Hit/miss/eviction counters and the store count, exposed via `metrics()`.
    private(set) var hits = 0
    private(set) var misses = 0
    private(set) var evictions = 0
    private(set) var stores = 0

    /// Invoked (synchronously, on the model lane) with a snapshot of a leaf
    /// just before it is evicted by `evictLRULeaf`. The owner (the
    /// `MLXGenerator`) uses this to serialize the evicted node to the SSD tier
    /// before its in-RAM state is dropped. `nil` (default) = eviction is a
    /// plain drop (no persistence).
    var onEvict: (@Sendable (RadixSSDStore.SnapshotNode) -> Void)?

    /// Set the eviction callback (the SSD tier's write point on eviction).
    func setOnEvict(_ closure: @escaping @Sendable (RadixSSDStore.SnapshotNode) -> Void) {
        self.onEvict = closure
    }

    /// `maxCacheBytes` bounds the radix cache's own estimated footprint (the
    /// per-cache budget, independent of the global memory-admission limit). At
    /// the cap, `store` evicts LRU leaves until under budget. `Int.max` (the
    /// default) means unlimited, preserving the pre-hardening behavior.
    public init(maxCacheBytes: Int = Int.max) {
        self.root = Self.emptyNode()
        self.maxCacheBytes = max(0, maxCacheBytes)
    }

    private static func emptyNode() -> RadixNode {
        RadixNode(
            tokens: [],
            cache: nil,
            hidden: nil,
            primary: nil,
            top2: nil,
            config: .default,
            namespace: "",
            lastAccessed: .now,
            children: [],
            diskState: nil
        )
    }
    /// Traverses the tree along matching token sub-arrays and returns the
    /// longest matching node path that carries a stored, unexpired,
    /// config-matching session state, together with that state.
    ///
    /// Supports partial node matching: when the LCP between a child's tokens
    /// and the remaining input is greater than zero but shorter than the
    /// child's full token segment, the match is valid up to the LCP provided
    /// the child's KV cache supports sequence trimming (or the LCP equals the
    /// child's full length, i.e. a full match). A stateless internal node
    /// (created by a split) can never be matched because it carries no state.
    ///
    /// Nodes whose age exceeds `ttlSeconds` are ignored. A hit refreshes the
    /// matched node's `lastAccessed` (LRU accounting).
    public func matchPrefix(
        tokens: [Int],
        config: ResolvedKVCacheConfig,
        namespace: String = "default",
        ttlSeconds: Int? = nil
    ) -> (prefixCount: Int, entry: CacheEntry?) {
        guard !tokens.isEmpty else { return (0, nil) }
        let ttl = TimeInterval(ttlSeconds ?? Int(defaultTTLSeconds))

        // Walk the tree, recording the matched nodes and their child-index
        // path from the root. Accumulate the LCP at each step.
        var path: [RadixNode] = []
        var indexPath: [Int] = []
        var lcpAtEachStep: [Int] = []
        var remaining = tokens
        var current = root
        while !remaining.isEmpty {
            // Find the child with the longest LCP with remaining.
            var bestIndex: Int? = nil
            var bestLCP = 0
            for (i, child) in current.children.enumerated() {
                let lcp = Self.commonPrefixLength(child.tokens, remaining)
                if lcp > bestLCP {
                    bestLCP = lcp
                    bestIndex = i
                }
            }
            guard let index = bestIndex, bestLCP > 0 else { break }
            let child = current.children[index]
            path.append(child)
            indexPath.append(index)
            lcpAtEachStep.append(bestLCP)
            remaining = Array(remaining[bestLCP...])
            current = child
            // If bestLCP < child.tokens.count, we've partially matched this
            // child. We can't continue walking past this child (the remaining
            // tokens diverge from the child's tokens), so we stop here.
            if bestLCP < child.tokens.count {
                break
            }
        }

        // Check from deepest to shallowest: the first node that has a valid
        // KV cache, matches the config, is unexpired, and (supports trimming
        // OR was fully matched) is the match.
        for depth in (0..<path.count).reversed() {
            let node = path[depth]
            guard let cache = node.cache, let hidden = node.hidden,
                  let primary = node.primary, let top2 = node.top2,
                  node.config == config, node.namespace == namespace else { continue }
            let age = node.lastAccessed.duration(to: .now)
            guard age <= .seconds(ttl) else { continue }

            // The matched prefix length up to this node.
            let matchedPrefix = lcpAtEachStep[0...depth].reduce(0, +)
            // The LCP at this step.
            let lcpHere = lcpAtEachStep[depth]
            let fullyMatched = (lcpHere == node.tokens.count)
            // Supports trimming: all cache entries are trimmable. In the
            // benchmark, the cache is `[]`, which is vacuously trimmable.
            let supportsTrimming = cache.allSatisfy { $0.isTrimmable }

            guard fullyMatched || supportsTrimming else { continue }

            self.hits += 1
            self.touch(indexPath: Array(indexPath[0...depth]))
            return (
                matchedPrefix,
                CacheEntry(
                    tokens: Array(tokens[0..<matchedPrefix]),
                    cache: cache,
                    hidden: hidden,
                    primary: primary,
                    top2: top2,
                    config: node.config,
                    namespace: node.namespace
                )
            )
        }
        self.misses += 1
        return (0, nil)
    }

    /// Inserts `entry.tokens` into the tree, splitting existing nodes when
    /// the new sequence shares only a partial prefix with a child branch.
    /// The entry's state is stored at the node whose full prefix equals
    /// `entry.tokens`, and `lastAccessed` is refreshed.
    public func store(_ entry: CacheEntry) {
        guard !entry.tokens.isEmpty else { return }
        self.insert(entry: entry, into: &self.root, remaining: entry.tokens)
        self.stores += 1
        self.enforceByteBudget()
    }

    /// Evict LRU leaves until the cache's estimated footprint is under the
    /// per-cache byte budget. No-op when unlimited (`Int.max`).
    private func enforceByteBudget() {
        guard self.maxCacheBytes < Int.max else { return }
        while self.totalEstimatedBytes() > self.maxCacheBytes {
            guard self.evictLRULeaf() != nil else { break }
        }
    }

    /// Sum of (full prefix length × conservative per-token KV bound) over all
    /// leaves. Shared-prefix nodes are counted per-leaf, so this is a
    /// conservative (\u{2265}) estimate — the same accounting `evictLRULeaf` uses.
    private func totalEstimatedBytes() -> Int {
        var bytes = 0
        var stack: [(node: RadixNode, prefix: Int)] = [(root, 0)]
        while let (node, prefix) = stack.popLast() {
            let nodePrefix = prefix + node.tokens.count
            if node.children.isEmpty {
                bytes += nodePrefix * Self.estimatedBytesPerToken
            }
            for child in node.children {
                stack.append((child, nodePrefix))
            }
        }
        return bytes
    }

    private func insert(entry: CacheEntry, into node: inout RadixNode, remaining: [Int]) {
        // Longest-common-prefix child (if any).
        var bestIndex: Int? = nil
        var bestLCP = 0
        for (i, child) in node.children.enumerated() {
            let lcp = Self.commonPrefixLength(child.tokens, remaining)
            if lcp > bestLCP {
                bestLCP = lcp
                bestIndex = i
            }
        }
        guard let bestIndex else {
            // No child shares a prefix: new leaf carrying the full state.
            node.children.append(Self.stateNode(tokens: remaining, entry: entry))
            return
        }
        let lcp = bestLCP
        if lcp == remaining.count {
            if lcp == node.children[bestIndex].tokens.count {
                // Exact existing node: refresh state and timestamp.
                var updated = node.children[bestIndex]
                updated.cache = entry.cache
                updated.hidden = entry.hidden
                updated.primary = entry.primary
                updated.top2 = entry.top2
                updated.config = entry.config
                updated.lastAccessed = .now
                node.children[bestIndex] = updated
            } else {
                // The new entry ends mid-edge: split the child at `lcp`. The
                // prefix node takes the new state; the suffix node keeps the
                // old state (its full prefix is unchanged).
                var suffix = node.children[bestIndex]
                suffix.tokens = Array(suffix.tokens[lcp...])
                var prefix = Self.stateNode(
                    tokens: Array(node.children[bestIndex].tokens[0..<lcp]),
                    entry: entry
                )
                prefix.children = [suffix]
                node.children[bestIndex] = prefix
            }
            return
        }
        // lcp < remaining.count: the new entry continues past the split point.
        if lcp == node.children[bestIndex].tokens.count {
            // Child edge fully consumed: descend into it.
            self.insert(
                entry: entry,
                into: &node.children[bestIndex],
                remaining: Array(remaining[lcp...])
            )
        } else {
            // Split the child at `lcp`: the prefix node retains a trimmed copy
            // of the original node's KV state (for the shared prefix), the
            // suffix node keeps the old state, and insertion continues into
            // the suffix.
            var suffix = node.children[bestIndex]
            suffix.tokens = Array(suffix.tokens[lcp...])
            var prefix = node.children[bestIndex]
            prefix.tokens = Array(prefix.tokens[0..<lcp])
            // Preserve the KV state on the prefix node by trimming the
            // original node's cache to `lcp` tokens.
            if let originalCache = node.children[bestIndex].cache {
                prefix.cache = Self.trimCache(originalCache, to: lcp)
                prefix.hidden = node.children[bestIndex].hidden
                prefix.primary = node.children[bestIndex].primary
                prefix.top2 = node.children[bestIndex].top2
            } else {
                prefix.cache = nil
                prefix.hidden = nil
                prefix.primary = nil
                prefix.top2 = nil
            }
            prefix.children = [suffix]
            node.children[bestIndex] = prefix
            self.insert(
                entry: entry,
                into: &node.children[bestIndex],
                remaining: Array(remaining[lcp...])
            )
        }
    }

    /// Evicts cached KV state if system memory usage exceeds pressure
    /// thresholds. Recursively evicts the least-recently-used (LRU) leaf
    /// nodes until the pressure drops below the threshold (or the tree is
    /// empty). Returns `true` if any node was evicted.
    ///
    /// Evicted bytes are accounted with the same conservative per-token KV
    /// bound as `MemoryAdmissionPolicy`; `Memory.clearCache()` is called once
    /// after eviction, as a measured recovery/eviction action.
    public func purgeIfMemoryPressure(
        activeBytes: Int,
        limitBytes: Int,
        thresholdFraction: Double
    ) -> Bool {
        guard limitBytes > 0 else { return false }
        let threshold = thresholdFraction * Double(limitBytes)
        guard Double(activeBytes) >= threshold else { return false }

        var freedBytes = 0
        while Double(activeBytes - freedBytes) >= threshold {
            guard let evicted = self.evictLRULeaf() else { break }
            freedBytes += evicted
        }
        if freedBytes > 0 {
            Memory.clearCache()
        }
        return freedBytes > 0
    }

    /// Removes the leaf node (no children) with the oldest `lastAccessed`,
    /// pruning stateless, childless ancestors. Returns the evicted node's
    /// estimated KV bytes, or `nil` when there is no leaf to evict.
    private func evictLRULeaf() -> Int? {
        guard let leaf = self.findLRULeaf() else { return nil }
        // Persist the evicted node to the SSD tier before dropping its in-RAM
        // state (the LRU-eviction write point). Only stateful leaves have
        // tensors to persist.
        if let onEvict = self.onEvict,
           let cache = leaf.node.cache, let hidden = leaf.node.hidden,
           let primary = leaf.node.primary, let top2 = leaf.node.top2 {
            onEvict(RadixSSDStore.SnapshotNode(
                fullPrefix: leaf.fullPrefix,
                entry: CacheEntry(
                    tokens: leaf.fullPrefix, cache: cache, hidden: hidden,
                    primary: primary, top2: top2, config: leaf.node.config,
                    namespace: leaf.node.namespace
                ),
                createdAt: Date()))
        }
        self.removeLeaf(path: leaf.path)
        self.evictions += 1
        return leaf.prefixTokens * Self.estimatedBytesPerToken
    }

    /// Best-effort removal of the stateful leaf whose full prefix exactly
    /// equals `tokens` (and which has no children). Returns the number of
    /// leaves removed (0 or 1). Does nothing when no such leaf exists — e.g.
    /// the entry was evicted, was split into an internal node plus a leaf,
    /// or was never stored. This is shared-pool removal: a session deletion
    /// releases its prefix only when no other entry shares it.
    public func remove(tokens: [Int], namespace: String) -> Int {
        guard !tokens.isEmpty else { return 0 }
        guard let path = self.findLeafPath(tokens: tokens, node: self.root, path: [], consumed: 0) else {
            return 0
        }
        self.removeLeaf(path: path)
        return 1
    }

    /// Finds the child-index path to a stateful, childless node whose full
    /// prefix exactly equals `tokens`, if one exists.
    private func findLeafPath(tokens: [Int], node: RadixNode, path: [Int], consumed: Int) -> [Int]? {
        if consumed == tokens.count {
            return (node.children.isEmpty && node.cache != nil) ? path : nil
        }
        guard consumed < tokens.count else { return nil }
        for (i, child) in node.children.enumerated() {
            let seg = child.tokens
            guard consumed + seg.count <= tokens.count else { continue }
            guard tokens[consumed..<(consumed + seg.count)] == seg[...] else { continue }
            if let result = self.findLeafPath(
                tokens: tokens, node: child, path: path + [i], consumed: consumed + seg.count
            ) {
                return result
            }
        }
        return nil
    }

    /// The child-index path from the root to the LRU leaf, that leaf's node,
    /// full prefix length, and full token prefix.
    private func findLRULeaf()
        -> (path: [Int], prefixTokens: Int, node: RadixNode, fullPrefix: [Int])? {
        var best: (path: [Int], prefixTokens: Int, node: RadixNode, fullPrefix: [Int],
                  instant: ContinuousClock.Instant)? = nil
        var stack: [(node: RadixNode, path: [Int], fullPrefix: [Int])] = [(root, [], [])]
        while let (node, path, fullPrefix) = stack.popLast() {
            if node.children.isEmpty, !path.isEmpty {
                if best == nil || node.lastAccessed < best!.instant {
                    best = (path, fullPrefix.count, node, fullPrefix, node.lastAccessed)
                }
            }
            for (i, child) in node.children.enumerated() {
                stack.append((child, path + [i], fullPrefix + child.tokens))
            }
        }
        return best.map {
            (path: $0.path, prefixTokens: $0.prefixTokens, node: $0.node, fullPrefix: $0.fullPrefix)
        }
    }

    /// Removes the leaf addressed by `path` (child indices from the root) and
    /// prunes stateless, childless ancestors up the tree.
    private func removeLeaf(path: [Int]) {
        _ = self.prune(&self.root, path: path)
    }

    /// Removes the subtree addressed by `path` below `node` and returns
    /// whether `node` itself has become prunable (stateless and childless).
    private func prune(_ node: inout RadixNode, path: [Int]) -> Bool {
        guard let first = path.first else {
            return true // `node` is the leaf: its parent drops it.
        }
        let prunable = self.prune(&node.children[first], path: Array(path.dropFirst()))
        if prunable {
            node.children.remove(at: first)
        }
        return node.children.isEmpty && node.cache == nil
    }

    /// Refresh `lastAccessed` on the node addressed by `indexPath` (child
    /// indices from the root).
    private func touch(indexPath: [Int]) {
        self.touch(&self.root, indexPath: indexPath)
    }

    private func touch(_ node: inout RadixNode, indexPath: [Int]) {
        if indexPath.isEmpty {
            node.lastAccessed = .now
            return
        }
        self.touch(&node.children[indexPath[0]], indexPath: Array(indexPath.dropFirst()))
    }

    private static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var i = 0
        while i < limit, a[i] == b[i] { i += 1 }
        return i
    }

    /// Trims each entry in a KV cache array so that it represents the state
    /// at `length` tokens. If an entry's current offset exceeds `length`, it
    /// is trimmed by `offset - length` tokens (removing the most recent
    /// tokens). Entries whose offset is already at or below `length` are
    /// left unchanged.
    private static func trimCache(_ cache: [any KVCache], to length: Int) -> [any KVCache] {
        cache.map { entry in
            let offset = entry.offset
            if offset > length {
                _ = entry.trim(offset - length)
            }
            return entry
        }
    }

    private static func stateNode(tokens: [Int], entry: CacheEntry) -> RadixNode {
        RadixNode(
            tokens: tokens,
            cache: entry.cache,
            hidden: entry.hidden,
            primary: entry.primary,
            top2: entry.top2,
            config: entry.config,
            namespace: entry.namespace,
            lastAccessed: .now,
            children: [],
            diskState: nil
        )
    }

    // MARK: - SSD tier integration

    /// DFS snapshot of every stateful node (the graceful-shutdown write point).
    /// Returns each node's full token prefix plus its `CacheEntry`.
    func snapshot() -> [RadixSSDStore.SnapshotNode] {
        var out: [RadixSSDStore.SnapshotNode] = []
        var stack: [(node: RadixNode, fullPrefix: [Int])] = [(root, [])]
        while let (node, fullPrefix) = stack.popLast() {
            let full = fullPrefix + node.tokens
            if !node.tokens.isEmpty,
               let cache = node.cache, let hidden = node.hidden,
               let primary = node.primary, let top2 = node.top2 {
                out.append(RadixSSDStore.SnapshotNode(
                    fullPrefix: full,
                    entry: CacheEntry(
                        tokens: full, cache: cache, hidden: hidden,
                        primary: primary, top2: top2, config: node.config,
                        namespace: node.namespace
                    ),
                    createdAt: Date()))
            }
            for child in node.children {
                stack.append((child, full))
            }
        }
        return out
    }

    /// Rebuild the in-RAM tree as a skeleton from a loaded prefix set: one
    /// node per entry with `diskState` set (no tensors). Re-inserting each
    /// fullPrefix recreates the tree topology; every node's `diskState` is set
    /// by its own set entry, so LCP match semantics (including partial matches
    /// at internal shared-prefix nodes) survive.
    func restoreSkeleton(nodes: [KVSSDIndexNode]) {
        self.root = Self.emptyNode()
        for node in nodes {
            self.insertSkeleton(
                into: &self.root, remaining: node.fullPrefix, state: node)
        }
    }

    private static func skeletonNode(tokens: [Int], state: KVSSDIndexNode) -> RadixNode {
        RadixNode(
            tokens: tokens, cache: nil, hidden: nil, primary: nil, top2: nil,
            config: state.config?.resolved ?? .default,
            namespace: state.namespace,
            lastAccessed: .now, children: [], diskState: state)
    }

    private func insertSkeleton(into node: inout RadixNode, remaining: [Int], state: KVSSDIndexNode) {
        var bestIndex: Int? = nil
        var bestLCP = 0
        for (i, child) in node.children.enumerated() {
            let lcp = Self.commonPrefixLength(child.tokens, remaining)
            if lcp > bestLCP {
                bestLCP = lcp
                bestIndex = i
            }
        }
        guard let bestIndex else {
            node.children.append(Self.skeletonNode(tokens: remaining, state: state))
            return
        }
        let lcp = bestLCP
        if lcp == remaining.count {
            if lcp == node.children[bestIndex].tokens.count {
                var child = node.children[bestIndex]
                child.diskState = state
                node.children[bestIndex] = child
            } else {
                var suffix = node.children[bestIndex]
                suffix.tokens = Array(suffix.tokens[lcp...])
                var prefix = node.children[bestIndex]
                prefix.tokens = Array(prefix.tokens[0..<lcp])
                prefix.diskState = state
                prefix.children = [suffix]
                node.children[bestIndex] = prefix
            }
            return
        }
        if lcp == node.children[bestIndex].tokens.count {
            self.insertSkeleton(
                into: &node.children[bestIndex],
                remaining: Array(remaining[lcp...]), state: state)
        } else {
            var suffix = node.children[bestIndex]
            suffix.tokens = Array(suffix.tokens[lcp...])
            var prefix = node.children[bestIndex]
            prefix.tokens = Array(prefix.tokens[0..<lcp])
            prefix.children = [suffix]
            node.children[bestIndex] = prefix
            self.insertSkeleton(
                into: &node.children[bestIndex],
                remaining: Array(remaining[lcp...]), state: state)
        }
    }

    /// Like `matchPrefix`, but a node is matchable if it has state in RAM
    /// (`cache != nil`) OR on disk (`diskState != nil`). A RAM hit returns the
    /// entry; a disk hit returns the `diskState` (and the node's child-index
    /// path, so the caller can promote it to RAM after lazy-loading). Used by
    /// the `MLXGenerator`; the benchmark uses `matchPrefix` (RAM-only).
    func matchPrefixForGeneration(
        tokens: [Int],
        config: ResolvedKVCacheConfig,
        namespace: String = "default",
        ttlSeconds: Int? = nil,
        diskTTLSeconds: Int = 86400
    ) -> (prefixCount: Int, entry: CacheEntry?, diskHit: KVSSDIndexNode?, diskIndexPath: [Int]?) {
        guard !tokens.isEmpty else { return (0, nil, nil, nil) }
        let ttl = TimeInterval(ttlSeconds ?? Int(defaultTTLSeconds))
        let diskTTL = TimeInterval(diskTTLSeconds)

        var path: [RadixNode] = []
        var indexPath: [Int] = []
        var lcpAtEachStep: [Int] = []
        var remaining = tokens
        var current = root
        while !remaining.isEmpty {
            var bestIndex: Int? = nil
            var bestLCP = 0
            for (i, child) in current.children.enumerated() {
                let lcp = Self.commonPrefixLength(child.tokens, remaining)
                if lcp > bestLCP {
                    bestLCP = lcp
                    bestIndex = i
                }
            }
            guard let index = bestIndex, bestLCP > 0 else { break }
            let child = current.children[index]
            path.append(child)
            indexPath.append(index)
            lcpAtEachStep.append(bestLCP)
            remaining = Array(remaining[bestLCP...])
            current = child
            if bestLCP < child.tokens.count { break }
        }

        for depth in (0..<path.count).reversed() {
            let node = path[depth]
            guard node.config == config, node.namespace == namespace else { continue }
            let matchedPrefix = lcpAtEachStep[0...depth].reduce(0, +)
            let lcpHere = lcpAtEachStep[depth]
            let fullyMatched = (lcpHere == node.tokens.count)
            let ip = Array(indexPath[0...depth])

            // RAM node (unchanged semantics vs `matchPrefix`).
            if let cache = node.cache, let hidden = node.hidden,
               let primary = node.primary, let top2 = node.top2 {
                let age = node.lastAccessed.duration(to: .now)
                guard age <= .seconds(ttl) else { continue }
                let supportsTrimming = cache.allSatisfy { $0.isTrimmable }
                guard fullyMatched || supportsTrimming else { continue }
                self.hits += 1
                self.touch(indexPath: ip)
                return (matchedPrefix, CacheEntry(
                    tokens: Array(tokens[0..<matchedPrefix]), cache: cache,
                    hidden: hidden, primary: primary, top2: top2,
                    config: node.config, namespace: node.namespace), nil, nil)
            }

            // Disk node: config guarantees trimmability; the `begin` desync
            // check (trimmableOffset == prefixCount) is the real gate.
            if let disk = node.diskState, disk.namespace == namespace {
                let age = Date().timeIntervalSince(disk.createdAt)
                guard age <= diskTTL else { continue }
                self.hits += 1
                self.touch(indexPath: ip)
                return (matchedPrefix, nil, disk, ip)
            }
        }
        self.misses += 1
        return (0, nil, nil, nil)
    }

    /// After the generator lazy-loads a disk node's tensors, promote it to RAM:
    /// set the in-RAM state and clear `diskState`. `indexPath` is the node's
    /// child-index path (as returned by `matchPrefixForGeneration`).
    func promoteToRAM(
        indexPath: [Int], entry: CacheEntry
    ) {
        self.promote(&self.root, indexPath: indexPath, entry: entry)
    }

    private func promote(_ node: inout RadixNode, indexPath: [Int], entry: CacheEntry) {
        guard let first = indexPath.first else { return }
        if indexPath.count == 1 {
            node.children[first].cache = entry.cache
            node.children[first].hidden = entry.hidden
            node.children[first].primary = entry.primary
            node.children[first].top2 = entry.top2
            node.children[first].config = entry.config
            node.children[first].namespace = entry.namespace
            node.children[first].diskState = nil
            node.children[first].lastAccessed = .now
            return
        }
        self.promote(&node.children[first], indexPath: Array(indexPath.dropFirst()), entry: entry)
    }

    /// Drop all cached state and clear the MLX allocator cache. Lifetime
    /// counters (`hits`/`misses`/`evictions`/`stores`) are NOT reset — they are
    /// process-lifetime observability totals, not per-clear figures.
    public func clear() {
        self.root = Self.emptyNode()
        Memory.clearCache()
    }

    /// A point-in-time snapshot of the cache's size and lifetime counters, for
    /// the observability endpoint. `entries` is the leaf count; `estimatedBytes`
    /// is the conservative per-leaf prefix accounting; `hitRate` is
    /// hits/(hits+misses) over the process lifetime.
    public struct Metrics: Sendable, Equatable {
        public let entries: Int
        public let estimatedBytes: Int
        public let hits: Int
        public let misses: Int
        public let evictions: Int
        public let stores: Int
        public var hitRate: Double {
            let total = hits + misses
            return total == 0 ? 0 : Double(hits) / Double(total)
        }
    }

    public func metrics() -> Metrics {
        var entries = 0
        var stack: [RadixNode] = [root]
        while let node = stack.popLast() {
            if node.children.isEmpty && !node.tokens.isEmpty {
                entries += 1
            }
            stack.append(contentsOf: node.children)
        }
        return Metrics(
            entries: entries,
            estimatedBytes: totalEstimatedBytes(),
            hits: hits,
            misses: misses,
            evictions: evictions,
            stores: stores
        )
    }
}

