// TokenizationCache.swift
//
// A bounded, actor-isolated cache of prompt tokenization results. It stores
// only the encoded token IDs (`[Int]`) for a fully formatted prompt string —
// no MLX state, no KV rows, no model weights. It is a pure-Swift value type
// owned by the `MLXGenerator` actor, so every access is serialized on the
// model-critical lane and no data race is possible.
//
// This is Stage 0 of the prefix-cache RFC: it removes redundant CPU
// tokenization for repeated prompts. It is deliberately NOT a KV/prefix cache:
// the runtime has no copy-on-write, so safe KV reuse is out of scope here.

import Foundation

struct TokenizationCache {
    /// The full cache key. Two entries are distinct unless every dimension
    /// that can change the tokenized output is equal:
    /// - `namespace`: model/head identity + tokenizer/template identity
    ///   (process-wide constant; the KV format is intentionally excluded
    ///   because it does not affect tokenization).
    /// - `enableThinking`: the template variant (thinking vs. non-thinking).
    /// - `formattedPrompt`: the exact, fully formatted prompt string.
    struct Key: Hashable {
        let namespace: String
        let enableThinking: Bool
        let formattedPrompt: String
    }

    /// A cached tokenization result with its LRU/TTL bookkeeping.
    private struct Entry {
        let tokenIDs: [Int]
        var lastAccess: Date
        var expiresAt: Date
        let byteSize: Int
    }

    /// Cumulative, server-wide cache counters. These are the source of truth
    /// for the `/metrics` endpoint; they are never reset per request.
    struct Stats {
        let hits: Int
        let misses: Int
        let evictions: Int
        let entryCount: Int
        let byteCount: Int
    }

    private var entries: [Key: Entry] = [:]
    /// LRU order: index 0 is the most recently used, the last element is the
    /// least recently used (the eviction victim).
    private var order: [Key] = []
    private var hits = 0
    private var misses = 0
    private var evictions = 0
    private var byteCount = 0
    private let maxEntries: Int
    private let maxBytes: Int
    private let defaultTTL: TimeInterval
    private let clock: () -> Date

    init(
        maxEntries: Int,
        maxBytes: Int,
        defaultTTL: TimeInterval,
        clock: @escaping () -> Date = { Date() }
    ) {
        self.maxEntries = max(1, maxEntries)
        self.maxBytes = max(1, maxBytes)
        self.defaultTTL = max(0, defaultTTL)
        self.clock = clock
    }

    /// Look up a key. Returns the cached token IDs on a hit (refreshing LRU
    /// order and TTL) or `nil` on a miss (including an expired entry, which is
    /// evicted and counted as a miss). Never throws.
    mutating func lookup(key: Key) -> ([Int]?, wasHit: Bool) {
        let now = clock()
        guard var entry = entries[key] else {
            misses += 1
            return (nil, false)
        }
        if now >= entry.expiresAt {
            remove(key: key)
            misses += 1
            return (nil, false)
        }
        hits += 1
        entry.lastAccess = now
        entry.expiresAt = now + defaultTTL
        entries[key] = entry
        moveToFront(key: key)
        return (entry.tokenIDs, true)
    }

    /// Insert (or replace) a key. Enforces the entry-count and byte budgets by
    /// evicting expired entries first, then least-recently-used entries. Never
    /// throws; a failed insert simply leaves the cache in a valid state.
    mutating func insert(key: Key, tokenIDs: [Int], ttl: TimeInterval? = nil) {
        let now = clock()
        let byteSize = key.formattedPrompt.utf8.count + tokenIDs.count * 8
        if entries[key] != nil {
            remove(key: key)
        }
        entries[key] = Entry(
            tokenIDs: tokenIDs,
            lastAccess: now,
            expiresAt: now + (ttl ?? defaultTTL),
            byteSize: byteSize
        )
        byteCount += byteSize
        order.insert(key, at: 0)
        evictToLimits(now: now)
    }

    /// Snapshot of the cumulative counters and current occupancy.
    func stats() -> Stats {
        Stats(
            hits: hits,
            misses: misses,
            evictions: evictions,
            entryCount: entries.count,
            byteCount: byteCount
        )
    }

    private mutating func remove(key: Key) {
        guard let entry = entries.removeValue(forKey: key) else { return }
        byteCount -= entry.byteSize
        if let idx = order.firstIndex(of: key) {
            order.remove(at: idx)
        }
    }

    private mutating func moveToFront(key: Key) {
        guard let idx = order.firstIndex(of: key), idx != 0 else { return }
        order.remove(at: idx)
        order.insert(key, at: 0)
    }

    private mutating func evictToLimits(now: Date) {
        var expired: [Key] = []
        for (key, entry) in entries where now >= entry.expiresAt {
            expired.append(key)
        }
        for key in expired {
            remove(key: key)
            evictions += 1
        }
        while byteCount > maxBytes, let victim = order.last {
            remove(key: victim)
            evictions += 1
        }
        while entries.count > maxEntries, let victim = order.last {
            remove(key: victim)
            evictions += 1
        }
    }
}
