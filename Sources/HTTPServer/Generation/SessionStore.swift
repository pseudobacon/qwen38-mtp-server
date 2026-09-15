import Foundation

/// Process-local, in-memory conversation/session store.
///
/// Sessions are keyed by an opaque server-assigned ID and are NOT persisted to
/// disk. Each session owns a conversation history (`messages`) used to re-render
/// the full chat template on the next completion — the server is stateless at
/// the model boundary. `prefixTokens` is the committed token prefix (prompt +
/// generated), stored for best-effort radix-cache release on delete; it is
/// diagnostic and does not guarantee a cache hit.
///
/// Lifecycle is bounded: a per-session TTL plus a global LRU cap (`maxSessions`)
/// prevent unbounded growth. One in-flight completion per session is enforced
/// via the `inFlight` flag.
actor SessionStore {
    private struct Entry {
        var messages: [ChatMessage]
        var tokenCount: Int
        var prefixTokens: [Int]?
        var inFlight: Bool
        var lastUsed: ContinuousClock.Instant
        var createdAt: Int
        var ttl: TimeInterval
    }

    private var entries: [String: Entry] = [:]
    private let maxSessions: Int
    private let defaultTTL: TimeInterval
    private let clock = ContinuousClock()

    init(maxSessions: Int, defaultTTL: TimeInterval) {
        self.maxSessions = max(1, maxSessions)
        self.defaultTTL = defaultTTL
    }

    // MARK: - CRUD

    /// Create an empty session. Evicts the LRU entry if over the cap.
    func create() -> SessionObject {
        let id = UUID().uuidString
        let createdAt = Int(Date().timeIntervalSince1970)
        entries[id] = Entry(
            messages: [],
            tokenCount: 0,
            prefixTokens: nil,
            inFlight: false,
            lastUsed: clock.now,
            createdAt: createdAt,
            ttl: defaultTTL
        )
        evictIfNeeded()
        return SessionObject(
            id: id, object: "session", created: createdAt,
            model: "", token_count: 0
        )
    }

    func exists(_ id: String) -> Bool {
        guard let entry = entries[id] else { return false }
        return !isExpired(entry)
    }

    func session(_ id: String) -> SessionObject? {
        guard let entry = entries[id], !isExpired(entry) else { return nil }
        touch(id)
        return SessionObject(
            id: id, object: "session", created: entry.createdAt,
            model: "", token_count: entry.tokenCount
        )
    }

    /// Delete a session, returning the cached prefix tokens (so the caller can
    /// release the radix entry) or `nil` if the session did not exist.
    func delete(_ id: String) -> [Int]? {
        guard let entry = entries.removeValue(forKey: id) else { return nil }
        return entry.prefixTokens
    }

    // MARK: - Completion lifecycle

    /// Return the current message history for building a completion request.
    func messages(_ id: String) -> [ChatMessage]? {
        guard let entry = entries[id], !isExpired(entry) else { return nil }
        touch(id)
        return entry.messages
    }

    /// Mark the session as having an in-flight completion. Returns `false` if
    /// the session is missing, expired, or already busy (409 at the route).
    func beginCompletion(_ id: String) -> Bool {
        guard var entry = entries[id], !isExpired(entry) else { return false }
        guard !entry.inFlight else { return false }
        entry.inFlight = true
        entry.lastUsed = clock.now
        entries[id] = entry
        return true
    }

    /// Commit a successful completion: append the request messages plus the
    /// assistant message, update the token count and cached prefix, and clear
    /// the in-flight flag.
    func finishCompletion(_ id: String, requestMessages: [ChatMessage], commit: CompletionCommit) {
        guard var entry = entries[id] else { return }
        entry.messages.append(contentsOf: requestMessages)
        entry.messages.append(commit.assistantMessage)
        entry.tokenCount = commit.promptTokens + commit.completionTokens
        entry.prefixTokens = commit.seedTokens + commit.completionTokenIDs
        entry.inFlight = false
        entry.lastUsed = clock.now
        entries[id] = entry
    }

    /// Clear the in-flight flag after a failed/aborted completion.
    func abortCompletion(_ id: String) {
        guard var entry = entries[id] else { return }
        entry.inFlight = false
        entry.lastUsed = clock.now
        entries[id] = entry
    }

    // MARK: - Eviction

    private func touch(_ id: String) {
        entries[id]?.lastUsed = clock.now
    }

    private func isExpired(_ entry: Entry) -> Bool {
        clock.now - entry.lastUsed > Duration.seconds(entry.ttl)
    }

    /// Remove expired entries, then evict LRU entries until under the cap.
    func evictIfNeeded() {
        // Expire first.
        for (id, entry) in entries where isExpired(entry) {
            entries.removeValue(forKey: id)
        }
        // LRU-evict the rest.
        while entries.count > maxSessions {
            guard let oldest = entries.min(by: { $0.value.lastUsed < $1.value.lastUsed })?.key else { return }
            entries.removeValue(forKey: oldest)
        }
    }
}
