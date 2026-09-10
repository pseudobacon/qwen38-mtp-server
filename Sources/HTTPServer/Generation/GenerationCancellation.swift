// GenerationCancellation.swift
//
// Request-scoped cancellation state for the Qwen 3.8 MTP server.
//
// This is a standalone, pure-Swift component: it owns only request-ID
// cancellation state. It holds no MLX arrays, model/session/KV objects,
// Vapor request/response objects, NIO writers, or tasks. It is not yet
// wired into MLXGenerator or OpenAIRouter; that is a later stage.
//
// Pre-registration cancellation: a `cancel` that arrives before the
// producer's `register` is retained as a *pending* cancellation. When the
// producer later calls `register`, the new active entry immediately reflects
// that pending cancellation, so a producer that checks `isCancelled` before
// or after `register` observes the cancellation and never starts model work
// for a generation the client has already abandoned.

import Foundation

/// A unique, request-scoped identifier for one generation.
///
/// Deliberately separate from the OpenAI `responseId` value: `responseId`
/// is an API-facing identifier, while a `GenerationID` is an internal
/// cancellation handle owned by the server.
struct GenerationID: Hashable, Sendable {
    let uuid: UUID

    /// Creates a fresh, unique generation ID.
    init() {
        self.uuid = UUID()
    }

    /// A stable, log-friendly representation, e.g. `gen-6F1C...`.
    var description: String {
        "gen-\(uuid.uuidString)"
    }
}

/// Why a generation was cancelled.
enum GenerationCancellationReason: Equatable, Sendable {
    /// The caller explicitly cancelled the request.
    case callerCancelled
    /// The stream consumer stopped consuming (e.g. client disconnect).
    case streamConsumerTerminated
    /// An SSE write failed.
    case sseWriteFailure

    /// A short, log-friendly name for the reason.
    var description: String {
        switch self {
        case .callerCancelled: return "caller_cancelled"
        case .streamConsumerTerminated: return "stream_consumer_terminated"
        case .sseWriteFailure: return "sse_write_failure"
        }
    }
}

/// Actor-owned, request-scoped cancellation state.
///
/// Invariants:
/// - `cancel` is idempotent; the first reason wins.
/// - Cancelling an unknown ID stores a *pending* cancellation that a later
///   `register` immediately reflects as an active, cancelled entry.
/// - `register` creates an active entry that reflects any pending
///   cancellation immediately.
/// - `isCancelled` and `cancellationReason` reflect pending cancellation so
///   that producer checks are safe before and after `register`.
/// - `deregister` removes either active or pending state; a later `register`
///   starts fresh and does not resurrect a removed cancellation.
/// - Cancelling one ID never affects another ID.
/// - The cancellation reason is observable until deregistration.
/// - The registry holds only `GenerationID` -> state: no MLX arrays,
///   model/session/KV objects, Vapor request/response objects, NIO writers,
///   or tasks.
actor GenerationCancellationRegistry {
    private enum State: Sendable {
        /// A cancellation was requested before the ID was registered.
        case pending(GenerationCancellationReason)
        /// The ID is registered; `reason` is `nil` when uncancelled.
        case active(reason: GenerationCancellationReason?)
    }

    private var states: [GenerationID: State] = [:]

    /// Registers `id` as an active generation.
    ///
    /// If a pending cancellation exists for `id`, the new active entry
    /// immediately reflects it (cancelled with the pending reason).
    ///
    /// Returns `false` if `id` is already registered (active).
    func register(_ id: GenerationID) -> Bool {
        switch states[id] {
        case nil:
            states[id] = .active(reason: nil)
            return true
        case .pending(let reason):
            states[id] = .active(reason: reason)
            return true
        case .active:
            return false
        }
    }

    /// Cancels `id` with `reason`.
    ///
    /// Idempotent: the first reason wins. Cancelling an unknown ID stores a
    /// pending cancellation that a later `register` will immediately reflect.
    func cancel(_ id: GenerationID, reason: GenerationCancellationReason) {
        switch states[id] {
        case nil:
            states[id] = .pending(reason)
        case .pending:
            // First reason wins; no-op.
            break
        case .active(let existing):
            if existing == nil {
                states[id] = .active(reason: reason)
            }
            // Already cancelled; first reason wins; no-op.
        }
    }

    /// True if `id` is currently registered (active, cancelled or not).
    func isRegistered(_ id: GenerationID) -> Bool {
        if case .active = states[id] { return true }
        return false
    }

    /// True if `id` is cancelled, whether via a pending pre-registration
    /// cancellation or an active cancellation.
    func isCancelled(_ id: GenerationID) -> Bool {
        switch states[id] {
        case .pending:
            return true
        case .active(let reason):
            return reason != nil
        case nil:
            return false
        }
    }

    /// The cancellation reason if `id` is cancelled (pending or active), else `nil`.
    func cancellationReason(for id: GenerationID) -> GenerationCancellationReason? {
        switch states[id] {
        case .pending(let reason):
            return reason
        case .active(let reason):
            return reason
        case nil:
            return nil
        }
    }

    /// Removes all state for `id` (active or pending). Idempotent. A later
    /// `register` starts fresh and does not resurrect a removed cancellation.
    func deregister(_ id: GenerationID) {
        states[id] = nil
    }
}