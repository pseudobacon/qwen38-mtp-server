// ModelRuntimeState.swift
//
// Actor-owned model-runtime state machine for the Qwen 3.8 MTP server.
//
// The HTTP layer never stores this state in Vapor application storage; it
// queries this actor. The actor holds only the state enum and no MLX tensors,
// model/session/KV objects, Vapor request/response objects, or tasks.

import Foundation
import Vapor

/// Actor-owned model-runtime state machine.
///
/// States:
/// - `.initializing`: model, tokenizer, and MTP head are loading.
/// - `.warming`: startup warmup of MTP shapes is in progress.
/// - `.ready`: model is loaded and warmed; serving requests.
/// - `.draining`: server is shutting down; new requests are rejected.
/// - `.failed(reason)`: model startup or warmup failed.
/// - `.memoryDegraded`: server is in a memory-degraded state.
public actor ModelRuntimeState {
    public enum State: Sendable, Equatable {
        case initializing
        case warming
        case ready
        case draining
        case failed(String)
        case memoryDegraded
    }

    private(set) public var state: State = .initializing

    /// True only when the runtime is fully loaded and warmed.
    public func isReady() -> Bool {
        state == .ready
    }

    /// Attempts a state transition. Returns `true` if the transition is legal
    /// and was applied, `false` otherwise (state unchanged).
    public func transition(to newState: State) -> Bool {
        guard canTransition(from: state, to: newState) else { return false }
        state = newState
        return true
    }

    /// A short diagnostic pair for the `/readyz` 503 body.
    public func diagnostic() -> (status: String, message: String) {
        switch state {
        case .initializing:
            return ("initializing", "Model, tokenizer, and MTP head are still loading.")
        case .warming:
            return ("warming", "Startup warmup of MTP shapes is in progress.")
        case .ready:
            return ("ready", "Model is loaded and warmed; serving requests.")
        case .draining:
            return ("draining", "Server is shutting down; new requests are rejected.")
        case .failed(let reason):
            return ("failed", "Model startup failed: \(reason)")
        case .memoryDegraded:
            return ("memory_degraded", "Server is in a memory-degraded state; new requests are rejected.")
        }
    }

    private func canTransition(from: State, to: State) -> Bool {
        switch (from, to) {
        case (.initializing, .warming), (.initializing, .failed):
            return true
        case (.warming, .ready), (.warming, .failed):
            return true
        case (.ready, .draining), (.ready, .memoryDegraded):
            return true
        case (.memoryDegraded, .ready), (.memoryDegraded, .draining):
            return true
        default:
            return false
        }
    }
}

/// `GET /healthz` response: 200 OK as long as the HTTP server is alive.
/// Never triggers or requires model inference.
public func healthzResponse() -> Response {
    let body = #"{"status":"ok"}"#
    return Response(
        status: .ok,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: Data(body.utf8))
    )
}

/// `GET /readyz` response: 200 OK only when the runtime is `.ready`;
/// otherwise 503 Service Unavailable with a diagnostic JSON body.
public func readyzResponse(_ runtimeState: ModelRuntimeState) async -> Response {
    guard await runtimeState.isReady() else {
        let (status, message) = await runtimeState.diagnostic()
        let payload: [String: String] = ["status": status, "message": message]
        let data = try! JSONEncoder().encode(payload)
        return Response(
            status: .serviceUnavailable,
            headers: ["Content-Type": "application/json; charset=utf-8"],
            body: .init(data: data)
        )
    }
    let body = #"{"status":"ready"}"#
    return Response(
        status: .ok,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: Data(body.utf8))
    )
}