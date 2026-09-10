// CancellationStreamBridge.swift
//
// A small, pure-Swift AsyncStream helper that exercises the request-scoped
// cancellation lifecycle (register -> check -> yield -> cleanup ->
// onTermination) without any MLX, model, tokenizer, or Vapor/NIO
// involvement. It is a test double for the MLXGenerator producer logic.

import Foundation

/// A minimal AsyncStream producer that mirrors the MLXGenerator cancellation
/// lifecycle:
/// - registers `id` before yielding,
/// - checks cancellation at each safe boundary (before each yield and before
///   finishing),
/// - yields up to `yieldCount` fragments when not cancelled,
/// - finishes the stream exactly once,
/// - deregisters `id` after the stream ends,
/// - cancels the registry entry with `.streamConsumerTerminated` on
///   termination if the ID is still registered.
///
/// `yieldCount` is the number of fragments to yield when not cancelled.
func makeCancellationStream(
    id: GenerationID,
    registry: GenerationCancellationRegistry,
    yieldCount: Int
) -> AsyncStream<Int> {
    AsyncStream(Int.self) { continuation in
        let task = Task {
            var emitted = 0

            do {
                // Register before yielding.
                _ = await registry.register(id)

                // Immediately check cancellation after registration.
                if await registry.isCancelled(id) {
                    throw GenerationCancelledError(id: id)
                }

                for _ in 0..<yieldCount {
                    // Check cancellation before each yield.
                    if await registry.isCancelled(id) {
                        throw GenerationCancelledError(id: id)
                    }

                    continuation.yield(emitted)
                    emitted += 1
                }

                // Check cancellation before finishing.
                if await registry.isCancelled(id) {
                    throw GenerationCancelledError(id: id)
                }

                continuation.finish()
            } catch is GenerationCancelledError {
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish()
            }

            // One explicit terminal cleanup path.
            await registry.deregister(id)
        }

        continuation.onTermination = { _ in
            Task {
                if await registry.isRegistered(id) {
                    await registry.cancel(id, reason: .streamConsumerTerminated)
                }
            }
            task.cancel()
        }
    }
}

/// A bounded-buffering AsyncStream producer that mirrors the MLXGenerator
/// cancellation lifecycle with `.bufferingOldest(bufferCapacity)`:
/// - registers `id` before yielding,
/// - checks cancellation at each safe boundary (before each yield and before
///   finishing),
/// - yields up to `yieldCount` fragments when not cancelled,
/// - treats a non-`.enqueued` yield result as a consumer termination and
///   cancels the registry entry with `.streamConsumerTerminated`,
/// - finishes the stream exactly once,
/// - deregisters `id` after the stream ends,
/// - cancels the registry entry with `.streamConsumerTerminated` on
///   termination if the ID is still registered.
///
/// `yieldCount` is the number of fragments to yield when not cancelled.
/// `bufferCapacity` is the maximum number of fragments the stream may buffer
/// before blocking the producer (mirrors `.bufferingOldest(8)` in
/// MLXGenerator).
func makeBoundedCancellationStream(
    id: GenerationID,
    registry: GenerationCancellationRegistry,
    yieldCount: Int,
    bufferCapacity: Int
) -> AsyncStream<Int> {
    AsyncStream(Int.self, bufferingPolicy: .bufferingOldest(bufferCapacity)) {
        continuation in
        let task = Task {
            var emitted = 0

            do {
                // Register before yielding.
                _ = await registry.register(id)

                // Immediately check cancellation after registration.
                if await registry.isCancelled(id) {
                    throw GenerationCancelledError(id: id)
                }

                for _ in 0..<yieldCount {
                    // Check cancellation before each yield.
                    if await registry.isCancelled(id) {
                        throw GenerationCancelledError(id: id)
                    }

                    // Yield and inspect the result, mirroring MLXGenerator's
                    // yieldAndCheck. With `.bufferingOldest` the only success
                    // result is `.enqueued`; `.dropped` and `.terminated` mean
                    // the consumer stopped or the stream is already finished.
                    switch continuation.yield(emitted) {
                    case .enqueued:
                        emitted += 1
                    case .terminated:
                        await registry.cancel(id, reason: .streamConsumerTerminated)
                        throw GenerationCancelledError(id: id)
                    case .dropped:
                        // Unreachable with `.bufferingOldest`, treated as failure.
                        await registry.cancel(id, reason: .streamConsumerTerminated)
                        throw GenerationCancelledError(id: id)
                    @unknown default:
                        emitted += 1
                    }
                }

                // Check cancellation before finishing.
                if await registry.isCancelled(id) {
                    throw GenerationCancelledError(id: id)
                }

                continuation.finish()
            } catch is GenerationCancelledError {
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish()
            }

            // One explicit terminal cleanup path.
            await registry.deregister(id)
        }

        continuation.onTermination = { _ in
            Task {
                if await registry.isRegistered(id) {
                    await registry.cancel(id, reason: .streamConsumerTerminated)
                }
            }
            task.cancel()
        }
    }
}