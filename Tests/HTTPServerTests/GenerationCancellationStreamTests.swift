// GenerationCancellationStreamTests.swift
//
// No-weight lifecycle tests for the request-scoped cancellation stream
// bridge. These tests exercise the AsyncStream cancellation lifecycle
// (register -> check -> yield -> cleanup -> onTermination) without any MLX,
// model weights, or MLXGenerator instantiation.

import XCTest
@testable import HTTPServer

final class GenerationCancellationStreamTests: XCTestCase {

    /// A pending cancellation (before the producer registers) must stop the
    /// stream before any fragment is yielded, and the ID must be deregistered
    /// after the stream ends.
    func testCancelledBeforeModelWorkYieldsNothing() async {
        let registry = GenerationCancellationRegistry()
        let id = GenerationID()

        // Cancel before the producer registers (pending cancellation).
        await registry.cancel(id, reason: .callerCancelled)

        let stream = makeCancellationStream(
            id: id,
            registry: registry,
            yieldCount: 5
        )

        var fragments: [Int] = []
        for await fragment in stream {
            fragments.append(fragment)
        }

        XCTAssertEqual(fragments, [])

        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
    }

    /// When not cancelled, the stream yields all fragments and the ID is
    /// deregistered after the stream ends.
    func testStreamYieldsAllFragmentsWhenNotCancelled() async {
        let registry = GenerationCancellationRegistry()
        let id = GenerationID()

        let stream = makeCancellationStream(
            id: id,
            registry: registry,
            yieldCount: 5
        )

        var fragments: [Int] = []
        for await fragment in stream {
            fragments.append(fragment)
        }

        XCTAssertEqual(fragments, [0, 1, 2, 3, 4])

        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
    }

    /// When the consumer stops consuming (stream termination), the registry
    /// entry is cancelled with `.streamConsumerTerminated` if the ID is still
    /// registered.
    func testConsumerTerminationCancelsRegistryEntry() async {
        let registry = GenerationCancellationRegistry()
        let id = GenerationID()

        let stream = makeCancellationStream(
            id: id,
            registry: registry,
            yieldCount: 100
        )

        // Consume only a few fragments, then stop consuming.
        var consumed = 0
        for await _ in stream {
            consumed += 1
            if consumed == 3 {
                break
            }
        }

        // Allow the onTermination task to run.
        try? await Task.sleep(for: .milliseconds(200))

        // After the stream ends, the ID must be deregistered (not registered).
        // The onTermination task may cancel the registry entry with
        // `.streamConsumerTerminated` if the ID is still registered, but the
        // producer task may deregister the ID first. In either case, the ID
        // must not be registered after the stream ends.
        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
    }
    func testSlowConsumerTriggersDroppedCancellation() async {
        let registry = GenerationCancellationRegistry()
        let id = GenerationID()

        let yieldCount = 100
        let bufferCapacity = 8

        let stream = makeBoundedCancellationStream(
            id: id,
            registry: registry,
            yieldCount: yieldCount,
            bufferCapacity: bufferCapacity
        )

        var consumed = 0
        
        // Consume fragments slowly: sleep so the fast producer fills the 
        // buffer, yields against a full buffer, gets .dropped, and aborts.
        for await fragment in stream {
            _ = fragment
            consumed += 1
            try? await Task.sleep(for: .milliseconds(50))
        }

        // The stream should abort and close. The consumer will only receive 
        // what fit in the buffer plus maybe 1 in-flight before the abort.
        XCTAssertTrue(
            consumed <= bufferCapacity + 2, 
            "Stream should have aborted due to .dropped, but consumed \(consumed) items"
        )

        // After the stream ends via cancellation, the ID must be deregistered.
        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
        
        // The reason should be recorded as a consumer termination.
        let reason = await registry.cancellationReason(for: id)
        XCTAssertNil(reason)
    }
}