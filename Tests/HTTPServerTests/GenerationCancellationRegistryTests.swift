// GenerationCancellationRegistryTests.swift
//
// Unit tests for the request-scoped cancellation registry.
//
// Pure Swift: no MLX, no model weights, and no MLXGenerator instantiation.

import XCTest
@testable import HTTPServer

final class GenerationCancellationRegistryTests: XCTestCase {
    private let registry = GenerationCancellationRegistry()

    func testRegistrationBeginsUncancelled() async {
        let id = GenerationID()
        let registered = await registry.register(id)
        XCTAssertTrue(registered)
        let isRegistered = await registry.isRegistered(id)
        XCTAssertTrue(isRegistered)
        let isCancelled = await registry.isCancelled(id)
        XCTAssertFalse(isCancelled)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertNil(reason)
    }

    func testRegisteringTheSameIDTwiceIsRejected() async {
        let id = GenerationID()
        let first = await registry.register(id)
        XCTAssertTrue(first)
        let second = await registry.register(id)
        XCTAssertFalse(second)
    }

    func testCancellationMarksTheMatchingIDWithTheExpectedReason() async {
        let id = GenerationID()
        _ = await registry.register(id)
        await registry.cancel(id, reason: .callerCancelled)
        let isCancelled = await registry.isCancelled(id)
        XCTAssertTrue(isCancelled)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertEqual(reason, .callerCancelled)
    }

    func testRepeatedCancellationIsIdempotentAndFirstReasonWins() async {
        let id = GenerationID()
        _ = await registry.register(id)
        await registry.cancel(id, reason: .callerCancelled)
        await registry.cancel(id, reason: .sseWriteFailure)
        await registry.cancel(id, reason: .streamConsumerTerminated)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertEqual(reason, .callerCancelled)
    }

    // MARK: - Pre-registration cancellation

    func testCancelBeforeRegisterIsObservedByRegisterWithTheOriginalReason() async {
        let id = GenerationID()
        await registry.cancel(id, reason: .streamConsumerTerminated)
        // Before registration the ID is not registered, but the pending
        // cancellation is already observable for safe producer checks.
        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
        let isCancelled = await registry.isCancelled(id)
        XCTAssertTrue(isCancelled)
        let pendingReason = await registry.cancellationReason(for: id)
        XCTAssertEqual(pendingReason, .streamConsumerTerminated)

        // Registering must create an active entry that immediately reflects
        // the pending cancellation.
        let registered = await registry.register(id)
        XCTAssertTrue(registered)
        let isRegisteredAfter = await registry.isRegistered(id)
        XCTAssertTrue(isRegisteredAfter)
        let isCancelledAfter = await registry.isCancelled(id)
        XCTAssertTrue(isCancelledAfter)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertEqual(reason, .streamConsumerTerminated)
    }

    func testFirstReasonWinsForRepeatedPreRegistrationCancel() async {
        let id = GenerationID()
        await registry.cancel(id, reason: .callerCancelled)
        await registry.cancel(id, reason: .sseWriteFailure)
        await registry.cancel(id, reason: .streamConsumerTerminated)
        let pendingReason = await registry.cancellationReason(for: id)
        XCTAssertEqual(pendingReason, .callerCancelled)

        _ = await registry.register(id)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertEqual(reason, .callerCancelled)
    }

    func testPendingCancelOfOneIDDoesNotAffectAnother() async {
        let a = GenerationID()
        let b = GenerationID()
        await registry.cancel(a, reason: .sseWriteFailure)

        let aCancelled = await registry.isCancelled(a)
        XCTAssertTrue(aCancelled)
        let aReason = await registry.cancellationReason(for: a)
        XCTAssertEqual(aReason, .sseWriteFailure)

        let bRegistered = await registry.register(b)
        XCTAssertTrue(bRegistered)
        let bCancelled = await registry.isCancelled(b)
        XCTAssertFalse(bCancelled)
        let bReason = await registry.cancellationReason(for: b)
        XCTAssertNil(bReason)
    }

    func testDeregisterBeforeRegisterClearsPendingCancellation() async {
        let id = GenerationID()
        await registry.cancel(id, reason: .callerCancelled)
        await registry.deregister(id)

        let isCancelledAfterDeregister = await registry.isCancelled(id)
        XCTAssertFalse(isCancelledAfterDeregister)
        let reasonAfterDeregister = await registry.cancellationReason(for: id)
        XCTAssertNil(reasonAfterDeregister)

        // A later registration must start fresh, uncancelled: the removed
        // pending cancellation is not resurrected.
        let registered = await registry.register(id)
        XCTAssertTrue(registered)
        let isCancelledAfterRegister = await registry.isCancelled(id)
        XCTAssertFalse(isCancelledAfterRegister)
        let reasonAfterRegister = await registry.cancellationReason(for: id)
        XCTAssertNil(reasonAfterRegister)
    }

    func testDeregistrationRemovesState() async {
        let id = GenerationID()
        _ = await registry.register(id)
        await registry.cancel(id, reason: .streamConsumerTerminated)
        await registry.deregister(id)
        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
        let isCancelled = await registry.isCancelled(id)
        XCTAssertFalse(isCancelled)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertNil(reason)
    }

    func testCancellationAfterDeregistrationStoresPendingCancellation() async {
        let id = GenerationID()
        _ = await registry.register(id)
        await registry.deregister(id)
        await registry.cancel(id, reason: .callerCancelled)
        let isRegistered = await registry.isRegistered(id)
        XCTAssertFalse(isRegistered)
        // The ID is not resurrected as registered, but the post-deregister
        // cancel is retained as a pending cancellation that a later
        // registration would immediately reflect.
        let isCancelled = await registry.isCancelled(id)
        XCTAssertTrue(isCancelled)
        let reason = await registry.cancellationReason(for: id)
        XCTAssertEqual(reason, .callerCancelled)
    }

    func testCancellingOneIDDoesNotAffectAnother() async {
        let a = GenerationID()
        let b = GenerationID()
        _ = await registry.register(a)
        _ = await registry.register(b)
        await registry.cancel(a, reason: .sseWriteFailure)
        let aCancelled = await registry.isCancelled(a)
        XCTAssertTrue(aCancelled)
        let aReason = await registry.cancellationReason(for: a)
        XCTAssertEqual(aReason, .sseWriteFailure)
        let bRegistered = await registry.isRegistered(b)
        XCTAssertTrue(bRegistered)
        let bCancelled = await registry.isCancelled(b)
        XCTAssertFalse(bCancelled)
        let bReason = await registry.cancellationReason(for: b)
        XCTAssertNil(bReason)
    }

    func testGenerationIDEqualityAndHashing() {
        let id = GenerationID()
        let copy = id
        let other = GenerationID()

        XCTAssertEqual(id, copy)
        XCTAssertEqual(id.hashValue, copy.hashValue)
        XCTAssertNotEqual(id, other)

        var keys: [GenerationID: String] = [:]
        keys[id] = "first"
        keys[copy] = "second"
        XCTAssertEqual(keys.count, 1)
        XCTAssertEqual(keys[id], "second")
        XCTAssertNil(keys[other])

        XCTAssertTrue(id.description.hasPrefix("gen-"))
        XCTAssertTrue(id.description.count > "gen-".count)
    }
}