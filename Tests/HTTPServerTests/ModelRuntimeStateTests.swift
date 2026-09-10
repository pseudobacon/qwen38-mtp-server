// ModelRuntimeStateTests.swift
//
// Unit tests for the actor-owned model-runtime state machine and the
// `/healthz` + `/readyz` endpoint response builders. No model weights are
// loaded; the tests exercise the state machine and the pure response
// builders directly.

import XCTest
import Vapor
@testable import HTTPServer

final class ModelRuntimeStateTests: XCTestCase {

    // MARK: - State machine transitions

    func testInitialStateIsInitializing() async {
        let state = ModelRuntimeState()
        let current = await state.state
        XCTAssertEqual(current, .initializing)
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testInitializingToWarming() async {
        let state = ModelRuntimeState()
        let ok = await state.transition(to: .warming)
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .warming)
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testWarmingToReady() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        let ok = await state.transition(to: .ready)
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .ready)
        let ready = await state.isReady()
        XCTAssertTrue(ready)
    }

    func testWarmingToFailed() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        let ok = await state.transition(to: .failed("warmup failed"))
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .failed("warmup failed"))
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testInitializingToFailed() async {
        let state = ModelRuntimeState()
        let ok = await state.transition(to: .failed("load failed"))
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .failed("load failed"))
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testReadyToDraining() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        _ = await state.transition(to: .ready)
        let ok = await state.transition(to: .draining)
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .draining)
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testReadyToMemoryDegraded() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        _ = await state.transition(to: .ready)
        let ok = await state.transition(to: .memoryDegraded)
        XCTAssertTrue(ok)
        let current = await state.state
        XCTAssertEqual(current, .memoryDegraded)
        let ready = await state.isReady()
        XCTAssertFalse(ready)
    }

    func testIllegalTransitionsAreRejected() async {
        let state = ModelRuntimeState()

        // initializing -> ready is illegal (must warm first).
        let ok1 = await state.transition(to: .ready)
        XCTAssertFalse(ok1)
        let s1 = await state.state
        XCTAssertEqual(s1, .initializing)

        // initializing -> draining is illegal.
        let ok2 = await state.transition(to: .draining)
        XCTAssertFalse(ok2)
        let s2 = await state.state
        XCTAssertEqual(s2, .initializing)

        // warming -> draining is illegal (must reach ready first).
        _ = await state.transition(to: .warming)
        let ok3 = await state.transition(to: .draining)
        XCTAssertFalse(ok3)
        let s3 = await state.state
        XCTAssertEqual(s3, .warming)

        // ready -> warming is illegal (no re-warm after ready).
        _ = await state.transition(to: .ready)
        let ok4 = await state.transition(to: .warming)
        XCTAssertFalse(ok4)
        let s4 = await state.state
        XCTAssertEqual(s4, .ready)

        // draining is terminal.
        _ = await state.transition(to: .draining)
        let ok5 = await state.transition(to: .ready)
        XCTAssertFalse(ok5)
        let ok6 = await state.transition(to: .warming)
        XCTAssertFalse(ok6)
        let s5 = await state.state
        XCTAssertEqual(s5, .draining)
    }

    // MARK: - /healthz endpoint

    func testHealthzReturns200() async {
        let response = healthzResponse()
        XCTAssertEqual(response.status, .ok)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"ok\""))
    }

    // MARK: - /readyz endpoint

    func testReadyzReturns200WhenReady() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        _ = await state.transition(to: .ready)

        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .ok)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"ready\""))
    }

    func testReadyzReturns503WhenInitializing() async {
        let state = ModelRuntimeState()
        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .serviceUnavailable)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"initializing\""))
        XCTAssertTrue(body.contains("\"message\""))
    }

    func testReadyzReturns503WhenWarming() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .serviceUnavailable)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"warming\""))
    }

    func testReadyzReturns503WhenDraining() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        _ = await state.transition(to: .ready)
        _ = await state.transition(to: .draining)
        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .serviceUnavailable)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"draining\""))
    }

    func testReadyzReturns503WhenFailed() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .failed("boom"))
        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .serviceUnavailable)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"failed\""))
        XCTAssertTrue(body.contains("boom"))
    }

    func testReadyzReturns503WhenMemoryDegraded() async {
        let state = ModelRuntimeState()
        _ = await state.transition(to: .warming)
        _ = await state.transition(to: .ready)
        _ = await state.transition(to: .memoryDegraded)
        let response = await readyzResponse(state)
        XCTAssertEqual(response.status, .serviceUnavailable)
        let body = String(decoding: response.body.data ?? Data(), as: UTF8.self)
        XCTAssertTrue(body.contains("\"status\":\"memory_degraded\""))
    }
}