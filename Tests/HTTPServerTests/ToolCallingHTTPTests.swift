// ToolCallingHTTPTests.swift
//
// Tests for the HTTP-layer tool-calling behavior that does not require model
// weights: the `finish_reason` computation shared by the streaming and
// non-streaming paths. The full request/response pipeline (SSE framing, model
// execution) is integration-level and exercised by the radix benchmark suite
// and the live-server harness; these tests pin the finish_reason contract.
//
// Contract: when one or more tool calls were produced and the model terminated
// normally ("stop"), the finish reason is "tool_calls"; truncation reasons
// ("length", "memory_pressure") are preserved.

import Testing
import Foundation
@testable import HTTPServer

@Test
func toolCallsWithStopReasonBecomeToolCalls() {
    #expect(toolCallFinishReason(toolCallCount: 1, finishedReason: "stop") == "tool_calls")
    #expect(toolCallFinishReason(toolCallCount: 3, finishedReason: "stop") == "tool_calls")
}

@Test
func noToolCallsPreserveStop() {
    #expect(toolCallFinishReason(toolCallCount: 0, finishedReason: "stop") == "stop")
}

@Test
func toolCallsDoNotMaskTruncationReasons() {
    // A truncated or memory-pressured generation keeps its reason even if some
    // (possibly partial) tool calls were parsed.
    #expect(toolCallFinishReason(toolCallCount: 1, finishedReason: "length") == "length")
    #expect(toolCallFinishReason(toolCallCount: 2, finishedReason: "memory_pressure") == "memory_pressure")
}
