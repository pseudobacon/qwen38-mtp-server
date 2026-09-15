// ToolCallingValidationTests.swift
//
// Unit tests for tool-calling request validation: tool schema checks,
// `tool_choice` resolution, `parallel_tool_calls`, and conversation sequencing
// (orphaned tool results, `tool_call_id` matching, unique call IDs).
//
// Pure Swift (no MLX, no model weights): runs on any machine. Every rejection
// is an OpenAI-shaped 400 with {"error": {"message", "type", "param", "code"}}.

import Testing
import Foundation
@testable import HTTPServer

// MARK: - Fixtures

private func makeTool(name: String, type: String = "function", parameters: JSONValue? = nil) -> ToolSpec {
    ToolSpec(
        type: type,
        function: ToolSpecFunction(name: name, description: nil, parameters: parameters)
    )
}

private func assistantCall(id: String, name: String) -> ChatMessage {
    ChatMessage(
        role: "assistant",
        content: nil,
        tool_calls: [
            ToolCall(id: id, type: "function", index: nil, function: .init(name: name, arguments: "{}"))
        ]
    )
}

private func toolResult(id: String, content: String = "ok") -> ChatMessage {
    ChatMessage(role: "tool", content: content, tool_call_id: id)
}

private func validRequest(
    messages: [ChatMessage] = [ChatMessage(role: "user", content: "Hi")],
    tools: [ToolSpec]? = nil,
    toolChoice: ToolChoice? = nil,
    parallelToolCalls: Bool? = nil
) -> ChatCompletionRequest {
    ChatCompletionRequest(
        model: ServerConfig.canonicalModelID,
        messages: messages,
        tools: tools,
        tool_choice: toolChoice,
        parallel_tool_calls: parallelToolCalls
    )
}

private func expectValidationError(_ request: ChatCompletionRequest, _ check: (OpenAIRequestError) -> Void) {
    do {
        _ = try ChatCompletionRequestValidator.validate(request: request, serverConfig: ServerConfig())
        Issue.record("expected validation to fail")
    } catch let error as OpenAIRequestError {
        check(error)
    } catch {
        Issue.record("unexpected error type: \(error)")
    }
}

// MARK: - Tool schema

@Test
func validToolsAreAccepted() throws {
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(tools: [makeTool(name: "get_weather")]),
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
}

@Test
func toolTypeMustBeFunction() {
    expectValidationError(validRequest(tools: [makeTool(name: "f", type: "retriever")])) { error in
        #expect(error.param == "tools")
        #expect(error.code == "invalid_value")
    }
}

@Test
func toolNameMustMatchPolicy() {
    // Spaces are not allowed in tool names.
    expectValidationError(validRequest(tools: [makeTool(name: "get weather")])) { error in
        #expect(error.param == "tools")
        #expect(error.code == "invalid_value")
    }
    // Over-long names are rejected.
    let long = String(repeating: "a", count: 65)
    expectValidationError(validRequest(tools: [makeTool(name: long)])) { error in
        #expect(error.param == "tools")
    }
}

@Test
func duplicateToolNamesAreRejected() {
    expectValidationError(validRequest(tools: [makeTool(name: "f"), makeTool(name: "f")])) { error in
        #expect(error.param == "tools")
        #expect(error.message.contains("Duplicate"))
    }
}

@Test
func toolParametersMustBeAnObject() {
    expectValidationError(validRequest(tools: [makeTool(name: "f", parameters: .array([]))])) { error in
        #expect(error.param == "tools")
        #expect(error.code == "invalid_value")
    }
}

@Test
func tooManyToolsAreRejected() {
    let many = (0...ChatCompletionRequestValidator.maxToolCount).map { makeTool(name: "t\($0)") }
    expectValidationError(validRequest(tools: many)) { error in
        #expect(error.param == "tools")
    }
}

// MARK: - tool_choice

@Test
func toolChoiceRequiredIsRejected() {
    expectValidationError(validRequest(tools: [makeTool(name: "f")], toolChoice: .required)) { error in
        #expect(error.param == "tool_choice")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func toolChoiceNamedIsRejected() {
    expectValidationError(validRequest(tools: [makeTool(name: "f")], toolChoice: .function(name: "f"))) { error in
        #expect(error.param == "tool_choice")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func toolChoiceAutoIsAccepted() throws {
    _ = try ChatCompletionRequestValidator.validate(
        request: validRequest(tools: [makeTool(name: "f")], toolChoice: .auto),
        serverConfig: ServerConfig()
    )
}

@Test
func toolChoiceNoneIsAccepted() throws {
    _ = try ChatCompletionRequestValidator.validate(
        request: validRequest(tools: [makeTool(name: "f")], toolChoice: ToolChoice.none),
        serverConfig: ServerConfig()
    )
}

// MARK: - parallel_tool_calls

@Test
func parallelToolCallsTrueIsRejected() {
    expectValidationError(validRequest(parallelToolCalls: true)) { error in
        #expect(error.param == "parallel_tool_calls")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func parallelToolCallsFalseIsAccepted() throws {
    _ = try ChatCompletionRequestValidator.validate(
        request: validRequest(parallelToolCalls: false),
        serverConfig: ServerConfig()
    )
}

// MARK: - Conversation sequencing

@Test
func validToolConversationIsAccepted() throws {
    _ = try ChatCompletionRequestValidator.validate(
        request: validRequest(messages: [
            ChatMessage(role: "user", content: "Weather?"),
            assistantCall(id: "call_1", name: "get_weather"),
            toolResult(id: "call_1", content: "Sunny"),
        ]),
        serverConfig: ServerConfig()
    )
}

@Test
func orphanedToolResultIsRejected() {
    expectValidationError(validRequest(messages: [
        ChatMessage(role: "user", content: "Weather?"),
        toolResult(id: "call_missing"),
    ])) { error in
        #expect(error.param == "messages")
        #expect(error.code == "invalid_value")
    }
}

@Test
func toolResultWithoutToolCallIdIsRejected() {
    expectValidationError(validRequest(messages: [
        ChatMessage(role: "user", content: "Weather?"),
        assistantCall(id: "call_1", name: "get_weather"),
        ChatMessage(role: "tool", content: "Sunny", tool_call_id: nil),
    ])) { error in
        #expect(error.param == "messages")
        #expect(error.message.contains("tool_call_id"))
    }
}

@Test
func duplicateToolCallIdsAcrossMessagesAreRejected() {
    expectValidationError(validRequest(messages: [
        ChatMessage(role: "user", content: "Weather?"),
        assistantCall(id: "call_1", name: "get_weather"),
        toolResult(id: "call_1"),
        assistantCall(id: "call_1", name: "get_weather"),
    ])) { error in
        #expect(error.param == "messages")
        #expect(error.message.contains("Duplicate"))
    }
}

@Test
func crossTurnToolCallIdMismatchIsRejected() {
    // The tool result references an id that no assistant message produced.
    expectValidationError(validRequest(messages: [
        ChatMessage(role: "user", content: "Weather?"),
        assistantCall(id: "call_A", name: "get_weather"),
        toolResult(id: "call_B"),
    ])) { error in
        #expect(error.param == "messages")
        #expect(error.code == "invalid_value")
    }
}