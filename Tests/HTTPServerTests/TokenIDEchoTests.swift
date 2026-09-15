// TokenIDEchoTests.swift
//
// Tests for the `include_token_ids` response extension and the session
// `CompletionCommit`. These are pure serialization/model tests: no model, no
// weights. They pin that (a) `token_ids` is omitted by default and round-trips
// when present, (b) the stored assistant history carries NO token IDs (it is
// for re-rendering only), and (c) `session_id` round-trips on the response.

import Testing
import Foundation
@testable import HTTPServer

@Test func chatMessageTokenIDsOmittedWhenNil() throws {
    let msg = ChatMessage(role: "assistant", content: "hi")
    let data = try JSONEncoder().encode(msg)
    let json = String(data: data, encoding: .utf8)!
    #expect(!json.contains("token_ids"))
}

@Test func chatMessageTokenIDsRoundTrip() throws {
    let msg = ChatMessage(role: "assistant", content: "hi", token_ids: [1, 2, 3])
    let data = try JSONEncoder().encode(msg)
    let json = String(data: data, encoding: .utf8)!
    #expect(json.contains("token_ids"))
    let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
    #expect(decoded.token_ids == [1, 2, 3])
}

@Test func completionCommitAssistantMessageCarriesNoTokenIDs() {
    let commit = CompletionCommit(
        content: "hi",
        reasoningContent: "thought",
        toolCalls: [],
        completionTokenIDs: [1, 2, 3],
        seedTokens: [10],
        promptTokens: 1,
        completionTokens: 3
    )
    let msg = commit.assistantMessage
    #expect(msg.role == "assistant")
    #expect(msg.content == "hi")
    #expect(msg.reasoning_content == "thought")
    #expect(msg.token_ids == nil)  // stored history is for re-rendering only
}

@Test func completionCommitAssistantMessageOmitsEmptyToolCalls() {
    let commit = CompletionCommit(
        content: nil,
        reasoningContent: nil,
        toolCalls: [],
        completionTokenIDs: [],
        seedTokens: [10],
        promptTokens: 1,
        completionTokens: 0
    )
    let msg = commit.assistantMessage
    #expect(msg.tool_calls == nil)  // empty tool calls collapse to nil
}

@Test func completionCommitAssistantMessageKeepsToolCalls() throws {
    let call = ToolCall(
        id: "call_1",
        type: "function",
        index: nil,
        function: .init(name: "f", arguments: "{}")
    )
    let commit = CompletionCommit(
        content: nil,
        reasoningContent: nil,
        toolCalls: [call],
        completionTokenIDs: [1],
        seedTokens: [10],
        promptTokens: 1,
        completionTokens: 1
    )
    let msg = commit.assistantMessage
    #expect(msg.tool_calls?.count == 1)
    let data = try JSONEncoder().encode(msg)
    let json = String(data: data, encoding: .utf8)!
    #expect(json.contains("call_1"))
    #expect(!json.contains("token_ids"))
}

@Test func chatCompletionResponseSessionIDRoundTrip() throws {
    let response = ChatCompletionResponse(
        id: "chatcmpl-1",
        object: "chat.completion",
        created: 0,
        model: "m",
        choices: [],
        usage: nil,
        session_id: "sess-1"
    )
    let data = try JSONEncoder().encode(response)
    let json = String(data: data, encoding: .utf8)!
    #expect(json.contains("sess-1"))
    let decoded = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
    #expect(decoded.session_id == "sess-1")
}

@Test func chatCompletionRequestIncludeTokenIDsDecode() throws {
    let body = """
    {"model": "m", "messages": [], "include_token_ids": true}
    """
    let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(body.utf8))
    #expect(req.include_token_ids == true)
}
