// ToolCallingRenderTests.swift
//
// Tests for prompt-rendering identity: `MLXGenerator.tokenizationCacheKey`
// must distinguish every prompt the chat template can render, so that two
// conversations differing only in a rendered field (tool schemas, tool_choice,
// assistant tool_calls, tool-result tool_call_id, reasoning) never share a
// cached tokenized prompt.
//
// The full HuggingFace template rendering is external (mlx-swift-lm) and
// requires model weights; these tests pin the identity contract that guards
// against cross-conversation contamination at the tokenization-cache layer.

import Testing
import Foundation
@testable import HTTPServer

private func msg(_ role: String, _ content: String? = nil) -> ChatMessage {
    ChatMessage(role: role, content: content)
}

private func assistantCall(_ id: String, _ name: String) -> ChatMessage {
    ChatMessage(
        role: "assistant",
        content: nil,
        tool_calls: [ToolCall(id: id, type: "function", index: nil, function: .init(name: name, arguments: "{}"))]
    )
}

private func toolResult(_ id: String) -> ChatMessage {
    ChatMessage(role: "tool", content: "ok", tool_call_id: id)
}

private func tool(_ name: String) -> ToolSpec {
    ToolSpec(type: "function", function: ToolSpecFunction(name: name, description: nil, parameters: nil))
}

@Test
func keyDiffersWithAndWithoutTools() {
    let messages = [msg("user", "Hi")]
    let withTools = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("f")], toolChoice: nil)
    let noTools = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: nil, toolChoice: nil)
    #expect(withTools != noTools)
}

@Test
func keyDiffersWithToolSchemas() {
    let messages = [msg("user", "Hi")]
    let a = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("weather")], toolChoice: nil)
    let b = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("search")], toolChoice: nil)
    #expect(a != b)
}

@Test
func keyDiffersWithToolChoice() {
    let messages = [msg("user", "Hi")]
    let auto = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("f")], toolChoice: .auto)
    let none = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("f")], toolChoice: ToolChoice.none)
    #expect(auto != none)
}

@Test
func keyDiffersWithAssistantToolCalls() {
    // An assistant message with a tool call renders differently (the XML
    // function block) than one without; the key must reflect it.
    let bare = MLXGenerator.tokenizationCacheKey(messages: [msg("assistant", "")], enableThinking: true, tools: [tool("f")], toolChoice: nil)
    let withCall = MLXGenerator.tokenizationCacheKey(messages: [assistantCall("call_1", "f")], enableThinking: true, tools: [tool("f")], toolChoice: nil)
    #expect(bare != withCall)
}

@Test
func keyDiffersWithToolCallId() {
    // Tool-result messages that differ only in tool_call_id must not collide.
    let a = MLXGenerator.tokenizationCacheKey(messages: [toolResult("call_A")], enableThinking: true, tools: [tool("f")], toolChoice: nil)
    let b = MLXGenerator.tokenizationCacheKey(messages: [toolResult("call_B")], enableThinking: true, tools: [tool("f")], toolChoice: nil)
    #expect(a != b)
}

@Test
func keyDiffersWithReasoningContent() {
    let bare = MLXGenerator.tokenizationCacheKey(messages: [msg("assistant", "hi")], enableThinking: true, tools: nil, toolChoice: nil)
    let withReasoning = MLXGenerator.tokenizationCacheKey(
        messages: [ChatMessage(role: "assistant", content: "hi", reasoning_content: "thought")],
        enableThinking: true, tools: nil, toolChoice: nil
    )
    #expect(bare != withReasoning)
}

@Test
func keyIsStableForIdenticalInput() {
    let messages = [
        msg("user", "Weather?"),
        assistantCall("call_1", "get_weather"),
        toolResult("call_1"),
    ]
    let a = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("get_weather")], toolChoice: nil)
    let b = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: [tool("get_weather")], toolChoice: nil)
    #expect(a == b)
}

@Test
func keyReflectsEnableThinking() {
    let messages = [msg("user", "Hi")]
    let on = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: true, tools: nil, toolChoice: nil)
    let off = MLXGenerator.tokenizationCacheKey(messages: messages, enableThinking: false, tools: nil, toolChoice: nil)
    #expect(on != off)
}