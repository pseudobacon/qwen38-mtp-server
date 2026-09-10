// NoThinkSlashCommandTests.swift
//
// Unit tests for the `/nothink` / `/no_think` slash-command interception in
// the chat-completions router. Pure Swift (no MLX, no model weights): they
// pin the stripping behavior, the native `/no_think` marker guarantee, and
// the downstream `enable_thinking` / prompt formatting consequences.

import Testing
import Foundation
@testable import HTTPServer

// Regression test: OpenAI clients commonly omit the `stream` field. The
// synthesized Codable decoder ignores default values, so `stream` must be
// optional in the wire shape; an omitted key decodes as `nil`, which the
// router treats as `false` (`request.stream ?? false`).
@Test func omittedStreamKeyDecodesAsNil() throws {
    let json = "{\"model\":\"qwen-3.8-27b-mtp\",\"messages\":[{\"role\":\"user\",\"content\":\"/nothink Hello\"}]}"
    let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    #expect(req.stream == nil)
    #expect(req.messages.last?.content == "/nothink Hello")
}

@Test func explicitStreamKeyDecodes() throws {
    let jsonTrue = "{\"model\":\"qwen-3.8-27b-mtp\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}"
    let reqTrue = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(jsonTrue.utf8))
    #expect(reqTrue.stream == true)
    let jsonFalse = "{\"model\":\"qwen-3.8-27b-mtp\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":false}"
    let reqFalse = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(jsonFalse.utf8))
    #expect(reqFalse.stream == false)
}

private func message(
    role: String,
    content: String?
) -> ChatMessage {
    ChatMessage(
        role: role,
        content: content,
        reasoning: nil,
        reasoning_content: nil
    )
}

private func request(
    messages: [ChatMessage],
    enableThinking: Bool? = nil,
    stream: Bool = false,
    maxTokens: Int? = nil
) -> ChatCompletionRequest {
    ChatCompletionRequest(
        model: ServerConfig.canonicalModelID,
        messages: messages,
        stream: stream,
        max_tokens: maxTokens,
        max_completion_tokens: nil,
        n: nil,
        temperature: nil,
        top_p: nil,
        top_k: nil,
        min_p: nil,
        repetition_penalty: nil,
        presence_penalty: nil,
        frequency_penalty: nil,
        stream_options: nil,
        stop: nil,
        context_window: nil,
        enable_thinking: enableThinking,
        mtp_enabled: nil,
        kv_cache_bits: nil,
        kv_cache_group_size: nil,
        quantized_kv_start: nil,
        ttl_seconds: nil,
        chat_template: nil,
        chat_template_kwargs: nil
    )
}

@Test("a leading /nothink is stripped and thinking is disabled")
func noThinkPrefixIsStripped() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "/nothink Hello")])
    )
    #expect(updated.messages.last?.content == "Hello /no_think")
    #expect(updated.enable_thinking == false)
}

@Test("a leading /no_think is stripped and thinking is disabled")
func noThinkUnderscorePrefixIsStripped() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "/no_think Hello")])
    )
    #expect(updated.messages.last?.content == "Hello /no_think")
    #expect(updated.enable_thinking == false)
}

@Test("leading whitespace before the command is tolerated")
func leadingWhitespaceIsTolerated() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "  /nothink Hello")])
    )
    #expect(updated.messages.last?.content == "Hello /no_think")
    #expect(updated.enable_thinking == false)
}

@Test("a bare /nothink command yields the bare native marker")
func bareCommandYieldsNativeMarker() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "/nothink")])
    )
    #expect(updated.messages.last?.content == "/no_think")
    #expect(updated.enable_thinking == false)
}

@Test("requests without the command are returned unchanged")
func noCommandLeavesRequestUnchanged() {
    let original = request(messages: [message(role: "user", content: "Hello")])
    let updated = applyNoThinkSlashCommand(to: original)
    #expect(updated.messages.last?.content == "Hello")
    #expect(updated.enable_thinking == nil)
}

@Test("a mid-string /nothink is not stripped (prefix-only semantics)")
func midStringCommandIsNotStripped() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "Hello /nothink world")])
    )
    #expect(updated.messages.last?.content == "Hello /nothink world")
    #expect(updated.enable_thinking == nil)
}

@Test("the command overrides an explicit enable_thinking: true")
func commandOverridesExplicitEnableThinking() {
    let updated = applyNoThinkSlashCommand(
        to: request(
            messages: [message(role: "user", content: "/nothink Hello")],
            enableThinking: true
        )
    )
    #expect(updated.enable_thinking == false)
}

@Test("only the last message is inspected")
func onlyLastMessageIsInspected() {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [
            message(role: "user", content: "/nothink first"),
            message(role: "assistant", content: "hi"),
            message(role: "user", content: "second"),
        ])
    )
    #expect(updated.messages[0].content == "/nothink first")
    #expect(updated.messages.last?.content == "second")
    #expect(updated.enable_thinking == nil)
}

@Test("other request fields survive the interception")
func otherFieldsSurvive() {
    let updated = applyNoThinkSlashCommand(
        to: request(
            messages: [message(role: "user", content: "/nothink Hello")],
            stream: true,
            maxTokens: 64
        )
    )
    #expect(updated.model == ServerConfig.canonicalModelID)
    #expect(updated.stream == true)
    #expect(updated.max_tokens == 64)
    #expect(updated.messages.last?.role == "user")
}

@Test("the stripped request resolves enableThinking=false downstream")
func strippedRequestResolvesThinkingDisabled() throws {
    let updated = applyNoThinkSlashCommand(
        to: request(messages: [message(role: "user", content: "/nothink Hello")])
    )
    let params = try SamplingParameters.fromRequest(
        updated,
        serverConfig: ServerConfig()
    )
    #expect(params.enableThinking == false)
}
