// ToolCallingSerializationTests.swift
//
// Serialization tests for the OpenAI-compatible tool-calling surface: the
// `ToolCall`/`ToolSpec`/`ToolChoice` Codable shapes and how completed tool
// calls are serialized into streaming `delta.tool_calls` chunks and
// non-streaming `message.tool_calls` responses.
//
// These are pure-Swift Codable tests (no MLX, no model weights), so they run
// on any machine without loading the model.

import Testing
import Foundation
@testable import HTTPServer

/// A completed tool call as the generator would emit it.
private func sampleToolCall(
    id: String = "call_abc123",
    name: String = "get_weather",
    arguments: String = "{\"location\":\"Paris\"}"
) -> ToolCall {
    ToolCall(
        id: id,
        type: "function",
        index: 0,
        function: ToolCall.ToolCallFunction(name: name, arguments: arguments)
    )
}

// MARK: - Streaming delta.tool_calls

@Test
func streamingChunkSerializesCompletedToolCallAsDeltaToolCalls() throws {
    let call = sampleToolCall()
    let chunk = ChatCompletionChunk(
        id: "chatcmpl_test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: ServerConfig.canonicalModelID,
        choices: [
            ChunkChoice(
                index: 0,
                delta: ChatMessage(
                    role: nil,
                    content: nil,
                    reasoning: nil,
                    reasoning_content: nil,
                    tool_calls: [call]
                ),
                finish_reason: nil
            )
        ],
        usage: nil
    )

    let data = try JSONEncoder().encode(chunk)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let choices = json["choices"] as! [[String: Any]]
    let delta = choices[0]["delta"] as! [String: Any]
    let toolCalls = delta["tool_calls"] as! [[String: Any]]

    #expect(toolCalls.count == 1)
    #expect(toolCalls[0]["id"] as? String == "call_abc123")
    #expect(toolCalls[0]["type"] as? String == "function")
    let function = toolCalls[0]["function"] as? [String: Any]
    #expect(function?["name"] as? String == "get_weather")
    #expect(function?["arguments"] as? String == "{\"location\":\"Paris\"}")

    // A nil finish_reason must be omitted from the chunk choice.
    #expect(choices[0]["finish_reason"] == nil)
    // A nil usage must be omitted from the chunk.
    #expect(json["usage"] == nil)
}

// MARK: - Non-streaming message.tool_calls

@Test
func nonStreamingResponseSerializesAccumulatedToolCalls() throws {
    let calls = [sampleToolCall(), sampleToolCall(id: "call_def456", name: "get_time")]
    let response = ChatCompletionResponse(
        id: "chatcmpl_test",
        object: "chat.completion",
        created: 1700000000,
        model: ServerConfig.canonicalModelID,
        choices: [
            CompletionChoice(
                index: 0,
                message: ChatMessage(
                    role: "assistant",
                    content: nil,
                    reasoning: nil,
                    reasoning_content: nil,
                    tool_calls: calls
                ),
                finish_reason: "tool_calls"
            )
        ],
        usage: Usage(prompt_tokens: 10, completion_tokens: 5, total_tokens: 15)
    )

    let data = try JSONEncoder().encode(response)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let choices = json["choices"] as! [[String: Any]]
    let message = choices[0]["message"] as! [String: Any]
    let toolCalls = message["tool_calls"] as! [[String: Any]]

    #expect(toolCalls.count == 2)
    #expect(toolCalls[0]["id"] as? String == "call_abc123")
    #expect(toolCalls[1]["id"] as? String == "call_def456")
    #expect(choices[0]["finish_reason"] as? String == "tool_calls")
}

// MARK: - Empty tool_calls omitted

@Test
func emptyToolCallsAreOmittedFromMessage() throws {
    let message = ChatMessage(
        role: "assistant",
        content: "Hello",
        reasoning: nil,
        reasoning_content: nil,
        tool_calls: []
    )
    let data = try JSONEncoder().encode(message)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(json["tool_calls"] == nil)
    #expect(json["content"] as? String == "Hello")
}

// MARK: - ToolSpec.toJSONString stability

@Test
func toolSpecToJSONStringIsStableAndFaithful() throws {
    let spec = ToolSpec(
        type: "function",
        function: ToolSpecFunction(
            name: "get_weather",
            description: "Get the weather for a location",
            parameters: JSONValue.object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object(["type": .string("string")])
                ]),
                "required": .array([.string("location")])
            ])
        )
    )

    let a = spec.toJSONString()
    let b = spec.toJSONString()
    #expect(a == b)

    // The string must be valid JSON that decodes back to the same spec.
    let data = a.data(using: .utf8)!
    let decoded = try JSONDecoder().decode(ToolSpec.self, from: data)
    #expect(decoded.type == "function")
    #expect(decoded.function.name == "get_weather")
    #expect(decoded.function.description == "Get the weather for a location")
}

// MARK: - ToolChoice round-trip

@Test
func toolChoiceRoundTripsThroughCodable() throws {
    let cases: [ToolChoice] = [
        .auto,
        .none,
        .required,
        .function(name: "get_weather")
    ]
    for choice in cases {
        let data = try JSONEncoder().encode(choice)
        let decoded = try JSONDecoder().decode(ToolChoice.self, from: data)
        let reencoded = try JSONEncoder().encode(decoded)
        // Compare the decoded values structurally (order-independent):
        // re-encoding the decoded value must round-trip to the same decoded value.
        let redecoded = try JSONDecoder().decode(ToolChoice.self, from: reencoded)
        #expect(toolChoiceKey(decoded) == toolChoiceKey(redecoded))
        #expect(toolChoiceKey(choice) == toolChoiceKey(decoded))
    }
}

@Test
func toolChoiceIsNoneOnlyForNoneCase() {
    #expect(ToolChoice.none.isNone)
    #expect(!ToolChoice.auto.isNone)
    #expect(!ToolChoice.required.isNone)
    #expect(!ToolChoice.function(name: "x").isNone)
}

/// A stable, order-independent key for a `ToolChoice` value, used to compare
/// round-tripped values without requiring `Equatable` conformance.
private func toolChoiceKey(_ choice: ToolChoice) -> String {
    switch choice {
    case .auto: return "auto"
    case .none: return "none"
    case .required: return "required"
    case .function(let name): return "function:\(name)"
    }
}
