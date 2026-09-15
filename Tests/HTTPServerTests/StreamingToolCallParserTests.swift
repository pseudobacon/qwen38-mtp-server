// StreamingToolCallParserTests.swift
//
// Unit tests for `StreamingToolCallParser`: incremental parsing of streaming
// model output into visible content, reasoning, and OpenAI-style tool calls.
//
// Pure Swift (no MLX, no model weights): runs on any machine. These tests pin
// the parse contract — well-formed tool calls are extracted, reasoning is
// separated, multiple calls get an incrementing index, and malformed / 
// unterminated blocks degrade to verbatim content (never throw, never crash).

import Testing
import Foundation
@testable import HTTPServer

// MARK: - Helpers

private let kToolOpen = StreamingToolCallParser.toolCallStartTag
private let kToolClose = StreamingToolCallParser.toolCallEndTag
private let kThinkEnd = StreamingToolCallParser.reasoningEndTag

private func runParser(
    _ text: String,
    toolCallsEnabled: Bool = true,
    enableThinking: Bool = false
) -> [GenerationFragment] {
    var parser = StreamingToolCallParser(toolCallsEnabled: toolCallsEnabled, enableThinking: enableThinking)
    var all = parser.parse(text)
    all.append(contentsOf: parser.finish())
    return all
}

private func contentOf(_ fragments: [GenerationFragment]) -> String {
    fragments.compactMap { if case .content(let s) = $0 { return s }; return nil }.joined()
}

private func reasoningOf(_ fragments: [GenerationFragment]) -> String {
    fragments.compactMap { if case .reasoning(let s) = $0 { return s }; return nil }.joined()
}

private func toolCallsOf(_ fragments: [GenerationFragment]) -> [ToolCall] {
    fragments.compactMap { if case .toolCall(let c) = $0 { return c }; return nil }
}

// MARK: - Well-formed tool calls

@Test
func wellFormedXMLToolCallIsParsed() {
    let input = "\(kToolOpen)\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n\(kToolClose)"
    let fragments = runParser(input)
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.name == "get_weather")
    #expect(calls[0].function.arguments == "{\"city\":\"Beijing\"}")
    #expect(calls[0].index == 0)
    #expect(calls[0].type == "function")
    // No spurious visible content from a well-formed block.
    #expect(contentOf(fragments).isEmpty)
}

@Test
func jsonDialectToolCallIsParsed() {
    let input = "\(kToolOpen)\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\", \"days\": 3}}\n\(kToolClose)"
    let fragments = runParser(input)
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.name == "get_weather")
    #expect(calls[0].function.arguments == "{\"city\":\"Beijing\",\"days\":3}")
}

@Test
func emptyParametersProduceEmptyArgumentsObject() {
    let input = "\(kToolOpen)\n<function=ping>\n</function>\n\(kToolClose)"
    let fragments = runParser(input)
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.name == "ping")
    #expect(calls[0].function.arguments == "{}")
}

@Test
func multipleToolCallsGetIncrementingIndex() {
    let input = """
    \(kToolOpen)\n<function=first>\n</function>\n\(kToolClose)
    \(kToolOpen)\n<function=second>\n<parameter=a>\n1\n</parameter>\n</function>\n\(kToolClose)
    """
    let fragments = runParser(input)
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 2)
    #expect(calls[0].function.name == "first")
    #expect(calls[0].index == 0)
    #expect(calls[1].function.name == "second")
    #expect(calls[1].index == 1)
}

// MARK: - Reasoning separation

@Test
func reasoningIsSeparatedFromContent() {
    // enableThinking=true: the parser opens in the reasoning state and treats
    // text as reasoning until the end tag.
    let input = "thinking deeply\(kThinkEnd)hello world"
    let fragments = runParser(input, toolCallsEnabled: true, enableThinking: true)
    #expect(reasoningOf(fragments) == "thinking deeply")
    #expect(contentOf(fragments) == "hello world")
    #expect(toolCallsOf(fragments).isEmpty)
}

@Test
func toolCallAfterReasoning() {
    let input = "reasoning\(kThinkEnd)\(kToolOpen)\n<function=f>\n</function>\n\(kToolClose)"
    let fragments = runParser(input, toolCallsEnabled: true, enableThinking: true)
    #expect(reasoningOf(fragments) == "reasoning")
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.name == "f")
}

// MARK: - Malformed / unterminated degradation

@Test
func malformedBlockIsEmittedAsContent() {
    // Neither valid XML nor JSON: emitted verbatim as content, never throws.
    let input = "\(kToolOpen)\nthis is not a valid block\(kToolClose)"
    let fragments = runParser(input)
    #expect(toolCallsOf(fragments).isEmpty)
    #expect(contentOf(fragments).contains("this is not a valid block"))
}

@Test
func unterminatedBlockIsFlushedAsContentOnFinish() {
    // Open tag + partial block, then generation ends: flushed as content.
    let input = "\(kToolOpen)\n<function=never_closed>"
    let fragments = runParser(input)
    #expect(toolCallsOf(fragments).isEmpty)
    #expect(contentOf(fragments).contains("<function=never_closed>"))
}

@Test
func toolCallTagSplitAcrossChunksIsNotPrematurelyEmitted() {
    // Feed the open marker one character at a time; the parser must retain the
    // suffix so a tag split across token boundaries is never emitted as content.
    var parser = StreamingToolCallParser(toolCallsEnabled: true, enableThinking: false)
    let full = "\(kToolOpen)\n<function=f>\n</function>\n\(kToolClose)"
    var fragments: [GenerationFragment] = []
    for ch in full {
        fragments.append(contentsOf: parser.parse(String(ch)))
    }
    fragments.append(contentsOf: parser.finish())
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.name == "f")
    // No content should leak the tag fragments.
    #expect(contentOf(fragments).isEmpty)
}

// MARK: - Disabled mode

@Test
func disabledModePassesContentThrough() {
    // toolCallsEnabled=false, enableThinking=false: pure content passthrough.
    let input = "hello \(kToolOpen) world"
    let fragments = runParser(input, toolCallsEnabled: false, enableThinking: false)
    #expect(contentOf(fragments) == input)
    #expect(toolCallsOf(fragments).isEmpty)
}

// MARK: - Argument serialization edge cases

@Test
func argumentValuesThatAreJSONLiteralsArePreserved() {
    // A parameter whose value is already a valid JSON literal (number, bool,
    // null, array, object) is emitted as-is rather than re-quoted.
    let input = "\(kToolOpen)\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n</function>\n\(kToolClose)"
    let fragments = runParser(input)
    let calls = toolCallsOf(fragments)
    #expect(calls.count == 1)
    #expect(calls[0].function.arguments == "{\"n\":42,\"b\":true}")
}