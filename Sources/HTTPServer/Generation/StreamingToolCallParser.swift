    // StreamingToolCallParser.swift
    //
    // Incrementally parses streaming model output into visible content, reasoning,
    // and OpenAI-style tool calls. It is a strict superset of
    // `SimpleStreamingReasoningParser`: when `toolCallsEnabled` is false it
    // delegates to that parser and is therefore byte-identical to it (same
    // scanning, same retained suffix). That equivalence is the gating mechanism
    // that keeps the no-tool pipeline unchanged.
    //
    // Single-pass states: idle / inReasoning / inToolCall.
    //   - inReasoning scans for the reasoning end tag only.
    //   - inToolCall scans for the tool-call block terminator only.
    //   - idle scans for the tool-call open tag (when enabled) and the reasoning
    //     open tag; the earliest match wins.
    //
    // The retained suffix is `longest relevant tag length - 1`, computed from the
    // tag string constants (never a magic literal), so a tag split across token
    // boundaries is never emitted prematurely.
    //
    // Tool-call block dialects (dispatched on the block's first non-whitespace
    // character):
    //   - XML (primary): <function=NAME><parameter=KEY>VALUE</parameter>...</function>
    //   - JSON (secondary): {"name": ..., "arguments": {...}}
    //
    // A malformed block is emitted verbatim as content (open tag + block + close
    // tag) and never throws. An unterminated block at finish() is flushed as
    // content. Multiple blocks get an incrementing index from a counter.

    import Foundation

    internal struct StreamingToolCallParser: Sendable {
        private enum State {
            case idle
            case inReasoning
            case inToolCall
        }

        // Tag constants are built from fragments so the full control-token literal
        // never appears as a single string in source.
        internal static let reasoningStartTag = "<" + "think>"
        internal static let reasoningEndTag = "</" + "think>"
        internal static let toolCallStartTag = "<tool_call>"
        internal static let toolCallEndTag = "</tool_call>"

        private let toolCallsEnabled: Bool
        private let enableThinking: Bool // 1. Store enableThinking

        // Shared reasoning path, used verbatim when tool calls are disabled.
        private var reasoning: SimpleStreamingReasoningParser

        // Tool-call state (only active when toolCallsEnabled).
        private var state: State
        private var buffer = ""
        private var toolCallIndex = 0
        private var toolCallBlock = ""

        init(toolCallsEnabled: Bool, enableThinking: Bool = true) {
            self.toolCallsEnabled = toolCallsEnabled
            self.enableThinking = enableThinking // Added property assignment
            self.reasoning = SimpleStreamingReasoningParser(enableThinking: enableThinking)
            self.state = enableThinking ? .inReasoning : .idle
        }

        /// Feed a decoded text chunk; returns the fragments it yields.
        mutating func parse(_ text: String) -> [GenerationFragment] {
            // FAST-PATH: If neither tools nor thinking are active, pass text straight through
            if !toolCallsEnabled && !enableThinking {
                return [.content(text)]
            }
            if !toolCallsEnabled {
                return reasoning.parse(text)
            }
            return parseEnabled(text)
        }

        /// Flush any retained text after generation ends.
        mutating func finish() -> [GenerationFragment] {
            // FAST-PATH: Nothing was buffered, return empty
            if !toolCallsEnabled && !enableThinking {
                return []
            }
            if !toolCallsEnabled {
                return reasoning.finish()
            }
            return finishEnabled()
        }

        // MARK: - Enabled state machine

        private mutating func parseEnabled(_ text: String) -> [GenerationFragment] {
            buffer += text
            var fragments: [GenerationFragment] = []
            var iterations = 0

            parseLoop: while !buffer.isEmpty {
                iterations += 1
                if iterations > 100000 {
                    // Defensive: malformed input must never crash the process.
                    // Flush whatever is left as content and stop.
                    if !buffer.isEmpty {
                        fragments.append(.content(buffer))
                    }
                    buffer = ""
                    break parseLoop
                }

                switch state {
                case .inReasoning:
                    if let endRange = buffer.range(of: Self.reasoningEndTag) {
                        let reasoningText = String(buffer[..<endRange.lowerBound])
                        if !reasoningText.isEmpty {
                            fragments.append(.reasoning(reasoningText))
                        }
                        buffer.removeSubrange(..<endRange.upperBound)
                        state = .idle
                        continue parseLoop
                    }

                    let retained = Self.reasoningEndTag.count - 1
                    if buffer.count > retained {
                        let splitIndex = buffer.index(buffer.endIndex, offsetBy: -retained)
                        let reasoningText = String(buffer[..<splitIndex])
                        buffer = String(buffer[splitIndex...])
                        if !reasoningText.isEmpty {
                            fragments.append(.reasoning(reasoningText))
                        }
                    }
                    break parseLoop

                case .inToolCall:
                    if let endRange = buffer.range(of: Self.toolCallEndTag) {
                        let block = toolCallBlock + String(buffer[..<endRange.lowerBound])
                        buffer.removeSubrange(..<endRange.upperBound)
                        state = .idle
                        toolCallBlock = ""
                        fragments.append(contentsOf: emitToolCall(block))
                        continue parseLoop
                    }

                    let retained = Self.toolCallEndTag.count - 1
                    if buffer.count > retained {
                        let splitIndex = buffer.index(buffer.endIndex, offsetBy: -retained)
                        toolCallBlock += String(buffer[..<splitIndex])
                        buffer = String(buffer[splitIndex...])
                    }
                    break parseLoop

                case .idle:
                    let toolRange = buffer.range(of: Self.toolCallStartTag)
                    let thinkRange = buffer.range(of: Self.reasoningStartTag)
                    let match: Range<String.Index>?
                    switch (toolRange, thinkRange) {
                    case (let t?, let th?):
                        match = t.lowerBound < th.lowerBound ? t : th
                    case (let t?, _):
                        match = t
                    case (_, let th?):
                        match = th
                    case (nil, nil):
                        match = nil
                    }

                    if let range = match {
                        let content = String(buffer[..<range.lowerBound])
                        if !content.isEmpty {
                            fragments.append(.content(content))
                        }
                        let matched = String(buffer[range])
                        buffer.removeSubrange(..<range.upperBound)
                        if matched == Self.toolCallStartTag {
                            state = .inToolCall
                            toolCallBlock = ""
                        } else {
                            state = .inReasoning
                        }
                        continue parseLoop
                    }

                    let retained =
                        max(Self.toolCallStartTag.count, Self.reasoningStartTag.count) - 1
                    let emitCount = buffer.count - retained
                    if emitCount > 0 {
                        let splitIndex = buffer.index(buffer.startIndex, offsetBy: emitCount)
                        let content = String(buffer[..<splitIndex])
                        buffer = String(buffer[splitIndex...])
                        if !content.isEmpty {
                            fragments.append(.content(content))
                        }
                    }

                    break parseLoop
                }
            }

            return fragments
        }

        private mutating func finishEnabled() -> [GenerationFragment] {
            var fragments: [GenerationFragment] = []

            switch state {
            case .inReasoning:
                if !buffer.isEmpty {
                    fragments.append(.reasoning(buffer))
                }
            case .inToolCall:
                // Unterminated block: flush the open tag plus everything seen so far
                // as content.
                let full = Self.toolCallStartTag + toolCallBlock + buffer
                if !full.isEmpty {
                    fragments.append(.content(full))
                }
            case .idle:
                if !buffer.isEmpty {
                    fragments.append(.content(buffer))
                }
            }

            buffer = ""
            toolCallBlock = ""
            return fragments
        }

        // MARK: - Tool-call block parsing

        /// Parses one complete block (the text between the open and close tags).
        /// Returns a `.toolCall` fragment when well-formed, otherwise emits the
        /// block verbatim as content. Never throws.
        private mutating func emitToolCall(_ block: String) -> [GenerationFragment] {
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            let parsed: (name: String, pairs: [(key: String, rawValue: String)])?

            switch trimmed.first {
            case "<":
                parsed = Self.parseXMLBlock(block)
            case "{":
                parsed = Self.parseJSONBlock(trimmed)
            default:
                parsed = nil
            }

            if let (name, pairs) = parsed {
                let arguments = Self.serializeArguments(pairs)
                let call = ToolCall(
                    id: "call_" + UUID().uuidString,
                    type: "function",
                    index: toolCallIndex,
                    function: ToolCall.ToolCallFunction(
                        name: name,
                        arguments: arguments
                    )
                )
                toolCallIndex += 1
                return [.toolCall(call)]
            }

            // Malformed: emit open tag + block + close tag verbatim as content.
            return [.content(Self.toolCallStartTag + block + Self.toolCallEndTag)]
        }

        /// XML dialect: <function=NAME> followed by zero or more
        /// <parameter=KEY>VALUE</parameter> pairs. Returns nil if the block does not
        /// match the expected shape.
        private static func parseXMLBlock(
            _ block: String
        ) -> (name: String, pairs: [(key: String, rawValue: String)])? {
            guard let fnStart = block.range(of: "<function=") else { return nil }
            let afterFn = block[fnStart.upperBound...]
            guard let fnEnd = afterFn.range(of: ">") else { return nil }
            let name = String(afterFn[..<fnEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }

            let rest = afterFn[fnEnd.upperBound...]
            var pairs: [(key: String, rawValue: String)] = []
            var searchStart = rest.startIndex

            while let pStart = rest.range(of: "<parameter=", range: searchStart..<rest.endIndex) {
                let afterP = rest[pStart.upperBound...]
                guard let pEnd = afterP.range(of: ">") else { return nil }
                let key = String(afterP[..<pEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else { return nil }

                let afterPEnd = afterP[pEnd.upperBound...]
                guard let pClose = afterPEnd.range(of: "</parameter>") else { return nil }
                var value = String(afterPEnd[..<pClose.lowerBound])
                // The chat template places each parameter value on its own line
                // (`<parameter=K>\nVALUE\n</parameter>`); strip exactly the one
                // formatting newline on each side, preserving any internal
                // newlines (multi-line values are supported).
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }

                pairs.append((key, value))
                searchStart = pClose.upperBound
            }

            return (name, pairs)
        }

        /// JSON dialect: a `{"name": ..., "arguments": {...}}` object. Returns the
        /// function name and the ordered (key, rawValue) argument pairs, or nil if
        /// the block is not a valid JSON object with a string `name` and an object
        /// `arguments`.
        private static func parseJSONBlock(
            _ text: String
        ) -> (name: String, pairs: [(key: String, rawValue: String)])? {
            var scanner = JSONScanner(text)
            guard scanner.parseObjectCapture() else { return nil }

            var name: String?
            var arguments: [(key: String, rawValue: String)]?

            for (key, raw) in scanner.capturedTopLevel {
                if key == "name" {
                    guard let decoded = scanner.decodeString(raw) else { return nil }
                    name = decoded
                } else if key == "arguments" {
                    guard let sub = scanner.parseOrderedObject(raw) else { return nil }
                    arguments = sub
                }
            }

            guard let name = name, let arguments = arguments else { return nil }
            return (name, arguments)
        }

        /// Serializes ordered (key, rawValue) pairs into a JSON object string.
        /// A value that is already a valid JSON literal is emitted as-is; otherwise
        /// it is JSON-encoded as a string. Key order is preserved.
        private static func serializeArguments(
            _ pairs: [(key: String, rawValue: String)]
        ) -> String {
            var parts: [String] = []
            for pair in pairs {
                let keyJSON = jsonString(pair.key)
                let valueJSON = isValidJSONLiteral(pair.rawValue)
                    ? pair.rawValue
                    : jsonString(pair.rawValue)
                parts.append("\(keyJSON):\(valueJSON)")
            }
            return "{\(parts.joined(separator: ","))}"
        }

        private static func isValidJSONLiteral(_ raw: String) -> Bool {
            guard let data = raw.data(using: .utf8) else { return false }
            // .fragmentsAllowed is required: a top-level scalar (string, number,
            // boolean, null) is a valid JSON literal but is rejected by
            // JSONSerialization without that option.
            return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) != nil
        }

        private static func jsonString(_ value: String) -> String {
            guard let data = try? JSONSerialization.data(
                withJSONObject: [value],
                options: []
            ) else {
                return "\"\(value)\""
            }
            // JSONSerialization produces ["value"]; strip the surrounding brackets.
            let full = String(decoding: data, as: UTF8.self)
            return String(full.dropFirst().dropLast())
        }
    }

    // MARK: - Minimal order-preserving JSON scanner

    /// A small recursive-descent JSON scanner that records top-level object
    /// members in source order (Swift's `JSONSerialization` loses key order).
    struct JSONScanner {
        private let chars: [Character]
        private var pos = 0

        /// Top-level members captured by `parseObjectCapture()`, in source order.
        var capturedTopLevel: [(key: String, rawValue: String)] = []

        init(_ text: String) {
            chars = Array(text)
        }

        private mutating func skipWS() {
            while pos < chars.count {
                let c = chars[pos]
                if c == " " || c == "\n" || c == "\t" || c == "\r" {
                    pos += 1
                } else {
                    break
                }
            }
        }

        /// Parses a top-level object, recording each member's key and the raw
        /// source text of its value (in order). Returns false on malformed input.
        mutating func parseObjectCapture() -> Bool {
            skipWS()
            guard pos < chars.count, chars[pos] == "{" else { return false }
            pos += 1
            skipWS()
            if pos < chars.count, chars[pos] == "}" {
                pos += 1
                return true
            }

            while true {
                skipWS()
                guard pos < chars.count, chars[pos] == "\"" else { return false }
                guard let key = parseString() else { return false }
                skipWS()
                guard pos < chars.count, chars[pos] == ":" else { return false }
                pos += 1
                skipWS()
                let valueStart = pos
                guard parseValue() else { return false }
                let rawValue = String(chars[valueStart..<pos])
                capturedTopLevel.append((key, rawValue))

                skipWS()
                if pos < chars.count, chars[pos] == "," {
                    pos += 1
                    continue
                } else if pos < chars.count, chars[pos] == "}" {
                    pos += 1
                    return true
                } else {
                    return false
                }
            }
        }

        /// Parses an object from `text`, returning its ordered (key, rawValue)
        /// members. Returns nil on malformed input.
        mutating func parseOrderedObject(_ text: String) -> [(key: String, rawValue: String)]? {
            var sub = JSONScanner(text)
            guard sub.parseObjectCapture() else { return nil }
            return sub.capturedTopLevel
        }

        /// Decodes a raw JSON string literal into its Swift string value.
        mutating func decodeString(_ raw: String) -> String? {
            var sub = JSONScanner(raw)
            return sub.parseString()
        }

        private mutating func parseString() -> String? {
            guard pos < chars.count, chars[pos] == "\"" else { return nil }
            pos += 1
            var result = ""
            while pos < chars.count {
                let c = chars[pos]
                if c == "\"" {
                    pos += 1
                    return result
                }
                if c == "\\" {
                    pos += 1
                    guard pos < chars.count else { return nil }
                    let esc = chars[pos]
                    switch esc {
                    case "\"": result.append("\"")
                    case "\\": result.append("\\")
                    case "/": result.append("/")
                    case "b": result.append("\u{0008}")
                    case "f": result.append("\u{000C}")
                    case "n": result.append("\n")
                    case "r": result.append("\r")
                    case "t": result.append("\t")
                    case "u":
                        pos += 1
                        guard pos + 4 <= chars.count else { return nil }
                        let hex = String(chars[pos..<pos + 4])
                        guard let code = UInt32(hex, radix: 16),
                            let scalar = UnicodeScalar(code) else { return nil }
                        result.append(Character(scalar))
                        pos += 4
                    default:
                        return nil
                    }
                    pos += 1
                } else {
                    result.append(c)
                    pos += 1
                }
            }
            return nil
        }

        private mutating func parseValue() -> Bool {
            skipWS()
            guard pos < chars.count else { return false }
            switch chars[pos] {
            case "{": return parseObject()
            case "[": return parseArray()
            case "\"": return parseString() != nil
            case "t": return matchLiteral("true")
            case "f": return matchLiteral("false")
            case "n": return matchLiteral("null")
            default: return parseNumber()
            }
        }

        private mutating func parseObject() -> Bool {
            guard pos < chars.count, chars[pos] == "{" else { return false }
            pos += 1
            skipWS()
            if pos < chars.count, chars[pos] == "}" {
                pos += 1
                return true
            }
            while true {
                skipWS()
                guard pos < chars.count, chars[pos] == "\"" else { return false }
                guard parseString() != nil else { return false }
                skipWS()
                guard pos < chars.count, chars[pos] == ":" else { return false }
                pos += 1
                guard parseValue() else { return false }
                skipWS()
                if pos < chars.count, chars[pos] == "," {
                    pos += 1
                    continue
                } else if pos < chars.count, chars[pos] == "}" {
                    pos += 1
                    return true
                } else {
                    return false
                }
            }
        }

        private mutating func parseArray() -> Bool {
            guard pos < chars.count, chars[pos] == "[" else { return false }
            pos += 1
            skipWS()
            if pos < chars.count, chars[pos] == "]" {
                pos += 1
                return true
            }
            while true {
                guard parseValue() else { return false }
                skipWS()
                if pos < chars.count, chars[pos] == "," {
                    pos += 1
                    continue
                } else if pos < chars.count, chars[pos] == "]" {
                    pos += 1
                    return true
                } else {
                    return false
                }
            }
        }

        private mutating func matchLiteral(_ s: String) -> Bool {
            let arr = Array(s)
            guard pos + arr.count <= chars.count else { return false }
            for i in 0..<arr.count where chars[pos + i] != arr[i] {
                return false
            }
            pos += arr.count
            return true
        }

        private mutating func parseNumber() -> Bool {
            let start = pos
            if pos < chars.count, chars[pos] == "-" { pos += 1 }
            guard pos < chars.count, chars[pos].isNumber else { return false }
            while pos < chars.count, chars[pos].isNumber { pos += 1 }
            if pos < chars.count, chars[pos] == "." {
                pos += 1
                guard pos < chars.count, chars[pos].isNumber else { return false }
                while pos < chars.count, chars[pos].isNumber { pos += 1 }
            }
            if pos < chars.count, chars[pos] == "e" || chars[pos] == "E" {
                pos += 1
                if pos < chars.count, chars[pos] == "+" || chars[pos] == "-" { pos += 1 }
                guard pos < chars.count, chars[pos].isNumber else { return false }
                while pos < chars.count, chars[pos].isNumber { pos += 1 }
            }
            return pos > start
        }
    }