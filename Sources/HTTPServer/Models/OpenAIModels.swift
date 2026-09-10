import Foundation

struct ChatCompletionRequest: Codable, Sendable {
    let model: String
    let messages: [ChatMessage]

    var stream: Bool?

    // Token limits
    var max_tokens: Int?
    var max_completion_tokens: Int?

    // Number of completions to generate. This server supports exactly one
    // completion; `n` must be 1 when present.
    var n: Int?

    // Sampling parameters accepted for OpenAI-compatible clients.
    // NOTE: Qwen36MTPBlockSession is currently greedy, so these are accepted/logged
    // and reserved for a future speculative-sampling implementation.
    var temperature: Float?
    var top_p: Float?
    var top_k: Int?
    var min_p: Float?
    var repetition_penalty: Float?
    var presence_penalty: Float?
    var frequency_penalty: Float?

    // Stream options
    var stream_options: StreamOptions?

    // Stop sequences: a single string or an array of strings. Generation
    // stops when any of these sequences appears in the committed output.
    var stop: StopSequences?

    // Custom server extensions
    var context_window: Int?
    var enable_thinking: Bool?
    var mtp_enabled: Bool?
    var kv_cache_bits: Int?
    var kv_cache_group_size: Int?
    var quantized_kv_start: Int?
    var ttl_seconds: Int?
    var chat_template: String?
    var chat_template_kwargs: [String: String]?

    // Tool calling
    var tools: [ToolSpec]?
    var tool_choice: ToolChoice?
}

struct StreamOptions: Codable, Sendable {
    let include_usage: Bool?
}

/// OpenAI `stop` field: either a single string or an array of strings.
struct StopSequences: Codable, Sendable {
    /// The stop sequences in request order (may include empty entries; the
    /// validator rejects those).
    let sequences: [String]

    init(sequences: [String]) {
        self.sequences = sequences
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(String.self) {
            self.sequences = [single]
        } else if let list = try? container.decode([String].self) {
            self.sequences = list
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "'stop' must be a string or an array of strings."
                )
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if sequences.count == 1 {
            try container.encode(sequences[0])
        } else {
            try container.encode(sequences)
        }
    }
}

// MARK: - ChatMessage (Omits nil keys)
struct ChatMessage: Codable, Sendable {
    let role: String?
    let content: String?
    let reasoning: String?
    let reasoning_content: String?
    let tool_calls: [ToolCall]?
    let tool_call_id: String?
    /// The tool name for `role: "tool"` (and legacy `role: "function"`)
    /// messages. Optional; ignored by the current chat template, which
    /// renders tool results from `content` only.
    let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, reasoning, reasoning_content, tool_calls, tool_call_id, name
    }

    init(
        role: String?,
        content: String?,
        reasoning: String? = nil,
        reasoning_content: String? = nil,
        tool_calls: [ToolCall]? = nil,
        tool_call_id: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let role = role { try container.encode(role, forKey: .role) }
        if let content = content { try container.encode(content, forKey: .content) }
        if let reasoning = reasoning { try container.encode(reasoning, forKey: .reasoning) }
        if let reasoning_content = reasoning_content { try container.encode(reasoning_content, forKey: .reasoning_content) }
        if let tool_calls = tool_calls, !tool_calls.isEmpty { try container.encode(tool_calls, forKey: .tool_calls) }
        if let tool_call_id = tool_call_id { try container.encode(tool_call_id, forKey: .tool_call_id) }
        if let name = name { try container.encode(name, forKey: .name) }
    }
}

/// A tool call on an assistant message (input history) or a generated tool
/// call (output). `arguments` is a JSON string, matching the OpenAI shape.
struct ToolCall: Codable, Sendable {
    let id: String?
    let type: String?
    let index: Int?
    let function: ToolCallFunction

    struct ToolCallFunction: Codable, Sendable {
        let name: String
        let arguments: String?
    }
}

/// Arbitrary JSON value, used for tool parameter schemas and other
/// free-form JSON fields. Decodes any JSON shape and re-encodes it
/// faithfully so the value can be handed to the chat template.
public enum JSONValue: Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "Unsupported JSON value")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }
}

/// A tool declaration in an OpenAI chat completion request.
public struct ToolSpec: Codable, Sendable {
    public let type: String
    public let function: ToolSpecFunction

    public init(type: String, function: ToolSpecFunction) {
        self.type = type
        self.function = function
    }
}

public extension ToolSpec {
    /// A stable JSON string representation, used as a cache key component.
    /// Object keys are sorted at every level so the string is deterministic
    /// for a given spec (JSONEncoder does not guarantee dictionary key order).
    func toJSONString() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let value = try? JSONDecoder().decode(JSONValue.self, from: data) else {
            return "unserializable"
        }
        return ToolSpec.canonicalJSON(value)
    }

    private static func canonicalJSON(_ value: JSONValue) -> String {
        switch value {
        case .null:
            return "null"
        case .bool(let b):
            return b ? "true" : "false"
        case .int(let i):
            return String(i)
        case .double(let d):
            return String(d)
        case .string(let s):
            return jsonString(s)
        case .array(let items):
            return "[" + items.map(canonicalJSON).joined(separator: ",") + "]"
        case .object(let obj):
            let entries = obj.keys.sorted().map { key in
                "\(jsonString(key)):\(canonicalJSON(obj[key]!))"
            }
            return "{" + entries.joined(separator: ",") + "}"
        }
    }

    private static func jsonString(_ s: String) -> String {
        var result = "\""
        for ch in s {
            switch ch {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            default: result.append(ch)
            }
        }
        result.append("\"")
        return result
    }
}

public struct ToolSpecFunction: Codable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue?

    public init(name: String, description: String?, parameters: JSONValue?) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// The `tool_choice` field. Mirrors the OpenAI shape: `"auto"`, `"none"`,
/// `"required"`, or `{"type": "function", "function": {"name": ...}}`.
public enum ToolChoice: Codable, Sendable {
    case auto
    case none
    case required
    case function(name: String)

    public var isNone: Bool {
        if case .none = self { return true }
        return false
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let raw = try? container.decode(String.self) {
            switch raw {
            case "auto":
                self = .auto
            case "none":
                self = .none
            case "required":
                self = .required
            default:
                throw DecodingError.dataCorrupted(
                    .init(codingPath: [], debugDescription: "Unsupported tool_choice value: \(raw)")
                )
            }
        } else if let object = try? container.decode(ToolChoiceFunction.self) {
            self = .function(name: object.function.name)
        } else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "tool_choice must be a string or a function object")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .auto:
            try container.encode("auto")
        case .none:
            try container.encode("none")
        case .required:
            try container.encode("required")
        case .function(let name):
            try container.encode(ToolChoiceFunction(type: "function", function: .init(name: name)))
        }
    }
}

private struct ToolChoiceFunction: Codable, Sendable {
    let type: String
    let function: ToolChoiceFunctionInner

    struct ToolChoiceFunctionInner: Codable, Sendable {
        let name: String
    }
}

struct Usage: Codable, Sendable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}

struct ChatCompletionResponse: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: Usage?
}

struct CompletionChoice: Codable, Sendable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String
}

// MARK: - ChunkChoice (Omits nil finish_reason)
struct ChunkChoice: Codable, Sendable {
    let index: Int
    let delta: ChatMessage
    let finish_reason: String?
    
    enum CodingKeys: String, CodingKey {
        case index, delta, finish_reason
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(index, forKey: .index)
        try container.encode(delta, forKey: .delta)
        if let finish_reason = finish_reason { try container.encode(finish_reason, forKey: .finish_reason) }
    }
}

// MARK: - ChatCompletionChunk (Omits nil usage)
struct ChatCompletionChunk: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChunkChoice]
    let usage: Usage?
    
    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(object, forKey: .object)
        try container.encode(created, forKey: .created)
        try container.encode(model, forKey: .model)
        try container.encode(choices, forKey: .choices)
        if let usage = usage { try container.encode(usage, forKey: .usage) }
    }
}

struct Model: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
    let context_length: Int?
}

struct ModelsListResponse: Codable, Sendable {
    let object: String
    let data: [Model]
}