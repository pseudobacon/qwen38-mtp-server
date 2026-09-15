// OpenAIValidation.swift
//
// Request validation and OpenAI-shaped error responses for the Qwen 3.8 MTP
// server.
//
// This layer is pure Swift (no MLX, no model weights) and runs before any
// model execution. It resolves the request against the server configuration
// and rejects anything the current runtime does not actually apply, so the
// API surface is truthful: every accepted field is applied, and every
// unsupported field is rejected with an OpenAI-shaped 400.

import Foundation
import Vapor

/// An OpenAI-shaped request error.
///
/// Serialized as:
/// ```json
/// {
///   "error": {
///     "message": "...",
///     "type": "invalid_request_error",
///     "param": "field_name",
///     "code": "invalid_value"
///   }
/// }
/// ```
///
/// `param` is omitted from the JSON when it is intentionally absent.
public struct OpenAIRequestError: Error, Sendable {
    public let message: String
    public let type: String
    public let param: String?
    public let code: String

    public init(
        message: String,
        type: String = "invalid_request_error",
        param: String?,
        code: String
    ) {
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}

/// The OpenAI error envelope.
public struct OpenAIErrorEnvelope: Codable, Sendable {
    public struct ErrorDetail: Codable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: ErrorDetail
}

/// Builds an OpenAI-shaped error envelope with an arbitrary HTTP status.
/// Reused by the session endpoints (404/409) which are not 400s.
public func openAIStatusErrorResponse(
    status: HTTPStatus,
    message: String,
    type: String = "invalid_request_error",
    param: String? = nil,
    code: String
) -> Response {
    let envelope = OpenAIErrorEnvelope(
        error: .init(message: message, type: type, param: param, code: code)
    )
    let data = try! JSONEncoder().encode(envelope)
    return Response(
        status: status,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: data)
    )
}

/// Builds the HTTP 400 response for an `OpenAIRequestError`.
public func openAIErrorResponse(_ error: OpenAIRequestError) -> Response {
    let envelope = OpenAIErrorEnvelope(
        error: .init(
            message: error.message,
            type: error.type,
            param: error.param,
            code: error.code
        )
    )
    // Encoding a struct of String/String? cannot fail.
    let data = try! JSONEncoder().encode(envelope)
    return Response(
        status: .badRequest,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: data)
    )
}

/// Builds the HTTP 429 response for a queue-overload condition. The body is
/// the exact OpenAI-shaped `engine_overloaded` error:
///
/// ```json
/// {
///   "error": {
///     "message": "The server is currently at capacity. Please try again later.",
///     "type": "server_error",
///     "param": null,
///     "code": "engine_overloaded"
///   }
/// }
/// ```
public func openAIOverloadedResponse() -> Response {
    let envelope = OpenAIErrorEnvelope(
        error: .init(
            message: "The server is currently at capacity. Please try again later.",
            type: "server_error",
            param: nil,
            code: "engine_overloaded"
        )
    )
    let data = try! JSONEncoder().encode(envelope)
    return Response(
        status: .tooManyRequests,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: data)
    )
}

/// Builds the HTTP 507 response for a memory-admission failure. The body is
/// the OpenAI-shaped `memory_budget_exceeded` error:
///
/// ```json
/// {
///   "error": {
///     "message": "Estimated KV cache ... exceeds the memory budget ...",
///     "type": "server_error",
///     "param": null,
///     "code": "memory_budget_exceeded"
///   }
/// }
/// ```
public func openAIMemoryErrorResponse(_ error: MemoryAdmissionPolicy.AdmissionFailure) -> Response {
    let envelope = OpenAIErrorEnvelope(
        error: .init(
            message: error.errorDescription ?? "Memory budget exceeded.",
            type: "server_error",
            param: nil,
            code: "memory_budget_exceeded"
        )
    )
    let data = try! JSONEncoder().encode(envelope)
    return Response(
        status: .insufficientStorage,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: data)
    )
}

/// Builds the HTTP 507 response for a transient prefill-buffer admission
/// failure (the quadratic dense scores buffer cannot fit in a single Metal
/// buffer). Same envelope shape as `openAIMemoryErrorResponse`, with the
/// `prefill_buffer_exceeded` code.
public func openAITransientBufferErrorResponse(
    _ error: MemoryAdmissionPolicy.TransientBufferFailure
) -> Response {
    let envelope = OpenAIErrorEnvelope(
        error: .init(
            message: error.errorDescription ?? "Prefill buffer exceeds the allocatable limit.",
            type: "server_error",
            param: nil,
            code: "prefill_buffer_exceeded"
        )
    )
    let data = try! JSONEncoder().encode(envelope)
    return Response(
        status: .insufficientStorage,
        headers: ["Content-Type": "application/json; charset=utf-8"],
        body: .init(data: data)
    )
}

/// Validates a `ChatCompletionRequest` against the server configuration and
/// the capabilities of the current runtime, before any model execution.
///
/// On success, returns the resolved `SamplingParameters` (the single policy
/// source shared with `MLXGenerator`). On failure, throws an
/// `OpenAIRequestError` that the router serializes as an OpenAI-shaped 400.
public enum ChatCompletionRequestValidator {

    /// Roles the prompt formatter actually handles. `tool` carries a tool
    /// result (`tool_call_id` / `name` optional); `function` is the legacy
    /// OpenAI function-calling spelling and is mapped to `tool` before
    /// template rendering.
    public static let supportedRoles: Set<String> = ["system", "assistant", "user", "tool", "function"]

    /// Maximum number of stop sequences a single request may provide.
    public static let maxStopSequences = 16

    /// Maximum length, in characters, of a single stop sequence.
    public static let maxStopSequenceLength = 64

    /// Maximum number of tool definitions a single request may provide.
    public static let maxToolCount = 128

    /// Maximum length, in characters, of a single tool function name.
    public static let maxToolNameLength = 64

    /// Tool function names must match `[a-zA-Z0-9_-]` (1...maxToolNameLength).
    static func isValidToolName(_ name: String) -> Bool {
        (!name.isEmpty)
            && name.count <= maxToolNameLength
            && name.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_" || $0 == "-") }
    }

    static func validate(
        request: ChatCompletionRequest,
        serverConfig: ServerConfig
    ) throws -> SamplingParameters {
        try validateModel(request: request, serverConfig: serverConfig)
        try validateMessages(request: request)
        try validateTools(request: request)
        try validateToolChoice(request: request)
        try validateParallelToolCalls(request: request)
        try validateToolConversation(request: request)
        try validateMaxTokens(request: request, serverConfig: serverConfig)
        try validateContextWindow(request: request, serverConfig: serverConfig)
        // `fromRequest` resolves the KV-cache configuration against the
        // runtime support table; unsupported K/V combinations throw
        // `KVCacheConfigError`, mapped here to an OpenAI-shaped 400.
        let params: SamplingParameters
        do {
            params = try SamplingParameters.fromRequest(
                request,
                serverConfig: serverConfig
            )
        } catch let error as KVCacheConfigError {
            throw OpenAIRequestError(
                message: error.message,
                param: error.param,
                code: "unsupported_parameter"
            )
        }
        try validateSampling(params: params)
        // MTP depth: the public API exposes only `mtp_enabled` (Bool). There is
        // no depth-extension field, so there is no range to validate here;
        // `MLXGenerator` maps `mtp_enabled` to the configured draft depth or
        // to the serial control depth (0).
        try validatePenalties(request: request)
        try validateChatTemplateFields(request: request)
        try validateTTL(request: request)
        try validateN(request: request)
        try validateStopSequences(request: request)
        return params
    }

    // MARK: - Model

    private static func validateModel(
        request: ChatCompletionRequest,
        serverConfig: ServerConfig
    ) throws {
        let model = request.model
        guard model == ServerConfig.canonicalModelID
            || serverConfig.modelAliases.contains(model) else {
            throw OpenAIRequestError(
                message: "Unknown model '\(model)'. The server loads '\(ServerConfig.canonicalModelID)'; additional aliases can be configured with --model-aliases.",
                param: "model",
                code: "invalid_value"
            )
        }
    }

    // MARK: - Messages

    private static func validateMessages(request: ChatCompletionRequest) throws {
        guard !request.messages.isEmpty else {
            throw OpenAIRequestError(
                message: "'messages' must contain at least one message.",
                param: "messages",
                code: "invalid_value"
            )
        }
        for (index, message) in request.messages.enumerated() {
            guard let rawRole = message.role else {
                throw OpenAIRequestError(
                    message: "messages[\(index)] is missing a role.",
                    param: "messages",
                    code: "invalid_value"
                )
            }
            let role = rawRole.lowercased()
            guard supportedRoles.contains(role) else {
                throw OpenAIRequestError(
                    message: "Unsupported message role '\(message.role ?? "<nil>")' at messages[\(index)]. Supported roles: system, assistant, user, tool, function.",
                    param: "messages",
                    code: "invalid_value"
                )
            }
        }
    }

    // MARK: - Tools

    /// Validates the `tools` array: each entry must be `type == "function"`
    /// with a well-formed, unique function name, and an optional `parameters`
    /// that is a JSON object. Size and name policy are enforced so malformed
    /// schemas are rejected before any model execution.
    private static func validateTools(request: ChatCompletionRequest) throws {
        guard let tools = request.tools, !tools.isEmpty else { return }

        if tools.count > maxToolCount {
            throw OpenAIRequestError(
                message: "'tools' may contain at most \(maxToolCount) definitions; got \(tools.count).",
                param: "tools",
                code: "invalid_value"
            )
        }

        var seenNames = Set<String>()
        for (index, tool) in tools.enumerated() {
            guard tool.type == "function" else {
                throw OpenAIRequestError(
                    message: "tools[\(index)].type must be \"function\"; got \"\(tool.type)\".",
                    param: "tools",
                    code: "invalid_value"
                )
            }
            let name = tool.function.name
            guard isValidToolName(name) else {
                throw OpenAIRequestError(
                    message: "tools[\(index)].function.name '\(name)' must match [a-zA-Z0-9_-] (1-\(maxToolNameLength) characters).",
                    param: "tools",
                    code: "invalid_value"
                )
            }
            guard seenNames.insert(name).inserted else {
                throw OpenAIRequestError(
                    message: "Duplicate tool name '\(name)' in 'tools'.",
                    param: "tools",
                    code: "invalid_value"
                )
            }
            if let parameters = tool.function.parameters {
                guard case .object = parameters else {
                    throw OpenAIRequestError(
                        message: "tools[\(index)].function.parameters must be a JSON object.",
                        param: "tools",
                        code: "invalid_value"
                    )
                }
            }
        }
    }

    /// `tool_choice` may only be `auto` (omit) or `none`. The Qwen 3.8 tool
    /// template has no mechanism to force a tool call or a specific named
    /// tool, so `required` and named selection are rejected with a clear 400
    /// rather than silently treated as `auto`.
    private static func validateToolChoice(request: ChatCompletionRequest) throws {
        guard let choice = request.tool_choice else { return }
        switch choice {
        case .auto, .none:
            return
        case .required:
            throw OpenAIRequestError(
                message: "'tool_choice: \"required\"' is not supported by the Qwen 3.8 tool template (it cannot force a tool call). Omit tool_choice (auto) or use 'none'.",
                param: "tool_choice",
                code: "unsupported_parameter"
            )
        case .function(let name):
            throw OpenAIRequestError(
                message: "Named 'tool_choice' (\"\(name)\") is not supported by the Qwen 3.8 tool template (it cannot force a specific tool). Omit tool_choice (auto) or use 'none'.",
                param: "tool_choice",
                code: "unsupported_parameter"
            )
        }
    }

    /// `parallel_tool_calls` is not a server-controlled toggle: the Qwen 3.8
    /// tool template has no parallel-call mode, and whether the model emits
    /// multiple tool calls in one assistant message is up to the model. Only
    /// `false` and omit are accepted; `true` is rejected clearly.
    private static func validateParallelToolCalls(request: ChatCompletionRequest) throws {
        if request.parallel_tool_calls == true {
            throw OpenAIRequestError(
                message: "'parallel_tool_calls: true' is not supported by the Qwen 3.8 tool template (no parallel-call mode toggle). Omit the field or pass false.",
                param: "parallel_tool_calls",
                code: "unsupported_parameter"
            )
        }
    }

    /// Conversation sequencing for tool calls and results:
    /// - every `role: "tool"` (or `"function"`) message must carry a
    ///   non-empty `tool_call_id` that references a prior assistant `tool_calls`
    ///   entry (no orphaned tool results, no cross-turn mismatch);
    /// - assistant `tool_calls` IDs must be unique within and across messages.
    private static func validateToolConversation(request: ChatCompletionRequest) throws {
        var assistantCallIDs = Set<String>()
        for (index, message) in request.messages.enumerated() {
            let rawRole = (message.role ?? "").lowercased()
            let role = rawRole == "function" ? "tool" : rawRole

            if role == "assistant" {
                let callIDs = (message.tool_calls ?? []).compactMap { $0.id }
                if Set(callIDs).count != callIDs.count {
                    throw OpenAIRequestError(
                        message: "messages[\(index)] (assistant) has duplicate tool call IDs.",
                        param: "messages",
                        code: "invalid_value"
                    )
                }
                for id in callIDs {
                    guard assistantCallIDs.insert(id).inserted else {
                        throw OpenAIRequestError(
                            message: "Duplicate tool call ID '\(id)' across messages.",
                            param: "messages",
                            code: "invalid_value"
                        )
                    }
                }
            } else if role == "tool" {
                guard let callID = message.tool_call_id, !callID.isEmpty else {
                    throw OpenAIRequestError(
                        message: "messages[\(index)] (tool) is missing a non-empty tool_call_id.",
                        param: "messages",
                        code: "invalid_value"
                    )
                }
                guard assistantCallIDs.contains(callID) else {
                    throw OpenAIRequestError(
                        message: "messages[\(index)] (tool) references tool_call_id '\(callID)' that does not match a prior assistant tool call (orphaned tool result).",
                        param: "messages",
                        code: "invalid_value"
                    )
                }
            }
        }
    }

    // MARK: - Max tokens

    private static func validateMaxTokens(
        request: ChatCompletionRequest,
        serverConfig: ServerConfig
    ) throws {
        if let maxTokens = request.max_tokens, maxTokens <= 0 {
            throw OpenAIRequestError(
                message: "'max_tokens' must be a positive integer.",
                param: "max_tokens",
                code: "invalid_value"
            )
        }
        if let maxCompletionTokens = request.max_completion_tokens, maxCompletionTokens <= 0 {
            throw OpenAIRequestError(
                message: "'max_completion_tokens' must be a positive integer.",
                param: "max_completion_tokens",
                code: "invalid_value"
            )
        }
        if let maxTokens = request.max_tokens, maxTokens > serverConfig.nPredict {
            throw OpenAIRequestError(
                message: "'max_tokens' (\(maxTokens)) exceeds the server generation limit (\(serverConfig.nPredict)).",
                param: "max_tokens",
                code: "invalid_value"
            )
        }
        if let maxCompletionTokens = request.max_completion_tokens, maxCompletionTokens > serverConfig.nPredict {
            throw OpenAIRequestError(
                message: "'max_completion_tokens' (\(maxCompletionTokens)) exceeds the server generation limit (\(serverConfig.nPredict)).",
                param: "max_completion_tokens",
                code: "invalid_value"
            )
        }
        if let maxTokens = request.max_tokens,
           let maxCompletionTokens = request.max_completion_tokens,
           maxTokens != maxCompletionTokens {
            throw OpenAIRequestError(
                message: "'max_tokens' and 'max_completion_tokens' were both provided and differ (\(maxTokens) vs \(maxCompletionTokens)); they must be equal.",
                param: "max_tokens",
                code: "invalid_value"
            )
        }
    }

    // MARK: - Context window

    private static func validateContextWindow(
        request: ChatCompletionRequest,
        serverConfig: ServerConfig
    ) throws {
        guard let contextWindow = request.context_window else { return }
        if contextWindow <= 0 {
            throw OpenAIRequestError(
                message: "'context_window' must be a positive integer.",
                param: "context_window",
                code: "invalid_value"
            )
        }
        if contextWindow > serverConfig.ctxSize {
            throw OpenAIRequestError(
                message: "'context_window' (\(contextWindow)) exceeds the server context limit (\(serverConfig.ctxSize)).",
                param: "context_window",
                code: "invalid_value"
            )
        }
    }

    // MARK: - Sampling

    /// Validates the resolved sampling values (request value, falling back to
    /// the server default).
    ///
    /// - `temperature`: any finite value; `<= 0` selects greedy decoding,
    ///   `> 0` selects serial target sampling (MTP depth 0). The sampler
    ///   applies no upper bound.
    /// - `top_p`: `[0, 1]`; the nucleus filter is a no-op at `1`.
    /// - `top_k`: `>= 0`; `0` disables top-k.
    /// - `min_p`: `[0, 1]`; `0` disables min-p.
    private static func validateSampling(params: SamplingParameters) throws {
        let temperature = params.temperature
        guard temperature.isFinite else {
            throw OpenAIRequestError(
                message: "'temperature' must be a finite number; NaN and infinity are not supported.",
                param: "temperature",
                code: "invalid_value"
            )
        }

        let topP = params.topP
        guard topP.isFinite, topP >= 0, topP <= 1 else {
            throw OpenAIRequestError(
                message: "'top_p' must be in [0, 1].",
                param: "top_p",
                code: "invalid_value"
            )
        }

        let topK = params.topK
        guard topK >= 0 else {
            throw OpenAIRequestError(
                message: "'top_k' must be >= 0 (0 disables top-k).",
                param: "top_k",
                code: "invalid_value"
            )
        }

        let minP = params.minP
        guard minP.isFinite, minP >= 0, minP <= 1 else {
            throw OpenAIRequestError(
                message: "'min_p' must be in [0, 1].",
                param: "min_p",
                code: "invalid_value"
            )
        }
    }

    // MARK: - Penalties

    /// Repetition, presence, and frequency penalties are applied to target
    /// logits before target token selection, in the serial target-only path
    /// (both greedy and sampled decoding). Supported ranges:
    ///
    /// - `repetition_penalty`: (0, 2]; default 1.0 (no-op). For every token
    ///   present in the request's token history (prompt + committed output
    ///   tokens, bounded by the context window), its logit is divided by the
    ///   penalty when non-negative and multiplied by it when negative
    ///   (llama.cpp convention).
    /// - `presence_penalty`: [-2, 2]; default 0.0 (no-op). Subtracted once
    ///   from the logit of each token present in the history.
    /// - `frequency_penalty`: [-2, 2]; default 0.0 (no-op). Subtracted as
    ///   `frequency_penalty * occurrences` from the logit of each token
    ///   present in the history.
    ///
    /// Out-of-range or non-finite values are rejected (400 `invalid_value`).
    /// When any penalty is non-default, the request is served at the serial
    /// control depth (0) even with `mtp_enabled: true`: the engine's
    /// speculative (draft/verify) path does not apply penalties to draft or
    /// verify logits, so a penalty request must run target-only for the
    /// penalties to take effect. Non-greedy (temperature > 0) requests
    /// without penalties DO use MTP: the engine implements mathematically
    /// exact target-distribution speculative sampling (see
    /// benchmarks/MTP-CORRECTNESS-CONTRACT.md).
    private static func validatePenalties(request: ChatCompletionRequest) throws {
        if let repetitionPenalty = request.repetition_penalty {
            guard repetitionPenalty.isFinite, repetitionPenalty > 0,
                  repetitionPenalty <= 2 else {
                throw OpenAIRequestError(
                    message: "'repetition_penalty' must be in (0, 2].",
                    param: "repetition_penalty",
                    code: "invalid_value"
                )
            }
        }
        if let presencePenalty = request.presence_penalty {
            guard presencePenalty.isFinite, presencePenalty >= -2,
                  presencePenalty <= 2 else {
                throw OpenAIRequestError(
                    message: "'presence_penalty' must be in [-2, 2].",
                    param: "presence_penalty",
                    code: "invalid_value"
                )
            }
        }
        if let frequencyPenalty = request.frequency_penalty {
            guard frequencyPenalty.isFinite, frequencyPenalty >= -2,
                  frequencyPenalty <= 2 else {
                throw OpenAIRequestError(
                    message: "'frequency_penalty' must be in [-2, 2].",
                    param: "frequency_penalty",
                    code: "invalid_value"
                )
            }
        }
    }

    // MARK: - Chat template

    /// `chat_template` is not applied by the current runtime.
    /// `chat_template_kwargs` is applied only for the `enable_thinking` key,
    /// and only with a recognized boolean value; any other key or value is
    /// rejected.
    private static func validateChatTemplateFields(request: ChatCompletionRequest) throws {
        if let chatTemplate = request.chat_template, !chatTemplate.isEmpty {
            throw OpenAIRequestError(
                message: "'chat_template' is not applied by the current runtime and is not supported.",
                param: "chat_template",
                code: "unsupported_parameter"
            )
        }
        if let kwargs = request.chat_template_kwargs {
            for (key, value) in kwargs {
                if key != "enable_thinking" {
                    throw OpenAIRequestError(
                        message: "chat_template_kwargs key '\(key)' is not applied by the current runtime and is not supported. Only 'enable_thinking' is supported.",
                        param: "chat_template_kwargs",
                        code: "unsupported_parameter"
                    )
                }
                let lower = value.lowercased()
                let recognized = ["true", "1", "yes", "false", "0", "no"]
                if !recognized.contains(lower) {
                    throw OpenAIRequestError(
                        message: "chat_template_kwargs['enable_thinking'] must be one of: true, 1, yes, false, 0, no.",
                        param: "chat_template_kwargs",
                        code: "invalid_value"
                    )
                }
            }
        }
    }

    // MARK: - n

    /// This server supports exactly one completion.
    private static func validateN(request: ChatCompletionRequest) throws {
        if let n = request.n, n != 1 {
            throw OpenAIRequestError(
                message: "'n' must be 1; this server supports exactly one completion.",
                param: "n",
                code: "invalid_value"
            )
        }
    }

    // MARK: - Stop sequences

    /// `stop` may be a single string or an array of strings. Empty sets,
    /// empty-string entries, oversized sets, and overlong sequences are
    /// rejected before any model execution.
    private static func validateStopSequences(request: ChatCompletionRequest) throws {
        guard let stop = request.stop else { return }
        let sequences = stop.sequences

        if sequences.isEmpty {
            throw OpenAIRequestError(
                message: "'stop' must contain at least one stop sequence.",
                param: "stop",
                code: "invalid_value"
            )
        }

        if sequences.count > maxStopSequences {
            throw OpenAIRequestError(
                message: "'stop' may contain at most \(maxStopSequences) sequences; got \(sequences.count).",
                param: "stop",
                code: "invalid_value"
            )
        }

        for (index, sequence) in sequences.enumerated() {
            if sequence.isEmpty {
                throw OpenAIRequestError(
                    message: "stop[\(index)] must not be an empty string.",
                    param: "stop",
                    code: "invalid_value"
                )
            }
            if sequence.count > maxStopSequenceLength {
                throw OpenAIRequestError(
                    message: "stop[\(index)] must be at most \(maxStopSequenceLength) characters; got \(sequence.count).",
                    param: "stop",
                    code: "invalid_value"
                )
            }
        }
    }

    // MARK: - TTL

    /// `ttl_seconds` is not applied by the current runtime: sessions are
    /// created per request and released when the request completes; there is
    /// no session TTL.
    private static func validateTTL(request: ChatCompletionRequest) throws {
        if request.ttl_seconds != nil {
            throw OpenAIRequestError(
                message: "'ttl_seconds' is not applied by the current runtime and is not supported.",
                param: "ttl_seconds",
                code: "unsupported_parameter"
            )
        }
    }
}