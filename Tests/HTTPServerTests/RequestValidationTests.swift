// RequestValidationTests.swift
//
// Unit tests for the request-validation layer of the Qwen 3.8 MTP server.
//
// The validator is pure Swift (no MLX, no model weights), so these tests run
// on any machine without loading the model. They pin the OpenAI-shaped error
// behavior: every rejection is a 400 with
// {"error": {"message", "type", "param", "code"}}.

import Testing
import Foundation
@testable import HTTPServer

/// Builds a `ChatMessage` with the output-only fields left nil.
private func message(
    role: String,
    content: String? = nil,
    reasoning: String? = nil,
    reasoningContent: String? = nil
) -> ChatMessage {
    ChatMessage(
        role: role,
        content: content,
        reasoning: reasoning,
        reasoning_content: reasoningContent
    )
}

/// A minimal valid request.
private func validRequest(
    model: String = ServerConfig.canonicalModelID,
    messages: [ChatMessage] = [message(role: "user", content: "Hello")]
) -> ChatCompletionRequest {
    ChatCompletionRequest(
        model: model,
        messages: messages
    )
}

/// Runs the validator and inspects the thrown `OpenAIRequestError`, failing
/// the test if validation succeeds or throws a different error.
private func expectValidationError(
    _ request: ChatCompletionRequest,
    serverConfig: ServerConfig = ServerConfig(),
    _ check: (OpenAIRequestError) -> Void
) {
    do {
        _ = try ChatCompletionRequestValidator.validate(
            request: request,
            serverConfig: serverConfig
        )
        Issue.record("expected validation to fail")
    } catch let error as OpenAIRequestError {
        check(error)
    } catch {
        Issue.record("unexpected error type: \(error)")
    }
}

// MARK: - Happy path

@Test
func validRequestResolvesToSamplingParameters() throws {
    let config = ServerConfig()
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(),
        serverConfig: config
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
    #expect(params.contextWindow == config.ctxSize)
    #expect(params.temperature == config.temp)
    #expect(params.topP == config.topP)
    #expect(params.topK == config.topK)
    #expect(params.minP == config.minP)
    #expect(params.mtpEnabled == (config.specDraftNMax > 0))
    #expect(params.enableThinking == true)
}

@Test
func configuredModelAliasIsAccepted() throws {
    var config = ServerConfig()
    config.modelAliases = ["qwen-27b", "qwen3.8"]
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(model: "qwen-27b"),
        serverConfig: config
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
}

// MARK: - Model

@Test
func unknownModelIsRejected() {
    expectValidationError(validRequest(model: "gpt-4")) { error in
        #expect(error.param == "model")
        #expect(error.code == "invalid_value")
        #expect(error.type == "invalid_request_error")
    }
}

// MARK: - Messages

@Test
func emptyMessagesAreRejected() {
    expectValidationError(validRequest(messages: [])) { error in
        #expect(error.param == "messages")
        #expect(error.code == "invalid_value")
    }
}

@Test
func unsupportedMessageRoleIsRejected() {
    expectValidationError(
        validRequest(messages: [message(role: "unknown", content: "x")])
    ) { error in
        #expect(error.param == "messages")
        #expect(error.code == "invalid_value")
    }
}

@Test
func toolRoleIsAccepted() throws {
    // `role: "tool"` carries a tool result; `tool_call_id` and `name` are
    // optional and accepted.
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(
            messages: [
                message(role: "user", content: "What is the weather?"),
                ChatMessage(
                    role: "tool",
                    content: "Sunny, 25C",
                    tool_call_id: "call_123",
                    name: "get_weather"
                )
            ]
        ),
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
}

@Test
func functionRoleIsAccepted() throws {
    // The legacy `role: "function"` spelling is accepted by validation and
    // mapped to `tool` before template rendering.
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(
            messages: [
                message(role: "user", content: "What is the weather?"),
                ChatMessage(
                    role: "function",
                    content: "Sunny, 25C",
                    tool_call_id: "call_123",
                    name: "get_weather"
                )
            ]
        ),
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
}

@Test
func toolMessageDecodesWithToolCallIdAndName() throws {
    let json = """
    {
      "model": "\(ServerConfig.canonicalModelID)",
      "messages": [
        {"role": "user", "content": "What is the weather?"},
        {"role": "tool", "content": "Sunny, 25C", "tool_call_id": "call_123", "name": "get_weather"}
      ]
    }
    """
    let request = try JSONDecoder().decode(
        ChatCompletionRequest.self,
        from: Data(json.utf8)
    )
    let toolMessage = request.messages[1]
    #expect(toolMessage.role == "tool")
    #expect(toolMessage.content == "Sunny, 25C")
    #expect(toolMessage.tool_call_id == "call_123")
    #expect(toolMessage.name == "get_weather")

    // The validator accepts the decoded request.
    _ = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
}

@Test
func reasoningFieldsOnInputMessagesAreAccepted() throws {
    // `reasoning` / `reasoning_content` on input messages are accepted and
    // safely ignored or passed through to the chat template builder; they no
    // longer trigger a 400 rejection.
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(
            messages: [
                message(role: "assistant", content: "x", reasoning: "r"),
                message(role: "user", content: "y", reasoningContent: "rc")
            ]
        ),
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
}

// MARK: - Token limits

@Test
func nonPositiveMaxTokensAreRejected() {
    var request = validRequest()
    request.max_tokens = 0
    expectValidationError(request) { error in
        #expect(error.param == "max_tokens")
        #expect(error.code == "invalid_value")
    }
}

@Test
func maxTokensAboveServerLimitAreRejected() {
    var config = ServerConfig()
    config.nPredict = 100
    var request = validRequest()
    request.max_tokens = 101
    expectValidationError(request, serverConfig: config) { error in
        #expect(error.param == "max_tokens")
        #expect(error.code == "invalid_value")
    }
}

@Test
func conflictingMaxTokensFieldsAreRejected() {
    var request = validRequest()
    request.max_tokens = 10
    request.max_completion_tokens = 20
    expectValidationError(request) { error in
        #expect(error.param == "max_tokens")
        #expect(error.code == "invalid_value")
    }
}

@Test
func maxTokensTakesPriorityOverMaxCompletionTokens() throws {
    // When both are provided and equal, `max_tokens` is the effective value.
    var request = validRequest()
    request.max_tokens = 512
    request.max_completion_tokens = 512
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == 512)
}

@Test
func maxCompletionTokensUsedWhenMaxTokensAbsent() throws {
    // When only `max_completion_tokens` is provided, it is the effective value.
    var request = validRequest()
    request.max_completion_tokens = 256
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == 256)
}

@Test
func defaultMaxTokensAppliedWhenNeitherProvided() throws {
    // When neither `max_tokens` nor `max_completion_tokens` is provided,
    // the effective max is the safe default (4096), not nPredict.
    let params = try ChatCompletionRequestValidator.validate(
        request: validRequest(),
        serverConfig: ServerConfig()
    )
    #expect(params.maxTokens == SamplingParameters.defaultMaxTokens)
    #expect(params.maxTokens == 4_096)
}

@Test
func contextWindowAboveServerLimitIsRejected() {
    var config = ServerConfig()
    config.ctxSize = 1000
    var request = validRequest()
    request.context_window = 1001
    expectValidationError(request, serverConfig: config) { error in
        #expect(error.param == "context_window")
        #expect(error.code == "invalid_value")
    }
}

@Test
func nonPositiveContextWindowIsRejected() {
    var request = validRequest()
    request.context_window = 0
    expectValidationError(request) { error in
        #expect(error.param == "context_window")
        #expect(error.code == "invalid_value")
    }
}

// MARK: - Sampling

@Test
func nanTemperatureIsRejected() {
    var request = validRequest()
    request.temperature = .nan
    expectValidationError(request) { error in
        #expect(error.param == "temperature")
        #expect(error.code == "invalid_value")
    }
}

@Test
func outOfRangeTopPIsRejected() {
    var request = validRequest()
    request.top_p = 1.5
    expectValidationError(request) { error in
        #expect(error.param == "top_p")
        #expect(error.code == "invalid_value")
    }
}

@Test
func negativeTopKIsRejected() {
    var request = validRequest()
    request.top_k = -1
    expectValidationError(request) { error in
        #expect(error.param == "top_k")
        #expect(error.code == "invalid_value")
    }
}

@Test
func outOfRangeMinPIsRejected() {
    var request = validRequest()
    request.min_p = 1.5
    expectValidationError(request) { error in
        #expect(error.param == "min_p")
        #expect(error.code == "invalid_value")
    }
}

@Test
func greedyAndSerialSamplingValuesAreAccepted() throws {
    // temperature <= 0 is the greedy path; temperature > 0 selects serial
    // target sampling (MTP depth 0). Both are applied by the runtime.
    var greedy = validRequest()
    greedy.temperature = 0.0
    _ = try ChatCompletionRequestValidator.validate(
        request: greedy,
        serverConfig: ServerConfig()
    )

    var sampled = validRequest()
    sampled.temperature = 0.7
    sampled.top_p = 0.9
    sampled.top_k = 40
    sampled.min_p = 0.1
    let params = try ChatCompletionRequestValidator.validate(
        request: sampled,
        serverConfig: ServerConfig()
    )
    #expect(params.temperature == 0.7)
    #expect(params.topP == 0.9)
    #expect(params.topK == 40)
    #expect(params.minP == 0.1)
}

// MARK: - KV cache

@Test
func kvCacheBitsAreRejected() {
    var request = validRequest()
    request.kv_cache_bits = 8
    expectValidationError(request) { error in
        #expect(error.param == "kv_cache_bits")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func nonDefaultKVGroupSizeIsAccepted() throws {
    var request = validRequest()
    request.kv_cache_group_size = 32
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.kvCacheConfig.groupSize == 32)
}

@Test
func nonDefaultQuantizedKVStartIsAccepted() throws {
    var request = validRequest()
    request.quantized_kv_start = 16
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.kvCacheConfig.quantizedKVStart == 16)
}

// MARK: - Penalties

@Test
func nonDefaultRepetitionPenaltyIsAccepted() throws {
    var request = validRequest()
    request.repetition_penalty = 1.2
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.repetitionPenalty == 1.2)
}

@Test
func nonDefaultPresencePenaltyIsAccepted() throws {
    var request = validRequest()
    request.presence_penalty = 0.5
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.presencePenalty == 0.5)
}

@Test
func nonDefaultFrequencyPenaltyIsAccepted() throws {
    var request = validRequest()
    request.frequency_penalty = -0.5
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.frequencyPenalty == -0.5)
}

@Test
func zeroRepetitionPenaltyIsRejected() {
    var request = validRequest()
    request.repetition_penalty = 0.0
    expectValidationError(request) { error in
        #expect(error.param == "repetition_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func repetitionPenaltyAboveTwoIsRejected() {
    var request = validRequest()
    request.repetition_penalty = 2.5
    expectValidationError(request) { error in
        #expect(error.param == "repetition_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func presencePenaltyAboveTwoIsRejected() {
    var request = validRequest()
    request.presence_penalty = 2.5
    expectValidationError(request) { error in
        #expect(error.param == "presence_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func presencePenaltyBelowNegativeTwoIsRejected() {
    var request = validRequest()
    request.presence_penalty = -2.5
    expectValidationError(request) { error in
        #expect(error.param == "presence_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func frequencyPenaltyAboveTwoIsRejected() {
    var request = validRequest()
    request.frequency_penalty = 2.5
    expectValidationError(request) { error in
        #expect(error.param == "frequency_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func frequencyPenaltyBelowNegativeTwoIsRejected() {
    var request = validRequest()
    request.frequency_penalty = -2.5
    expectValidationError(request) { error in
        #expect(error.param == "frequency_penalty")
        #expect(error.code == "invalid_value")
    }
}

@Test
func defaultPenaltyValuesAreAccepted() throws {
    var request = validRequest()
    request.repetition_penalty = 1.0
    request.presence_penalty = 0.0
    request.frequency_penalty = 0.0
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.repetitionPenalty == 1.0)
    #expect(params.presencePenalty == 0.0)
    #expect(params.frequencyPenalty == 0.0)
    #expect(!params.hasNonDefaultPenalties)
}

// MARK: - Chat template / TTL

@Test
func ttlSecondsAreRejected() {
    var request = validRequest()
    request.ttl_seconds = 60
    expectValidationError(request) { error in
        #expect(error.param == "ttl_seconds")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func chatTemplateIsRejected() {
    var request = validRequest()
    request.chat_template = "custom"
    expectValidationError(request) { error in
        #expect(error.param == "chat_template")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func unsupportedChatTemplateKwargsAreRejected() {
    var request = validRequest()
    request.chat_template_kwargs = ["foo": "bar"]
    expectValidationError(request) { error in
        #expect(error.param == "chat_template_kwargs")
        #expect(error.code == "unsupported_parameter")
    }
}

@Test
func unrecognizedEnableThinkingKwargsAreRejected() {
    var request = validRequest()
    request.chat_template_kwargs = ["enable_thinking": "maybe"]
    expectValidationError(request) { error in
        #expect(error.param == "chat_template_kwargs")
        #expect(error.code == "invalid_value")
    }
}

@Test
func enableThinkingKwargsAreAccepted() throws {
    var request = validRequest()
    request.chat_template_kwargs = ["enable_thinking": "false"]
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.enableThinking == false)
}

// MARK: - n

@Test
func nGreaterThanOneIsRejected() {
    var request = validRequest()
    request.n = 2
    expectValidationError(request) { error in
        #expect(error.param == "n")
        #expect(error.code == "invalid_value")
    }
}

@Test
func nEqualToOneIsAccepted() throws {
    var request = validRequest()
    request.n = 1
    _ = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
}

// MARK: - Stop sequences

@Test
func validSingleStopSequenceIsAccepted() throws {
    var request = validRequest()
    request.stop = StopSequences(sequences: ["\n"])
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.stopSequences == ["\n"])
}

@Test
func validStopSequenceArrayIsAccepted() throws {
    var request = validRequest()
    request.stop = StopSequences(sequences: ["STOP", "END"])
    let params = try ChatCompletionRequestValidator.validate(
        request: request,
        serverConfig: ServerConfig()
    )
    #expect(params.stopSequences == ["STOP", "END"])
}

@Test
func emptyStopArrayIsRejected() {
    var request = validRequest()
    request.stop = StopSequences(sequences: [])
    expectValidationError(request) { error in
        #expect(error.param == "stop")
        #expect(error.code == "invalid_value")
    }
}

@Test
func emptyStopStringEntryIsRejected() {
    var request = validRequest()
    request.stop = StopSequences(sequences: ["ok", ""])
    expectValidationError(request) { error in
        #expect(error.param == "stop")
        #expect(error.code == "invalid_value")
    }
}

@Test
func tooManyStopSequencesAreRejected() {
    var request = validRequest()
    request.stop = StopSequences(
        sequences: (0..<ChatCompletionRequestValidator.maxStopSequences + 1).map { "s\($0)" }
    )
    expectValidationError(request) { error in
        #expect(error.param == "stop")
        #expect(error.code == "invalid_value")
    }
}

@Test
func overlongStopSequenceIsRejected() {
    var request = validRequest()
    request.stop = StopSequences(
        sequences: [String(repeating: "a", count: ChatCompletionRequestValidator.maxStopSequenceLength + 1)]
    )
    expectValidationError(request) { error in
        #expect(error.param == "stop")
        #expect(error.code == "invalid_value")
    }
}

// MARK: - Error envelope

@Test
func errorEnvelopeOmitsAbsentParamAndIsValidJSON() throws {
    let error = OpenAIRequestError(
        message: "Request body is not valid JSON.",
        param: nil,
        code: "invalid_value"
    )
    let response = openAIErrorResponse(error)
    #expect(response.status == .badRequest)
    #expect(
        response.headers["Content-Type"].contains { $0.hasPrefix("application/json") }
    )
    guard let body = response.body.string else {
        Issue.record("expected a string response body")
        return
    }
    let decoded = try JSONDecoder().decode(OpenAIErrorEnvelope.self, from: Data(body.utf8))
    #expect(decoded.error.message == "Request body is not valid JSON.")
    #expect(decoded.error.type == "invalid_request_error")
    #expect(decoded.error.code == "invalid_value")
    #expect(decoded.error.param == nil)
    // `param` must be omitted, not encoded as null.
    #expect(!body.contains("\"param\""))
}