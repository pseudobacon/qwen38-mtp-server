import Foundation
import Logging
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

/// Typed bridge between model generation and OpenAI SSE serialization.
enum GenerationFragment: Sendable {
    case content(String)
    case reasoning(String)

    /// A single parsed tool call (function-calling). Carries the OpenAI
    /// `tool_calls` element: id, index, and the function name + arguments
    /// (arguments as a JSON string). Never merged by the coalescer.
    case toolCall(ToolCall)

    case finished(
        reason: String,
        promptTokens: Int,
        completionTokens: Int
    )

    /// Generator-side telemetry for one generation. Never written to SSE;
    /// consumed only by the router's metrics collection.
    case metrics(GenerationMetrics)
}

extension GenerationFragment {
    /// The text payload of the fragment, for content and reasoning fragments.
    /// Returns an empty string for non-text fragments.
    var text: String {
        switch self {
        case .content(let text): return text
        case .reasoning(let text): return text
        case .toolCall: return ""
        case .finished: return ""
        case .metrics: return ""
        }
    }

    /// Returns a copy of the fragment carrying the given text payload,
    /// preserving the fragment kind. Non-text fragments are returned as-is.
    func withText(_ newText: String) -> GenerationFragment {
        switch self {
        case .content: return .content(newText)
        case .reasoning: return .reasoning(newText)
        case .toolCall: return self
        case .finished: return self
        case .metrics: return self
        }
    }
}

/// Thrown when a matching-ID cancellation is observed at a safe boundary.
struct GenerationCancelledError: Error {
    let id: GenerationID
}

/// Incrementally separates Qwen `<think>...</think>` output from ordinary
/// visible content while safely handling tags split across token boundaries.
internal struct SimpleStreamingReasoningParser: Sendable {
    private var buffer = ""
    private var insideReasoning = false

    init(enableThinking: Bool = false) {
        self.insideReasoning = enableThinking
    }

    mutating func parse(_ text: String) -> [GenerationFragment] {
        buffer += text

        var fragments: [GenerationFragment] = []

        while !buffer.isEmpty {
            if insideReasoning {
                if let endRange = buffer.range(of: "</" + "think>") {
                    let reasoning = String(buffer[..<endRange.lowerBound])

                    if !reasoning.isEmpty {
                        fragments.append(.reasoning(reasoning))
                    }

                    buffer.removeSubrange(..<endRange.upperBound)
                    insideReasoning = false
                    continue
                }

                let retainedSuffixLength = 16
                if buffer.count > retainedSuffixLength {
                    let splitIndex = buffer.index(
                        buffer.endIndex,
                        offsetBy: -retainedSuffixLength
                    )
                    let reasoning = String(buffer[..<splitIndex])
                    buffer = String(buffer[splitIndex...])

                    if !reasoning.isEmpty {
                        fragments.append(.reasoning(reasoning))
                    }
                }

                break
            }

            if let startRange = buffer.range(of: "<" + "think>") {
                let content = String(buffer[..<startRange.lowerBound])

                if !content.isEmpty {
                    fragments.append(.content(content))
                }

                buffer.removeSubrange(..<startRange.upperBound)
                insideReasoning = true
                continue
            }

            let retainedSuffixLength = 6
            let emitCount = buffer.count - retainedSuffixLength

            if emitCount > 0 {
                let splitIndex = buffer.index(
                    buffer.startIndex,
                    offsetBy: emitCount
                )
                let content = String(buffer[..<splitIndex])
                buffer = String(buffer[splitIndex...])

                if !content.isEmpty {
                    fragments.append(.content(content))
                }
            }

            break
        }

        return fragments
    }

    mutating func finish() -> [GenerationFragment] {
        guard !buffer.isEmpty else { return [] }
        let fragment: GenerationFragment = insideReasoning ? .reasoning(buffer) : .content(buffer)
        buffer = ""
        return [fragment]
    }
}

/// Merges consecutive same-channel fragments into single fragments.
internal func coalesceFragments(_ fragments: [GenerationFragment]) -> [GenerationFragment] {
    var result: [GenerationFragment] = []

    for fragment in fragments {
        if let last = result.last {
            switch (last, fragment) {
            case (.content(let a), .content(let b)):
                result[result.count - 1] = .content(a + b)
                continue
            case (.reasoning(let a), .reasoning(let b)):
                result[result.count - 1] = .reasoning(a + b)
                continue
            default:
                break
            }
        }
        result.append(fragment)
    }

    return result
}

/// Server-side sampling surface for one chat-completions request.
struct SamplingParameters: Sendable {
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let repetitionPenalty: Float
    let presencePenalty: Float
    let frequencyPenalty: Float

    let maxTokens: Int
    let contextWindow: Int

    let enableThinking: Bool
    let mtpEnabled: Bool
    let prefillChunkSize: Int

    let stopSequences: [String]
    let kvCacheConfig: ResolvedKVCacheConfig
    let ttlSeconds: Int?
    let toolCallsEnabled: Bool

    init(
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float,
        presencePenalty: Float,
        frequencyPenalty: Float,
        maxTokens: Int,
        contextWindow: Int,
        enableThinking: Bool,
        mtpEnabled: Bool,
        prefillChunkSize: Int,
        stopSequences: [String],
        kvCacheConfig: ResolvedKVCacheConfig,
        ttlSeconds: Int?,
        toolCallsEnabled: Bool = false
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.maxTokens = maxTokens
        self.contextWindow = contextWindow
        self.enableThinking = enableThinking
        self.mtpEnabled = mtpEnabled
        self.prefillChunkSize = prefillChunkSize
        self.stopSequences = stopSequences
        self.kvCacheConfig = kvCacheConfig
        self.ttlSeconds = ttlSeconds
        self.toolCallsEnabled = toolCallsEnabled
    }

    static let defaultContextWindow = 262_144
    static let defaultMaxTokens = 4_096

    static func fromRequest(
        _ request: ChatCompletionRequest,
        serverConfig: ServerConfig
    ) throws -> SamplingParameters {
        let kwargs = request.chat_template_kwargs ?? [:]

        let kwargsEnableThinking: Bool? = {
            guard let raw = kwargs["enable_thinking"]?.lowercased() else { return nil }
            if raw == "true" || raw == "1" || raw == "yes" { return true }
            if raw == "false" || raw == "0" || raw == "no" { return false }
            return nil
        }()

        let kvCacheConfig: ResolvedKVCacheConfig
        if let bits = request.kv_cache_bits {
            // Per-request override: resolve via the llama.cpp-style K/V path.
            do {
                kvCacheConfig = try ResolvedKVCacheConfig.resolve(
                    kType: String(bits),
                    vType: String(bits),
                    groupSize: request.kv_cache_group_size ?? 64,
                    quantizedKVStart: request.quantized_kv_start ?? 0
                )
            } catch let error as KVCacheConfigError where error.param == "cache_type_k" || error.param == "cache_type_v" {
                throw KVCacheConfigError(
                    message: "Unsupported kv_cache_bits value \(bits). The Qwen 3.8 runtime supports f16, q8, q4, and q2.",
                    param: "kv_cache_bits"
                )
            }
        } else {
            // Launch-time scheme: resolve via the unified scheme path.
            kvCacheConfig = ResolvedKVCacheConfig.resolve(
                scheme: serverConfig.kvScheme,
                groupSize: request.kv_cache_group_size ?? serverConfig.kvGroupSize,
                quantizedKVStart: request.quantized_kv_start ?? 0
            )
        }

        let toolCallsEnabled = serverConfig.toolsEnabled
            && !(request.tools?.isEmpty ?? true)
            && !(request.tool_choice?.isNone ?? false)

        return SamplingParameters(
            temperature: request.temperature ?? serverConfig.temp,
            topP: request.top_p ?? serverConfig.topP,
            topK: request.top_k ?? serverConfig.topK,
            minP: request.min_p ?? serverConfig.minP,
            repetitionPenalty: request.repetition_penalty ?? serverConfig.repeatPenalty,
            presencePenalty: request.presence_penalty ?? serverConfig.presencePenalty,
            frequencyPenalty: request.frequency_penalty ?? serverConfig.frequencyPenalty,
            maxTokens: request.max_tokens
                ?? request.max_completion_tokens
                ?? SamplingParameters.defaultMaxTokens,
            contextWindow: request.context_window ?? serverConfig.ctxSize,
            // `enable_thinking` defaults to `true` (Qwen3 reasoning-model
            // behavior: the response opens with thinking tokens unless the
            // caller explicitly disables it via the field or
            // `chat_template_kwargs`). This is context, not a sampling bug,
            // but it measurably lowers MTP draft acceptance on short prompts
            // (~55% thinking-on vs ~74% thinking-off at depth 3, T=0), so the
            // `mean_mtp_acceptance_rate` metric is prompt/context-dependent
            // and must be read against the request's thinking mode.
            enableThinking: request.enable_thinking ?? kwargsEnableThinking ?? true,
            mtpEnabled: request.mtp_enabled ?? (serverConfig.specDraftNMax > 0),
            prefillChunkSize: serverConfig.prefillChunkSize,
            stopSequences: request.stop?.sequences ?? [],
            kvCacheConfig: kvCacheConfig,
            ttlSeconds: request.ttl_seconds,
            toolCallsEnabled: toolCallsEnabled
        )
    }

    var isGreedy: Bool { temperature <= 0.0 }

    var hasNonDefaultPenalties: Bool {
        repetitionPenalty != 1.0 || presencePenalty != 0.0 || frequencyPenalty != 0.0
    }
}

private struct InferenceContext: @unchecked Sendable {
    let model: any Qwen38MTPTarget
    let tokenizer: any MLXLMCommon.Tokenizer
}

/// Background-execution actor for the Qwen 3.8 MTP server.
actor MLXGenerator {
    let modelPath: String
    let mtpHeadPath: String
    let maxDraftDepth: Int

    private let model: any Qwen38MTPTarget
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let stopTokens: Set<Int>

    let memoryAdmissionPolicy: MemoryAdmissionPolicy
    let memoryRecoveryPolicy: MemoryRecoveryPolicy
    private var memoryRecoveryCount = 0

    private let cancellationRegistry = GenerationCancellationRegistry()

    /// Actor-isolated radix-tree store of completed-session KV state for
    /// multi-session and branched prefix reuse.
    private let kvCacheManager = RadixKVCacheManager()

    private var tokenizationCache: TokenizationCache
    private let cacheNamespace: String
    private let logger: Logger

    init(
        modelPath: String = "./weights",
        mtpHeadPath: String = "./mtp-head",
        maxDraftDepth: Int = MLXFastConstants.qwenMTPMaxDepth,
        runtimeState: ModelRuntimeState? = nil,
        memoryLimitBytes: Int = 32 * 1024 * 1024 * 1024,
        systemSafetyReserveBytes: Int = 2 * 1024 * 1024 * 1024,
        memoryRecoveryEnabled: Bool = true,
        memoryPressureThreshold: Double = 0.9,
        kvScheme: KVCacheScheme = .fp16,
        kvTailSize: Int = 1024,
        tokenizationCacheMaxEntries: Int = 1024,
        tokenizationCacheMaxBytes: Int = 256 * 1024 * 1024,
        tokenizationCacheTTLSeconds: Int = 300,
        logger: Logger = Logger(label: "HTTPServer.MLXGenerator")
    ) async throws {
        self.modelPath = modelPath
        self.mtpHeadPath = mtpHeadPath
        self.maxDraftDepth = maxDraftDepth
        self.logger = logger

        self.cacheNamespace = Self.cacheNamespace(modelPath: modelPath, mtpHeadPath: mtpHeadPath)
        self.tokenizationCache = TokenizationCache(
            maxEntries: tokenizationCacheMaxEntries,
            maxBytes: tokenizationCacheMaxBytes,
            defaultTTL: TimeInterval(tokenizationCacheTTLSeconds)
        )

        let targetURL = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()

        // W4: draft-head quantization selection. MLX_QWEN_MTP_HEAD_QUANT:
        //   1/true/on — force the 4-bit sibling tree (<headPath>/q4),
        //   4-bit group-64 affine, produced by benchmarks/make_q4_head.py;
        //   fails loudly if the tree is missing (explicit operator intent).
        //   0/false/off — force the pinned BF16 tree (rollback state).
        //   unset — default ON since the W4 verdict (2026-09-14, essay +8.6 %
        //   / specdec +6.1 %, 24/24 reps bit-exact). The 4-bit tree is
        //   REQUIRED: a missing tree is a loud startup failure, never a
        //   silent BF16 fallback (post-W4 hardening — a headline measured on
        //   the fallback would be the stale-binary error class). A fresh
        //   checkout runs benchmarks/make_q4_head.py once to obtain it.
        // The target-verify path keeps the committed stream bit-identical
        // across both head states on the benchmark fixtures (W4 A/B matrix);
        // the knob changes draft quality (acceptance) and per-round head
        // weight traffic.
        let headBase = URL(fileURLWithPath: mtpHeadPath).resolvingSymlinksInPath()
        let q4URL = headBase.appendingPathComponent("q4")
        let q4Present = FileManager.default.fileExists(
            atPath: q4URL.appendingPathComponent("model.safetensors").path)
        let headURL: URL
        let headVariant: String
        switch ProcessInfo.processInfo.environment["MLX_QWEN_MTP_HEAD_QUANT"] {
        case .some(let v) where ["1", "true", "on"].contains(v.lowercased()):
            guard q4Present else {
                throw MLXFastError.invalidInput(
                    "MLX_QWEN_MTP_HEAD_QUANT=1 but the quantized head tree is missing: "
                    + q4URL.appendingPathComponent("model.safetensors").path
                    + " — generate it with benchmarks/make_q4_head.py")
            }
            headURL = q4URL
            headVariant = "4-bit quantized (MLX_QWEN_MTP_HEAD_QUANT=1)"
        case .some(let v) where ["0", "false", "off"].contains(v.lowercased()):
            headURL = headBase
            headVariant = "BF16 pinned (MLX_QWEN_MTP_HEAD_QUANT=0)"
        case .some(let v):
            throw MLXFastError.invalidInput(
                "MLX_QWEN_MTP_HEAD_QUANT must be 1/true/on or 0/false/off, got \"\(v)\"")
        case .none:
            guard q4Present else {
                throw MLXFastError.invalidInput(
                    "MTP head default is the 4-bit tree but it is missing: "
                    + q4URL.appendingPathComponent("model.safetensors").path
                    + " — generate it with benchmarks/make_q4_head.py (or set "
                    + "MLX_QWEN_MTP_HEAD_QUANT=0 for an explicit BF16 rollback)")
            }
            headURL = q4URL
            headVariant = "4-bit quantized (default ON)"
        }
        print("MLXLM: MTP head selected: \(headURL.path) — \(headVariant)")
        // Load-time engagement proof for the draft depth (post-W4 queue,
        // 2026-09-14): the session's default policy pins k = 2
        // (Qwen38MTPBlockSession.defaultDraftDepth); QWEN_MTP_DRAFT_K
        // overrides. Effective k = min(offer cap, forced ?? default).
        let forcedDraftK = ProcessInfo.processInfo.environment["QWEN_MTP_DRAFT_K"]
            .flatMap { Int($0) }
        let effectiveDraftK = Swift.min(
            maxDraftDepth, forcedDraftK ?? Qwen38MTPBlockSession.defaultDraftDepth)
        print("MLXLM: MTP draft depth: k=\(effectiveDraftK)"
            + (forcedDraftK.map { " (forced via QWEN_MTP_DRAFT_K=\($0))" }
               ?? " (default \(Qwen38MTPBlockSession.defaultDraftDepth); offer cap "
                  + "\(maxDraftDepth); override QWEN_MTP_DRAFT_K)"))

        let (loadedModel, loadedTokenizer) = try Qwen38MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL,
            headDirectory: headURL
        ) { _ in
            let context = try waitForMLXAsync {
                try await LLMModelFactory.shared.load(
                    from: targetURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
            return (context.model, context.tokenizer)
        }

        guard let model = loadedModel as? any Qwen38MTPTarget else {
            throw MLXFastError.invalidInput("Loaded model is not an MTP-capable Qwen model")
        }

        guard model.hasMTPHead else {
            throw MLXFastError.invalidInput("MTP head failed to attach to backbone")
        }

        eval(model)

        let modelBaselineBytes = Memory.activeMemory
        self.memoryAdmissionPolicy = MemoryAdmissionPolicy(
            memoryLimitBytes: memoryLimitBytes,
            systemSafetyReserveBytes: systemSafetyReserveBytes,
            modelBaselineBytes: modelBaselineBytes,
            keyBits: kvScheme.keyBits,
            valueBits: kvScheme.valueBits,
            tailSize: kvTailSize
        )
        self.memoryRecoveryPolicy = MemoryRecoveryPolicy(
            enabled: memoryRecoveryEnabled,
            pressureThresholdFraction: memoryPressureThreshold
        )

        self.model = model
        self.tokenizer = loadedTokenizer
        self.stopTokens = resolveStopTokens(directory: targetURL, tokenizer: loadedTokenizer, logger: logger)

        _ = await runtimeState?.transition(to: .warming)

        let warmup = try Qwen38MTPBlockSession(model: model, stopTokens: stopTokens)
        let warmDepth = min(max(1, maxDraftDepth), MLXFastConstants.qwenMTPMaxDepth)
        try warmup.warmAllDepths(maxDepth: warmDepth)
    }

    func cancelGeneration(id: GenerationID, reason: GenerationCancellationReason) async {
        await cancellationRegistry.cancel(id, reason: reason)
    }

    func recordMemoryRecovery() {
        memoryRecoveryCount += 1
    }

    private func toSendableValue(_ value: Any) -> (any Sendable)? {
        if let string = value as? String { return string }
        if let number = value as? NSNumber {
            if CFBooleanGetTypeID() == CFGetTypeID(number) { return number.boolValue }
            return number
        }
        if let dict = value as? [String: Any] { return convertToSendableDict(dict) }
        if let array = value as? [Any] { return array.compactMap { toSendableValue($0) } }
        return nil
    }

    private func convertToSendableDict(_ dict: [String: Any]) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        for (key, val) in dict {
            if let sendableVal = toSendableValue(val) { result[key] = sendableVal }
        }
        return result
    }

    private func parseJSONArguments(_ jsonString: String) -> [String: any Sendable] {
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return convertToSendableDict(json)
    }

    private func applyChatTemplate(
        messages: [ChatMessage],
        enableThinking: Bool,
        tools: [ToolSpec]?,
        toolChoice: ToolChoice?
    ) throws -> [Int] {
        let hfMessages: [[String: any Sendable]] = messages.map { message in
            // The Qwen 3.8 chat template renders tool results under
            // `role == "tool"`; the legacy `function` spelling is mapped
            // onto it so both accepted roles reach the template builder.
            let rawRole = (message.role ?? "user").lowercased()
            let role = rawRole == "function" ? "tool" : rawRole
            var item: [String: any Sendable] = [
                "role": role,
                "content": message.content ?? "",
            ]

            if let reasoning = message.reasoning_content, !reasoning.isEmpty {
                item["reasoning_content"] = reasoning
            }

            if let calls = message.tool_calls, !calls.isEmpty {
                item["tool_calls"] = calls.map { call -> [String: any Sendable] in
                    let argsDict = parseJSONArguments(call.function.arguments ?? "{}")
                    var callDict: [String: any Sendable] = [
                        "function": [
                            "name": call.function.name,
                            "arguments": argsDict,
                        ] as [String: any Sendable]
                    ]
                    if let id = call.id { callDict["id"] = id }
                    if let type = call.type { callDict["type"] = type }
                    return callDict
                }
            }

            if role == "tool" {
                if let callID = message.tool_call_id {
                    item["tool_call_id"] = callID
                }
                if let name = message.name {
                    item["name"] = name
                }
            }

            return item
        }

        let kwargs: [String: any Sendable] = [
            "enable_thinking": enableThinking,
            "add_generation_prompt": true,
        ]

        var hfTools: [[String: any Sendable]]? = nil
        if let tools = tools, !tools.isEmpty, toolChoice?.isNone != true {
            let encoder = JSONEncoder()
            if let data = try? encoder.encode(tools),
               let jsonDicts = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                hfTools = jsonDicts.map { convertToSendableDict($0) }
            }
        }

        do {
            return try tokenizer.applyChatTemplate(
                messages: hfMessages,
                tools: hfTools,
                additionalContext: kwargs
            )
        } catch {
            logger.logString(.error, "applyChatTemplate failed: \(error)")
            throw MLXFastError.invalidInput("chat template rendering failed: \(error)")
        }
    }

    func applyChatTemplateWithCache(
        messages: [ChatMessage],
        enableThinking: Bool,
        tools: [ToolSpec]?,
        toolChoice: ToolChoice?
    ) throws -> (tokenIDs: [Int], wasHit: Bool) {
        let toolsKey = tools?.map { $0.toJSONString() }.joined(separator: "|") ?? "none"
        let choiceKey = toolChoice.map { String(describing: $0) } ?? "auto"
        let msgsKey = messages.map { "\($0.role ?? ""):\($0.content ?? ""):\($0.reasoning_content ?? "")" }.joined(separator: "||")
        let promptCacheString = "thinking=\(enableThinking)|choice=\(choiceKey)|tools=\(toolsKey)|msgs=\(msgsKey)"

        let key = TokenizationCache.Key(
            namespace: cacheNamespace,
            enableThinking: enableThinking,
            formattedPrompt: promptCacheString
        )

        let (cached, wasHit) = tokenizationCache.lookup(key: key)
        if let cached = cached {
            return (cached, wasHit)
        }

        let tokenIDs = try applyChatTemplate(
            messages: messages,
            enableThinking: enableThinking,
            tools: tools,
            toolChoice: toolChoice
        )

        tokenizationCache.insert(key: key, tokenIDs: tokenIDs)
        return (tokenIDs, false)
    }

    func estimatePromptTokens(
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) -> Int {
        do {
            let (tokenIDs, _) = try applyChatTemplateWithCache(
                messages: request.messages,
                enableThinking: samplingParams.enableThinking,
                tools: request.tools,
                toolChoice: request.tool_choice
            )
            return tokenIDs.count
        } catch {
            logger.logString(.error, "estimatePromptTokens failed: \(error)")
            return 0
        }
    }

    func tokenizationCacheStats() -> TokenizationCache.Stats {
        tokenizationCache.stats()
    }

    private static func cacheNamespace(modelPath: String, mtpHeadPath: String) -> String {
        "qwen3.8-mtp|model=\(modelPath)|head=\(mtpHeadPath)|template=manual-qwen"
    }

    func generateStream(
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) -> AsyncStream<GenerationFragment> {
        generateStream(
            id: GenerationID(),
            request: request,
            samplingParams: samplingParams
        )
    }

    func generateStream(
        id: GenerationID,
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) -> AsyncStream<GenerationFragment> {
        logger.logString(
            .debug,
            "generateStream started: model=\(request.model) stream=\(request.stream ?? false) messages=\(request.messages.count)"
        )

        let registry = self.cancellationRegistry
        let stopTokens = self.stopTokens
        let maxDraftDepth = self.maxDraftDepth
        let memoryRecoveryPolicy = self.memoryRecoveryPolicy
        let memoryAdmissionPolicy = self.memoryAdmissionPolicy
        let logger = self.logger

        let (seedTokens, wasCacheHit): ([Int], Bool)
        do {
            (seedTokens, wasCacheHit) = try applyChatTemplateWithCache(
                messages: request.messages,
                enableThinking: samplingParams.enableThinking,
                tools: request.tools,
                toolChoice: request.tool_choice
            )
        } catch {
            logger.logString(.error, "Failed to apply chat template: \(error)")
            return AsyncStream { continuation in
                continuation.yield(.finished(reason: "error", promptTokens: 0, completionTokens: 0))
                continuation.finish()
            }
        }

        let inferenceContext = InferenceContext(model: self.model, tokenizer: self.tokenizer)

        return AsyncStream(
            GenerationFragment.self,
            bufferingPolicy: .bufferingOldest(1024)
        ) { continuation in
            let task = Task.detached {
                let model = inferenceContext.model
                let tokenizer = inferenceContext.tokenizer
                
                var emitted = 0
                var promptTokens = 0
                var prefillSeconds = 0.0
                var decodeSeconds = 0.0
                var rounds = 0
                let decodeDepth = samplingParams.mtpEnabled ? maxDraftDepth : MLXFastConstants.qwenMTPSerialControlDepth
                var proposedDraftTokens = 0
                var acceptedDraftTokens = 0
                var finishReason = "length"

                func fragmentKindName(_ fragment: GenerationFragment) -> String {
                    switch fragment {
                    case .content: return "content"
                    case .reasoning: return "reasoning"
                    case .toolCall: return "toolCall"
                    case .finished: return "finished"
                    case .metrics: return "metrics"
                    }
                }

                func yieldAndCheck(_ fragment: GenerationFragment) async throws {
                    switch continuation.yield(fragment) {
                    case .enqueued: break
                    case .terminated, .dropped:
                        await registry.cancel(id, reason: .streamConsumerTerminated)
                        throw GenerationCancelledError(id: id)
                    @unknown default: break
                    }
                }

                func yieldMetrics() async {
                    let cancellationCause = await registry.cancellationReason(for: id)?.description
                    continuation.yield(.metrics(GenerationMetrics(
                        prefillSeconds: prefillSeconds,
                        decodeSeconds: decodeSeconds,
                        rounds: rounds,
                        proposedDraftTokens: proposedDraftTokens,
                        acceptedDraftTokens: acceptedDraftTokens,
                        effectiveDraftDepth: decodeDepth,
                        finishReason: finishReason,
                        promptTokens: promptTokens,
                        completionTokens: emitted,
                        cancellationCause: cancellationCause,
                        tokenizationCacheHit: wasCacheHit
                    )))
                }

                var generationFailed = false
                var exportedSessionState: (tokens: [Int], cache: [any KVCache], hidden: MLXArray, primary: Int, top2: ([Int], [Double]))?

                do {
                    _ = await registry.register(id)

                    if await registry.isCancelled(id) { throw GenerationCancelledError(id: id) }

                    let contextWindow = max(1, samplingParams.contextWindow)
                    let maxPromptTokens = max(1, contextWindow - 16)

                    let effectiveSeedTokens: [Int] = seedTokens.count > maxPromptTokens
                        ? Array(seedTokens.suffix(maxPromptTokens))
                        : seedTokens

                    promptTokens = effectiveSeedTokens.count
                    let maxTokens = max(1, min(samplingParams.maxTokens, contextWindow - promptTokens))

                    if await registry.isCancelled(id) { throw GenerationCancelledError(id: id) }

                    let kvCacheConfig = samplingParams.kvCacheConfig

                    var generateParameters = GenerateParameters()
                    generateParameters.maxTokens = maxTokens
                    generateParameters.kvBits = kvCacheConfig.kvBits
                    generateParameters.kvGroupSize = kvCacheConfig.groupSize
                    generateParameters.quantizedKVStart = kvCacheConfig.quantizedKVStart
                    generateParameters.temperature = samplingParams.temperature
                    generateParameters.topP = samplingParams.topP
                    generateParameters.topK = samplingParams.topK
                    generateParameters.minP = samplingParams.minP
                    generateParameters.repetitionPenalty = samplingParams.repetitionPenalty
                    generateParameters.presencePenalty = samplingParams.presencePenalty
                    generateParameters.frequencyPenalty = samplingParams.frequencyPenalty

                    // Stage 3: Memory pressure eviction hook
                    let activeBytes = Memory.activeMemory
                    let limitBytes = memoryAdmissionPolicy.memoryLimitBytes
                    let pressureThreshold = memoryRecoveryPolicy.pressureThresholdFraction

                    let purged = await self.kvCacheManager.purgeIfMemoryPressure(
                        activeBytes: activeBytes,
                        limitBytes: limitBytes,
                        thresholdFraction: pressureThreshold
                    )
                    if purged {
                        logger.logString(.info, "Pre-generation memory pressure triggered KV cache eviction.")
                    }

                    // A/B (QWEN_MTP_POSTNORM=0): initialize the session with
                    // `postNorm: false` so the hidden row feeding the next
                    // draft skips the final RMSNorm; default (unset/1) keeps
                    // `postNorm: true`.
                    let currentSession = try Qwen38MTPBlockSession(
                        model: model,
                        stopTokens: stopTokens,
                        generateParameters: generateParameters,
                        postNorm: ProcessInfo.processInfo.environment["QWEN_MTP_POSTNORM"] != "0",
                        sampling: MTPSamplingConfig(
                            temperature: samplingParams.temperature,
                            topP: samplingParams.topP,
                            topK: samplingParams.topK,
                            minP: samplingParams.minP,
                            repetitionPenalty: samplingParams.repetitionPenalty,
                            presencePenalty: samplingParams.presencePenalty,
                            frequencyPenalty: samplingParams.frequencyPenalty
                        ),
                        historyLimit: samplingParams.contextWindow,
                        prefillChunkSize: samplingParams.prefillChunkSize
                    )

                    defer { currentSession.release() }

                    let prefillStart = ContinuousClock.now

                    // Stage 1: Consult Radix KV Cache Manager
                    let (prefixCount, cachedEntry) = await self.kvCacheManager.matchPrefix(
                        tokens: effectiveSeedTokens,
                        config: samplingParams.kvCacheConfig,
                        ttlSeconds: samplingParams.ttlSeconds
                    )

                    _ = try currentSession.begin(
                        seedTokens: effectiveSeedTokens,
                        prefixCount: prefixCount,
                        reusableCache: cachedEntry?.cache,
                        reusableHidden: cachedEntry?.hidden,
                        reusablePrimary: cachedEntry?.primary,
                        reusableTop2: cachedEntry?.top2
                    )

                    let prefillElapsed = prefillStart.duration(to: .now)
                    prefillSeconds = Double(prefillElapsed.components.seconds) + Double(prefillElapsed.components.attoseconds) / 1e18

                    let decodeStart = ContinuousClock.now
                    var done = false
                    var reasoningParser = StreamingToolCallParser(
                        toolCallsEnabled: samplingParams.toolCallsEnabled,
                        enableThinking: samplingParams.enableThinking
                    )
                    var stopMatcher = StopSequenceMatcher(sequences: samplingParams.stopSequences)
                    var stoppedByStopSequence = false

                    // TEMPORARY step-level diagnostics (task: isolate the TTLT
                    // throughput regression). Gate: QWEN_MTP_STEP_TRACE=1.
                    // High-resolution wall clock around each generateRound plus
                    // the round's offered/accepted draft counts. Written
                    // directly to stderr via fputs to avoid a per-round logger
                    // actor hop in the hot loop; never enable in a timed run.
                    let stepTrace =
                        ProcessInfo.processInfo.environment["QWEN_MTP_STEP_TRACE"] == "1"
                    var stepLatencyTotalMs = 0.0

                    while emitted < maxTokens && !done {
                        if await registry.isCancelled(id) { throw GenerationCancelledError(id: id) }
                        try Task.checkCancellation()

                        let tStep0 = stepTrace
                            ? DispatchTime.now().uptimeNanoseconds : 0
                        let result = try currentSession.generateRound(depth: decodeDepth)
                        if stepTrace {
                            let stepNs =
                                DispatchTime.now().uptimeNanoseconds - tStep0
                            stepLatencyTotalMs += Double(stepNs) / 1e6
                            let line = String(format:
                                "MTP-STEP round=%d depth=%d offered=%d accepted=%d committed=%d stepMs=%.4f\n",
                                rounds + 1,
                                decodeDepth,
                                result.acceptedDraftCount + result.rejectedDraftCount,
                                result.acceptedDraftCount,
                                result.tokens.count,
                                Double(stepNs) / 1e6
                            )
                            _ = line.withCString { fputs($0, stderr) }
                        }

                        if await registry.isCancelled(id) { throw GenerationCancelledError(id: id) }

                        // Live memory pressure safeguard: if system memory reaches
                        // critical limits (>90% utilized), halt generation safely
                        // and return the completed response up to this token.
                        let activeBytes = Memory.activeMemory
                        let limitBytes = memoryAdmissionPolicy.memoryLimitBytes
                        if limitBytes > 0 && Double(activeBytes) > 0.9 * Double(limitBytes) {
                            finishReason = "memory_pressure"
                            done = true
                            break
                        }

                        rounds += 1
                        proposedDraftTokens += result.acceptedDraftCount + result.rejectedDraftCount
                        acceptedDraftTokens += result.acceptedDraftCount

                        var validTokens: [Int] = []
                        for token in result.tokens {
                            if await registry.isCancelled(id) { throw GenerationCancelledError(id: id) }
                            try Task.checkCancellation()

                            if stopTokens.contains(token) {
                                finishReason = "stop"
                                done = true
                                break
                            }

                            if emitted + validTokens.count >= maxTokens {
                                finishReason = "length"
                                done = true
                                break
                            }

                            validTokens.append(token)
                        }

                        if !validTokens.isEmpty {
                            let text = tokenizer.decode(tokenIds: validTokens)

                            if text.contains("<|im_end|>") || text.contains("<|endoftext|>") {
                                finishReason = "stop"
                                done = true
                            } else {
                                let fragments = coalesceFragments(reasoningParser.parse(text))
                                for fragment in fragments {
                                    switch fragment {
                                    case .toolCall:
                                        try await yieldAndCheck(fragment)

                                    case .content(let fragText), .reasoning(let fragText):
                                        let emitText = stopMatcher.consume(fragText)
                                        if !emitText.isEmpty {
                                            try await yieldAndCheck(fragment.withText(emitText))
                                        }
                                        if stopMatcher.isStopped {
                                            finishReason = "stop"
                                            done = true
                                            stoppedByStopSequence = true
                                            break
                                        }

                                    default:
                                        try await yieldAndCheck(fragment)
                                    }
                                }

                                emitted += validTokens.count
                                if !done && emitted >= maxTokens {
                                    finishReason = "length"
                                    done = true
                                }
                            }
                        }
                    }

                    let decodeElapsed = decodeStart.duration(to: .now)
                    decodeSeconds = Double(decodeElapsed.components.seconds) + Double(decodeElapsed.components.attoseconds) / 1e18

                    if stepTrace && rounds > 0 {
                        // Item D: append whitespace-free QMV verify dispatch
                        // counters (cumulative process lifetime) so the cell
                        // runner can parse engagement per run. Empty
                        // histograms render "-".
                        // Item D: append whitespace-free QMV verify dispatch
                        // counters (cumulative process lifetime) so the cell
                        // runner can parse engagement per run. Empty
                        // histograms render "-". Appended via concatenation —
                        // String(format:) %s expects a C string, not a Swift
                        // String.
                        let qmvTokens = Qwen35QMVVerifyDispatch
                            .summaryTokens().joined(separator: " ")
                        let summary = String(format:
                            "MTP-STEP-SUMMARY rounds=%d proposed=%d accepted=%d acceptedPerStep=%.4f avgStepMs=%.4f decodeSeconds=%.4f committed=%d",
                            rounds,
                            proposedDraftTokens,
                            acceptedDraftTokens,
                            Double(acceptedDraftTokens) / Double(rounds),
                            stepLatencyTotalMs / Double(rounds),
                            decodeSeconds,
                            emitted
                        ) + " " + qmvTokens + "\n"
                        _ = summary.withCString { fputs($0, stderr) }
                    }

                    if !stoppedByStopSequence {
                        for fragment in reasoningParser.finish() {
                            try await yieldAndCheck(fragment)
                        }
                    }

                    try await yieldAndCheck(
                        .finished(
                            reason: finishReason,
                            promptTokens: promptTokens,
                            completionTokens: emitted
                        )
                    )

                    await yieldMetrics()

                    // Export session state for KV reuse
                    exportedSessionState = currentSession.exportState()
                    continuation.finish()

                } catch is GenerationCancelledError {
                    await yieldMetrics()
                    continuation.finish()
                } catch is CancellationError {
                    await yieldMetrics()
                    continuation.finish()
                } catch {
                    generationFailed = true
                    await yieldMetrics()
                    continuation.finish()
                }

                await registry.deregister(id)

                // Store completed session state into Radix Tree KV Cache Manager
                if !generationFailed, let state = exportedSessionState {
                    await self.kvCacheManager.store(
                        RadixKVCacheManager.CacheEntry(
                            tokens: state.tokens,
                            cache: state.cache,
                            hidden: state.hidden,
                            primary: state.primary,
                            top2: state.top2,
                            config: samplingParams.kvCacheConfig
                        )
                    )
                }

                // Opt-in memory recovery
                if generationFailed {
                    if memoryRecoveryPolicy.shouldRecover(
                        reason: .failedRequest,
                        activeBytes: Memory.activeMemory,
                        limitBytes: memoryAdmissionPolicy.memoryLimitBytes
                    ) {
                        Memory.clearCache()
                        await self.recordMemoryRecovery()
                    }
                } else if memoryRecoveryPolicy.shouldRecover(
                    reason: .memoryPressure,
                    activeBytes: Memory.activeMemory,
                    limitBytes: memoryAdmissionPolicy.memoryLimitBytes
                ) {
                    Memory.clearCache()
                    await self.recordMemoryRecovery()
                }
            }

            continuation.onTermination = { _ in
                Task {
                    if await registry.isRegistered(id) {
                        await registry.cancel(id, reason: .streamConsumerTerminated)
                    }
                }
                task.cancel()
            }
        }
    }

    func generateTokenIDs(
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) async throws -> [Int] {
        let stopTokens = self.stopTokens
        let maxDraftDepth = self.maxDraftDepth
        let inferenceContext = InferenceContext(model: self.model, tokenizer: self.tokenizer)
        let model = inferenceContext.model

        let seedTokens = try applyChatTemplate(
            messages: request.messages,
            enableThinking: samplingParams.enableThinking,
            tools: request.tools,
            toolChoice: request.tool_choice
        )
        let maxTokens = max(1, samplingParams.maxTokens)
        let contextWindow = max(1, samplingParams.contextWindow)
        let maxPromptTokens = max(1, contextWindow - maxTokens)

        let effectiveSeedTokens: [Int] = seedTokens.count > maxPromptTokens
            ? Array(seedTokens.suffix(maxPromptTokens))
            : seedTokens

        var generateParameters = GenerateParameters()
        generateParameters.maxTokens = maxTokens
        generateParameters.temperature = samplingParams.temperature
        generateParameters.topP = samplingParams.topP
        generateParameters.topK = samplingParams.topK
        generateParameters.minP = samplingParams.minP

        let session = try Qwen38MTPBlockSession(
            model: model,
            stopTokens: stopTokens,
            generateParameters: generateParameters,
            sampling: MTPSamplingConfig(
                temperature: samplingParams.temperature,
                topP: samplingParams.topP,
                topK: samplingParams.topK,
                minP: samplingParams.minP,
                repetitionPenalty: samplingParams.repetitionPenalty,
                presencePenalty: samplingParams.presencePenalty,
                frequencyPenalty: samplingParams.frequencyPenalty
            ),
            historyLimit: samplingParams.contextWindow,
            prefillChunkSize: samplingParams.prefillChunkSize
        )

        _ = try session.begin(seedTokens: effectiveSeedTokens)

        let decodeDepth = samplingParams.mtpEnabled ? maxDraftDepth : MLXFastConstants.qwenMTPSerialControlDepth
        var allTokens: [Int] = []
        var done = false

        while allTokens.count < maxTokens && !done {
            let result = try session.generateRound(depth: decodeDepth)
            for token in result.tokens {
                if allTokens.count >= maxTokens || stopTokens.contains(token) {
                    done = true
                    break
                }
                allTokens.append(token)
            }
        }

        return allTokens
    }
}

extension MLXGenerator: GenerationProvider {}

private func resolveStopTokens(
    directory: URL,
    tokenizer: (any MLXLMCommon.Tokenizer)?,
    logger: Logger
) -> Set<Int> {
    var ids = Set<Int>()
    ids.insert(248046) // <|im_end|>
    ids.insert(248044) // <|endoftext|>

    for name in ["config.json", "generation_config.json"] {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(name)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            continue
        }

        for key in ["eos_token_id", "pad_token_id"] {
            switch root[key] {
            case let value as Int: ids.insert(value)
            case let values as [Any]: ids.formUnion(values.compactMap { $0 as? Int })
            default: break
            }
        }
    }

    if let eos = tokenizer?.eosTokenId { ids.insert(eos) }
    return ids
}

private final class MLXAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

private func waitForMLXAsync<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = MLXAsyncResultBox<T>()
    let unsafeBox = box

    Task.detached {
        do {
            unsafeBox.result = .success(try await operation())
        } catch {
            unsafeBox.result = .failure(error)
        }
        semaphore.signal()
    }

    semaphore.wait()

    guard let result = box.result else {
        throw MLXFastError.invalidInput("the MTP async model load completed without a result")
    }

    return try result.get()
}

private extension Logger {
    func logString(_ level: Logger.Level, _ message: String) {
        log(level: level, "\(message)")
    }
}