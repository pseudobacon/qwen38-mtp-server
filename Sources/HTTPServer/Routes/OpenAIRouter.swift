// OpenAIRouter.swift
//
// Wires the OpenAI-compatible `POST /v1/chat/completions` route to the
// `MLXGenerator` actor.
//
// The generator actor is the only place MLX is touched; this file only
// bridges the actor's `AsyncStream` to the HTTP layer.

import Foundation
import Vapor

/// Computes the OpenAI `finish_reason` for a chat completion given the number
/// of tool calls produced and the generator's terminal reason. When one or
/// more tool calls were produced and the model terminated normally ("stop"),
/// the reason is `tool_calls` (OpenAI semantics); otherwise the generator's
/// reason ("stop"/"length"/"memory_pressure") is preserved.
func toolCallFinishReason(toolCallCount: Int, finishedReason: String) -> String {
    toolCallCount > 0 && finishedReason == "stop" ? "tool_calls" : finishedReason
}

/// Registers OpenAI-compatible routes on `app`, backed by `generator` and
/// the single-lane `scheduler`. Every chat-completion request (streaming or
/// non-streaming) is admitted through `scheduler.schedule`, which bounds the
/// FIFO queue and rejects overload with HTTP 429 `engine_overloaded`.
///
/// Before scheduling, the request is checked against the generator's
/// `MemoryAdmissionPolicy`: if the estimated KV-cache footprint cannot fit in
/// the configured memory budget, the request is rejected with an OpenAI-shaped
/// 507 before any model execution.
func registerOpenAIRoutes(
    on app: Application,
    scheduler: GenerationScheduler?,
    runtimeState: ModelRuntimeState,
    generator: MLXGenerator?,
    metricsCollector: MetricsCollector,
    sessionStore: SessionStore
) {
    // Aggregate server metrics over the bounded rolling window. Read-only:
    // it never touches the model executor or the generation lane.
    app.get("metrics") { _ async throws -> Response in
        var summary = await metricsCollector.summary()
        // Merge the cumulative tokenization-cache counters (Stage 0) from the
        // generator. These are server-wide and not derived from the rolling
        // request window. No prompt content is ever logged.
        if let generator {
            let stats = await generator.tokenizationCacheStats()
            summary.tokenizationCacheHits = stats.hits
            summary.tokenizationCacheMisses = stats.misses
            summary.tokenizationCacheEvictions = stats.evictions
            summary.tokenizationCacheEntries = stats.entryCount
            summary.tokenizationCacheBytes = stats.byteCount
            let total = stats.hits + stats.misses
            summary.tokenizationCacheHitRate = total > 0
                ? Double(stats.hits) / Double(total)
                : nil
        }
        let json = try JSONEncoder().encode(summary)
        return Response(
            status: .ok,
            headers: [
                "Content-Type": "application/json; charset=utf-8"
            ],
            body: .init(data: Data(json))
        )
    }

    app.post("v1", "chat", "completions") { req async throws -> Response in
        // Admission gate: reject with 503 unless the runtime is `.ready` and
        // the scheduler is available. This stops admitting new
        // request streams during `.draining`, `.failed`, `.warming`,
        // `.initializing`, and `.memoryDegraded`.
        guard await runtimeState.isReady(), let scheduler, let generator else {
            return await readyzResponse(runtimeState)
        }

        // Decode the request body. Malformed JSON is an invalid request
        // shape and gets an OpenAI-shaped 400 before any model execution.
        let decoded: ChatCompletionRequest
        do {
            decoded = try req.content.decode(ChatCompletionRequest.self)
        } catch {
            return openAIErrorResponse(
                OpenAIRequestError(
                    message: "Request body is not valid JSON.",
                    param: nil,
                    code: "invalid_value"
                )
            )
        }

        // Intercept the `/nothink` / `/no_think` slash command: if the last
        // message's content starts with the command, strip it from the user
        // text and force `enable_thinking: false` so the prompt formatter
        // applies the Qwen `/no_think` marker convention instead of feeding
        // the raw command to the model.
        let request = applyNoThinkSlashCommand(to: decoded)

        let serverConfig = app.storage[ServerConfigKey.self] ?? ServerConfig()

        return try await performChatCompletion(
            app: app,
            request: request,
            serverConfig: serverConfig,
            runtimeState: runtimeState,
            scheduler: scheduler,
            generator: generator,
            metricsCollector: metricsCollector,
            logger: req.logger,
            sessionID: nil,
            onFinished: nil
        )
    }

    // Shared completion pipeline for the stateless and session endpoints.
    // `request` is the fully-rendered request (messages already merged for a
    // session). `sessionID` (when set) is echoed on the non-streaming
    // response. `onFinished` (when set) is invoked exactly once with the
    // outcome: `(success, commit)`; on success `commit` carries the assistant
    // message fields and the committed token IDs.
    @Sendable
    func performChatCompletion(
        app: Application,
        request: ChatCompletionRequest,
        serverConfig: ServerConfig,
        runtimeState: ModelRuntimeState,
        scheduler: GenerationScheduler,
        generator: MLXGenerator,
        metricsCollector: MetricsCollector,
        logger: Logger,
        sessionID: String?,
        onFinished: (@Sendable (Bool, CompletionCommit?) async -> Void)?
    ) async throws -> Response {
        // Validate the request against the server configuration and the
        // capabilities of the current runtime, before any model execution.
        // Every rejection is an OpenAI-shaped 400.
        let samplingParams: SamplingParameters
        do {
            samplingParams = try ChatCompletionRequestValidator.validate(
                request: request,
                serverConfig: serverConfig
            )
        } catch let error as OpenAIRequestError {
            return openAIErrorResponse(error)
        } catch {
            // Validation only throws OpenAIRequestError; anything else is a
            // server bug and propagates as a 500.
            throw error
        }

        let model = request.model
        let stream = request.stream ?? false
        let maxTokens = samplingParams.maxTokens
        let contextWindow = samplingParams.contextWindow
        let temperature = samplingParams.temperature
        let topP = samplingParams.topP
        let topK = samplingParams.topK
        let minP = samplingParams.minP
        let enableThinking = samplingParams.enableThinking
        let mtpEnabled = samplingParams.mtpEnabled
        var logMessage = "chat.completions: model=\(model) stream=\(stream)"
        logMessage += " maxTokens=\(maxTokens) contextWindow=\(contextWindow)"
        logMessage += " temperature=\(temperature) topP=\(topP) topK=\(topK)"
        logMessage += " minP=\(minP) enableThinking=\(enableThinking)"
        logMessage += " mtpEnabled=\(mtpEnabled)"
        logger.debug("\(logMessage)")

        let responseId = "chatcmpl-\(UUID().uuidString)"
        let timestamp = Int(Date().timeIntervalSince1970)

        // Router-owned, request-scoped generation identity. Deliberately
        // separate from the OpenAI-facing `responseId`: this is the server's
        // internal cancellation handle for this one generation, passed to the
        // generator and to every matching `cancelGeneration` call.
        let generationID = GenerationID()

        // Whether the caller asked for a usage report. For streaming this
        // gates the dedicated usage-only chunk; for non-streaming the usage is
        // always included in the single response.
        let includeUsage = request.stream_options?.include_usage == true
        let includeTokenIDs = request.include_token_ids == true

        // Memory-aware admission control: estimate the prompt token count
        // (CPU-only tokenization, no model forward) and check the estimated
        // KV-cache footprint against the configured memory budget. This runs
        // BEFORE scheduling and BEFORE any model execution, so an over-budget
        // request is rejected with an OpenAI-shaped 507 without touching the
        // single-lane queue or the model.
            let promptTokens = await generator.estimatePromptTokens(
                request: request,
                samplingParams: samplingParams
            )
            let promptTokenBudget = max(0, contextWindow - maxTokens)
            let effectivePromptTokens = min(promptTokens, promptTokenBudget)

            do {
                // Cap the completion estimate at 4096: the admission check is a
                // pre-flight bound, not a guarantee. Even if the user requests
                // a large max_tokens, the actual generation is bounded by the
                // context window and the KV cache grows incrementally.
                let effectiveCompletionTokens = min(maxTokens, 4096)
                try generator.memoryAdmissionPolicy.check(
                    promptTokens: effectivePromptTokens,
                    completionTokens: effectiveCompletionTokens
                )
            } catch let error as MemoryAdmissionPolicy.AdmissionFailure {
                return openAIMemoryErrorResponse(error)
            }

        // Admit the request through the single-lane scheduler. This suspends
        // without blocking the HTTP thread until the lane is free. If the
        // queue is full, it throws `queueFull` and we return an OpenAI-shaped
        // 429 before any model execution.
        let admissionStart = ContinuousClock.now
        let generationStream: AsyncStream<GenerationFragment>
        do {
            generationStream = try await scheduler.schedule(
                id: generationID,
                request: request,
                samplingParams: samplingParams
            )
        } catch GenerationScheduler.SchedulerError.queueFull {
            return openAIOverloadedResponse()
        }
        let queueWaitSeconds = durationSeconds(admissionStart, .now)

        if request.stream == true {
            return Response(
                status: .ok,
                headers: [
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                ],
                body: Response.Body(stream: { writer in
                    Task {
                        let encoder = JSONEncoder()
                        let allocator = ByteBufferAllocator()

                        // DIAGNOSTIC (temporary, privacy-safe): complete SSE trace.
                        // Enabled only when QWEN_STREAM_DIAG=1. Logs request ID,
                        // event sequence, stage, and counts/flags. No prompt or
                        // completion text is logged.
                        let sseDiagEnabled =
                            ProcessInfo.processInfo.environment["QWEN_STREAM_DIAG"] == "1"
                        var sseSeq = 0

                        func sseDiagLog(_ stage: String, _ fields: [String]) {
                            guard sseDiagEnabled else { return }
                            sseSeq += 1
                            let elapsed = admissionStart.duration(to: .now)
                            let elapsedMs = Double(elapsed.components.attoseconds) / 1e15
                            let fieldStr = fields.joined(separator: " ")
                            logger.log(
                                level: .info,
                                Logger.Message(
                                    stringLiteral: "STREAMDIAG id=\(generationID) "
                                        + "evt=\(sseSeq) stage=\(stage) "
                                        + "elapsed_ms=\(String(format: "%.1f", elapsedMs)) "
                                        + fieldStr
                                )
                            )
                        }

                        // Helper to emit individual SSE chunks over the wire.
                        func writeSSE<T: Encodable>(_ object: T) async throws {
                            let json = try encoder.encode(object)
                            var buffer = allocator.buffer(capacity: json.count + 8)
                            buffer.writeString("data: ")
                            buffer.writeBytes(json)
                            buffer.writeString("\n\n")
                            let writeStart = ContinuousClock.now
                            sseDiagLog("SSE_WRITE_ATTEMPT", [
                                "bytes=\(buffer.readableBytes)"
                            ])
                            do {
                                _ = try await writer.write(.buffer(buffer)).get()
                                let writeMs = Double(
                                    writeStart.duration(to: .now).components.attoseconds
                                ) / 1e15
                                sseDiagLog("SSE_FLUSH_COMPLETE", [
                                    "bytes=\(buffer.readableBytes)",
                                    "write_ms=\(String(format: "%.1f", writeMs))",
                                    "ok=true"
                                ])
                            } catch {
                                sseDiagLog("SSE_FLUSH_COMPLETE", [
                                    "bytes=\(buffer.readableBytes)",
                                    "ok=false"
                                ])
                                throw error
                            }
                        }

                        // Telemetry state for this request. `firstContentTime`
                        // is the admission-to-first-committed-byte TTFT
                        // measurement; `generatorMetrics` carries the
                        // generator-side measurements yielded at stream end.
                        var finalReason = "stop"
                        var finalUsage: Usage?
                        var finalTokenIDs: [Int] = []
                        var streamContent = ""
                        var streamReasoning = ""
                        var firstContentTime: ContinuousClock.Instant?
                        var generatorMetrics: GenerationMetrics?
                        var toolCalls: [ToolCall] = []

                        /// Builds the per-request metrics and records them in
                        /// the collector asynchronously. Fire-and-forget: the
                        /// recording hop never blocks the SSE stream.
                        func recordMetrics() {
                            let metrics = RequestMetrics(
                                requestID: generationID.description,
                                modelAlias: request.model,
                                isStream: true,
                                queueWaitSeconds: queueWaitSeconds,
                                ttftSeconds: firstContentTime.map {
                                    durationSeconds(admissionStart, $0)
                                },
                                generationSeconds: durationSeconds(admissionStart, .now),
                                prefillSeconds: generatorMetrics?.prefillSeconds ?? 0,
                                decodeSeconds: generatorMetrics?.decodeSeconds ?? 0,
                                promptTokens: generatorMetrics?.promptTokens ?? 0,
                                completionTokens: generatorMetrics?.completionTokens ?? 0,
                                mtpEnabled: mtpEnabled,
                                effectiveDraftDepth: generatorMetrics?.effectiveDraftDepth ?? 0,
                                mtpRounds: generatorMetrics?.rounds ?? 0,
                                proposedDraftTokens: generatorMetrics?.proposedDraftTokens ?? 0,
                                acceptedDraftTokens: generatorMetrics?.acceptedDraftTokens ?? 0,
                                finishReason: generatorMetrics?.finishReason ?? finalReason,
                                cancellationCause: generatorMetrics?.cancellationCause,
                                memoryAdmission: .admitted,
                                tokenizationCacheHit: generatorMetrics?.tokenizationCacheHit ?? false
                            )
                            Task { await metricsCollector.record(metrics) }
                        }

                        do {
                            // 1. Initial assistant chunk
                            let initialChunk = ChatCompletionChunk(
                                id: responseId,
                                object: "chat.completion.chunk",
                                created: timestamp,
                                model: request.model,
                                choices: [
                                    ChunkChoice(
                                        index: 0,
                                        delta: ChatMessage(
                                            role: "assistant",
                                            content: nil,
                                            reasoning: nil,
                                            reasoning_content: nil
                                        ),
                                        finish_reason: nil
                                    )
                                ],
                                usage: nil
                            )
                            try await writeSSE(initialChunk)

                            // 2. Real-time generator event loop (admitted stream)
                            for await fragment in generationStream {
                                switch fragment {
                                case .content(let content):
                                    if firstContentTime == nil {
                                        firstContentTime = .now
                                    }
                                    streamContent += content
                                    let chunk = ChatCompletionChunk(
                                        id: responseId,
                                        object: "chat.completion.chunk",
                                        created: timestamp,
                                        model: request.model,
                                        choices: [
                                            ChunkChoice(
                                                index: 0,
                                                delta: ChatMessage(
                                                    role: nil,
                                                    content: content,
                                                    reasoning: nil,
                                                    reasoning_content: nil
                                                ),
                                                finish_reason: nil
                                            )
                                        ],
                                        usage: nil
                                    )
                                    try await writeSSE(chunk)

                                case .reasoning(let reasoning):
                                    if firstContentTime == nil {
                                        firstContentTime = .now
                                    }
                                    streamReasoning += reasoning
                                    let chunk = ChatCompletionChunk(
                                        id: responseId,
                                        object: "chat.completion.chunk",
                                        created: timestamp,
                                        model: request.model,
                                        choices: [
                                            ChunkChoice(
                                                index: 0,
                                                delta: ChatMessage(
                                                    role: nil,
                                                    content: nil,
                                                    reasoning: reasoning,
                                                    reasoning_content: reasoning
                                                ),
                                                finish_reason: nil
                                            )
                                        ],
                                        usage: nil
                                    )
                                    try await writeSSE(chunk)

                                case .finished(let reason, let promptTokens, let completionTokens, let completionTokenIDs):
                                    // When the model produced one or more tool
                                    // calls and terminated normally ("stop"),
                                    // the finish reason is "tool_calls"
                                    // (OpenAI semantics).
                                    finalReason = toolCallFinishReason(
                                        toolCallCount: toolCalls.count,
                                        finishedReason: reason
                                    )
                                    finalTokenIDs = completionTokenIDs
                                    if includeUsage {
                                        finalUsage = Usage(
                                            prompt_tokens: promptTokens,
                                            completion_tokens: completionTokens,
                                            total_tokens: promptTokens + completionTokens
                                        )
                                    }

                                case .metrics(let metrics):
                                    // Telemetry only: never written to SSE.
                                    generatorMetrics = metrics

                                case .toolCall(let call):
                                    // Serialize the completed tool call as a
                                    // delta chunk. OpenAI streams tool calls as
                                    // `delta.tool_calls` entries; each completed
                                    // call is emitted as a single delta with the
                                    // full call (id, type, function name + args).
                                    toolCalls.append(call)
                                    let chunk = ChatCompletionChunk(
                                        id: responseId,
                                        object: "chat.completion.chunk",
                                        created: timestamp,
                                        model: request.model,
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
                                    try await writeSSE(chunk)
                                }
                            }

                            // 3. Final chunk with finish_reason
                            let finalChunk = ChatCompletionChunk(
                                id: responseId,
                                object: "chat.completion.chunk",
                                created: timestamp,
                                model: request.model,
                                choices: [
                                    ChunkChoice(
                                        index: 0,
                                        delta: ChatMessage(
                                            role: nil,
                                            content: nil,
                                            reasoning: nil,
                                            reasoning_content: nil,
                                            token_ids: includeTokenIDs ? finalTokenIDs : nil
                                        ),
                                        finish_reason: finalReason
                                    )
                                ],
                                usage: nil
                            )
                            try await writeSSE(finalChunk)

                            // 4. Dedicated usage chunk (if requested)
                            if let finalUsage = finalUsage {
                                let usageChunk = ChatCompletionChunk(
                                    id: responseId,
                                    object: "chat.completion.chunk",
                                    created: timestamp,
                                    model: request.model,
                                    choices: [],
                                    usage: finalUsage
                                )
                                try await writeSSE(usageChunk)
                            }

                            // 5. [DONE] signal
                            var doneBuffer = allocator.buffer(capacity: 16)
                            doneBuffer.writeString("data: [DONE]\n\n")
                            _ = try await writer.write(.buffer(doneBuffer)).get()

                            // Close response stream
                            _ = try await writer.write(.end).get()

                            if let onFinished {
                                let seedTokens = await generator.promptTokenIDs(
                                    request: request, samplingParams: samplingParams
                                )
                                let commit = CompletionCommit(
                                    content: streamContent,
                                    reasoningContent: streamReasoning.isEmpty ? nil : streamReasoning,
                                    toolCalls: toolCalls,
                                    completionTokenIDs: finalTokenIDs,
                                    seedTokens: seedTokens,
                                    promptTokens: finalUsage?.prompt_tokens ?? 0,
                                    completionTokens: finalUsage?.completion_tokens ?? 0
                                )
                                await onFinished(true, commit)
                            }

                            recordMetrics()
                        } catch {
                            // A write failed: cancel only the matching generation
                            // with `.sseWriteFailure`, log the ID and error only,
                            // and return. Abandoning the `for await` loop
                            // terminates the producer; no further writes happen.
                            await scheduler.cancel(id: generationID, reason: .sseWriteFailure)
                            logger.warning(
                                "SSE write failed for generation \(generationID); error=\(error.localizedDescription)"
                            )
                            if let onFinished {
                                await onFinished(false, nil)
                            }
                            recordMetrics()
                        }
                        // Always release the lane, whether the generation
                        // completed, failed, or was cancelled.
                        await scheduler.releaseLane(id: generationID)
                    }
                })
            )
        }

        // Non-streaming response block.
        var content = ""
        var reasoningContent = ""
        var finishReason = "stop"
        var usage: Usage?
        var toolCalls: [ToolCall] = []
        var completionTokenIDs: [Int] = []
        var firstFragmentTime: ContinuousClock.Instant?
        var generatorMetrics: GenerationMetrics?
        var metricsRecorded = false

        /// Records the per-request metrics exactly once, asynchronously.
        /// Fire-and-forget: the recording hop never blocks the response.
        func recordMetrics() {
            guard !metricsRecorded else { return }
            metricsRecorded = true
            let metrics = RequestMetrics(
                requestID: generationID.description,
                modelAlias: request.model,
                isStream: false,
                queueWaitSeconds: queueWaitSeconds,
                ttftSeconds: firstFragmentTime.map {
                    durationSeconds(admissionStart, $0)
                },
                generationSeconds: durationSeconds(admissionStart, .now),
                prefillSeconds: generatorMetrics?.prefillSeconds ?? 0,
                decodeSeconds: generatorMetrics?.decodeSeconds ?? 0,
                promptTokens: generatorMetrics?.promptTokens ?? 0,
                completionTokens: generatorMetrics?.completionTokens ?? 0,
                mtpEnabled: mtpEnabled,
                effectiveDraftDepth: generatorMetrics?.effectiveDraftDepth ?? 0,
                mtpRounds: generatorMetrics?.rounds ?? 0,
                proposedDraftTokens: generatorMetrics?.proposedDraftTokens ?? 0,
                acceptedDraftTokens: generatorMetrics?.acceptedDraftTokens ?? 0,
                finishReason: generatorMetrics?.finishReason ?? finishReason,
                cancellationCause: generatorMetrics?.cancellationCause,
                memoryAdmission: .admitted,
                tokenizationCacheHit: generatorMetrics?.tokenizationCacheHit ?? false
            )
            Task { await metricsCollector.record(metrics) }
        }
        defer { recordMetrics() }

        // Use withTaskCancellationHandler so that if the client disconnects
        // (e.g. during a long MTP prefill), the cancellation fires instantly
        // without waiting for the next token to drop into the `for await` loop.
        await withTaskCancellationHandler {
            for await fragment in generationStream {
                // We still check here as a fallback guard
                if Task.isCancelled {
                    break
                }
                if firstFragmentTime == nil {
                    firstFragmentTime = .now
                }

                switch fragment {
                case .content(let text):
                    content += text
                case .reasoning(let text):
                    reasoningContent += text
                case .finished(let reason, let promptTokens, let completionTokens, let tokenIDs):
                    finishReason = reason
                    completionTokenIDs = tokenIDs
                    usage = Usage(
                        prompt_tokens: promptTokens,
                        completion_tokens: completionTokens,
                        total_tokens: promptTokens + completionTokens
                    )
                case .metrics(let metrics):
                    // Telemetry only: never written to the response.
                    generatorMetrics = metrics

                case .toolCall(let call):
                    // Accumulate the completed tool call for the final
                    // non-streaming response. Each `.toolCall` fragment is a
                    // fully-parsed call (id, type, function name + arguments).
                    toolCalls.append(call)
                }
            }
        } onCancel: {
            // This closure runs immediately on a concurrent executor when the HTTP task is cancelled.
            Task {
                await scheduler.cancel(id: generationID, reason: .streamConsumerTerminated)
            }
        }

        // Always release the lane, whether the generation completed, failed,
        // or was cancelled.
        await scheduler.releaseLane(id: generationID)

        // Never fabricate a successful completion after cancellation.
        try Task.checkCancellation()

        // Notify the session (if any) of a successful completion. The stored
        // assistant message carries no token IDs (it is for re-rendering);
        // token IDs are diagnostic-only and live on the API response.
        if let onFinished {
            let seedTokens = await generator.promptTokenIDs(
                request: request, samplingParams: samplingParams
            )
            let commit = CompletionCommit(
                content: content,
                reasoningContent: reasoningContent.isEmpty ? nil : reasoningContent,
                toolCalls: toolCalls,
                completionTokenIDs: completionTokenIDs,
                seedTokens: seedTokens,
                promptTokens: usage?.prompt_tokens ?? 0,
                completionTokens: usage?.completion_tokens ?? 0
            )
            await onFinished(true, commit)
        }

        let response = ChatCompletionResponse(
            id: responseId,
            object: "chat.completion",
            created: timestamp,
            model: request.model,
            choices: [
                CompletionChoice(
                    index: 0,
                    message: ChatMessage(
                        role: "assistant",
                        content: content,
                        reasoning: nil,
                        reasoning_content: reasoningContent.isEmpty ? nil : reasoningContent,
                        tool_calls: toolCalls.isEmpty ? nil : toolCalls,
                        token_ids: includeTokenIDs ? completionTokenIDs : nil
                    ),
                    finish_reason: toolCallFinishReason(
                        toolCallCount: toolCalls.count,
                        finishedReason: finishReason
                    )
                )
            ],
            usage: usage,
            session_id: sessionID
        )

        let json = try JSONEncoder().encode(response)

        return Response(
            status: .ok,
            headers: [
                "Content-Type": "application/json; charset=utf-8"
            ],
            body: .init(data: Data(json))
        )
    }

    // MARK: - Session API
    //
    // Sessions are process-local, in-memory conversation owners. A session
    // guarantees conversation ownership and history continuity; it does NOT
    // guarantee a KV-cache hit (the prefix is re-rendered and the radix cache
    // is consulted opportunistically, exactly like the stateless path).

    // POST /v1/sessions — create an empty session.
    app.post("v1", "sessions") { req async throws -> Response in
        guard await runtimeState.isReady() else {
            return await readyzResponse(runtimeState)
        }
        let session = await sessionStore.create()
        let data = try JSONEncoder().encode(session)
        return Response(
            status: .created,
            headers: ["Content-Type": "application/json; charset=utf-8"],
            body: .init(data: Data(data))
        )
    }

    // GET /v1/sessions/:id — inspect a session (history + token count).
    app.get("v1", "sessions", ":id") { req async throws -> Response in
        guard await runtimeState.isReady() else {
            return await readyzResponse(runtimeState)
        }
        guard let id = req.parameters.get("id"),
              let session = await sessionStore.session(id) else {
            return openAIStatusErrorResponse(
                status: .notFound,
                message: "Session not found",
                code: "session_not_found"
            )
        }
        let data = try JSONEncoder().encode(session)
        return Response(
            status: .ok,
            headers: ["Content-Type": "application/json; charset=utf-8"],
            body: .init(data: Data(data))
        )
    }

    // DELETE /v1/sessions/:id — delete a session and release its cached prefix.
    app.delete("v1", "sessions", ":id") { req async throws -> Response in
        guard await runtimeState.isReady(), let generator else {
            return await readyzResponse(runtimeState)
        }
        guard let id = req.parameters.get("id") else {
            return openAIStatusErrorResponse(
                status: .badRequest, message: "Missing session id", code: "invalid_value"
            )
        }
        let freedPrefix = await sessionStore.delete(id)
        if let tokens = freedPrefix, !tokens.isEmpty {
            await generator.releaseCachedPrefix(tokens: tokens)
        }
        let data = try JSONEncoder().encode(SessionDeleteResponse(deleted: true))
        return Response(
            status: .ok,
            headers: ["Content-Type": "application/json; charset=utf-8"],
            body: .init(data: Data(data))
        )
    }

    // POST /v1/sessions/:id/completions — generate within a session. The body is
    // a normal chat-completion request whose `messages` are the new turn(s);
    // they are appended to the session history, and the assistant reply is
    // committed back on success.
    app.post("v1", "sessions", ":id", "completions") { req async throws -> Response in
        guard await runtimeState.isReady(), let scheduler, let generator else {
            return await readyzResponse(runtimeState)
        }
        guard let id = req.parameters.get("id") else {
            return openAIStatusErrorResponse(
                status: .badRequest, message: "Missing session id", code: "invalid_value"
            )
        }
        let decoded: ChatCompletionRequest
        do {
            decoded = try req.content.decode(ChatCompletionRequest.self)
        } catch {
            return openAIErrorResponse(
                OpenAIRequestError(
                    message: "Request body is not valid JSON.",
                    param: nil,
                    code: "invalid_value"
                )
            )
        }
        guard let baseMessages = await sessionStore.messages(id) else {
            return openAIStatusErrorResponse(
                status: .notFound, message: "Session not found", code: "session_not_found"
            )
        }
        guard await sessionStore.beginCompletion(id) else {
            return openAIStatusErrorResponse(
                status: .conflict,
                message: "Session already has an in-flight completion",
                code: "session_busy"
            )
        }
        var request = applyNoThinkSlashCommand(to: decoded)
        let requestPart = request.messages
        request.messages = baseMessages + requestPart
        let serverConfig = app.storage[ServerConfigKey.self] ?? ServerConfig()
        do {
            return try await performChatCompletion(
                app: app,
                request: request,
                serverConfig: serverConfig,
                runtimeState: runtimeState,
                scheduler: scheduler,
                generator: generator,
                metricsCollector: metricsCollector,
                logger: req.logger,
                sessionID: id,
                onFinished: { success, commit in
                    if success, let commit {
                        await sessionStore.finishCompletion(
                            id, requestMessages: requestPart, commit: commit
                        )
                    } else {
                        await sessionStore.abortCompletion(id)
                    }
                }
            )
        } catch {
            await sessionStore.abortCompletion(id)
            throw error
        }
    }
}

/// Intercepts the `/nothink` and `/no_think` slash commands.
///
/// If the last message's content starts with `/nothink` or `/no_think`
/// (optionally after leading whitespace), the command is stripped from the
/// message text — the rest of the user prompt is preserved intact — and the
/// returned request carries `enable_thinking: false`, so
/// `SamplingParameters.fromRequest` resolves thinking as disabled. The
/// stripped user prompt is then guaranteed to end with the native Qwen
/// `/no_think` soft-switch marker (appended when absent), so the model
/// switches to non-thinking mode through its own weights instead of any
/// prefilled thought block. The request is returned unchanged when the
/// command is not present. Only the last message is inspected; the command
/// is matched as a prefix, never mid-string.
func applyNoThinkSlashCommand(to request: ChatCompletionRequest) -> ChatCompletionRequest {
    guard let last = request.messages.indices.last else { return request }
    let message = request.messages[last]
    guard let content = message.content else { return request }

    let trimmed = content.trimmingCharacters(in: .whitespaces)
    let prefixes = ["/nothink", "/no_think"]
    guard let prefix = prefixes.first(where: { trimmed.hasPrefix($0) }) else {
        return request
    }

    var rest = trimmed.dropFirst(prefix.count)
    while let c = rest.first, c.isWhitespace {
        rest = rest.dropFirst()
    }

    // Guarantee the native Qwen `/no_think` soft-switch marker: the stripped
    // user prompt must end with (or at least contain) it, so the model
    // switches to non-thinking mode through its own weights.
    var updatedContent = String(rest)
    if !updatedContent.contains("/no_think") {
        updatedContent = updatedContent.isEmpty
            ? "/no_think"
            : updatedContent + " /no_think"
    }

    var messages = request.messages
    messages[last] = ChatMessage(
        role: message.role,
        content: updatedContent,
        reasoning: message.reasoning,
        reasoning_content: message.reasoning_content,
        tool_calls: message.tool_calls,
        tool_call_id: message.tool_call_id
    )

    return ChatCompletionRequest(
        model: request.model,
        messages: messages,
        stream: request.stream,
        max_tokens: request.max_tokens,
        max_completion_tokens: request.max_completion_tokens,
        n: request.n,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        repetition_penalty: request.repetition_penalty,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        stream_options: request.stream_options,
        stop: request.stop,
        context_window: request.context_window,
        enable_thinking: false,
        mtp_enabled: request.mtp_enabled,
        kv_cache_bits: request.kv_cache_bits,
        kv_cache_group_size: request.kv_cache_group_size,
        quantized_kv_start: request.quantized_kv_start,
        ttl_seconds: request.ttl_seconds,
        chat_template: request.chat_template,
        chat_template_kwargs: request.chat_template_kwargs,
        tools: request.tools,
        tool_choice: request.tool_choice
    )
}

/// Converts a `ContinuousClock` span to fractional seconds.
private func durationSeconds(
    _ start: ContinuousClock.Instant,
    _ end: ContinuousClock.Instant
) -> Double {
    let elapsed = start.duration(to: end)
    return Double(elapsed.components.seconds)
        + Double(elapsed.components.attoseconds) / 1e18
}