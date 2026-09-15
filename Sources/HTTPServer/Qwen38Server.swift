import Vapor
import Logging
import MLX
import MLXLLM

@main
struct QwenServer {
    static func main() async throws {
        let config = ServerConfig.fromCommandLine()

        // Set MLX cache limit
        MLX.Memory.cacheLimit = config.memoryLimitGB * 1024 * 1024 * 1024

        // 1. Detect environment and bootstrap logging
        // Vapor rejects unknown command-line flags, so hand it only the
        // arguments that are not server-specific flags.
        var env = try Environment.detect(arguments: ServerConfig.vaporArguments())
        try LoggingSystem.bootstrap(from: &env)

        // 2. Create the async Application instance
        let app = try await Application.make(env)

        // 3. Apply network configuration
        app.http.server.configuration.hostname = config.host
        app.http.server.configuration.port = config.port

        app.routes.defaultMaxBodySize = "64mb"

        // Structured startup log: configuration values only. No `print()`
        // on the server path. Built in separate statements so the compiler
        // can type-check each one.
        var startupLog = "Server configuration: host=\(config.host) "
            + "port=\(config.port)"
        startupLog += " model=\(config.model) mtpHead=\(config.mtpHead)"
        startupLog += " ctxSize=\(config.ctxSize) nPredict=\(config.nPredict)"
        startupLog += " temp=\(config.temp) topP=\(config.topP) topK=\(config.topK)"
        startupLog += " minP=\(config.minP) repeatPenalty=\(config.repeatPenalty)"
        startupLog += " presencePenalty=\(config.presencePenalty)"
        startupLog += " frequencyPenalty=\(config.frequencyPenalty)"
        startupLog += " specDraftNMax=\(config.specDraftNMax)"
        startupLog += " cacheTypeK=\(config.cacheTypeK) cacheTypeV=\(config.cacheTypeV)"
        startupLog += " memoryLimitGB=\(config.memoryLimitGB)"
        startupLog += " tokenizationCache=entries:\(config.tokenizationCacheMaxEntries)"
        startupLog += " bytes:\(config.tokenizationCacheMaxBytes) ttl:\(config.tokenizationCacheTTLSeconds)s"
        app.logger.info("\(startupLog)")

        // 4. Model-runtime state machine (actor-owned; never stored in Vapor
        //     application storage). Starts in `.initializing`.
        let runtimeState = ModelRuntimeState()

        // 4b. Validate the launch KV-cache configuration against the runtime
        //     support table. If the operator configured an unsupported K/V
        //     format (e.g. q8_0), fail startup loudly rather than rejecting
        //     every request with a 400.
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: config.cacheTypeK,
                vType: config.cacheTypeV,
                groupSize: 64,
                quantizedKVStart: 0
            )
        } catch let error as KVCacheConfigError {
            _ = await runtimeState.transition(to: .failed(error.message))
            app.logger.error("Launch KV-cache config invalid: \(error.message)")
            try? await app.asyncShutdown()
            return
        }

        // 4c. Instantiate the MLX generator actor with config. All blocking
        //     MLX work is confined to this actor; the HTTP layer only consumes
        //     the `AsyncStream` it yields. The generator transitions the state
        //     to `.warming` before the deterministic startup shape warm.
        //     On any load/warmup failure the state is marked `.failed` and the
        //     server keeps serving `/healthz` + `/readyz` (503) but rejects
        //     chat-completion requests.
        let generator: MLXGenerator?
        do {
            generator = try await MLXGenerator(
                modelPath: config.model,
                mtpHeadPath: config.mtpHead,
                maxDraftDepth: config.specDraftNMax,
                runtimeState: runtimeState,
                memoryLimitBytes: config.memoryLimitGB * 1024 * 1024 * 1024,
                systemSafetyReserveBytes: config.systemSafetyReserveGB * 1024 * 1024 * 1024,
                memoryRecoveryEnabled: config.memoryRecoveryEnabled,
                memoryPressureThreshold: config.memoryPressureThreshold,
                kvScheme: config.kvScheme,
                kvTailSize: config.kvTailSize,
                tokenizationCacheMaxEntries: config.tokenizationCacheMaxEntries,
                tokenizationCacheMaxBytes: config.tokenizationCacheMaxBytes,
                tokenizationCacheTTLSeconds: config.tokenizationCacheTTLSeconds
            )
            _ = await runtimeState.transition(to: .ready)
            app.logger.info("Model runtime is ready.")
        } catch {
            _ = await runtimeState.transition(to: .failed(error.localizedDescription))
            app.logger.error("Model startup failed: \(error)")
            generator = nil
        }

        // 4c. Instantiate the single-lane generation scheduler. It is a
        //     long-lived actor that admits at most one active generation and
        //     bounds the FIFO queue to `config.maxQueueDepth`. Overload is
        //     rejected with HTTP 429 `engine_overloaded`.
        let scheduler = generator.map {
            GenerationScheduler(provider: $0, maxQueueSize: config.maxQueueDepth)
        }

        // 5a. Bounded, actor-isolated metrics store for per-request
        //     telemetry. Recording is a single actor hop and never blocks
        //     the model executor or the SSE stream.
        let metricsCollector = MetricsCollector()

        // 5b. Process-local, in-memory session store for the conversation API.
        //     Bounded by LRU cap and idle TTL; no disk persistence.
        let sessionStore = SessionStore(
            maxSessions: config.maxSessions,
            defaultTTL: TimeInterval(config.sessionTTLSeconds)
        )

        // 5. Store config for use in request handlers
        app.storage[ServerConfigKey.self] = config

        // 5b. Operational endpoints. `/healthz` is alive as long as the HTTP
        //     server runs; `/readyz` is 200 only when the runtime is `.ready`.
        app.get("healthz") { _ in healthzResponse() }
        app.get("readyz") { _ async throws -> Response in
            await readyzResponse(runtimeState)
        }

        // Dynamic /v1/models route driven by server configuration
        app.get("v1", "models") { req -> Response in
            let json = """
            {
              "object": "list",
              "data": [
                {
                  "id": "qwen3.8-27b-mtp",
                  "object": "model",
                  "created": 1700000000,
                  "owned_by": "local"
                }
              ]
            }
            """
            var headers = HTTPHeaders()
            headers.contentType = .json
            return Response(status: .ok, headers: headers, body: .init(string: json))
        }

        // 5c. Graceful shutdown: transition to `.draining` and let the active
        //     generation finish or hit its cooperative cancellation boundary.
        app.lifecycle.use(
            ModelShutdownHandler(runtimeState: runtimeState, scheduler: scheduler)
        )

        do {
            app.get { req in
                "Qwen 3.8 MTP Server is running\nConfig: \(config)"
            }

            registerOpenAIRoutes(
                on: app,
                scheduler: scheduler,
                runtimeState: runtimeState,
                generator: generator,
                metricsCollector: metricsCollector,
                sessionStore: sessionStore
            )

            try await app.execute()
        } catch {
            app.logger.report(error: error)
            try? await app.asyncShutdown()
            throw error
        }

        // 6. Clean up async resources on successful exit
        try await app.asyncShutdown()
    }
}

/// Vapor lifecycle handler that drives the model-runtime state machine into
/// `.draining` on application shutdown. It stops admitting new request
/// streams (the router rejects them with 503 once the state is not `.ready`)
/// and lets the active generation task finish or hit its cooperative
/// cancellation boundary before the process exits.
struct ModelShutdownHandler: LifecycleHandler {
    let runtimeState: ModelRuntimeState
    let scheduler: GenerationScheduler?

    func shutdownAsync(_ application: Application) async {
        // Item D: final cumulative QMV verify dispatch counters for the run
        // (shutdown-time output is allowed; per-call output is not). The
        // per-request step summary already carries the same counters.
        application.logger.info(
            "QMV verify dispatch (Item D): \(Qwen35QMVVerifyDispatch.summary())")
        _ = await runtimeState.transition(to: .draining)
        application.logger.info("Model runtime draining; waiting for active generation.")
        if let scheduler {
            await scheduler.drain(timeout: .seconds(30))
        }
    }
}

// Storage key for server config
struct ServerConfigKey: StorageKey {
    typealias Value = ServerConfig
}