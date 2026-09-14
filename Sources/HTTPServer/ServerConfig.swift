// ServerConfig.swift
//
// Launch-time server configuration (llama.cpp-style command-line flags),
// with environment-variable overrides for deployment. Generation
// parameters here are *defaults*: every one is overridable per-request
// through the OpenAI-compatible API.

import Foundation

struct ServerConfig: Sendable {
    // Network
    var host: String = "127.0.0.1"
    var port: Int = 8000

    // Model paths
    var model: String = "./weights"
    var mtpHead: String = "./mtp-head"

    /// The canonical ID of the single loaded model, as reported by
    /// `GET /v1/models`. Requests must name this ID or one of the
    /// configured aliases.
    static let canonicalModelID = "qwen3.8-27b-mtp"

    /// Additional model IDs accepted as aliases for the loaded model.
    /// Configured with `--model-aliases` (comma-separated) or
    /// `QWEN_MODEL_ALIASES`. Empty by default.
    var modelAliases: [String] = []

    // Context & Generation (llama.cpp names, act as server defaults)
    var ctxSize: Int = 262144          // --ctx-size, -c
    var nPredict: Int = 262144          // --n-predict, --predict, -n
    var temp: Float = 0.0              // --temp, --temperature
    var topK: Int = 20                  // --top-k
    var topP: Float = 0.95              // --top-p
    var minP: Float = 0.05              // --min-p
    var repeatPenalty: Float = 1.0     // --repeat-penalty
    var presencePenalty: Float = 0.0   // --presence-penalty
    var frequencyPenalty: Float = 0.0  // --frequency-penalty

    // MTP & KV Cache
    var specDraftNMax: Int = 3         // --spec-draft-n-max (0 = disabled)
    var cacheTypeK: String = "kvarn8"     // --cache-type-k, -ctk
    var cacheTypeV: String = "kvarn4"     // --cache-type-v, -ctv
    /// Maximum number of tokens in a single prefill forward pass. Long prompts
    /// are split into chunks of this size so the Metal GPU command buffer is
    /// not blocked by one giant prefill, keeping ITL predictable across
    /// concurrent streams. 0 disables chunking (single-pass prefill).
    var prefillChunkSize: Int = 512    // --prefill-chunk-size

    // Tool calling (server-wide gate; per-request `tools`/`tool_choice`
    // still control individual requests).
    var toolsEnabled: Bool = true      // --tools-enabled (default: true)

    // Memory
    var memoryLimitGB: Int = 44        // --memory-limit (MLX specific)
    var systemSafetyReserveGB: Int = 4 // OS + runtime safety reserve
    var memoryRecoveryEnabled: Bool = true   // opt-in Memory.clearCache() hook
    var memoryPressureThreshold: Double = 0.9 // fraction of limit = pressure

    // KV cache quantization scheme (unified; overrides cacheTypeK/cacheTypeV)
    var kvScheme: KVCacheScheme = .fp16   // --kv-scheme
    var kvGroupSize: Int? = nil            // --kv-group-size (nil = scheme default)
    var kvBits: Int? = nil                 // --kv-bits (symmetric override)
    /// Number of leading tokens kept in FP16 (unquantized) before the
    /// quantized KV cache begins. The active generation window stays in FP16
    /// so speculative verification is numerically exact.
    var kvTailSize: Int = 1024             // --kv-tail-size, --cache-type-v-tail

    // Tokenization cache (Stage 0 of the prefix-cache RFC). Bounded, actor-
    // isolated cache of prompt tokenization results. Stores only encoded token
    // IDs — no MLX/KV state. Safe KV reuse is out of scope (no copy-on-write).
    var tokenizationCacheMaxEntries: Int = 1024
    var tokenizationCacheMaxBytes: Int = 256 * 1024 * 1024
    var tokenizationCacheTTLSeconds: Int = 300

    // Scheduling
    /// Maximum number of chat-completion requests that may wait in the
    /// single-lane generation queue before a new request is rejected with
    /// HTTP 429 `engine_overloaded`.
    var maxQueueDepth: Int = 8         // --max-queue-depth

    /// Flags that consume a following value argument.
    private static let valueTakingFlags: Set<String> = [
        "--host", "-H",
        "--port", "-p",
        "--model", "-m",
        "--mtp-head",
        "--model-aliases",
        "--ctx-size", "-c",
        "--n-predict", "--predict", "-n",
        "--temp", "--temperature",
        "--top-k",
        "--top-p",
        "--min-p",
        "--repeat-penalty",
        "--presence-penalty",
        "--frequency-penalty",
        "--spec-draft-n-max",
        "--prefill-chunk-size",
        "--cache-type-k", "-ctk",
        "--cache-type-v", "-ctv",
        "--kv-scheme",
        "--kv-group-size",
        "--kv-bits",
        "--kv-tail-size", "--cache-type-v-tail",
        "--memory-limit",
        "--max-queue-depth",
    ]

    /// Flags that take no value.
    private static let valuelessFlags: Set<String> = [
        "--help",
        "--tools-enabled",
        "--tools-disabled",
    ]

    /// The subset of `CommandLine.arguments` that Vapor's
    /// `Environment.detect` accepts: the executable path plus any
    /// Vapor-native flags (e.g. `--env`). All server-specific flags and
    /// their values are stripped, because Vapor rejects unknown commands.
    static func vaporArguments() -> [String] {
        let args = CommandLine.arguments
        var kept: [String] = []
        var skipNext = false

        for (index, arg) in args.enumerated() {
            if index == 0 {
                kept.append(arg)
                continue
            }

            if skipNext {
                skipNext = false
                continue
            }

            if valueTakingFlags.contains(arg) {
                skipNext = true
                continue
            }

            if valuelessFlags.contains(arg) {
                continue
            }

            kept.append(arg)
        }

        return kept
    }

    static func fromCommandLine() -> ServerConfig {
        var config = ServerConfig()
        let args = CommandLine.arguments
        var kvSchemeExplicitlySet = false

        for (index, arg) in args.enumerated() {
            guard index > 0 else { continue }

            switch arg {
            case "--host", "-H":
                if index + 1 < args.count { config.host = args[index + 1] }
            case "--port", "-p":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.port = val }
            case "--model", "-m":
                if index + 1 < args.count { config.model = args[index + 1] }
            case "--mtp-head":
                if index + 1 < args.count { config.mtpHead = args[index + 1] }
            case "--model-aliases":
                if index + 1 < args.count {
                    config.modelAliases = parseCommaSeparatedList(args[index + 1])
                }
            case "--ctx-size", "-c":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.ctxSize = val }
            case "--n-predict", "--predict", "-n":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.nPredict = val }
            case "--temp", "--temperature":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.temp = val }
            case "--top-k":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.topK = val }
            case "--top-p":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.topP = val }
            case "--min-p":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.minP = val }
            case "--repeat-penalty":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.repeatPenalty = val }
            case "--presence-penalty":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.presencePenalty = val }
            case "--frequency-penalty":
                if index + 1 < args.count, let val = Float(args[index + 1]) { config.frequencyPenalty = val }
            case "--spec-draft-n-max":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.specDraftNMax = val }
            case "--prefill-chunk-size":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.prefillChunkSize = val }
            case "--cache-type-k", "-ctk":
                if index + 1 < args.count { config.cacheTypeK = args[index + 1] }
            case "--cache-type-v", "-ctv":
                if index + 1 < args.count { config.cacheTypeV = args[index + 1] }
            case "--kv-scheme":
                if index + 1 < args.count, let scheme = KVCacheScheme(raw: args[index + 1]) {
                    config.kvScheme = scheme
                    kvSchemeExplicitlySet = true
                }
            case "--kv-group-size":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.kvGroupSize = val }
            case "--kv-bits":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.kvBits = val }
            case "--kv-tail-size", "--cache-type-v-tail":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.kvTailSize = val }
            case "--memory-limit":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.memoryLimitGB = val }
            case "--max-queue-depth":
                if index + 1 < args.count, let val = Int(args[index + 1]) { config.maxQueueDepth = val }
            case "--tools-enabled":
                config.toolsEnabled = true
            case "--tools-disabled":
                config.toolsEnabled = false
            case "--help":
                printHelp()
                exit(0)
            default:
                break
            }
        }

        // Environment variable overrides (Docker friendly)
        if let val = ProcessInfo.processInfo.environment["QWEN_HOST"] { config.host = val }
        if let val = ProcessInfo.processInfo.environment["QWEN_PORT"], let intVal = Int(val) { config.port = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_MODEL"] { config.model = val }
        if let val = ProcessInfo.processInfo.environment["QWEN_MTP_HEAD"] { config.mtpHead = val }
        if let val = ProcessInfo.processInfo.environment["QWEN_MODEL_ALIASES"] {
            config.modelAliases = parseCommaSeparatedList(val)
        }
        if let val = ProcessInfo.processInfo.environment["QWEN_MEMORY_LIMIT_GB"], let intVal = Int(val) { config.memoryLimitGB = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_SYSTEM_SAFETY_RESERVE_GB"], let intVal = Int(val) { config.systemSafetyReserveGB = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_MEMORY_RECOVERY"], let boolVal = Bool(val) { config.memoryRecoveryEnabled = boolVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_MEMORY_PRESSURE_THRESHOLD"], let doubleVal = Double(val) { config.memoryPressureThreshold = doubleVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_MAX_QUEUE_DEPTH"], let intVal = Int(val) { config.maxQueueDepth = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_PREFILL_CHUNK_SIZE"], let intVal = Int(val) { config.prefillChunkSize = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_TOKENIZATION_CACHE_MAX_ENTRIES"], let intVal = Int(val) { config.tokenizationCacheMaxEntries = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_TOKENIZATION_CACHE_MAX_BYTES"], let intVal = Int(val) { config.tokenizationCacheMaxBytes = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_TOKENIZATION_CACHE_TTL_SECONDS"], let intVal = Int(val) { config.tokenizationCacheTTLSeconds = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_KV_SCHEME"], let scheme = KVCacheScheme(raw: val) { config.kvScheme = scheme }
        if let val = ProcessInfo.processInfo.environment["QWEN_KV_GROUP_SIZE"], let intVal = Int(val) { config.kvGroupSize = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_KV_BITS"], let intVal = Int(val) { config.kvBits = intVal }
        if let val = ProcessInfo.processInfo.environment["QWEN_KV_TAIL_SIZE"], let intVal = Int(val) { config.kvTailSize = intVal }
        if let val = ProcessInfo.processInfo.environment["LLAMA_ARG_CACHE_TYPE_K"] { config.cacheTypeK = val }
        if let val = ProcessInfo.processInfo.environment["LLAMA_ARG_CACHE_TYPE_V"] { config.cacheTypeV = val }

        // Derive kvScheme from the K/V cache-type pair when --kv-scheme was
        // not explicitly provided. The pair is always present (defaults to
        // "f16"/"f16"), so this is a no-op for the default configuration.
        if !kvSchemeExplicitlySet {
            if let derived = KVCacheScheme.fromKVPair(kType: config.cacheTypeK, vType: config.cacheTypeV) {
                config.kvScheme = derived
            }
        }

        return config
    }

    /// Splits a comma-separated list, trimming whitespace and dropping
    /// empty entries.
    static func parseCommaSeparatedList(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    // Helper to map llama.cpp cache types to MLX kvBits
    var kvCacheBits: Int? {
        let kType = cacheTypeK.lowercased()
        if kType == "q8_0" || kType == "q8" { return 8 }
        if kType == "q4_0" || kType == "q4" { return 4 }
        return nil // f16 or default
    }

    static func printHelp() {
        print("""
        Qwen 3.8 MTP Server - OpenAI-compatible API server with speculative decoding

        USAGE:
            HTTPServer [OPTIONS]

        SERVER OPTIONS:
            --host, -H <host>               Hostname to bind (default: 127.0.0.1)
            --port, -p <port>               Port to listen on (default: 8000)
            --help                          Show this help message

        MODEL OPTIONS:
            --model, -m <path>              Path to model weights (default: ./weights)
            --mtp-head <path>               Path to MTP head weights (default: ./mtp-head)
            --model-aliases <list>          Comma-separated model ID aliases for
                                            '\(ServerConfig.canonicalModelID)' (default: none)

        CONTEXT & GENERATION (Server defaults, overridable via API):
            --ctx-size, -c <int>            Context window size (default: 262144)
            --n-predict, --predict, -n <int> Max tokens to predict (default: 16384)
            --temp, --temperature <float>   Temperature (default: 0.0)
            --top-k <int>                   Top-k sampling (default: 0)
            --top-p <float>                 Top-p sampling (default: 1.0)
            --min-p <float>                 Min-p sampling (default: 0.0)
            --repeat-penalty <float>        Repeat penalty (default: 1.0)
            --presence-penalty <float>      Presence penalty (default: 0.0)
            --frequency-penalty <float>     Frequency penalty (default: 0.0)

        MTP & MEMORY OPTIONS:
            --spec-draft-n-max <int>        Max speculative draft depth (default: 8, 0 = disabled)
            --prefill-chunk-size <int>      Max tokens per prefill forward pass (default: 512,
                                            0 = single-pass prefill)
            --cache-type-k, -ctk <type>     KV cache K type: f16, f32, q8_0, q4_0, q4_1, q2_0,
                                             kvarn8, kvarn4, kvarn2 (default: f16)
            --cache-type-v, -ctv <type>     KV cache V type: f16, f32, q8_0, q4_0, q4_1, q2_0,
                                             kvarn8, kvarn4, kvarn2 (default: f16)
             --kv-scheme <scheme>            Unified KV cache scheme: fp16, affine8, affine4, affine2,
                                             turbo8v4, turbo4v8, turbo8v3, turbo8v2, kvarn_k4v2_g128
                                             (aliases: k8v4, affine8v4 -> turbo8v4; kvarn8, kvarn4, kvarn2;
                                             default: fp16)
                                             Overrides the scheme derived from -ctk/-ctv.
             --kv-group-size <int>           Quantization group size override (default: scheme's own)
             --kv-bits <int>                 Bit-width override for symmetric schemes (default: scheme's own)
             --kv-tail-size <int>            Leading tokens kept in FP16 before quantized KV begins
                                             (alias: --cache-type-v-tail; default: 1024)
            --memory-limit <int>            MLX cache limit in GB (default: 32)

         SCHEDULING OPTIONS:
             --max-queue-depth <int>         Max requests waiting in the single-lane
                                             generation queue before 429 rejection
                                             (default: 8)

         TOOL CALLING OPTIONS:
             --tools-enabled                 Enable tool calling (default: true)
             --tools-disabled                Disable tool calling server-wide


        ENVIRONMENT VARIABLES:
            QWEN_HOST, QWEN_PORT, QWEN_MODEL, QWEN_MTP_HEAD, QWEN_MODEL_ALIASES,
            QWEN_MEMORY_LIMIT_GB, QWEN_MAX_QUEUE_DEPTH, QWEN_PREFILL_CHUNK_SIZE,
            QWEN_KV_SCHEME, QWEN_KV_GROUP_SIZE, QWEN_KV_BITS, QWEN_KV_TAIL_SIZE,
            LLAMA_ARG_CACHE_TYPE_K, LLAMA_ARG_CACHE_TYPE_V,
            MLX_QWEN_MTP_HEAD_QUANT   (1/true/on force the 4-bit draft head
                                       <QWEN_MTP_HEAD>/q4; 0/false/off the
                                       pinned BF16 head; unset = default ON,
                                       falling back to BF16 with a loud log if
                                       the q4 tree is missing)

        EXAMPLES:
            HTTPServer --port 8080 --model /models/qwen-27b
            HTTPServer --temp 0.7 --top-p 0.9 --top-k 40
            HTTPServer --spec-draft-n-max 0  # Disable MTP
            HTTPServer --model-aliases qwen-27b,qwen3.8
        """)
    }
}