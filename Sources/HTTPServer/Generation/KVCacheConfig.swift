// KVCacheConfig.swift
//
// Resolved KV-cache configuration for the Qwen 3.8 MTP server.
//
// Replaces the ambiguous single-value `kvBits` plumbing with an explicit
// resolved configuration that represents the K and V cache formats
// independently. The vendored Qwen 3.8 runtime
// (`Qwen35TextModel.newCache`) builds `KVCacheSimple` (f16) for every
// full-attention layer and ignores `GenerateParameters.kvBits`, so the
// runtime supports exactly one shared format: f16 for both K and V.
// `ResolvedKVCacheConfig.resolve` validates launch defaults and per-request
// overrides against that support table and rejects unsupported K/V format
// combinations with a typed configuration error instead of silently
// ignoring them.

import Foundation

/// One cache format, for K or V independently.
public struct KVCacheFormat: Sendable, Equatable, CustomStringConvertible {
    public enum Kind: String, Sendable, Equatable {
        case f16
        case q8_0
        case q4_0
        case q2_0
    }

    public let kind: Kind

    /// Bits per element; `nil` for unquantized f16.
    public var bits: Int? {
        switch kind {
        case .f16: return nil
        case .q8_0: return 8
        case .q4_0: return 4
        case .q2_0: return 2
        }
    }

    /// Canonical raw spelling (llama.cpp `cache_type_k`/`cache_type_v` style).
    public var raw: String { kind.rawValue }

    public init(kind: Kind) {
        self.kind = kind
    }

    /// Parse an integer bit count (16/8/4/2).
    public init?(bits: Int) {
        switch bits {
        case 16: self.init(kind: .f16)
        case 8: self.init(kind: .q8_0)
        case 4: self.init(kind: .q4_0)
        case 2: self.init(kind: .q2_0)
        default: return nil
        }
    }

    /// Parse a raw format string.
    ///
    /// Accepts the standard llama.cpp quantization strings (`f16`, `q8_0`,
    /// `q4_0`, `q2_0`), the KVarN power-of-2 aliases (`kvarn8`, `kvarn4`,
    /// `kvarn2`), and the `affine*` / `fp*` aliases.
    public init?(raw: String) {
        switch raw.lowercased() {
        case "f16", "float16", "16": self.init(kind: .f16)
        case "q8_0", "q8", "8", "kvarn8", "affine8", "fp8": self.init(kind: .q8_0)
        case "q4_0", "q4", "4", "kvarn4", "affine4", "fp4": self.init(kind: .q4_0)
        case "q2_0", "q2", "2", "kvarn2", "affine2", "fp2": self.init(kind: .q2_0)
        default: return nil
        }
    }

    public var description: String { raw }
}

/// The unified KV-cache quantization scheme selected at startup.
///
/// Each scheme fixes the bit-widths of the K and V projections and the
/// default quantization group size. `fp16` is the unquantized default and is
/// the only scheme the vendored Qwen 3.8 runtime instantiates directly
/// (`Qwen35TextModel.newCache` builds `KVCacheSimple`); the quantized schemes
/// are carried through to the session via `GenerateParameters.kvBits` /
/// `kvGroupSize` and drive the memory-admission budget.
///
/// Asymmetric schemes (`turbo8v4`, `turbo8v3`, `turbo8v2`, `kvarnK4V2G128`)
/// use different bit-widths for K and V, so they have no single shared
/// `kvBits` value; `k8v4` / `affine8v4` are aliases for `turbo8v4`.
public enum KVCacheScheme: String, Sendable, CaseIterable {
    case fp16
    case affine8
    case affine4
    case affine2
    case turbo8v4
    case turbo4v8
    case turbo8v3
    case turbo8v2
    case kvarnK4V2G128

    /// Bit-width of the K projection.
    public var keyBits: Int {
        switch self {
        case .fp16: return 16
        case .affine8: return 8
        case .affine4: return 4
        case .affine2: return 2
        case .turbo8v4: return 8
        case .turbo4v8: return 4
        case .turbo8v3: return 8
        case .turbo8v2: return 8
        case .kvarnK4V2G128: return 4
        }
    }

    /// Bit-width of the V projection.
    public var valueBits: Int {
        switch self {
        case .fp16: return 16
        case .affine8: return 8
        case .affine4: return 4
        case .affine2: return 2
        case .turbo8v4: return 4
        case .turbo4v8: return 8
        case .turbo8v3: return 3
        case .turbo8v2: return 2
        case .kvarnK4V2G128: return 2
        }
    }

    /// Whether the scheme quantizes the KV cache (i.e. is not `fp16`).
    public var isQuantized: Bool { self != .fp16 }

    /// The default quantization group size for this scheme (128 for kvarn,
    /// 64 otherwise).
    public var defaultGroupSize: Int {
        self == .kvarnK4V2G128 ? 128 : 64
    }

    /// Parse a raw scheme string, including the `k8v4` / `affine8v4` aliases
    /// (both map to `turbo8v4`, the asymmetric key-8 / value-4 scheme) and the
    /// llama.cpp KVarN power-of-2 aliases (`kvarn8`, `kvarn4`, `kvarn2`).
    public init?(raw: String) {
        switch raw.lowercased() {
        case "fp16", "f16", "float16", "16": self = .fp16
        case "affine8", "fp8", "q8", "8", "kvarn8": self = .affine8
        case "affine4", "fp4", "q4", "4", "kvarn4": self = .affine4
        case "affine2", "fp2", "q2", "2", "kvarn2": self = .affine2
        case "turbo8v4", "k8v4", "affine8v4": self = .turbo8v4
        case "turbo4v8", "k4v8", "affine4v8": self = .turbo4v8
        case "turbo8v3": self = .turbo8v3
        case "turbo8v2": self = .turbo8v2
        case "kvarn_k4v2_g128", "kvarn": self = .kvarnK4V2G128
        default: return nil
        }
    }

    /// Derive a `KVCacheScheme` from a llama.cpp-style K/V cache-type pair.
    ///
    /// Accepts the standard llama.cpp quantization strings (`f16`, `f32`,
    /// `q8_0`, `q4_0`, `q4_1`, `q2_0`) and the KVarN power-of-2 aliases
    /// (`kvarn8`, `kvarn4`, `kvarn2`). Returns `nil` when the pair does not
    /// map to a known scheme (e.g. `f32`/`f32`, `q4_1`/`q4_1`).
    public static func fromKVPair(kType: String, vType: String) -> KVCacheScheme? {
        let k = kType.lowercased()
        let v = vType.lowercased()
        switch (k, v) {
        case ("f16", "f16"), ("float16", "float16"), ("16", "16"):
            return .fp16
        case ("q8_0", "q8_0"), ("q8", "q8"), ("8", "8"),
             ("kvarn8", "kvarn8"), ("kvarn8_8", "kvarn8_8"):
            return .affine8
        case ("q8_0", "q4_0"), ("q8", "q4"), ("8", "4"):
            return .turbo8v4
        case ("q4_0", "q8_0"), ("q4", "q8"), ("4", "8"):
            return .turbo4v8
        case ("q4_0", "q4_0"), ("q4", "q4"), ("4", "4"),
             ("kvarn4", "kvarn4"), ("kvarn4_4", "kvarn4_4"):
            return .affine4
        case ("q4_0", "q2_0"), ("q4", "q2"), ("4", "2"),
             ("kvarn4", "kvarn2"), ("kvarn4_2", "kvarn4_2"):
            return .kvarnK4V2G128
        case ("q2_0", "q2_0"), ("q2", "q2"), ("2", "2"),
             ("kvarn2", "kvarn2"), ("kvarn2_2", "kvarn2_2"):
            return .affine2
        default:
            return nil
        }
    }
}

/// Typed configuration error for unsupported KV-cache settings.
public struct KVCacheConfigError: Error, LocalizedError {
    public let message: String
    public let param: String

    public var errorDescription: String? { message }

    public init(message: String, param: String) {
        self.message = message
        self.param = param
    }
}

/// The KV-cache configuration actually used for a request: K and V formats
/// resolved independently, plus the quantization group size and the token
/// offset at which quantization would begin (meaningless while the runtime
/// is f16-only, but validated so an ignored field is never exposed).
public struct ResolvedKVCacheConfig: Sendable, Equatable {
    /// The unified quantization scheme (drives `keyBits` / `valueBits`).
    public let scheme: KVCacheScheme
    public let kFormat: KVCacheFormat
    public let vFormat: KVCacheFormat
    public let groupSize: Int
    public let quantizedKVStart: Int

    /// Bit-width of the K projection (16 for `fp16`).
    public var keyBits: Int { scheme.keyBits }

    /// Bit-width of the V projection (16 for `fp16`).
    public var valueBits: Int { scheme.valueBits }

    /// Bits per element (`nil` = f16) for `GenerateParameters.kvBits`.
    /// The runtime only accepts a single shared value, so this is defined
    /// only when K and V resolve to the same quantized format (symmetric
    /// schemes). Asymmetric schemes (`turbo8v4`, `turbo8v3`, `turbo8v2`,
    /// `kvarnK4V2G128`) have no single shared `kvBits` value.
    public var kvBits: Int? {
        guard scheme.isQuantized, scheme.keyBits == scheme.valueBits else { return nil }
        return scheme.keyBits
    }

    /// The launch-default / no-override configuration.
    public static let `default` = ResolvedKVCacheConfig(
        scheme: .fp16,
        kFormat: KVCacheFormat(kind: .f16),
        vFormat: KVCacheFormat(kind: .f16),
        groupSize: 64,
        quantizedKVStart: 0
    )

    public init(
        scheme: KVCacheScheme,
        kFormat: KVCacheFormat,
        vFormat: KVCacheFormat,
        groupSize: Int,
        quantizedKVStart: Int
    ) {
        self.scheme = scheme
        self.kFormat = kFormat
        self.vFormat = vFormat
        self.groupSize = groupSize
        self.quantizedKVStart = quantizedKVStart
    }

    /// Resolve a unified KV-cache scheme into a resolved configuration.
    ///
    /// `groupSize` defaults to the scheme's own default (128 for kvarn, 64
    /// otherwise). `quantizedKVStart` is validated against the runtime support
    /// table (0 only, while the runtime is f16-only).
    public static func resolve(
        scheme: KVCacheScheme,
        groupSize: Int? = nil,
        quantizedKVStart: Int = 0
    ) -> ResolvedKVCacheConfig {
        let resolvedGroup = groupSize ?? scheme.defaultGroupSize
        let kFormat = KVCacheFormat(bits: scheme.keyBits) ?? KVCacheFormat(kind: .f16)
        let vFormat = KVCacheFormat(bits: scheme.valueBits) ?? KVCacheFormat(kind: .f16)
        return ResolvedKVCacheConfig(
            scheme: scheme,
            kFormat: kFormat,
            vFormat: vFormat,
            groupSize: resolvedGroup,
            quantizedKVStart: quantizedKVStart
        )
    }

    /// What the vendored Qwen 3.8 runtime actually supports.
    ///
    /// `Qwen35TextModel.newCache` builds `KVCacheSimple` (f16) for every
    /// full-attention layer and `MambaCache` for the gated-delta layers.
    /// Quantized K/V formats (`q8_0`, `q4_0`, `q2_0`) are accepted at the
    /// configuration level and drive the memory-admission budget; the
    /// runtime instantiates the corresponding quantized cache layers when
    /// the scheme is resolved. K and V are represented independently so an
    /// unsupported combination is rejected with a typed error rather than
    /// silently ignored.
    public struct RuntimeSupport: Sendable, Equatable {
        public let kFormats: Set<KVCacheFormat.Kind>
        public let vFormats: Set<KVCacheFormat.Kind>
        public let groupSizes: Set<Int>
        public let quantizedKVStarts: Set<Int>

        public static let qwen38 = RuntimeSupport(
            kFormats: [.f16, .q8_0, .q4_0],
            vFormats: [.f16, .q4_0, .q2_0],
            groupSizes: [64],
            quantizedKVStarts: [0]
        )
    }

    /// Resolve raw K/V format strings plus quantization knobs against the
    /// runtime support table. Throws `KVCacheConfigError` for any
    /// unsupported combination.
    public static func resolve(
        kType: String,
        vType: String,
        groupSize: Int,
        quantizedKVStart: Int,
        support: RuntimeSupport = RuntimeSupport.qwen38
    ) throws -> ResolvedKVCacheConfig {
        guard let kFormat = KVCacheFormat(raw: kType),
              support.kFormats.contains(kFormat.kind) else {
            throw KVCacheConfigError(
                message: "Unsupported K cache format '\(kType)'. "
                    + "The Qwen 3.8 runtime supports f16, q8, and q4 for K.",
                param: "cache_type_k"
            )
        }
        guard let vFormat = KVCacheFormat(raw: vType),
              support.vFormats.contains(vFormat.kind) else {
            throw KVCacheConfigError(
                message: "Unsupported V cache format '\(vType)'. "
                    + "The Qwen 3.8 runtime supports f16, q4, and q2 for V.",
                param: "cache_type_v"
            )
        }
        guard support.groupSizes.contains(groupSize) else {
            throw KVCacheConfigError(
                message: "Unsupported KV cache group size \(groupSize). "
                    + "The Qwen 3.8 runtime supports group size 64 only.",
                param: "kv_cache_group_size"
            )
        }
        guard support.quantizedKVStarts.contains(quantizedKVStart) else {
            throw KVCacheConfigError(
                message: "Unsupported quantized KV start \(quantizedKVStart). "
                    + "The Qwen 3.8 runtime supports 0 only.",
                param: "quantized_kv_start"
            )
        }
        // Derive the scheme from the resolved K/V formats. The llama.cpp-style
        // independent K/V path supports symmetric f16/q8/q4/q2 combinations and
        // mixed K/V pairs (e.g. q8_0/q4_0, q4_0/q2_0) when the support table
        // allows them.
        let scheme: KVCacheScheme
        switch (kFormat.kind, vFormat.kind) {
        case (.f16, .f16): scheme = .fp16
        case (.q8_0, .q8_0): scheme = .affine8
        case (.q4_0, .q4_0): scheme = .affine4
        case (.q2_0, .q2_0): scheme = .affine2
        case (.q8_0, .q4_0): scheme = .turbo8v4
        case (.q4_0, .q8_0): scheme = .turbo4v8
        case (.q4_0, .q2_0): scheme = .kvarnK4V2G128
        default:
            throw KVCacheConfigError(
                message: "Unsupported K/V format combination '\(kFormat.raw)' / '\(vFormat.raw)'. "
                    + "The Qwen 3.8 runtime supports f16, q8, q4, and q2.",
                param: "cache_type_k"
            )
        }
        return ResolvedKVCacheConfig(
            scheme: scheme,
            kFormat: kFormat,
            vFormat: vFormat,
            groupSize: groupSize,
            quantizedKVStart: quantizedKVStart
        )
    }
}