// MemoryAdmission.swift
//
// Memory-aware admission control and the opt-in memory-recovery hook for
// the Qwen 3.8 MTP server.
//
// `MemoryAdmissionPolicy` computes the estimated KV-cache footprint of a
// request BEFORE any model execution and rejects requests that cannot fit
// in the configured memory budget. The 262144-token context window is a
// policy maximum, not a promise: admission bounds what is actually
// allocatable given model weights + safety reserve.
//
// `MemoryRecoveryPolicy` is the opt-in hook that may call
// `MLX.Memory.clearCache()`. It is triggered only by failed requests or
// measured memory-pressure events; it is completely bypassed on the normal
// fast-path request, which must never clear the global MLX cache.

import Foundation
import MLXLLM

/// Frozen Qwen 3.8 27B text-tower geometry, mirrored from
/// `MLXFastConstants` (64 layers on a 4-layer hybrid repeat; 4 KV heads;
/// head_dim 256). The admission bound deliberately counts every layer as a
/// KV-carrying layer: it is a conservative upper bound, not a measurement.
public enum Qwen38KVGeometry {
    public static let numLayers = MLXFastConstants.numHiddenLayers
    public static let numKeyValueHeads = 4
    public static let headDim = 256
}

/// Pre-generation memory admission.
///
/// Estimated footprint:
///
///     ModelBaseline + SafetyReserve
///         + 2 × N_layers × N_kv_heads × D_head × L_total × bytes_per_element
///
/// where `L_total = promptTokens + completionTokens` and the leading 2
/// covers K and V. The bound counts all `N_layers` as KV-carrying (only 16
/// of 64 actually are), so it over-estimates by design.
public struct MemoryAdmissionPolicy: Sendable {
    /// Total memory budget in bytes (`--memory-limit`).
    public let memoryLimitBytes: Int
    /// OS + runtime safety reserve in bytes.
    public let systemSafetyReserveBytes: Int
    /// Measured post-load model footprint in bytes (`Memory.activeMemory`
    /// right after the weights are materialized).
    public let modelBaselineBytes: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    /// Bit-width of the K projection (16 for f16).
    public let keyBits: Int
    /// Bit-width of the V projection (16 for f16).
    public let valueBits: Int
    /// Number of leading tokens kept in FP16 (unquantized) before the
    /// quantized KV cache begins. The active generation window stays in FP16
    /// so speculative verification is numerically exact.
    public let tailSize: Int

    public init(
        memoryLimitBytes: Int,
        systemSafetyReserveBytes: Int,
        modelBaselineBytes: Int,
        numLayers: Int = Qwen38KVGeometry.numLayers,
        numKVHeads: Int = Qwen38KVGeometry.numKeyValueHeads,
        headDim: Int = Qwen38KVGeometry.headDim,
        keyBits: Int = 16,
        valueBits: Int = 16,
        tailSize: Int = 1024
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.systemSafetyReserveBytes = systemSafetyReserveBytes
        self.modelBaselineBytes = modelBaselineBytes
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.tailSize = tailSize
    }

    /// KV-cache budget left for a request after the model baseline and the
    /// safety reserve.
    public var kvBudgetBytes: Int {
        max(0, memoryLimitBytes - systemSafetyReserveBytes - modelBaselineBytes)
    }

    /// Estimated KV-cache growth in bytes for a request of
    /// `promptTokens + completionTokens` total tokens.
    ///
    /// The first `tailSize` tokens are kept in FP16 (2 bytes per element for
    /// both K and V, i.e. 4 bytes total per element). The remaining tokens use
    /// the quantized bit rate: `N_layers × N_kv_heads × D_head × L ×
    /// (keyBits + valueBits) / 8`. For f16 (16/16) with `tailSize >= total`
    /// this reduces to the original `2 × N × H × D × L × 2` formula.
    public func kvCacheGrowthBytes(promptTokens: Int, completionTokens: Int) -> Int {
        let total = max(0, promptTokens) + max(0, completionTokens)
        let tail = min(tailSize, total)
        let remaining = total - tail
        // FP16 tail: 2 bytes (K) + 2 bytes (V) = 4 bytes per element.
        let tailBytes = Int(Double(numLayers) * Double(numKVHeads) * Double(headDim)
            * Double(tail) * 4.0)
        // Quantized remainder: (keyBits + valueBits) / 8 bytes per element.
        let quantBytes = Int(Double(numLayers) * Double(numKVHeads) * Double(headDim)
            * Double(remaining) * (Double(keyBits) / 8 + Double(valueBits) / 8))
        return tailBytes + quantBytes
    }

    /// Reject a request whose estimated KV growth exceeds the budget.
    public func check(promptTokens: Int, completionTokens: Int) throws {
        let growth = kvCacheGrowthBytes(
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )
        guard growth <= kvBudgetBytes else {
            throw AdmissionFailure(
                estimatedBytes: growth,
                budgetBytes: kvBudgetBytes,
                promptTokens: max(0, promptTokens),
                completionTokens: max(0, completionTokens)
            )
        }
    }

    /// A request that cannot fit in the memory budget.
    public struct AdmissionFailure: Error, LocalizedError {
        public let estimatedBytes: Int
        public let budgetBytes: Int
        public let promptTokens: Int
        public let completionTokens: Int

        public var errorDescription: String? {
            "Estimated KV cache \(estimatedBytes) bytes exceeds the "
                + "memory budget of \(budgetBytes) bytes "
                + "(prompt \(promptTokens) + completion \(completionTokens) tokens)."
        }
    }
}


/// Opt-in memory-recovery hook.
///
/// `MLX.Memory.clearCache()` is a measured recovery/eviction action, not a
/// per-request step. It is triggered only by failed requests or by a
/// measured memory-pressure event, and only when the operator enabled it.
/// Normal requests bypass it entirely.
public struct MemoryRecoveryPolicy: Sendable {
    public enum Reason: Sendable {
        case failedRequest
        case memoryPressure
    }

    /// Whether the recovery hook is enabled at all.
    public let enabled: Bool
    /// Fraction of the memory limit at which active memory counts as
    /// pressure (e.g. 0.9 = 90%).
    public let pressureThresholdFraction: Double

    public init(enabled: Bool, pressureThresholdFraction: Double = 0.9) {
        self.enabled = enabled
        self.pressureThresholdFraction = pressureThresholdFraction
    }

    /// Decide whether to run `MLX.Memory.clearCache()` after a request.
    public func shouldRecover(
        reason: Reason,
        activeBytes: Int,
        limitBytes: Int
    ) -> Bool {
        guard enabled else { return false }
        switch reason {
        case .failedRequest:
            return true
        case .memoryPressure:
            guard limitBytes > 0 else { return false }
            return Double(activeBytes) >= pressureThresholdFraction * Double(limitBytes)
        }
    }
}
