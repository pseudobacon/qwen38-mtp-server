import XCTest
@testable import HTTPServer

/// Pure-Swift tests for Prompt 7: explicit KV-cache format resolution and
/// memory-aware admission control. No model weights are loaded.
final class KVCacheConfigTests: XCTestCase {

    func testDefaultConfigIsF16Shared() {
        let config = ResolvedKVCacheConfig.default
        XCTAssertEqual(config.kFormat, KVCacheFormat(kind: .f16))
        XCTAssertEqual(config.vFormat, KVCacheFormat(kind: .f16))
        XCTAssertEqual(config.groupSize, 64)
        XCTAssertEqual(config.quantizedKVStart, 0)
        // f16 is unquantized, so no shared `kvBits` value exists.
        XCTAssertNil(config.kvBits)
    }

    func testKVCacheFormatBitsParsing() {
        XCTAssertEqual(KVCacheFormat(bits: 16)?.kind, .f16)
        XCTAssertNil(KVCacheFormat(bits: 16)?.bits)
        XCTAssertEqual(KVCacheFormat(bits: 8)?.kind, .q8_0)
        XCTAssertEqual(KVCacheFormat(bits: 8)?.bits, 8)
        XCTAssertEqual(KVCacheFormat(bits: 4)?.kind, .q4_0)
        XCTAssertEqual(KVCacheFormat(bits: 4)?.bits, 4)
        XCTAssertNil(KVCacheFormat(bits: 7))
        XCTAssertNil(KVCacheFormat(bits: 0))
    }

    func testKVCacheFormatRawParsing() {
        XCTAssertEqual(KVCacheFormat(raw: "f16")?.kind, .f16)
        XCTAssertEqual(KVCacheFormat(raw: "FLOAT16")?.kind, .f16)
        XCTAssertEqual(KVCacheFormat(raw: "16")?.kind, .f16)
        XCTAssertEqual(KVCacheFormat(raw: "q8")?.kind, .q8_0)
        XCTAssertEqual(KVCacheFormat(raw: "q4_0")?.kind, .q4_0)
        XCTAssertEqual(KVCacheFormat(raw: "4")?.kind, .q4_0)
        XCTAssertEqual(KVCacheFormat(raw: "q2")?.kind, .q2_0)
        XCTAssertEqual(KVCacheFormat(raw: "q2_0")?.kind, .q2_0)
        XCTAssertEqual(KVCacheFormat(raw: "kvarn8")?.kind, .q8_0)
        XCTAssertEqual(KVCacheFormat(raw: "kvarn4")?.kind, .q4_0)
        XCTAssertEqual(KVCacheFormat(raw: "kvarn2")?.kind, .q2_0)
        XCTAssertEqual(KVCacheFormat(raw: "affine8")?.kind, .q8_0)
        XCTAssertEqual(KVCacheFormat(raw: "affine4")?.kind, .q4_0)
        XCTAssertEqual(KVCacheFormat(raw: "affine2")?.kind, .q2_0)
        XCTAssertNil(KVCacheFormat(raw: "bf16"))
        XCTAssertNil(KVCacheFormat(raw: ""))
    }

    func testResolveAcceptsF16Shared() throws {
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "f16",
            vType: "f16",
            groupSize: 64,
            quantizedKVStart: 0
        )
        XCTAssertEqual(config, ResolvedKVCacheConfig.default)
        XCTAssertNil(config.kvBits)
    }

    func testResolveAcceptsCanonicalRawSpellings() throws {
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "float16",
            vType: "16",
            groupSize: 64,
            quantizedKVStart: 0
        )
        XCTAssertEqual(config.kFormat.kind, .f16)
        XCTAssertEqual(config.vFormat.kind, .f16)
    }

    func testResolveRejectsUnsupportedKFormat() {
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "q8_0",
                vType: "f16",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unsupported K format")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_k")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveRejectsUnsupportedVFormat() {
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "f16",
                vType: "q8_0",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unsupported V format")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_v")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - KVCacheScheme.fromKVPair

    func testFromKVPairF16F16() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "f16", vType: "f16"), .fp16)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "F16", vType: "f16"), .fp16)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "float16", vType: "float16"), .fp16)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "16", vType: "16"), .fp16)
    }

    func testFromKVPairQ8Q8() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q8_0", vType: "q8_0"), .affine8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q8", vType: "q8"), .affine8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "8", vType: "8"), .affine8)
    }

    func testFromKVPairQ8Q4() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q8_0", vType: "q4_0"), .turbo8v4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q8", vType: "q4"), .turbo8v4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "8", vType: "4"), .turbo8v4)
    }

    func testFromKVPairQ4Q8() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4_0", vType: "q8_0"), .turbo4v8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4", vType: "q8"), .turbo4v8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "4", vType: "8"), .turbo4v8)
    }

    func testFromKVPairQ4Q4() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4_0", vType: "q4_0"), .affine4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4", vType: "q4"), .affine4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "4", vType: "4"), .affine4)
    }

    func testFromKVPairQ4Q2() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4_0", vType: "q2_0"), .kvarnK4V2G128)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "q4", vType: "q2"), .kvarnK4V2G128)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "4", vType: "2"), .kvarnK4V2G128)
    }

    func testFromKVPairUnmappedReturnsNil() {
        // f32 is accepted as a valid llama.cpp string but maps to no scheme.
        XCTAssertNil(KVCacheScheme.fromKVPair(kType: "f32", vType: "f32"))
        // q4_1 is accepted but maps to no scheme.
        XCTAssertNil(KVCacheScheme.fromKVPair(kType: "q4_1", vType: "q4_1"))
        // Mixed f16/q8_0 is not a known scheme.
        XCTAssertNil(KVCacheScheme.fromKVPair(kType: "f16", vType: "q8_0"))
        // Empty strings.
        XCTAssertNil(KVCacheScheme.fromKVPair(kType: "", vType: ""))
    }

    // MARK: - KVCacheScheme.fromKVPair (KVarN power-of-2 aliases)

    func testFromKVPairKvarn8() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn8", vType: "kvarn8"), .affine8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn8_8", vType: "kvarn8_8"), .affine8)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "KVARN8", vType: "kvarn8"), .affine8)
    }

    func testFromKVPairKvarn4() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn4", vType: "kvarn4"), .affine4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn4_4", vType: "kvarn4_4"), .affine4)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "KVARN4", vType: "kvarn4"), .affine4)
    }

    func testFromKVPairKvarn4Kvarn2() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn4", vType: "kvarn2"), .kvarnK4V2G128)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn4_2", vType: "kvarn4_2"), .kvarnK4V2G128)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "KVARN4", vType: "kvarn2"), .kvarnK4V2G128)
    }

    func testFromKVPairKvarn2() {
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn2", vType: "kvarn2"), .affine2)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "kvarn2_2", vType: "kvarn2_2"), .affine2)
        XCTAssertEqual(KVCacheScheme.fromKVPair(kType: "KVARN2", vType: "kvarn2"), .affine2)
    }

    // MARK: - KVCacheScheme affine2

    func testAffine2SchemeProperties() {
        let scheme = KVCacheScheme.affine2
        XCTAssertEqual(scheme.keyBits, 2)
        XCTAssertEqual(scheme.valueBits, 2)
        XCTAssertTrue(scheme.isQuantized)
        XCTAssertEqual(scheme.defaultGroupSize, 64)
        // Symmetric 2-bit scheme has a shared kvBits value.
        let resolved = ResolvedKVCacheConfig.resolve(scheme: .affine2)
        XCTAssertEqual(resolved.kvBits, 2)
    }

    func testAffine2RawParsing() {
        XCTAssertEqual(KVCacheScheme(raw: "affine2"), .affine2)
        XCTAssertEqual(KVCacheScheme(raw: "kvarn2"), .affine2)
        XCTAssertEqual(KVCacheScheme(raw: "fp2"), .affine2)
        XCTAssertEqual(KVCacheScheme(raw: "2"), .affine2)
        XCTAssertNil(KVCacheScheme(raw: "kvarn3"))
    }

    // MARK: - ResolvedKVCacheConfig.resolve (expanded quantized K/V support)

    func testResolveAcceptsQ8Q8() throws {
        // q8_0 K + q8_0 V: q8_0 is in kFormats but NOT in vFormats.
        // This combination is rejected because V does not support q8_0.
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "q8_0",
                vType: "q8_0",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError: q8_0 is not a supported V format")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_v")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveAcceptsQ4Q4() throws {
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "q4_0",
            vType: "q4_0",
            groupSize: 64,
            quantizedKVStart: 0
        )
        XCTAssertEqual(config.scheme, .affine4)
        XCTAssertEqual(config.kFormat.kind, .q4_0)
        XCTAssertEqual(config.vFormat.kind, .q4_0)
        XCTAssertEqual(config.kvBits, 4)
    }

    func testResolveRejectsQ2AsKFormat() {
        // q2_0 is not in kFormats (only f16, q8_0, q4_0).
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "q2_0",
                vType: "q2_0",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError: q2_0 is not a supported K format")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_k")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveAcceptsQ4Q2() throws {
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "q4_0",
            vType: "q2_0",
            groupSize: 64,
            quantizedKVStart: 0
        )
        XCTAssertEqual(config.scheme, .kvarnK4V2G128)
        XCTAssertEqual(config.kFormat.kind, .q4_0)
        XCTAssertEqual(config.vFormat.kind, .q2_0)
        // Asymmetric scheme has no shared kvBits.
        XCTAssertNil(config.kvBits)
    }

    func testResolveAcceptsKvarnAliases() throws {
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "kvarn4",
            vType: "kvarn2",
            groupSize: 64,
            quantizedKVStart: 0
        )
        XCTAssertEqual(config.scheme, .kvarnK4V2G128)
    }

    func testResolveRejectsUnsupportedKFormatQ2() {
        // q2_0 is not in kFormats (only f16, q8_0, q4_0).
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "q2_0",
                vType: "f16",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unsupported K format q2_0")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_k")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveRejectsUnsupportedVFormatQ8() {
        // q8_0 is not in vFormats (only f16, q4_0, q2_0).
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "f16",
                vType: "q8_0",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unsupported V format q8_0")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_v")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveRejectsInvalidCombinationF16Q4() {
        // f16 K + q4_0 V: both formats are individually supported, but the
        // combination is not a known scheme.
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "f16",
                vType: "q4_0",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for invalid f16/q4_0 combination")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_k")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - MemoryAdmissionPolicy tail-size math

    func testKvCacheGrowthTailSizeAllFP16() {
        // tailSize >= total: all tokens are FP16, so the result equals the
        // original 2 × N × H × D × L × 2 formula.
        let policy = MemoryAdmissionPolicy(
            memoryLimitBytes: 1 << 40,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0,
            keyBits: 4,
            valueBits: 4,
            tailSize: 1024
        )
        // 150 tokens < 1024 tail, so all FP16:
        // 64 × 4 × 256 × 150 × 4 = 39_321_600
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 100, completionTokens: 50),
            39_321_600
        )
    }

    func testKvCacheGrowthTailSizePartialQuantized() {
        // tailSize = 100, total = 150: first 100 tokens FP16, last 50 quantized.
        let policy = MemoryAdmissionPolicy(
            memoryLimitBytes: 1 << 40,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0,
            keyBits: 4,
            valueBits: 4,
            tailSize: 100
        )
        // FP16 tail: 64 × 4 × 256 × 100 × 4 = 26_214_400
        // Quantized remainder: 64 × 4 × 256 × 50 × (4/8 + 4/8) = 64 × 4 × 256 × 50 × 1 = 3_276_800
        // Total: 26_214_400 + 3_276_800 = 29_491_200
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 100, completionTokens: 50),
            29_491_200
        )
    }

    func testKvCacheGrowthTailSizeZero() {
        // tailSize = 0: all tokens quantized.
        let policy = MemoryAdmissionPolicy(
            memoryLimitBytes: 1 << 40,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0,
            keyBits: 4,
            valueBits: 4,
            tailSize: 0
        )
        // 64 × 4 × 256 × 150 × (4/8 + 4/8) = 64 × 4 × 256 × 150 × 1 = 9_830_400
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 100, completionTokens: 50),
            9_830_400
        )
    }

    func testKvCacheGrowthTailSizeDefaultFP16() {
        // Default policy (f16/f16, tailSize=1024): all tokens FP16.
        let policy = MemoryAdmissionPolicy(
            memoryLimitBytes: 1 << 40,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        // 64 × 4 × 256 × 150 × 4 = 39_321_600
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 100, completionTokens: 50),
            39_321_600
        )
    }

    func testResolveRejectsUnknownFormatString() {
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "q2",
                vType: "f16",
                groupSize: 64,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unknown format string")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "cache_type_k")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveRejectsUnsupportedGroupSize() {
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "f16",
                vType: "f16",
                groupSize: 32,
                quantizedKVStart: 0
            )
            XCTFail("Expected KVCacheConfigError for unsupported group size")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "kv_cache_group_size")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testResolveRejectsUnsupportedQuantizedStart() {
        do {
            _ = try ResolvedKVCacheConfig.resolve(
                kType: "f16",
                vType: "f16",
                groupSize: 64,
                quantizedKVStart: 128
            )
            XCTFail("Expected KVCacheConfigError for unsupported quantized start")
        } catch let error as KVCacheConfigError {
            XCTAssertEqual(error.param, "quantized_kv_start")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testKVBitsOnlyForSharedQuantizedFormats() throws {
        // Custom support table where both K and V accept q8_0.
        let sharedQ8 = ResolvedKVCacheConfig.RuntimeSupport(
            kFormats: [.q8_0],
            vFormats: [.q8_0],
            groupSizes: [64],
            quantizedKVStarts: [0]
        )
        let config = try ResolvedKVCacheConfig.resolve(
            kType: "q8_0",
            vType: "q8_0",
            groupSize: 64,
            quantizedKVStart: 0,
            support: sharedQ8
        )
        XCTAssertEqual(config.kvBits, 8)

        // Mixed K/V quantized formats: no single shared `kvBits` value.
        let mixed = ResolvedKVCacheConfig.RuntimeSupport(
            kFormats: [.q8_0, .q4_0],
            vFormats: [.q8_0, .q4_0],
            groupSizes: [64],
            quantizedKVStarts: [0]
        )
        let mixedConfig = try ResolvedKVCacheConfig.resolve(
            kType: "q8_0",
            vType: "q4_0",
            groupSize: 64,
            quantizedKVStart: 0,
            support: mixed
        )
        XCTAssertNil(mixedConfig.kvBits)
    }
}

/// Pure-Swift tests for the pre-generation memory admission math and the
/// opt-in memory-recovery policy. No model weights are loaded.
final class MemoryAdmissionTests: XCTestCase {

    /// Small deterministic geometry so the expected byte counts are exact.
    private func smallPolicy(
        memoryLimitBytes: Int,
        systemSafetyReserveBytes: Int,
        modelBaselineBytes: Int,
        keyBits: Int = 16,
        valueBits: Int = 16
    ) -> MemoryAdmissionPolicy {
        MemoryAdmissionPolicy(
            memoryLimitBytes: memoryLimitBytes,
            systemSafetyReserveBytes: systemSafetyReserveBytes,
            modelBaselineBytes: modelBaselineBytes,
            numLayers: 2,
            numKVHeads: 1,
            headDim: 4,
            keyBits: keyBits,
            valueBits: valueBits
        )
    }

    func testKvBudgetBytes() {
        let policy = smallPolicy(
            memoryLimitBytes: 1_000,
            systemSafetyReserveBytes: 200,
            modelBaselineBytes: 300
        )
        XCTAssertEqual(policy.kvBudgetBytes, 500)

        // Budget floors at zero when limit < reserve + baseline.
        let starved = smallPolicy(
            memoryLimitBytes: 100,
            systemSafetyReserveBytes: 200,
            modelBaselineBytes: 300
        )
        XCTAssertEqual(starved.kvBudgetBytes, 0)
    }

    func testKvCacheGrowthFormula() {
        // 2 (K and V) x 2 layers x 1 head x 4 dim x 15 tokens x 2 bytes
        let policy = smallPolicy(
            memoryLimitBytes: 10_000,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 10, completionTokens: 5),
            480
        )
    }

    func testKvCacheGrowthDefaultGeometry() {
        let policy = MemoryAdmissionPolicy(
            memoryLimitBytes: 1 << 40,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        // 2 x 64 layers x 4 heads x 256 dim x 150 tokens x 2 bytes
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: 100, completionTokens: 50),
            39_321_600
        )
    }

    func testKvCacheGrowthClampsNegativeInputs() {
        let policy = smallPolicy(
            memoryLimitBytes: 10_000,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: -5, completionTokens: 10),
            policy.kvCacheGrowthBytes(promptTokens: 0, completionTokens: 10)
        )
        XCTAssertEqual(
            policy.kvCacheGrowthBytes(promptTokens: -1, completionTokens: -1),
            0
        )
    }

    func testCheckPassesUnderBudget() {
        let policy = smallPolicy(
            memoryLimitBytes: 10_000,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        XCTAssertNoThrow(
            try policy.check(promptTokens: 10, completionTokens: 5)
        )
    }

    func testCheckPassesAtExactBudgetBoundary() {
        // Growth of (10, 5) under the small geometry is exactly 480 bytes.
        let policy = smallPolicy(
            memoryLimitBytes: 480,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        XCTAssertNoThrow(
            try policy.check(promptTokens: 10, completionTokens: 5)
        )
    }

    func testCheckThrowsOverBudget() {
        let policy = smallPolicy(
            memoryLimitBytes: 100,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0
        )
        do {
            try policy.check(promptTokens: 10, completionTokens: 5)
            XCTFail("Expected AdmissionFailure for over-budget request")
        } catch let error as MemoryAdmissionPolicy.AdmissionFailure {
            XCTAssertEqual(error.estimatedBytes, 480)
            XCTAssertEqual(error.budgetBytes, 100)
            XCTAssertEqual(error.promptTokens, 10)
            XCTAssertEqual(error.completionTokens, 5)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testAdmissionFailureErrorDescription() {
        let failure = MemoryAdmissionPolicy.AdmissionFailure(
            estimatedBytes: 480,
            budgetBytes: 100,
            promptTokens: 10,
            completionTokens: 5
        )
        let description = failure.errorDescription ?? ""
        XCTAssertTrue(description.contains("480"))
        XCTAssertTrue(description.contains("100"))
        XCTAssertTrue(description.contains("10"))
        XCTAssertTrue(description.contains("5"))
    }

    func testRecoveryPolicyBypassedWhenDisabled() {
        let policy = MemoryRecoveryPolicy(enabled: false)
        XCTAssertFalse(
            policy.shouldRecover(
                reason: .failedRequest,
                activeBytes: 1 << 40,
                limitBytes: 1 << 30
            )
        )
        XCTAssertFalse(
            policy.shouldRecover(
                reason: .memoryPressure,
                activeBytes: 1 << 40,
                limitBytes: 1 << 30
            )
        )
    }

    func testRecoveryPolicyFailedRequestAlwaysRecovers() {
        let policy = MemoryRecoveryPolicy(enabled: true)
        XCTAssertTrue(
            policy.shouldRecover(
                reason: .failedRequest,
                activeBytes: 0,
                limitBytes: 1 << 30
            )
        )
    }

    func testRecoveryPolicyMemoryPressureThreshold() {
        let policy = MemoryRecoveryPolicy(
            enabled: true,
            pressureThresholdFraction: 0.9
        )
        let limit = 1_000
        // At or above the threshold: recover.
        XCTAssertTrue(
            policy.shouldRecover(
                reason: .memoryPressure,
                activeBytes: 900,
                limitBytes: limit
            )
        )
        // Below the threshold: no recovery.
        XCTAssertFalse(
            policy.shouldRecover(
                reason: .memoryPressure,
                activeBytes: 899,
                limitBytes: limit
            )
        )
        // Zero limit: never recover on pressure.
        XCTAssertFalse(
            policy.shouldRecover(
                reason: .memoryPressure,
                activeBytes: 1_000,
                limitBytes: 0
            )
        )
    }

    // MARK: - Transient prefill-buffer admission

    private func transientPolicy(
        memoryLimitBytes: Int,
        chunkedPrefillEnabled: Bool,
        numQueryHeads: Int = 24,
        chunkedTileSize: Int = 512,
        metalMaxBufferBytes: Int = 30_150_672_384
    ) -> MemoryAdmissionPolicy {
        MemoryAdmissionPolicy(
            memoryLimitBytes: memoryLimitBytes,
            systemSafetyReserveBytes: 0,
            modelBaselineBytes: 0,
            numQueryHeads: numQueryHeads,
            chunkedPrefillEnabled: chunkedPrefillEnabled,
            chunkedTileSize: chunkedTileSize,
            metalMaxBufferBytes: metalMaxBufferBytes
        )
    }

    func testTransientBufferDenseQuadratic() {
        let policy = transientPolicy(memoryLimitBytes: 1 << 40, chunkedPrefillEnabled: false)
        // nQHeads(24) x L x L x 2 (one layer's dense scores, bf16).
        XCTAssertEqual(policy.transientPrefillBufferBytes(promptTokens: 100), 24 * 100 * 100 * 2)
        XCTAssertEqual(
            policy.transientPrefillBufferBytes(promptTokens: 32768),
            24 * 32768 * 32768 * 2
        )
    }

    func testTransientBufferChunkedLinear() {
        let policy = transientPolicy(memoryLimitBytes: 1 << 40, chunkedPrefillEnabled: true)
        // nQHeads(24) x tile(512) x L x 2 — linear in L.
        XCTAssertEqual(
            policy.transientPrefillBufferBytes(promptTokens: 32768),
            24 * 512 * 32768 * 2
        )
    }

    func testCheckRejectsOversizedDensePrefill() {
        // Dense 32K scores buffer is 51.6 GB > 30.15 GB Metal cap. The KV
        // growth (8.6 GB) fits the 1 TiB budget, so the *transient* gate trips.
        let policy = transientPolicy(memoryLimitBytes: 1 << 40, chunkedPrefillEnabled: false)
        do {
            try policy.check(promptTokens: 32768, completionTokens: 128)
            XCTFail("Expected TransientBufferFailure for dense 32K prefill")
        } catch let error as MemoryAdmissionPolicy.TransientBufferFailure {
            XCTAssertEqual(error.promptTokens, 32768)
            XCTAssertFalse(error.chunkedPrefillEnabled)
            XCTAssertGreaterThan(error.transientBytes, error.safeBytes)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testCheckAdmitsChunkedPrefillAt32K() {
        // Chunked 32K buffer is 0.8 GB, well under the cap. Same KV growth as
        // the dense case above, so only the transient gate differs.
        let policy = transientPolicy(memoryLimitBytes: 1 << 40, chunkedPrefillEnabled: true)
        XCTAssertNoThrow(try policy.check(promptTokens: 32768, completionTokens: 128))
    }
}