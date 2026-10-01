// swift-tools-version: 6.2
//
// qwen38-mtp-server: 3-layer Qwen 3.8 MTP server.
//
//   Layer 1: `../mlx-swift-lm` (local fork)  -> products MLXLLM, MLXLMCommon
//   Layer 2: `MLXFastModel`                  -> Metal runtime glue / model factory
//   Layer 3: `HTTPServer` (executable)       -> OpenAI-compatible API
//
// `mlx-swift` is pinned to the exact fork revision the engine uses (pseudobacon
// 472c262, metal custom-kernel dispatch fix) so SwiftPM resolves a single shared
// `MLX` across the fork and this package. The URL must match the engine's: the
// same package identity from two URLs makes SwiftPM keep the root's location with
// the engine's fork-only revision, which fails to check out ("unable to read tree").
import PackageDescription

let package = Package(
    name: "qwen38-mtp-server",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .executable(name: "qwen38-mtp-server", targets: ["HTTPServer"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift-lm"),
        // Same fork + exact pin as `../mlx-swift-lm` (pseudobacon 00f401ca:
        // dispatch fix + qmm_nax repair) so SwiftPM resolves ONE shared
        // `MLX` across the fork and this package. The URL must match the
        // engine's: the same package identity from two URLs makes SwiftPM
        // keep the root's location with the engine's fork-only revision,
        // which fails to check out ("unable to read tree").
        .package(url: "https://github.com/pseudobacon/mlx-swift", .revision("00f401ca8b290ddfa7a66c18009930664b244751")),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "602.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/vapor/vapor", "4.102.1" ..< "5.0.0"),
        // `#huggingFaceTokenizerLoader()` + the chat template (Tokenizers).
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(
            name: "MLXFastModel",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "SwiftSyntax", package: "swift-syntax"),
            ]
        ),
        .executableTarget(
            name: "HTTPServer",
            dependencies: [
                .target(name: "MLXFastModel"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Vapor", package: "vapor"),
            ]
        ),
        .testTarget(
            name: "HTTPServerTests",
            dependencies: [
                "MLXFastModel",
                "HTTPServer",
                .product(name: "Vapor", package: "vapor"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
    ]
)