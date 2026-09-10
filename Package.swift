// swift-tools-version: 6.2
//
// qwen38-mtp-server: 3-layer Qwen 3.8 MTP server.
//
//   Layer 1: `../mlx-swift-lm` (local fork)  -> products MLXLLM, MLXLMCommon
//   Layer 2: `MLXFastModel`                  -> Metal runtime glue / model factory
//   Layer 3: `HTTPServer` (executable)       -> OpenAI-compatible API
//
// `mlx-swift` is pinned to the same range the fork uses (0.31.6) so SwiftPM
// resolves a single shared `MLX` across the fork and this package.
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
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "602.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/vapor/vapor", "4.102.1" ..< "5.0.0"),
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
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Vapor", package: "vapor"),
            ]
        ),
        .testTarget(
            name: "HTTPServerTests",
            dependencies: [
                "MLXFastModel",
                "HTTPServer",
                .product(name: "Vapor", package: "vapor"),
            ]
        ),
    ]
)