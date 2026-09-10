import Vapor
import Logging

/// Minimal entry point for checkpoint 1: validates the 3-layer dependency
/// graph (local `../mlx-swift-lm` fork + `MLXFastModel` + `HTTPServer`) and
/// the `mlx-swift` 0.31.6 resolution. The full server wiring lands in a
/// later checkpoint.
@main
struct Qwen38MTPServer {
    static func main() async throws {
        var env = try Environment.detect()
        try LoggingSystem.bootstrap(from: &env)

        let app = try await Application.make(env)
        app.get("healthz") { _ in "ok" }
        try await app.execute()
    }
}