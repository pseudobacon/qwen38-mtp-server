import XCTest
import MLX
import MLXFastModel

final class HTTPServerSmokeTests: XCTestCase {
    /// Verifies the `MLXFastModel` target builds against the resolved
    /// `mlx-swift` (0.31.6) and that `MLXFast.metalKernel` compiles and runs.
    func testMetalKernelProbeCompiles() throws {
        _ = try MLXFastProbe.buildProbeKernel()
    }
}
