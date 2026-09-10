import MLX

/// Compile-time probe that verifies the resolved `mlx-swift` (0.31.6) exposes
/// the inline Metal-kernel API the ported Qwen38 engine relies on:
/// `MLXFast.metalKernel(name:inputNames:outputNames:source:header:...) ->
/// MLXFast.MLXFastKernel`. Deliberately trivial: it must compile and construct.
public enum MLXFastProbe {
    public static func buildProbeKernel() throws -> MLXFast.MLXFastKernel {
        return try MLXFast.metalKernel(
            name: "qwen38_probe",
            inputNames: ["x"],
            outputNames: ["out"],
            source: "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + 1.0f;",
            ensureRowContiguous: false
        )
    }
}