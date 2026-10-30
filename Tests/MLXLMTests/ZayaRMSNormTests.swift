import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Zaya RMSNorm")
struct ZayaRMSNormTests {
    @Test("Fused RMSNorm matches the fp32 reference formula")
    func fusedMatchesReference() {
        MLXMetalTestLock.withLock {
            let x = MLXArray([
                Float(0.25), -0.5, 1.25, 2.0,
                Float(-1.5), 0.75, 0.5, -0.25,
                Float(3.0), -2.0, 1.0, 0.125,
                Float(-0.75), -1.25, 2.5, -3.0,
            ]).reshaped(2, 2, 4).asType(.float16)

            let norm = ZayaRMSNorm(dimensions: 4, eps: 1e-6)
            let actual = norm(x)

            let xf = x.asType(.float32)
            let variance = (xf * xf).mean(axis: -1, keepDims: true)
            let reference = (xf * rsqrt(variance + 1e-6)).asType(x.dtype)

            let maxDelta = (actual.asType(.float32) - reference.asType(.float32))
                .abs().max().item(Float.self)
            #expect(maxDelta < 1e-3)
            #expect(actual.dtype == x.dtype)
            #expect(actual.shape == x.shape)
        }
    }

    @Test("Fused scaled L2 normalize matches the fp32 reference formula",
          arguments: [DType.float32, DType.float16])
    func scaledL2NormalizeMatchesReference(dtype: DType) {
        MLXMetalTestLock.withLock {
            let x = MLXArray([
                Float(0.25), -0.5, 1.25, 2.0,
                Float(-1.5), 0.75, 0.5, -0.25,
                Float(3.0), -2.0, 1.0, 0.125,
                Float(-0.75), -1.25, 2.5, -3.0,
            ]).reshaped(2, 2, 4).asType(dtype)
            let scale = Float(3.0)

            let actual = zayaScaledL2Normalize(x, scale: scale)

            let xf = x.asType(.float32)
            let norm = sqrt((xf * xf).sum(axis: -1, keepDims: true) + 1e-6)
            let reference = (xf * (scale / norm)).asType(x.dtype)

            let maxDelta = (actual.asType(.float32) - reference.asType(.float32))
                .abs().max().item(Float.self)
            #expect(maxDelta < (dtype == .float16 ? 2.5e-3 : 1e-5))
            #expect(actual.dtype == x.dtype)
            #expect(actual.shape == x.shape)
        }
    }
}

@Suite("Zaya ResScale")
struct ZayaResScaleTests {
    @Test("ZayaResScale affine merge matches reference for default scale/bias")
    func applyMatchesReferenceFormula() {
        MLXMetalTestLock.withLock {
            let layer = ZayaResScale()
            let hiddenStates = MLXArray([
                Float(1.0), -2.0, 3.0, -4.0
            ]).asType(.float32).reshaped(1, 1, 4)
            let residual = MLXArray([
                Float(0.25), -0.5, 1.5, -1.25
            ]).asType(.float32).reshaped(1, 1, 4)

            let scaled = layer.apply(residual: residual, hiddenStates: hiddenStates)
            let expectedResidual = residual
            let expectedHidden = hiddenStates

            #expect(
                (scaled.residual! - expectedResidual).asType(.float32).abs().max().item(Float.self)
                    < 1e-6)
            #expect(
                (scaled.hiddenStates - expectedHidden).asType(.float32).abs().max().item(Float.self)
                    < 1e-6)
        }
    }

}

@Suite("Zaya JANGTQ backend selection", .serialized)
struct ZayaJANGTQBackendSelectionTests {
    /// Streaming experts are retired as a production mechanism:
    /// `JANGTQStreamingExperts.explicitStreamingEnabled` is `false` whatever
    /// the environment says, so the legacy flag must leave Zaya's JANGTQ
    /// experts on `TurboQuantSwitchGLU`.
    @Test("Legacy streaming experts flag keeps TurboQuantSwitchGLU")
    func legacyStreamingExpertsFlagKeepsTurboQuantBackend() throws {
        // Building the experts is MLX work, so it holds the lock, as the
        // file's other suites do.
        try MLXMetalTestLock.withLock {
            let previous = getenv("MLXPRESS_STREAMING_EXPERTS").map { String(cString: $0) }
            setenv("MLXPRESS_STREAMING_EXPERTS", "1", 1)
            defer {
                if let previous {
                    setenv("MLXPRESS_STREAMING_EXPERTS", previous, 1)
                } else {
                    unsetenv("MLXPRESS_STREAMING_EXPERTS")
                }
            }

            let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: """
            {
              "model_type": "zaya",
              "hidden_size": 8,
              "ffn_hidden_size": 16,
              "num_experts": 2,
              "weight_format": "mxtq",
              "mxtq_bits": 2
            }
            """.data(using: .utf8)!)

            let experts = ZayaExperts(
                cfg.textConfig,
                context: .jangtq(gateUpBits: 2, downBits: 2, seed: 42),
                layerIdx: 1)

            #expect(experts.switchMLP is TurboQuantSwitchGLU)
            #expect(!(experts.switchMLP is StreamingTurboQuantSwitchGLU))
        }
    }
}
