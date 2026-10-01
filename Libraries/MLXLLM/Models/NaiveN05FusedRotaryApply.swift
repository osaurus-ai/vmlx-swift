import MLX
import MLXLMCommon

/// One default-off decode dispatch for the existing, already-rounded phases.
/// Trigonometry, position construction, projections and cache mutation remain
/// owned by the reference graph. This kernel has no model/phase array capture.
enum NaiveN05FusedRotaryApply {
    static func requested(environment: [String: String]) -> Bool {
        environment["VMLX_NAIVE_FUSED_ROTARY_APPLY"] == "1"
    }

    /// Only the current attention Q/K and sparse-indexer Q/K roles. Metadata
    /// admission never evaluates an input or reads a GPU buffer on the host.
    static func admits(_ x: MLXArray, cosine: MLXArray, sine: MLXArray,
                       dimensions: Int) -> Bool {
        #if canImport(Metal)
            guard Device.defaultDevice().deviceType == .gpu,
                !CompiledDecodeTrace.isActive,
                dimensions == 64, x.ndim == 4, x.dtype == .bfloat16,
                x.dim(0) == 1, x.dim(2) == 1,
                cosine.shape == [1, 1, 1, 32], cosine.dtype == .bfloat16,
                sine.shape == cosine.shape, sine.dtype == cosine.dtype
            else { return false }
            let heads = x.dim(1)
            switch x.dim(3) {
            case 192: return heads == 4 || heads == 8 || heads == 64
            case 128: return heads == 1 || heads == 16
            default: return false
            }
        #else
            return false
        #endif
    }

    private static let kernel = MLXFast.metalKernel(
        name: "naive_n05_bf16_precise_phase_rotary_apply_v1",
        inputNames: ["x", "cosine", "sine"], outputNames: ["rotated"],
        source: """
            uint d = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            if (d >= D || h >= uint(x_shape[1])) return;
            uint row = h * D;
            if (d >= 64) {
                rotated[row + d] = x[row + d];
                return;
            }
            {
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            uint j = d % 32;
            T a = x[row + j];
            T b = x[row + 32 + j];
            T c = cosine[j];
            T s = sine[j];
            // Match the reference's T-typed binary primitives and their stores.
            // Each product is rounded to T before the separate add/subtract.
            if (d < 32) {
                T ac = T(a * c);
                T bs = T(b * s);
                rotated[row + d] = T(ac - bs);
            } else {
                T bc = T(b * c);
                T a_s = T(a * s);
                rotated[row + d] = T(bc + a_s);
            }
            }
            """, ensureRowContiguous: true)

    static func apply(_ x: MLXArray, cosine: MLXArray, sine: MLXArray,
                      dimensions: Int) -> MLXArray? {
        guard admits(x, cosine: cosine, sine: sine, dimensions: dimensions)
        else { return nil }
        // B=1, T=1, H<=64, D<=192 bounds every Int32 grid/template conversion
        // and every UInt32 shader index to at most 12,287. A strided view is
        // copied by the existing encoder only if not already row contiguous;
        // the lazy output owns its inputs and encoder-owned copy temporaries.
        return kernel([x, cosine, sine],
            template: [("T", DType.bfloat16), ("D", x.dim(3))],
            grid: (x.dim(3), x.dim(1), 1), threadGroup: (128, 1, 1),
            outputShapes: [x.shape], outputDTypes: [.bfloat16])[0]
    }
}
