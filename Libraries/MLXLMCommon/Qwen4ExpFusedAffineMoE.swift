import Foundation
import MLX
import MLXFast

/// Exact-shape Qwen4Exp/Ornith q2/q3/q4/q5/q6 affine routed-MoE decode kernels.
///
/// Gate and up share one packed-weight walk. The second kernel applies the
/// router scores while reducing the down projections directly into the
/// hidden vector, avoiding a materialized `[routes, hidden]` routed
/// output. Inputs and outputs are BF16; quantized dot products and reduction
/// accumulate in FP32 registers.
enum Qwen4ExpFusedAffineMoE {
    typealias Reducer = (MLXArray, MLXArray, MLXArray) -> MLXArray?

    private struct Shape: Equatable {
        let inputDimensions: Int
        let expertDimensions: Int
        let routes: Int
    }

    /// Deliberately exact, audited production contracts. Do not turn this into
    /// an arbitrary-shape fast path without adding parity and live rows.
    private static let qwen4ExpShape = Shape(
        inputDimensions: 2560, expertDimensions: 640, routes: 10)
    private static let ornith35Shape = Shape(
        inputDimensions: 2048, expertDimensions: 512, routes: 8)
    private static let supportedBits = Set([2, 3, 4, 5, 6])
    private static let supportedGroupSizes = Set([32, 64])
    /// Decode and native-MTP verification only. A single decode token is one
    /// row; native MTP verifies the current token plus at most three drafts.
    /// Larger prompt/prefill batches deliberately stay on the generic path.
    private static let maximumRows = 4

    private static let enabled: Bool = {
        let raw =
            ProcessInfo.processInfo.environment[
                "VMLINUX_QWEN4_EXP_FUSED_AFFINE_MOE"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()

    private static let pairKernel = MLXFast.metalKernel(
        name: "vmlx_qwen4_q4g64_pair_swiglu",
        inputNames: ["x", "gw", "gs", "gb", "uw", "us", "ub", "inds"],
        outputNames: ["act"],
        source: """
            uint tid = thread_position_in_grid.x;
            uint sgid = tid / 32u;
            uint lane = thread_index_in_simdgroup;
            uint m = sgid % EXPERT_DIM;
            uint route_row = sgid / EXPERT_DIM;
            uint k = route_row % ROUTES;
            uint row = route_row / ROUTES;
            uint e = uint(inds[row * ROUTES + k]);
            size_t rowoff = size_t(e) * EXPERT_DIM + m;
            constexpr uint G_VALUES_PER_PACK = (G_BITS == 3 || G_BITS == 5) ? 8u : (G_BITS == 6 ? 4u : 32u / G_BITS);
            constexpr uint U_VALUES_PER_PACK = (U_BITS == 3 || U_BITS == 5) ? 8u : (U_BITS == 6 ? 4u : 32u / U_BITS);
            constexpr uint G_BYTES_PER_PACK = G_BITS == 5 ? 5u : ((G_BITS == 3 || G_BITS == 6) ? 3u : 4u);
            constexpr uint U_BYTES_PER_PACK = U_BITS == 5 ? 5u : ((U_BITS == 3 || U_BITS == 6) ? 3u : 4u);
            constexpr uint G_PACKS_INPUT = INPUT_DIM / G_VALUES_PER_PACK;
            constexpr uint U_PACKS_INPUT = INPUT_DIM / U_VALUES_PER_PACK;
            constexpr uint G_PACKS_PER_GROUP = G_GROUP_SIZE / G_VALUES_PER_PACK;
            constexpr uint U_PACKS_PER_GROUP = U_GROUP_SIZE / U_VALUES_PER_PACK;
            constexpr ulong G_CODE_MASK = (ulong(1) << G_BITS) - ulong(1);
            constexpr ulong U_CODE_MASK = (ulong(1) << U_BITS) - ulong(1);
            const device uint8_t* grow = reinterpret_cast<const device uint8_t*>(gw)
                + rowoff * G_PACKS_INPUT * G_BYTES_PER_PACK;
            const device uint8_t* urow = reinterpret_cast<const device uint8_t*>(uw)
                + rowoff * U_PACKS_INPUT * U_BYTES_PER_PACK;
            size_t gsoff = rowoff * (INPUT_DIM / G_GROUP_SIZE);
            size_t usoff = rowoff * (INPUT_DIM / U_GROUP_SIZE);
            float gacc = 0.0f;
            float uacc = 0.0f;
            for (uint pack_idx = lane; pack_idx < G_PACKS_INPUT; pack_idx += 32u) {
              uint grp = pack_idx / G_PACKS_PER_GROUP;
              float gsc = float(gs[gsoff + grp]);
              float gbi = float(gb[gsoff + grp]);
              const device uint8_t* gp = grow + pack_idx * G_BYTES_PER_PACK;
              ulong gwrd;
              if (G_BITS == 5) {
                gwrd = ulong(gp[0]) | (ulong(gp[1]) << 8u) | (ulong(gp[2]) << 16u)
                    | (ulong(gp[3]) << 24u) | (ulong(gp[4]) << 32u);
              } else if (G_BITS == 3 || G_BITS == 6) {
                gwrd = ulong(gp[0]) | (ulong(gp[1]) << 8u) | (ulong(gp[2]) << 16u);
              } else {
                gwrd = ulong(*reinterpret_cast<const device uint32_t*>(gp));
              }
              uint xbase = pack_idx * G_VALUES_PER_PACK;
              float qsg = 0.0f;
              float xs = 0.0f;
              for (uint j = 0; j < G_VALUES_PER_PACK; ++j) {
                float xv = float(x[row * INPUT_DIM + xbase + j]);
                xs += xv;
                qsg += xv * float((gwrd >> (G_BITS * j)) & G_CODE_MASK);
              }
              gacc += gsc * qsg + gbi * xs;
            }
            for (uint pack_idx = lane; pack_idx < U_PACKS_INPUT; pack_idx += 32u) {
              uint grp = pack_idx / U_PACKS_PER_GROUP;
              float usc = float(us[usoff + grp]);
              float ubi = float(ub[usoff + grp]);
              const device uint8_t* up = urow + pack_idx * U_BYTES_PER_PACK;
              ulong uwrd;
              if (U_BITS == 5) {
                uwrd = ulong(up[0]) | (ulong(up[1]) << 8u) | (ulong(up[2]) << 16u)
                    | (ulong(up[3]) << 24u) | (ulong(up[4]) << 32u);
              } else if (U_BITS == 3 || U_BITS == 6) {
                uwrd = ulong(up[0]) | (ulong(up[1]) << 8u) | (ulong(up[2]) << 16u);
              } else {
                uwrd = ulong(*reinterpret_cast<const device uint32_t*>(up));
              }
              uint xbase = pack_idx * U_VALUES_PER_PACK;
              float qsu = 0.0f;
              float xs = 0.0f;
              for (uint j = 0; j < U_VALUES_PER_PACK; ++j) {
                float xv = float(x[row * INPUT_DIM + xbase + j]);
                xs += xv;
                qsu += xv * float((uwrd >> (U_BITS * j)) & U_CODE_MASK);
              }
              uacc += usc * qsu + ubi * xs;
            }
            gacc = simd_sum(gacc);
            uacc = simd_sum(uacc);
            if (lane == 0u) {
              float g = float(T(gacc));
              float u = float(T(uacc));
              float activated = g / (1.0f + metal::fast::exp(-g)) * u;
              act[(row * ROUTES + k) * EXPERT_DIM + m] = T(activated);
            }
            """,
        // Native MTP supplies lazy/sliced multi-row input, route, and score
        // views. The kernel indexes those arrays as flat row-major storage, so
        // accepting arbitrary strides silently reads the wrong tokens/routes.
        // MLX only materializes inputs that need it; dense single-row decode
        // remains the same allocation-free contract.
        ensureRowContiguous: true)

    private static let downKernel = MLXFast.metalKernel(
        name: "vmlx_qwen4_q4g64_weighted_down10",
        inputNames: ["act", "dw", "ds", "db", "inds", "scores"],
        outputNames: ["out"],
        source: """
            uint tid = thread_position_in_grid.x;
            uint sgid = tid / 32u;
            uint lane = thread_index_in_simdgroup;
            uint d = sgid % INPUT_DIM;
            uint row = sgid / INPUT_DIM;
            float weighted = 0.0f;
            for (uint k = 0; k < ROUTES; ++k) {
              uint e = uint(inds[row * ROUTES + k]);
              size_t rowoff = size_t(e) * INPUT_DIM + d;
              constexpr uint VALUES_PER_PACK = (BITS == 3 || BITS == 5) ? 8u : (BITS == 6 ? 4u : 32u / BITS);
              constexpr uint BYTES_PER_PACK = BITS == 5 ? 5u : ((BITS == 3 || BITS == 6) ? 3u : 4u);
              constexpr uint PACKS_HIDDEN = EXPERT_DIM / VALUES_PER_PACK;
              constexpr uint PACKS_PER_GROUP = GROUP_SIZE / VALUES_PER_PACK;
              constexpr ulong CODE_MASK = (ulong(1) << BITS) - ulong(1);
              const device uint8_t* dwrow = reinterpret_cast<const device uint8_t*>(dw)
                  + rowoff * PACKS_HIDDEN * BYTES_PER_PACK;
              size_t soff = rowoff * (EXPERT_DIM / GROUP_SIZE);
              float acc = 0.0f;
              for (uint pack_idx = lane; pack_idx < PACKS_HIDDEN; pack_idx += 32u) {
                uint grp = pack_idx / PACKS_PER_GROUP;
                float sc = float(ds[soff + grp]);
                float bi = float(db[soff + grp]);
                const device uint8_t* dp = dwrow + pack_idx * BYTES_PER_PACK;
                ulong wrd;
                if (BITS == 5) {
                  wrd = ulong(dp[0]) | (ulong(dp[1]) << 8u) | (ulong(dp[2]) << 16u)
                      | (ulong(dp[3]) << 24u) | (ulong(dp[4]) << 32u);
                } else if (BITS == 3 || BITS == 6) {
                  wrd = ulong(dp[0]) | (ulong(dp[1]) << 8u) | (ulong(dp[2]) << 16u);
                } else {
                  wrd = ulong(*reinterpret_cast<const device uint32_t*>(dp));
                }
                size_t abase = (size_t(row) * ROUTES + k) * EXPERT_DIM
                    + pack_idx * VALUES_PER_PACK;
                float qs = 0.0f;
                float xs = 0.0f;
                for (uint j = 0; j < VALUES_PER_PACK; ++j) {
                  float av = float(act[abase + j]);
                  xs += av;
                  qs += av * float((wrd >> (BITS * j)) & CODE_MASK);
                }
                acc += sc * qs + bi * xs;
              }
              acc = simd_sum(acc);
              weighted += float(scores[row * ROUTES + k]) * acc;
            }
            if (lane == 0u) out[row * INPUT_DIM + d] = T(weighted);
            """,
        // `act` is dense, but `indices` and `scores` may be strided views from
        // top-k selection during multi-token verification. This kernel also
        // indexes them as flat row-major storage.
        ensureRowContiguous: true)

    private static let reportLock = NSLock()
    nonisolated(unsafe) private static var reportedShapes = Set<String>()
    nonisolated(unsafe) private static var didReportConstructionRejection = false
    nonisolated(unsafe) private static var didReportInvocationRejection = false

    private static func reportConstructionRejection(
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear
    ) {
        reportLock.lock()
        defer { reportLock.unlock() }
        guard !didReportConstructionRejection else { return }
        didReportConstructionRejection = true
        FileHandle.standardError.write(
            Data(
                ("[Qwen4Exp] fused_affine_moe_decode=construction_rejected"
                    + " gate=\(gate.inputDims)x\(gate.outputDims):\(gate.bits)b:g\(gate.groupSize):\(gate.mode):\(gate.weight.dtype):\(gate.scales.dtype):\(String(describing: gate.biases?.dtype))"
                    + " up=\(up.inputDims)x\(up.outputDims):\(up.bits)b:g\(up.groupSize):\(up.mode):\(up.weight.dtype):\(up.scales.dtype):\(String(describing: up.biases?.dtype))"
                    + " down=\(down.inputDims)x\(down.outputDims):\(down.bits)b:g\(down.groupSize):\(down.mode):\(down.weight.dtype):\(down.scales.dtype):\(String(describing: down.biases?.dtype))\n")
                    .utf8))
    }

    private static func reportInvocationRejection(
        input: MLXArray, indices: MLXArray, scores: MLXArray
    ) {
        reportLock.lock()
        defer { reportLock.unlock() }
        guard !didReportInvocationRejection else { return }
        didReportInvocationRejection = true
        FileHandle.standardError.write(
            Data(
                ("[Qwen4Exp] fused_affine_moe_decode=invocation_rejected"
                    + " input=\(input.shape):\(input.dtype):size\(input.size)"
                    + " indices=\(indices.shape):\(indices.dtype):size\(indices.size)"
                    + " scores=\(scores.shape):\(scores.dtype):size\(scores.size)\n").utf8))
    }

    private static func reportActivation(shape: Shape, rows: Int) {
        reportLock.lock()
        defer { reportLock.unlock() }
        let label = "\(shape.inputDimensions)x\(shape.expertDimensions):topk\(shape.routes):rows\(rows)"
        guard reportedShapes.insert(label).inserted else { return }
        // Built piecewise on purpose, for the reason already recorded above the
        // `compiled_gdn_front=rejected` diagnostic in Qwen35.swift: as a single `+` chain with
        // interpolations wrapped in `Data(...).utf8`, this exceeds the type checker's budget and the
        // file does not compile on Swift 6.4 (swiftlang-6.4.0.33.1) — "unable to type-check this
        // expression in reasonable time". Identical output.
        var diag = "[Qwen4Exp] fused_affine_moe_decode=active rows=\(rows)"
        diag += " shape=\(shape.inputDimensions)x\(shape.expertDimensions) topk=\(shape.routes)"
        diag += " mixed_q2_q3_q4_q5_q6_g32_g64 input=bfloat16 output=bfloat16"
        diag += " affine_metadata=bf16_or_f16 router_scores=bf16_or_f32"
        diag += " accumulator=float32"
        diag += " weighted_down=true\n"
        FileHandle.standardError.write(Data(diag.utf8))
    }

    private static func supports(_ projection: QuantizedSwitchLinear) -> Bool {
        // JANG_4M stores affine metadata as BF16 while JANG_1L stores the
        // same affine scale and zero-point payloads as F16. The Metal kernels
        // convert either storage type to FP32 registers before accumulation
        // and still write BF16 activations/results. Excluding F16 here only
        // forced the 1L bundle onto the slower generic routed path.
        let metadataDType = projection.scales.dtype
        return supportedBits.contains(projection.bits)
            && supportedGroupSizes.contains(projection.groupSize)
            && projection.mode == .affine
            && projection.weight.dtype == .uint32
            && (metadataDType == .bfloat16 || metadataDType == .float16)
            && projection.biases?.dtype == metadataDType
    }

    static func makeReducer(
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear
    ) -> Reducer? {
        let shape = Shape(
            inputDimensions: gate.inputDims,
            expertDimensions: gate.outputDims,
            routes: gate.inputDims == ornith35Shape.inputDimensions
                && gate.outputDims == ornith35Shape.expertDimensions
                ? ornith35Shape.routes : qwen4ExpShape.routes)
        guard enabled,
            shape == qwen4ExpShape || shape == ornith35Shape,
            up.inputDims == shape.inputDimensions,
            up.outputDims == shape.expertDimensions,
            down.inputDims == shape.expertDimensions,
            down.outputDims == shape.inputDimensions,
            supports(gate), supports(up), supports(down),
            let gateBiases = gate.biases, let upBiases = up.biases,
            let downBiases = down.biases
        else {
            reportConstructionRejection(gate: gate, up: up, down: down)
            return nil
        }

        let gateWeight = gate.weight
        let gateScales = gate.scales
        let upWeight = up.weight
        let upScales = up.scales
        let downWeight = down.weight
        let downScales = down.scales
        let gateUpBits = gate.bits
        let upBits = up.bits
        let downBits = down.bits
        let gateGroupSize = gate.groupSize
        let upGroupSize = up.groupSize
        let downGroupSize = down.groupSize

        return { input, indices, scores in
            // The 4M bundle quantizes its router and produces BF16 scores.
            // The 1L bundle keeps the router dense F32, so precise softmax
            // produces F32 scores. The reduction already converts scores to
            // FP32 registers; both are native inputs and neither changes the
            // BF16 activation/result contract.
            let rows = input.size / shape.inputDimensions
            let leadingShape = Array(input.shape.dropLast())
            guard input.dtype == .bfloat16,
                scores.dtype == .bfloat16 || scores.dtype == .float32,
                rows >= 1, rows <= maximumRows,
                input.size == rows * shape.inputDimensions,
                input.dim(-1) == shape.inputDimensions,
                indices.size == rows * shape.routes, indices.dim(-1) == shape.routes,
                scores.size == rows * shape.routes, scores.dim(-1) == shape.routes,
                Array(indices.shape.dropLast()) == leadingShape,
                Array(scores.shape.dropLast()) == leadingShape
            else {
                reportInvocationRejection(input: input, indices: indices, scores: scores)
                return nil
            }

            let pairShape =
                Array(input.shape.dropLast())
                + [shape.routes, shape.expertDimensions]
            let pair = pairKernel(
                [
                    input,
                    gateWeight, gateScales, gateBiases,
                    upWeight, upScales, upBiases,
                    indices,
                ],
                template: [
                    ("T", input.dtype),
                    ("G_BITS", gateUpBits),
                    ("U_BITS", upBits),
                    ("G_GROUP_SIZE", gateGroupSize),
                    ("U_GROUP_SIZE", upGroupSize),
                    ("INPUT_DIM", shape.inputDimensions),
                    ("EXPERT_DIM", shape.expertDimensions),
                    ("ROUTES", shape.routes),
                ],
                grid: (32 * rows * shape.expertDimensions * shape.routes, 1, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [pairShape],
                outputDTypes: [input.dtype])[0]
            let output = downKernel(
                [
                    pair,
                    downWeight, downScales, downBiases,
                    indices, scores,
                ],
                template: [
                    ("T", input.dtype),
                    ("BITS", downBits),
                    ("GROUP_SIZE", downGroupSize),
                    ("INPUT_DIM", shape.inputDimensions),
                    ("EXPERT_DIM", shape.expertDimensions),
                    ("ROUTES", shape.routes),
                ],
                grid: (32 * rows * shape.inputDimensions, 1, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [input.shape],
                outputDTypes: [input.dtype])[0]

            reportActivation(shape: shape, rows: rows)
            return output
        }
    }
}
