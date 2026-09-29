import Foundation
import MLX
import MLXFast

/// Packed, expert-sorted prefill. The caller supplies rows in the projection's
/// basis; rotation is never inferred from a name or applied a second time.
/// NAX fuses gate/up, activation and optional H32 output rotation. Steel keeps
/// the same packed representation on older devices.
final class JANGHPrefillKernel {
    enum Backend: String { case nax, steel }
    private let bits: Int
    private let upBits: Int
    private let codebookHeader: String
    private var kernels: [String: MLXFast.MLXFastKernel] = [:]
    private let lock = NSLock()
    private let rotation = JANGHRowRotation()
    private static let offsetKernel = MLXFast.metalKernel(
        name: "jangh_prefill_sorted_offsets", inputNames: ["indices", "meta"],
        outputNames: ["offsets"], source: """
            const uint g = thread_position_in_grid.x;
            if (g > uint(meta[1])) return;
            int lo = 0, hi = meta[0];
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (indices[mid] < g) lo = mid + 1; else hi = mid;
            }
            offsets[g] = lo;
            """, ensureRowContiguous: false)


    init(contract: JANGHFormatContract, module: String, upModule: String? = nil) throws {
        guard let projection = contract.projections[module],
            upModule == nil || contract.projections[upModule!] != nil
        else { throw JANGHFormatContract.ValidationError.invalid("missing prefill projection") }
        bits = projection.bits
        upBits = upModule.flatMap { contract.projections[$0]?.bits } ?? bits
        var header = "template <int bits> METAL_FUNC float tq_level(uint q);\n"
        for width in Set([bits, upBits]).sorted() {
            guard let book = contract.codebooks[width] else {
                throw JANGHFormatContract.ValidationError.invalid("missing prefill codebook")
            }
            header += "template <> METAL_FUNC float tq_level<\(width)>(uint q) { "
            if width == 2 {
                let v = book.levels.map { "\(Float($0))f" }
                header += "float lo = (q & 1u) ? \(v[1]) : \(v[0]); "
                header += "float hi = (q & 1u) ? \(v[3]) : \(v[2]); return (q & 2u) ? hi : lo; }\n"
            } else {
                header += "float u = float(q) - \(Float((1 << width) - 1) / 2)f; "
                header += "return u * fma(\(Float(book.beta))f, u*u, \(Float(book.alpha))f); }\n"
            }
        }
        codebookHeader = header
    }

    static var nativeBackend: Backend {
        #if os(macOS)
            guard #available(macOS 26.2, *) else { return .steel }
            let architecture = GPU.deviceInfo().architecture
            let parts = architecture.split(separator: "g").last.map(String.init) ?? ""
            let digits = String(parts.prefix(while: { $0.isNumber }))
            guard let generation = Int(digits), let kind = parts.dropFirst(digits.count).first else {
                return .steel
            }
            return generation >= (kind == "p" ? 18 : 17) ? .nax : .steel
        #else
            return .steel
        #endif
    }

    /// The native NAX fragment helper assumes relaxed-precision layout.
    /// F32 full-precision fragments have a different layout on the measured
    /// device; the strict-descriptor experiment failed parity. Until a separate
    /// F32 fragment implementation is proven, keep F32 on full-precision steel.
    static func resolvedBackend(dtype: DType, requested: Backend) -> Backend {
        dtype == .float32 ? .steel : requested
    }

    private func kernel(dtype: DType, backend: Backend, fused: Bool, rotate: Bool, width: Int) -> MLXFast.MLXFastKernel {
        let type = dtype == .bfloat16 ? "bfloat16_t" : dtype == .float16 ? "half" : "float"
        let key = "\(backend.rawValue)_\(type)_\(width)_\(upBits)_\(fused)_\(rotate)"
        lock.lock()
        defer { lock.unlock() }
        if let value = kernels[key] { return value }
        let value: MLXFast.MLXFastKernel
        if backend == .nax {
            let source = """
                constexpr int PAD = 64 + 16 / sizeof(\(type));
                threadgroup \(type) Wg[64 * PAD];
                threadgroup \(type) Wu[\(fused ? 64 : 1) * PAD];
                tq_gather_qmm_nax<\(type), \(width), \(upBits), \(fused), \(rotate)>(
                    x,wg,sg,wu,su,offsets,y,meta[0],meta[1],meta[2],meta[3],lim[0],Wg,Wu,
                    threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            value = MLXFast.metalKernel(
                name: "jangh_prefill_" + key,
                inputNames: ["x", "wg", "sg", "wu", "su", "offsets", "meta", "lim"],
                outputNames: ["y"], source: source,
                header: JANGHPrefillHeaders.nax + codebookHeader + JANGHPrefillSource.loader + JANGHPrefillSource.nax,
                ensureRowContiguous: false)
        } else {
            let source = """
                constexpr int PAD = 32 + 16 / sizeof(\(type));
                threadgroup \(type) Xs[16 * PAD];
                threadgroup \(type) Ws[32 * PAD];
                tq_gather_qmm_steel<\(type), \(width)>(x,wg,sg,offsets,y,meta[0],meta[1],meta[2],meta[3],Xs,Ws,
                    threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            value = MLXFast.metalKernel(
                name: "jangh_prefill_" + key,
                inputNames: ["x", "wg", "sg", "offsets", "meta"], outputNames: ["y"], source: source,
                header: JANGHPrefillHeaders.steel + codebookHeader + JANGHPrefillSource.loader + JANGHPrefillSource.steel,
                ensureRowContiguous: false)
        }
        kernels[key] = value
        return value
    }

    func projectSorted(
        _ input: MLXArray, packed: MLXArray, scales: MLXArray, indices: MLXArray,
        upPacked: MLXArray? = nil, upScales: MLXArray? = nil,
        limit: Float? = nil, rotateOutput: Bool = false, backend: Backend? = nil
    ) throws -> MLXArray {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) > 0,
            input.dim(1).isMultiple(of: 64), [.float16, .bfloat16, .float32].contains(input.dtype),
            packed.ndim == 3, packed.dim(0) > 0, packed.dim(1) > 0,
            indices.ndim == 1, indices.size == input.dim(0), indices.dtype == .uint32,
            (upPacked == nil) == (upScales == nil),
            limit == nil || (limit!.isFinite && limit! > 0)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH prefill inputs") }
        let m = input.dim(0), k = input.dim(1), n = packed.dim(1), experts = packed.dim(0)
        guard [m, n, k, experts].allSatisfy({ $0 <= Int(Int32.max) }),
            experts < Int(Int32.max),
            !rotateOutput || n.isMultiple(of: 32)
        else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH prefill dimensions") }
        func validate(_ weights: MLXArray, _ scale: MLXArray, bits: Int) throws {
            guard weights.dtype == .uint32, weights.shape == [experts, n, k / 32 * bits],
                scale.dtype == .float16, scale.shape == [experts, n]
            else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH prefill bank") }
            try JANGHBankLayout.requireReadyRowContiguous(weights, role: "prefill packed")
            try JANGHBankLayout.requireReadyRowContiguous(scale, role: "prefill scales")
        }
        try validate(packed, scales, bits: bits)
        if let upPacked, let upScales { try validate(upPacked, upScales, bits: upBits) }
        let selected = Self.resolvedBackend(dtype: input.dtype, requested: backend ?? Self.nativeBackend)
        let x = contiguous(input)
        // MLX sends <8-element inputs through constant pointers, while the
        // tiled helper requires device pointers for its index vector.
        let idx = contiguous(indices.size < 8
            ? concatenated([indices, broadcast(indices[(indices.size - 1)...], to: [8 - indices.size])]) : indices)
        let metadata = MLXArray([Int32(m), Int32(n), Int32(k), Int32(experts)])
        // GPU lower bounds exactly match gather_mm_offsets. The extra group
        // collects invalid sorted IDs so existing fail-closed NaNs are preserved.
        let offsets = Self.offsetKernel(
            [idx, MLXArray([Int32(m), Int32(experts)])],
            grid: (experts + 1, 1, 1), threadGroup: (min(experts + 1, 256), 1, 1),
            outputShapes: [[max(8, experts + 1)]], outputDTypes: [.int32])[0]
        func deviceArray(_ array: MLXArray) -> MLXArray {
            array.size < 8 ? concatenated([array.flattened(), MLXArray.zeros([8 - array.size], dtype: array.dtype)]) : array
        }
        if selected == .nax {
            return kernel(dtype: input.dtype, backend: .nax, fused: upPacked != nil, rotate: rotateOutput, width: bits)(
                [x, deviceArray(packed), deviceArray(scales), deviceArray(upPacked ?? packed), deviceArray(upScales ?? scales), offsets, metadata, MLXArray([limit ?? 0])],
                grid: (((n + 63) / 64) * 128, min(m, (m + 63) / 64 + experts), 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [input.dtype])[0]
        }
        func single(_ w: MLXArray, _ s: MLXArray, _ width: Int) -> MLXArray {
            return kernel(dtype: input.dtype, backend: .steel, fused: false, rotate: false, width: width)(
                [x, deviceArray(w), deviceArray(s), offsets, metadata], grid: (((n + 31) / 32) * 64, min(m, (m + 15) / 16 + experts), 1), threadGroup: (64, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [input.dtype])[0]
        }
        var gate = single(packed, scales, bits)
        if let upPacked, let upScales {
            var up = single(upPacked, upScales, upBits).asType(.float32)
            gate = gate.asType(.float32)
            if let limit { gate = minimum(gate, limit); up = clip(up, min: -limit, max: limit) }
            gate = (gate * sigmoid(gate) * up).asType(input.dtype)
        }
        if rotateOutput {
            gate = try rotation(gate, outputDType: input.dtype)
        }
        return gate
    }
}
