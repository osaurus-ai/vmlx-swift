import Foundation
import MLX
import MLXFast

/// Packed, expert-sorted prefill. The caller supplies rows in the projection's
/// basis; rotation is never inferred from a name or applied a second time.
/// NAX fuses gate/up, activation and optional H32 output rotation. Steel keeps
/// the same packed representation on older devices.
final class JANGHPrefillKernel {
    enum Backend: String { case nax, steel }
    let enableDenseSmallTile: Bool
    private let bits: Int
    private let upBits: Int
    private let codebookHeader: String
    private var kernels: [String: MLXFast.MLXFastKernel] = [:]
    private let lock = NSLock()
    private let rotation = JANGHRowRotation()

    init(
        contract: JANGHFormatContract, module: String, upModule: String? = nil,
        enableDenseSmallTile: Bool = false
    ) throws {
        self.enableDenseSmallTile = enableDenseSmallTile
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

    /// Largest row count routed to the 16-row NAX tile. `VMLX_JANGH_NAX_SMALL_TILE_MAX=0` keeps the 64-row tile.
    static let smallTileMaxRows: Int = {
        if let raw = RuntimeEnvironment.value("VMLX_JANGH_NAX_SMALL_TILE_MAX"), let v = Int(raw) {
            return v
        }
        return 16
    }()

    /// Dense verify-window split-K (`projectDenseSplitK`). `VMLX_JANGH_SPLITK=0` restores the 16-row tile.
    static let denseSplitKEnabled: Bool = RuntimeEnvironment.value("VMLX_JANGH_SPLITK") != "0"

    /// Local diagnostic for the portable verify-window kernel. Not a default.
    static let denseSteelSplitKEnabled: Bool = RuntimeEnvironment.value("VMLX_JANGH_STEEL_SPLITK") == "1"

    // Only immutable dense-bank owners opt in; routed/K2 callers retain their old tile.
    private let smallTileAdmission = JANGHDenseFastAdmission(enabled: true)

    static var nativeBackend: Backend {
        // A conservative diagnostic can disable NAX on this machine. It never
        // enables instructions unavailable on the actual device.
        if RuntimeEnvironment.value("VMLX_JANGH_PREFILL_BACKEND") == "steel" { return .steel }
        #if os(macOS)
            guard #available(macOS 26.2, *) else { return .steel }
            return backend(forMetalArchitecture: GPU.deviceInfo().architecture)
        #else
            return .steel
        #endif
    }

    static func backend(forMetalArchitecture architecture: String) -> Backend {
        // Metal uses ...g<generation><kind>; g is also the trailing kind
        // for base GPUs. Splitting on g discards that final kind.
        guard let kind = architecture.last, "gspd".contains(kind) else { return .steel }
        let stem = architecture.dropLast()
        guard let marker = stem.lastIndex(of: "g"),
            let generation = Int(stem[stem.index(after: marker)...])
        else {
            return .steel
        }
        return generation >= (kind == "p" ? 18 : 17) ? .nax : .steel
    }

    /// The native NAX fragment helper assumes relaxed-precision layout.
    /// F32 full-precision fragments have a different layout on the measured
    /// device; the strict-descriptor experiment failed parity. Until a separate
    /// F32 fragment implementation is proven, keep F32 on full-precision steel.
    static func resolvedBackend(dtype: DType, requested: Backend) -> Backend {
        dtype == .float32 ? .steel : requested
    }

    private func kernel(
        dtype: DType, backend: Backend, fused: Bool, rotate: Bool, width: Int, tileRows: Int = 64
    )
        -> MLXFast.MLXFastKernel
    {
        let type = dtype == .bfloat16 ? "bfloat16_t" : dtype == .float16 ? "half" : "float"
        let key =
            "\(backend.rawValue)_\(type)_\(width)_\(upBits)_\(fused)_\(rotate)"
            + (tileRows == 64 ? "" : "_m\(tileRows)")
        lock.lock()
        defer { lock.unlock() }
        if let value = kernels[key] { return value }
        let value: MLXFast.MLXFastKernel
        if backend == .nax {
            let source = """
                constexpr int PAD = 64 + 16 / sizeof(\(type));
                threadgroup \(type) Wg[64 * PAD];
                threadgroup \(type) Wu[\(fused ? 64 : 1) * PAD];
                tq_gather_qmm_nax<\(type), \(width), \(upBits), \(fused), \(rotate), \(tileRows)>(
                    x,wg,sg,wu,su,indices,y,meta[0],meta[1],meta[2],meta[3],lim[0],Wg,Wu,
                    threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            value = MLXFast.metalKernel(
                name: "jangh_prefill_" + key,
                inputNames: ["x", "wg", "sg", "wu", "su", "indices", "meta", "lim"],
                outputNames: ["y"], source: source,
                header: JANGHPrefillHeaders.nax + codebookHeader + JANGHPrefillSource.loader + JANGHPrefillSource.nax,
                ensureRowContiguous: false)
        } else {
            let source = """
                constexpr int PAD = 32 + 16 / sizeof(\(type));
                threadgroup \(type) Xs[16 * PAD];
                threadgroup \(type) Ws[32 * PAD];
                tq_gather_qmm_steel<\(type), \(width)>(x,wg,sg,indices,y,meta[0],meta[1],meta[2],meta[3],Xs,Ws,
                    threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            value = MLXFast.metalKernel(
                name: "jangh_prefill_" + key,
                inputNames: ["x", "wg", "sg", "indices", "meta"], outputNames: ["y"], source: source,
                header: JANGHPrefillHeaders.steel + codebookHeader + JANGHPrefillSource.loader + JANGHPrefillSource.steel,
                ensureRowContiguous: false)
        }
        kernels[key] = value
        return value
    }

    /// Dense verify windows (<= 16 rows, one bank, plain linear): split-K NAX tile (`JANGHPrefillSource.naxSplitK`).
    /// Returns nil when the shape is not eligible. Output dtype = input dtype.
    func projectDenseSplitK(_ input: MLXArray, packed: MLXArray, scales: MLXArray) -> MLXArray? {
        if Self.nativeBackend == .steel, Self.denseSteelSplitKEnabled {
            return projectDenseSteelSplitK(input, packed: packed, scales: scales)
        }
        guard input.ndim == 2, input.dim(0) >= 1, input.dim(0) <= 16, packed.ndim == 3, packed.dim(0) == 1,
            [.float16, .bfloat16].contains(input.dtype), Self.resolvedBackend(dtype: input.dtype, requested: Self.nativeBackend) == .nax
        else { return nil }
        let m = input.dim(0), k = input.dim(1), n = packed.dim(1)
        guard k.isMultiple(of: 64), n.isMultiple(of: 64) else { return nil }
        let tiles = n / 64
        let steps = k / 64
        var split = 1
        while split < 16 && tiles * split < 1024 && steps / (split * 2) >= 8 { split *= 2 }
        let type = input.dtype == .bfloat16 ? "bfloat16_t" : "half"
        let key = "splitk_\(type)_\(bits)"
        lock.lock()
        let kernel: MLXFast.MLXFastKernel
        if let cached = kernels[key] {
            kernel = cached
        } else {
            let source = """
                constexpr int PAD = 64 + 16 / sizeof(\(type));
                threadgroup \(type) Wg[64 * PAD];
                tq_dense_qmm_nax_splitk<\(type), \(bits), 16>(
                    x, wg, sg, yp, meta[0], meta[1], meta[2], meta[3], Wg,
                    threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                """
            kernel = MLXFast.metalKernel(
                name: "jangh_dense_" + key, inputNames: ["x", "wg", "sg", "meta"], outputNames: ["yp"],
                source: source,
                header: JANGHPrefillHeaders.nax + codebookHeader + JANGHPrefillSource.loader
                    + JANGHPrefillSource.naxSplitK,
                ensureRowContiguous: false)
            kernels[key] = kernel
        }
        lock.unlock()
        let metadata = MLXArray([Int32(m), Int32(n), Int32(k), Int32(split)])
        let partial = kernel(
            [contiguous(input), packed, scales, metadata],
            grid: (tiles * 64, (m + 15) / 16, split), threadGroup: (64, 1, 1),
            outputShapes: [[split, m, n]], outputDTypes: [.float32])[0]
        return (split == 1 ? partial[0] : partial.sum(axis: 0)).asType(input.dtype)
    }

    /// Packed single-bank diagnostic for small verify windows on pre-NAX
    /// devices. Partitions K across workgroups, reduces FP32 partials, and
    /// retains the original packed bank. No permanent decoded weight cache.
    func projectDenseSteelSplitK(_ input: MLXArray, packed: MLXArray, scales: MLXArray) -> MLXArray? {
        guard input.ndim == 2, (1...16).contains(input.dim(0)),
            [.float16, .bfloat16].contains(input.dtype), packed.ndim == 3,
            packed.dim(0) == 1, packed.dtype == .uint32, scales.dtype == .float16
        else { return nil }
        let m = input.dim(0), k = input.dim(1), n = packed.dim(1)
        guard k > 0, k.isMultiple(of: 64), n > 0,
            [m, n, k].allSatisfy({ $0 <= Int(Int32.max) }),
            packed.shape == [1, n, k / 32 * bits], scales.shape == [1, n]
        else { return nil }
        do {
            try JANGHBankLayout.requireReadyRowContiguous(packed, role: "Steel split-K packed")
            try JANGHBankLayout.requireReadyRowContiguous(scales, role: "Steel split-K scales")
        } catch { return nil }
        let tiles = (n + 31) / 32, steps = k / 32
        var split = 1
        while split < 16 && tiles * split < 1024 && steps / (split * 2) >= 8 { split *= 2 }
        let type = input.dtype == .bfloat16 ? "bfloat16_t" : "half"
        let key = "steel_splitk_\(type)_\(bits)"
        lock.lock()
        let built: MLXFast.MLXFastKernel
        if let cached = kernels[key] {
            built = cached
        } else {
            let source = """
                constexpr int PAD = 32 + 16 / sizeof(\(type));
                threadgroup \(type) Xs[16 * PAD];
                threadgroup \(type) Ws[32 * PAD];
                tq_dense_qmm_steel_splitk<\(type), \(bits)>(
                    x,wg,sg,yp,meta[0],meta[1],meta[2],meta[3],Xs,Ws,
                    threadgroup_position_in_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
                """
            built = MLXFast.metalKernel(
                name: "jangh_dense_" + key, inputNames: ["x", "wg", "sg", "meta"], outputNames: ["yp"],
                source: source,
                header: JANGHPrefillHeaders.steel + codebookHeader + JANGHPrefillSource.loader
                    + JANGHPrefillSource.steelSplitK,
                ensureRowContiguous: false)
            kernels[key] = built
        }
        lock.unlock()
        // Kernel pointers are device pointers; pad genuinely tiny banks as in
        // projectSorted rather than letting MLX select constant pointers.
        func deviceArray(_ array: MLXArray) -> MLXArray {
            array.size < 8 ? concatenated([array.flattened(), MLXArray.zeros([8 - array.size], dtype: array.dtype)]) : array
        }
        let partial = built(
            [contiguous(input), deviceArray(packed), deviceArray(scales),
             MLXArray([Int32(m), Int32(n), Int32(k), Int32(split)])],
            grid: (tiles * 64, 1, split), threadGroup: (64, 1, 1),
            outputShapes: [[split, m, n]], outputDTypes: [.float32])[0]
        return (split == 1 ? partial[0] : partial.sum(axis: 0)).asType(input.dtype)
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
        func deviceArray(_ array: MLXArray) -> MLXArray {
            array.size < 8 ? concatenated([array.flattened(), MLXArray.zeros([8 - array.size], dtype: array.dtype)]) : array
        }
        if selected == .nax {
            let inputs = [
                x, deviceArray(packed), deviceArray(scales), deviceArray(upPacked ?? packed),
                deviceArray(upScales ?? scales), idx, metadata, MLXArray([limit ?? 0]),
            ]
            func run(_ tileRows: Int) -> MLXArray {
                let threads = tileRows == 16 ? 64 : 128
                return kernel(
                    dtype: input.dtype, backend: .nax, fused: upPacked != nil, rotate: rotateOutput,
                    width: bits, tileRows: tileRows)(
                        inputs, grid: (((n + 63) / 64) * threads, (m + tileRows - 1) / tileRows, 1),
                        threadGroup: (threads, 1, 1), outputShapes: [[m, n]],
                        outputDTypes: [input.dtype])[0]
            }
            // Verify windows (<= 16 rows) use the 16-row tile (2 simdgroups = 64 threads): same per-element MMA
            // sequence as the 64-row tile, so it is admitted per dense owner/input variant only if bitwise equal to it.
            if enableDenseSmallTile, experts == 1, m <= Self.smallTileMaxRows,
                smallTileAdmission.admits(
                    key:
                        "\(input.dtype)_\(m)_\(k)_\(n)_\(bits)_\(upBits)_\(upPacked != nil)_\(rotateOutput)_\(limit?.bitPattern ?? 0)",
                    compare: { (run(16), run(64)) })
            {
                return run(16)
            }
            return run(64)
        }
        func single(_ w: MLXArray, _ s: MLXArray, _ width: Int) -> MLXArray {
            return kernel(dtype: input.dtype, backend: .steel, fused: false, rotate: false, width: width)(
                [x, deviceArray(w), deviceArray(s), idx, metadata], grid: (((n + 31) / 32) * 64, (m + 15) / 16, 1), threadGroup: (64, 1, 1),
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
