import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Multi-row mxfp8 QMV for DeepSeek-V4.1 JANGHT bundles: 3...7 activation rows through an mxfp8 Linear with each
/// weight read and decoded once (port of vMLX Python `vmlx_engine/metal/fp8_rows_qmv.py`).
///
/// Why: a dSpark verify (and the draft's block) feeds a few token rows through every mxfp8 projection. MLX's mxfp8
/// `quantizedMM` serves M >= 2 with wider kernels whose cost grows with M although a 1-row mxfp8 QMV is
/// bandwidth-bound and the extra rows need no extra weight bytes (M5 Max, 40 layers, 6 rows: wo_b 3.05 -> 7.04 ms,
/// wq_b 2.48 -> 4.33 ms against 1 row; with this kernel 4.24 and 3.19).
///
/// How: MLX's `fp_qmv_fast_impl` (the M=1 kernel for K % 256 == 0) with every row in the same threadgroup: per
/// 256-value block each lane loads its 8 activations of every row, then for each of its output rows decodes the 8
/// E4M3 weights and the E8M0 group scale once (exact, as MLX's conversions) and runs MLX's per-row qdot sequence on
/// them, then `simd_sum`. Each row therefore equals the M=1 call bit for bit, so verify rows match 1-row decode on
/// these projections. Every (K, N, rows, dtype, batched) is admitted only after a bit-equality check against stock
/// M=1 calls on the real weights; a shape that does not match keeps `quantizedMM` (logged once).
/// `VMLX_FP8_ROWS_QMV=0` keeps the stock kernels.
enum DeepseekV41FP8Rows {
    static let requested: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_FP8_ROWS_QMV"]?.lowercased() ?? "1"
        return !["0", "false", "off", "no"].contains(raw)
    }()

    /// (output rows per simdgroup, simdgroups per threadgroup, x registers in the input dtype) per row count; none
    /// changes a row's arithmetic, only register pressure. At 2 rows MLX's wide QMV is as fast and at 8 rows every
    /// configuration measured slower (Python sweep), so those keep the stock kernel.
    static func configuration(rows: Int) -> (rps: Int, nsg: Int, nativeX: Bool)? {
        switch rows {
        case 3, 4: return (2, 2, false)
        case 5, 6, 7: return (2, 2, true)
        default: return nil
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var admitted: [String: Bool] = [:]

    /// `linear(x)`, through the multi-row kernel when `linear` is a bias-free mxfp8 `QuantizedLinear` and `x` holds
    /// an admitted number of rows.
    static func apply(_ linear: Linear, _ x: MLXArray) -> MLXArray {
        guard requested, let q = linear as? QuantizedLinear, q.mode == .mxfp8, q.bias == nil,
            let y = rows(x, weight: q.weight, scales: q.scales)
        else { return linear(x) }
        return y
    }

    /// `x` `[..., M, K]` through the mxfp8 weight `[N, K/4]` (scales `[N, K/32]`), or batched weights `[B, N, K/4]`
    /// with `x` `[B, M, K]` (the grouped wo_a): `[..., M, N]`, or nil when the call is not served here.
    static func rows(_ x: MLXArray, weight: MLXArray, scales: MLXArray) -> MLXArray? {
        guard requested, x.ndim >= 2,
            // METAL-ONLY: case 1. Guards the Metal kernel; other devices keep quantizedMM.
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let m = x.dim(-2)
        let k = x.dim(-1)
        guard let config = configuration(rows: m) else { return nil }
        let batched = weight.ndim == 3
        let lead = Array(x.shape.dropLast(2))
        let x3: MLXArray
        if batched {
            guard x.ndim == 3, x.dim(0) == weight.dim(0), scales.ndim == 3, scales.dim(0) == weight.dim(0)
            else { return nil }
            x3 = x
        } else {
            guard weight.ndim == 2, scales.ndim == 2, lead.reduce(1, *) == 1 else { return nil }
            x3 = x.reshaped(1, m, k)
        }
        guard [DType.bfloat16, .float16, .float32].contains(x3.dtype), weight.dtype == .uint32,
            scales.dtype == .uint8, k % 256 == 0, weight.dim(-1) * 4 == k, scales.dim(-1) * 32 == k,
            scales.dim(-2) == weight.dim(-2)
        else { return nil }
        let n = weight.dim(-2)
        let key = "\(k)x\(n)r\(m)\(x3.dtype)\(batched)"
        guard admit(key, x3: x3, weight: weight, scales: scales, batched: batched, config: config) else {
            return nil
        }
        let y = launch(x3, weight, scales, batched: batched, config: config)
        return batched ? y : y.reshaped(lead + [m, n])
    }

    private static func admit(
        _ key: String, x3: MLXArray, weight: MLXArray, scales: MLXArray, batched: Bool,
        config: (rps: Int, nsg: Int, nativeX: Bool)
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let verdict = admitted[key] { return verdict }
        // The check evaluates; a compiled trace never admits (it keeps the stock path until an eager call has).
        if CompiledDecodeTrace.isActive { return false }
        let probe = (MLXRandom.normal(x3.shape, key: MLXRandom.key(1234)) * 2).asType(x3.dtype)
        let got = launch(probe, weight, scales, batched: batched, config: config)
        var ok = true
        for r in 0 ..< x3.dim(1) {
            let ref = quantizedMM(
                probe[0..., r ..< r + 1, 0...], weight, scales: scales, biases: nil, transpose: true,
                groupSize: 32, bits: 8, mode: .mxfp8)
            if !arrayEqual(got[0..., r ..< r + 1, 0...], ref).item(Bool.self) {
                ok = false
                break
            }
        }
        admitted[key] = ok
        if !ok {
            NSLog("[DSV41 fp8 rows] \(key): not bit-equal to the stock M=1 kernel; keeping quantizedMM")
        }
        return ok
    }

    private static func launch(
        _ x3: MLXArray, _ weight: MLXArray, _ scales: MLXArray, batched: Bool,
        config: (rps: Int, nsg: Int, nativeX: Bool)
    ) -> MLXArray {
        let b = x3.dim(0)
        let m = x3.dim(1)
        let k = x3.dim(2)
        let n = weight.dim(-2)
        var (rps, nsg) = (config.rps, config.nsg)
        if n % (rps * nsg) != 0 { (rps, nsg) = (1, 1) }
        let nativeX = config.nativeX && x3.dtype != .float32
        return kernel(
            [x3, weight, scales],
            template: [
                ("T", x3.dtype), ("XT", nativeX ? x3.dtype : .float32), ("K", k), ("N", n), ("ROWS", m),
                ("WBATCH", batched), ("RPS", rps), ("NSG", nsg),
            ],
            grid: (32, nsg * (n / (rps * nsg)), b),
            threadGroup: (32, nsg, 1),
            outputShapes: [[b, m, n]],
            outputDTypes: [x3.dtype])[0]
    }

    /// One kernel; the template arguments specialize it per shape.
    // METAL-ONLY: case 2. Metal kernel; without a Metal device `rows` returns nil and the caller keeps quantizedMM.
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_fp8_rows_qmv", inputNames: ["x", "w", "scales"], outputNames: ["y"], source: source,
        header: header)

    private static let header = """
        inline float vmlx_e4m3_value(uint8_t b) {
          // MLX fp8_e4m3::operator float16_t, then float: exact
          uint16_t v = b & 127;
          uint16_t sign_bit = ((uint16_t)((b >> 7) & 1)) << 15;
          uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
          half converted = as_type<half>(u);
          return static_cast<float>(converted * 256.0);
        }
        inline float vmlx_e8m0_value(uint8_t b) {
          // MLX fp8_e8m0::operator float
          uint32_t out = (b == 0 ? 0x400000 : (static_cast<uint16_t>(b) << 23));
          return as_type<float>(out);
        }
        """

    private static let source = """
          constexpr int values_per_thread = 8;  // pack_factor (4) * packs_per_thread (2), 8-bit
          constexpr int block_size = values_per_thread * 32;
          constexpr int group_size = 32;
          const uint3 tid = threadgroup_position_in_grid;
          const uint simd_gid = simdgroup_index_in_threadgroup;
          const uint simd_lid = thread_index_in_simdgroup;
          const uint b = tid.z;

          const device uint8_t* ws = (const device uint8_t*)w + (WBATCH ? (size_t)b * N * K : 0);
          const device uint8_t* sc = scales + (WBATCH ? (size_t)b * N * (K / group_size) : 0);
          const int out_row = tid.y * (NSG * RPS) + simd_gid * RPS;
          ws += (size_t)out_row * K + simd_lid * values_per_thread;
          sc += (size_t)out_row * (K / group_size) + simd_lid / (group_size / values_per_thread);
          const device T* xs = x + (size_t)b * ROWS * K + simd_lid * values_per_thread;

          XT x_thread[ROWS][values_per_thread];
          float result[ROWS][RPS];
          for (int r = 0; r < ROWS; r++)
            for (int row = 0; row < RPS; row++) result[r][row] = 0.0f;

          for (int k = 0; k < K; k += block_size) {
            for (int r = 0; r < ROWS; r++) {
              for (int i = 0; i < values_per_thread; i++) x_thread[r][i] = xs[r * K + i];
            }
            for (int row = 0; row < RPS; row++) {
              const device uint8_t* wl = ws + row * K;
              float s = vmlx_e8m0_value(sc[row * (K / group_size)]);
              float wv[values_per_thread];
              for (int i = 0; i < values_per_thread; i++) wv[i] = vmlx_e4m3_value(wl[i]);
              for (int r = 0; r < ROWS; r++) {
                float accum = 0.0f;
                for (int i = 0; i < values_per_thread; i++) accum += float(x_thread[r][i]) * wv[i];
                result[r][row] += s * accum;
              }
            }
            ws += block_size;
            sc += block_size / group_size;
            xs += block_size;
          }
          for (int r = 0; r < ROWS; r++) {
            for (int row = 0; row < RPS; row++) {
              float v = simd_sum(result[r][row]);
              if (simd_lid == 0) {
                y[((size_t)b * ROWS + r) * N + out_row + row] = static_cast<T>(v);
              }
            }
          }
        """
}
