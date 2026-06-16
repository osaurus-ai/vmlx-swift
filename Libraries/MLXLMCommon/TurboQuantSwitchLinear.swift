//
// TurboQuantSwitchLinear — drop-in replacement for `SwitchLinear` that uses
// the JANGTQ codebook+Hadamard Metal kernels instead of `gather_qmm`.
// Created by Jinho Jang (eric@jangq.ai).
//
// Storage:
//   - `packed`  : uint32, shape (n_experts, out_features, packed_in)
//                 — codebook indices, 16 vals × 2 bits per uint32
//   - `norms`   : fp16,   shape (n_experts, out_features)
//                 — per-row L2 norm
//   - `signs`   : fp32,   shape (in_features,)
//                 — Hadamard sign vector (loaded from sidecar)
//   - `codebook`: fp32,   shape (4,)  for 2-bit
//                 — Lloyd-Max centroids (loaded from sidecar)
//
// `signs` and `codebook` are NOT module parameters — they're cached at
// load time in `JANGTQRuntimeCache` so multiple layers with the same
// `in_features` share the same MLXArray.
//
// `forward(x, indices)` does:
//   1. Hadamard rotate `x` (with `signs`) → `x_rot`  [P3 multiblock]
//   2. ONE Metal dispatch for the weighted dot products through the
//      codebook lookup, exactly mirroring `gather_qmm` semantics.
//
// For SwiGLU MoE blocks (gate+up+down), the higher-level
// `TurboQuantSwitchGLU` chains three of these via the fused gate+up
// kernel and the gather kernel. See `TurboQuantSwitchGLU` below.
//

import Foundation
import MLX
import MLXNN

/// Backed by the JANGTQ codebook kernels. Single matmul per call; no fused
/// gate+up. Use `TurboQuantSwitchGLU` for the full SwiGLU path.
public class TurboQuantSwitchLinear: Module {
    @ParameterInfo(key: "tq_packed") public var packed: MLXArray
    @ParameterInfo(key: "tq_norms")  public var norms: MLXArray

    public let inFeatures: Int
    public let outFeatures: Int
    public let numExperts: Int
    public let bits: Int
    public let mxtqSeed: Int

    public init(
        inFeatures: Int, outFeatures: Int, numExperts: Int,
        bits: Int = 2, seed: Int = 42,
        useStreamingPlaceholders: Bool = false
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.numExperts = numExperts
        self.bits = bits
        self.mxtqSeed = seed
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Initialize with zeros — the loader will overwrite with real data.
        if useStreamingPlaceholders || JANGTQStreamingExperts.isEnabled {
            self._packed.wrappedValue = MLXArray.zeros([1, 1, 1], dtype: .uint32)
            self._norms.wrappedValue  = MLXArray.zeros([1, 1], dtype: .float16)
        } else {
            self._packed.wrappedValue = MLXArray.zeros([numExperts, outFeatures, packedCols], dtype: .uint32)
            self._norms.wrappedValue  = MLXArray.zeros([numExperts, outFeatures], dtype: .float16)
        }
        super.init()
    }

    /// Single-matmul forward (gate-only or up-only or down-only). For the
    /// fused gate+up+SwiGLU + down path, use `TurboQuantSwitchGLU` which
    /// dispatches the two specialized kernels in one chain.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Look up or deterministically materialize signs + codebook from the runtime cache.
        let signs = JANGTQRuntimeCache.shared.requiredSigns(
            inFeatures: inFeatures, seed: mxtqSeed)
        let codebook = JANGTQRuntimeCache.shared.requiredCodebook(
            inFeatures: inFeatures, bits: bits)

        // Hadamard rotate input — accepts shape (..., in_features), returns fp32.
        let xRot = JANGTQKernels.hadamardRotate(x, signs: signs, dim: inFeatures)

        // Reshape to (batch, in_features) for the kernel.
        let batch = xRot.size / inFeatures
        let xFlat = xRot.reshaped([batch, inFeatures])

        // Number of expert slots K (last dim of indices)
        let K = indices.dim(-1)
        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        let y = JANGTQKernels.gatherTQTopK(
            xRot: xFlat, packed: packed, norms: norms,
            codebook: codebook, rhsIndices: idxFlat,
            batchTokens: batch, K: K,
            inFeatures: inFeatures, outFeatures: outFeatures, bits: bits
        )
        return y.reshaped(indices.shape + [outFeatures])
    }
}

/// Drop-in replacement for `SwitchGLU` that uses JANGTQ kernels for the
/// three projections. Mirrors the Python `_fused_switchglu_call` fast path
/// from `jang-tools/jang_tools/load_jangtq.py`.
public class TurboQuantSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: TurboQuantSwitchLinear
    @ModuleInfo(key: "up_proj")   public var upProj:   TurboQuantSwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: TurboQuantSwitchLinear

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    /// Legacy "all projections same bits" view. For uniform configs
    /// (JANGTQ2 = 2-bit everywhere, JANGTQ4 = 4-bit everywhere) this
    /// equals `gateUpBits == downBits`. For mixed-precision configs
    /// (JANGTQ_K, e.g. MiniMax-M2.7-JANGTQ_K with gate=2/up=2/down=4)
    /// this returns the gate+up width; callers needing the down width
    /// should read `downBits` directly.
    public var bits: Int { gateUpBits }
    /// Codebook bit width shared by gate_proj and up_proj. They MUST
    /// match because the fused `fusedGateUpSwiGLU` Metal kernel uses a
    /// single `bits` parameter for both. Bundles where gate ≠ up are
    /// not currently supported.
    public let gateUpBits: Int
    /// Codebook bit width for down_proj. Independent of gate/up. The
    /// `gatherTQ` Metal kernel takes this as its `bits` parameter on
    /// the down dispatch.
    public let downBits: Int
    public let mxtqSeed: Int
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// SwiGLU clamp magnitude. `0.0` (default) preserves ordinary SwiGLU
    /// `silu(gate) * up` for every non-DSV4 caller — output is bit-identical
    /// to the pre-2026-05-04 path. DSV4 sets this to `10.0` to activate the
    /// limited-SwiGLU expression `silu(min(gate, 10)) * clip(up, -10, 10)`
    /// the Python `_dsv4_swiglu` reference uses (mlx_model.py L1090) and
    /// the codex_dsv4_fixkit Python runtime patch installs.
    public let swigluLimit: Float

    /// Convenience constructor — every projection at the same bit
    /// width. Backwards-compatible with all existing JANGTQ2 / JANGTQ4
    /// callers (Qwen35JANGTQ, MiniMaxJANGTQ-2bit, DSV4JANGTQ uniform,
    /// NemotronH JANGTQ, etc.).
    public convenience init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        bits: Int = 2, seed: Int = 42,
        swigluLimit: Float = 0.0
    ) {
        self.init(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            gateUpBits: bits, downBits: bits, seed: seed, swigluLimit: swigluLimit)
    }

    /// Per-projection-bits constructor — for mixed-precision configs
    /// like JANGTQ_K (gate=2 / up=2 / down=4). gate and up MUST share
    /// a width because the fused gate+up Metal kernel takes a single
    /// `bits` parameter; bundles where gate ≠ up are rejected at
    /// model-load time.
    public init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        gateUpBits: Int, downBits: Int, seed: Int = 42,
        swigluLimit: Float = 0.0,
        useStreamingPlaceholders: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.gateUpBits = gateUpBits
        self.downBits = downBits
        self.mxtqSeed = seed
        self.swigluLimit = swigluLimit
        self._gateProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: gateUpBits, seed: seed,
            useStreamingPlaceholders: useStreamingPlaceholders
        )
        self._upProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: gateUpBits, seed: seed,
            useStreamingPlaceholders: useStreamingPlaceholders
        )
        self._downProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims, outFeatures: inputDims,
            numExperts: numExperts, bits: downBits, seed: seed,
            useStreamingPlaceholders: useStreamingPlaceholders
        )
        super.init()
    }

    /// Optional decode fast-path cache keyed by `(batchTokens, K, projection
    /// bits, swigluLimit)`. The compiled body runs the full rotate -> fused
    /// gate/up SwiGLU -> rotate -> down gather chain as one MLX graph.
    ///
    /// This remains opt-in because real MiniMax M2.7, Ling JANGTQ2, and DSV4
    /// runs on the local M5 Max match the Python `jang_tools` finding:
    /// compiling the whole SwitchGLU path can regress decode into the ~30 tok/s
    /// band or hurt the BatchEngine production path. The plain custom Metal
    /// kernel chain is the production default; set
    /// `VMLX_TQ_SWITCH_GLU_COMPILE=1` only for targeted experiments.
    private var compiledCache: [String: ([MLXArray]) -> [MLXArray]] = [:]

    private static let compiledFastPathExplicitOn: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_TQ_SWITCH_GLU_COMPILE"]?
            .lowercased()
        return raw == "1" || raw == "true" || raw == "on" || raw == "yes"
    }()

    private static let compiledFastPathExplicitOff: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_TQ_SWITCH_GLU_COMPILE"]?
            .lowercased()
        return raw == "0" || raw == "false" || raw == "off" || raw == "no"
    }()

    private static let compiledFastPathThreshold: Int = {
        let env = ProcessInfo.processInfo.environment
        return Int(env["VMLX_TQ_SWITCH_GLU_COMPILE_THRESHOLD"] ?? "128") ?? 128
    }()

    /// Forward through the JANGTQ MoE MLP fast path.
    /// `x` shape: `(batch, seq, hidden)`. `indices` shape: `(batch, seq, K)`.
    /// Returns `(batch, seq, K, hidden)` to match `SwitchGLU` semantics —
    /// caller multiplies by router scores and sums over the K dim.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Codebook lookup uses PER-PROJECTION bit widths so JANGTQ_K
        // (gate=2 / up=2 / down=4) loads the right table for each
        // dispatch. Uniform-bit bundles fall through with
        // gateUpBits == downBits, matching the legacy single-bits path.
        let signsIn = JANGTQRuntimeCache.shared.requiredSigns(
            inFeatures: inputDims, seed: mxtqSeed)
        let signsDn = JANGTQRuntimeCache.shared.requiredSigns(
            inFeatures: hiddenDims, seed: mxtqSeed)
        let cbGate = JANGTQRuntimeCache.shared.requiredCodebook(
            inFeatures: inputDims, bits: gateUpBits)
        let cbDown = JANGTQRuntimeCache.shared.requiredCodebook(
            inFeatures: hiddenDims, bits: downBits)

        // The decode broadcast pattern: x has shape (batch, seq, hidden),
        // indices has shape (batch, seq, K). Each token uses K experts.
        // We flatten (batch, seq) → 1 batch row for the kernel.
        let inputDims = self.inputDims
        let xSize = x.size
        let batchTokens = xSize / inputDims
        let xFlat = x.reshaped([batchTokens, inputDims])

        let K = indices.dim(-1)
        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        let useCompiledFastPath = HardwareInfo.isCompiledDecodeSupported
            && !Self.compiledFastPathExplicitOff
            && Self.compiledFastPathExplicitOn
            && Self.compiledFastPathThreshold > 0
            && indices.size <= Self.compiledFastPathThreshold
        if useCompiledFastPath {
            let limitKey = Int((swigluLimit * 1000).rounded())
            let cacheKey = "bt\(batchTokens).K\(K).gb\(gateUpBits).db\(downBits).lim\(limitKey)"
            if compiledCache[cacheKey] == nil {
                let inDim = self.inputDims
                let outDim = self.hiddenDims
                let gateBitsLocal = self.gateUpBits
                let downBitsLocal = self.downBits
                let bt = batchTokens
                let kLocal = K
                let swigluLimitLocal = self.swigluLimit
                let body: ([MLXArray]) -> [MLXArray] = { args in
                    let xR = JANGTQKernels.hadamardRotate(
                        args[0], signs: args[7], dim: inDim)
                    let xAct_ = JANGTQKernels.fusedGateUpSwiGLU(
                        xRot: xR,
                        packedGate: args[1], normsGate: args[2],
                        packedUp: args[3], normsUp: args[4],
                        codebook: args[9], rhsIndices: args[11],
                        batchTokens: bt, K: kLocal,
                        inFeatures: inDim, outFeatures: outDim,
                        bits: gateBitsLocal,
                        swigluLimit: swigluLimitLocal)
                    let xActR = JANGTQKernels.hadamardRotate(
                        xAct_, signs: args[8], dim: outDim)
                    let yLocal = JANGTQKernels.gatherTQ(
                        xRot: xActR,
                        packed: args[5], norms: args[6],
                        codebook: args[10], rhsIndices: args[11],
                        nRows: bt * kLocal,
                        inFeatures: outDim, outFeatures: inDim,
                        bits: downBitsLocal)
                    return [yLocal]
                }
                // Shape-specific compile is intentional here. MLX's
                // `CustomKernel` primitive in the current osaurus pin
                // does not implement `output_shapes`, so shapeless
                // compile traps when it tries to infer shapes for the
                // JANGTQ Metal kernels. We key by exact decode shape
                // above, so the shape-specific trace is the right fit.
                compiledCache[cacheKey] = compile(shapeless: false, body)
            }
            let compiled = compiledCache[cacheKey]!
            let outputs = compiled([
                xFlat,
                gateProj.packed, gateProj.norms,
                upProj.packed, upProj.norms,
                downProj.packed, downProj.norms,
                signsIn, signsDn,
                cbGate, cbDown,
                idxFlat,
            ])
            var outShape = indices.shape
            outShape.append(inputDims)
            return outputs[0].reshaped(outShape).asType(x.dtype)
        }

        // 1. Rotate input
        let xRot = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)

        // 2. Fused gate+up+SwiGLU — broadcast mode: K_meta = K so the kernel
        //    can compute token_idx = dispatch_idx / K and k_idx = dispatch_idx % K.
        //    Total dispatches = batchTokens * K.
        //    `swigluLimit > 0` activates the DSV4 limited-SwiGLU clamp inside
        //    the kernel; `0.0` (default for every non-DSV4 model) keeps the
        //    historical ordinary `silu(gate) * up` expression bit-for-bit.
        let xAct = JANGTQKernels.fusedGateUpSwiGLU(
            xRot: xRot,
            packedGate: gateProj.packed, normsGate: gateProj.norms,
            packedUp: upProj.packed, normsUp: upProj.norms,
            codebook: cbGate, rhsIndices: idxFlat,
            batchTokens: batchTokens, K: K,
            inFeatures: inputDims, outFeatures: hiddenDims, bits: gateUpBits,
            swigluLimit: swigluLimit
        )
        // xAct shape: (batchTokens * K, hidden_dims)

        // 3. Hadamard rotate x_act — one row per (token, expert) pair.
        let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)

        // 4. Gather TQ matmul (down_proj) — per-row mode, one row per pair.
        let y = JANGTQKernels.gatherTQ(
            xRot: xActRot,
            packed: downProj.packed, norms: downProj.norms,
            codebook: cbDown, rhsIndices: idxFlat,
            nRows: batchTokens * K,
            inFeatures: hiddenDims, outFeatures: inputDims, bits: downBits
        )
        // y shape: (batchTokens * K, inputDims)

        // Reshape to match SwitchGLU's output: (batch, seq, K, inputDims)
        var outShape = indices.shape
        outShape.append(inputDims)
        return y.reshaped(outShape).asType(x.dtype)
    }
}

extension TurboQuantSwitchGLU: SwitchGLULayer {}
