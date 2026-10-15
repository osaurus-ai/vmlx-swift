// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/ada_modulation.py.
//
// `emb` is reshaped to [B, dim, 2, 3] by the transformer: axis 2 = {attn, mlp},
// axis 3 = {shift, scale, gate}. Modulation adds the timestep modulation to the
// learned per-stream params.
import MLX
import MLXNN

/// The 6 learned modulation vectors for one stream (vid or txt). Keys match the
/// exported checkpoint: `params_vid.attn_shift`, etc.
public final class AdaParams: Module {
    @ParameterInfo(key: "attn_shift") var attnShift: MLXArray
    @ParameterInfo(key: "attn_scale") var attnScale: MLXArray
    @ParameterInfo(key: "attn_gate") var attnGate: MLXArray
    @ParameterInfo(key: "mlp_shift") var mlpShift: MLXArray
    @ParameterInfo(key: "mlp_scale") var mlpScale: MLXArray
    @ParameterInfo(key: "mlp_gate") var mlpGate: MLXArray

    public init(_ dim: Int) {
        self._attnShift.wrappedValue = MLXArray.zeros([dim])
        self._attnScale.wrappedValue = MLXArray.ones([dim])
        self._attnGate.wrappedValue = MLXArray.zeros([dim])
        self._mlpShift.wrappedValue = MLXArray.zeros([dim])
        self._mlpScale.wrappedValue = MLXArray.ones([dim])
        self._mlpGate.wrappedValue = MLXArray.zeros([dim])
        super.init()
    }

    func shift(_ layer: AdaLayer) -> MLXArray { layer == .attn ? attnShift : mlpShift }
    func scale(_ layer: AdaLayer) -> MLXArray { layer == .attn ? attnScale : mlpScale }
    func gate(_ layer: AdaLayer) -> MLXArray { layer == .attn ? attnGate : mlpGate }
}

public enum AdaLayer { case attn, mlp }
public enum AdaMode { case modIn, modOut }

public final class AdaModulation: Module {
    @ModuleInfo(key: "params_all") var paramsAll: AdaParams?
    @ModuleInfo(key: "params_vid") var paramsVid: AdaParams?
    @ModuleInfo(key: "params_txt") var paramsTxt: AdaParams?
    let isLastLayer: Bool
    let shared: Bool

    public init(dim: Int, shared: Bool = false, isLastLayer: Bool = false) {
        self.isLastLayer = isLastLayer
        self.shared = shared
        if shared {
            self._paramsAll.wrappedValue = AdaParams(dim)
        } else {
            self._paramsVid.wrappedValue = AdaParams(dim)
            self._paramsTxt.wrappedValue = isLastLayer ? nil : AdaParams(dim)
        }
        super.init()
    }

    public func modulateVid(_ hidden: MLXArray, _ emb: MLXArray, _ layer: AdaLayer, _ mode: AdaMode)
        -> MLXArray
    {
        apply(hidden, emb, shared ? paramsAll! : paramsVid!, layer, mode)
    }

    public func modulateTxt(_ hidden: MLXArray, _ emb: MLXArray, _ layer: AdaLayer, _ mode: AdaMode)
        -> MLXArray
    {
        if isLastLayer { return hidden }
        return apply(hidden, emb, shared ? paramsAll! : paramsTxt!, layer, mode)
    }

    private func apply(
        _ hidden: MLXArray, _ emb: MLXArray, _ p: AdaParams, _ layer: AdaLayer, _ mode: AdaMode
    ) -> MLXArray {
        let layerIdx = layer == .attn ? 0 : 1
        let mod = emb[0..., 0..., layerIdx]  // [B, dim, 3]
        switch mode {
        case .modIn:
            let shift = mod[.ellipsis, 0].expandedDimensions(axis: 1) + p.shift(layer)
            let scale = mod[.ellipsis, 1].expandedDimensions(axis: 1) + p.scale(layer)
            return hidden * scale + shift
        case .modOut:
            let gate = mod[.ellipsis, 2].expandedDimensions(axis: 1) + p.gate(layer)
            return hidden * gate
        }
    }
}
