import Foundation

/// Super-VL's source architecture requires a final vision LayerNorm when its
/// language configuration contains a next-token prediction head. This is an
/// architectural condition, independent of whether speculation is enabled.
func nemotronVisionFinalLayerNormPresent(
    tensorNames: Set<String>, nextTokenLayers: Int
) throws -> Bool {
    let prefix = "vision_projector.vision_final_layernorm."
    let hasWeight = tensorNames.contains(prefix + "weight")
    let hasBias = tensorNames.contains(prefix + "bias")
    guard hasWeight == hasBias, hasWeight == (nextTokenLayers > 0) else {
        throw DecodingError.dataCorrupted(.init(
            codingPath: [],
            debugDescription:
                "Nemotron vision final LayerNorm requires both weight and bias when num_nextn_predict_layers > 0; "
                + "found weight=\(hasWeight), bias=\(hasBias), next-token layers=\(nextTokenLayers)."))
    }
    return hasWeight
}
