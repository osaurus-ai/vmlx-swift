import Foundation

struct Flux3TransformerConfig: Decodable {
    let attentionHeadDim, numAttentionHeads, numLayers, numSingleLayers: Int
    let inChannels, condInChannels, jointAttentionDim, timestepChannels: Int
    let mlpRatio: Float
    let axesDimsRope: [Int]
    let ropeTheta: Float
    let flowMap: Bool
    var hidden: Int { attentionHeadDim * numAttentionHeads }
}

/// Host float64 trigonometry, float32 tables, adjacent-pair rotation.
/// Separate from MLX so small reference positions can be checked on the CPU.
enum Flux3RotaryTables {
    static func tables(ids: [[Float]], axes: [Int], theta: Double) -> (cos: [Float], sin: [Float]) {
        var cs: [Float] = [], ss: [Float] = []
        for row in ids {
            for (axis, dim) in axes.enumerated() {
                for j in stride(from: 0, to: dim, by: 2) {
                    let angle = Double(row[axis]) / pow(theta, Double(j) / Double(dim))
                    cs.append(Float(Foundation.cos(angle))); ss.append(Float(Foundation.sin(angle)))
                }
            }
        }
        return (cs, ss)
    }
}

struct Flux3PromptPlan: Equatable {
    let ids, mask, positions: [Int32]
    init(tokens: [Int], padID: Int, maxTokens: Int, padMultiple: Int) {
        let real = min(tokens.count, maxTokens)
        let count = min(((real + padMultiple - 1) / padMultiple) * padMultiple, maxTokens)
        ids = tokens.prefix(real).map(Int32.init) + [Int32](repeating: Int32(padID), count: count - real)
        mask = [Int32](repeating: 1, count: real) + [Int32](repeating: 0, count: count - real)
        positions = (0..<real).map(Int32.init) + [Int32](repeating: 1, count: count - real)
    }
    static func textIDs(length: Int) -> [[Float]] { (0..<length).map { [0, 0, 0, Float($0)] } }
    static func imageIDs(height: Int, width: Int, layer: Int = 0) -> [[Float]] {
        (0..<height).flatMap { row in (0..<width).map { col in [0, Float(row) * 0.5, Float(col) * 0.5, Float(layer)] } }
    }
}
