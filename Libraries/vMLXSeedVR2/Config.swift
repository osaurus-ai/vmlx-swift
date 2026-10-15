// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import Foundation

/// SeedVR2 transformer configuration. Defaults match the 3B checkpoint
/// (mflux `SeedVR2Transformer.__init__`). The 7B architecture is not supported.
public struct SeedVR2Config: Codable, Sendable {
    public var vidInChannels: Int = 33
    public var vidOutChannels: Int = 16
    public var vidDim: Int = 2560
    public var txtInDim: Int = 5120
    public var heads: Int = 20
    public var headDim: Int = 128
    public var expandRatio: Int = 4
    public var ropeOnText: Bool = true
    public var normEps: Float = 1e-5
    public var patchSize: [Int] = [1, 2, 2]
    public var numLayers: Int = 32
    public var mmLayers: Int = 10
    public var ropeDim: Int = 128
    public var window: [Int] = [4, 3, 3]

    public static let r3B = SeedVR2Config()

}
