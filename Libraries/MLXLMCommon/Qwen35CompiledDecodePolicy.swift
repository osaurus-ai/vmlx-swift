// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Architecture/environment policy gating the Qwen 3.5 MoE text backbone's
/// trusted routed-MoE decode region (`Qwen4ExpCompiledRoutedSwitchGLU` in
/// `SwitchLayers.swift`).
///
/// The region is reached from `SwitchGLU` only when `compileSeparatedDecode` is
/// set, and both Qwen 3.5 constructions must agree on when: the text-only model
/// in `MLXLLM/Models/Qwen35.swift` and the VLM in `MLXVLM/Models/Qwen35.swift`.
/// Keeping the predicate here, next to the region itself, stops the two callers
/// from drifting apart.
///
/// The region keeps its own exact-shape, dtype, quantization, and single-token
/// guards at call time, so this policy only decides which `SwitchGLU` instances
/// are allowed to attempt it. Everything else — prefill, unsupported shapes,
/// disabled configuration — falls back to the eager routed path unchanged.
public enum Qwen35CompiledDecodePolicy {

    /// Explicit opt-out (or forced enable) for the compiled decode regions:
    /// `0`/`false` disables, any other value enables. When absent, the
    /// architecture match below decides. The legacy `VMLINUX_` spelling is
    /// honored through `RuntimeEnvironment`.
    public static let environmentVariable = "VMLX_QWEN35_COMPILE_DECODE_REGIONS"

    /// Whether a Qwen 3.5 MoE text backbone with the given shape may use the
    /// compiled routed-MoE decode region. Only the validated
    /// `qwen3_5_moe_text` topology is eligible; neighboring shapes stay eager.
    public static func shouldCompileDecodeRegions(
        modelType: String,
        hiddenSize: Int,
        hiddenLayers: Int,
        fullAttentionInterval: Int,
        numExperts: Int,
        numExpertsPerTok: Int,
        moeIntermediateSize: Int,
        linearNumKeyHeads: Int,
        linearNumValueHeads: Int,
        linearKeyHeadDim: Int,
        linearValueHeadDim: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if let override = RuntimeEnvironment.value(environmentVariable, in: environment) {
            return override != "0" && override.lowercased() != "false"
        }
        return modelType == "qwen3_5_moe_text"
            && hiddenSize == 2048
            && hiddenLayers == 40
            && fullAttentionInterval == 4
            && numExperts == 256
            && numExpertsPerTok == 8
            && moeIntermediateSize == 512
            && linearNumKeyHeads == 16
            && linearNumValueHeads == 32
            && linearKeyHeadDim == 128
            && linearValueHeadDim == 128
    }
}
