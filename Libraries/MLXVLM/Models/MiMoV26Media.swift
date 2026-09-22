// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// MiMo's image/video geometry and pixel math. The bundle's processor_config,
/// rather than its legacy Qwen preprocessor_config, supplies the pixel budgets.
enum MiMoV26Pixels {
    static func targetSize(height: Int, width: Int, factor: Int, minimum: Int, maximum: Int)
        throws -> (Int, Int)
    {
        guard height > 0, width > 0, factor > 0, minimum > 0, maximum >= minimum else {
            throw VLMError.processing("Invalid MiMo resize geometry or pixel budget")
        }
        var h = Double(height), w = Double(width)
        if min(height, width) < factor {
            let scale = Double(factor) / min(h, w)
            h = (h * scale).rounded(.toNearestOrEven)
            w = (w * scale).rounded(.toNearestOrEven)
        } else if max(h, w) / min(h, w) > 200 {
            throw VLMError.processing("MiMo image aspect ratio exceeds 200")
        }
        let unit = Double(factor)
        var rh = (h / unit).rounded(.toNearestOrEven) * unit
        var rw = (w / unit).rounded(.toNearestOrEven) * unit
        if rh * rw > Double(maximum) {
            let scale = sqrt(h * w / Double(maximum))
            rh = floor(h / scale / unit) * unit
            rw = floor(w / scale / unit) * unit
        } else if rh * rw < Double(minimum) {
            let scale = sqrt(Double(minimum) / (h * w))
            rh = ceil(h * scale / unit) * unit
            rw = ceil(w * scale / unit) * unit
        }
        guard rh >= unit, rw >= unit, rh < Double(Int.max), rw < Double(Int.max) else {
            throw VLMError.processing("MiMo resize produced an invalid output size")
        }
        return (Int(rh), Int(rw))
    }

    /// align_corners=false, no antialias, Float32 bilinear interpolation.
    /// Input/output are NCHW and retain the reference's 0...255 pixel scale.
    static func resize(_ input: MLXArray, height: Int, width: Int) throws -> MLXArray {
        guard input.ndim == 4, input.dim(1) == 3, input.dim(2) > 0, input.dim(3) > 0,
            height > 0, width > 0 else {
            throw VLMError.processing("MiMo resize expects nonempty NCHW RGB frames")
        }
        func taps(_ count: Int, _ output: Int) -> (MLXArray, MLXArray, MLXArray) {
            let scale = Float(count) / Float(output)
            var first: [Int32] = [], second: [Int32] = [], fraction: [Float] = []
            for i in 0..<output {
                let position = max(scale * (Float(i) + 0.5) - 0.5, 0)
                let lower = min(Int(floor(position)), count - 1)
                first.append(Int32(lower)); second.append(Int32(min(lower + 1, count - 1)))
                fraction.append(position - Float(lower))
            }
            return (MLXArray(first), MLXArray(second), MLXArray(fraction))
        }
        let (y0, y1, fy) = taps(input.dim(2), height)
        let (x0, x1, fx) = taps(input.dim(3), width)
        let y = fy.reshaped(1, 1, height, 1), x = fx.reshaped(1, 1, 1, width)
        let source = input.asType(.float32)
        let rows = source.take(y0, axis: 2) * (1 - y) + source.take(y1, axis: 2) * y
        return rows.take(x0, axis: 3) * (1 - x) + rows.take(x1, axis: 3) * x
    }

    static func patches(_ input: MLXArray, height: Int, width: Int,
                        patch: Int, merge: Int, temporal: Int) throws -> (MLXArray, THW) {
        guard patch > 0, merge > 0, temporal > 0, input.ndim == 4, input.dim(0) > 0,
            height.isMultiple(of: patch * merge), width.isMultiple(of: patch * merge) else {
            throw VLMError.processing("Invalid MiMo patch geometry")
        }
        let mean = MLXArray([Float(123.675), 116.28, 103.53]).reshaped(1, 3, 1, 1)
        let std = MLXArray([Float(58.395), 57.12, 57.375]).reshaped(1, 3, 1, 1)
        let frames = (try resize(input, height: height, width: width) - mean) / std
        // Qwen's patch packing has the same C,T,P,P and merge-unit order.
        return try QwenVL.patchify(images: [frames], mergeSize: merge,
                                  patchSize: patch, temporalPatchSize: temporal)
    }

    static func videoSamples(total: Int, sourceFPS: Double, targetFPS: Double,
                             minimum: Int, maximum: Int) throws -> [Int] {
        guard total >= 2, sourceFPS.isFinite, sourceFPS > 0, targetFPS.isFinite,
            targetFPS > 0, minimum > 0, maximum >= minimum else {
            throw VLMError.processing("Invalid MiMo video sampling parameters")
        }
        let lower = ((minimum + 1) / 2) * 2
        let upper = (min(maximum, total) / 2) * 2
        let desired = Double(total) / sourceFPS * targetFPS
        let count = Int(floor(min(min(max(desired, Double(lower)), Double(upper)), Double(total)) / 2)) * 2
        guard count >= 2 else { throw VLMError.processing("MiMo video needs at least two frames") }
        return (0..<count).map { Int(floor(Double($0) * Double(total - 1) / Double(count - 1))) }
    }

    static func timestamp(_ seconds: Double) throws -> String {
        guard seconds.isFinite, seconds >= 0, seconds < Double(Int.max) else {
            throw VLMError.processing("Invalid MiMo video timestamp")
        }
        return String(format: "%02d:%02d", Int(seconds / 60), Int(seconds.truncatingRemainder(dividingBy: 60)))
    }
}
