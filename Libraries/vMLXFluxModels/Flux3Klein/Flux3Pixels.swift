import Foundation
import vMLXFluxKit

/// CPU RGB8 reference processing. Lanczos3 uses Pillow's separable, normalized
/// signed 22-bit coefficients and rounds/clamps each pass to RGB8.
struct Flux3Pixels: Equatable {
    let width, height: Int
    let rgb: [UInt8]
    init(width: Int, height: Int, rgb: [UInt8]) throws {
        guard width > 0, height > 0, width <= Int.max / height / 3, rgb.count == width * height * 3 else {
            throw FluxError.invalidRequest("invalid reference RGB buffer")
        }
        self.width = width; self.height = height; self.rgb = rgb
    }
    func oriented(_ orientation: Int) throws -> Self {
        let swapped = (5...8).contains(orientation)
        let w = swapped ? height : width, h = swapped ? width : height
        var result = [UInt8](repeating: 0, count: rgb.count)
        for y in 0..<h { try FluxInterruptScope.check(); for x in 0..<w {
            let sx: Int, sy: Int
            switch orientation {
            case 2: sx = width - 1 - x; sy = y
            case 3: sx = width - 1 - x; sy = height - 1 - y
            case 4: sx = x; sy = height - 1 - y
            case 5: sx = y; sy = x
            case 6: sx = y; sy = height - 1 - x
            case 7: sx = width - 1 - y; sy = height - 1 - x
            case 8: sx = width - 1 - y; sy = x
            default: sx = x; sy = y
            }
            for c in 0..<3 { result[(y * w + x) * 3 + c] = rgb[(sy * width + sx) * 3 + c] }
        } }
        return try Self(width: w, height: h, rgb: result)
    }
    private static func coefficients(input: Int, output: Int) -> [(start: Int, weights: [Int64])] {
        let scale = Double(input) / Double(output), filterScale = max(1, scale)
        let support = filterScale * 3, precision = Double(1 << 22)
        func kernel(_ x: Double) -> Double {
            let a = abs(x)
            if a == 0 { return 1 }
            if a >= 3 { return 0 }
            let p = x * Double.pi
            return sin(p) / p * (sin(p / 3) / (p / 3))
        }
        return (0..<output).map { i in
            let center = (Double(i) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5)), end = min(input, Int(center + support + 0.5))
            let values = (start..<end).map { kernel((Double($0) - center + 0.5) / filterScale) }
            let sum = values.reduce(0, +)
            let weights = values.map { value -> Int64 in
                let normalized = sum == 0 ? value : value / sum
                return Int64(normalized * precision + (normalized < 0 ? -0.5 : 0.5))
            }
            return (start, weights)
        }
    }
    func resized(width targetW: Int, height targetH: Int) throws -> Self {
        guard targetW > 0, targetH > 0 else { throw FluxError.invalidRequest("empty resize") }
        var pixels = rgb
        if targetW != width {
            let coefficients = Self.coefficients(input: width, output: targetW)
            var next = [UInt8](repeating: 0, count: targetW * height * 3)
            for y in 0..<height { try FluxInterruptScope.check(); for x in 0..<targetW { for c in 0..<3 {
                let coeff = coefficients[x]
                var value: Int64 = 1 << 21
                for (j, weight) in coeff.weights.enumerated() { value += Int64(pixels[(y * width + coeff.start + j) * 3 + c]) * weight }
                next[(y * targetW + x) * 3 + c] = UInt8(clamping: value >> 22)
            } } }
            pixels = next
        }
        if targetH != height {
            let coefficients = Self.coefficients(input: height, output: targetH)
            var next = [UInt8](repeating: 0, count: targetW * targetH * 3)
            for y in 0..<targetH { try FluxInterruptScope.check(); for x in 0..<targetW { for c in 0..<3 {
                let coeff = coefficients[y]
                var value: Int64 = 1 << 21
                for (j, weight) in coeff.weights.enumerated() { value += Int64(pixels[((coeff.start + j) * targetW + x) * 3 + c]) * weight }
                next[(y * targetW + x) * 3 + c] = UInt8(clamping: value >> 22)
            } } }
            pixels = next
        }
        return try Self(width: targetW, height: targetH, rgb: pixels)
    }
    func prepared(plan: Flux3ReferencePlan, single: Bool) throws -> Self {
        let resized = try resized(width: plan.resized.width, height: plan.resized.height)
        let w = plan.canvas.width, h = plan.canvas.height
        var result = [UInt8](repeating: 0, count: w * h * 3)
        func padded(_ index: Int, side: Int) -> Int {
            if plan.symmetricPadding {
                let period = side * 2, i = (index % period + period) % period
                return i < side ? i : period - 1 - i
            }
            let period = 2 * (side - 1), i = (index % period + period) % period
            return i < side ? i : period - i
        }
        let left = single ? (w - resized.width) / 2 : -(plan.crop?.left ?? 0)
        let top = single ? (h - resized.height) / 2 : -(plan.crop?.top ?? 0)
        for y in 0..<h { try FluxInterruptScope.check(); for x in 0..<w {
            let sx = single ? padded(x - left, side: resized.width) : x - left
            let sy = single ? padded(y - top, side: resized.height) : y - top
            for c in 0..<3 { result[(y * w + x) * 3 + c] = resized.rgb[(sy * resized.width + sx) * 3 + c] }
        } }
        return try Self(width: w, height: h, rgb: result)
    }
    var normalizedNCHW: [Float] {
        let count = width * height
        var result = [Float](repeating: 0, count: count * 3)
        for i in 0..<count { for c in 0..<3 { result[c * count + i] = Float(rgb[i * 3 + c]) / 127.5 - 1 } }
        return result
    }
}
