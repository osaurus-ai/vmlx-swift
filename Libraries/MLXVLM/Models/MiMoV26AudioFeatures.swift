// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Accelerate
import Foundation
import MLX

/// Native audio frontend: sinc-Hann resampling and magnitude HTK log-mel.
/// Double precision matches the reference preprocessing before its Float32
/// encoder boundary; Accelerate performs the DFT and filterbank multiplication.
enum MiMoV26AudioFeatures {
    struct Configuration: Decodable, Sendable {
        let sampleRate: Int
        let fft: Int
        let hop: Int
        let window: Int
        let bins: Int
        let minimumFrequency: Double
        let maximumFrequency: Double?

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sampling_rate", fft = "nfft", hop = "hop_length"
            case window = "window_size", bins = "n_mels"
            case minimumFrequency = "fmin", maximumFrequency = "fmax"
        }
    }

    static func resample(_ input: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        guard sourceRate > 0, targetRate > 0, !input.isEmpty else {
            throw VLMError.processing("MiMo audio requires nonempty PCM and positive sample rates")
        }
        if sourceRate == targetRate { return input }
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let divisor = gcd(sourceRate, targetRate)
        let original = sourceRate / divisor, target = targetRate / divisor
        let base = Double(min(original, target)) * 0.99
        let width = Int(ceil(6 * Double(original) / base))
        let taps = 2 * width + original
        let count = Int(ceil(Double(target) * Double(input.count) / Double(original)))
        // Cached-transform reference kernels are Float32, with the convolution
        // accumulated in Double. Generate each phase without a large phase bank.
        var output = Array(repeating: Float(0), count: count)
        for phase in 0..<min(target, count) {
            let offset = Double(-Float(phase) / Float(target))
            var kernel = Array(repeating: Double(0), count: taps)
            for tap in 0..<taps {
                let t = min(6, max(-6, (offset + Double(tap - width) / Double(original)) * base))
                let hann = pow(cos(t * .pi / 12), 2)
                let angle = t * .pi
                let sinc = angle == 0 ? 1 : sin(angle) / angle
                kernel[tap] = Double(Float(sinc * hann * base / Double(original)))
            }
            var position = phase
            var block = 0
            while position < count {
                let start = block * original - width
                let lower = max(0, -start), upper = min(taps, input.count - start)
                var sum = Double(0)
                if lower < upper {
                    for tap in lower..<upper { sum += Double(input[start + tap]) * kernel[tap] }
                }
                output[position] = Float(sum)
                position += target; block += 1
            }
        }
        return output
    }

    static func logMel(_ samples: [Float], configuration c: Configuration) throws -> MLXArray {
        guard c.fft > 0, c.fft.isMultiple(of: 2), c.hop > 0,
            c.window > 0, c.window <= c.fft, c.bins > 0, c.sampleRate > 0,
            c.minimumFrequency >= 0,
            (c.maximumFrequency ?? Double(c.sampleRate / 2)) > c.minimumFrequency,
            samples.count > c.fft / 2
        else { throw VLMError.processing("Invalid MiMo mel configuration or audio shorter than reflect padding") }
        guard let setup = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(c.fft), .FORWARD) else {
            throw VLMError.processing("Accelerate cannot construct the configured MiMo audio DFT")
        }
        defer { vDSP_DFT_DestroySetupD(setup) }
        let pad = c.fft / 2, frequencies = c.fft / 2 + 1
        let count = 1 + samples.count / c.hop
        let left = (c.fft - c.window) / 2
        var window = Array(repeating: Double(0), count: c.fft)
        for i in 0..<c.window { window[left + i] = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(c.window)) }
        var real = Array(repeating: Double(0), count: c.fft)
        let imaginary = Array(repeating: Double(0), count: c.fft)
        var outputReal = real, outputImaginary = real
        var magnitudes = Array(repeating: Double(0), count: count * frequencies)
        for frame in 0..<count {
            for tap in 0..<c.fft {
                var index = frame * c.hop + tap - pad
                if index < 0 { index = -index }
                if index >= samples.count { index = 2 * samples.count - 2 - index }
                real[tap] = Double(samples[index]) * window[tap]
            }
            vDSP_DFT_ExecuteD(setup, real, imaginary, &outputReal, &outputImaginary)
            for bin in 0..<frequencies {
                magnitudes[frame * frequencies + bin] = hypot(outputReal[bin], outputImaginary[bin])
            }
        }
        let maximum = c.maximumFrequency ?? Double(c.sampleRate / 2)
        func mel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }
        let lower = mel(c.minimumFrequency), upper = mel(maximum)
        let points = (0..<(c.bins + 2)).map {
            700 * (pow(10, (lower + (upper - lower) * Double($0) / Double(c.bins + 1)) / 2595) - 1)
        }
        var filters = Array(repeating: Double(0), count: frequencies * c.bins)
        for f in 0..<frequencies {
            let hz = Double(f) * Double(c.sampleRate / 2) / Double(frequencies - 1)
            for m in 0..<c.bins {
                let down = (hz - points[m]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - hz) / (points[m + 2] - points[m + 1])
                filters[f * c.bins + m] = max(0, min(down, up))
            }
        }
        var output = Array(repeating: Double(0), count: count * c.bins)
        vDSP_mmulD(magnitudes, 1, filters, 1, &output, 1,
                   vDSP_Length(count), vDSP_Length(c.bins), vDSP_Length(frequencies))
        return MLXArray(output.map { Float(log(max($0, 1e-7))) }).reshaped(count, c.bins)
    }
}
