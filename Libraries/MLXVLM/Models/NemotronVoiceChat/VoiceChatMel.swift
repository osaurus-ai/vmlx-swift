// NemotronLabs VoiceChat — the mel front-end.
//
// Matches NeMo's `AudioToMelSpectrogramPreprocessor` as this bundle configures
// it, and the reference `mlx_audio.stt.models.nemotron_asr.audio`:
//
//   preemphasis 0.97 · symmetric Hann win_length 400 centre-padded to n_fft 512
//   reflect-padded centred STFT, hop 160 · power spectrum (mag_power 2.0)
//   Slaney mel filters, 128 bins · log(x + 2^-24)
//   normalize "NA" — features are NOT normalised, which is what this model
//   was trained with. Applying per-feature normalisation here would shift
//   every encoder input and degrade ASR in a way nothing else reports.

import Accelerate
import Foundation
import MLX

/// Log-mel features for a mono 16 kHz waveform → (1, frames, 128).
public func voiceChatLogMelSpectrogram(
    _ waveform: [Float], config: VoiceChatPreprocessorConfiguration
) -> MLXArray {
    let sampleRate = config.sampleRate
    let nFFT = config.nFFT
    let winLength = Int((config.windowSize * Double(sampleRate)).rounded())
    let hopLength = Int((config.windowStride * Double(sampleRate)).rounded())
    let nMels = config.features
    let preemph: Float = 0.97
    let logGuard = Float(pow(2.0, -24.0))

    // Preemphasis: y[n] = x[n] - 0.97 x[n-1].
    var x = waveform
    if x.count > 1 {
        for i in stride(from: x.count - 1, to: 0, by: -1) {
            x[i] = x[i] - preemph * x[i - 1]
        }
    }

    // Hann(win_length) centre-padded to n_fft, matching torch.stft / NeMo.
    var window = [Float](repeating: 0, count: nFFT)
    let padLeft = (nFFT - winLength) / 2
    for i in 0 ..< winLength {
        window[padLeft + i] = Float(
            0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(winLength - 1)))
    }

    // Centred STFT with reflect padding.
    let half = nFFT / 2
    var padded = [Float]()
    padded.reserveCapacity(x.count + nFFT)
    for i in 0 ..< half {
        padded.append(x[Swift.min(half - i, x.count - 1)])
    }
    padded.append(contentsOf: x)
    for i in 0 ..< half {
        padded.append(x[Swift.max(x.count - 2 - i, 0)])
    }

    let frames = Swift.max(1, (padded.count - nFFT) / hopLength + 1)
    let bins = nFFT / 2 + 1
    var power = [Float](repeating: 0, count: frames * bins)

    let log2n = vDSP_Length(log2(Double(nFFT)))
    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return MLXArray.zeros([1, frames, nMels])
    }
    defer { vDSP_destroy_fftsetup(setup) }

    var realPart = [Float](repeating: 0, count: half)
    var imagPart = [Float](repeating: 0, count: half)

    for f in 0 ..< frames {
        var frame = [Float](repeating: 0, count: nFFT)
        let start = f * hopLength
        for i in 0 ..< nFFT where start + i < padded.count {
            frame[i] = padded[start + i] * window[i]
        }
        realPart.withUnsafeMutableBufferPointer { realBuffer in
            imagPart.withUnsafeMutableBufferPointer { imagBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
                frame.withUnsafeBufferPointer { framePtr in
                    framePtr.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: half
                    ) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                // vDSP packs DC in realp[0] and Nyquist in imagp[0], and
                // returns values scaled by 2.
                let dc = split.realp[0] / 2
                let nyquist = split.imagp[0] / 2
                power[f * bins] = dc * dc
                power[f * bins + bins - 1] = nyquist * nyquist
                for k in 1 ..< half {
                    let re = split.realp[k] / 2
                    let im = split.imagp[k] / 2
                    power[f * bins + k] = re * re + im * im
                }
            }
        }
    }

    let filters = voiceChatSlaneyMelFilters(
        sampleRate: sampleRate, nFFT: nFFT, nMels: nMels)
    var mel = [Float](repeating: 0, count: frames * nMels)
    for f in 0 ..< frames {
        for m in 0 ..< nMels {
            var sum: Float = 0
            let row = filters[m]
            for k in 0 ..< bins where row[k] != 0 {
                sum += row[k] * power[f * bins + k]
            }
            mel[f * nMels + m] = Foundation.log(sum + logGuard)
        }
    }
    return MLXArray(mel, [1, frames, nMels])
}

/// Slaney-normalised mel filterbank, (nMels, nFFT/2+1).
func voiceChatSlaneyMelFilters(sampleRate: Int, nFFT: Int, nMels: Int) -> [[Float]] {
    let bins = nFFT / 2 + 1
    let fMin: Double = 0
    let fMax = Double(sampleRate) / 2

    func hzToMel(_ hz: Double) -> Double {
        let fMinMel = 0.0, fSp = 200.0 / 3.0
        let minLogHz = 1000.0, minLogMel = (minLogHz - 0.0) / fSp
        let logStep = Foundation.log(6.4) / 27.0
        return hz >= minLogHz
            ? minLogMel + Foundation.log(hz / minLogHz) / logStep
            : fMinMel + hz / fSp
    }
    func melToHz(_ mel: Double) -> Double {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0, minLogMel = (minLogHz - 0.0) / fSp
        let logStep = Foundation.log(6.4) / 27.0
        return mel >= minLogMel
            ? minLogHz * Foundation.exp(logStep * (mel - minLogMel))
            : mel * fSp
    }

    let melMin = hzToMel(fMin), melMax = hzToMel(fMax)
    let melPoints = (0 ... (nMels + 1)).map {
        melToHz(melMin + (melMax - melMin) * Double($0) / Double(nMels + 1))
    }
    let fftFreqs = (0 ..< bins).map { Double($0) * Double(sampleRate) / Double(nFFT) }

    var filters = [[Float]](repeating: [Float](repeating: 0, count: bins), count: nMels)
    for m in 0 ..< nMels {
        let left = melPoints[m], center = melPoints[m + 1], right = melPoints[m + 2]
        // Slaney normalisation: unit AREA per filter, not unit peak.
        let enorm = 2.0 / (right - left)
        for k in 0 ..< bins {
            let freq = fftFreqs[k]
            var value = 0.0
            if freq >= left && freq <= center, center > left {
                value = (freq - left) / (center - left)
            } else if freq > center && freq <= right, right > center {
                value = (right - freq) / (right - center)
            }
            filters[m][k] = Float(value * enorm)
        }
    }
    return filters
}
