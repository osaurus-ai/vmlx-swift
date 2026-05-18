import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

@main
struct OmniAudioChunkStabilityBench {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)

        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["BENCH_MODEL"], !modelPath.isEmpty else {
            throw benchError("Set BENCH_MODEL to a local Nemotron Omni bundle path")
        }

        let audioPath = env["BENCH_AUDIO_FILE"]
            ?? "Tests/MLXLMTests/Resources/audio_only.mov"
        let sampleRate = max(1, Int(env["BENCH_SAMPLE_RATE"] ?? "16000") ?? 16_000)
        let tolerances = parseDoubles(
            env["BENCH_STABILITY_TOLERANCES"],
            defaultValue: [0.1, 0.01, 0.001, 0.0001])
        let failOnUnstable = (env["BENCH_FAIL_ON_UNSTABLE"] ?? "0") == "1"
        let defaultTolerance = Double(env["BENCH_STABILITY_TOLERANCE"] ?? "0.01") ?? 0.01

        try await OmniAudioChunkStabilityRunner.run(
            modelPath: modelPath,
            audioPath: audioPath,
            sampleRate: sampleRate,
            tolerances: tolerances,
            defaultTolerance: defaultTolerance,
            failOnUnstable: failOnUnstable)
    }
}

enum OmniAudioChunkStabilityRunner {
    struct EncodedClip {
        let samples: Int
        let sampleRate: Int
        let encodeMs: Double
        let tokens: Int
        let hidden: Int
        let values: [Float]

        var seconds: Double { Double(samples) / Double(sampleRate) }
    }

    static func run(
        modelPath: String,
        audioPath: String,
        sampleRate: Int,
        tolerances: [Double],
        defaultTolerance: Double,
        failOnUnstable: Bool
    ) async throws {
        let modelDir = URL(fileURLWithPath: modelPath)
        let audioURL = URL(fileURLWithPath: audioPath)

        guard FileManager.default.fileExists(atPath: modelDir.appending(path: "config_omni.json").path)
        else {
            throw benchError("config_omni.json not found; expected a Nemotron Omni bundle")
        }
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw benchError("audio file not found: \(audioURL.path)")
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        let loadMs = elapsedMs(since: loadStart)
        guard let omni = context.model as? NemotronHOmni else {
            throw benchError("loaded \(type(of: context.model)); expected NemotronHOmni")
        }

        printJSON([
            "event": "load",
            "model": modelDir.lastPathComponent,
            "model_path": modelDir.path,
            "load_ms": rounded(loadMs),
            "rss_mib": rounded(currentRSSMiB()),
        ])

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let pcm = try nemotronOmniLoadAudioFile(audioURL, targetSampleRate: Double(sampleRate))
        let decodeMs = elapsedMs(since: decodeStart)
        let checkpoints = checkpointSampleCounts(
            totalSamples: pcm.count,
            sampleRate: sampleRate,
            env: ProcessInfo.processInfo.environment)

        printJSON([
            "event": "audio_source",
            "model": modelDir.lastPathComponent,
            "audio_file": audioURL.path,
            "samples": pcm.count,
            "sample_rate": sampleRate,
            "duration_ms": rounded(Double(pcm.count) * 1000.0 / Double(sampleRate)),
            "decode_ms": rounded(decodeMs),
            "checkpoint_samples": checkpoints,
            "tolerances": tolerances,
            "default_tolerance": defaultTolerance,
            "rss_mib": rounded(currentRSSMiB()),
        ])

        let full = try encode(omni: omni, samples: pcm, sampleRate: sampleRate)
        printJSON(encodeEvent(full, model: modelDir.lastPathComponent, label: "final"))

        var clips: [EncodedClip] = []
        for count in checkpoints where count < pcm.count {
            let prefix = Array(pcm.prefix(count))
            let encoded = try encode(omni: omni, samples: prefix, sampleRate: sampleRate)
            clips.append(encoded)
            printJSON(encodeEvent(encoded, model: modelDir.lastPathComponent, label: "prefix"))
        }
        clips.append(full)

        var comparisonCount = 0
        var unstableComparisons = 0
        for (index, clip) in clips.enumerated() where clip.samples < full.samples {
            let finalComparison = compare(
                prefix: clip,
                reference: full,
                referenceName: "final",
                tolerances: tolerances,
                defaultTolerance: defaultTolerance)
            comparisonCount += 1
            if !(finalComparison["chunk_concat_safe_default"] as? Bool ?? false) {
                unstableComparisons += 1
            }
            printJSON(finalComparison)

            if index + 1 < clips.count {
                let nextComparison = compare(
                    prefix: clip,
                    reference: clips[index + 1],
                    referenceName: "next",
                    tolerances: tolerances,
                    defaultTolerance: defaultTolerance)
                comparisonCount += 1
                if !(nextComparison["chunk_concat_safe_default"] as? Bool ?? false) {
                    unstableComparisons += 1
                }
                printJSON(nextComparison)
            }
        }

        printJSON([
            "event": "summary",
            "model": modelDir.lastPathComponent,
            "audio_file": audioURL.path,
            "comparisons": comparisonCount,
            "unstable_comparisons_default_tolerance": unstableComparisons,
            "default_tolerance": defaultTolerance,
            "chunk_concat_safe_default": unstableComparisons == 0,
            "rss_mib": rounded(currentRSSMiB()),
        ])

        if failOnUnstable && unstableComparisons > 0 {
            throw benchError(
                "Parakeet embeddings are not prefix-stable at tolerance \(defaultTolerance)")
        }
    }

    private static func encode(
        omni: NemotronHOmni,
        samples: [Float],
        sampleRate: Int
    ) throws -> EncodedClip {
        guard !samples.isEmpty else { throw benchError("cannot encode empty audio") }
        let start = CFAbsoluteTimeGetCurrent()
        let embedding = omni.extractAudioEmbeds(waveform: samples)
        MLX.eval(embedding)
        let encodeMs = elapsedMs(since: start)
        let values = embedding.asType(.float32).asArray(Float.self)
        return EncodedClip(
            samples: samples.count,
            sampleRate: sampleRate,
            encodeMs: encodeMs,
            tokens: embedding.dim(0),
            hidden: embedding.dim(1),
            values: values)
    }

    private static func compare(
        prefix: EncodedClip,
        reference: EncodedClip,
        referenceName: String,
        tolerances: [Double],
        defaultTolerance: Double
    ) -> [String: Any] {
        let comparableTokens = min(prefix.tokens, reference.tokens)
        let hidden = min(prefix.hidden, reference.hidden)
        var tokenMaxAbs = Array(repeating: 0.0, count: comparableTokens)
        var totalAbs = 0.0
        var totalSq = 0.0
        var maxAbs = 0.0
        var comparedValues = 0

        for token in 0 ..< comparableTokens {
            var tokenMax = 0.0
            for channel in 0 ..< hidden {
                let lhs = Double(prefix.values[token * prefix.hidden + channel])
                let rhs = Double(reference.values[token * reference.hidden + channel])
                let diff = abs(lhs - rhs)
                tokenMax = max(tokenMax, diff)
                maxAbs = max(maxAbs, diff)
                totalAbs += diff
                totalSq += diff * diff
                comparedValues += 1
            }
            tokenMaxAbs[token] = tokenMax
        }

        var stableByTolerance = [String: Int]()
        var rollbackByTolerance = [String: Int]()
        var safeByTolerance = [String: Bool]()
        for tolerance in tolerances {
            let stable = stableLeadingTokens(
                tokenMaxAbs: tokenMaxAbs,
                tolerance: tolerance)
            stableByTolerance[toleranceKey(tolerance)] = stable
            rollbackByTolerance[toleranceKey(tolerance)] = max(0, comparableTokens - stable)
            safeByTolerance[toleranceKey(tolerance)] = stable == comparableTokens
        }
        let stableDefault = stableLeadingTokens(
            tokenMaxAbs: tokenMaxAbs,
            tolerance: defaultTolerance)

        return [
            "event": "compare",
            "reference": referenceName,
            "prefix_samples": prefix.samples,
            "reference_samples": reference.samples,
            "prefix_seconds": rounded(prefix.seconds),
            "reference_seconds": rounded(reference.seconds),
            "prefix_tokens": prefix.tokens,
            "reference_tokens": reference.tokens,
            "comparable_tokens": comparableTokens,
            "hidden_size": hidden,
            "mean_abs": rounded(comparedValues > 0 ? totalAbs / Double(comparedValues) : 0),
            "rms": rounded(comparedValues > 0 ? sqrt(totalSq / Double(comparedValues)) : 0),
            "max_abs": rounded(maxAbs),
            "stable_tokens_by_tolerance": stableByTolerance,
            "rollback_tokens_by_tolerance": rollbackByTolerance,
            "chunk_concat_safe_by_tolerance": safeByTolerance,
            "default_tolerance": defaultTolerance,
            "stable_tokens_default": stableDefault,
            "rollback_tokens_default": max(0, comparableTokens - stableDefault),
            "chunk_concat_safe_default": stableDefault == comparableTokens,
            "rss_mib": rounded(currentRSSMiB()),
        ]
    }

    private static func stableLeadingTokens(tokenMaxAbs: [Double], tolerance: Double) -> Int {
        for rollback in 0 ... tokenMaxAbs.count {
            let keep = tokenMaxAbs.count - rollback
            if keep == 0 || tokenMaxAbs.prefix(keep).allSatisfy({ $0 <= tolerance }) {
                return keep
            }
        }
        return 0
    }

    private static func encodeEvent(
        _ clip: EncodedClip,
        model: String,
        label: String
    ) -> [String: Any] {
        [
            "event": "encode",
            "label": label,
            "model": model,
            "samples": clip.samples,
            "sample_rate": clip.sampleRate,
            "seconds": rounded(clip.seconds),
            "audio_tokens": clip.tokens,
            "hidden_size": clip.hidden,
            "encode_ms": rounded(clip.encodeMs),
            "rss_mib": rounded(currentRSSMiB()),
        ]
    }

    private static func checkpointSampleCounts(
        totalSamples: Int,
        sampleRate: Int,
        env: [String: String]
    ) -> [Int] {
        if let raw = env["BENCH_CHUNK_SECONDS"], !raw.isEmpty {
            let parsed = parseDoubles(raw, defaultValue: [])
                .map { max(1, min(totalSamples, Int(($0 * Double(sampleRate)).rounded()))) }
            return Array(Set(parsed + [totalSamples])).sorted()
        }

        let step = max(0.1, Double(env["BENCH_CHUNK_STEP_SECONDS"] ?? "1.0") ?? 1.0)
        var counts = Set<Int>()
        var seconds = step
        let totalSeconds = Double(totalSamples) / Double(sampleRate)
        while seconds < totalSeconds {
            counts.insert(max(1, min(totalSamples, Int((seconds * Double(sampleRate)).rounded()))))
            seconds += step
        }
        counts.insert(totalSamples)
        return Array(counts).sorted()
    }

    private static func toleranceKey(_ value: Double) -> String {
        String(format: "%.6g", value)
    }

    private static func printJSON(_ fields: [String: Any]) {
        do {
            let data = try JSONSerialization.data(
                withJSONObject: fields, options: [.sortedKeys])
            let json = String(data: data, encoding: .utf8) ?? "{}"
            print("OMNI_AUDIO_CHUNK_STABILITY \(json)")
        } catch {
            print("OMNI_AUDIO_CHUNK_STABILITY {\"event\":\"encode_error\",\"error\":\"\(error)\"}")
        }
    }
}

private func elapsedMs(since start: CFAbsoluteTime) -> Double {
    (CFAbsoluteTimeGetCurrent() - start) * 1000.0
}

private func rounded(_ value: Double) -> Double {
    guard value.isFinite else { return value }
    return (value * 10).rounded() / 10
}

private func currentRSSMiB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<natural_t>.stride)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0,
                &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.resident_size) / (1024.0 * 1024.0)
}

private func parseDoubles(_ raw: String?, defaultValue: [Double]) -> [Double] {
    guard let raw, !raw.isEmpty else { return defaultValue }
    let values = raw.split(separator: ",").compactMap {
        Double($0.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    return values.isEmpty ? defaultValue : values
}

private func benchError(_ message: String) -> NSError {
    NSError(
        domain: "OmniAudioChunkStabilityBench",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: message])
}
