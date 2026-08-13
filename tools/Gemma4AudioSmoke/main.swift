// Audio-input/text-output VLM runtime proof.
//
// Loads a local bundle, feeds a real audio file plus a text prompt through
// the model's processor and decoder, and optionally runs a three-turn text
// conversation through the same loaded model.
//
// Usage:
//   GEMMA4_SMOKE_MODEL=/path/to/bundle \
//   GEMMA4_SMOKE_AUDIO=/path/to/audio.wav \
//   GEMMA4_SMOKE_PROMPT="What do you hear?" \
//   GEMMA4_SMOKE_MAX_TOKENS=64 \
//   swift run Gemma4AudioSmoke
//
// Requires the MLX metallib next to the executable —
// run scripts/prepare-mlx-metal.sh first.

import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
@preconcurrency import VMLXTokenizers

private func currentPhysicalFootprintBytes() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(
                mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
        }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : nil
}

private final class PhysicalFootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private let timer: DispatchSourceTimer
    private var peakBytes: UInt64 = 0

    init(interval: TimeInterval = 0.1) {
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }

    private func sample() {
        guard let bytes = currentPhysicalFootprintBytes() else { return }
        lock.lock()
        peakBytes = max(peakBytes, bytes)
        lock.unlock()
    }

    func stop() -> UInt64 {
        sample()
        timer.cancel()
        lock.lock()
        defer { lock.unlock() }
        return peakBytes
    }
}

@main
struct Gemma4AudioSmoke {
    private struct TurnResult {
        let text: String
    }

    private static func hasProtocolMarker(_ text: String) -> Bool {
        let lower = text.lowercased()
        return ["<|", "<think", "</think", "<tool_call", "</tool_call"].contains {
            lower.contains($0)
        }
    }

    private static func generatePreparedTurn(
        context: ModelContext, input: LMInput, parameters: GenerateParameters,
        label: String
    ) async throws -> TurnResult {
        let stream = try MLXLMCommon.generate(
            input: input, parameters: parameters, context: context)
        var text = ""
        var reasoningCharacterCount = 0
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if let chunk = event.chunk { text += chunk }
            if let reasoning = event.reasoning { reasoningCharacterCount += reasoning.count }
            if let info = event.info { completion = info }
        }
        guard let completion else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 10,
                userInfo: [NSLocalizedDescriptionKey: "\(label) emitted no completion telemetry"])
        }
        print(
            String(
                format: "[smoke] %@ generated %d tokens at %.1f tok/s, stop=%@",
                label, completion.generationTokenCount, completion.tokensPerSecond,
                String(describing: completion.stopReason)))
        print("[smoke] \(label) reasoning channel characters: \(reasoningCharacterCount)")
        print("[smoke] \(label) output: \(text)")
        guard completion.generationTokenCount > 0, completion.tokensPerSecond > 0 else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 11,
                userInfo: [NSLocalizedDescriptionKey: "\(label) emitted no measurable generation"])
        }
        guard case .stop = completion.stopReason else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 12,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(label) stopped by \(completion.stopReason)"
                ])
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            !hasProtocolMarker(text)
        else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 13,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "\(label) output is empty or leaks a protocol marker"
                ])
        }
        guard reasoningCharacterCount == 0 else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 15,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "\(label) emitted \(reasoningCharacterCount) reasoning characters"
                ])
        }
        return TurnResult(text: text)
    }

    private static func generateChatTurn(
        context: ModelContext, chat: [Chat.Message], parameters: GenerateParameters,
        additionalContext: [String: any Sendable]?, label: String
    ) async throws -> TurnResult {
        let input = try await context.processor.prepare(
            input: UserInput(chat: chat, additionalContext: additionalContext))
        nonisolated(unsafe) let sendableInput = input
        return try await generatePreparedTurn(
            context: context, input: sendableInput, parameters: parameters, label: label)
    }

    private static func runThreeTurnProof(
        context: ModelContext, parameters: GenerateParameters,
        initialUserMessage: Chat.Message, firstResponse: String,
        additionalContext: [String: any Sendable]?
    ) async throws {
        print("[smoke] starting three-turn audio conversation proof")
        var history: [Chat.Message] = [
            initialUserMessage,
            .assistant(firstResponse),
            .user("According to the transcription, name two vegetables that were mentioned."),
        ]
        let turn2 = try await generateChatTurn(
            context: context, chat: history, parameters: parameters,
            additionalContext: additionalContext, label: "turn 2")
        history.append(.assistant(turn2.text))
        history.append(.user("Was mutton mentioned in the speech? Answer yes or no."))
        let turn3 = try await generateChatTurn(
            context: context, chat: history, parameters: parameters,
            additionalContext: additionalContext, label: "turn 3")
        let second = turn2.text.lowercased()
        let third = turn3.text.lowercased()
        let first = firstResponse.lowercased()
        let transcriptTerms = ["turnip", "carrot", "potato", "mutton"]
        let transcriptHits = transcriptTerms.filter { first.contains($0) }
        let vegetableHits = ["turnip", "carrot", "potato"].filter { second.contains($0) }
        guard transcriptHits.count == transcriptTerms.count,
            vegetableHits.count >= 2,
            third.contains("yes")
        else {
            throw NSError(
                domain: "AudioVLMRuntimeProof", code: 14,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "three-turn audio recall failed: \(firstResponse) | \(turn2.text) | \(turn3.text)"
                ])
        }
        print("[smoke] three-turn audio conversation proof PASS")
    }

    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)
        let footprintSampler = PhysicalFootprintSampler()
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["GEMMA4_SMOKE_MODEL"], !modelPath.isEmpty else {
            fputs("Set GEMMA4_SMOKE_MODEL to a local audio VLM bundle path\n", stderr)
            exit(1)
        }
        guard let audioPath = env["GEMMA4_SMOKE_AUDIO"], !audioPath.isEmpty else {
            fputs("Set GEMMA4_SMOKE_AUDIO to a wav/audio file path\n", stderr)
            exit(1)
        }
        let prompt = env["GEMMA4_SMOKE_PROMPT"] ?? "What do you hear in this audio clip?"
        let maxTokens = max(1, Int(env["GEMMA4_SMOKE_MAX_TOKENS"] ?? "64") ?? 64)

        let modelDir = URL(fileURLWithPath: modelPath)
        let audioURL = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            fputs("audio file not found: \(audioURL.path)\n", stderr)
            exit(1)
        }

        print("[smoke] loading \(modelDir.lastPathComponent) ...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context = try await MLXLMCommon.loadModel(
            from: modelDir, using: #huggingFaceTokenizerLoader())
        print(
            String(
                format: "[smoke] loaded %@ in %.1fs (model type: %@)",
                modelDir.lastPathComponent,
                CFAbsoluteTimeGetCurrent() - loadStart,
                String(describing: type(of: context.model))))

        let initialUserMessage = Chat.Message.user(prompt, audios: [.url(audioURL)])
        let chat: [Chat.Message] = [initialUserMessage]
        let additionalContext: [String: any Sendable]?
        if let raw = env["GEMMA4_SMOKE_ENABLE_THINKING"]?.lowercased() {
            let enabled = raw == "1" || raw == "true" || raw == "yes" || raw == "on"
            additionalContext = ["enable_thinking": enabled]
            print("[smoke] explicit template mode: enable_thinking=\(enabled)")
        } else {
            additionalContext = nil
            print("[smoke] template mode: bundle default")
        }
        let userInput = UserInput(chat: chat, additionalContext: additionalContext)

        let prepareStart = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: userInput)
        let promptTokens = lmInput.text.tokens.dim(-1)
        print(
            String(
                format: "[smoke] processor.prepare: %.0f ms, prompt tokens: %d, audio: %@",
                (CFAbsoluteTimeGetCurrent() - prepareStart) * 1000,
                promptTokens,
                lmInput.audio.map {
                    "waveform shape \($0.waveform.shape), "
                        + "preEncoded: \($0.preEncodedEmbedding != nil)"
                } ?? "nil"))

        var parameters = GenerateParameters(
            generationConfig: context.configuration.generationDefaults)
        parameters.maxTokens = maxTokens
        if let rawTemperature = env["GEMMA4_SMOKE_TEMPERATURE"],
            let temperature = Float(rawTemperature)
        {
            parameters.temperature = temperature
            print("[smoke] explicit temperature override: \(temperature)")
        }
        let repetitionPenalty = parameters.repetitionPenalty.map { String($0) } ?? "nil"
        print(
            "[smoke] generation parameters: temperature=\(parameters.temperature), "
                + "top_p=\(parameters.topP), top_k=\(parameters.topK), "
                + "min_p=\(parameters.minP), repetition_penalty="
                + repetitionPenalty)

        nonisolated(unsafe) let sendableInput = lmInput
        let firstTurn = try await generatePreparedTurn(
            context: context, input: sendableInput, parameters: parameters, label: "audio turn 1")
        if (env["GEMMA4_SMOKE_MULTI_TURN"] ?? "0") == "1" {
            try await runThreeTurnProof(
                context: context, parameters: parameters,
                initialUserMessage: initialUserMessage, firstResponse: firstTurn.text,
                additionalContext: additionalContext)
        }
        let currentBytes = currentPhysicalFootprintBytes() ?? 0
        let peakBytes = footprintSampler.stop()
        print(
            String(
                format: "[smoke] physical footprint current=%.3f GiB peak_sampled=%.3f GiB",
                Double(currentBytes) / Double(1 << 30),
                Double(peakBytes) / Double(1 << 30)))
        print("[smoke] PASS")
    }
}
