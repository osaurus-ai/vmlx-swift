import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import VMLXTokenizers

/// Numeric diagnostic only. Osaurus application proof is recorded separately.
enum D3DecisionProbe {
    static func run(directory: URL) async throws {
        let env = ProcessInfo.processInfo.environment
        if let image = env["D3_DUMP_IMAGE"], let destination = env["D3_DUMP_PIXELS"] {
            let configuration = try D3Model.mediaProcessorConfiguration(directory: directory)
            guard let input = CIImage(contentsOf: URL(fileURLWithPath: image)) else {
                throw DecisionRuntimeError.invalid("Diagnostic image could not be decoded")
            }
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
            let processor = Qwen3VLProcessor(
                configuration, tokenizer: tokenizer, decisionMediaProcessing: true)
            let (pixels, _) = try processor.preprocess(images: [input], processing: nil)
            try save(arrays: ["pixels": pixels], url: URL(fileURLWithPath: destination))
            if env["D3_PIXELS_ONLY"] == "1" { return }
        }
        let rowsURL = URL(fileURLWithPath: env["D3_ROWS"]!)
        let outputURL = URL(fileURLWithPath: env["D3_OUTPUT"]!)
        let rows = try String(contentsOf: rowsURL, encoding: .utf8).split(separator: "\n")
        let loadStart = Date.timeIntervalSinceReferenceDate
        let model = try await D3Model.load(
            directory: directory, tokenizerLoader: #huggingFaceTokenizerLoader())
        print(
            "D3_LOADED seconds=\(Date.timeIntervalSinceReferenceDate - loadStart) active=\(Memory.activeMemory) cache=\(Memory.cacheMemory)"
        )
        var output: [String: JSONValue] = [:]
        var inputReceipts: [String: JSONValue] = [:]
        let limit = Int(env["D3_LIMIT"] ?? "0") ?? 0
        for (index, line) in rows.enumerated() {
            if limit > 0 && index >= limit { break }
            let row = try DecisionJSON.parse(Data(line.utf8))
            let id = row["id"]!.string!
            // Suite rows carry normalized questions without their default choice type.
            var raw = row["question"]!
            if raw["type"] == nil, let entries = raw.entries {
                raw = .object([("type", .string("choice"))] + entries)
            }
            let base = rowsURL.deletingLastPathComponent()
            let images: [URL] = (row["images"]?.array ?? []).map {
                base.appendingPathComponent($0.string!)
            }
            let videos: [URL] = (row["videos"]?.array ?? []).map {
                base.appendingPathComponent($0.string!)
            }
            if raw["type"]?.string == "box" {
                let r = try await model.decideBox(
                    state: row["state"] ?? .null,
                    instructions: raw["instructions"]?.describe
                        ?? "Choose the best matching option.", images: images)
                output[id] = .object(["box": .object(r.box.mapValues(JSONValue.double))])
                print(
                    "D3_BOX index=\(index) tokens=\(r.inputTokens) state=\(r.stateTokens) seconds=\(r.seconds) box=\(r.box)"
                )
                try JSONEncoder().encode(output).write(to: outputURL, options: .atomic)
                Memory.clearCache()
                continue
            }
            let question = try D3Question(raw)
            let r = try await model.decide(
                state: row["state"] ?? .null, question: question, images: images, videos: videos)
            output[id] = .array(r.probabilities.map { .double(Double($0)) })
            inputReceipts[id] = .object([
                "tokens": .array(r.tokenIDs.map(JSONValue.int)),
                "image_grid_thw": .array(
                    r.imageGrids.map { .array([$0.t, $0.h, $0.w].map(JSONValue.int)) }),
                "video_grid_thw": .array(
                    r.videoGrids.map { .array([$0.t, $0.h, $0.w].map(JSONValue.int)) }),
            ])
            try JSONEncoder().encode(inputReceipts).write(
                to: URL(fileURLWithPath: outputURL.path + ".inputs.json"), options: .atomic)
            print(
                "D3_ROW index=\(index) kind=\(row["kind"]?.string ?? "") tokens=\(r.inputTokens) state=\(r.stateTokens) seconds=\(r.seconds) prompt_tps=\(r.promptTokensPerSecond) active=\(Memory.activeMemory) cache=\(Memory.cacheMemory) probs=\(r.probabilities)"
            )
            try JSONEncoder().encode(output).write(to: outputURL, options: .atomic)
            Memory.clearCache()
        }
    }
}
