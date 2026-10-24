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
        let rowsURL = URL(fileURLWithPath: env["D3_ROWS"]!)
        let outputURL = URL(fileURLWithPath: env["D3_OUTPUT"]!)
        let rows = try String(contentsOf: rowsURL, encoding: .utf8).split(separator: "\n")
        let loadStart = Date.timeIntervalSinceReferenceDate
        let model = try await D3Model.load(directory: directory, tokenizerLoader: #huggingFaceTokenizerLoader())
        print("D3_LOADED seconds=\(Date.timeIntervalSinceReferenceDate - loadStart) active=\(Memory.activeMemory) cache=\(Memory.cacheMemory)")
        var output: [String: [Float]] = [:]
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
            guard raw["type"]?.string != "box", raw["type"]?.string != "track" else { continue }
            let question = try D3Question(raw)
            let base = rowsURL.deletingLastPathComponent()
            let images: [URL] = (row["images"]?.array ?? []).map { base.appendingPathComponent($0.string!) }
            let videos: [URL] = (row["videos"]?.array ?? []).map { base.appendingPathComponent($0.string!) }
            let r = try await model.decide(state: row["state"] ?? .null, question: question, images: images, videos: videos)
            output[id] = r.probabilities
            print("D3_ROW index=\(index) kind=\(row["kind"]?.string ?? "") tokens=\(r.inputTokens) state=\(r.stateTokens) seconds=\(r.seconds) prompt_tps=\(r.promptTokensPerSecond) active=\(Memory.activeMemory) cache=\(Memory.cacheMemory) probs=\(r.probabilities)")
            try JSONEncoder().encode(output).write(to: outputURL, options: .atomic)
            Memory.clearCache()
        }
    }
}
