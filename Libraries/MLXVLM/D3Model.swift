import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A decision model is deliberately not a LanguageModel/NativeMTPModel. Its owner serializes
/// requests, holds only weights between calls, and never enters a generation iterator.
public actor D3Model {
    public let bundle: DecisionBundle
    public let tokenizer: any Tokenizer
    private var backbone: Qwen35?
    private var readout: MLXArray?
    private let processor: Qwen3VLProcessor
    private var resources: DecisionLoadResources?

    private init(bundle: DecisionBundle, tokenizer: any Tokenizer, backbone: Qwen35,
        readout: MLXArray, processor: Qwen3VLProcessor, resources: DecisionLoadResources) {
        self.bundle = bundle; self.tokenizer = tokenizer; self.backbone = backbone
        self.readout = readout; self.processor = processor; self.resources = resources
    }

    public static func load(directory: URL, tokenizerLoader: any TokenizerLoader,
        loadConfiguration: LoadConfiguration = .default) async throws -> D3Model {
        let bundle = try DecisionBundle(directory: directory)
        try Task.checkCancellation()
        let resources = DecisionLoadResources(directory: directory, configuration: loadConfiguration)
        let previousAllocatorCacheLimit = resources.previousAllocatorCacheLimit
        defer { if let cap = previousAllocatorCacheLimit { Memory.cacheLimit = cap } }
        let readoutData = try Data(contentsOf: bundle.readoutURL)
        let hash = SHA256.hash(data: readoutData).map { String(format: "%02x", $0) }.joined()
        guard hash == bundle.readoutSHA256 else { throw DecisionRuntimeError.invalid("D3 readout SHA256 mismatch") }
        let arrays = try MLX.loadArrays(data: readoutData)
        guard let readout = arrays["weight"], arrays.count == 1,
            readout.shape == bundle.readoutShape, readout.dtype == .float32 else {
            throw DecisionRuntimeError.invalid("D3 readout must be float32 [255, hidden]")
        }
        // transformers' declared Qwen2Tokenizer uses L+, unlike tokenizer.json's [L M]+.
        // Normalize only this explicit decision tokenizer in a temporary small-file view.
        let view = FileManager.default.temporaryDirectory.appendingPathComponent("d3-tokenizer-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: view, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: view) }
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            where file.pathExtension != "safetensors" && file.lastPathComponent != "tokenizer.json" {
            try FileManager.default.createSymbolicLink(at: view.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        let tokenData = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        let tokenizerConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))) as? [String: Any]
        guard tokenizerConfig?["tokenizer_class"] as? String == "Qwen2Tokenizer" else {
            throw DecisionRuntimeError.invalid("D3 requires the declared Qwen2Tokenizer contract")
        }
        func normalize(_ value: Any) -> Any {
            if let object = value as? [String: Any] {
                return object.mapValues { normalize($0) }
            }
            if let array = value as? [Any] { return array.map(normalize) }
            if let text = value as? String {
                return text.replacingOccurrences(of: "[\\p{L}\\p{M}]+", with: "\\p{L}+")
            }
            return value
        }
        let normalized = normalize(try JSONSerialization.jsonObject(with: tokenData))
        try JSONSerialization.data(withJSONObject: normalized).write(to: view.appendingPathComponent("tokenizer.json"))
        let tokenizer = try await tokenizerLoader.load(from: view)
        try D3Prompt.validateCodes(tokenizer: tokenizer, configuration: bundle.configuration)
        var configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let custom = try Qwen35JANGHPreparation.loadIfDeclared(directory: directory, configurationData: configData)
        if let custom { configData = custom.dense.ordinaryConfiguration }
        let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configData)
        guard config.textConfiguration.hiddenSize == bundle.readoutShape[1],
            config.textConfiguration.numExperts == 0, config.textConfiguration.mtpNumHiddenLayers == 0 else {
            throw DecisionRuntimeError.invalid("D3 backbone/readout dimensions or head declarations disagree")
        }
        let backbone = try custom?.construct(configurationData: configData, requesting: nil, decisionBackbone: true)
            ?? Qwen35(decisionConfiguration: config)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        try Task.checkCancellation()
        try loadWeights(modelDirectory: directory, model: backbone, quantization: base.quantization,
            perLayerQuantization: base.perLayerQuantization, jangConfig: JangLoader.loadConfig(at: directory))
        backbone.train(false)
        eval(backbone, readout)
        try Task.checkCancellation()
        var processorConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json"))) as! [String: Any]
        // Vendor D3 video normalization, distinct from unrestricted generative chat video.
        processorConfig["video_preprocessor_config"] = ["fps": 2, "min_frames": 4, "max_frames": 32,
            "min_pixels": 4096, "max_pixels": 200704 * 32] as [String: Any]
        let pc = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: processorConfig))
        return D3Model(bundle: bundle, tokenizer: tokenizer, backbone: backbone, readout: readout,
            processor: Qwen3VLProcessor(pc, tokenizer: tokenizer, videoTimestampGroups: true), resources: resources)
    }

    /// Host calls under its GPU producer gate before releasing that gate.
    public func unload() {
        backbone = nil
        readout = nil
        resources = nil
    }

    public struct BoxResult: Sendable {
        public let box: [String: Double]
        public let confidence: Double
        public let visible: Double?
        public let inputTokens: Int
        public let stateTokens: Int
        public let seconds: Double
    }

    public func decideBox(state: DecisionJSON, instructions: String, images: [URL] = [],
        videos: [URL] = [], presence: Bool = false, at seconds: Double? = nil) async throws -> BoxResult {
        guard let backbone, let readout else { throw DecisionRuntimeError.invalid("Decision model has been unloaded") }
        let kind = presence ? "track" : "box"
        guard case .object(let support)? = bundle.configuration.systemone,
            case .array(let types)? = support["question_types"], types.contains(.string(kind)) else {
            throw DecisionRuntimeError.invalid(kind + "_unsupported")
        }
        let start = Date.timeIntervalSinceReferenceDate
        let suffix = D3Box.suffix(presence: presence)
        let text = D3Box.user(state: state, instructions: instructions, codes: bundle.configuration.codes,
            presence: presence, at: seconds)
        var content: [[String: any Sendable]] = images.map { _ in ["type": "image"] }
        content += videos.map { _ in ["type": "video"] }
        content.append(["type": "text", "text": text])
        let input = UserInput(messages: [["role": "system", "content": D3Prompt.system],
            ["role": "user", "content": content]], images: images.map(UserInput.Image.url),
            videos: videos.map(UserInput.Video.url), additionalContext: ["enable_thinking": false])
        let prepared = try await processor.prepare(input: input, assistantSuffix: suffix)
        let ids = prepared.text.tokenIds ?? prepared.text.tokens.asArray(Int32.self).map(Int.init)
        let split = try D3Prompt.stateTokenCount(ids: ids, state: state, tokenizer: tokenizer)
        guard let piece = tokenizer.incrementalByteLevelDecoder else {
            throw DecisionRuntimeError.invalid("D3 Box requires lossless token offsets")
        }
        var bytes: [UInt8] = [], ends: [Int] = []
        for id in ids {
            switch piece(id) { case .bytes(let b): bytes += b; case .literal(let s): bytes += s.utf8; case .ignored: break }
            ends.append(bytes.count)
        }
        guard bytes.suffix(suffix.utf8.count).elementsEqual(suffix.utf8) else {
            throw DecisionRuntimeError.invalid("D3 Box marker suffix does not match token bytes")
        }
        var end = bytes.count - suffix.utf8.count
        let names = (presence ? ["visible"] : []) + D3Box.edgeNames
        var rows: [Int] = []
        for name in names {
            end += (name + ":").utf8.count
            guard let row = ends.lastIndex(of: end) else {
                throw DecisionRuntimeError.invalid("Box colon is not a tokenizer boundary")
            }
            rows.append(row)
            end += 1 // newline, except after the final marker (unused)
        }
        let hidden = try backbone.decisionHidden(input: prepared, stateTokenCount: split)
        var distributions: [[Float]] = []
        for (i,row) in rows.enumerated() {
            let count = presence && i == 0 ? 2 : 128
            let logits = matmul(hidden[0,row,0...].asType(.float32), readout.T)[0..<count]
                / bundle.configuration.temperature
            let p = softmax(logits, axis: -1); eval(p)
            distributions.append(p.asArray(Float.self))
        }
        let visible = presence ? Double(distributions.removeFirst()[1]) : nil
        let box = try D3Box.decode(edges: distributions)
        var table: [[Double]] = []
        if case .array(let points)? = support["box_calibration"] {
            table = try points.map { point in
                guard case .array(let xy) = point, xy.count == 2 else {
                    throw DecisionRuntimeError.invalid("Invalid Box calibration point")
                }
                return try xy.map { value in
                    switch value { case .double(let v): return v; case .int(let v): return Double(v)
                    default: throw DecisionRuntimeError.invalid("Invalid Box calibration coordinate") }
                }
            }
        }
        try Task.checkCancellation()
        return BoxResult(box: box, confidence: D3Box.confidence(edges: distributions, box: box, calibration: table),
            visible: visible, inputTokens: ids.count, stateTokens: split,
            seconds: Date.timeIntervalSinceReferenceDate - start)
    }

    public struct Result: Sendable {
        public let probabilities: [Float]
        public let answer: JSONValue
        public let inputTokens: Int
        public let stateTokens: Int
        public let seconds: Double
        public var promptTokensPerSecond: Double { Double(inputTokens) / max(seconds, 1e-9) }
    }

    public func decide(state: DecisionJSON, question: D3Question,
        images: [URL] = [], videos: [URL] = []) async throws -> Result {
        try Task.checkCancellation()
        guard let backbone, let readout else { throw DecisionRuntimeError.invalid("Decision model has been unloaded") }
        let start = Date.timeIntervalSinceReferenceDate
        let text = D3Prompt.user(state: state, question: question, codes: bundle.configuration.codes)
        var content: [[String: any Sendable]] = images.map { _ in ["type": "image"] }
        content += videos.map { _ in ["type": "video"] }
        content.append(["type": "text", "text": text])
        let input = UserInput(messages: [["role": "system", "content": D3Prompt.system],
            ["role": "user", "content": content]], images: images.map(UserInput.Image.url), videos: videos.map(UserInput.Video.url),
            additionalContext: ["enable_thinking": false])
        let prepared = try await processor.prepare(input: input)
        let ids = prepared.text.tokenIds ?? prepared.text.tokens.asArray(Int32.self).map(Int.init)
        let split = try D3Prompt.stateTokenCount(ids: ids, state: state, tokenizer: tokenizer)
        if let length = bundle.configuration.validatedLength, ids.count > length {
            FileHandle.standardError.write(Data("[D3] validated_length_exceeded tokens=\(ids.count) validated=\(length); proceeding without truncation\n".utf8))
        }
        let hidden = try backbone.decisionHidden(input: prepared,
            stateTokenCount: bundle.configuration.attentionMode == .prefixBidirectional ? split : nil)
        let last = hidden[0, -1, 0...].asType(.float32)
        let logits = matmul(last, readout.T)[0..<question.keys.count] / bundle.configuration.temperature
        let p = softmax(logits, axis: -1)
        eval(p)
        try Task.checkCancellation()
        let probabilities = p.asArray(Float.self)
        return Result(probabilities: probabilities, answer: try question.answer(probabilities: probabilities),
            inputTokens: ids.count, stateTokens: split, seconds: Date.timeIntervalSinceReferenceDate - start)
    }
}
