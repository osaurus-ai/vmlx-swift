import Foundation
import MLXPress

@main
struct MLXPressSelfCheck {
    static func main() throws {
        try checkBundleFacts()
        try checkActivityCompressionGate()
        try checkMemorySnapshotTelemetry()
        try checkReadinessChecklist()
        try checkLoadConfigurationDefaults()
        try checkBaseDecodeParameters()
        print("MLXPress self-check passed")
    }
}

private func checkBundleFacts() throws {
    let directory = try TemporaryModelDirectory()
    defer { directory.remove() }

    try directory.writeJSON(
        name: "config.json",
        object: [
            "model_type": "test_moe",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        ])
    try directory.writeJSON(name: "tokenizer.json", object: [:])
    try directory.writeJSON(name: "model.safetensors.index.json", object: [:])
    try directory.writeBytes(name: "model-00001-of-00001.safetensors", count: 1_000)
    try directory.writeBytes(name: "jangtq_runtime.safetensors", count: 7)

    let facts = MLXPressBundleFacts.inspect(at: directory.url)

    try expect(facts.format == .jangTQ, "expected JANGTQ format")
    try expect(facts.modelType == "test_moe", "expected model_type")
    try expect(facts.totalSafetensorsBytes == 1_007, "expected safetensors byte count")
    try expect(facts.isRouted, "expected routed bundle")
    try expect(facts.numRoutedExperts == 8, "expected routed expert count")
    try expect(facts.topK == 2, "expected top-k count")
    try expect(facts.hasTokenizerJSON, "expected tokenizer.json")
    try expect(facts.hasSafetensorsIndex, "expected safetensors index")
    try expect(facts.autoCompressionEligible, "expected auto compression eligibility")

    let nested = try TemporaryModelDirectory()
    defer { nested.remove() }
    try nested.writeJSON(
        name: "config.json",
        object: [
            "model_type": "wrapper",
            "text_config": [
                "model_type": "inner_moe",
                "num_experts": 16,
                "top_k_experts": 4,
            ],
        ])
    try nested.writeBytes(name: "weights.safetensors", count: 5)

    let nestedFacts = MLXPressBundleFacts.inspect(at: nested.url)
    try expect(nestedFacts.format == .mlx, "expected plain MLX format")
    try expect(nestedFacts.modelType == "wrapper", "expected outer model_type precedence")
    try expect(nestedFacts.totalSafetensorsBytes == 5, "expected nested byte count")
    try expect(nestedFacts.isRouted, "expected nested routed metadata")
    try expect(nestedFacts.numRoutedExperts == 16, "expected nested routed expert count")
    try expect(nestedFacts.topK == 4, "expected nested top-k count")
}

private func checkActivityCompressionGate() throws {
    let directory = try TemporaryModelDirectory()
    defer { directory.remove() }
    try directory.writeJSON(name: "config.json", object: ["model_type": "test"])
    try directory.writeBytes(name: "weights.safetensors", count: 1_000)
    let facts = MLXPressBundleFacts.inspect(at: directory.url)

    let pass = MLXPressActivityCompressionCheck(
        bundleFacts: facts,
        preLoad: memory(physicalFootprintBytes: 10_000),
        postLoad: memory(physicalFootprintBytes: 10_300),
        maxFootprintPercent: 30)
    try expect(pass.verdict == .passed, "expected activity gate pass at threshold")
    try expect(pass.footprintIncreaseBytes == 300, "expected pass byte delta")
    try expect(pass.maxAllowedFootprintIncreaseBytes == 300, "expected max allowed bytes")
    try expect(pass.footprintIncreasePercent == 30, "expected pass ratio")

    let fail = MLXPressActivityCompressionCheck(
        bundleFacts: facts,
        preLoad: memory(physicalFootprintBytes: 10_000),
        postLoad: memory(physicalFootprintBytes: 10_301),
        maxFootprintPercent: 30)
    try expect(fail.verdict == .failed, "expected activity gate fail above threshold")
    try expect(fail.footprintIncreaseBytes == 301, "expected fail byte delta")

    let noModelBytes = try TemporaryModelDirectory()
    defer { noModelBytes.remove() }
    try noModelBytes.writeJSON(name: "config.json", object: ["model_type": "test"])
    let zeroFacts = MLXPressBundleFacts.inspect(at: noModelBytes.url)
    let unavailable = MLXPressActivityCompressionCheck(
        bundleFacts: zeroFacts,
        preLoad: memory(physicalFootprintBytes: 10_000),
        postLoad: memory(physicalFootprintBytes: 10_001),
        maxFootprintPercent: 30)
    try expect(unavailable.verdict == .unavailable, "expected unavailable gate without model bytes")
}

private func checkMemorySnapshotTelemetry() throws {
    let snapshot = MLXPressMemorySnapshot.current()
    try expect(snapshot.physicalMemoryBytes > 0, "expected physical memory telemetry")
    try expect(
        snapshot.mlxPeakMemoryBytes >= snapshot.mlxActiveMemoryBytes,
        "expected MLX peak memory to cover active memory")
}

private func checkReadinessChecklist() throws {
    let kimi = try TemporaryModelDirectory(name: "Kimi-K2.6-Small-JANGTQ")
    defer { kimi.remove() }
    try kimi.writeJSON(
        name: "config.json",
        object: [
            "model_type": "kimi_k25",
            "text_config": [
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "kimi_k2",
                "n_routed_experts": 211,
                "num_experts_per_tok": 8,
                "qk_rope_head_dim": 64,
                "rope_theta": 50_000,
                "rope_scaling": ["type": "yarn"],
            ],
            "vision_config": [
                "mm_hidden_size": 1152,
                "video_attn_type": "spatial_temporal",
            ],
        ])
    try kimi.writeJSON(name: "jang_config.json", object: ["has_vision": false])
    try kimi.writeBytes(name: "model.safetensors", count: 1_000)
    try kimi.writeBytes(name: "jangtq_runtime.safetensors", count: 7)

    let kimiFacts = MLXPressBundleFacts.inspect(at: kimi.url)
    let kimiReadiness = MLXPressModelReadinessChecklist.build(for: kimiFacts)
    try expect(kimiReadiness.family == "Kimi K2.x", "expected Kimi readiness family")
    try expect(
        kimiFacts.architecture.attentionKinds.contains("routed-moe"),
        "expected Kimi routed-MoE architecture signal")
    try expect(
        kimiFacts.architecture.attentionKinds.contains("mla-partial-rope-attention"),
        "expected Kimi MLA partial-RoPE attention signal")
    try expect(
        kimiFacts.architecture.matmulKinds.contains("jangtq-hadamard-rotation"),
        "expected Kimi JANGTQ Hadamard matmul signal")
    try expect(
        kimiFacts.architecture.matmulKinds.contains("randomized-hadamard-pow2-blocks"),
        "expected Kimi non-power-of-two Hadamard block signal")
    try expect(
        kimiFacts.architecture.positionEncodings.contains { $0.contains("partial-rope") },
        "expected Kimi partial RoPE architecture signal")
    try expect(
        kimiFacts.architecture.positionVectorKinds.contains { $0.contains("partial-rope-head-vector") },
        "expected Kimi partial RoPE vector signal")
    try expect(
        kimiFacts.architecture.cacheEncodingKinds.contains("tqdiskserializer-v2-layer-kind-tags"),
        "expected Kimi cache encoder signal")
    try expect(
        kimiFacts.architecture.cacheEncodingKinds.contains { $0.contains("cache-policy-salt") },
        "expected Kimi cache-policy salt signal")
    try expect(
        !kimiFacts.architecture.attentionKinds.contains("vl-encoder-plus-text-attention"),
        "expected explicit Kimi has_vision=false to suppress media gates")
    try expect(
        kimiFacts.architecture.mediaCacheKinds.isEmpty,
        "expected explicit Kimi has_vision=false to suppress media cache gates")
    try expect(kimiReadiness.overallState == .partial, "expected Kimi partial readiness until long proof")
    try expect(
        kimiReadiness.items.contains { $0.gate == "kimi-short-no-thinking-decode" && $0.state == .partial },
        "expected Kimi short decode partial gate")
    try expect(
        kimiReadiness.items.contains { $0.gate == "hadamard-tq-matmul-contract" && $0.state == .partial },
        "expected Kimi Hadamard/TQ matmul partial gate")
    try expect(
        kimiReadiness.items.contains { $0.gate == "cache-block-storage-encode" && $0.state == .partial },
        "expected Kimi cache block storage gate")
    try expect(
        kimiReadiness.items.contains { $0.gate == "parser-autodetect-stack" && $0.state == .partial },
        "expected Kimi parser autodetect gate")
    try expect(
        kimiReadiness.items.contains { $0.gate == "cold-warm-deviation-proof" && $0.state == .partial },
        "expected Kimi cold/warm deviation gate")
    try expect(
        kimiReadiness.items.contains { $0.gate == "synchronous-prompt-boundary-rederive" && $0.state == .partial },
        "expected Kimi routed prompt-boundary rederive gate")
    try expect(
        kimiReadiness.requiredProofs.contains { $0.contains("longer Kimi MLXPress cache-stack decode") },
        "expected Kimi long decode proof requirement")
    try expect(
        kimiReadiness.requiredProofs.contains { $0.contains("per-turn artifact") },
        "expected per-turn proof artifact requirement")

    let qwenVL = try TemporaryModelDirectory(name: "Qwen3-VL-JANGTQ")
    defer { qwenVL.remove() }
    try qwenVL.writeJSON(
        name: "config.json",
        object: [
            "model_type": "qwen3_vl",
            "num_hidden_layers": 2,
            "rope_scaling": ["mrope_section": [1, 1, 1]],
            "vision_config": ["model_type": "qwen3_vl"],
        ])
    try qwenVL.writeBytes(name: "model.safetensors", count: 1_000)

    let qwenVLFacts = MLXPressBundleFacts.inspect(at: qwenVL.url)
    let qwenVLReadiness = MLXPressModelReadinessChecklist.build(for: qwenVLFacts)
    try expect(qwenVLReadiness.family == "Qwen VL", "expected Qwen VL readiness family")
    try expect(
        qwenVLFacts.architecture.positionEncodings.contains("2d-3d-mrope-grid"),
        "expected Qwen VL 2D/3D MRoPE architecture signal")
    try expect(
        qwenVLFacts.architecture.positionVectorKinds.contains { $0.contains("qwen-vl-image-grid-thw") },
        "expected Qwen VL image grid vector signal")
    try expect(
        qwenVLFacts.architecture.positionVectorKinds.contains { $0.contains("qwen-vl-video-grid-thw") },
        "expected Qwen VL video grid vector signal")
    try expect(
        qwenVLFacts.architecture.positionVectorKinds.contains("qwen-vl-position-ids[3,batch,seq]"),
        "expected Qwen VL 3-axis position IDs vector signal")
    try expect(
        qwenVLFacts.architecture.positionVectorKinds.contains("qwen-vl-mrope-delta-vector"),
        "expected Qwen VL MRoPE delta vector signal")
    try expect(
        qwenVLFacts.architecture.cacheEncodingKinds.contains { $0.contains("media-scope-salt") },
        "expected Qwen VL media cache encoding signal")
    try expect(
        qwenVLFacts.architecture.cacheEncodingKinds.contains("vl-partial-hit-rejected-for-media-token-region"),
        "expected Qwen VL partial media hit rejection signal")
    try expect(
        qwenVLFacts.architecture.mediaCacheKinds.contains("mrope-grid-position-ids"),
        "expected Qwen VL media cache architecture signal")
    try expect(qwenVLReadiness.overallState == .blocked, "expected Qwen VL blocked until media proof")
    try expect(
        qwenVLReadiness.items.contains { $0.gate == "vl-vector-media-cache-proof" && $0.state == .missing },
        "expected Qwen VL media cache proof gate")
    try expect(
        qwenVLReadiness.requiredProofs.contains { $0.contains("2D/3D MRoPE") },
        "expected Qwen VL MRoPE proof requirement")

    let qwen36Vision = try TemporaryModelDirectory(name: "Qwen3.6-27B-JANG_4M-CRACK")
    defer { qwen36Vision.remove() }
    try qwen36Vision.writeJSON(
        name: "config.json",
        object: [
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": [
                "model_type": "qwen3_5_text",
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 16,
                "rope_parameters": [
                    "rope_type": "default",
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                    "mrope_interleaved": true,
                ],
            ],
            "vision_config": [
                "model_type": "qwen3_5",
                "patch_size": 16,
                "temporal_patch_size": 2,
            ],
        ])
    try qwen36Vision.writeBytes(name: "model.safetensors", count: 1_000)

    let qwen36VisionFacts = MLXPressBundleFacts.inspect(at: qwen36Vision.url)
    let qwen36VisionReadiness = MLXPressModelReadinessChecklist.build(for: qwen36VisionFacts)
    try expect(qwen36VisionReadiness.family == "Qwen3.6 MoE", "expected Qwen3.6 readiness family")
    try expect(
        qwen36VisionFacts.architecture.attentionKinds.contains("vl-encoder-plus-text-attention"),
        "expected non-empty Qwen vision_config to mark VLM attention")
    try expect(
        qwen36VisionFacts.architecture.positionVectorKinds.contains("qwen-vl-position-ids[3,batch,seq]"),
        "expected Qwen3.6 vision config to require Qwen VL position IDs")
    try expect(
        qwen36VisionFacts.architecture.mediaCacheKinds.contains("mrope-grid-position-ids"),
        "expected Qwen3.6 vision config to require media cache identity")
    try expect(qwen36VisionReadiness.overallState == .blocked, "expected Qwen3.6 VLM blocked until media proof")
    try expect(
        qwen36VisionReadiness.items.contains { $0.gate == "vl-vector-media-cache-proof" && $0.state == .missing },
        "expected Qwen3.6 VLM media cache proof gate")
    try expect(
        qwen36VisionReadiness.requiredProofs.contains { $0.contains("real media-cache proof") },
        "expected detected vision config media proof requirement")

    let minimax = try TemporaryModelDirectory(name: "MiniMax-M2.7-JANGTQ_K-CRACK")
    defer { minimax.remove() }
    try minimax.writeJSON(
        name: "config.json",
        object: [
            "model_type": "minimax_m2",
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
        ])
    try minimax.writeBytes(name: "model.safetensors", count: 1_000)
    try minimax.writeBytes(name: "jangtq_runtime.safetensors", count: 7)

    let minimaxReadiness = MLXPressModelReadinessChecklist.build(
        for: MLXPressBundleFacts.inspect(at: minimax.url))
    try expect(minimaxReadiness.family == "MiniMax M2", "expected MiniMax readiness family")
    try expect(minimaxReadiness.overallState == .partial, "expected MiniMax partial readiness")
    try expect(
        minimaxReadiness.items.contains { $0.gate == "minimax-thinking-on-final-answer" && $0.state == .partial },
        "expected MiniMax thinking-on partial gate")
}

private func checkLoadConfigurationDefaults() throws {
    let defaultConfiguration = MLXPressLoadConfiguration.default
    let plainConfiguration = MLXPressLoadConfiguration.plain

    try expect(defaultConfiguration.compression == .auto(envFallback: true), "expected default auto compression")
    try expect(defaultConfiguration.allocatorCacheLimit == .decodeCacheDefault, "expected default allocator cap")
    try expect(defaultConfiguration.memoryLimit == .default, "expected default memory limit")
    try expect(defaultConfiguration.useMmapSafetensors, "expected default mmap loader")
    try expect(!defaultConfiguration.enableRouterAdvice, "expected router advice default-off")
    try expect(defaultConfiguration.disableDecodeFusedGateUp, "expected fused gate/up disabled")
    try expect(!defaultConfiguration.enableActiveExpertStreaming, "expected active streaming default-off")
    try expect(defaultConfiguration.cache.enabled, "expected cache stack default-on")
    try expect(!defaultConfiguration.cache.usePagedCache, "expected paged cache default-off")
    try expect(defaultConfiguration.cache.enableDiskCache, "expected disk cache default-on")
    try expect(defaultConfiguration.cache.defaultKVMode == .turboQuant(keyBits: 3, valueBits: 3), "expected TurboQuant KV default")

    try expect(plainConfiguration.compression == .disabled, "expected plain compression disabled")
    try expect(plainConfiguration.allocatorCacheLimit == .unlimited, "expected plain allocator cap unlimited")
    try expect(plainConfiguration.memoryLimit == .unlimited, "expected plain memory unlimited")
    try expect(!plainConfiguration.useMmapSafetensors, "expected plain mmap disabled")
    try expect(!plainConfiguration.enableRouterAdvice, "expected plain router advice disabled")
    try expect(!plainConfiguration.disableDecodeFusedGateUp, "expected plain fused path allowed")
    try expect(!plainConfiguration.enableActiveExpertStreaming, "expected plain active streaming disabled")
    try expect(!plainConfiguration.cache.enabled, "expected plain cache stack disabled")
}

private func checkBaseDecodeParameters() throws {
    let defaults = MLXPressDefaultGenerateParameters()
    try expect(defaults.draftStrategy?.kindName == "none", "expected default MTP/spec-decode off")

    var requested = MLXPressDefaultGenerateParameters()
    requested.draftStrategy = .dflash(
        drafterPath: URL(fileURLWithPath: "/tmp/not-loaded"),
        blockSize: 4)
    let normalized = MLXPressBaseDecodeParameters(requested)
    try expect(normalized.draftStrategy?.kindName == "none", "expected base decode to force MTP/spec-decode off")
}

private func memory(physicalFootprintBytes: UInt64?) -> MLXPressMemorySnapshot {
    MLXPressMemorySnapshot(
        residentSizeBytes: physicalFootprintBytes ?? 0,
        physicalFootprintBytes: physicalFootprintBytes,
        physicalMemoryBytes: 128 * 1024 * 1024)
}

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() {
        throw SelfCheckError.failed(message)
    }
}

private enum SelfCheckError: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

private struct TemporaryModelDirectory {
    let url: URL

    init(name: String? = nil) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxpress-selfcheck", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        url = root.appendingPathComponent(name ?? UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true)
    }

    func writeJSON(name: String, object: [String: Any]) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys])
        try data.write(to: url.appendingPathComponent(name))
    }

    func writeBytes(name: String, count: Int) throws {
        try Data(repeating: 0, count: count)
            .write(to: url.appendingPathComponent(name))
    }

    func remove() {
        try? FileManager.default.removeItem(at: url)
    }
}
