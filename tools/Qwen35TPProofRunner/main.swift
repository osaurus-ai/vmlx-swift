import Foundation
import MLX
import MLXDistributedTP
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
@preconcurrency import VMLXTokenizers

@main
enum Qwen35TPProofRunner {
    static func main() async {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        let rank = Int(env["MLX_RANK"] ?? "0") ?? 0
        let worldSize = Int(env["MLX_WORLD_SIZE"] ?? "1") ?? 1
        let backend = env["MLX_DIST_BACKEND"] ?? "ring"
        let strict = (env["TP_STRICT"] ?? "1") != "0"
        let modelPath = env["TP_MODEL_PATH"] ?? ""
        let outputPath = env["TP_OUTPUT_PATH"] ?? "/tmp/qwen35_tp_rank\(rank).json"
        let promptTokens = parseTokens(
            env["TP_PROMPT_TOKEN_IDS"],
            fallback: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13])
        let maxNewTokens = Int(env["TP_MAX_NEW_TOKENS"] ?? "8") ?? 8
        let cacheDir = env["TP_CACHE_DIR"] ?? "/tmp/vmlx-qwen35-tp-cache/rank\(rank)"
        let enableDiskCache = env["TP_L2_DISK_CACHE"] != "0"
        let kvMode = parseKVMode(env["TP_KV_MODE"])

        log("rank=\(rank) world_size=\(worldSize) backend=\(backend) strict=\(strict)")
        log("model_path=\(modelPath)")
        log("output_path=\(outputPath)")
        log("max_new_tokens=\(maxNewTokens) kv_mode=\(describe(kvMode))")
        log("cache_dir=\(cacheDir) disk_l2=\(enableDiskCache)")

        if env["TP_SMOKE"] == "1" {
            let group = Group(strict: strict, backend: backend)
            let input = MLXArray((0 ..< 8).map(Float.init))
            let summed = Collectives.allSum(input, group: group)
            MLX.eval(summed)
            writeJSON(outputPath, [
                "mode": "smoke",
                "rank": rank,
                "world_size": group.size,
                "backend": backend,
                "all_sum": summed.asArray(Float.self),
            ])
        }

        guard !modelPath.isEmpty else {
            fail("TP_MODEL_PATH not set")
        }

        let group = Group(strict: strict, backend: backend)
        log("group rank=\(group.rank) size=\(group.size) multi=\(group.isMultiRank)")

        let context: ModelContext
        do {
            context = try await loadModel(
                from: URL(fileURLWithPath: modelPath),
                using: #huggingFaceTokenizerLoader())
        } catch {
            fail("loadModel failed: \(error)")
        }
        log("loaded model \(type(of: context.model))")

        var replaced: Set<String> = []
        if group.isMultiRank {
            replaced = ShardingPlan.qwen35.apply(to: context.model, group: group)
            log("qwen35 sharding replacements=\(replaced.count)")
            if replaced.isEmpty {
                fail("Qwen35 sharding plan produced zero replacements for \(type(of: context.model))")
            }
            MLX.eval(context.model)
        }

        var parameters = GenerateParameters(
            maxTokens: maxNewTokens,
            kvMode: kvMode,
            temperature: 0,
            prefillStepSize: 512)
        parameters.topP = 1.0
        parameters.topK = 0
        parameters.minP = 0

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            enableDiskCache: enableDiskCache,
            diskCacheDir: URL(fileURLWithPath: cacheDir),
            modelKey: "qwen35-tp-rank\(rank)-world\(group.size)-\(type(of: context.model))",
            defaultKVMode: kvMode))

        let input = LMInput(text: LMInput.Text(tokens: MLXArray(promptTokens.map(Int32.init)).reshaped([1, promptTokens.count])))

        let started = Date()
        var iterator: TokenIterator
        do {
            iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: parameters,
                cacheCoordinator: coordinator)
        } catch {
            fail("TokenIterator init failed: \(error)")
        }
        let prefillElapsed = Date().timeIntervalSince(started)

        let decodeStarted = Date()
        var generated: [Int] = []
        while let token = iterator.next() {
            generated.append(token)
            log("generated[\(generated.count - 1)]=\(token)")
            if generated.count >= maxNewTokens {
                break
            }
        }
        let decodeElapsed = Date().timeIntervalSince(decodeStarted)
        iterator.storeCacheAfterGeneration(generatedTokenIds: generated, includeGeneratedBoundary: false)
        let snapshot = coordinator.snapshotStats()
        let decoded = context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)

        writeJSON(outputPath, [
            "mode": "qwen35_tp_decode",
            "rank": rank,
            "world_size": group.size,
            "backend": backend,
            "model_type": String(describing: type(of: context.model)),
            "sharding_plan": "qwen35",
            "sharding_replacements": replaced.sorted(),
            "sharding_replacement_count": replaced.count,
            "qwen35_ssm_boundary": "GatedDelta/SSM companion layers are replicated pending recurrent-state cache parity proof.",
            "prompt_tokens": promptTokens,
            "generated_tokens": generated,
            "decoded": decoded,
            "kv_mode": describe(kvMode),
            "cache_stats": cacheStats(snapshot),
            "prefill_seconds": prefillElapsed,
            "decode_seconds": decodeElapsed,
            "tokens_per_second": tokensPerSecond(generated.count, decodeElapsed),
        ])
    }
}

private func parseTokens(_ raw: String?, fallback: [Int]) -> [Int] {
    guard let raw, !raw.isEmpty else { return fallback }
    let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return parsed.isEmpty ? fallback : parsed
}

private func parseKVMode(_ raw: String?) -> KVQuantizationMode {
    guard let raw = raw?.lowercased(), !raw.isEmpty, raw != "none" else {
        return .none
    }
    if raw == "turboquant" || raw == "tq" {
        return .turboQuant(keyBits: 3, valueBits: 3)
    }
    if raw.hasPrefix("tq:") || raw.hasPrefix("turboquant:") {
        let bits = raw.split(separator: ":", maxSplits: 1).dropFirst().first ?? ""
        let parts = bits.split(separator: ",").compactMap { Int($0) }
        return .turboQuant(
            keyBits: parts.first ?? 3,
            valueBits: parts.dropFirst().first ?? parts.first ?? 3)
    }
    fail("unsupported TP_KV_MODE=\(raw)")
}

private func describe(_ mode: KVQuantizationMode) -> String {
    switch mode {
    case .none:
        return "none"
    case .affine(let bits, let groupSize):
        return "affine(bits:\(bits),group:\(groupSize))"
    case .turboQuant(let keyBits, let valueBits):
        return "turboquant(keyBits:\(keyBits),valueBits:\(valueBits))"
    }
}

private func cacheStats(_ snapshot: CacheCoordinatorStatsSnapshot) -> [String: Any] {
    var out: [String: Any] = [
        "paged_enabled": snapshot.pagedEnabled,
        "disk_enabled": snapshot.diskEnabled,
        "is_hybrid": snapshot.isHybrid,
        "is_paged_incompatible": snapshot.isPagedIncompatible,
        "ssm": [
            "hit_count": snapshot.ssmStats.hits,
            "miss_count": snapshot.ssmStats.misses,
            "re_derives": snapshot.ssmStats.reDerives,
        ],
    ]
    if let disk = snapshot.diskStats {
        out["disk"] = [
            "hit_count": disk.hits,
            "miss_count": disk.misses,
            "stores": disk.stores,
            "max_size_bytes": disk.maxSizeBytes,
        ]
    }
    if let paged = snapshot.pagedStats {
        out["paged"] = [
            "total_blocks": paged.totalBlocks,
            "allocated_blocks": paged.allocatedBlocks,
            "free_blocks": paged.freeBlocks,
            "hit_count": paged.cacheHits,
            "miss_count": paged.cacheMisses,
            "evictions": paged.evictions,
        ]
    }
    return out
}

private func tokensPerSecond(_ count: Int, _ elapsed: TimeInterval) -> Double {
    guard count > 0, elapsed > 0 else { return 0 }
    let value = Double(count) / elapsed
    return value.isFinite ? value : 0
}

private func writeJSON(_ path: String, _ payload: [String: Any]) -> Never {
    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        exit(0)
    } catch {
        fail("failed to write JSON \(path): \(error)")
    }
}

@inline(__always) private func log(_ message: String) {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    print("[Qwen35TPProofRunner rank=\(rank)] \(message)")
}

private func fail(_ message: String) -> Never {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    FileHandle.standardError.write(Data("[Qwen35TPProofRunner rank=\(rank)] ERROR: \(message)\n".utf8))
    exit(2)
}
