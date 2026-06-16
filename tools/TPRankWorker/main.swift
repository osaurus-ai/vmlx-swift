// Copyright © 2026 Jinho Jang. All rights reserved.
//
// TPRankWorker — minimal rank entry point for the size-2 tensor-parallel
// loopback proof. One process per rank; env vars (`MLX_RANK`,
// `MLX_WORLD_SIZE`, `MLX_JACCL_COORDINATOR`, optionally
// `MLX_IBV_DEVICES`) are read by `mlx_distributed_init` at startup.
// The launcher (`Tools/tp-launch.sh`) sets them before exec.
//
// Behavior per invocation:
//   1. `Group(strict: env["TP_STRICT"] != "0", backend: env["MLX_DIST_BACKEND"])`.
//      Default backend is `"ring"` (TCP) for first-pass validation; flip
//      to `"jaccl"` (RDMA) once ring proves the TP math correct.
//   2. Load the model bundle at `TP_MODEL_PATH`.
//   3. Apply a model-family sharding plan if `world_size > 1` (no-op on size-1).
//   4. Either:
//      - Run one prefill forward pass on a fixed token sequence
//        (`TP_PROMPT_TOKEN_IDS` env, comma-separated), or
//      - When `TP_MAX_NEW_TOKENS > 0`, run a cache-backed deterministic
//        decode loop through `TokenIterator`.
//   5. Materialize logits or generated token ids to `TP_OUTPUT_PATH`.
//   6. Exit 0.
//
// Bit-identity test: run once with `MLX_WORLD_SIZE=1` (single-rank
// baseline) producing `baseline.f32`, then run twice with
// `MLX_RANK=0/1 MLX_WORLD_SIZE=2` (size-2 TP) producing `tp_rank0.f32`.
// Compare baseline vs rank0 to within 1e-3 tolerance.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXDistributedJACCL
import MLXDistributedTP
import MLXHuggingFace
import MLXNN
@preconcurrency import VMLXTokenizers

@main
struct TPRankWorker {
    static func main() async {
        setvbuf(stdout, nil, _IONBF, 0)

        let env = ProcessInfo.processInfo.environment
        let rank = Int(env["MLX_RANK"] ?? "0") ?? 0
        let worldSize = Int(env["MLX_WORLD_SIZE"] ?? "1") ?? 1
        let backend = env["MLX_DIST_BACKEND"] ?? "ring"
        let strict = (env["TP_STRICT"] ?? "1") != "0"
        let modelPath = env["TP_MODEL_PATH"] ?? ""
        let outputPath = env["TP_OUTPUT_PATH"]
            ?? "/tmp/tp_rank\(rank).f32"
        let promptTokens: [Int] = (env["TP_PROMPT_TOKEN_IDS"] ?? "1,2,3,4,5,6,7,8")
            .split(separator: ",")
            .compactMap { Int($0) }
        let batchPromptTokens = parseBatchPromptTokens(env["TP_BATCH_PROMPT_TOKEN_IDS"])
        let smokeMode = env["TP_SMOKE"] == "1"
        let requestedPlan = env["TP_SHARDING_PLAN"]?.lowercased()
        let preShardLoad = env["TP_PRESHARD_LOAD"] == "1"
        let maxNewTokens = Int(env["TP_MAX_NEW_TOKENS"] ?? "0") ?? 0
        let maxBatchSize = Int(env["TP_MAX_BATCH_SIZE"] ?? "\(max(1, batchPromptTokens.count))")
            ?? max(1, batchPromptTokens.count)
        let kvMode = parseKVMode(env["TP_KV_MODE"])
        let enableCacheCoordinator = env["TP_ENABLE_CACHE_COORDINATOR"] == "1"
            || env["TP_PREFIX_CACHE"] == "1"
            || env["TP_L2_DISK_CACHE"] == "1"
        let enableDiskCache = env["TP_L2_DISK_CACHE"] == "1"
        let cacheDir = env["TP_CACHE_DIR"]
            ?? "/tmp/vmlx-tp-cache/\(sanitizedCacheComponent(modelPath))/rank\(rank)"
        let temperature = Float(env["TP_TEMPERATURE"] ?? "0") ?? 0
        let prefillStepSize = Int(env["TP_PREFILL_STEP_SIZE"] ?? "512") ?? 512

        log("rank=\(rank) world_size=\(worldSize) backend=\(backend) strict=\(strict)")
        log("model_path=\(modelPath)")
        log("output_path=\(outputPath)")
        log("prompt_tokens=\(promptTokens)")
        log("batch_prompt_count=\(batchPromptTokens.count) max_batch_size=\(maxBatchSize)")
        log("requested_plan=\(requestedPlan ?? "auto")")
        log("pre_shard_load=\(preShardLoad)")
        log("max_new_tokens=\(maxNewTokens)")
        log("kv_mode=\(describe(kvMode))")
        log("cache_coordinator=\(enableCacheCoordinator) l2_disk=\(enableDiskCache) cache_dir=\(cacheDir)")
        log("temperature=\(temperature) prefill_step_size=\(prefillStepSize)")
        log("smoke=\(smokeMode)")

        if smokeMode {
            // Minimal allSum smoke — no model, no sharding. If this
            // works, the issue is upstream of collectives. If this
            // hangs, the issue IS the collective itself.
            log("[smoke] init Group")
            let g = Group(strict: strict, backend: backend)
            log("[smoke] Group rank=\(g.rank) size=\(g.size)")
            let payload: [Float] = (0..<8).map { Float($0 + rank * 100) }
            let x = MLXArray(payload)
            log("[smoke] before allSum local=\(payload)")
            let summed = Collectives.allSum(x, group: g)
            MLX.eval(summed)
            let arr = summed.asArray(Float.self)
            log("[smoke] after  allSum sum=\(arr)")
            log("[smoke] DONE")
            exit(0)
        }

        guard !modelPath.isEmpty else {
            err("TP_MODEL_PATH not set")
            exit(2)
        }

        // 1. Init distributed group.
        log("initializing Group strict=\(strict) backend=\(backend)")
        if backend == "jaccl", worldSize > 1 {
            log("JACCL.isAvailable() = \(JACCL.isAvailable())")
        }
        let group = worldSize > 1
            ? Group(strict: strict, backend: backend)
            : Group.singleProcessTest(rank: rank, size: 1)
        log("Group rank=\(group.rank) size=\(group.size) isMultiRank=\(group.isMultiRank)")

        if worldSize > 1 && group.size != worldSize {
            err("declared MLX_WORLD_SIZE=\(worldSize) but mlx_distributed_init returned size=\(group.size)")
        }

        // 2. Load model bundle.
        let modelDir = URL(fileURLWithPath: modelPath)
        log("loading model from \(modelDir.path)")
        let context: ModelContext
        do {
            if preShardLoad, group.isMultiRank {
                context = try await ModelLoadHooks.$preLoadModelMutation.withValue({ model in
                    let (planName, plan) = selectShardingPlan(for: model, requested: requestedPlan)
                    log("pre-load applying ShardingPlan.\(planName) (world_size=\(group.size))")
                    let replaced = plan.apply(to: model, group: group)
                    log("pre-load sharding replaced \(replaced.count) modules")
                    if replaced.isEmpty {
                        throw TPRankWorkerError.noPreLoadShardingRewrites(modelType: String(describing: type(of: model)))
                    }
                    for path in Array(replaced.sorted().prefix(5)) {
                        log("  pre-load replaced: \(path)")
                    }
                }) {
                    try await loadModel(
                        from: modelDir, using: #huggingFaceTokenizerLoader())
                }
            } else {
                context = try await loadModel(
                    from: modelDir, using: #huggingFaceTokenizerLoader())
            }
        } catch {
            err("loadModel failed: \(error)")
            exit(3)
        }
        log("model loaded: \(type(of: context.model))")

        // 3. Apply sharding plan iff multi-rank.
        if group.isMultiRank {
            if preShardLoad {
                log("TP_PRESHARD_LOAD=1 — sharding was applied before loadWeights; skipping post-load rewrite")
            } else {
                let (planName, plan) = selectShardingPlan(for: context.model, requested: requestedPlan)
                log("applying ShardingPlan.\(planName) (world_size=\(group.size))")
                let replaced = plan.apply(to: context.model, group: group)
                log("sharding replaced \(replaced.count) Linears")
                if replaced.isEmpty {
                    err("zero modules were replaced — directive keys do not match \(type(of: context.model)); refusing unsharded TP")
                    exit(6)
                } else {
                    let sample = Array(replaced.sorted().prefix(5))
                    for path in sample {
                        log("  replaced: \(path)")
                    }
                }
            }
            MLX.eval(context.model)
        } else {
            log("world_size=1 — sharding plan is a no-op (baseline mode)")
        }

        if maxNewTokens > 0, !batchPromptTokens.isEmpty {
            let parameters = makeGenerateParameters(
                maxTokens: maxNewTokens,
                kvMode: kvMode,
                temperature: temperature,
                prefillStepSize: prefillStepSize)
            let coordinator = makeCacheCoordinator(
                enabled: enableCacheCoordinator,
                enableDiskCache: enableDiskCache,
                cacheDir: cacheDir,
                rank: rank,
                groupSize: group.size,
                model: context.model,
                kvMode: kvMode)
            log("running BatchEngine continuous decode for \(batchPromptTokens.count) prompts")
            let batchResult = await runBatchDecode(
                context: context,
                promptBatches: batchPromptTokens,
                parameters: parameters,
                maxBatchSize: maxBatchSize,
                group: group,
                coordinator: coordinator)
            tryWriteJSON(
                outputPath: outputPath,
                payload: [
                    "rank": rank,
                    "world_size": group.size,
                    "backend": backend,
                    "model_type": String(describing: type(of: context.model)),
                    "mode": "batch",
                    "prompt_batches": batchPromptTokens,
                    "results": batchResult.results,
                    "engine": batchResult.engine,
                    "kv_mode": describe(kvMode),
                    "token_authority": group.isMultiRank ? "rank0_send_recv_per_slot" : "local",
                    "cache_coordinator": enableCacheCoordinator,
                    "l2_disk_cache": enableDiskCache,
                    "cache_dir": cacheDir,
                    "cache_stats": cacheStatsDictionary(coordinator?.snapshotStats()),
                ])
            log("batch decode wrote \(batchResult.results.count) result(s) to \(outputPath)")
            log("DONE rank=\(rank) — exiting 0")
            exit(0)
        }

        if maxNewTokens > 0 {
            let parameters = makeGenerateParameters(
                maxTokens: maxNewTokens,
                kvMode: kvMode,
                temperature: temperature,
                prefillStepSize: prefillStepSize)

            let coordinator = makeCacheCoordinator(
                enabled: enableCacheCoordinator,
                enableDiskCache: enableDiskCache,
                cacheDir: cacheDir,
                rank: rank,
                groupSize: group.size,
                model: context.model,
                kvMode: kvMode)

            log("running cache-backed decode for \(maxNewTokens) generated tokens")
            let promptArray = MLXArray(promptTokens.map { Int32($0) })
                .reshaped([1, promptTokens.count])
            let input = LMInput(text: LMInput.Text(tokens: promptArray))
            let samplerOverride: LogitSampler? = group.isMultiRank
                ? RankZeroTokenSampler(base: parameters.sampler(), group: group)
                : nil
            let iteratorStart = Date()
            var iterator: TokenIterator
            do {
                iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    parameters: parameters,
                    cacheCoordinator: coordinator,
                    samplerOverride: samplerOverride)
            } catch {
                err("TokenIterator init failed: \(error)")
                exit(7)
            }
            let prefillElapsed = Date().timeIntervalSince(iteratorStart)

            let generationStart = Date()
            var generated: [Int] = []
            while let token = iterator.next() {
                generated.append(token)
                log("generated[\(generated.count - 1)]=\(token)")
                if generated.count >= maxNewTokens { break }
            }
            let generationElapsed = Date().timeIntervalSince(generationStart)
            let cacheDiagnostics = iterator.cacheDiagnostics()
            iterator.storeCacheAfterGeneration(
                generatedTokenIds: generated,
                includeGeneratedBoundary: false)
            let totalElapsed = Date().timeIntervalSince(iteratorStart)
            let decoded = context.tokenizer.decode(
                tokenIds: generated,
                skipSpecialTokens: false)
            let snapshot = coordinator?.snapshotStats()
            let promptTPS = tokensPerSecond(count: promptTokens.count, elapsed: prefillElapsed)
            let decodeTPS = tokensPerSecond(count: generated.count, elapsed: generationElapsed)
            let totalTPS = tokensPerSecond(
                count: promptTokens.count + generated.count,
                elapsed: totalElapsed)
            tryWriteJSON(
                outputPath: outputPath,
                payload: [
                    "rank": rank,
                    "world_size": group.size,
                    "backend": backend,
                    "model_type": String(describing: type(of: context.model)),
                    "prompt_tokens": promptTokens,
                    "generated_tokens": generated,
                    "decoded": decoded,
                    "kv_mode": describe(kvMode),
                    "token_authority": group.isMultiRank ? "rank0_send_recv" : "local",
                    "cache_coordinator": enableCacheCoordinator,
                    "l2_disk_cache": enableDiskCache,
                    "cache_dir": cacheDir,
                    "live_cache": cacheDiagnostics,
                    "cache_stats": cacheStatsDictionary(snapshot),
                    "prompt_time": prefillElapsed,
                    "generate_time": generationElapsed,
                    "elapsed_ms": totalElapsed * 1000,
                    "prompt_tps": promptTPS,
                    "decode_tps": decodeTPS,
                    "tokens_per_second": totalTPS,
                ])
            log(String(format: "timing prompt_time=%.3fs generate_time=%.3fs prompt_tps=%.3f decode_tps=%.3f total_tps=%.3f",
                       prefillElapsed, generationElapsed, promptTPS, decodeTPS, totalTPS))
            log("decode wrote \(generated.count) tokens to \(outputPath)")
            log("DONE rank=\(rank) — exiting 0")
            exit(0)
        }

        // 4. Run one forward pass on the fixed token sequence.
        log("running forward pass on \(promptTokens.count) tokens")
        let tokens = MLXArray(promptTokens.map { Int32($0) })
            .reshaped([1, promptTokens.count])
        let logits = context.model(tokens, cache: nil)
        MLX.eval(logits)
        log("logits shape=\(logits.shape) dtype=\(logits.dtype)")

        // 5. Cast to fp32 and dump.
        let logitsFP32 = logits.asType(.float32)
        MLX.eval(logitsFP32)

        let shape = logitsFP32.shape
        let count = shape.reduce(1, *)
        let floats: [Float] = logitsFP32.asArray(Float.self)
        let data = floats.withUnsafeBytes { Data($0) }
        let expectedBytes = count * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            err("logits byte count mismatch: got \(data.count), expected \(expectedBytes)")
            exit(4)
        }

        // Write [ndim u32][dim u32 × ndim][count u64][raw f32 bytes].
        var header = Data()
        var ndim = UInt32(shape.count).littleEndian
        header.append(Data(bytes: &ndim, count: MemoryLayout<UInt32>.size))
        for d in shape {
            var dimVal = UInt32(d).littleEndian
            header.append(Data(bytes: &dimVal, count: MemoryLayout<UInt32>.size))
        }
        var n = UInt64(count).littleEndian
        header.append(Data(bytes: &n, count: MemoryLayout<UInt64>.size))

        let outURL = URL(fileURLWithPath: outputPath)
        do {
            try (header + data).write(to: outURL)
        } catch {
            err("failed to write \(outputPath): \(error)")
            exit(5)
        }
        log("wrote \(expectedBytes) bytes (\(count) Float32) to \(outputPath)")
        log("DONE rank=\(rank) — exiting 0")
    }
}

private enum TPRankWorkerError: Error, CustomStringConvertible {
    case noPreLoadShardingRewrites(modelType: String)

    var description: String {
        switch self {
        case .noPreLoadShardingRewrites(let modelType):
            return "TP_PRESHARD_LOAD=1 produced zero sharding rewrites for \(modelType)"
        }
    }
}

private struct RankZeroTokenSampler: LogitSampler {
    let base: LogitSampler
    let group: Group

    func sample(logits: MLXArray) -> MLXArray {
        guard group.isMultiRank else {
            return base.sample(logits: logits)
        }

        if group.rank == 0 {
            let token = base.sample(logits: logits)
            var dependencies: [MLXArray] = [token]
            for dst in 1 ..< group.size {
                dependencies.append(Collectives.send(token, to: dst, group: group))
            }
            MLX.eval(dependencies)
            return token
        }

        // Force the local forward/cache-update graph to materialize even
        // though this rank does not own sampling.
        MLX.eval(logits)
        let like = MLXArray([Int32(0)])
        let token = Collectives.recvLike(like, from: 0, group: group)
        MLX.eval(token)
        return token
    }
}

private func makeGenerateParameters(
    maxTokens: Int,
    kvMode: KVQuantizationMode,
    temperature: Float,
    prefillStepSize: Int
) -> GenerateParameters {
    var parameters = GenerateParameters(
        maxTokens: maxTokens,
        kvMode: kvMode,
        temperature: temperature,
        prefillStepSize: prefillStepSize)
    if temperature == 0 {
        // Keep every rank deterministic. Stochastic sampling needs an
        // explicit rank-0 token broadcast path before production use.
        parameters.topP = 1.0
        parameters.topK = 0
        parameters.minP = 0
    }
    return parameters
}

private func makeCacheCoordinator(
    enabled: Bool,
    enableDiskCache: Bool,
    cacheDir: String,
    rank: Int,
    groupSize: Int,
    model: Module,
    kvMode: KVQuantizationMode
) -> CacheCoordinator? {
    guard enabled else { return nil }
    let cfg = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: enableDiskCache,
        diskCacheDir: URL(fileURLWithPath: cacheDir),
        modelKey: "tp-rank\(rank)-world\(groupSize)-\(type(of: model))",
        defaultKVMode: kvMode)
    log("cache coordinator enabled: paged=true disk=\(enableDiskCache) rank_namespaced=true")
    return CacheCoordinator(config: cfg)
}

private struct BatchDecodeResult: Sendable {
    let index: Int
    let promptTokens: [Int]
    let generatedTokens: [Int]
    let decoded: String
    let info: CompletionInfoPayload
}

private struct CompletionInfoPayload: Sendable {
    let promptTokenCount: Int
    let generationTokenCount: Int
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let stopReason: String
    let tokensPerSecond: Double
    let unclosedReasoning: Bool
}

private func runBatchDecode(
    context: ModelContext,
    promptBatches: [[Int]],
    parameters: GenerateParameters,
    maxBatchSize: Int,
    group: Group,
    coordinator: CacheCoordinator?
) async -> (results: [[String: Any]], engine: [String: Any]) {
    let engine = BatchEngine(
        context: context,
        maxBatchSize: max(1, maxBatchSize),
        cacheCoordinator: coordinator)
    var streams: [(index: Int, prompt: [Int], stream: AsyncStream<BatchGeneration>)] = []
    for (index, prompt) in promptBatches.enumerated() {
        let promptArray = MLXArray(prompt.map { Int32($0) })
            .reshaped([1, prompt.count])
        let input = LMInput(text: LMInput.Text(tokens: promptArray))
        let samplerOverride: BatchSamplerOverride? = group.isMultiRank
            ? BatchSamplerOverride(RankZeroTokenSampler(base: parameters.sampler(), group: group))
            : nil
        let (_, stream) = await engine.submit(
            input: input,
            parameters: parameters,
            samplerOverride: samplerOverride)
        streams.append((index: index, prompt: prompt, stream: stream))
    }

    let results = await withTaskGroup(of: BatchDecodeResult.self) { group in
        for item in streams {
            group.addTask {
                var tokens: [Int] = []
                var infoPayload = CompletionInfoPayload.empty
                for await event in item.stream {
                    switch event {
                    case .token(let token):
                        tokens.append(token)
                    case .info(let info):
                        infoPayload = completionInfoDictionary(info)
                    }
                }
                return BatchDecodeResult(
                    index: item.index,
                    promptTokens: item.prompt,
                    generatedTokens: tokens,
                    decoded: context.tokenizer.decode(
                        tokenIds: tokens,
                        skipSpecialTokens: false),
                    info: infoPayload)
            }
        }
        var collected: [BatchDecodeResult] = []
        for await result in group {
            collected.append(result)
        }
        return collected.sorted {
            $0.index < $1.index
        }
    }

    let highWater = await engine.activeCountHighWatermarkForDiagnostics
    let compatibilitySplits = await engine.decodeCompatibilitySplitCountForDiagnostics
    let tqCompressions = await engine.turboQuantCompressionCountForDiagnostics
    await engine.shutdown()
    return (
        results.map(batchResultDictionary),
        [
            "max_batch_size": max(1, maxBatchSize),
            "active_high_watermark": highWater,
            "decode_compatibility_splits": compatibilitySplits,
            "turboquant_compressions": tqCompressions,
            "token_authority": group.isMultiRank ? "rank0_send_recv_per_slot" : "local",
        ])
}

private func batchResultDictionary(_ result: BatchDecodeResult) -> [String: Any] {
    [
        "index": result.index,
        "prompt_tokens": result.promptTokens,
        "generated_tokens": result.generatedTokens,
        "decoded": result.decoded,
        "info": completionInfoDictionary(result.info),
    ]
}

private func completionInfoDictionary(_ info: CompletionInfoPayload) -> [String: Any] {
    [
        "prompt_token_count": info.promptTokenCount,
        "generation_token_count": info.generationTokenCount,
        "prompt_time": info.promptTime,
        "generate_time": info.generateTime,
        "stop_reason": info.stopReason,
        "tokens_per_second": info.tokensPerSecond,
        "unclosed_reasoning": info.unclosedReasoning,
    ]
}

private func completionInfoDictionary(_ info: GenerateCompletionInfo) -> CompletionInfoPayload {
    CompletionInfoPayload(
        promptTokenCount: info.promptTokenCount,
        generationTokenCount: info.generationTokenCount,
        promptTime: info.promptTime,
        generateTime: info.generateTime,
        stopReason: describe(info.stopReason),
        tokensPerSecond: info.tokensPerSecond.isFinite ? info.tokensPerSecond : 0,
        unclosedReasoning: info.unclosedReasoning)
}

private extension CompletionInfoPayload {
    static let empty = CompletionInfoPayload(
        promptTokenCount: 0,
        generationTokenCount: 0,
        promptTime: 0,
        generateTime: 0,
        stopReason: "missing",
        tokensPerSecond: 0,
        unclosedReasoning: false)
}

private func describe(_ reason: GenerateStopReason) -> String {
    switch reason {
    case .stop:
        return "stop"
    case .length:
        return "length"
    case .cancelled:
        return "cancelled"
    }
}

private func parseBatchPromptTokens(_ raw: String?) -> [[Int]] {
    guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return []
    }
    return raw.split(separator: ";").map { prompt in
        prompt.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }.filter { !$0.isEmpty }
}

private func selectShardingPlan(
    for model: Module,
    requested: String?
) -> (name: String, plan: ShardingPlan) {
        if let requested {
            switch requested {
        case "mimo", "mimo_v2", "mimo-v2", "mimo_v2_flash", "mimov2":
            return ("mimoV2", .mimoV2)
        case "llama", "mistral":
            return ("llama", .llama)
        default:
            fatalError("Unsupported TP_SHARDING_PLAN=\(requested). Supported: auto, mimo_v2, llama")
        }
    }

    if model is MiMoV2FlashModel {
        return ("mimoV2", .mimoV2)
    }
    return ("llama", .llama)
}

private func parseKVMode(_ raw: String?) -> KVQuantizationMode {
    guard let raw = raw?.lowercased(), !raw.isEmpty, raw != "none" else {
        return .none
    }
    if raw == "turboquant" || raw == "tq" {
        return .turboQuant(keyBits: 3, valueBits: 3)
    }
    if raw.hasPrefix("turboquant:") || raw.hasPrefix("tq:") {
        let bits = raw.split(separator: ":", maxSplits: 1).dropFirst().first ?? ""
        let parts = bits.split(separator: ",").compactMap { Int($0) }
        return .turboQuant(
            keyBits: parts.first ?? 3,
            valueBits: parts.dropFirst().first ?? parts.first ?? 3)
    }
    fatalError("Unsupported TP_KV_MODE=\(raw). Supported: none, turboquant, tq, tq:K,V")
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

private func sanitizedCacheComponent(_ value: String) -> String {
    let fallback = "model"
    guard !value.isEmpty else { return fallback }
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
    let mapped = value.unicodeScalars.map { scalar -> Character in
        allowed.contains(scalar) ? Character(scalar) : "_"
    }
    let result = String(mapped).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    return result.isEmpty ? fallback : result
}

private func cacheStatsDictionary(_ snapshot: CacheCoordinatorStatsSnapshot?) -> [String: Any] {
    guard let snapshot else { return [:] }
    var result: [String: Any] = [
        "paged_enabled": snapshot.pagedEnabled,
        "disk_enabled": snapshot.diskEnabled,
        "is_hybrid": snapshot.isHybrid,
        "is_paged_incompatible": snapshot.isPagedIncompatible,
    ]
    if let paged = snapshot.pagedStats {
        result["paged"] = [
            "total_blocks": paged.totalBlocks,
            "allocated_blocks": paged.allocatedBlocks,
            "free_blocks": paged.freeBlocks,
            "hit_count": paged.cacheHits,
            "miss_count": paged.cacheMisses,
            "evictions": paged.evictions,
        ]
    }
    if let disk = snapshot.diskStats {
        result["disk"] = [
            "hit_count": disk.hits,
            "miss_count": disk.misses,
            "stores": disk.stores,
            "max_size_bytes": disk.maxSizeBytes,
        ]
    }
    result["ssm"] = [
        "hit_count": snapshot.ssmStats.hits,
        "miss_count": snapshot.ssmStats.misses,
        "re_derives": snapshot.ssmStats.reDerives,
    ]
    return result
}

private func tokensPerSecond(count: Int, elapsed: TimeInterval) -> Double {
    guard count > 0, elapsed > 0 else { return 0 }
    let value = Double(count) / elapsed
    return value.isFinite ? value : 0
}

private func tryWriteJSON(outputPath: String, payload: [String: Any]) {
    do {
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
    } catch {
        err("failed to write JSON \(outputPath): \(error)")
        exit(8)
    }
}

// MARK: - Logging helpers

@inline(__always) private func log(_ msg: String) {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    print("[TPRankWorker rank=\(rank)] \(msg)")
}

@inline(__always) private func err(_ msg: String) {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    FileHandle.standardError.write(Data(
        "[TPRankWorker rank=\(rank)] ERROR: \(msg)\n".utf8))
}
