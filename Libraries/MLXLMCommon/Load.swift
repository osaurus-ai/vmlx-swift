// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private func isPreservedMTPWeightKey(_ key: String) -> Bool {
    let lower = key.lowercased()
    return lower.hasPrefix("mtp.")
        || lower.hasPrefix("model.mtp_layers.")
        || lower.contains(".mtp.")
        || lower.contains(".mtp_layers.")
}

private func loadSafetensorsHeaderNamesForBaseLoad(_ url: URL) throws -> [String] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        return []
    }
    var headerLength: UInt64 = 0
    for (index, byte) in lengthData.enumerated() {
        headerLength |= UInt64(byte) << UInt64(index * 8)
    }
    guard headerLength > 0, headerLength <= 64 * 1024 * 1024 else {
        return []
    }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == Int(headerLength),
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        return []
    }
    return header.keys.filter { $0 != "__metadata__" }
}

private func modelIndexContainsPreservedMTPWeight(at modelDirectory: URL) -> Bool? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard let data = try? Data(contentsOf: indexURL),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let weightMap = json["weight_map"] as? [String: Any]
    else {
        return nil
    }
    return weightMap.keys.contains(where: isPreservedMTPWeightKey)
}

private func loadJangConfigSanitizeMetadata(at modelDirectory: URL) -> [String: String] {
    guard let url = JangLoader.findConfigPath(at: modelDirectory),
        let data = try? Data(contentsOf: url),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        return [:]
    }

    var metadata: [String: String] = [:]
    func set(_ key: String, _ value: Any?) {
        if let value = value as? String, !value.isEmpty {
            metadata[key] = value
        }
    }

    set("norm_convention", json["norm_convention"])
    set("weight_format", json["weight_format"])
    if let runtime = json["runtime"] as? [String: Any] {
        set("runtime.norm_convention", runtime["norm_convention"])
        if metadata["norm_convention"] == nil {
            set("norm_convention", runtime["norm_convention"])
        }
    }
    return metadata
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and updates the model with the weights.
///
/// When a JANG model is detected (via `jangConfig`), per-layer bit widths are
/// inferred from tensor shapes automatically. Standard MLX models are unaffected.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    jangConfig: JangConfig? = nil,
    loadPreservedMTP: Bool = false
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // Resolve symlinks (mlxstudio uses symlinked model directories)
    let modelDirectory = modelDirectory.resolvingSymlinksInPath()

    // JANGTQ-native detection: `weight_format: "mxtq"` means the bundle
    // ships tq_packed/tq_norms tensors that should be consumed RAW by
    // TurboQuantSwitchGLU. The sidecar is preferred, but newer runtime-cache
    // code can deterministically regenerate signs/codebooks when it is absent.
    let jangtqSidecarURL = modelDirectory.appendingPathComponent("jangtq_runtime.safetensors")
    let hasJANGTQSidecar = FileManager.default.fileExists(atPath: jangtqSidecarURL.path)
    let declaresJANGTQNative: Bool = {
        guard let jangConfigURL = JangLoader.findConfigPath(at: modelDirectory),
            let configData = try? Data(contentsOf: jangConfigURL),
            let configJSON = try? JSONSerialization.jsonObject(with: configData)
                as? [String: Any]
        else { return false }
        let weightFormat = (configJSON["weight_format"] as? String)?.lowercased()
        let profile = (configJSON["profile"] as? String)?.lowercased() ?? ""
        return weightFormat == "mxtq" || profile.contains("jangtq")
    }()
    let isJANGTQNative = hasJANGTQSidecar || declaresJANGTQNative

    if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        // iter 25: collect candidate shard URLs first so we can detect
        // CORRUPT bundles that contain MULTIPLE concurrent shard sets
        // (e.g. an incomplete `model-NNNNN-of-00113.safetensors` partial
        // sitting alongside a complete `model-NNNNN-of-00115.safetensors`
        // — common after a re-download into a non-empty directory).
        // Mixed sets overwrite each other during the load loop, leaving
        // tensor shapes inconsistent and producing nonsense per-layer
        // quant inferences (the JANG_2L "uint32 vs bfloat16" fatal in
        // quantized_matmul). Detect and surface explicitly.
        // VMLX_DSV4_SKIP_SIDECAR=1 deliberately skips the prestacked
        // routed-expert overlay `jangtq_stacked.safetensors` even when
        // it's present in the bundle. Use this to validate the
        // sidecar-free load path (DSV4 bundles rebuilt with
        // `routed_expert_layout: prestacked` ship the stacked tensors
        // directly in the main model shards — the sidecar becomes
        // redundant). Cost when set: zero — the loader skips one file.
        // Benefit: proves the new bundle layout works before we drop
        // the sidecar entirely from HF releases. Phase B of the
        // sidecar-deprecation plan in
        // docs/OSAURUS-DSV4-INTEGRATION.md.
        let skipDSV4Sidecar =
            ProcessInfo.processInfo.environment["VMLX_DSV4_SKIP_SIDECAR"] == "1"
        var allShardURLs: [URL] = []
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors" else { continue }
            if url.lastPathComponent == "jangtq_runtime.safetensors" { continue }
            if skipDSV4Sidecar
                && url.lastPathComponent == "jangtq_stacked.safetensors"
            {
                FileHandle.standardError.write(Data(
                    "[loadWeights] VMLX_DSV4_SKIP_SIDECAR=1 — skipping jangtq_stacked.safetensors\n".utf8))
                continue
            }
            allShardURLs.append(url)
        }
        // Detect mixed `model-NNNNN-of-MMMMM.safetensors` sets: parse
        // the trailing `MMMMM` total and group by it. >1 distinct total
        // means we're looking at multiple shard sets in one directory.
        let shardTotalRegex = try! NSRegularExpression(
            pattern: #"model-\d+-of-(\d+)\.safetensors$"#)
        var totalsSeen: [Int: Int] = [:]   // total → count
        for url in allShardURLs {
            let n = url.lastPathComponent
            let nr = NSRange(n.startIndex..<n.endIndex, in: n)
            if let m = shardTotalRegex.firstMatch(in: n, range: nr),
               m.numberOfRanges >= 2,
               let totalRange = Range(m.range(at: 1), in: n),
               let total = Int(n[totalRange])
            {
                totalsSeen[total, default: 0] += 1
            }
        }
        if totalsSeen.count > 1 {
            // Pick the LARGEST total whose count equals that total
            // (= the COMPLETE shard set). All others are partials.
            // NB: `totalsSeen` is a Dictionary — its iteration order is
            // randomized per process, so `.first(where:)` would pick an
            // ARBITRARY complete set when two coexist (garbage-on-some-loads,
            // fixed-by-reload). `.max()` over the matches is deterministic AND
            // matches the "largest" intent stated above.
            let completeTotal = totalsSeen.filter { $0.key == $0.value }.keys.max()
                ?? totalsSeen.keys.max()!
            let summary = totalsSeen
                .map { "\($0.value)/\($0.key)" }
                .sorted()
                .joined(separator: ", ")
            let completeTag = String(format: "%05d", completeTotal)
            let warning = "[loadWeights] WARNING: bundle "
                + "\(modelDirectory.path) contains MULTIPLE concurrent "
                + "shard sets (\(summary)). Using only "
                + "`-of-\(completeTag).safetensors`. Delete the partial "
                + "set(s) to silence this warning.\n"
            FileHandle.standardError.write(Data(warning.utf8))
            allShardURLs = allShardURLs.filter {
                $0.lastPathComponent.hasSuffix(
                    "-of-\(String(format: "%05d", completeTotal)).safetensors")
            }
        }
        // Canonical loader entry. On patched osaurus mlx-swift pins,
        // `loadArraysAndMetadata(url:)` honors the safetensors mmap
        // environment set by `ModelFactory.withMmapSafetensorsEnv`,
        // returning MLX arrays backed by whole-shard file mappings.
        // Older pins ignore the env and fall back to the stock
        // pread/allocator path. `MmapSafetensorsLoader.swift` remains
        // a header/parser test utility, not the production tensor path.
        let prestackedRoutedKeys = (try? JangPressPrestacker
            .prestackedRoutedReplacementKeys(in: modelDirectory)) ?? []
        var skippedPrestackedSourceTensors = 0
        var skippedStreamingSourceTensors = 0
        var skippedPreservedMTPTensors = 0
        var skippedPreservedMTPShards = 0
        let shouldFilterPreservedMTP =
            !loadPreservedMTP && (modelIndexContainsPreservedMTPWeight(at: modelDirectory) ?? true)
        let streamingRoutedExperts =
            JANGTQStreamingExperts.isEnabled
            || JANGTQStreamingExperts.shouldAutoEnableNemotronUltra(
                modelDirectory: modelDirectory)
            || JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(
                modelDirectory: modelDirectory)
        if streamingRoutedExperts {
            JANGTQStreamingExperts.configureModelDirectory(modelDirectory)
        }
        for url in allShardURLs {
            if shouldFilterPreservedMTP,
                let headerNames = try? loadSafetensorsHeaderNamesForBaseLoad(url),
                !headerNames.isEmpty,
                headerNames.allSatisfy(isPreservedMTPWeightKey)
            {
                skippedPreservedMTPShards += 1
                skippedPreservedMTPTensors += headerNames.count
                continue
            }
            let (w, m) = try loadArraysAndMetadata(url: url)
            let isPrestackedShard = url.lastPathComponent == "jangpress-prestacked.safetensors"
            for (key, value) in w {
                if shouldFilterPreservedMTP, isPreservedMTPWeightKey(key) {
                    skippedPreservedMTPTensors += 1
                    continue
                }
                if streamingRoutedExperts,
                   JANGTQStreamingExperts.isStreamableRoutedTensorKey(key)
                {
                    skippedStreamingSourceTensors += 1
                    continue
                }
                if !isPrestackedShard,
                   let replacementKey = JangPressPrestacker.prestackedReplacementKey(
                    forPerExpertKey: key),
                   prestackedRoutedKeys.contains(replacementKey)
                {
                    skippedPrestackedSourceTensors += 1
                    continue
                }
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
        if skippedPrestackedSourceTensors > 0 {
            FileHandle.standardError.write(Data(
                "[loadWeights] using MLXPress prestacked routed overlay; skipped \(skippedPrestackedSourceTensors) original per-expert tensor(s)\n".utf8))
        }
        if skippedStreamingSourceTensors > 0 {
            FileHandle.standardError.write(Data(
                "[loadWeights] using MLXPress active-expert streaming; skipped \(skippedStreamingSourceTensors) per-expert tensor(s) during weight load\n".utf8))
        }
        if skippedPreservedMTPTensors > 0 {
            FileHandle.standardError.write(Data(
                "[loadWeights] preserved MTP tensors are isolated from base AR load; skipped \(skippedPreservedMTPTensors) tensor(s) across \(skippedPreservedMTPShards) MTP-only shard(s)\n".utf8))
        }
        if loadPreservedMTP {
            FileHandle.standardError.write(Data(
                "[loadWeights] native MTP requested; preserved MTP tensors are included in model update\n".utf8))
        }
    }

    let jangTensorManifest = try JangLoader.loadTensorQuantizationManifest(
        at: modelDirectory)
    if let jangTensorManifest {
        try JangLoader.validateTensorQuantizationManifest(
            jangTensorManifest, against: weights)
    }

    // Affine-1 JANG bundles opt in through a schema-2 manifest. Validate that
    // contract before model sanitization changes any weight paths, then leave
    // the packed tensors untouched for MLX's native affine-1 Metal kernels.
    if let contract = try JangLoader.loadAffine1RuntimeContract(at: modelDirectory) {
        for modulePath in contract.modulePaths {
            let key = "\(modulePath).weight"
            guard let packed = weights[key], packed.dtype == .uint32,
                packed.shape.last.map({ $0 > 0 }) == true
            else {
                throw JangLoaderError.loadFailed(
                    "affine-1 manifest weight must be a non-empty uint32 tensor: \(key)")
            }
        }
        FileHandle.standardError.write(Data(
            "[Load] JANG affine-1 validated \(contract.modulePaths.count) native 1-bit weight(s)\n".utf8))
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    //
    // Cap MetalAllocator's buffer_cache_ during sanitize() to keep the
    // per-shard `MLX.stacked()` intermediate buffers from ballooning the
    // pool to 100+ GB on high-shard JANGTQ bundles (MiniMax 117 shards,
    // Mistral 3.5 78 shards, Holo3 39+). The cache only helps with reuse
    // during inference, not during a one-shot load — capping it here
    // forces freed intermediates to actually release back to the OS
    // instead of accumulating in the pool.
    //
    // Restored to the prior limit after sanitize so steady-state
    // inference performance is unaffected.
    let priorCacheLimit = MLX.Memory.cacheLimit
    MLX.Memory.cacheLimit = 1 * 1024 * 1024 * 1024  // 1 GB during load
    defer {
        MLX.Memory.cacheLimit = priorCacheLimit
    }
    for (key, value) in loadJangConfigSanitizeMetadata(at: modelDirectory)
        where metadata[key] == nil
    {
        metadata[key] = value
    }
    weights = model.sanitize(weights: weights, metadata: metadata)

    // JANGTQ native: load the signs/codebook sidecar into the runtime cache
    // before model.update() so TurboQuantSwitchGLU has everything it needs
    // on first forward.
    if hasJANGTQSidecar {
        do {
            try JANGTQRuntimeCache.shared.loadSidecar(from: jangtqSidecarURL)
        } catch {
            print("[loadWeights] JANGTQ sidecar load failed: \(error)")
            throw error
        }
    } else if declaresJANGTQNative {
        FileHandle.standardError.write(Data(
            "[loadWeights] JANGTQ runtime sidecar missing; generating deterministic signs/codebooks on demand\n".utf8))
    }

    // Dequantize MoE gate/router weights from quantized uint32 -> float.
    // The model definitions keep router gates as plain Linear modules for
    // routing precision; bundled `.gate.{weight,scales,biases}` tensors are a
    // storage detail, not a signal to replace those gates with QuantizedLinear.
    // This must run for standard quantized bundles too, not only JANG bundles:
    // some Qwen3.6 MXFP-stamped artifacts carry affine router companions.
    // Safe for JANGTQ-native too: the dequant only touches `.*.gate.*` keys,
    // not the `tq_packed`/`tq_norms` expert projections.
    let declaredAffineQuantization = perLayerQuantization?.quantization
    let gateDefaultQuantization = declaredAffineQuantization ?? quantization
    let gateGroupSize = gateDefaultQuantization?.groupSize ?? jangConfig?.quantization.blockSize
    if let gateGroupSize {
        let gateBitWidths = Array(Set(
            (jangConfig?.quantization.bitWidthsUsed ?? [])
                + [gateDefaultQuantization?.bits].compactMap { $0 }
        )).sorted()
        JangLoader.dequantizeMoEGates(
            weights: &weights,
            groupSize: gateGroupSize,
            bitWidthsUsed: gateBitWidths,
            hiddenSizeHint: readHiddenSizeHint(at: modelDirectory))
    }

    // Determine quantization: JANG models infer per-layer bit widths from tensor shapes.
    // Standard MLX models use the quantization from config.json as before.
    // Safe for JANGTQ-native: infer only walks `.scales` keys, so it picks
    // up the affine 8-bit attention / embed / lm_head and ignores the
    // tq_packed expert projections.
    let effectivePerLayerQuantization: BaseConfiguration.PerLayerQuantization?
    if let jangConfig {
        // Prefer config.json's explicit `quantization.group_size` over
        // jangConfig.blockSize when the jang_config doesn't carry quant
        // metadata of its own (e.g., DSV4-Flash bundles ship
        // `weight_format: "bf16"` even on quantized variants — the
        // global group_size is in config.json instead).
        //
        // 2026-04-28: when jangConfig has explicit `bit_widths_used`
        // (signals real JANG conversion), `inferPerLayerQuantization`
        // ignores `overrideGroupSize` and uses jangConfig.blockSize as
        // the authoritative prior — see `JangLoader.swift` for the
        // root-cause writeup of the (8, 32) ≡ (4, 64) shape ambiguity
        // that crashed Cascade-2 JANG_4M / Nemotron-Omni MXFP4 with
        // mid-prefill rmsNorm. We do NOT merge config.json's per-layer
        // dict here because that dict can be HF-tooling-stale (claims
        // `(gs=32, bits=8)` for layers actually packed at `(gs=64,
        // bits=4)`) and would re-introduce the wrong values.
        let hiddenHint = readHiddenSizeHint(at: modelDirectory)
        let hiddenSizePerLayerInputHint = readHiddenSizePerLayerInputHint(at: modelDirectory)
        let linearAttnValueDimHint = readLinearAttnValueDimHint(at: modelDirectory)
        let expertIntermediateSizeHint = readExpertIntermediateSizeHint(at: modelDirectory)
        let validInDims = readValidInDims(at: modelDirectory)
        let attentionOutputDimHints = readAttentionOutputDimHintsForJANGQuantization(
            at: modelDirectory)
        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig,
            hiddenSizeHint: hiddenHint,
            hiddenSizePerLayerInputHint: hiddenSizePerLayerInputHint,
            linearAttnValueDimHint: linearAttnValueDimHint,
            expertIntermediateSizeHint: expertIntermediateSizeHint,
            validInDims: validInDims,
            attentionOutputDimHints: attentionOutputDimHints,
            declaredDefaultQuantization: declaredAffineQuantization ?? quantization,
            declaredPerLayerQuantization: perLayerQuantization,
            declaredManifestQuantization: jangTensorManifest?.entries ?? [:])

        if !inferred.perLayerQuantization.isEmpty {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] JANG shape walk produced \(inferred.perLayerQuantization.count) per-layer quant override(s) over default (bits=\(b), gs=\(g))\n".utf8))
        }
        func variants(_ key: String) -> [String] {
            var seen = Set<String>()
            var out: [String] = []
            func add(_ value: String) {
                if seen.insert(value).inserted {
                    out.append(value)
                }
            }

            add(key)
            if key.contains(".attn.") || key.hasSuffix(".attn") {
                add(key.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                if key.hasSuffix(".attn") {
                    add(String(key.dropLast(".attn".count)) + ".self_attn")
                }
            }
            if key.hasPrefix("language_model.model.") {
                add(String(key.dropFirst("language_model.".count)))
                add(String(key.dropFirst("language_model.model.".count)))
            } else if key.hasPrefix("language_model.") {
                add(String(key.dropFirst("language_model.".count)))
            } else if key.hasPrefix("model.") {
                add("language_model.\(key)")
            } else {
                add("model.\(key)")
                add("language_model.\(key)")
                add("language_model.model.\(key)")
            }
            return out
        }

        var merged = inferred.perLayerQuantization
        for (key, value) in inferred.perLayerQuantization {
            for variant in variants(key) where merged[variant] == nil {
                merged[variant] = value
            }
            let modelPrefixed = "model.\(key)"
            if merged[modelPrefixed] == nil { merged[modelPrefixed] = value }
            for variant in variants(modelPrefixed) where merged[variant] == nil {
                merged[variant] = value
            }
        }
        if let perLayerQuantization {
            for (key, value) in perLayerQuantization.perLayerQuantization {
                for variant in variants(key) {
                    if merged[variant] == nil { merged[variant] = value }
                    let modelPrefixed = "model.\(variant)"
                    if merged[modelPrefixed] == nil { merged[modelPrefixed] = value }
                }
                if key.hasPrefix("language_model.model.") {
                    let stripped = String(key.dropFirst("language_model.".count))
                    for variant in variants(stripped) where merged[variant] == nil {
                        merged[variant] = value
                    }
                } else if key.hasPrefix("language_model.") {
                    let stripped = String(key.dropFirst("language_model.".count))
                    for variant in variants(stripped) where merged[variant] == nil {
                        merged[variant] = value
                    }
                }
            }
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: declaredAffineQuantization ?? inferred.quantization,
            perLayerQuantization: merged
        )
        if ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1" {
            let topQ = declaredAffineQuantization ?? inferred.quantization
            FileHandle.standardError.write(Data(
                "[merge-diag] top-level quantization = \(topQ.map { "(b=\($0.bits), gs=\($0.groupSize), mode=\($0.mode.rawValue))" } ?? "NIL"); merged_count=\(merged.count); inferred_count=\(inferred.perLayerQuantization.count); explicit_count=\(perLayerQuantization?.perLayerQuantization.count ?? 0); hidden_hint=\(hiddenHint.map(String.init) ?? "nil"); valid_dims=\(validInDims.sorted())\n".utf8))
        }
    } else if let perLayerQuantization {
        // Remap perLayerQuantization keys to match sanitized weight paths.
        // Config.json uses VLM-prefixed keys like "language_model.model.layers.0..."
        // LLM sanitize strips to "model.layers.0..." but VLM keeps "language_model.model.layers.0..."
        // Keep BOTH original and stripped keys so it works for both paths.
        var remappedPerLayer = perLayerQuantization.perLayerQuantization
        for (key, value) in perLayerQuantization.perLayerQuantization {
            if key.hasPrefix("language_model.model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            } else if key.hasPrefix("language_model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            }
        }

        // Defense-in-depth: cross-check the config-supplied per-layer
        // overrides against actual safetensors shapes. If a bundle's
        // config.json was re-stamped (or a converter bug emitted wrong
        // bits / group_size), the runtime would otherwise silently
        // corrupt dequant. Shape walk is authoritative — it overrides
        // disagreeing entries and logs a one-line summary so users can
        // see when a patch was applied.
        if let shapeInferred = JangLoader.inferPerLayerQuantizationFromShapes(
            weights: weights,
            defaultBits: perLayerQuantization.quantization?.bits,
            defaultGroupSize: perLayerQuantization.quantization?.groupSize,
            defaultMode: perLayerQuantization.quantization?.mode ?? .affine)
        {
            // A declared per-module (bits, groupSize) is "shape-consistent" when it
            // unpacks the actual `.weight`/`.scales` to integer dims that agree:
            //   inputDim = packed*32 / bits   (must divide cleanly)
            //   inputDim / numGroups == groupSize
            // Some packed widths satisfy MULTIPLE bit-widths (e.g. 512 → bits∈{2,4,8}),
            // so the shape walk's first-match guess can disagree with a config that is
            // itself perfectly consistent. In that case the explicit config is the
            // author's stated intent and the shape walk's guess is ambiguous — keep
            // config. Only override when config is DEMONSTRABLY broken (its declared
            // bits don't unpack consistently with the on-disk shapes). Was: Laguna-M.1
            // shipped correct 4-bit dense-MLP, the walk re-stamped it 8-bit (512 is
            // ambiguous), and the 4-bit weights dequantized to garbage.
            func declaredBitsAreShapeConsistent(
                _ path: String, _ cq: BaseConfiguration.Quantization
            ) -> Bool {
                guard let w = weights["\(path).weight"], let s = weights["\(path).scales"],
                    let packed = w.shape.last, let numGroups = s.shape.last,
                    cq.bits > 0, numGroups > 0, (packed * 32) % cq.bits == 0
                else { return false }
                let inputDim = (packed * 32) / cq.bits
                return inputDim % numGroups == 0 && inputDim / numGroups == cq.groupSize
            }
            var corrections = 0
            for (path, expected) in shapeInferred.perLayerQuantization {
                if case .quantize(let q) = expected {
                    if let configured = remappedPerLayer[path],
                        case .quantize(let cq) = configured,
                        cq.bits == q.bits && cq.groupSize == q.groupSize && cq.mode == q.mode
                    { continue }
                    // Keep an explicit config that is itself shape-consistent (the walk
                    // disagreed only because the packed width is bit-width ambiguous).
                    if let configured = remappedPerLayer[path],
                        case .quantize(let cq) = configured,
                        declaredBitsAreShapeConsistent(path, cq)
                    { continue }
                    remappedPerLayer[path] = expected
                    corrections += 1
                }
            }
            if corrections > 0 {
                FileHandle.standardError.write(
                    Data("[Load] config per-layer quant disagreed with safetensors shapes — patched \(corrections) layer(s) from shape walk\n".utf8))
            }
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: perLayerQuantization.quantization,
            perLayerQuantization: remappedPerLayer
        )
    } else if let quantization {
        // Bundle has top-level quantization but no per-layer overrides.
        // Walk every `.scales` key and infer (bits, gs) from shapes.
        // This catches bundles whose `config.json` says e.g.
        // `bits: 8` uniformly while individual modules are actually
        // mixed (8-bit attention + 2-bit routed MoE). The algorithm
        // is idempotent: when the config matches reality the inferred
        // map adds no per-layer overrides.
        let inferred =
            JangLoader.inferPerLayerQuantizationFromShapes(
                weights: weights,
                defaultBits: quantization.bits,
                defaultGroupSize: quantization.groupSize,
                defaultMode: quantization.mode)
        if let inferred, !inferred.perLayerQuantization.isEmpty {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] non-JANG shape walk produced \(inferred.perLayerQuantization.count) per-layer quant override(s) over default (bits=\(b), gs=\(g))\n".utf8))
        }
        effectivePerLayerQuantization = inferred
            ?? BaseConfiguration.PerLayerQuantization(
                quantization: quantization, perLayerQuantization: [:])
    } else {
        // No quantization signal in config.json at all — but the
        // bundle may STILL be quantized (e.g., a stripped config).
        // If `.scales` keys exist, infer fully from shapes.
        let inferred =
            JangLoader.inferPerLayerQuantizationFromShapes(weights: weights)
        if let inferred {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] config has no quant block — shape walk inferred default (bits=\(b), gs=\(g)) plus \(inferred.perLayerQuantization.count) override(s)\n".utf8))
        }
        effectivePerLayerQuantization = inferred
    }

    // quantize if needed
    if quantization != nil || effectivePerLayerQuantization != nil {
        func quantizedWeightBaseCandidates(_ path: String) -> [String] {
            var seen = Set<String>()
            var out: [String] = []
            func add(_ value: String) {
                if seen.insert(value).inserted {
                    out.append(value)
                }
            }

            add(path)
            if path.hasPrefix("language_model.model.") {
                add(String(path.dropFirst("language_model.".count)))
                add(String(path.dropFirst("language_model.model.".count)))
            } else if path.hasPrefix("language_model.") {
                add(String(path.dropFirst("language_model.".count)))
            } else if path.hasPrefix("model.") {
                add("language_model.\(path)")
                add(String(path.dropFirst("model.".count)))
            } else {
                add("model.\(path)")
                add("language_model.\(path)")
                add("language_model.model.\(path)")
            }
            return out
        }

        // Inline quantize with error logging instead of try! crash
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            let baseCandidates = quantizedWeightBaseCandidates(path)
            let matchedBase = baseCandidates.first {
                weights["\($0).weight"] != nil && weights["\($0).scales"] != nil
            }
            guard let matchedBase,
                let loadedWeight = weights["\(matchedBase).weight"],
                let loadedScales = weights["\(matchedBase).scales"]
            else { return nil }
            let biasesKey = "\(matchedBase).biases"
            let biasKey = "\(matchedBase).bias"
            let tup: (groupSize: Int, bits: Int, mode: QuantizationMode)?
            if let effectivePerLayerQuantization {
                let explicit = baseCandidates.lazy.compactMap { candidate -> BaseConfiguration.Quantization? in
                    guard let option = effectivePerLayerQuantization.perLayerQuantization[candidate] else {
                        return nil
                    }
                    switch option {
                    case .skip:
                        return nil
                    case .quantize(let quantization):
                        return quantization
                    }
                }.first
                tup = (explicit ?? effectivePerLayerQuantization.quantization)?.asTuple
            } else {
                tup = quantization?.asTuple
            }
            guard let resolvedQuantization = tup else { return nil }
            let gs = resolvedQuantization.groupSize
            let b = resolvedQuantization.bits
            var mode = resolvedQuantization.mode
            if weights[biasesKey] != nil && (mode == .mxfp4 || mode == .mxfp8) {
                // MXFP kernels have no affine zero-point/bias companion. If a
                // converter stamped the bundle as MXFP but emitted `.biases`,
                // the tensor payload is affine and must be loaded that way.
                mode = .affine
            }

            let quantBiases =
                (mode == .mxfp4 || mode == .mxfp8) ? nil : weights[biasesKey]

            // Pre-quantized safetensors already provide `.weight` +
            // `.scales` (+ optional quant `.biases`). Build the quantized
            // module from those arrays directly instead of quantizing the
            // randomly initialized placeholder module and immediately
            // overwriting it during `model.update(parameters:)`. This is
            // especially important for routed-MoE `SwitchLinear`, where the
            // placeholder can be tens of GB on Ling/DSV4-class bundles.
            if let linear = m as? Linear {
                return (path, QuantizedLinear(
                    weight: loadedWeight,
                    bias: weights[biasKey] ?? linear.bias,
                    scales: loadedScales,
                    biases: quantBiases,
                    groupSize: gs, bits: b, mode: mode))
            }

            if let switchLinear = m as? SwitchLinear {
                return (path, QuantizedSwitchLinear(
                    inputDims: switchLinear.inputDims,
                    outputDims: switchLinear.outputDims,
                    numExperts: switchLinear.numExperts,
                    weight: loadedWeight,
                    bias: weights[biasKey] ?? switchLinear.bias,
                    scales: loadedScales,
                    biases: quantBiases,
                    groupSize: gs, bits: b, mode: mode))
            }

            if m is Embedding {
                return (path, QuantizedEmbedding(
                    weight: loadedWeight,
                    scales: loadedScales,
                    biases: quantBiases,
                    groupSize: gs,
                    bits: b,
                    mode: mode))
            }

            if let q = quantizeSingle(layer: m, groupSize: gs, bits: b, mode: mode) {
                return (path, q)
            }
            return nil
        }
        do {
            try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
        } catch {
            print("[loadWeights] quantize model.update failed: \(error)")
            for (path, mod) in updates.prefix(5) {
                print("  update path: \(path) → \(type(of: mod))")
            }
            throw error
        }
    }

    // apply the loaded weights
    // Use .noUnusedKeys instead of .all — MXFP4/MXFP8 quantized layers don't have .biases
    // in the weight files, but QuantizedLinear's optional .biases property gets initialized
    // by the quantize step. Strict .all verification would fail on the missing keys.
    do {
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.noUnusedKeys])
    }

    // `weights` is only a load/update staging dictionary. Drop it before
    // any post-load dtype materialization so quantized bundles do not keep
    // a second complete copy of the original safetensor arrays alive while
    // the model parameters are being converted in place.
    weights.removeAll(keepingCapacity: false)
    MLX.Memory.clearCache()

    // Tied-head load-time quantization: quantize a plain fp16/bf16
    // `embed_tokens` at load. Tied-embedding families (Gemma4 QAT) ship
    // the decoder quantized but the 262k-vocab embedding unquantized;
    // `asLinear` then streams the full fp16 table per decoded token
    // (E2B: ~1.07 GB/token), which dominates decode bandwidth. llama.cpp
    // ships the equivalent output head quantized (Q6_K-class) in the GGUF
    // baselines this engine is compared against.
    //
    // Policy precedence: VMLX_QUANT_TIED_HEAD_BITS env (bench override) >
    // host-set TiedHeadQuantizationPolicy (server settings) > off.
    // Applied ONLY when the bundle itself is quantized — an fp16 bundle's
    // head is part of the bundle's declared precision contract.
    let envHeadBits = ProcessInfo.processInfo.environment["VMLX_QUANT_TIED_HEAD_BITS"]
        .flatMap(Int.init)
    let policyHeadQuant = TiedHeadQuantizationPolicy.current
    let bundleIsQuantized = quantization != nil || perLayerQuantization != nil
    if let headBits = envHeadBits ?? (bundleIsQuantized ? policyHeadQuant?.bits : nil),
        headBits > 0
    {
        let headGroupSize =
            ProcessInfo.processInfo.environment["VMLX_QUANT_TIED_HEAD_GS"].flatMap(Int.init)
            ?? policyHeadQuant?.groupSize ?? 64
        var headUpdates: [(String, Module)] = []
        for (path, mod) in model.namedModules() {
            guard path.hasSuffix("embed_tokens"), type(of: mod) == Embedding.self,
                let emb = mod as? Embedding
            else { continue }
            headUpdates.append(
                (path, QuantizedEmbedding(emb, groupSize: headGroupSize, bits: headBits)))
        }
        if !headUpdates.isEmpty {
            try model.update(modules: ModuleChildren.unflattened(headUpdates), verify: .none)
            MLX.eval(model)
            MLX.Memory.clearCache()
            FileHandle.standardError.write(Data(
                "[Load] quantized tied embedding head bits=\(headBits) gs=\(headGroupSize) paths=\(headUpdates.map(\.0))\n".utf8))
        }
    }

    // Convert float16/float32 parameters to bfloat16 to prevent AsType cascades.
    // float16 causes AsType when mixed with internal float32 ops (softmax, RMSNorm).
    // bfloat16 shares float32's exponent range, so promotion is cheaper/eliminated.
    //
    // JANGTQ caveat: TurboQuant Metal kernels infer their signature from
    // `tq_norms`, and casting those norms to bf16 breaks routed projections
    // (verified on MiniMax M2.7 JANGTQ_2L). Keep the JANGTQ tensors raw.
    //
    // For non-mmap JANGTQ loads, still convert non-TQ Mamba/attention/router
    // weights so resident decode avoids preventable fp16 AsType cascades. For
    // mmap/JangPress loads, default to preserving file-backed tensor residency
    // unless the bundle has live proof that affine quant metadata must be
    // materialized to bf16 while raw TurboQuant tensors stay untouched.
    let mmapSafetensorsActive = envFlag("MLX_SAFETENSORS_MMAP")
        || envFlag("VMLINUX_MMAP_SAFETENSORS")
    let allowJANGTQMmapBFloat16 = envFlag("VMLINUX_JANGTQ_BF16_MMAP")
        || envFlag("MLX_JANGTQ_BF16_MMAP")
    let autoJANGTQMmapBFloat16 = requiresJANGTQMmapBFloat16(modelDirectory)
    if !isJANGTQNative || !mmapSafetensorsActive || allowJANGTQMmapBFloat16
        || autoJANGTQMmapBFloat16
    {
        convertToBFloat16(
            model: model,
            shouldSkip: isJANGTQNative ? isJANGTQParameterKey : { _ in false })
    }

    eval(model)
    MLX.Memory.clearCache()
}

/// Convert float16/float32 model parameters to bfloat16 for MoE performance.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing ~50% speed regression for MoE models where gate routing runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
/// Quantization scales/biases are ALSO converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
///
/// Keep the conversion chunked. Ling MXFP4 carries tens of GB of affine
/// scale/bias metadata; converting every array into one dictionary keeps
/// both fp16 and bf16 copies alive until the final eval and can push peak
/// memory past 100 GB. Chunking bounds the transient extra allocation while
/// preserving the dtype contract.
private func isJANGTQParameterKey(_ key: String) -> Bool {
    key.hasSuffix(".tq_packed") || key.hasSuffix(".tq_norms")
}

private func requiresJANGTQMmapBFloat16(_ modelDirectory: URL) -> Bool {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard
        let data = try? Data(contentsOf: configURL),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        return false
    }
    let config = (object["text_config"] as? [String: Any]) ?? object
    let modelType = ((config["model_type"] as? String) ?? (object["model_type"] as? String) ?? "")
        .lowercased()
    if modelType == "nemotron_h" {
        return true
    }
    guard modelType == "qwen3_5_moe" || modelType == "qwen3_5_moe_text" else {
        return false
    }
    let layerTypes = config["layer_types"] as? [String] ?? []
    return layerTypes.contains("linear_attention")
}

private func envFlag(_ key: String) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[key]?.lowercased() else {
        return false
    }
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

private func convertToBFloat16(
    model: Module,
    shouldSkip: (String) -> Bool = { _ in false }
) {
    let convertibleParams: [(key: String, convertedBytes: Int)] = {
        let flat = model.parameters().flattened()
        return flat.compactMap { key, array in
            guard !shouldSkip(key) else {
                return nil
            }
            guard array.ctx.ctx != nil else {
                return nil
            }
            guard array.dtype == .float16 || array.dtype == .float32 else {
                return nil
            }
            return (key: key, convertedBytes: estimatedByteCount(array, as: .bfloat16))
        }
    }()

    guard !convertibleParams.isEmpty else { return }

    let chunkLimit = bfloat16ConversionChunkLimit()
    var index = 0
    while index < convertibleParams.count {
        var converted = [String: MLXArray]()
        var convertedBytes = 0

        do {
            let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            while index < convertibleParams.count {
                let entry = convertibleParams[index]
                if !converted.isEmpty,
                    convertedBytes + entry.convertedBytes > chunkLimit
                {
                    break
                }

                guard let array = current[entry.key],
                    array.ctx.ctx != nil,
                    array.dtype == DType.float16 || array.dtype == DType.float32
                else {
                    index += 1
                    continue
                }

                converted[entry.key] = array.asType(DType.bfloat16)
                convertedBytes += entry.convertedBytes
                index += 1
            }
        }

        guard !converted.isEmpty else { continue }

        let values = Array(converted.values)
        MLX.eval(values)

        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            print("[convertToBFloat16] model.update failed: \(error)")
        }
        MLX.Memory.clearCache()
    }
}

private func bfloat16ConversionChunkLimit() -> Int {
    let env = ProcessInfo.processInfo.environment
    if let raw = env["VMLX_BF16_CONVERT_CHUNK_MB"],
        let mb = Int(raw), mb > 0
    {
        return mb * 1024 * 1024
    }
    return 256 * 1024 * 1024
}

private func estimatedByteCount(_ array: MLXArray, as dtype: DType) -> Int {
    let elements = array.shape.reduce(1) { partial, dim in
        partial * max(dim, 1)
    }
    return elements * dtypeByteWidth(dtype)
}

private func dtypeByteWidth(_ dtype: DType) -> Int {
    if dtype == .bool || dtype == .int8 || dtype == .uint8 {
        return 1
    }
    if dtype == .float16 || dtype == .bfloat16
        || dtype == .int16 || dtype == .uint16
    {
        return 2
    }
    if dtype == .int64 || dtype == .uint64 {
        return 8
    }
    return 4
}

private func readHiddenSizeHint(at modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    if let textConfig = top["text_config"] as? [String: Any],
       let hiddenSize = textConfig["hidden_size"] as? Int, hiddenSize > 0
    {
        return hiddenSize
    }
    if let hiddenSize = top["hidden_size"] as? Int, hiddenSize > 0 {
        return hiddenSize
    }
    return nil
}

private func readHiddenSizePerLayerInputHint(at modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    let config = (top["text_config"] as? [String: Any]) ?? top
    if let value = config["hidden_size_per_layer_input"] as? Int, value > 0 {
        return value
    }
    return nil
}

private func readLinearAttnValueDimHint(at modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    let config = (top["text_config"] as? [String: Any]) ?? top
    guard let valueHeads = config["linear_num_value_heads"] as? Int,
          let valueHeadDim = config["linear_value_head_dim"] as? Int,
          valueHeads > 0, valueHeadDim > 0
    else { return nil }
    return valueHeads * valueHeadDim
}

private func readExpertIntermediateSizeHint(at modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    let config = (top["text_config"] as? [String: Any]) ?? top
    for key in ["moe_intermediate_size", "expert_intermediate_size", "intermediate_size"] {
        if let value = config[key] as? Int, value > 0 {
            return value
        }
    }
    return nil
}

/// Build architecture-valid input dimensions for JANG bit/group-size
/// disambiguation. Shape math alone can make (8, 32), (4, 64), and
/// (2, 128) look equivalent; these dimensions constrain Qwen hybrid SSM,
/// MLA, and MoE projections to real model widths.
private func readValidInDims(at modelDirectory: URL) -> Set<Int> {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [] }
    let config = (top["text_config"] as? [String: Any]) ?? top
    var dims = Set<Int>()
    func add(_ value: Any?) {
        if let int = value as? Int, int > 0 {
            dims.insert(int)
        }
    }

    add(config["hidden_size"])
    add(config["intermediate_size"])
    add(config["moe_intermediate_size"])
    add(config["expert_intermediate_size"])
    add(config["shared_expert_intermediate_size"])
    add(config["hidden_size_per_layer_input"])
    add(config["q_lora_rank"])
    add(config["kv_lora_rank"])
    add(config["v_head_dim"])
    add(config["qk_nope_head_dim"])
    add(config["qk_rope_head_dim"])

    if let numLayers = config["num_hidden_layers"] as? Int,
       let perLayerInput = config["hidden_size_per_layer_input"] as? Int,
       numLayers > 0, perLayerInput > 0
    {
        dims.insert(numLayers * perLayerInput)
    }

    let headDims = [
        config["head_dim"],
        config["swa_head_dim"],
        config["global_head_dim"],
    ].compactMap { $0 as? Int }
    let headCounts = [
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["swa_num_attention_heads"],
        config["swa_num_key_value_heads"],
        config["num_global_key_value_heads"],
    ].compactMap { $0 as? Int }
    for headDim in headDims where headDim > 0 {
        for count in headCounts where count > 0 {
            dims.insert(headDim * count)
        }
    }

    let valueHeadDims = [
        config["v_head_dim"],
        config["swa_v_head_dim"],
        config["global_v_head_dim"],
    ].compactMap { $0 as? Int }
    let valueHeadCounts = [
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["swa_num_attention_heads"],
        config["swa_num_key_value_heads"],
        config["num_global_key_value_heads"],
    ].compactMap { $0 as? Int }
    for valueHeadDim in valueHeadDims where valueHeadDim > 0 {
        for count in valueHeadCounts where count > 0 {
            dims.insert(valueHeadDim * count)
        }
    }

    if let nope = config["qk_nope_head_dim"] as? Int,
       let rope = config["qk_rope_head_dim"] as? Int,
       let heads = config["num_attention_heads"] as? Int,
       nope > 0, rope > 0, heads > 0
    {
        dims.insert((nope + rope) * heads)
    }

    if let groups = config["o_groups"] as? Int,
       let rank = config["o_lora_rank"] as? Int
    {
        let groupedOut = groups * rank
        if groupedOut > 0 { dims.insert(groupedOut) }
    }

    if let keyHeads = config["linear_num_key_heads"] as? Int,
       let keyHeadDim = config["linear_key_head_dim"] as? Int
    {
        let keyDim = keyHeads * keyHeadDim
        if keyDim > 0 { dims.insert(keyDim) }

        if let valueHeads = config["linear_num_value_heads"] as? Int,
           let valueHeadDim = config["linear_value_head_dim"] as? Int
        {
            let valueDim = valueHeads * valueHeadDim
            if valueDim > 0 { dims.insert(valueDim) }

            let convDim = keyDim * 2 + valueDim
            if convDim > 0 { dims.insert(convDim) }
        }
    }

    return dims
}

func readAttentionOutputDimHintsForJANGQuantization(at modelDirectory: URL) -> Set<Int> {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
          let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [] }
    let config = (top["text_config"] as? [String: Any]) ?? top
    var dims = Set<Int>()

    func add(headsKey: String, valueDimKey: String, fallbackDimKey: String) {
        guard let heads = config[headsKey] as? Int, heads > 0 else { return }
        if let valueDim = config[valueDimKey] as? Int, valueDim > 0 {
            dims.insert(heads * valueDim)
        } else if let headDim = config[fallbackDimKey] as? Int, headDim > 0 {
            dims.insert(heads * headDim)
        }
    }

    add(
        headsKey: "num_attention_heads",
        valueDimKey: "v_head_dim",
        fallbackDimKey: "head_dim")
    add(
        headsKey: "swa_num_attention_heads",
        valueDimKey: "swa_v_head_dim",
        fallbackDimKey: "swa_head_dim")
    add(
        headsKey: "num_global_attention_heads",
        valueDimKey: "global_v_head_dim",
        fallbackDimKey: "global_head_dim")

    return dims
}
