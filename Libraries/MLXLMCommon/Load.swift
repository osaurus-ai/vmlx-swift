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
            let completeTotal = totalsSeen.first(where: { $0.key == $0.value })?.key
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
        let streamingRoutedExperts = JANGTQStreamingExperts.isEnabled
        for url in allShardURLs {
            if !loadPreservedMTP,
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
                if !loadPreservedMTP, isPreservedMTPWeightKey(key) {
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

    // JANG: dequantize MoE gate weights from quantized uint32 → float.
    // Gates are stored at 8-bit (CRITICAL tier) but may have different group_size
    // than the body. Dequantizing resolves ambiguous bit/group_size inference.
    // Safe for JANGTQ-native too: the dequant only touches `.*.gate.*` keys,
    // not the `tq_packed`/`tq_norms` expert projections.
    let declaredAffineQuantization = perLayerQuantization?.quantization
    if let jangConfig {
        let gateBitWidths = Array(Set(
            jangConfig.quantization.bitWidthsUsed
                + [declaredAffineQuantization?.bits].compactMap { $0 }
        )).sorted()
        JangLoader.dequantizeMoEGates(
            weights: &weights,
            groupSize: declaredAffineQuantization?.groupSize
                ?? jangConfig.quantization.blockSize,
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
        let linearAttnValueDimHint = readLinearAttnValueDimHint(at: modelDirectory)
        let validInDims = readValidInDims(at: modelDirectory)
        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig,
            hiddenSizeHint: hiddenHint,
            linearAttnValueDimHint: linearAttnValueDimHint,
            validInDims: validInDims,
            declaredDefaultQuantization: declaredAffineQuantization ?? quantization)

        if !inferred.perLayerQuantization.isEmpty {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] JANG shape walk produced \(inferred.perLayerQuantization.count) per-layer quant override(s) over default (bits=\(b), gs=\(g))\n".utf8))
        }
        func variants(_ key: String) -> [String] {
            var out = [key]
            if key.contains(".attn.") || key.hasSuffix(".attn") {
                out.append(key.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                if key.hasSuffix(".attn") {
                    out.append(String(key.dropLast(".attn".count)) + ".self_attn")
                }
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
            var corrections = 0
            for (path, expected) in shapeInferred.perLayerQuantization {
                if case .quantize(let q) = expected {
                    if let configured = remappedPerLayer[path],
                        case .quantize(let cq) = configured,
                        cq.bits == q.bits && cq.groupSize == q.groupSize && cq.mode == q.mode
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
        // Inline quantize with error logging instead of try! crash
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            let weightKey = "\(path).weight"
            let scalesKey = "\(path).scales"
            let biasesKey = "\(path).biases"
            let biasKey = "\(path).bias"
            guard let loadedWeight = weights[weightKey],
                let loadedScales = weights[scalesKey]
            else { return nil }
            let tup: (groupSize: Int, bits: Int, mode: QuantizationMode)?
            if let effectivePerLayerQuantization {
                tup = effectivePerLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                tup = quantization?.asTuple
            }
            guard let (gs, b, mode) = tup else { return nil }

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

    // Convert all float16/float32 parameters to bfloat16 to prevent AsType cascades.
    // float16 causes AsType when mixed with internal float32 ops (softmax, RMSNorm).
    // bfloat16 shares float32's exponent range, so promotion is cheaper/eliminated.
    //
    // JANGTQ bypass: Python baseline runs with fp16 TurboQuant norms, and the
    // JANGTQ Metal kernels infer their signature from the norm dtype. Casting
    // those norms to bf16 breaks the gate/up/down projections (verified on
    // MiniMax M2.7 JANGTQ_2L). JANGTQ dispatches are already fp32 internally,
    // so there's no fp16↔fp32 ping-pong to collapse. Skip the cast entirely.
    if !isJANGTQNative {
        convertToBFloat16(model: model)
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
private func convertToBFloat16(model: Module) {
    let convertibleParams: [(key: String, convertedBytes: Int)] = {
        let flat = model.parameters().flattened()
        return flat.compactMap { key, array in
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
    add(config["q_lora_rank"])
    add(config["kv_lora_rank"])
    add(config["v_head_dim"])
    add(config["qk_nope_head_dim"])
    add(config["qk_rope_head_dim"])

    let headDims = [config["head_dim"], config["global_head_dim"]].compactMap { $0 as? Int }
    let headCounts = [
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["num_global_key_value_heads"],
    ].compactMap { $0 as? Int }
    for headDim in headDims where headDim > 0 {
        for count in headCounts where count > 0 {
            dims.insert(headDim * count)
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
