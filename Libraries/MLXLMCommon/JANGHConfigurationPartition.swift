import Foundation

/// Metadata admission only. The ordinary view must never be used without constructing
/// every custom module from `contract`: its false overrides exclude those modules from
/// generic quantization, rather than converting their weights to affine quantization.
struct JANGHConfigurationPartition: Sendable {
    enum Failure: Error, Equatable { case invalid(String) }

    let contract: JANGHFormatContract
    let modelType: String
    let customModules: Set<String>
    /// Both quantization aliases contain the same ordinary plan plus explicit skips.
    /// Unknown ordinary modes remain present for BaseConfiguration's strict decoder.
    let ordinaryConfiguration: Data
    /// Canonical root metadata consumed by the existing JANGH format validator.
    let customConfiguration: Data

    init(configuration: Data, sidecar: Data? = nil) throws {
        var root = try Self.object(configuration)
        guard let modelType = root["model_type"] as? String, !modelType.isEmpty else {
            throw Failure.invalid("missing model_type; no architecture alias is inferred")
        }
        // This first adapter has one authoritative root owner. Reject a second owner
        // rather than silently selecting one or rewriting architecture-local metadata.
        if let text = root["text_config"] as? [String: Any],
            ["quantization", "quantization_config", "jangtq"].contains(where: {
                text[$0] != nil && !(text[$0] is NSNull)
            })
        {
            throw Failure.invalid("nested JANGH metadata ownership is not supported")
        }
        let primary = try Self.optionalObject(root, "quantization")
        let alias = try Self.optionalObject(root, "quantization_config")
        guard primary != nil || alias != nil else {
            throw Failure.invalid("missing quantization plan")
        }
        let normalizedPrimary = try primary.map(Self.flatten)
        let normalizedAlias = try alias.map(Self.flatten)
        if let normalizedPrimary, let normalizedAlias,
            try Self.encode(normalizedPrimary) != Self.encode(normalizedAlias)
        {
            // JANGHT bundles (a `jangt` header) are produced and run by a loader that reads ONLY `quantization`
            // (vMLX Python `install_jangt(config["quantization"])`); Naive-N0.5 JANGHT2.5 ships a stale
            // `quantization_config` copied from its JANGH2 parent. For those bundles `quantization` is the plan
            // the producer validated. Every other custom bundle keeps refusing a conflict.
            guard root["jangt"] is [String: Any] else {
                throw Failure.invalid("conflicting quantization aliases")
            }
            LoadDiagnostics.write(Data(
                "[JANGH] JANGHT bundle: quantization_config differs from quantization; using quantization\n".utf8))
        }
        let plan = normalizedPrimary ?? normalizedAlias!
        var header = try Self.optionalObject(root, "jangtq")
        if let sidecar {
            let other = try Self.object(sidecar)
            // MiniCPM-V 4.7 JANGH2 is produced by the same converter: authoritative header in config.json, sidecar
            // with chat/runtime metadata and a scalar summary.
            if modelType == "flint", let authoritative = header {
                // Flint keeps its executable plan/header in config.json. The
                // converter sidecar contains descriptions and AWQ/GPTQ statistics.
                guard other["format"] as? String == "jangtq2",
                    let version = other["format_version"],
                    try Self.encode(["v": version]) == Self.encode(["v": 2]),
                    other["quantization_config"] == nil
                else { throw Failure.invalid("unsupported Flint JANGH sidecar contract") }
                if let declared = try Self.optionalObject(other, "jangtq"),
                    try Self.encode(authoritative) != Self.encode(declared)
                {
                    throw Failure.invalid("conflicting Flint config and sidecar JANGH headers")
                }
                if let summary = try Self.optionalObject(other, "quantization") {
                    let descriptions: Set<String> = [
                        "routed_experts", "non_experts", "router", "embed_tokens", "lm_head", "norms_sinks",
                    ]
                    let statistics: Set<String> = [
                        "gptq_vs_rtn", "awq_moe_in", "awq_attn", "expert_units_jangh",
                    ]
                    guard Set(summary.keys).isSubset(of: descriptions.union(statistics)),
                        summary.allSatisfy({ key, value in
                            descriptions.contains(key) ? value is String : Self.isNumericCalibrationMetadata(value)
                        })
                    else { throw Failure.invalid("Flint sidecar quantization must be calibration metadata") }
                }
            } else if modelType == "lfm2_moe", let authoritative = header,
                other["format"] as? String == "jang", other["format_version"] as? String == "2.0"
            {
                // The LFM converter stores the executable plan/header in config.json.
                // jang_config.json is descriptive chat/calibration metadata, not a second plan.
                guard other["quantization_config"] == nil else {
                    throw Failure.invalid("LFM sidecar quantization ownership is not supported")
                }
                if let declared = try Self.optionalObject(other, "jangtq"),
                    try Self.encode(authoritative) != Self.encode(declared)
                {
                    throw Failure.invalid("conflicting LFM config and sidecar JANGH headers")
                }
                if let summary = try Self.optionalObject(other, "quantization") {
                    guard !summary.keys.contains(where: { $0.contains(".") }),
                        summary["per_tensor"] == nil, summary["quantization_config"] == nil,
                        summary["method"] as? String == "jang", summary["bits"] is Int,
                        summary["group_size"] is Int, summary["bit_widths_used"] is [Int],
                        let modes = summary["modes"] as? [String], modes.count == 2,
                        Set(modes) == ["affine", "jangtq2"]
                    else { throw Failure.invalid("LFM sidecar quantization must be summary metadata") }
                    guard summary["bits"] as? Int == plan["bits"] as? Int,
                        summary["group_size"] as? Int == plan["group_size"] as? Int
                    else { throw Failure.invalid("LFM sidecar summary disagrees with the config plan") }
                    if let routed = summary["routed_experts"] as? [String: Any] {
                        let customEntries = plan.filter { ($0.value as? [String: Any])?["mode"] as? String == "jangtq2" }
                        func widths(_ roles: [String]) -> [Int] {
                            Set(customEntries.compactMap { name, value in
                                roles.contains(String(name.split(separator: ".").last ?? ""))
                                    ? (value as? [String: Any])?["bits"] as? Int : nil
                            }).sorted()
                        }
                        guard !routed.keys.contains(where: { $0.contains(".") }), routed["per_tensor"] == nil,
                            routed["mode"] as? String == "jangtq2",
                            routed["gate_up_bits"] as? [Int] == widths(["gate_proj", "up_proj"]),
                            routed["down_bits"] as? [Int] == widths(["down_proj"])
                        else { throw Failure.invalid("LFM routed summary disagrees with the config plan") }
                    } else if summary["routed_experts"] != nil {
                        throw Failure.invalid("invalid LFM routed summary")
                    }
                }
            } else if modelType == "qwen3_5" || modelType == "minicpmv4_7" || modelType == "qwen3_5_moe", let authoritative = header {
                // This converter's sidecar carries chat/runtime metadata and a
                // scalar summary, not another per-module quantization owner.
                let decision = other["decision"] as? [String: Any]
                let runtime = other["runtime"] as? [String: Any]
                let decisionMetadataOnly = modelType == "qwen3_5"
                    && (root["architectures"] as? [String]) == ["Qwen3_5Model"]
                    && decision?["format_id"] as? String == "d3-code-readout-v1"
                    && decision?["generates_text"] as? Bool == false
                    && decision?["pooling"] as? String == "last"
                    && runtime?["lm_head"] as? Bool == false
                    && runtime?["bundle_has_mtp"] as? Bool == false
                    && runtime?["mtp_layers"] as? Int == 0
                    && other["format"] == nil && other["format_version"] == nil
                    && other["weight_format"] == nil
                let version = other["format_version"]
                let versionedSidecar = other["format"] as? String == "jangtq2"
                    && (version as? Int == 2 || version as? String == "2.0")
                // D3 deliberately puts the executable v2 header/plan only in config.json.
                // Its headless decision stamp is metadata, not a second format owner.
                guard versionedSidecar || decisionMetadataOnly else {
                    throw Failure.invalid("unsupported dense Qwen JANGH sidecar version")
                }
                if other["jangtq"] != nil {
                    guard let declared = try Self.optionalObject(other, "jangtq"),
                        try Self.encode(authoritative) == Self.encode(declared)
                    else { throw Failure.invalid("conflicting config and sidecar JANGH headers") }
                }
                guard other["quantization_config"] == nil else {
                    throw Failure.invalid("sidecar quantization ownership is not supported")
                }
                if let summary = try Self.optionalObject(other, "quantization") {
                    if modelType == "qwen3_5_moe" {
                        // Ornith's sidecar is calibration provenance, not an
                        // executable plan. The config header/plan above remains
                        // authoritative; reject module names and plan-like keys.
                        let fields: Set<String> = [
                            "method", "profile", "actual_bits", "block_size", "calibrated",
                            "imatrix", "gptq", "awq_layers", "calibration_method",
                            "quantization_method", "bit_widths_used", "quantization_scheme",
                            "quantization_backend", "experts_gptq_vs_rtn", "awq", "layout",
                            "experts_jangh_units",
                        ]
                        guard Set(summary.keys).isSubset(of: fields),
                            summary["method"] as? String == "jang-gptq",
                            summary["block_size"] is Int,
                            summary["bit_widths_used"] is [Int]
                        else { throw Failure.invalid("Qwen MoE sidecar must be calibration metadata") }
                    } else {
                        // `method` / `layout` are descriptive strings (MiniCPM-V 4.7), never a per-module plan.
                        guard
                            Set(summary.keys).isSubset(of: [
                                "bits", "group_size", "bit_widths_used", "method", "layout",
                            ] + (decisionMetadataOnly ? ["unquantized_linears"] : [])),
                            summary["bits"] is Int, summary["group_size"] is Int,
                            summary["bit_widths_used"] is [Int],
                            summary["method"] == nil || summary["method"] is String,
                            summary["layout"] == nil || summary["layout"] is String
                        else {
                            throw Failure.invalid(
                                "dense Qwen sidecar quantization must be summary metadata")
                        }
                    }
                    if decisionMetadataOnly, let paths = summary["unquantized_linears"] {
                        guard let names = paths as? [String], Set(names).count == names.count,
                            names.allSatisfy({ plan[$0] as? Bool == false }) else {
                            throw Failure.invalid("D3 unquantized summary disagrees with the authoritative plan")
                        }
                    }

                }
            } else {
                // Do not identify the format using the legacy label alone.
                guard other["format"] as? String == "jangtq2",
                    let version = other["format_version"],
                    try Self.encode(["v": version]) == Self.encode(["v": 2]),
                    let otherHeader = try Self.optionalObject(other, "jangtq")
                else { throw Failure.invalid("unsupported or missing JANGH sidecar contract") }
                if let header, try Self.encode(header) != Self.encode(otherHeader) {
                    throw Failure.invalid("conflicting config and sidecar JANGH headers")
                }
                // A quantization plan in a sidecar would introduce another unvalidated owner.
                guard other["quantization"] == nil, other["quantization_config"] == nil else {
                    throw Failure.invalid("sidecar quantization ownership is not supported")
                }
                header = otherHeader
            }
        }
        // A pure-JANGT bundle (no `jangtq2` projection, a `jangt` header) carries no codebooks and may omit `jangtq`;
        // the format contract still refuses a `jangtq2` projection without one.
        let declaresCodebookProjection = plan.values.contains {
            ($0 as? [String: Any])?["mode"] as? String == "jangtq2"
        }
        guard header != nil || (!declaresCodebookProjection && root["jangt"] is [String: Any]) else {
            throw Failure.invalid("missing authoritative JANGH header")
        }
        var canonical = root
        canonical["jangtq"] = header
        canonical["quantization"] = plan
        canonical["quantization_config"] = plan
        let customConfiguration = try Self.encode(canonical)
        let contract = try JANGHFormatContract(configuration: customConfiguration)
        let customModules = Set(contract.projections.keys).union(contract.trellis.keys)
        // This adapter admits only the checkpoint namespace audited for routed
        // JANGH banks. Other complete triples may be valid future formats, but
        // are unsupported here (shared-expert and attention aliases in the
        // generic decoder must not broaden custom-bank admission implicitly).
        let denseK2 = modelType == "k2_horizon" && root["mlp_layout"] as? String == "dense_jangh_down"
        // Dense Qwen3.5-family JANGH (Qwen3.8-27B JANGH2, 2026-10-06): every dense decoder MLP projection is a
        // one-expert codebook bank at `language_model.model.layers.L.mlp.{gate,up,down}_proj` (complete
        // triples are enforced by JANGHFormatContract). Nothing else may be custom in such a bundle.
        let denseQwen35 = modelType == "qwen3_5"
        for name in customModules {
            if denseK2 { continue } // Exact down-only namespace was validated by the format contract.
            if denseQwen35 {
                let parts = name.split(separator: ".", omittingEmptySubsequences: false)
                guard parts.count == 6, parts[0] == "language_model", parts[1] == "model",
                    parts[2] == "layers", let layer = Int(parts[3]), layer >= 0,
                    String(layer) == String(parts[3]), parts[4] == "mlp",
                    ["gate_proj", "up_proj", "down_proj"].contains(String(parts[5]))
                else {
                    throw Failure.invalid("unsupported dense qwen3_5 JANGH module path \(name)")
                }
                continue
            }
            var components = name.split(separator: ".", omittingEmptySubsequences: false)
            // DeepSeek-V4.1 JANGHT: stacked routed banks at `layers.L.ffn.experts.{gate,up,down}_proj` (the release
            // checkpoint namespace; the dSpark `mtp.*` banks are loaded by the draft, not here). That type only.
            if modelType == "deepseek_v41" {
                guard components.count == 5, components[0] == "layers", let layer = Int(components[1]), layer >= 0,
                    String(layer) == String(components[1]), components[2] == "ffn", components[3] == "experts",
                    ["gate_proj", "up_proj", "down_proj"].contains(String(components[4]))
                else { throw Failure.invalid("unsupported deepseek_v41 custom module path \(name)") }
                continue
            }
            // Nemotron-H latent MoE (Nemotron 3.5 Bolt): gate-less relu2 banks at
            // `backbone.layers.L.mixer.switch_mlp.{fc1,fc2}`; admitted for that model type only.
            if modelType == "nemotron_h" {
                guard components.count == 6, components[0] == "backbone",
                    components[1] == "layers", let layer = Int(components[2]), layer >= 0,
                    String(layer) == String(components[2]), components[3] == "mixer",
                    components[4] == "switch_mlp",
                    JANGHFormatContract.relu2Roles.contains(String(components[5]))
                else { throw Failure.invalid("unsupported nemotron_h custom module path \(name)") }
                continue
            }
            // MiniCPM-V 4.7 (Qwen3.5-MoE text under a VLM wrapper) keeps its routed banks at
            // `language_model.model.layers.L.mlp.switch_mlp.*`; admitted for that model type only.
            if (modelType == "minicpmv4_7" || modelType == "qwen3_5_moe"),
                components.first == "language_model" {
                components.removeFirst()
            }
            guard components.count == 6, components[0] == "model",
                components[1] == "layers", let layer = Int(components[2]), layer >= 0,
                String(layer) == String(components[2]),
                components[3] == (modelType == "lfm2_moe" ? "feed_forward" : "mlp"),
                components[4] == "switch_mlp"
            else { throw Failure.invalid("unsupported custom routed module path \(name)") }
        }
        // Generic lookup also tries model/language_model wrapper spellings. An
        // ordinary alias could otherwise win by lookup order over our explicit skip.
        let canonicalCustom = Set(customModules.map(Self.unwrappedModule))
        guard canonicalCustom.count == customModules.count else {
            throw Failure.invalid("duplicate custom module wrapper aliases")
        }
        for name in plan.keys where !customModules.contains(name) {
            guard !canonicalCustom.contains(Self.unwrappedModule(name)) else {
                throw Failure.invalid("ordinary/custom module wrapper alias collision \(name)")
            }
        }
        var ordinary = plan
        for name in customModules { ordinary[name] = false }
        root["jangtq"] = header
        root["quantization"] = ordinary
        root["quantization_config"] = ordinary
        self.contract = contract
        self.modelType = modelType
        self.customModules = customModules
        self.customConfiguration = customConfiguration
        self.ordinaryConfiguration = try Self.encode(root)
    }

    private static func object(_ data: Data) throws -> [String: Any] {
        guard data.count <= 16 * 1024 * 1024,
            let object = try JANGHHeaderAdapter.validateJSON(data) as? [String: Any]
        else { throw Failure.invalid("invalid or oversized metadata object") }
        return object
    }

    private static func isNumericCalibrationMetadata(_ value: Any) -> Bool {
        if let number = value as? NSNumber { return number.doubleValue.isFinite }
        if let values = value as? [String: Any] {
            return values.values.allSatisfy(isNumericCalibrationMetadata)
        }
        if let values = value as? [Any] { return values.allSatisfy(isNumericCalibrationMetadata) }
        return false
    }

    private static func optionalObject(_ owner: [String: Any], _ name: String) throws
        -> [String: Any]?
    {
        guard let value = owner[name], !(value is NSNull) else { return nil }
        guard let result = value as? [String: Any] else {
            throw Failure.invalid("invalid metadata object \(name)")
        }
        return result
    }

    private static func flatten(_ source: [String: Any]) throws -> [String: Any] {
        var result = source
        if let value = result.removeValue(forKey: "per_tensor") {
            guard let entries = value as? [String: Any] else {
                throw Failure.invalid("invalid per_tensor plan")
            }
            for (name, entry) in entries {
                guard name.contains("."), entry is [String: Any] else {
                    throw Failure.invalid("invalid per_tensor module \(name)")
                }
                if let prior = result[name],
                    try encode(["v": prior]) != encode(["v": entry])
                {
                    throw Failure.invalid("contradictory flat and per_tensor module \(name)")
                }
                result[name] = entry
            }
        }
        return result
    }

    private static func unwrappedModule(_ name: String) -> String {
        var result = name
        if result.hasPrefix("language_model.") { result.removeFirst("language_model.".count) }
        if result.hasPrefix("model.") { result.removeFirst("model.".count) }
        return result
    }

    private static func encode(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
}
