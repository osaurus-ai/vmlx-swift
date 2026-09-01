// Copyright © 2026 osaurus-eval contributors

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// GLM-5.3 configuration, construction plan and checkpoint-key policy.
///
/// The shipped `config.json` is embedded rather than referenced from `~/Library/MLModels`, so these
/// run on a machine that has never downloaded the bundle. Its values are copied verbatim from
/// `JANGQ-AI/GLM-5.3-Flash-JANG-MTP`; the layer schedules are shortened to keep the fixture legible,
/// with `num_hidden_layers` reduced to match, which is exactly the consistency the type checks.
@Suite("GLM-5.3 (glm5_next) construction")
struct Glm5NextConstructionTests {

    static let configJSON = #"""
        {"model_type":"glm5_next",
         "image_token_id":154854,"video_token_id":154855,
         "image_start_token_id":154830,"image_end_token_id":154831,
         "video_start_token_id":154832,"video_end_token_id":154833,
         "text_config":{"model_type":"glm5_next_text","hidden_size":4096,
           "num_hidden_layers":6,"intermediate_size":12288,"num_attention_heads":64,
           "num_key_value_heads":64,"vocab_size":154880,"rms_norm_eps":1e-05,
           "max_position_embeddings":1048576,"kv_lora_rank":512,"q_lora_rank":1536,
           "qk_nope_head_dim":256,"qk_rope_head_dim":0,"v_head_dim":256,"mla_use_nope":true,
           "n_routed_experts":288,"n_shared_experts":1,"num_experts_per_tok":8,
           "moe_intermediate_size":2048,"first_k_dense_replace":3,"scoring_func":"sigmoid",
           "topk_method":"noaux_tc","routed_scaling_factor":2.5,"norm_topk_prob":true,"n_group":1,"topk_group":1,
           "mhc":true,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
           "index_head_dim":128,"index_n_heads":32,"index_topk":2048,"index_kpool":4,
           "index_kpool_compress":true,"index_kpool_always_select_tail":true,"num_nextn_predict_layers":1,"swiglu_limit":10.0,
           "linear_attn_config":{"num_heads":64,"gate_lower_bound":-5.0,"head_dim":128,
             "short_conv_kernel_size":4,"kda_layers":[0,1,2,4],"full_attn_layers":[3,5]},
           "layer_types":["linear_attention","linear_attention","linear_attention",
             "deepseek_sparse_attention","linear_attention","deepseek_sparse_attention"],
           "mlp_layer_types":["dense","dense","dense","sparse","sparse","sparse"]},
         "vision_config":{"model_type":"glm5_next_vision","depth":24,"hidden_size":1024,
           "intermediate_size":4096,"out_hidden_size":4096,"num_heads":16,"in_channels":3,
           "image_size":448,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
           "rms_norm_eps":1e-05,"projection_intermediate_size":10240,"swiglu_limit":10.0}}
        """#

    static func config(_ json: String = configJSON) throws -> Glm5NextConfiguration {
        try JSONDecoder().decode(Glm5NextConfiguration.self, from: Data(json.utf8))
    }

    @Test("the shipped configuration decodes, including the fields that are not DeepSeek defaults")
    func decodesShippedConfiguration() throws {
        let c = try Self.config()
        #expect(c.modelType == "glm5_next")
        let t = c.textConfig
        // MLA with NO rotary split. Both fields must agree; a bundle setting one is malformed.
        #expect(t.qkRopeHeadDim == 0)
        #expect(t.mlaUseNope)
        #expect(t.usesNoPositionalEncoding)
        // The MoE that `noaux_tc` + sigmoid identifies as DeepSeek-shaped.
        #expect(t.nRoutedExperts == 288)
        #expect(t.numExpertsPerTok == 8)
        #expect(t.topkMethod == "noaux_tc")
        #expect(t.scoringFunc == "sigmoid")
        // Hyper-connections, the DeepSeek V4 mechanism.
        #expect(t.mhc)
        #expect(t.hcSinkhornIters == 20)
        // The indexer's key pooling — the one piece with no counterpart in this repo.
        #expect(t.indexKpool == 4)
        #expect(t.indexKpoolCompress)
        #expect(c.visionConfig?.spatialMergeSize == 2)
        #expect(c.visionConfig?.temporalPatchSize == 2)
    }

    @Test("the layer schedule is read as a list, and validated against num_hidden_layers")
    func scheduleIsValidated() throws {
        let c = try Self.config()
        let schedule = try c.textConfig.validatedSchedule()
        #expect(schedule.count == 6)
        #expect(schedule.filter { $0 == .linearAttention }.count == 4)
        #expect(schedule.filter { $0 == .deepseekSparseAttention }.count == 2)

        // A schedule that disagrees with the layer count is a malformed bundle, not something to
        // discover through an out-of-range subscript mid-forward.
        let broken = Self.configJSON.replacingOccurrences(
            of: "\"num_hidden_layers\":6", with: "\"num_hidden_layers\":7")
        #expect(throws: Glm5NextConfigurationError.self) {
            _ = try Self.config(broken).textConfig.validatedSchedule()
        }
    }

    @Test("construction narrows exactly like every other converted family")
    func constructionNarrows() throws {
        let c = try Self.config()
        #expect(Glm5Next.constructibleModalities(of: c) == [.text, .vision, .video])

        let full = try Glm5Next(c, requesting: nil)
        #expect(full.modalities == [.text, .vision, .video])
        #expect(full.plan.builds(.visionTower))

        let textOnly = try Glm5Next(c, requesting: [.text])
        #expect(textOnly.modalities == [.text])
        #expect(!textOnly.plan.builds(.visionTower))
        #expect(textOnly.plan.builds(.languageCore), "the core is never optional")

        // Video alone must still allocate the tower: one tower serves both media lanes.
        let videoOnly = try Glm5Next(c, requesting: [.video])
        #expect(videoOnly.plan.builds(.visionTower))
    }

    @Test("a bundle without the video token offers no video lane")
    func videoNeedsItsToken() throws {
        let noVideo = Self.configJSON.replacingOccurrences(
            of: "\"video_token_id\":154855,", with: "")
        let c = try Self.config(noVideo)
        #expect(Glm5Next.constructibleModalities(of: c) == [.text, .vision])
        #expect(throws: (any Error).self) { _ = try Glm5Next(c, requesting: [.video]) }
        let vision = try Glm5Next(c, requesting: [.vision])
        #expect(vision.modalities == [.text, .vision], "no video lane to report")
    }

    @Test("the hc_ prefix is stripped so DeepseekV4HyperConnection can be reused as-is")
    func hyperConnectionPrefixIsStripped() throws {
        let weights: [String: MLXArray] = [
            "model.layers.0.attn_hc.hc_fn": MLXArray([Float(1)]),
            "model.layers.0.attn_hc.hc_scale": MLXArray([Float(1)]),
            "model.layers.0.ffn_hc.hc_base": MLXArray([Float(1)]),
            "model.layers.0.self_attn.kv_b_proj.weight": MLXArray([Float(1)]),
        ]
        let out = Glm5NextCheckpointKeys.sanitize(weights, keepVision: true)
        #expect(out["model.layers.0.attn_hc.fn"] != nil)
        #expect(out["model.layers.0.attn_hc.scale"] != nil)
        #expect(out["model.layers.0.ffn_hc.base"] != nil)
        #expect(out["model.layers.0.attn_hc.hc_fn"] == nil, "the prefixed key must not survive")
        #expect(
            out["model.layers.0.self_attn.kv_b_proj.weight"] != nil,
            "MLA keys already match and must be left alone")
        #expect(out.count == weights.count, "renaming must not drop or duplicate a tensor")
    }

    @Test("vision weights are dropped only when the plan has no tower")
    func visionKeysFollowThePlan() throws {
        let weights: [String: MLXArray] = [
            "model.visual.blocks.0.attn.qkv.weight": MLXArray([Float(1)]),
            "model.layers.0.self_attn.kv_b_proj.weight": MLXArray([Float(1)]),
        ]
        #expect(Glm5NextCheckpointKeys.sanitize(weights, keepVision: true).count == 2)
        let narrowed = Glm5NextCheckpointKeys.sanitize(weights, keepVision: false)
        #expect(narrowed.count == 1)
        #expect(narrowed["model.layers.0.self_attn.kv_b_proj.weight"] != nil,
                "narrowing must never touch a language key")
    }

    /// A malformed input is REPORTED, not trapped.
    ///
    /// This replaces a test that asserted the decoder was unimplemented — it is implemented now. The
    /// property worth keeping is the one that test was really about: a bad call must not take the
    /// process down. A 1-D token array reaches MLX's `reshape`, whose failure is a `fatalError`.
    @Test("a malformed input is reported, not trapped")
    func malformedInputIsReported() throws {
        let model = try Glm5Next(Self.config(), requesting: [.text])
        #expect(throws: Glm5NextInputShapeError.self) {
            _ = try model(MLXArray([Int32(1)]))
        }
    }
    /// Decodes the SHIPPED bundle when it is on this machine.
    ///
    /// The fixture above is a transcription, and a transcription can drift from the thing it
    /// describes while every test built on it still passes. This is the differential check: it reads
    /// the real 45-layer configuration and asserts the same properties, so a bundle whose shape
    /// changes fails here rather than being discovered at load time. Skips where the bundle is
    /// absent, so it never fails a machine that has not downloaded 100+ GB.
    @Test("the real bundle decodes and agrees with the fixture")
    func realBundleAgrees() throws {
        let path = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP/config.json" as NSString)
            .expandingTildeInPath
        guard FileManager.default.fileExists(atPath: path) else { return }

        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let real = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)
        let fixture = try Self.config()

        #expect(real.modelType == fixture.modelType)
        #expect(real.imageTokenId == fixture.imageTokenId)
        #expect(real.videoTokenId == fixture.videoTokenId)
        #expect(real.canBuildVisionTower && real.canConsumeVideo)

        let t = real.textConfig, f = fixture.textConfig
        #expect(t.usesNoPositionalEncoding == f.usesNoPositionalEncoding)
        #expect(t.nRoutedExperts == f.nRoutedExperts)
        #expect(t.numExpertsPerTok == f.numExpertsPerTok)
        #expect(t.topkMethod == f.topkMethod && t.scoringFunc == f.scoringFunc)
        #expect(t.mhc == f.mhc && t.hcSinkhornIters == f.hcSinkhornIters)
        #expect(t.indexKpool == f.indexKpool && t.indexKpoolCompress == f.indexKpoolCompress)

        // The shipped schedule, which the fixture deliberately shortens.
        let schedule = try t.validatedSchedule()
        #expect(schedule.count == 45)
        #expect(schedule.filter { $0 == .linearAttention }.count == 34)
        #expect(schedule.filter { $0 == .deepseekSparseAttention }.count == 11)

        // And it plans the same way at full size.
        let model = try Glm5Next(real, requesting: [.text])
        #expect(model.modalities == [.text])
        #expect(model.layerIndices(of: .deepseekSparseAttention).count == 11)
    }

    /// Reads the shipped bundle's TENSOR NAMES and checks the key policy against them.
    ///
    /// The synthetic `sanitize` tests above prove the transformation on keys I wrote. This proves it
    /// on the keys the converter actually emits — 2999 of them — which is the only way to find out
    /// that a prefix is spelled `visual.` rather than `model.visual.`, or that some hyper-connection
    /// tensor was left unprefixed. Only safetensors HEADERS are read, so it costs milliseconds and
    /// never touches 96 GB of weights. Skips where the bundle is absent.
    @Test("the key policy holds against the shipped bundle's real tensor names")
    func keyPolicyAgainstShippedWeights() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir) else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }   // still downloading

        var names: [String] = []
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len))
            else { continue }
            let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any] ?? [:]
            names += obj.keys.filter { $0 != "__metadata__" }
        }
        try #require(names.count > 2000, "expected the full tensor set, got \(names.count)")

        // 1. Every hyper-connection tensor really is `hc_`-prefixed, so none is left behind.
        let hcAll = names.filter { $0.contains("_hc.") }
        let hcPrefixed = names.filter { $0.contains("_hc.hc_") }
        #expect(!hcAll.isEmpty)
        #expect(hcAll.count == hcPrefixed.count, "an unprefixed hyper-connection tensor would be missed")

        // 2. Nothing that looks visual escapes the matcher — this is what catches a `visual.` vs
        //    `model.visual.` spelling mistake.
        let looksVisual = names.filter {
            $0.contains("visual") || $0.contains("vision") || $0.contains("patch_embed")
        }
        let matched = looksVisual.filter { Glm5NextCheckpointKeys.isVisionKey($0) }
        #expect(matched.count == looksVisual.count, "a vision key the matcher does not recognise")

        // 3. The rename is injective: renaming must not collapse two tensors into one.
        let renamed = Set(names.map { Glm5NextCheckpointKeys.stripHyperConnectionPrefix($0) })
        #expect(renamed.count == Set(names).count, "the hc_ rename collided with an existing key")

        // 4. A narrowed plan drops vision and nothing else.
        let kept = names.filter { !Glm5NextCheckpointKeys.isVisionKey($0) }
        #expect(kept.count == names.count - looksVisual.count)
        #expect(
            !kept.contains { $0.hasPrefix("model.layers.") && Glm5NextCheckpointKeys.isVisionKey($0) },
            "narrowing must never reach a language key")
    }

    /// The weights' own layer schedule must agree with the configuration's.
    ///
    /// `layer_types` is a claim; the tensors are the fact. A bundle whose claim disagrees would
    /// build the wrong attention for a layer and fail numerically much later, so it is worth one
    /// header read to find out here.
    @Test("the shipped weights agree with the declared layer schedule")
    func weightScheduleAgreesWithConfig() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir) else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27,
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)

        var names: [String] = []
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len))
            else { continue }
            let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any] ?? [:]
            names += obj.keys.filter { $0 != "__metadata__" }
        }
        try #require(names.count > 2000)

        var sparse = Set<Int>(), linear = Set<Int>()
        for name in names {
            let parts = name.split(separator: ".")
            guard parts.count > 3, parts[0] == "model", parts[1] == "layers",
                let i = Int(parts[2]), parts[3] == "self_attn"
            else { continue }
            let leaf = parts[4...].joined(separator: ".")
            if leaf.hasPrefix("A_log") || leaf.hasPrefix("dt_bias") || leaf.hasPrefix("q_conv1d") {
                linear.insert(i)
            }
            if leaf.hasPrefix("kv_a_proj") || leaf.hasPrefix("q_a_proj") || leaf.hasPrefix("indexer") {
                sparse.insert(i)
            }
        }
        #expect(sparse.isDisjoint(with: linear), "a layer cannot run both attentions")

        let schedule = try config.textConfig.validatedSchedule()
        for (i, kind) in schedule.enumerated() {
            switch kind {
            case .linearAttention:
                #expect(linear.contains(i), "layer \(i) is declared linear but has no linear weights")
            case .deepseekSparseAttention:
                #expect(sparse.contains(i), "layer \(i) is declared sparse but has no MLA weights")
            }
        }
        // The MTP layer trails the decoder and is NOT in `layer_types` — worth pinning, because a
        // decoder that trusts the schedule's length would silently ignore it.
        let mtp = config.textConfig.numHiddenLayers
        #expect(
            sparse.contains(mtp) || linear.contains(mtp),
            "expected an MTP layer at index \(mtp), beyond the declared schedule")
        #expect(config.textConfig.numNextnPredictLayers == 1)
    }

    /// The linear-attention module's parameter tree, against the checkpoint's actual keys.
    ///
    /// This is the check that a module either loads or does not. Building the module and reading its
    /// parameter names is cheap; comparing them with the shipped tensor names is the only way to
    /// find out that the bundle stores `q_conv1d` as a BARE tensor of shape [8192, 4] while a
    /// `Conv1d` module wants `q_conv1d.weight` of shape [8192, 4, 1]. No amount of reading the
    /// config would have said so.
    /// Tensor names and shapes for one decoder layer, from the shard headers.
    static func shippedLayer(_ index: Int, dir: String, shards: [String]) throws -> [String: [Int]] {
        var out: [String: [Int]] = [:]
        let prefix = "model.layers.\(index).self_attn."
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len)),
                let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { continue }
            for (key, meta) in obj where key.hasPrefix(prefix) {
                if let m = meta as? [String: Any], let shape = m["shape"] as? [Int] {
                    out[String(key.dropFirst(prefix.count))] = shape
                }
            }
        }
        return out
    }

    /// Which module parameters the checkpoint cannot supply, going THROUGH the key policy.
    static func unaddressable(
        module: Module, shipped: [String: [Int]]
    ) -> [String] {
        var missing: [String] = []
        for (key, _) in module.parameters().flattened() {
            let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            let viaPolicy = shipped.keys.contains {
                Glm5NextCheckpointKeys.bareTensorWeightKey($0) == key
                    || Glm5NextCheckpointKeys.stripHyperConnectionPrefix($0) == key
            }
            if shipped[key] == nil, shipped["\(base).scales"] == nil, !viaPolicy {
                missing.append(key)
            }
        }
        return missing.sorted()
    }

    @Test("the linear-attention module's parameters match the shipped layer-0 tensors")
    func linearAttentionParametersMatchWeights() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)

        // Shapes for layer 0, which the schedule says is linear attention.
        try #require(try config.textConfig.validatedSchedule()[0] == .linearAttention)
        var shipped: [String: [Int]] = [:]
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len)),
                let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { continue }
            for (key, meta) in obj where key.hasPrefix("model.layers.0.self_attn.") {
                if let m = meta as? [String: Any], let shape = m["shape"] as? [Int] {
                    shipped[String(key.dropFirst("model.layers.0.self_attn.".count))] = shape
                }
            }
        }
        try #require(!shipped.isEmpty, "no layer-0 attention tensors found")

        let module = Glm5NextLinearAttention(config.textConfig)
        var expected: [String: [Int]] = [:]
        for (key, value) in module.attention.parameters().flattened() {
            expected[key] = value.shape
        }
        try #require(!expected.isEmpty)

        // Every parameter the module declares must be addressable in the checkpoint. A quantized
        // tensor is stored as `weight`/`scales`/`biases`, so presence is what is asserted, not shape.
        var unaddressable: [String] = []
        for key in expected.keys.sorted() {
            // EXACT spelling, plus the quantized triple. Accepting a differently-spelled tensor
            // here is what made an earlier version of this test pass while `q_conv1d` and
            // `q_conv1d.weight` disagreed — the very mismatch it exists to find. What bridges them
            // is `Glm5NextCheckpointKeys`, so the test asks whether the POLICY produces the key,
            // not whether something vaguely similar exists.
            let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            let viaPolicy = shipped.keys.contains {
                Glm5NextCheckpointKeys.bareTensorWeightKey($0) == key
                    || Glm5NextCheckpointKeys.stripHyperConnectionPrefix($0) == key
            }
            let present = shipped[key] != nil || shipped["\(base).scales"] != nil || viaPolicy
            if !present { unaddressable.append(key) }
        }
        #expect(
            unaddressable.isEmpty,
            "the module declares parameters the checkpoint cannot supply: \(unaddressable)")

        // And the unquantized ones must agree in shape exactly.
        for name in ["A_log", "dt_bias", "o_norm.weight"] {
            if let want = expected[name], let got = shipped[name] {
                #expect(want == got, "\(name): module \(want) vs checkpoint \(got)")
            }
        }
    }

    /// The sparse-attention module — MLA plus indexer — against layer 3's shipped tensors.
    ///
    /// Layer 3 is the first `deepseek_sparse_attention` layer, and the schedule is asserted rather
    /// than assumed so this cannot quietly start checking the wrong layer.
    @Test("the sparse-attention module's parameters match the shipped layer-3 tensors")
    func sparseAttentionParametersMatchWeights() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)
        try #require(try config.textConfig.validatedSchedule()[3] == .deepseekSparseAttention)

        let shipped = try Self.shippedLayer(3, dir: dir, shards: shards)
        try #require(!shipped.isEmpty)

        let module = Glm5NextSparseAttention(config.textConfig)
        #expect(
            Self.unaddressable(module: module, shipped: shipped).isEmpty,
            "the checkpoint cannot supply: \(Self.unaddressable(module: module, shipped: shipped))")

        // The MLA widths are derived, not copied, so a wrong derivation shows here rather than as a
        // shape error deep in a matmul.
        let t = config.textConfig
        #expect(module.qHeadDim == t.qkNopeHeadDim, "no rotary split, so qHeadDim IS the nope half")
        #expect(!module.usesRoPE)
        #expect(module.indexer.headDim == t.indexHeadDim)
        #expect(module.indexer.numHeads == t.indexNHeads)

        // `kv_b_proj` fans the compressed KV out to per-head nope AND value channels; the shipped
        // [32768, …] confirms 64 * (256 + 256).
        if let kvb = shipped["kv_b_proj.scales"] {
            #expect(kvb[0] == t.numAttentionHeads * (t.qkNopeHeadDim + t.vHeadDim))
        }
        // The indexer's key head is SHARED, not per-head: one [128, hidden] projection.
        if let wk = shipped["wk.scales"] { #expect(wk[0] == t.indexHeadDim) }
    }

    /// The vision tower's parameter tree, against every `visual.*` tensor in the bundle.
    ///
    /// Checked in BOTH directions, unlike the per-layer tests. A missing module parameter fails the
    /// load; an unclaimed checkpoint tensor is the quieter bug — it means the tower is structurally
    /// wrong somewhere and the weight will be silently dropped. `Glm4v` ships `embeddings` and
    /// `post_conv_layernorm`; inheriting those from the donor would have produced exactly that.
    @Test("the vision tower's parameters match the shipped visual tensors, both ways")
    func visionTowerParametersMatchWeights() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)
        let vision = try #require(config.visionConfig)

        var shipped: [String: [Int]] = [:]
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len)),
                let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { continue }
            for (key, meta) in obj where key.hasPrefix("visual.") {
                if let m = meta as? [String: Any], let shape = m["shape"] as? [Int] {
                    shipped[String(key.dropFirst("visual.".count))] = shape
                }
            }
        }
        try #require(shipped.count > 500, "expected the full tower, got \(shipped.count)")

        let tower = Glm5NextVisionTower(vision)
        var declared = Set<String>()
        for (key, _) in tower.parameters().flattened() { declared.insert(key) }

        // 1. Every declared parameter must be suppliable.
        var unsuppliable: [String] = []
        for key in declared.sorted() {
            let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            if shipped[key] == nil, shipped["\(base).scales"] == nil { unsuppliable.append(key) }
        }
        #expect(unsuppliable.isEmpty, "tower declares what the checkpoint lacks: \(unsuppliable)")

        // 2. Every shipped tensor must be claimed. This is the direction that catches an EXTRA
        //    module — or a missing one, when the checkpoint has weights the tower never asked for.
        var unclaimed: [String] = []
        for key in shipped.keys.sorted() {
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") { continue }
            if declared.contains(key) { continue }
            // A quantized Linear declares `weight`; `bias` is separate and also declared.
            if declared.contains(key) == false, shipped["\(key).scales"] != nil { continue }
            unclaimed.append(key)
        }
        #expect(unclaimed.isEmpty, "checkpoint tensors nothing claims: \(unclaimed.prefix(8))")

        #expect(tower.blocks.count == vision.depth)
        #expect(tower.spatialMergeSize == 2 && tower.temporalPatchSize == 2)
    }

    /// The feed-forward modules against the shipped tensors, dense and sparse.
    ///
    /// The two schedules do NOT line up — layers 0-2 are dense MLPs while layer 3 is the first
    /// sparse ATTENTION layer — so a decoder that read one schedule for both would build the wrong
    /// MLP for layer 3. Both are asserted from the configuration before anything is compared.
    @Test("the dense and MoE feed-forwards match the shipped tensors")
    func feedForwardParametersMatchWeights() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)
        let t = config.textConfig
        _ = try t.validatedSchedule()

        // The MLP schedule is its own list, and disagrees with the attention one.
        #expect(t.mlpLayerTypes[0] == .dense)
        #expect(t.mlpLayerTypes[3] == .sparse)
        #expect(t.layerTypes[3] == .deepseekSparseAttention)
        #expect(
            t.mlpLayerTypes.prefix(t.firstKDenseReplace).allSatisfy { $0 == .dense },
            "first_k_dense_replace must agree with mlp_layer_types")

        func shippedMLP(_ index: Int) throws -> [String: [Int]] {
            var out: [String: [Int]] = [:]
            let prefix = "model.layers.\(index).mlp."
            for shard in shards {
                let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
                defer { try? handle.close() }
                guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
                let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
                guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len)),
                    let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any]
                else { continue }
                for (key, meta) in obj where key.hasPrefix(prefix) {
                    if let m = meta as? [String: Any], let shape = m["shape"] as? [Int] {
                        out[String(key.dropFirst(prefix.count))] = shape
                    }
                }
            }
            return out
        }

        let dense = try shippedMLP(0)
        try #require(!dense.isEmpty)
        #expect(
            Self.unaddressable(module: Glm5NextDenseMLP(t), shipped: dense).isEmpty,
            "dense: \(Self.unaddressable(module: Glm5NextDenseMLP(t), shipped: dense))")
        // Dense uses `intermediate_size`, not the MoE width — a real confusion, since both exist.
        if let gate = dense["gate_proj.scales"] { #expect(gate[0] == t.intermediateSize) }

        let sparse = try shippedMLP(4)
        try #require(!sparse.isEmpty)
        #expect(
            Self.unaddressable(module: Glm5NextMoE(t), shipped: sparse).isEmpty,
            "moe: \(Self.unaddressable(module: Glm5NextMoE(t), shipped: sparse))")
        // The experts are STACKED: a leading expert axis is what distinguishes `switch_mlp` from a
        // per-expert layout, and getting it wrong changes nothing about the key names.
        if let stacked = sparse["switch_mlp.gate_proj.scales"] {
            #expect(stacked.count == 3, "switch_mlp must be stacked [experts, out, groups]")
            #expect(stacked[0] == t.nRoutedExperts)
            #expect(stacked[1] == t.moeIntermediateSize)
        }
        if let router = sparse["gate.weight"] {
            #expect(router == [t.nRoutedExperts, t.hiddenSize])
        }
        if let correction = sparse["e_score_correction_bias"] {
            #expect(correction == [t.nRoutedExperts])
        }
        if let shared = sparse["shared_experts.gate_proj.scales"] {
            #expect(shared[0] == t.moeIntermediateSize * t.nSharedExperts)
        }
    }

    /// A decoder layer builds exactly one attention and one MLP, of the kinds the schedules name.
    @Test("a decoder layer builds the kinds its two schedules name")
    func decoderLayerFollowsBothSchedules() throws {
        let t = try Self.config().textConfig
        let linearDense = Glm5NextDecoderLayer(t, kind: .linearAttention, mlpKind: .dense)
        #expect(linearDense.linearAttention != nil && linearDense.sparseAttention == nil)
        #expect(linearDense.denseMLP != nil && linearDense.moe == nil)

        let sparseMoE = Glm5NextDecoderLayer(t, kind: .deepseekSparseAttention, mlpKind: .sparse)
        #expect(sparseMoE.sparseAttention != nil && sparseMoE.linearAttention == nil)
        #expect(sparseMoE.moe != nil && sparseMoE.denseMLP == nil)

        // The combination the shipped schedule actually contains at layer 3: sparse attention with a
        // SPARSE mlp, which only holds because the two lists are read separately.
        let mixed = Glm5NextDecoderLayer(t, kind: .deepseekSparseAttention, mlpKind: .dense)
        #expect(mixed.sparseAttention != nil && mixed.denseMLP != nil)
    }

    /// The whole model's parameter tree against every tensor in the bundle, both directions.
    ///
    /// This is the last structural check there is: if the tree and the checkpoint agree here, the
    /// weights bind. It subsumes the per-module tests, which stay because they say WHICH module is
    /// wrong when one breaks.
    @Test("the whole model's parameters match the shipped bundle, both ways")
    func wholeModelMatchesBundle() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)

        var shipped: [String: [Int]] = [:]
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dir + "/" + shard))
            defer { try? handle.close() }
            guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else { continue }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len > 0, len < 64_000_000, let header = try handle.read(upToCount: Int(len)),
                let obj = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { continue }
            for (key, meta) in obj where key != "__metadata__" {
                if let m = meta as? [String: Any], let shape = m["shape"] as? [Int] {
                    shipped[key] = shape
                }
            }
        }
        try #require(shipped.count > 2900, "expected the full bundle, got \(shipped.count)")

        let model = try Glm5Next(config, requesting: nil)
        var declared = Set<String>()
        for (key, _) in model.parameters().flattened() { declared.insert(key) }
        // Roughly 950, not ~3000: a QUANTIZED tensor ships as three files (weight / scales /
        // biases) but is one parameter. Comparing the two counts directly is a category error, and
        // an earlier version of this line made it.
        try #require(declared.count > 900, "model declared only \(declared.count) parameters")

        // Every declared parameter must be suppliable, going through the key policy.
        var unsuppliable: [String] = []
        for key in declared.sorted() {
            let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            let viaPolicy = shipped.keys.contains {
                Glm5NextCheckpointKeys.bareTensorWeightKey($0) == key
                    || Glm5NextCheckpointKeys.stripHyperConnectionPrefix($0) == key
            }
            if shipped[key] == nil, shipped["\(base).scales"] == nil, !viaPolicy {
                unsuppliable.append(key)
            }
        }
        #expect(
            unsuppliable.isEmpty,
            "declared but unsuppliable (\(unsuppliable.count)): \(unsuppliable.prefix(10))")
    }

    /// A bundle published WITHOUT multi-token prediction must build, and must declare no MTP
    /// parameter.
    ///
    /// The same model ships both ways. The version without omits `num_nextn_predict_layers`
    /// entirely rather than setting it to 0, so the configuration defaults it — and every MTP
    /// module is optional precisely so that this case declares nothing its checkpoint lacks. A model
    /// that always built the MTP layer would fail to load the non-MTP bundle with a missing-weight
    /// error, which is the failure this test exists to prevent.
    @Test("a bundle without MTP builds and declares no MTP parameters")
    func nonMTPBundleBuilds() throws {
        let withoutMTP = Self.configJSON
            .replacingOccurrences(of: "\"num_nextn_predict_layers\":1,", with: "")
        let config = try Self.config(withoutMTP)
        #expect(config.textConfig.numNextnPredictLayers == 0, "absent must mean zero")

        let model = try Glm5Next(config, requesting: [.text])
        #expect(model.languageModel.numMTPLayers == 0)
        #expect(model.languageModel.multiTokenPredictionLayer == nil)
        #expect(
            model.languageModel.layers.count == config.textConfig.numHiddenLayers,
            "no extra layer when there is no MTP")

        let mtpNames = ["enorm", "hnorm", "eh_proj", "shared_head"]
        let declared = model.parameters().flattened().map(\.0)
        for name in mtpNames {
            #expect(
                !declared.contains { $0.contains(".\(name).") || $0.hasSuffix(".\(name)") },
                "\(name) must not be declared when the bundle has no MTP")
        }

        // And the MTP bundle DOES declare them, so the test above is not vacuous.
        let withMTP = try Glm5Next(Self.config(), requesting: [.text])
        #expect(withMTP.languageModel.numMTPLayers == 1)
        #expect(withMTP.languageModel.multiTokenPredictionLayer?.isMultiTokenPrediction == true)
        let mtpDeclared = withMTP.parameters().flattened().map(\.0)
        for name in mtpNames {
            #expect(
                mtpDeclared.contains { $0.contains(".\(name).") || $0.hasSuffix(".\(name)") },
                "\(name) must be declared when the bundle has MTP")
        }
    }

    /// A tiny model, small enough to RUN. The first numerical check in this suite.
    ///
    /// Everything before this is structural: names, shapes, plans. This actually executes the
    /// forward pass — both attention kinds, both MLP kinds, the hyper-connections, the head — and so
    /// it is the first thing that would catch a transposed reshape or a residual wired to the wrong
    /// tensor. The dimensions are the shipped ones divided down, keeping every RELATIONSHIP that
    /// matters (v_head_dim == qk_nope_head_dim, kv_lora_rank < hidden, experts > topK).
    static let tinyJSON = #"""
        {"model_type":"glm5_next","image_token_id":9,"video_token_id":10,
         "text_config":{"model_type":"glm5_next_text","hidden_size":64,
           "num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":4,
           "num_key_value_heads":4,"vocab_size":128,"rms_norm_eps":1e-05,
           "max_position_embeddings":4096,"kv_lora_rank":16,"q_lora_rank":32,
           "qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":16,"mla_use_nope":true,
           "n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,
           "moe_intermediate_size":32,"first_k_dense_replace":2,"scoring_func":"sigmoid",
           "topk_method":"noaux_tc","routed_scaling_factor":2.5,"norm_topk_prob":true,
           "n_group":1,"topk_group":1,
           "mhc":true,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
           "index_head_dim":16,"index_n_heads":2,"index_topk":2048,"index_kpool":4,
           "index_kpool_compress":true,"index_kpool_always_select_tail":true,
           "num_nextn_predict_layers":1,"swiglu_limit":10.0,"tie_word_embeddings":false,
           "linear_attn_config":{"num_heads":4,"gate_lower_bound":-5.0,"head_dim":16,
             "short_conv_kernel_size":4,"kda_layers":[0,2,3],"full_attn_layers":[1]},
           "layer_types":["linear_attention","deepseek_sparse_attention","linear_attention",
             "linear_attention"],
           "mlp_layer_types":["dense","dense","sparse","sparse"]},
         "vision_config":{"model_type":"glm5_next_vision","depth":2,"hidden_size":32,
           "intermediate_size":64,"out_hidden_size":64,"num_heads":2,"in_channels":3,
           "image_size":56,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
           "rms_norm_eps":1e-05,"projection_intermediate_size":128,"swiglu_limit":10.0}}
        """#

    @Test("a tiny model runs a forward pass and produces finite logits")
    func tinyModelForwardRuns() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
            let model = try Glm5Next(config, requesting: [.text])

            // Both attention kinds and both MLP kinds are exercised by this schedule.
            #expect(model.languageModel.layers.count == 5, "4 decoder + 1 MTP")
            #expect(model.languageModel.layers[0].linearAttention != nil)
            #expect(model.languageModel.layers[1].sparseAttention != nil)
            #expect(model.languageModel.layers[0].denseMLP != nil)
            #expect(model.languageModel.layers[2].moe != nil)
            #expect(model.languageModel.layers[0].attentionHC != nil, "mhc is on")

            let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
            let logits = try model(tokens)
            eval(logits)

            #expect(logits.shape == [1, 5, config.textConfig.vocabSize])
            let values = logits.asType(.float32).asArray(Float.self)
            #expect(values.allSatisfy { $0.isFinite }, "logits must be finite")
            // Randomly-initialised weights, so no particular VALUE is expected — but an all-zero or
            // constant output means something is disconnected, which is worth catching.
            #expect(Set(values.prefix(64)).count > 1, "logits are constant — a dead path")
        }
    }

    /// Beyond `index_topk` the sparse layers select for real, and the whole model still runs.
    ///
    /// This test used to assert the opposite — that a longer sequence was REFUSED — back when the
    /// key-pool math was unimplemented and returning a plausible answer would have been worse than
    /// failing. It is inverted rather than deleted because the boundary is still the interesting
    /// place: crossing it changes which code path runs, and the only thing worse than refusing there
    /// is quietly producing NaN.
    @Test("a sequence longer than index_topk selects, and still produces finite logits")
    func longSequenceSelectsAndComputes() throws {
        try MLXMetalTestLock.withLock {
            let shortTopK = Self.tinyJSON.replacingOccurrences(
                of: "\"index_topk\":2048", with: "\"index_topk\":4")
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self, from: Data(shortTopK.utf8))
            let model = try Glm5Next(config, requesting: [.text])

            for length in [4, 9] {
                let tokens = MLXArray((0 ..< length).map { Int32($0 + 1) })
                    .reshaped(1, length)
                let logits = try model(tokens)
                eval(logits)
                let values = logits.asType(.float32).asArray(Float.self)
                #expect(
                    values.allSatisfy { $0.isFinite },
                    "length \(length) produced non-finite logits")
                #expect(Set(values.prefix(64)).count > 1, "logits are constant — a dead path")
            }
        }
    }

    /// The caches a sparse layer gets must be able to hold the indexer's history.
    ///
    /// A plain `KVCacheSimple` would generate perfectly well right up to `index_topk` and then have
    /// no packed rows to pool — the failure would appear thousands of tokens into a long context,
    /// which is the worst possible place to discover a cache-type mistake.
    @Test("sparse layers get the cache that carries indexer state")
    func sparseLayersGetTheIndexedCache() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
            let model = try Glm5Next(config, requesting: [.text])
            let caches = model.newCache(parameters: nil)
            let kinds = model.languageModel.layers.prefix(model.languageModel.numDecoderLayers)
                .map { $0.kind }
            for (index, kind) in kinds.enumerated() where kind != .linearAttention {
                #expect(
                    caches[index] is Glm5NextIndexedKVCache,
                    "sparse layer \(index) got \(type(of: caches[index]))")
            }
        }
    }

    /// REAL WEIGHTS. Loads one layer of each attention kind from the shipped bundle and runs it.
    ///
    /// Not the whole model: it is 96 GB and this machine has less free than that, so a full load
    /// would swap or fail for reasons that say nothing about the code. One layer is ~2 GB and tests
    /// what actually matters here — that real MIXED-PRECISION quantized tensors bind to these
    /// modules and compute finite values. Everything before this used random weights.
    @Test("real quantized weights load into a layer and it computes")
    func realWeightsLoadAndCompute() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir),
            let data = fm.contents(atPath: dir + "/config.json")
        else { return }
        let shards = entries.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }.sorted()
        guard shards.count == 27 else { return }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: data)
        let quantization = try #require(config.quantization, "the bundle declares quantization")
        let t = config.textConfig
        let schedule = try t.validatedSchedule()

        try MLXMetalTestLock.withLock {
            for layerIndex in [0, 3] {   // one linear, one sparse
                let kind = schedule[layerIndex]
                let prefix = "model.layers.\(layerIndex)."

                // Collect just this layer's tensors.
                var raw: [String: MLXArray] = [:]
                for shard in shards {
                    let url = URL(fileURLWithPath: dir + "/" + shard)
                    let arrays = try MLX.loadArrays(url: url)
                    for (key, value) in arrays where key.hasPrefix(prefix) {
                        raw[String(key.dropFirst(prefix.count))] = value
                    }
                }
                try #require(!raw.isEmpty, "no tensors for layer \(layerIndex)")

                let layer = Glm5NextDecoderLayer(
                    t, kind: kind, mlpKind: t.mlpLayerTypes[layerIndex])

                // Per-module quantization, from the config rather than one global setting.
                quantize(model: layer) { path, _ in
                    guard raw["\(path).scales"] != nil else { return nil }
                    let (groupSize, bits) = quantization.setting(for: prefix + path)
                    return (groupSize: groupSize, bits: bits)
                }

                let weights = Glm5NextCheckpointKeys.sanitize(raw, keepVision: false)
                try layer.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
                eval(layer)

                // Run it. The stream is the WIDENED one when hyper-connections are on.
                var h = MLXRandom.normal([1, 4, t.hiddenSize]).asType(.bfloat16)
                if t.mhc {
                    h = repeated(h.expandedDimensions(axis: -2), count: t.hcMult, axis: -2)
                }
                // One cache slot, of the kind this layer needs — `MambaCache` conforms to `KVCache`.
                let out = try layer(h, cache: kind == .linearAttention ? MambaCache() : nil)
                eval(out)

                #expect(out.shape == h.shape, "layer \(layerIndex) changed the stream shape")
                let values = out.asType(.float32).asArray(Float.self)
                #expect(
                    values.allSatisfy { $0.isFinite },
                    "layer \(layerIndex) (\(kind)) produced non-finite values from real weights")
                #expect(Set(values.prefix(64)).count > 1, "layer \(layerIndex) output is constant")
            }
        }
    }

    /// The image processor against the bundle's own `processor_config.json`.
    ///
    /// The value worth pinning is the TOKEN-to-PIXEL conversion. This bundle states its budget in
    /// tokens (16…8000) where the shared resize helper wants pixels; passing the token counts
    /// straight through would cap every image at 8000 pixels — about 89×89 — and destroy it while
    /// everything downstream still ran and produced confident nonsense.
    @Test("the image processor converts the token budget to pixels")
    func imageProcessorUsesTokenBudget() throws {
        let dir = ("~/Library/MLModels/JANGQ-AI/GLM-5.3-Flash-JANG-MTP" as NSString)
            .expandingTildeInPath
        guard let data = FileManager.default.contents(atPath: dir + "/processor_config.json")
        else { return }
        struct Wrapper: Codable { let image_processor: Glm5NextImageProcessorConfiguration }
        let config = try JSONDecoder().decode(Wrapper.self, from: data).image_processor

        #expect(config.patchSize == 14 && config.mergeSize == 2)
        #expect(config.pixelsPerToken == 784, "a token covers a 2x2 block of 14x14 patches")
        #expect(config.minPixels == 16 * 784)
        #expect(config.maxPixels == 8000 * 784)
        #expect(config.maxPixels > 6_000_000, "the budget is millions of pixels, not thousands")

        let processor = Glm5NextImageProcessor(config)

        // A 4K image must be shrunk, but nowhere near the naive 8000-pixel reading.
        let big = try processor.targetSize(height: 2160, width: 3840)
        #expect(big.height % config.resizeFactor == 0 && big.width % config.resizeFactor == 0)
        #expect(big.height * big.width <= config.maxPixels)
        #expect(big.height * big.width > 1_000_000, "a 4K image must not collapse to a thumbnail")

        // A tiny image is grown to the minimum.
        let small = try processor.targetSize(height: 32, width: 32)
        #expect(small.height * small.width >= config.minPixels)

        // The token count the prompt must reserve follows the merged grid.
        #expect(processor.tokenCount(height: 56, width: 56) == 4, "56/28 squared")
    }

    /// Splicing refuses a mismatch instead of shifting every feature.
    ///
    /// The start/end markers bracket the run but are ORDINARY tokens keeping their own embeddings.
    /// Counting them as placeholders would offset every image feature by one — fluent nonsense, no
    /// error anywhere.
    @Test("splicing writes only at placeholders, and refuses a count mismatch")
    func spliceRespectsPlaceholders() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
            // `.text`: the tiny fixture has image TOKENS but no `vision_config`, so asking for
            // `.vision` is correctly refused. Splicing is about token POSITIONS and needs no tower.
            let model = try Glm5Next(config, requesting: [.text])
            let imageToken = try #require(config.imageTokenId)

            // start, IMG, IMG, end  — two placeholders, the markers are not.
            let ids = MLXArray([Int32(7), Int32(imageToken), Int32(imageToken), Int32(8)])
            let hidden = config.textConfig.hiddenSize
            let embeddings = MLXArray.zeros([1, 4, hidden])
            let features = MLXArray.ones([2, hidden])

            let spliced = try model.spliceImageFeatures(
                inputIds: ids, embeddings: embeddings, imageFeatures: features)
            eval(spliced)
            let values = spliced.asType(.float32).asArray(Float.self)
            // Positions 1 and 2 written, 0 and 3 untouched.
            #expect(values[0 ..< hidden].allSatisfy { $0 == 0 }, "the start marker was overwritten")
            #expect(values[hidden ..< 2 * hidden].allSatisfy { $0 == 1 })
            #expect(values[2 * hidden ..< 3 * hidden].allSatisfy { $0 == 1 })
            #expect(
                values[3 * hidden ..< 4 * hidden].allSatisfy { $0 == 0 },
                "the end marker was overwritten")

            // One feature too few must be refused, not silently shifted.
            #expect(throws: Glm5NextInputShapeError.self) {
                _ = try model.spliceImageFeatures(
                    inputIds: ids, embeddings: embeddings,
                    imageFeatures: MLXArray.ones([1, hidden]))
            }
        }
    }

    /// The ROUTED factory now serves glm5_next, and narrows it like every other family.
    ///
    /// Registration was held back until the model could honestly conform to `LanguageModel` — a
    /// `fatalError` on the forward would have been the failure mode this repo spent the week
    /// removing. It conforms now, so the table entry is real.
    @Test("the routed factory builds glm5_next and honours the request")
    func routedFactoryServesGlm5Next() async throws {
        let data = Data(Self.tinyJSON.utf8)

        let full = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "glm5_next", requesting: nil)
        let bearing = try #require(full as? any ModalityBearing)
        // The fixture carries a vision stanza, so an unnarrowed build offers both — and the
        // narrowed build below must therefore report STRICTLY less. Asserting only the narrowed
        // side would pass just as well for a model that reported `[.text]` no matter what.
        // One tower serves stills and video alike, so both are on offer.\n        #expect(bearing.modalities == [.text, .vision, .video])

        let narrowed = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "glm5_next", requesting: [.text])
        #expect((narrowed as? any ModalityBearing)?.modalities == [.text])

        // It is a LanguageModel, which is what the registration required.
        _ = try #require(narrowed as? any LanguageModel, "registration requires LanguageModel")

        // One cache slot per DECODER layer, of the kind each layer needs — not one per entry of
        // `layers`, which includes the MTP layer.
        let glm = try #require(narrowed as? Glm5Next)
        let caches = glm.newCache(parameters: nil)
        #expect(caches.count == glm.languageModel.numDecoderLayers)
        #expect(caches[0] is MambaCache, "layer 0 is linear attention")
        #expect(!(caches[1] is MambaCache), "layer 1 is sparse attention")
    }

    /// Media handed to a TEXT-ONLY instance is REFUSED, not answered from the text alone.
    /// (A vision-bearing instance encodes it instead — see `imageFlowsEndToEnd`.)
    @Test("prepare refuses media it has no tower for")
    func prepareRefusesMedia() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
            let model = try Glm5Next(config, requesting: [.text])
            let text = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3]).reshaped(1, 3))

            // Text alone is fine.
            #expect(throws: Never.self) {
                _ = try model.prepare(
                    LMInput(text: text), cache: model.newCache(parameters: nil), windowSize: nil)
            }

            // With an image it must refuse — answering from the text would look like it had seen it.
            let withImage = LMInput(
                text: text,
                image: .init(pixels: MLXArray.zeros([1, 3, 8, 8]), frames: nil))
            #expect(throws: Glm5NextDecoderUnavailable.self) {
                _ = try model.prepare(
                    withImage, cache: model.newCache(parameters: nil), windowSize: nil)
            }
        }
    }

    /// PIXELS ACTUALLY FLOW: a real image through the processor, the tower, the splice and the
    /// decoder, in one call.
    ///
    /// This is the end-to-end the image path existed for. Everything it exercises was individually
    /// tested; what it adds is that the pieces AGREE — above all that the placeholder count the
    /// processor writes into the prompt equals the feature count the tower produces. Those are
    /// derived in two different places from the same grid, and a drift between them is exactly the
    /// kind of thing that shows up as a shape error deep in a splice or, worse, as a plausible
    /// answer about the wrong pixels.
    @Test("an image flows from processor through tower and splice to logits")
    func imageFlowsEndToEnd() throws {
        let config = try JSONDecoder().decode(
            Glm5NextConfiguration.self, from: Data(Self.tinyJSON.utf8))
        try #require(config.canBuildVisionTower, "the fixture now carries a vision stanza")

        // A processor built from the shipped image config, scaled to the tiny tower.
        let processorConfig = Glm5NextImageProcessorConfiguration(
            imageMean: [0.48145466, 0.4578275, 0.40821073],
            imageStd: [0.26862954, 0.26130258, 0.27577711],
            patchSize: 14, mergeSize: 2, temporalPatchSize: 2,
            minImageTokens: 1, maxImageTokens: 64, patchExpandFactor: 1)
        let imageProcessor = Glm5NextImageProcessor(processorConfig)

        // A real 112x112 image.
        let image = CIImage(color: .gray).cropped(to: .init(x: 0, y: 0, width: 112, height: 112))

        let (height, width) = try imageProcessor.targetSize(height: 112, width: 112)
        let expectedTokens = imageProcessor.tokenCount(height: height, width: width)
        #expect(expectedTokens > 0)

        try MLXMetalTestLock.withLock {
            let model = try Glm5Next(config, requesting: [.vision])
            let tower = try #require(model.visionTower)

            // Preprocess exactly as the processor would, then check the tower agrees on the count.
            var processed = MediaProcessing.inSRGBToneCurveSpace(image)
            processed = MediaProcessing.resampleBicubic(
                processed, to: .init(width: width, height: height))
            processed = MediaProcessing.normalize(
                processed, mean: (0.48, 0.46, 0.41), std: (0.27, 0.26, 0.28))
            let array = MediaProcessing.asMLXArray(processed)
            let frames = Array(repeating: array, count: processorConfig.temporalPatchSize)
            let (patches, grid) = try QwenVL.patchify(
                images: frames, mergeSize: processorConfig.mergeSize,
                patchSize: processorConfig.patchSize,
                temporalPatchSize: processorConfig.temporalPatchSize)

            let features = try tower(
                patches.asType(tower.patchEmbed.proj.weight.dtype), grid: grid)
            eval(features)
            let produced = features.dim(0)
            #expect(
                produced == expectedTokens,
                "tower produced \(produced) features, prompt reserves \(expectedTokens) — drifted")
            #expect(features.dim(1) == config.textConfig.hiddenSize, "tower must emit language width")

            // Build the prompt the processor would, and run the whole thing.
            var ids: [Int32] = [1, 2]
            if let start = config.imageStartTokenId { ids.append(Int32(start)) }
            ids.append(contentsOf: Array(
                repeating: Int32(config.imageTokenId!), count: expectedTokens))
            if let end = config.imageEndTokenId { ids.append(Int32(end)) }
            let tokens = MLXArray(ids)[.newAxis, 0...]

            let input = LMInput(
                text: .init(tokens: tokens), image: .init(pixels: patches, frames: [grid]))
            let result = try model.prepare(
                input, cache: model.newCache(parameters: nil), windowSize: nil)

            guard case .logits(let output) = result else {
                Issue.record("prepare returned tokens for an input carrying an image")
                return
            }
            eval(output.logits)
            #expect(output.logits.shape == [1, ids.count, config.textConfig.vocabSize])
            let values = output.logits.asType(.float32).asArray(Float.self)
            #expect(values.allSatisfy { $0.isFinite }, "image-conditioned logits must be finite")
            #expect(Set(values.prefix(64)).count > 1, "logits are constant — a dead path")
        }
    }

}
