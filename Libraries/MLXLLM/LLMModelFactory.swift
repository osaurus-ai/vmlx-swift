// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Creates a function that decodes configuration data and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(from:configuration:progressHandler:)``.
public enum LLMTypeRegistry {

    // Split into functions to help the compiler type-check the large model registry
    private static func coreModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "mistral": create(LlamaConfiguration.self, LlamaModel.init),
            "llama": create(LlamaConfiguration.self, LlamaModel.init),
            "phi": create(PhiConfiguration.self, PhiModel.init),
            "phi3": create(Phi3Configuration.self, Phi3Model.init),
            "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
            "gemma": create(GemmaConfiguration.self, GemmaModel.init),
            "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
            "gemma3": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3_text": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3n": create(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
            "gemma4": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            "gemma4_text": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            "gemma4_unified": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            "gemma4_unified_text": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            // Block-diffusion family — generation runs through
            // BlockDiffusionTokenIterator, never the AR TokenIterator.
            "diffusion_gemma": create(DiffusionGemmaConfiguration.self, DiffusionGemmaModel.init),
            "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
            "qwen3": create(Qwen3Configuration.self, Qwen3Model.init),
            "qwen3_moe": create(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
            "qwen3_next": create(Qwen3NextConfiguration.self, Qwen3NextModel.init),
            "qwen3_5": create(Qwen35Configuration.self, Qwen35Model.init),
            "qwen3_5_moe": { data in
                // Peek at weight_format — "mxtq" routes to the JANGTQ variant,
                // which swaps the routed-expert SwitchGLU for TurboQuantSwitchGLU
                // so the codebook Metal kernels run instead of gather_qmm.
                // Same model_type is reused by Qwen 3.6.
                struct FormatCheck: Codable {
                    let weightFormat: String?
                    enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
                }
                if let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: data),
                    check.weightFormat == "mxtq"
                {
                    let config = try JSONDecoder.json5().decode(
                        Qwen35JANGTQConfiguration.self, from: data)
                    return Qwen35JANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
                return Qwen35MoEModel(config)
            },
            "qwen3_5_text": create(Qwen35TextConfiguration.self, Qwen35TextModel.init),
        ]
    }

    private static func extendedModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "mistral4": create(Mistral4Configuration.self, Mistral4Model.init),
            "minicpm": create(MiniCPMConfiguration.self, MiniCPMModel.init),
            "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
            "cohere": create(CohereConfiguration.self, CohereModel.init),
            "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
            "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
            // DeepSeek-V3 / Kimi K2.6 (kimi_k25, kimi_k2) — all share
            // the same MLA + MoE text backbone. Factory peeks
            // `weight_format == "mxtq"` and routes to the JANGTQ
            // variant (routed experts via TurboQuantSwitchGLU + codebook
            // Metal kernels) when present; standard affine / fp8
            // bundles continue to use `DeepseekV3Model`.
            //
            // Coverage:
            //   - model_type = deepseek_v3  : DeepSeek-V3 upstream + JANGTQ
            //   - model_type = kimi_k25     : Kimi K2.6 REAP-30/50 + JANGTQ
            //   - model_type = kimi_k2      : pre-K2.6 Kimi naming
            //
            // Reference:
            //   jang/research/KIMI-K2.6-VMLX-INTEGRATION.md §2 (Swift)
            //   jang/research/KIMI-K2.6-IMPLEMENTATION.md §4.1 (MLA)
            //
            // `DeepseekV3Attention` materializes K/V before SDPA, but it still
            // needs the MLA L==1 fp32 SDPA cast during decode. The helper also
            // avoids a second cache update after MLA has already appended K/V.
            "deepseek_v3": dispatchDeepseekV3Family,
            "kimi_k25": dispatchDeepseekV3Family,
            "kimi_k2": dispatchDeepseekV3Family,
            "granite": create(GraniteConfiguration.self, GraniteModel.init),
            "granitemoehybrid": create(
                GraniteMoeHybridConfiguration.self, GraniteMoeHybridModel.init),
            "mimo": create(MiMoConfiguration.self, MiMoModel.init),
            "mimo_v2": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
            "mimo_v2_flash": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
            "minimax": create(MiniMaxConfiguration.self, MiniMaxModel.init),
            // MiniMax-M3 (MSA / Lightning-Indexer). The text decoder is shared by
            // the plain and VL outer types; text-only sanitize drops vision keys.
            "minimax_m3": create(MiniMaxM3Configuration.self, MiniMaxM3Model.init),
            "minimax_m3_vl": create(MiniMaxM3Configuration.self, MiniMaxM3Model.init),
            "minimax_m2": { data in
                // Peek at weight_format — "mxtq" routes to the JANGTQ variant,
                // which swaps the MoE SwitchGLU for TurboQuantSwitchGLU so the
                // codebook Metal kernels run instead of gather_qmm. Attention
                // path is unchanged.
                struct FormatCheck: Codable {
                    let weightFormat: String?
                    enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
                }
                if let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: data),
                    check.weightFormat == "mxtq"
                {
                    let config = try JSONDecoder.json5().decode(
                        MiniMaxJANGTQConfiguration.self, from: data)
                    return MiniMaxJANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data)
                return MiniMaxModel(config)
            },
            "glm4": create(GLM4Configuration.self, GLM4Model.init),
            "glm4_moe": create(GLM4MoEConfiguration.self, GLM4MoEModel.init),
            "glm4_moe_lite": create(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
            "acereason": create(Qwen2Configuration.self, Qwen2Model.init),
            "falcon_h1": create(FalconH1Configuration.self, FalconH1Model.init),
            "bitnet": create(BitnetConfiguration.self, BitnetModel.init),
            "smollm3": create(SmolLM3Configuration.self, SmolLM3Model.init),
            "ernie4_5": create(Ernie45Configuration.self, Ernie45Model.init),
            "lfm2": create(LFM2Configuration.self, LFM2Model.init),
            // Laguna (Poolside agentic-coding 33B/3B-active MoE). 40-layer
            // hybrid SWA + full attention with per-layer head count
            // (48 full / 64 SWA), dual RoPE, q_norm/k_norm, sigmoid +
            // bias-corrected routing over 256 routed experts top-8 + 1
            // shared. See `Models/Laguna.swift`. JANGTQ variant (paired
            // JANGTQDenseLinear port) follows the Mistral 3 family
            // pattern in a follow-up.
            "laguna": { data in
                // 2026-05-01: Real Laguna mxtq bundles are MIXED-QUANT.
                //
                //   - Dense paths (attention Q/K/V/O, layer-0 dense MLP
                //     gate/up/down, shared expert, router gate) ship as
                //     MLX standard affine quant — `weight` packed uint32 +
                //     `scales` + `biases` of shape [out, in/group_size].
                //     Vanilla `Linear` works because the MLX loader auto-
                //     substitutes `QuantizedLinear` based on the
                //     top-level `quantization: { bits, group_size }`
                //     field in `config.json`.
                //
                //   - Routed MoE experts ship as JANGTQ codebook —
                //     stacked tensors at `layers.<i>.mlp.experts.{gate_up_proj,
                //     down_proj}.{tq_packed,tq_norms}`, with gate and up
                //     FUSED on the out-dim axis. `LagunaModel` wires
                //     these through `TurboQuantSwitchGLU` (same
                //     primitive DSV4 / Mistral 4 / NemotronH JANGTQ use)
                //     and the model's `sanitize` splits the fused tensor
                //     at load time.
                //
                // 2026-05-02 update: also support **mxfp4** Laguna bundles.
                // Those ship the routed experts as MLX standard affine
                // quant (`weight` + `scales` + `biases` per gate_up_proj /
                // down_proj — gate_up still FUSED on the out-dim axis).
                // Detect by `weight_format != "mxtq"` AND no `mxtq_bits`
                // top-level key; route to the affine `SwitchGLU` MoE path
                // (jangtq=nil) instead of the codebook
                // `TurboQuantSwitchGLU` path. Sanitize splits the fused
                // gate_up_proj for both formats.
                struct LagunaProbe: Codable {
                    let weightFormat: String?
                    let mxtqBits: Int?
                    let mxtqSeed: Int?
                    enum CodingKeys: String, CodingKey {
                        case weightFormat = "weight_format"
                        case mxtqBits = "mxtq_bits"
                        case mxtqSeed = "mxtq_seed"
                    }
                }
                let probe = try? JSONDecoder.json5().decode(LagunaProbe.self, from: data)
                let cfg = try JSONDecoder.json5().decode(
                    LagunaConfiguration.self, from: data)
                // Laguna-M.1 (Mistral lineage): per-element gating, SEPARATE
                // q/k/v/o_proj, per-module affine quant, all-full-attention.
                // Architecturally distinct from the poolside XS.2 fused-qkv /
                // JANGTQ path below — route to the dedicated affine engine.
                if cfg.gateMode == .perElement {
                    return LagunaM1Model(cfg)
                }
                let isMxtq =
                    probe?.weightFormat?.lowercased() == "mxtq"
                    || probe?.mxtqBits != nil
                let jangtqContext: LagunaMoEContext?
                if isMxtq {
                    jangtqContext = LagunaMoEContext(
                        bits: probe?.mxtqBits ?? 2,
                        mxtqSeed: probe?.mxtqSeed ?? 42)
                } else {
                    // mxfp4 / affine bundle: nil context → SwitchGLU path.
                    jangtqContext = nil
                }
                return LagunaModel(cfg, jangtq: jangtqContext)
            },
        ]
    }

    private static func additionalModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "baichuan_m1": create(BaichuanM1Configuration.self, BaichuanM1Model.init),
            "exaone4": create(Exaone4Configuration.self, Exaone4Model.init),
            "gpt_oss": create(GPTOSSConfiguration.self, GPTOSSModel.init),
            "lille-130m": create(Lille130mConfiguration.self, Lille130mModel.init),
            "olmoe": create(OlmoEConfiguration.self, OlmoEModel.init),
            "olmo2": create(Olmo2Configuration.self, Olmo2Model.init),
            "olmo3": create(Olmo3Configuration.self, Olmo3Model.init),
            "bailing_moe": create(BailingMoeConfiguration.self, BailingMoeModel.init),
            "bailing_hybrid": create(BailingHybridConfiguration.self, BailingHybridModel.init),
            "bailing_moe_v2_5": create(BailingHybridConfiguration.self, BailingHybridModel.init),
            "lfm2_moe": create(LFM2MoEConfiguration.self, LFM2MoEModel.init),
            "step": dispatchStep3p5,
            "step3p5": dispatchStep3p5,
            "step3p7": dispatchStep3p5,
            "nanochat": create(NanoChatConfiguration.self, NanoChatModel.init),
            "nemotron_h": dispatchNemotronH,
            "afmoe": create(AfMoEConfiguration.self, AfMoEModel.init),
            "jamba_3b": create(JambaConfiguration.self, JambaModel.init),
            "mistral3": dispatchMistral3LLM,
            // `ministral3` is the inner text_config.model_type for Mistral 3.5
            // VLM bundles, but text-only Mistral 3.5 LLM bundles can also expose
            // it at the OUTER level. Register both keys to the same dispatch so
            // either shape resolves correctly. The vision_config gate inside
            // the closure still routes VLM bundles to VLMModelFactory if a user
            // configured a Mistral 3.5 VLM bundle to expose `ministral3` outer.
            "ministral3": dispatchMistral3LLM,
            "apertus": create(ApertusConfiguration.self, ApertusModel.init),
            "zaya": dispatchZaya,
            // DSV4 (DeepSeek-V4-Flash / -Pro): architecturally distinct
            // from DSV3 — mHC residual stream, CSA/HCA hybrid attention,
            // sqrtsoftplus gate, grouped low-rank O, sliding window,
            // hash routing on layers 0-2. Would PRODUCE GARBAGE if routed
            // through DeepseekV3Model. Throw a clear error pointing at
            // the port plan instead of silently dispatching to DSV3.
            //
            // Model weights for DSV4 are also FP4 + FP8 mixed — the
            // standard safetensors loader doesn't know how to dequant
            // them, so even a DSV3-shaped fallback would fail at load
            // time with a less-useful error.
            //
            // Swift port plan + status:
            //   Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md
            "deepseek_v4": dispatchDeepseekV4,
            "hy_v3": create(Hy3Configuration.self, Hy3Model.init),
            "hy3":   create(Hy3Configuration.self, Hy3Model.init),
            "hy-v3": create(Hy3Configuration.self, Hy3Model.init),
        ]
    }

    /// 2026-05-06 ZAYA1 CCA-attention port. Uses the context pattern
    /// (à la NemotronH/Laguna): a single `ZayaModel` class drives all
    /// four bundle variants — JANGTQ2/JANGTQ4 (codebook-quantized routed
    /// experts), MXFP4 (affine routed experts), and the BF16 base model.
    /// CCA state (`conv_state`, `prev_hs`) round-trips via `ZayaCCACache`
    /// (LayerKind.zayaCCA) and `BatchZayaCCACache`; the BatchEngine
    /// auto-detects ZAYA caches and flips paged-incompatible+hybrid.
    private static func dispatchZaya(data: Data) throws -> any LanguageModel {
        struct Probe: Codable {
            let weightFormat: String?
            let mxtqBits: ProbeBits?
            let mxtqGateUpBits: Int?
            let mxtqDownBits: Int?
            let mxtqSeed: Int?
            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
                case mxtqBits = "mxtq_bits"
                case mxtqGateUpBits = "mxtq_gate_up_bits"
                case mxtqDownBits = "mxtq_down_bits"
                case mxtqSeed = "mxtq_seed"
            }
        }
        // mxtq_bits in the bundle may be a flat int, a per-role dict
        // (e.g. `{routed_expert: 2, attention: 8, ...}`), or a JANGTQ_K
        // per-projection dict. Accept all real metadata shapes, but reject
        // gate/up width disagreement because the fused gate+up kernel cannot
        // dispatch mixed widths.
        enum ProbeBits: Codable {
            case flat(Int)
            case roles([String: Int])
            case projections(gateUp: Int?, down: Int?)
            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let v = try? c.decode(Int.self) {
                    self = .flat(v); return
                }
                if let v = try? c.decode([String: Int].self) {
                    self = .roles(v); return
                }
                struct ProjectionBits: Decodable {
                    let gateProj: Int?
                    let upProj: Int?
                    let downProj: Int?
                    enum CodingKeys: String, CodingKey {
                        case gateProj = "gate_proj"
                        case upProj = "up_proj"
                        case downProj = "down_proj"
                    }
                }
                struct NestedBits: Decodable {
                    let routedExpert: ProjectionBits?
                    enum CodingKeys: String, CodingKey { case routedExpert = "routed_expert" }
                }
                if let v = try? c.decode(NestedBits.self),
                    let routedExpert = v.routedExpert
                {
                    let gate = routedExpert.gateProj
                    let up = routedExpert.upProj
                    if let gate, let up, gate != up {
                        throw DecodingError.dataCorruptedError(
                            in: c,
                            debugDescription:
                                "mxtq_bits.routed_expert gate_proj and up_proj must match for fused gate/up kernels")
                    }
                    self = .projections(
                        gateUp: gate ?? up,
                        down: routedExpert.downProj)
                    return
                }
                throw DecodingError.typeMismatch(
                    ProbeBits.self,
                    .init(codingPath: decoder.codingPath,
                          debugDescription: "mxtq_bits must be Int, [String: Int], or routed projection metadata"))
            }
        }
        let config = try JSONDecoder.json5().decode(ZayaConfiguration.self, from: data)
        let probe = try? JSONDecoder.json5().decode(Probe.self, from: data)

        // Resolve routed-expert bit width from the probe, accepting both shapes.
        let routedBits: Int? = {
            switch probe?.mxtqBits {
            case .flat(let v): return v
            case .roles(let dict): return dict["routed_expert"] ?? dict.values.first
            case .projections(let gateUp, let down): return gateUp ?? down
            case nil: return nil
            }
        }()
        let projectedBits: (gateUp: Int?, down: Int?) = {
            switch probe?.mxtqBits {
            case .projections(let gateUp, let down): return (gateUp, down)
            default: return (nil, nil)
            }
        }()

        let weightFormat = (probe?.weightFormat ?? config.textConfig.weightFormat)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let isJANGTQ =
            weightFormat == "mxtq"
            || weightFormat == "jangtq2"
            || weightFormat == "jangtq4"
            || routedBits != nil

        let context: ZayaMoEContext?
        if isJANGTQ {
            let gateUpBits = probe?.mxtqGateUpBits ?? projectedBits.gateUp ?? routedBits ?? 2
            let downBits = probe?.mxtqDownBits ?? projectedBits.down ?? routedBits ?? 2
            let seed = probe?.mxtqSeed ?? 42
            context = .jangtq(gateUpBits: gateUpBits, downBits: downBits, seed: seed)
        } else {
            context = nil
        }
        return ZayaModel(config, moe: context)
    }

    private static func dispatchNemotronH(data: Data) throws -> any LanguageModel {
        enum ProbeBits: Decodable {
            case int(Int)
            case routed(up: Int?, down: Int?)

            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let value = try? container.decode(Int.self) {
                    self = .int(value)
                    return
                }
                let keyed = try decoder.container(keyedBy: CodingKeys.self)
                if let routed = try keyed.decodeIfPresent(
                    RoutedExpert.self, forKey: .routedExpert)
                {
                    self = .routed(up: routed.upProj ?? routed.gateProj, down: routed.downProj)
                } else {
                    throw DecodingError.dataCorruptedError(
                        in: container,
                        debugDescription: "mxtq_bits must be an Int or contain routed_expert projection bits")
                }
            }

            var uniformBits: Int? {
                switch self {
                case .int(let value):
                    return value
                case .routed(let up, let down):
                    if let up, let down, up != down {
                        return nil
                    }
                    return up ?? down
                }
            }

            private enum CodingKeys: String, CodingKey {
                case routedExpert = "routed_expert"
            }

            private struct RoutedExpert: Decodable {
                let gateProj: Int?
                let upProj: Int?
                let downProj: Int?

                enum CodingKeys: String, CodingKey {
                    case gateProj = "gate_proj"
                    case upProj = "up_proj"
                    case downProj = "down_proj"
                }
            }
        }

        struct Probe: Decodable {
            let weightFormat: String?
            let format: String?
            let mxtqBits: ProbeBits?
            let routedExpertBits: Int?
            let mxtqSeed: Int?

            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
                case format
                case mxtqBits = "mxtq_bits"
                case routedExpertBits = "routed_expert_bits"
                case mxtqSeed = "mxtq_seed"
            }
        }

        let config = try JSONDecoder.json5().decode(NemotronHConfiguration.self, from: data)
        let probe = try? JSONDecoder.json5().decode(Probe.self, from: data)
        let weightFormat = probe?.weightFormat?.lowercased()
        let jangFormat = probe?.format?.lowercased()
        let mxtqBits = probe?.mxtqBits?.uniformBits
        let isJANGTQ =
            weightFormat == "mxtq"
            || jangFormat?.contains("jangtq") == true
            || weightFormat == "jangtq2"
            || weightFormat == "jangtq4"
            || mxtqBits != nil
            || probe?.routedExpertBits != nil

        guard isJANGTQ else {
            return NemotronHModel(config)
        }

        guard let bits = mxtqBits ?? probe?.routedExpertBits else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "NemotronH JANGTQ requires uniform routed expert bits for fc1/fc2"))
        }

        return NemotronHModel(
            jangtqContext: NemotronHJANGTQContext(
                bits: bits,
                mxtqSeed: probe?.mxtqSeed ?? 42),
            configuration: config)
    }

    /// Dispatcher for the DeepSeek-V3 family (deepseek_v3, kimi_k25,
    /// kimi_k2). Peeks `weight_format` in config.json — `"mxtq"`
    /// routes to `DeepseekV3JANGTQModel` (TurboQuantSwitchGLU for
    /// routed experts + Metal codebook kernels); every other value
    /// routes to the standard `DeepseekV3Model`.
    ///
    /// Keeps as a top-level helper (not an inline closure) so the
    /// three model_type entries in `extendedModels()` share one code
    /// path. Any future DeepSeek-V3-family alias just adds a dict
    /// entry pointing here.
    /// DSV4 dispatch placeholder. Throws a structured error pointing
    /// at the port plan until `DeepseekV4.swift` /
    /// `DeepseekV4JANGTQ.swift` land. Keeping the registration means:
    ///
    ///   - osaurus gets a CLEAR error saying DSV4 isn't supported yet
    ///     instead of a cryptic "bundle silently produced garbage";
    ///   - model_type reasoning/tool plumbing (which already handles
    ///     `deepseek*` prefix correctly) keeps working — just the
    ///     forward pass is gated;
    ///   - test harness can exercise the `kimi_k25`/`deepseek_v3`
    ///     paths without triggering DSV4 errors.
    ///
    /// Follow `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md` to finish.
    private static func dispatchDeepseekV4(data: Data) throws -> any LanguageModel {
        // DeepseekV4 (JANGTQ + JANG family). The right variant depends
        // on whether routed experts are stored as MXTQ codebook
        // (TurboQuantSwitchGLU) or plain affine (SwitchGLU).
        //
        // Detection priority:
        //   1. `weight_format: "mxtq"` in config.json — authoritative
        //      when present (jang_config.json typically carries this
        //      but some bundles stamp it on config.json instead).
        //   2. `DSV4_FORCE_JANGTQ=1` env override — for bundles with
        //      mislabeled jang_config (we've seen "bf16" stamped on
        //      JANGTQ bundles in the wild). Sets the JANGTQ path.
        //   3. Heuristic: DSV4 + `quantization.bits in {2, 4}` AND
        //      `quantization.group_size == 32` AND no overriding
        //      affine signal → JANGTQ. Reflects research §5: the
        //      only DSV4-Flash distributions are JANGTQ_2L/4 and
        //      JANG_2L/4. Both quant ladders. JANG_2L/JANG4 use
        //      uniform affine (no `tq_packed` keys) — but they're
        //      experimental per the bundle cheat-sheet, not the
        //      primary production target.
        //   4. Fallback: affine `DeepseekV4Model`.
        struct FormatCheck: Codable {
            let weightFormat: String?
            let quantization: QuantInfo?
            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
                case quantization
            }
        }
        struct QuantInfo: Codable {
            let bits: Int?
            let groupSize: Int?
            enum CodingKeys: String, CodingKey {
                case bits
                case groupSize = "group_size"
            }
        }

        let config = try JSONDecoder.json5().decode(
            DeepseekV4Configuration.self, from: data)
        let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: data)
        let forced = ProcessInfo.processInfo.environment["DSV4_FORCE_JANGTQ"] == "1"
        let weightFormat = check?.weightFormat?.lowercased()
        let isMxtqStamp =
            weightFormat == "mxtq" || weightFormat == "jangtq2"
            || weightFormat == "jangtq4"

        // 2026-04-26: bundles with mislabeled `weight_format: "bf16"`
        // but real JANGTQ codebook tensors are auto-corrected by
        // `_load`'s merge step BEFORE this dispatcher fires — when a
        // valid `jangtq_runtime.safetensors` sidecar is present, the
        // merge forces `weight_format = "mxtq"`. By the time we
        // reach this point the stamp is authoritative, so the
        // existing `isMxtqStamp` check is sufficient.
        if isMxtqStamp || forced {
            // mxtqBits sourcing — the routed-MoE codebook lives in
            // `jangtq_runtime.safetensors` keyed `codebook.{inFeatures}.
            // {bits}`. The bits THERE are authoritative — the
            // `config.json` `quantization.bits` field describes the
            // AFFINE non-routed block (often 8 for JANGTQ_2L) and
            // doesn't match the codebook bits. Mismatch caused
            // `TurboQuantSwitchLinear.forward` to fatalError("sidecar
            // not loaded") when bits=8 was searched against a bits=2
            // codebook (2026-04-25 reproducer on DSV4-Flash JANGTQ
            // bundle whose config.json was regenerated with bits=8).
            //
            // Resolution priority:
            //   1. `DSV4_JANGTQ_BITS` env override (4 for JANGTQ4
            //      bundles, 2 for JANGTQ_2L).
            //   2. Authoritative `weight_format` stamp:
            //      `jangtq4` → 4, `jangtq2`/`mxtq` → 2.
            //   3. Forced path (no stamp, env-only): default to 2 (the
            //      canonical JANGTQ_2L distribution).
            //   4. Heuristic — config.json bits, only when in {2, 4}.
            //      Anything else is the affine non-routed bits and
            //      doesn't match the codebook.
            let env = ProcessInfo.processInfo.environment
            let envBits = (env["DSV4_JANGTQ_BITS"]).flatMap { Int($0) }
            let stampBits: Int? = {
                switch weightFormat {
                case "jangtq4": return 4
                case "jangtq2", "mxtq": return 2
                default: return nil
                }
            }()
            // Codebook bit-widths the runtime kernels currently support.
            // Centralized so all bit filters in this routing block stay
            // in sync. Extend ONLY when the corresponding converter
            // packing convention is verified end-to-end; the Metal
            // kernel's `vals_per_u32 = 32 / bits` math is generic but
            // the converter side has to match (e.g., JANGTQ3 / 3-bit
            // packing is NOT shipped — leave it out).
            let validRoutedBits: Set<Int> = [2, 4]
            let configBits: Int? = {
                guard let b = check?.quantization?.bits, validRoutedBits.contains(b)
                else { return nil }
                return b
            }()
            // `routed_expert_bits` field — populated by `_load`'s
            // resolution chain (sidecar codebook sniff / profile / etc.)
            // when the bundle's config didn't ship it directly. Read
            // here so we prefer it over `configBits` (which can be the
            // affine non-routed bits = 8, useless for routed-MoE).
            struct RoutedBitsCheck: Codable {
                let routedExpertBits: Int?
                enum CodingKeys: String, CodingKey {
                    case routedExpertBits = "routed_expert_bits"
                }
            }
            let routedBits: Int? = {
                guard let r = (try? JSONDecoder.json5().decode(
                    RoutedBitsCheck.self, from: data))?.routedExpertBits,
                    validRoutedBits.contains(r)
                else { return nil }
                return r
            }()
            let hasLayerBitPlan = !config.routedExpertLayerBits.isEmpty
            let uniformBits =
                envBits
                ?? (hasLayerBitPlan ? nil : (stampBits ?? routedBits ?? configBits))
            return DeepseekV4JANGTQModel(config, mxtqBits: uniformBits, mxtqSeed: 42)
        }
        return DeepseekV4Model(config)
    }

    /// Shared dispatch for `mistral3` and `ministral3` outer model_types.
    /// Mistral 3 / 3.5 LLM-only bundles can expose either spelling at the
    /// outer level. Both flow through the same logic:
    ///
    ///   1. If `vision_config` is present → throw `unsupportedModelType`
    ///      so the factory iterator falls through to VLMModelFactory.
    ///   2. If inner `text_config.model_type == "mistral4"` → Mistral4Model
    ///      (Mistral 3 wrapper around a Mistral 4 text decoder).
    ///   3. If `weight_format == "mxtq"` → Mistral3TextJANGTQModel.
    ///   4. Otherwise → vanilla Mistral3TextModel.
    static func dispatchMistral3LLM(data: Data) throws -> any LanguageModel {
        // 1. Vision gate — VLM bundles must not be loaded under this LLM route.
        struct VisionGate: Codable {
            let visionConfig: AnyDecodable?
            enum CodingKeys: String, CodingKey { case visionConfig = "vision_config" }
        }
        struct AnyDecodable: Codable {}
        if let gate = try? JSONDecoder.json5().decode(VisionGate.self, from: data),
            gate.visionConfig != nil
        {
            throw ModelFactoryError.unsupportedModelType(
                "mistral3/ministral3 (has vision_config — route via VLMModelFactory)")
        }

        // 2. Mistral3 / Ministral3 wrapper around Mistral4 text decoder.
        struct TextConfigCheck: Codable {
            let textConfig: TextModelType?
            struct TextModelType: Codable {
                let modelType: String?
                enum CodingKeys: String, CodingKey { case modelType = "model_type" }
            }
            enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
        }
        if let check = try? JSONDecoder.json5().decode(TextConfigCheck.self, from: data),
            check.textConfig?.modelType == "mistral4"
        {
            let config = try JSONDecoder.json5().decode(Mistral4Configuration.self, from: data)
            return Mistral4Model(config)
        }

        // 3. JANGTQ dispatch — `weight_format == "mxtq"` routes to
        //    Mistral3TextJANGTQModel which uses JANGTQDenseLinear for
        //    attention Q/K/V/O + MLP gate/up/down (consuming `.tq_packed` +
        //    `.tq_norms` safetensors instead of a flat `.weight`).
        struct JANGTQProbe: Codable {
            let weightFormat: String?
            let mxtqBits: Int?
            let mxtqSeed: Int?
            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
                case mxtqBits = "mxtq_bits"
                case mxtqSeed = "mxtq_seed"
            }
        }
        if let probe = try? JSONDecoder.json5().decode(JANGTQProbe.self, from: data),
            probe.weightFormat?.lowercased() == "mxtq"
        {
            let config = try JSONDecoder.json5().decode(
                Mistral3TextConfiguration.self, from: data)
            return Mistral3TextJANGTQModel(
                config,
                bits: probe.mxtqBits ?? 2,
                seed: probe.mxtqSeed ?? 42
            )
        }

        // 4. Vanilla.
        let config = try JSONDecoder.json5().decode(Mistral3TextConfiguration.self, from: data)
        return Mistral3TextModel(config)
    }

    private static func dispatchDeepseekV3Family(data: Data) throws -> any LanguageModel {
        struct FormatCheck: Codable {
            let weightFormat: String?
            enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
        }

        // Kimi-K2.6 (model_type=kimi_k25) wraps the LLM config under
        // `text_config` (the bundle is multimodal-shaped even when no
        // vision tower is loaded by this LLM-only path). Plain DeepSeek
        // V3 bundles have the LLM config at the top level. Detect by
        // looking for a `text_config` object and re-encoding it as the
        // primary payload before the existing decoders run — mirrors
        // Mistral3VLM's text_config unwrap pattern.
        let effectiveData: Data
        do {
            if let json = try JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed]) as? [String: Any],
                let textConfig = json["text_config"] as? [String: Any]
            {
                // Carry weight_format / mxtq_bits / mxtq_seed forward
                // from the top-level into the unwrapped text_config so
                // dispatchers downstream still see them.
                var merged = textConfig
                for k in ["weight_format", "mxtq_bits", "mxtq_seed",
                          "model_type", "quantization"]
                {
                    if merged[k] == nil, let v = json[k] {
                        merged[k] = v
                    }
                }
                effectiveData = try JSONSerialization.data(
                    withJSONObject: merged, options: [])
            } else {
                effectiveData = data
            }
        } catch {
            effectiveData = data
        }

        if let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: effectiveData),
            check.weightFormat == "mxtq"
        {
            let config = try JSONDecoder.json5().decode(
                DeepseekV3JANGTQConfiguration.self, from: effectiveData)
            return DeepseekV3JANGTQModel(config)
        }
        let config = try JSONDecoder.json5().decode(
            DeepseekV3Configuration.self, from: effectiveData)
        return DeepseekV3Model(config)
    }

    private static func dispatchStep3p5(data: Data) throws -> any LanguageModel {
        struct Probe: Codable {
            let weightFormat: String?
            let mxtqBits: Int?
            let mxtqGateUpBits: Int?
            let mxtqDownBits: Int?
            let mxtqSeed: Int?
            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
                case mxtqBits = "mxtq_bits"
                case mxtqGateUpBits = "mxtq_gate_up_bits"
                case mxtqDownBits = "mxtq_down_bits"
                case mxtqSeed = "mxtq_seed"
            }
        }
        let probe = try? JSONDecoder.json5().decode(Probe.self, from: data)
        let config = try JSONDecoder.json5().decode(Step3p5Configuration.self, from: data)
        let isMxtq =
            probe?.weightFormat?.lowercased() == "mxtq"
            || probe?.mxtqBits != nil
            || probe?.mxtqGateUpBits != nil
            || probe?.mxtqDownBits != nil
        let jangtq: Step3p5JANGTQContext?
        if isMxtq {
            let gateUpBits = probe?.mxtqGateUpBits ?? probe?.mxtqBits ?? 2
            let downBits = probe?.mxtqDownBits ?? probe?.mxtqBits ?? gateUpBits
            jangtq = Step3p5JANGTQContext(
                gateUpBits: gateUpBits,
                downBits: downBits,
                seed: probe?.mxtqSeed ?? 42)
        } else {
            jangtq = nil
        }
        return Step3p5Model(config, jangtq: jangtq)
    }

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(
        creators: coreModels().merging(extendedModels()) { a, _ in a }
            .merging(additionalModels()) { a, _ in a }
    )
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The Python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = LLMRegistry(modelConfigurations: all())

    static public let smolLM_135M_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    static public let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    static public let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    static public let deepSeekR1_7B_4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    static public let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    static public let phi3_5_4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let phi3_5MoE = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_9b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_2b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma3_1B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-1b-it-qat-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E4B_it_lm_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E2B_it_lm_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E4B_it_lm_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E2B_it_lm_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        defaultPrompt: "why is the sky blue?"
    )

    static public let qwen2_5_7b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen2_5_1_5b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_0_6b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-0.6B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_1_7b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-1.7B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_4b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-4B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_8b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-8B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3MoE_30b_a3b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-30B-A3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    static public let llama3_1_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_1B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_3B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let deepseek_r1_4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let granite3_3_2b_4bit = ModelConfiguration(
        id: "mlx-community/granite-3.3-2b-instruct-4bit",
        defaultPrompt: ""
    )

    static public let mimo_7b_sft_4bit = ModelConfiguration(
        id: "mlx-community/MiMo-7B-SFT-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let glm4_9b_4bit = ModelConfiguration(
        id: "mlx-community/GLM-4-9B-0414-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .glm4
    )

    static public let acereason_7b_4bit = ModelConfiguration(
        id: "mlx-community/AceReason-Nemotron-7B-4bit",
        defaultPrompt: ""
    )

    static public let bitnet_b1_58_2b_4t_4bit = ModelConfiguration(
        id: "mlx-community/bitnet-b1.58-2B-4T-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let baichuan_m1_14b_instruct_4bit = ModelConfiguration(
        id: "mlx-community/Baichuan-M1-14B-Instruct-4bit-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let smollm3_3b_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM3-3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let ernie_45_0_3BPT_bf16_ft = ModelConfiguration(
        id: "mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let lfm2_1_2b_4bit = ModelConfiguration(
        id: "mlx-community/LFM2-1.2B-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .lfm2
    )

    static public let exaone_4_0_1_2b_4bit = ModelConfiguration(
        id: "mlx-community/exaone-4.0-1.2b-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let lille_130m_bf16 = ModelConfiguration(
        id: "mlx-community/lille-130m-instruct-bf16",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let olmoe_1b_7b_0125_instruct_4bit = ModelConfiguration(
        id: "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let olmo_2_1124_7B_Instruct_4bit = ModelConfiguration(
        id: "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let ling_mini_2_2bit = ModelConfiguration(
        id: "mlx-community/Ling-mini-2.0-2bit-DWQ",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let granite_4_0_h_tiny_4bit_dwq = ModelConfiguration(
        id: "mlx-community/Granite-4.0-H-Tiny-4bit-DWQ",
        defaultPrompt: ""
    )

    static public let lfm2_8b_a1b_3bit_mlx = ModelConfiguration(
        id: "mlx-community/LFM2-8B-A1B-3bit-MLX",
        defaultPrompt: "",
        toolCallFormat: .lfm2
    )

    static public let nanochat_d20_mlx = ModelConfiguration(
        id: "dnakov/nanochat-d20-mlx",
        defaultPrompt: ""
    )

    static public let gpt_oss_20b_MXFP4_Q8 = ModelConfiguration(
        id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let jamba_3b = ModelConfiguration(
        id: "mlx-community/AI21-Jamba-Reasoning-3B-bf16",
        defaultPrompt: ""
    )

    static public let gemma4_27b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-27b-it-4bit",
        defaultPrompt: "Explain quantum computing briefly.",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma4_12b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-12b-it-4bit",
        defaultPrompt: "What is the meaning of life?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma4_27b_it_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-27b-it-qat-4bit",
        defaultPrompt: "Explain quantum computing briefly.",
        extraEOSTokens: ["<end_of_turn>"]
    )

    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
            deepSeekR1_7B_4bit,
            gemma2bQuantized,
            gemma_2_2b_it_4bit,
            gemma_2_9b_it_4bit,
            gemma3_1B_qat_4bit,
            gemma3n_E4B_it_lm_bf16,
            gemma3n_E2B_it_lm_bf16,
            gemma3n_E4B_it_lm_4bit,
            gemma3n_E2B_it_lm_4bit,
            granite3_3_2b_4bit,
            granite_4_0_h_tiny_4bit_dwq,
            llama3_1_8B_4bit,
            llama3_2_1B_4bit,
            llama3_2_3B_4bit,
            llama3_8B_4bit,
            mistral7B4bit,
            mistralNeMo4bit,
            openelm270m4bit,
            phi3_5MoE,
            phi3_5_4bit,
            phi4bit,
            qwen205b4bit,
            qwen2_5_7b,
            qwen2_5_1_5b,
            qwen3_0_6b_4bit,
            qwen3_1_7b_4bit,
            qwen3_4b_4bit,
            qwen3_8b_4bit,
            qwen3MoE_30b_a3b_4bit,
            smolLM_135M_4bit,
            deepseek_r1_4bit,
            mimo_7b_sft_4bit,
            glm4_9b_4bit,
            acereason_7b_4bit,
            bitnet_b1_58_2b_4t_4bit,
            smollm3_3b_4bit,
            ernie_45_0_3BPT_bf16_ft,
            lfm2_1_2b_4bit,
            baichuan_m1_14b_instruct_4bit,
            exaone_4_0_1_2b_4bit,
            lille_130m_bf16,
            olmoe_1b_7b_0125_instruct_4bit,
            olmo_2_1124_7B_Instruct_4bit,
            ling_mini_2_2bit,
            lfm2_8b_a1b_3bit_mlx,
            nanochat_d20_mlx,
            gpt_oss_20b_MXFP4_Q8,
            jamba_3b,
            gemma4_27b_it_4bit,
            gemma4_12b_it_4bit,
            gemma4_27b_it_qat_4bit,
        ]
    }

}

@available(*, deprecated, renamed: "LLMRegistry", message: "Please use LLMRegistry directly.")
public typealias ModelRegistry = LLMRegistry

private struct LLMUserInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let modelType: String?
    let messageGenerator: MessageGenerator
    let defaultAdditionalContext: [String: any Sendable]?

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        modelType: String?,
        messageGenerator: MessageGenerator,
        defaultAdditionalContext: [String: any Sendable]? = nil
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.modelType = modelType
        self.messageGenerator = messageGenerator
        self.defaultAdditionalContext = defaultAdditionalContext
    }

    func prepare(input: UserInput) throws -> LMInput {
        let additionalContext = mergedAdditionalContext(input.additionalContext)
        let bailingMessages = BailingThinkingTemplateContext.apply(
            to: messageGenerator.generate(from: input),
            modelType: modelType,
            additionalContext: additionalContext
        )
        var messages = NemotronToolChoiceTemplateContext.apply(
            to: bailingMessages,
            modelType: modelType,
            additionalContext: additionalContext
        )
        if shouldCompactGemma4RequiredToolHistory(additionalContext) {
            messages = compactGemma4RequiredToolHistory(messages)
        }
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: additionalContext)
            let cachePrefixTokenCounts = canonicalHistoryCacheBoundaries(
                messages: messages,
                tools: input.tools,
                additionalContext: additionalContext,
                promptTokens: promptTokens)

            return LMInput(
                tokens: MLXArray(promptTokens),
                tokenIds: promptTokens,
                cacheScopeSalt: cacheScopeSalt(from: additionalContext),
                cachePrefixTokenCounts: cachePrefixTokenCounts,
                toolSchemas: input.tools)
        } catch TokenizerError.missingChatTemplate {
            print(
                "No chat template was included or provided, so converting messages to simple text format. This is not optimal for model performance, so applications should provide a chat template if none is included with the model."
            )
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(
                tokens: MLXArray(promptTokens),
                tokenIds: promptTokens,
                cacheScopeSalt: cacheScopeSalt(from: additionalContext),
                toolSchemas: input.tools)
        }
    }

    private func mergedAdditionalContext(
        _ requestContext: [String: any Sendable]?
    ) -> [String: any Sendable]? {
        guard defaultAdditionalContext != nil || requestContext != nil else {
            return nil
        }
        var merged: [String: any Sendable] = defaultAdditionalContext ?? [:]
        if let requestContext {
            for (key, value) in requestContext {
                merged[key] = value
            }
        }
        return DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            merged.isEmpty ? nil : merged,
            modelType: modelType
        )
    }

    private func shouldCompactGemma4RequiredToolHistory(
        _ additionalContext: [String: any Sendable]?
    ) -> Bool {
        guard let modelType else { return false }
        let normalized = modelType
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        let compact = normalized.replacingOccurrences(of: "_", with: "")
        guard compact.hasPrefix("gemma4") else {
            return false
        }
        if (additionalContext?["tool_choice"] as? String) == "required" {
            return true
        }
        if let name = additionalContext?["tool_choice_name"] as? String,
           !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return true
        }
        return false
    }

    private func compactGemma4RequiredToolHistory(
        _ messages: [[String: any Sendable]]
    ) -> [[String: any Sendable]] {
        guard let latestUserIndex = messages.lastIndex(where: {
            let role = $0["role"] as? String
            return role == "user" || role == "developer"
        }) else {
            return messages
        }

        var compacted: [[String: any Sendable]] = []
        compacted.reserveCapacity(messages.count)
        for message in messages[..<latestUserIndex] {
            guard let role = message["role"] as? String else { continue }
            if role == "system" || role == "developer" {
                compacted.append(message)
            }
        }
        compacted.append(messages[latestUserIndex])
        if latestUserIndex + 1 < messages.endIndex {
            compacted.append(contentsOf: messages[(latestUserIndex + 1)...])
        }
        return compacted
    }

    private func canonicalHistoryCacheBoundaries(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        promptTokens: [Int]
    ) -> [Int] {
        guard let controllable = tokenizer as? any GenerationPromptControllableTokenizer else {
            return []
        }
        guard let historyTokens = try? controllable.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            addGenerationPrompt: false)
        else {
            return []
        }
        guard !historyTokens.isEmpty,
              historyTokens.count < promptTokens.count,
              promptTokens.prefix(historyTokens.count).elementsEqual(historyTokens)
        else {
            return []
        }
        return [historyTokens.count]
    }

    static func defaultContext(
        modelType: String?,
        capabilities: JangCapabilities?
    ) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]

        // Trust the bundle's capability stamp. The runtime must NOT
        // auto-rewrite or apply family-based default-template overrides
        // by model_type prefix — that hides bundle-metadata bugs and
        // contradicts a properly-stamped bundle's declaration. Fix bundle
        // metadata at the source instead.
        //
        // The one universal rule that DOES belong here: a bundle that
        // explicitly stamps `supports_thinking == false` (Ling/Bailing-style
        // non-thinking models) gets `enable_thinking=false` seeded so a
        // stale chat template can't produce noise. Per-request overrides
        // are still respected.
        if capabilities?.supportsThinking == false {
            context["enable_thinking"] = false
        }

        _ = modelType  // intentionally unused — see above
        return context.isEmpty ? nil : context
    }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama3_8B_4bit)
/// ```
public final class LLMModelFactory: ModelFactory {

    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry: AbstractModelRegistry

    static func mergeJANGTQSidecarStartupMetadata(
        _ configData: Data,
        modelDirectory: URL
    ) -> Data {
        guard var configDict = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any]
        else { return configData }

        let sidecarURL = modelDirectory.appending(component: "jangtq_runtime.safetensors")
        guard let sniffed = JANGTQRuntimeCache.sniffCodebookBits(at: sidecarURL) else {
            return configData
        }

        let priorFormat = (configDict["weight_format"] as? String) ?? "(unset)"
        let isMxtqStampAlready = (priorFormat.lowercased() == "mxtq")
            || (priorFormat.lowercased() == "jangtq1")
            || (priorFormat.lowercased() == "jangtq2")
            || (priorFormat.lowercased() == "jangtq4")
        if !isMxtqStampAlready {
            configDict["weight_format"] = "mxtq"
            FileHandle.standardError.write(
                Data("[Load] sidecar codebook present (\(sniffed)-bit) — forced weight_format \"mxtq\" (was: \"\(priorFormat)\"); fix the bundle's startup metadata\n".utf8))
        }
        if configDict["mxtq_bits"] == nil {
            configDict["mxtq_bits"] = sniffed
        }
        if configDict["routed_expert_bits"] == nil {
            configDict["routed_expert_bits"] = configDict["mxtq_bits"]
        }
        if var textConfig = configDict["text_config"] as? [String: Any] {
            for key in [
                "weight_format", "mxtq_seed", "mxtq_bits",
                "mxtq_gate_up_bits", "mxtq_down_bits",
                "routed_expert_bits",
            ] {
                if textConfig[key] == nil, let value = configDict[key] {
                    textConfig[key] = value
                }
            }
            configDict["text_config"] = textConfig
        }
        return (try? JSONSerialization.data(withJSONObject: configDict)) ?? configData
    }

    /// Bundle names of always-reasoning models that emit `<think>...</think>`
    /// intrinsically but ship NO declarative reasoning signal (empty chat
    /// template, no `reasoning_parser` stamp, no `enable_thinking` kwarg). These
    /// can't be caught by template or `model_type` heuristics, so they're
    /// recognised by name and given the think_xml parser. Keep this list narrow
    /// — only models confirmed to ALWAYS reason without any other signal.
    private static func bundleNameImpliesIntrinsicThinking(_ modelDirectory: URL) -> Bool {
        let name = modelDirectory.lastPathComponent.lowercased()
        return name.contains("vibethinker")
    }

    private static func chatTemplateText(modelDirectory: URL) -> String? {
        let templateURL = modelDirectory.appending(component: "chat_template.jinja")
        if let template = try? String(contentsOf: templateURL, encoding: .utf8) {
            return template
        }
        let tokenizerConfigURL = modelDirectory.appending(component: "tokenizer_config.json")
        guard
            let data = try? Data(contentsOf: tokenizerConfigURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let template = object["chat_template"] as? String
        else {
            return nil
        }
        return template
    }

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let modelDirectory = configuration.modelDirectory

        // Load config.json once and decode for both base config and model-specific config
        let configurationURL = modelDirectory.appending(component: "config.json")
        var configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // JANGTQ: merge `weight_format`, `mxtq_bits`, `mxtq_seed` from
        // jang_config.json into config.json so per-type creator closures can
        // dispatch on them (e.g. minimax_m2 → MiniMaxJANGTQModel). The real
        // `weight_format` for JANGTQ models lives in jang_config.json, not
        // config.json — without this merge the factory never sees "mxtq" and
        // falls through to the standard non-TQ model path.
        let jangConfigURL = modelDirectory.appending(component: "jang_config.json")
        if let jangData = try? Data(contentsOf: jangConfigURL),
            var configDict = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any],
            let jangDict = (try? JSONSerialization.jsonObject(with: jangData)) as? [String: Any]
        {
            let jangFormat = (jangDict["format"] as? String)?.lowercased()
            let jangWeightFormat = (jangDict["weight_format"] as? String)?.lowercased()
            let hasJANGTQSidecar = FileManager.default.fileExists(
                atPath: modelDirectory.appending(component: "jangtq_runtime.safetensors").path)
            let shouldMergeJANGTQMetadata =
                hasJANGTQSidecar
                || jangFormat?.contains("jangtq") == true
                || jangWeightFormat == "mxtq"
                || jangWeightFormat?.contains("jangtq") == true

            if shouldMergeJANGTQMetadata {
            for key in ["weight_format", "mxtq_seed"] {
                if configDict[key] == nil, let v = jangDict[key] {
                    configDict[key] = v
                }
            }
            // mxtq_bits resolution (2026-05-04 JANGTQ_K extension).
            // Bundles ship `mxtq_bits` in three shapes:
            //   1. Flat Int — uniform bit width across all roles
            //   2. Per-role dict like
            //        {"routed_expert": 2, "attention": 8, ...}
            //      → use routed_expert as the scalar.
            //   3. Per-PROJECTION nested dict (JANGTQ_K) like
            //        {"routed_expert": {"gate_proj": 2, "down_proj": 4,
            //                           "up_proj": 2},
            //         "attention": 8, ...}
            //      → emit separate `mxtq_gate_up_bits` /
            //        `mxtq_down_bits` keys for model configs that opt
            //        in (MiniMaxJANGTQ etc.); the legacy `mxtq_bits`
            //        Int is also overwritten with the gate_up width so
            //        callers that only know about uniform-bit configs
            //        keep working.
            //
            // Either `config.json` or `jang_config.json` may carry the
            // nested dict — try both. Always REPLACE configDict's
            // `mxtq_bits` with the resolved Int so the downstream
            // Codable decoders (which expect Int) don't see a dict.
            let bitsMaps: [[String: Any]] = [
                configDict["mxtq_bits"] as? [String: Any],
                jangDict["mxtq_bits"] as? [String: Any],
            ].compactMap { $0 }
            func intValue(_ value: Any?) -> Int? {
                if let value = value as? Int {
                    return value
                }
                if let value = value as? NSNumber {
                    return value.intValue
                }
                return nil
            }
            for bitsMap in bitsMaps {
                let routedAny = bitsMap["routed_expert"]
                if let routedInt = intValue(routedAny) {
                    configDict["mxtq_bits"] = routedInt
                    break
                } else if let routedDict = routedAny as? [String: Any] {
                    let gate = intValue(routedDict["gate_proj"])
                    let up = intValue(routedDict["up_proj"])
                    let down = intValue(routedDict["down_proj"])
                    // gate and up MUST share a width — the fused gate+up
                    // Metal kernel takes a single `bits` parameter.
                    if let g = gate, let u = up, g == u {
                        configDict["mxtq_bits"] = g
                        configDict["mxtq_gate_up_bits"] = g
                        if let d = down {
                            configDict["mxtq_down_bits"] = d
                        }
                        break
                    } else if let g = gate, let u = up {
                        throw ModelFactoryError.configurationFileError(
                            jangConfigURL.lastPathComponent,
                            configuration.name,
                            NSError(
                                domain: "LLMModelFactory",
                                code: 4,
                                userInfo: [
                                    NSLocalizedDescriptionKey:
                                        "mxtq_bits.routed_expert gate_proj (\(g)) and up_proj (\(u)) differ; fused gate/up JANGTQ kernels require matching widths"
                                ]))
                    } else if let g = gate ?? up {
                        configDict["mxtq_bits"] = g
                        configDict["mxtq_gate_up_bits"] = g
                        if let d = down {
                            configDict["mxtq_down_bits"] = d
                        }
                        break
                    }
                }
            }
            // 2026-04-26 robustness: bundles in the wild ship inconsistent
            // routed-bits fields (some have `mxtq_bits` Int, some
            // `routed_expert_bits` top-level, some only the nested dict
            // above, some nothing at all). Cascade through the remaining
            // signals so the runtime never silently picks the wrong
            // codebook bits and produces garbage:
            //
            //   1. `routed_expert_bits` Int at jang_config top-level
            //   2. Sniff actual codebook bits from the sidecar's
            //      `codebook.{inFeatures}.{bits}` keys (most reliable —
            //      determined at quantization time, can't drift)
            //   3. `profile` string convention ("JANGTQ4" → 4, etc.)
            //   4. Fall through to the per-config decoder default
            if configDict["mxtq_bits"] == nil,
                let routed = jangDict["routed_expert_bits"] as? Int
            {
                configDict["mxtq_bits"] = routed
            }
            // Sidecar sniff — the conclusive "is this JANGTQ?" signal.
            // A non-empty codebook in `jangtq_runtime.safetensors`
            // means routed-MoE experts use the TurboQuant codebook
            // path, regardless of what the bundle's `weight_format`
            // stamp says. Some bundles in the wild ship
            // `weight_format: "bf16"` (mislabeled — the bundle is
            // actually JANGTQ); without this auto-correction the
            // dispatch would route to the plain affine class and
            // fail with "Unhandled keys ['tq_norms', 'tq_packed']"
            // 60+ shards into the weight load.
            //
            // When detected, force `weight_format = "mxtq"` so the
            // existing dispatch logic routes to JANGTQ via its
            // normal stamp path — no new convention, no implicit
            // fields, just patching the bundle metadata in-memory
            // to match the actual on-disk reality. Logs a one-line
            // diagnostic so operators can see when the override
            // fired (a hint to repair the bundle's stamp at source).
            let sidecarURL = modelDirectory.appending(
                component: "jangtq_runtime.safetensors")
            if let sniffed = JANGTQRuntimeCache.sniffCodebookBits(
                at: sidecarURL)
            {
                let priorFormat =
                    (configDict["weight_format"] as? String) ?? "(unset)"
                let isMxtqStampAlready = (priorFormat.lowercased() == "mxtq")
                    || (priorFormat.lowercased() == "jangtq1")
                    || (priorFormat.lowercased() == "jangtq2")
                    || (priorFormat.lowercased() == "jangtq4")
                if !isMxtqStampAlready {
                    configDict["weight_format"] = "mxtq"
                    FileHandle.standardError.write(
                        Data("[Load] sidecar codebook present (\(sniffed)-bit) — forced weight_format \"mxtq\" (was: \"\(priorFormat)\"); fix the bundle's jang_config.json\n".utf8))
                }
                if configDict["mxtq_bits"] == nil {
                    configDict["mxtq_bits"] = sniffed
                }
            }
            if configDict["mxtq_bits"] == nil,
                let profile = jangDict["profile"] as? String,
                let pBits = jangtqBitsFromProfile(profile)
            {
                configDict["mxtq_bits"] = pBits
            }
            // Mirror the resolved value into BOTH conventional field
            // names so each model's decoder sees it under the field
            // it expects without needing a cross-cutting side-channel:
            //   - `mxtq_bits`         — Qwen35JANGTQ family
            //   - `routed_expert_bits` — DSV4 family
            // Idempotent: only fills missing fields, never overwrites
            // values the bundle already shipped.
            if let resolved = configDict["mxtq_bits"] as? Int {
                if configDict["routed_expert_bits"] == nil {
                    configDict["routed_expert_bits"] = resolved
                }
            }
            // VL-wrapped configs (Qwen3.5-VL, Qwen3.6-VL) put the LLM fields
            // inside `text_config`. The Qwen35JANGTQ decoder tries the
            // top-level first then falls back to decoding from `text_config`,
            // so the routed-bits resolution we just performed must ALSO be
            // mirrored into `text_config` for the nested decode path to
            // see it. Includes `routed_expert_bits` (DSV4-VL family
            // convention) so both decoder shapes work without a separate
            // injection pass.
            //
            // Mirror is unconditional fill-when-missing — never overwrite
            // a value the bundle's text_config already explicitly set.
            // This deliberately differs from a "skip if values match"
            // guard which could fall through and leave text_config nil
            // when top-level was just resolved by our cascade above
            // (see vmlx-swift §421/§425 — that bug class is closed here
            // by always running the mirror, not by adding a guard).
            if var textConfig = configDict["text_config"] as? [String: Any] {
                for key in [
                    "weight_format", "mxtq_seed", "mxtq_bits",
                    "mxtq_gate_up_bits", "mxtq_down_bits",
                    "routed_expert_bits",
                ] {
                    if textConfig[key] == nil, let v = configDict[key] {
                        textConfig[key] = v
                    }
                }
                configDict["text_config"] = textConfig
            }
            }
            if let merged = try? JSONSerialization.data(withJSONObject: configDict) {
                configData = merged
            }
        }
        configData = Self.mergeJANGTQSidecarStartupMetadata(
            configData,
            modelDirectory: modelDirectory)

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let earlyJangConfig: JangConfig?
        if JangLoader.isJangModel(at: modelDirectory) {
            earlyJangConfig = try JangLoader.loadConfig(at: modelDirectory)
        } else {
            earlyJangConfig = nil
        }
        let earlyMTPStatus = try? MTPBundleInspector.inspect(
            modelDirectory: modelDirectory,
            jangConfig: earlyJangConfig)
        let loadNativeMTP = try NativeMTPActivation.shouldLoadNativeMTPWeights(
            configData: configData,
            baseModelType: baseConfig.modelType,
            status: earlyMTPStatus)
        if !loadNativeMTP {
            configData = try NativeMTPActivation.scrubInactiveMTPConfig(configData)
        }

        if JANGTQStreamingExperts.shouldAutoEnableNemotronUltra(modelDirectory: modelDirectory) {
            JANGTQStreamingExperts.configureModelDirectory(modelDirectory)
        }
        if JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(modelDirectory: modelDirectory) {
            JANGTQStreamingExperts.configureModelDirectory(modelDirectory)
            if var configDict = try? JSONSerialization.jsonObject(with: configData)
                as? [String: Any]
            {
                configDict["auto_streaming_experts"] = true
                if var textConfig = configDict["text_config"] as? [String: Any] {
                    textConfig["auto_streaming_experts"] = true
                    configDict["text_config"] = textConfig
                }
                if let updated = try? JSONSerialization.data(withJSONObject: configDict) {
                    configData = updated
                }
            }
        }

        // Determine effective model type for the LLM factory.
        // VLM configs may wrap a different text decoder (e.g., mistral3 VLM wraps mistral4 text).
        // If text_config.model_type exists and differs from the top-level, prefer it when
        // it's a registered type — it means the text decoder is a different architecture.
        struct TextConfigModelType: Codable {
            let modelType: String?
            enum CodingKeys: String, CodingKey { case modelType = "model_type" }
        }
        struct TextConfigWrapper: Codable {
            let textConfig: TextConfigModelType?
            enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
        }
        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch {
            // Top-level model_type failed (e.g. "mistral3" is a VLM type not in LLM registry,
            // or the config couldn't be decoded for that type).
            // Try text_config.model_type as fallback (e.g. "mistral4" text decoder).
            if let wrapper = try? JSONDecoder.json5().decode(TextConfigWrapper.self, from: configData),
                let textModelType = wrapper.textConfig?.modelType,
                textModelType != baseConfig.modelType
            {
                do {
                    model = try await typeRegistry.createModel(
                        configuration: configData, modelType: textModelType)
                } catch let innerError as DecodingError {
                    throw ModelFactoryError.configurationDecodingError(
                        configurationURL.lastPathComponent, configuration.name, innerError)
                }
            } else if let decodingError = error as? DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, decodingError)
            } else {
                throw error
            }
        }

        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        let generationConfig =
            if let generationData = try? Data(contentsOf: generationConfigURL) {
                try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: generationData)
            } else {
                nil as GenerationConfigFile?
            }
        // Load EOS token IDs from config.json, nested text_config, with optional
        // override from generation_config.json.
        var eosTokenIds = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: baseConfig,
            configurationData: configData,
            generationConfig: generationConfig)
        if baseConfig.modelType == "deepseek_v4" {
            // DSV4's Python runtime treats EOS plus both role-boundary
            // sentinels as hard stops: {1, 128803, 128804}. The public
            // bundle's config/generation_config omit 128804
            // (`<｜Assistant｜>`), which lets role tokens leak into decode
            // instead of terminating cleanly.
            eosTokenIds.formUnion([1, 128803, 128804])
        }

        // Detect JANG model — if jang_config.json exists, load it for per-layer quantization.
        // Standard MLX models skip this entirely (jangConfig stays nil).
        let jangConfig = earlyJangConfig
        let mtpStatus = earlyMTPStatus

        // Build a ModelConfiguration with loaded EOS token IDs and tool call format.
        //
        // Tool-format resolution priority (highest first):
        //   1. Caller-supplied `configuration.toolCallFormat` (explicit override).
        //   2. JANG `capabilities.tool_parser` stamp from jang_config.json — authoritative
        //      when set, covers the short family names (`qwen`, `minimax`, `glm47`, …) the
        //      JANG converter stamps vs the canonical enum raw values.
        //   3. `ToolCallFormat.infer(from: modelType)` heuristic on config.json's
        //      `model_type`. Last resort for non-JANG standard MLX models.
        //
        // Previous code called `infer()` before the JANG load, so JANG models whose
        // `tool_parser` stamp disagreed with `model_type` were silently miscategorised.
        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds
        let chatTemplate = Self.chatTemplateText(modelDirectory: modelDirectory)
        if mutableConfiguration.toolCallFormat == nil {
            // New DSV4-era stamp lives under `chat.tool_calling.parser`
            // (e.g. `"dsml"` for DeepSeek-V4-Flash). Keep the legacy
            // `capabilities.tool_parser` path as the second priority
            // — older bundles use it, and the new schema inherits it
            // as a fallback. Finally `model_type` infer.
            let chatStamped = ToolCallFormat.fromCapabilityName(
                jangConfig?.chat?.toolCalling?.parser)
            let templateAwareJang =
                ParserResolution.toolCall(
                    capabilities: jangConfig?.capabilities,
                    modelType: baseConfig.modelType,
                    chatTemplate: chatTemplate
                ).format
            mutableConfiguration.toolCallFormat =
                chatStamped
                ?? templateAwareJang
                ?? ToolCallFormat.infer(from: baseConfig.modelType)
        }

        // Reasoning-parser stamp: same precedence ladder as the tool-call
        // format. JANG `capabilities.reasoning_parser` wins when present;
        // otherwise we pick a parser off the model_type heuristic. The
        // stamp is the short capability name (e.g. `"qwen3_6"`, `"gemma4"`);
        // Evaluate + BatchEngine resolve it to a live ReasoningParser
        // instance via `ReasoningParser.fromCapabilityName(_:)`.
        //
        // CORRECTNESS CRITICAL: historically this was a reverse-allowlist
        // that defaulted ANY model_type outside {gemma4, gemma, mistral}
        // to `"think_xml"`. `think_xml` starts with `startInReasoning: true`
        // to match Qwen's `<think>`-prefilled prompt tail — so for every
        // non-reasoning family (LFM2, LLaMA, Phi, StarCoder2, Cohere,
        // OpenELM, InternLM2, NanoChat, …) every decoded chunk
        // came out as `Generation.reasoning(_)` and osaurus rendered the
        // entire answer into the thinking block. Fixed by flipping to an
        // explicit allowlist of the model_types that ACTUALLY emit a
        // `<think>…</think>` envelope natively (Qwen 3.x, DeepSeek-V3/V4,
        // GLM 4/5, MiniMax M2+, Kimi K2.x, Nemotron-H). Every other
        // model_type falls through to `"none"` and emits plain `.chunk`.
        if mutableConfiguration.reasoningParserName == nil {
            if Self.bundleNameImpliesIntrinsicThinking(modelDirectory) {
                // Always-reasoning model that emits <think>...</think>
                // intrinsically with NO declarative signal anywhere — empty
                // chat template, no reasoning_parser stamp, no enable_thinking
                // kwarg (e.g. VibeThinker, a qwen2 reasoner). Template- and
                // model_type-based detection both correctly see "no thinking
                // template" and would leave the CoT leaking into visible
                // content. Recognise it by bundle name and apply the think_xml
                // parser (`<think>…</think>` stripping; startInReasoning=false
                // since the tag is emitted by the model, not the prompt tail).
                mutableConfiguration.reasoningParserName = "think_xml"
            } else if ParserResolution.shouldIgnoreReasoningStamp(
                capabilities: jangConfig?.capabilities,
                modelType: baseConfig.modelType)
            {
                let resolvedReasoning = ParserResolution.reasoning(
                    capabilities: jangConfig?.capabilities,
                    modelType: baseConfig.modelType,
                    chatTemplate: chatTemplate)
                mutableConfiguration.reasoningParserName =
                    resolvedReasoning.parser == nil
                    ? "none"
                    : (resolvedReasoning.source == .chatTemplate
                        ? "qwen3"
                        : reasoningStampFromModelType(baseConfig.modelType))
            } else if let stamp = jangConfig?.capabilities?.reasoningParser {
                mutableConfiguration.reasoningParserName = stamp
            } else {
                let resolvedReasoning = ParserResolution.reasoning(
                    capabilities: jangConfig?.capabilities,
                    modelType: baseConfig.modelType,
                    chatTemplate: chatTemplate)
                mutableConfiguration.reasoningParserName =
                    resolvedReasoning.parser == nil
                    ? "none"
                    : (resolvedReasoning.source == .chatTemplate
                        ? "qwen3"
                        : reasoningStampFromModelType(baseConfig.modelType))
            }
        }

        // Load tokenizer and weights in parallel.
        //
        // JANG / JANGTQ bundles ship weights-only — the snapshot directory
        // usually has no `tokenizer.json`. Falling back to the source model's
        // cached tokenizer is the only way the chat template gets applied.
        // `resolveTokenizerDirectory` returns the original directory unchanged
        // when there is no fallback to perform, so this is a no-op for
        // standard models.
        let jangResolvedDir = JangLoader.resolveTokenizerDirectory(
            for: configuration.tokenizerDirectory)
        let templateResolvedDir = JangLoader.resolveChatTemplateSidecarSubstitution(
            for: jangResolvedDir)
        // Then rewrite `tokenizer_class` if swift-transformers doesn't know
        // it (TokenizersBackend → Qwen2Tokenizer for Qwen VL). Returns
        // the input unchanged for already-supported classes.
        let tokenizerDirectory = JangLoader.resolveTokenizerClassSubstitution(
            for: templateResolvedDir)
        async let tokenizerTask = tokenizerLoader.load(from: tokenizerDirectory)

        // When JANG, skip config.json's perLayerQuantization — JANG infers correct
        // per-layer bits from tensor shapes. This avoids creating QuantizedLinear at
        // the wrong bit width (which can't be re-quantized later).
        // BUT: still pass `quantization` (the global config.json group_size /
        // bits) so JangLoader.inferPerLayerQuantization gets the correct
        // `knownGroupSize` even when jang_config.json doesn't carry quant
        // metadata (e.g. DSV4-Flash bundles ship `weight_format: "bf16"`).
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            quantization: jangConfig != nil ? baseConfig.quantizationContainer?.quantization : nil,
            // 2026-04-28: pass perLayerQuantization through even when JANG;
            // loadWeights merges config.json's explicit per-layer dict on
            // top of the shape walk to fix mis-inference of (bits, gs) pairs
            // that share a packed shape (e.g., (4,64) ≡ (8,32)). Closes
            // Cascade-2 JANG_4M / Nemotron-Omni MXFP4 first-prefill rmsNorm.
            perLayerQuantization: baseConfig.perLayerQuantization,
            jangConfig: jangConfig,
            loadPreservedMTP: loadNativeMTP)

        let tokenizer = try await tokenizerTask

        let messageGenerator =
            if let model = model as? LLMModel {
                model.messageGenerator(tokenizer: tokenizer)
            } else {
                DefaultMessageGenerator()
            }

        // Build a ModelConfiguration for the ModelContext. When the JANG
        // fallback resolved to a different directory than the caller
        // requested, surface that in the `tokenizerSource` so any re-load
        // via this config uses the same tokenizer.
        let tokenizerSource: TokenizerSource? =
            tokenizerDirectory == modelDirectory
            ? nil
            : .directory(tokenizerDirectory)
        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: tokenizerSource,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: mutableConfiguration.extraEOSTokens,
            eosTokenIds: mutableConfiguration.eosTokenIds,
            toolCallFormat: mutableConfiguration.toolCallFormat,
            reasoningParserName: mutableConfiguration.reasoningParserName,
            generationDefaults: generationConfig,
            mtpStatus: mtpStatus)

        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer, configuration: modelConfig,
            modelType: baseConfig.modelType,
            messageGenerator: messageGenerator,
            defaultAdditionalContext: LLMUserInputProcessor.defaultContext(
                modelType: baseConfig.modelType,
                capabilities: jangConfig?.capabilities))

        return .init(
            configuration: modelConfig, model: model, processor: processor,
            tokenizer: tokenizer)
    }

}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
        LLMModelFactory.shared
    }
}
