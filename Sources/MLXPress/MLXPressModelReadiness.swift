import Foundation

public enum MLXPressReadinessState: String, Sendable, Codable, Equatable {
    case proven
    case created
    case partial
    case blocked
    case missing
    case notApplicable = "not_applicable"
}

public struct MLXPressReadinessItem: Sendable, Codable, Equatable {
    public let gate: String
    public let state: MLXPressReadinessState
    public let detail: String
    public let evidence: String?

    public init(
        gate: String,
        state: MLXPressReadinessState,
        detail: String,
        evidence: String? = nil
    ) {
        self.gate = gate
        self.state = state
        self.detail = detail
        self.evidence = evidence
    }
}

public struct MLXPressModelReadinessChecklist: Sendable, Codable, Equatable {
    public let family: String
    public let attentionArchitecture: String
    public let loadMethod: String
    public let overallState: MLXPressReadinessState
    public let summary: String
    public let items: [MLXPressReadinessItem]
    public let requiredProofs: [String]
    public let blockers: [String]

    public init(
        family: String,
        attentionArchitecture: String,
        loadMethod: String,
        overallState: MLXPressReadinessState,
        summary: String,
        items: [MLXPressReadinessItem],
        requiredProofs: [String],
        blockers: [String]
    ) {
        self.family = family
        self.attentionArchitecture = attentionArchitecture
        self.loadMethod = loadMethod
        self.overallState = overallState
        self.summary = summary
        self.items = items
        self.requiredProofs = requiredProofs
        self.blockers = blockers
    }

    public static func build(for facts: MLXPressBundleFacts) -> MLXPressModelReadinessChecklist {
        let family = MLXPressModelFamily.from(facts: facts)
        var items = commonItems(for: facts, family: family)
        items.append(contentsOf: familyItems(for: facts, family: family))

        let blockers = items
            .filter { $0.state == .blocked || $0.state == .missing }
            .map { "\($0.gate): \($0.detail)" }
        let overall: MLXPressReadinessState
        if blockers.contains(where: { _ in true }) {
            overall = .blocked
        } else if items.contains(where: { $0.state == .partial }) {
            overall = .partial
        } else if items.contains(where: { $0.state == .created }) {
            overall = .created
        } else {
            overall = .proven
        }

        return MLXPressModelReadinessChecklist(
            family: family.displayName,
            attentionArchitecture: family.attentionArchitecture,
            loadMethod: readinessLoadMethod(for: facts),
            overallState: overall,
            summary: family.summary,
            items: items,
            requiredProofs: readinessRequiredProofs(for: family, facts: facts),
            blockers: blockers)
    }
}

private enum MLXPressModelFamily: Sendable, Equatable {
    case minimax
    case hy3
    case kimi
    case qwenVL
    case qwen36
    case zaya
    case ling
    case deepseekV4
    case generic

    static func from(facts: MLXPressBundleFacts) -> MLXPressModelFamily {
        let modelType = facts.modelType?.lowercased() ?? ""
        let name = facts.directory.lastPathComponent.lowercased()
        let probe = "\(modelType) \(name)"
        if probe.contains("minimax") { return .minimax }
        if probe.contains("hy_v3") || probe.contains("hy3") || probe.contains("hunyuan") {
            return .hy3
        }
        if probe.contains("kimi") { return .kimi }
        if probe.contains("qwen3_vl") || probe.contains("qwen2_vl")
            || probe.contains("qwen2_5_vl") || probe.contains("qwen2.5-vl")
        {
            return .qwenVL
        }
        if probe.contains("qwen3_5_moe") || probe.contains("qwen3.6") || probe.contains("qwen36") {
            return .qwen36
        }
        if probe.contains("zaya") { return .zaya }
        if probe.contains("bailing") || probe.contains("ling") { return .ling }
        if probe.contains("deepseek_v4") || probe.contains("dsv4") { return .deepseekV4 }
        return .generic
    }

    var displayName: String {
        switch self {
        case .minimax: return "MiniMax M2"
        case .hy3: return "Hy3 / Hunyuan"
        case .kimi: return "Kimi K2.x"
        case .qwenVL: return "Qwen VL"
        case .qwen36: return "Qwen3.6 MoE"
        case .zaya: return "ZAYA"
        case .ling: return "Ling / Bailing Hybrid"
        case .deepseekV4: return "DeepSeek V4 Flash"
        case .generic: return "Generic MLX/JANG"
        }
    }

    var attentionArchitecture: String {
        switch self {
        case .minimax:
            return "dense self-attention + routed MoE"
        case .hy3:
            return "dense self-attention + shared-expert routed MoE + MTP weights"
        case .kimi:
            return "DeepSeek-V3/Kimi dense attention lineage + large routed MoE"
        case .qwenVL:
            return "Qwen VL dense attention + image/video encoder + 2D/3D MRoPE"
        case .qwen36:
            return "Qwen MoE attention, possible hybrid/path-dependent variants"
        case .zaya:
            return "text/VL MoE + ZAYA CCA companion cache"
        case .ling:
            return "Bailing hybrid attention/MoE, companion-state audit required"
        case .deepseekV4:
            return "DSV4 compressor/indexer attention + routed MoE"
        case .generic:
            return "unknown or dense MLX/JANG family"
        }
    }

    var summary: String {
        switch self {
        case .minimax:
            return "Low-RAM MLXPress rows exist for thinking-off MiniMax; thinking-on final-answer closure and tool rows remain gated."
        case .hy3:
            return "Uniform Hy3 has coherent MLXPress rows; Hy3 JANGTQ_K load is fixed but decode peak/coherency remain blocked."
        case .kimi:
            return "Kimi low-RAM load and short no-thinking decode are proven for Kimi Small; longer decode, thinking/tool, multi-turn, and warm-cache rows remain gated."
        case .qwenVL:
            return "Qwen VL needs media-salted cache keys, 2D/3D MRoPE position proof, and image/video cold/warm rows before MLXPress cache hits are safe."
        case .qwen36:
            return "Qwen3.6 MoE plumbing exists; combined-repo MLXPress cache-stack proof is still required."
        case .zaya:
            return "ZAYA needs CCA companion-cache handling and VL media-key proof before full cache hits are safe."
        case .ling:
            return "Ling/Bailing needs current combined-repo low-RAM and hybrid companion-state proof."
        case .deepseekV4:
            return "DSV4 is a special topology; compressor/indexer cache state must be keyed and validated separately."
        case .generic:
            return "Use the global gates before making any MLXPress claim."
        }
    }

    var positionEncodingContract: String {
        switch self {
        case .qwenVL:
            return "2D/3D MRoPE with image/video grid positions; cache restore must preserve per-sequence position ids, not only scalar offsets."
        case .qwen36:
            return "RoPE/MRoPE-family Qwen offsets with possible hybrid SSM layers; verify graph-visible offsets for paged/TurboQuant cache paths."
        case .kimi:
            return "DeepSeek/Kimi partial RoPE path using qk_rope_head_dim; dense/sliding cache restores must preserve RoPE offsets."
        case .deepseekV4:
            return "Dual RoPE theta for normal/compressor-indexer layers; disk cache must preserve compressor/indexer topology."
        case .zaya:
            return "Text RoPE plus VL/media position handling where loaded; CCA companion state is path-dependent."
        case .hy3:
            return "Hy3 dense attention RoPE parameters from rope_parameters; verify no MTP warm path leaks into base MLXPress."
        case .minimax:
            return "MiniMax dense attention RoPE offset path; routed MoE proof must not disturb KV position continuity."
        case .ling:
            return "Hybrid attention/linear recurrence offsets; companion state is required for path-dependent layers."
        case .generic:
            return "Unknown position encoding; inspect RoPE/MRoPE/ALiBi/sliding-window contract before cache-hit claims."
        }
    }

    var cacheStorageContract: String {
        switch self {
        case .qwenVL:
            return "Paged/disk hashes must include media cacheScopeSalt; VLM LMInput construction must carry image/video processor salt."
        case .zaya:
            return "Disk L2 must persist or reject ZayaCCACache conv_state/prev_hs plus media salt for VL rows."
        case .ling, .qwen36:
            return "Mamba/Arrays companion state must be serialized or restored with SSMStateCache; partial hits without companion state are false hits."
        case .deepseekV4:
            return "TQDiskSerializer must store HybridPoolCache rotating window, compressor/indexer pools, and incomplete-window buffers."
        case .kimi:
            return "Dense/sliding KV can use paged blocks plus TQDiskSerializer rotating/KV kinds; no SSM companion unless a future config adds one."
        default:
            return "Paged CacheBlock chain hashes, DiskCache safetensors payloads, and TQDiskSerializer layer-kind tags must match the family cache type."
        }
    }

    var hybridCompanionState: MLXPressReadinessState {
        switch self {
        case .zaya, .ling, .qwen36, .deepseekV4:
            return .partial
        case .qwenVL:
            return .notApplicable
        case .generic:
            return .missing
        default:
            return .notApplicable
        }
    }

    var mediaCacheState: MLXPressReadinessState {
        switch self {
        case .qwenVL, .zaya:
            return .missing
        default:
            return .notApplicable
        }
    }
}

private func commonItems(
    for facts: MLXPressBundleFacts,
    family: MLXPressModelFamily
) -> [MLXPressReadinessItem] {
    let architectureDetail =
        "\(family.attentionArchitecture); observed config: \(facts.architecture.attentionSummary)"
    let matmulDetail =
        "Observed matmul kernels: \(facts.architecture.matmulSummary). "
    let positionDetail =
        "\(family.positionEncodingContract) Observed config: \(facts.architecture.positionSummary). Position vectors: \(facts.architecture.positionVectorSummary)."
    let cacheStorageDetail =
        "\(family.cacheStorageContract) Observed storage: \(facts.architecture.cacheStorageSummary). Observed encode: \(facts.architecture.cacheEncodingSummary)."

    let activeStreamingState: MLXPressReadinessState
    let activeStreamingDetail: String
    if facts.format == .jangTQ && facts.isRouted {
        activeStreamingState = .created
        activeStreamingDetail = "Routed JANGTQ defaults to compression-first canonical mmap residency: resident compute with macOS cold-page reclaim/compression. Active-expert streaming is an explicit fallback/diagnostic path when the resident path is not safe yet; a coherent 1 tok/s streaming row is not the target methodology, permanent prestack is explicit opt-in only, and --ephemeral-prestack is a temporary no-permanent-overlay diagnostic."
    } else if facts.isRouted {
        activeStreamingState = .notApplicable
        activeStreamingDetail = "Routed bundle is not JANGTQ; use mmap-backed affine/JANG path and family proof."
    } else {
        activeStreamingState = .notApplicable
        activeStreamingDetail = "Dense bundle; active expert streaming is not required."
    }

    let cacheCompanionState: MLXPressReadinessState
    let cacheCompanionDetail: String
    switch family {
    case .zaya, .ling, .deepseekV4, .qwen36:
        cacheCompanionState = .partial
        cacheCompanionDetail = "Family may carry path-dependent cache state; disk/full hits must serialize or reject companion state."
    default:
        cacheCompanionState = .created
        cacheCompanionDetail = "Paged KV, disk L2, and TurboQuant KV configuration surfaces are wired; runtime proof is still per-family."
    }

    let hasMediaCacheContract = !facts.architecture.mediaCacheKinds.isEmpty
    let mediaCacheState = hasMediaCacheContract ? .missing : family.mediaCacheState
    let mediaCacheDetail: String
    if mediaCacheState == .missing {
        mediaCacheDetail = "VL rows need real image/video payloads, vector/media processor identity, MRoPE/grid position proof, and media-keyed cold/warm cache hits. Observed media cache identity: \(facts.architecture.mediaCacheSummary)."
    } else {
        mediaCacheDetail = "No VL/video proof required for this text-only family row unless a VLM wrapper is loaded."
    }

    return [
        MLXPressReadinessItem(
            gate: "single-low-ram-load-method",
            state: .created,
            detail: "MLXPress load configuration uses mmap safetensors, bounded allocator cache, memory cap, compression-first routed-weight residency, and no default prestack overlay; ephemeral prestack is explicit."),
        MLXPressReadinessItem(
            gate: "attention-architecture-classified",
            state: family == .generic ? .missing : .created,
            detail: architectureDetail),
        MLXPressReadinessItem(
            gate: "hadamard-tq-matmul-contract",
            state: facts.format == .jangTQ ? .partial : .notApplicable,
            detail: facts.format == .jangTQ
                ? "\(matmulDetail)JANGTQ rows must prove Hadamard rotation plus TurboQuant gather/fused matmul without materializing full expert stacks; streamed chunks must stay below the Activity Monitor gate."
                : "\(matmulDetail)Plain MLX/affine row; no JANGTQ Hadamard/TurboQuant matmul contract for this bundle."),
        MLXPressReadinessItem(
            gate: "rope-mrope-position-contract",
            state: family == .generic ? .missing : .partial,
            detail: positionDetail),
        MLXPressReadinessItem(
            gate: "active-expert-streaming-or-equivalent",
            state: activeStreamingState,
            detail: activeStreamingDetail),
        MLXPressReadinessItem(
            gate: "permanent-prestack-disabled-by-default",
            state: .created,
            detail: "Permanent prestacked routed overlays require MLXPRESS_PRESTACK=1 or JANGPRESS_PRESTACK=1. The typed --ephemeral-prestack path may create a process-lifetime temp overlay, but the normal MLXPress path must not write permanent overlays."),
        MLXPressReadinessItem(
            gate: "token-speed-telemetry",
            state: .created,
            detail: "CLI emits prompt/decode token/s when generation completes; missing telemetry is a blocked row."),
        MLXPressReadinessItem(
            gate: "coherency-gates",
            state: .created,
            detail: "CLI has visible/reasoning loop checks, min visible/generated token checks, expected-output checks, and length-stop failure mode."),
        MLXPressReadinessItem(
            gate: "multi-turn-harness",
            state: .created,
            detail: "Repeated --turn prompts run in one loaded session; per-turn telemetry and coherency gates are emitted."),
        MLXPressReadinessItem(
            gate: "parser-autodetect-stack",
            state: family == .generic ? .missing : .partial,
            detail: "Runtime must select ReasoningParser and ToolCallFormat from model capabilities, chat/JANG config stamps, or model-type inference; each family still needs MLXPress-on no-thinking, thinking, and tool-call transcript proof."),
        MLXPressReadinessItem(
            gate: "per-turn-ram-speed-artifact",
            state: .created,
            detail: "Validation rows must persist stdout/stderr plus per-turn prompt/decode token/s, Activity Monitor post-load/post-decode/peak gates, cache-hit tier, and no-loop verdicts."),
        MLXPressReadinessItem(
            gate: "cold-warm-deviation-proof",
            state: family == .generic ? .missing : .partial,
            detail: "scripts/compare-cache-deviation.sh provides cache-off/cold/warm comparison with isolated disk L2; every family still needs a green row or an explicit skip-off rationale."),
        MLXPressReadinessItem(
            gate: "synchronous-prompt-boundary-rederive",
            state: family.hybridCompanionState == .notApplicable && !facts.isRouted
                ? .notApplicable
                : .partial,
            detail: "Hybrid rows must prove synchronous prompt-boundary SSMReDerive or companion capture/restoration; routed rows must separately prove router/expert warm advice does not increase Activity Monitor footprint or change output. Detached async SSM rederive is not a production path."),
        MLXPressReadinessItem(
            gate: "cache-stack-surfaces",
            state: cacheCompanionState,
            detail: cacheCompanionDetail),
        MLXPressReadinessItem(
            gate: "cache-block-storage-encode",
            state: family == .generic ? .missing : .partial,
            detail: cacheStorageDetail),
        MLXPressReadinessItem(
            gate: "hybrid-companion-state-split",
            state: family.hybridCompanionState,
            detail: family.hybridCompanionState == .notApplicable
                ? "No known path-dependent SSM/CCA/compressor companion state for this row."
                : "Cache hits must include or reject all non-KV companion state before reusing the prefix. Observed companion state: \(facts.architecture.companionStateSummary). Observed split: \(facts.architecture.hybridSplitSummary)."),
        MLXPressReadinessItem(
            gate: "vl-vector-media-cache-proof",
            state: mediaCacheState,
            detail: mediaCacheDetail)
    ]
}

private func familyItems(
    for facts: MLXPressBundleFacts,
    family: MLXPressModelFamily
) -> [MLXPressReadinessItem] {
    switch family {
    case .minimax:
        return [
            MLXPressReadinessItem(
                gate: "minimax-routing-precision",
                state: .created,
                detail: "Gate matmul must stay fp32 before top-k to avoid nondeterministic expert picks."),
            MLXPressReadinessItem(
                gate: "minimax-thinking-on-final-answer",
                state: .partial,
                detail: "Reasoning text can be coherent, but production pass requires final visible answer closure without max-token fake pass.")
        ]
    case .hy3:
        let kLike = facts.directory.lastPathComponent.lowercased().contains("k")
        return [
            MLXPressReadinessItem(
                gate: "hy3-mixed-bit-routed-experts",
                state: .created,
                detail: "JANGTQ_K gate/up and down projection bits are decoded separately."),
            MLXPressReadinessItem(
                gate: "hy3-k-decode-memory-and-coherency",
                state: kLike ? .blocked : .partial,
                detail: kLike
                    ? "Hy3 K load is low-RAM, but strict decode still crosses the Activity Monitor gate and short-answer coherency failed."
                    : "Uniform Hy3 has prior coherent rows; rerun current combined-repo proof before widening claims.")
        ]
    case .kimi:
        return [
            MLXPressReadinessItem(
                gate: "kimi-short-no-thinking-decode",
                state: .partial,
                detail: "Kimi Small now has base and MLXPress cache-stack short no-thinking rows with visible output and low peak footprint; longer strict rows still need memory and coherency proof."),
            MLXPressReadinessItem(
                gate: "kimi-effective-top-k",
                state: .partial,
                detail: "Config may expose num_experts_per_tok and top_k with different meanings; runtime must use the effective MoE top-k."),
            MLXPressReadinessItem(
                gate: "kimi-parser-stack",
                state: .created,
                detail: "Kimi thinking and Kimi K2 tool-call parser surfaces exist, but no MLXPress-on tool/multi-turn proof is green yet.")
        ]
    case .qwenVL:
        return [
            MLXPressReadinessItem(
                gate: "qwen-vl-mrope-grid-proof",
                state: .missing,
                detail: "Qwen VL must prove image_grid/video_grid 2D/3D MRoPE position IDs across cold/warm cache hits."),
            MLXPressReadinessItem(
                gate: "qwen-vl-media-salt-proof",
                state: .missing,
                detail: "Same text with different images/videos must miss or use distinct media-salted cache entries.")
        ]
    case .qwen36:
        return [
            MLXPressReadinessItem(
                gate: "qwen36-moe-jangtq-runtime",
                state: .created,
                detail: "Qwen MoE/JANGTQ model path exists; combined-repo low-RAM cache-stack rows still need to run."),
            MLXPressReadinessItem(
                gate: "qwen36-path-dependent-cache-audit",
                state: .partial,
                detail: "Confirm whether this exact bundle has SSM or other companion state before accepting full disk hits.")
        ]
    case .zaya:
        return [
            MLXPressReadinessItem(
                gate: "zaya-cca-cache",
                state: .partial,
                detail: "Zaya CCA cache types exist; disk L2 hits must include or reject CCA companion state."),
            MLXPressReadinessItem(
                gate: "zaya-vl-media-cache",
                state: .missing,
                detail: "VL image/video cache keys and real media payload proof are required for ZAYA VL.")
        ]
    case .ling:
        return [
            MLXPressReadinessItem(
                gate: "ling-hybrid-state-cache",
                state: .partial,
                detail: "Bailing/Ling hybrid state must be audited before disk/full-hit reuse is production-safe."),
            MLXPressReadinessItem(
                gate: "ling-current-low-ram-proof",
                state: .missing,
                detail: "Current combined-repo MLXPress-on low-RAM multi-turn proof has not been rerun.")
        ]
    case .deepseekV4:
        return [
            MLXPressReadinessItem(
                gate: "dsv4-compressor-indexer-cache",
                state: .partial,
                detail: "Compressor/indexer cache state must be included in keys and deviation proofs."),
            MLXPressReadinessItem(
                gate: "dsv4-special-topology-proof",
                state: .missing,
                detail: "Do not generalize DSV4 cache behavior to Kimi/MiniMax/Hy3 without a family-specific row.")
        ]
    case .generic:
        return [
            MLXPressReadinessItem(
                gate: "family-specific-runtime-audit",
                state: .missing,
                detail: "No family-specific MLXPress checklist exists for this model type yet.")
        ]
    }
}

private func readinessRequiredProofs(
    for family: MLXPressModelFamily,
    facts: MLXPressBundleFacts
) -> [String] {
    var proofs = [
        "inspect row showing model type, routed experts, top-k, format, and safetensors bytes",
        "load row with Activity Monitor physical footprint below gate",
        "generation row with prompt/decode token/s emitted",
        "coherent no-loop visible output with enough generated tokens",
        "multi-turn row in one loaded session",
        "cache-off/cold/warm or cold/warm deviation row with disk L2 isolation",
        "per-turn artifact with stdout/stderr, token/s, Activity Monitor gates, cache-hit tier, and no-loop verdict",
        "parser-autodetect row covering no-thinking, reasoning, and tool-call modes where supported",
        "reasoning-on row and tool-call row when the family supports them"
    ]
    if facts.format == .jangTQ && facts.isRouted {
        proofs.append("compression-first mmap row showing resident compute semantics, routed tensors loaded as canonical weights, low Activity Monitor footprint, no permanent prestack overlay, usable decode token/s, and low file/page pressure")
    }
    switch family {
    case .zaya:
        proofs.append("CCA companion cache hit/reject proof and real image/video media-cache proof")
    case .qwenVL:
        proofs.append("Qwen VL 2D/3D MRoPE position-id proof with media-salted cold/warm cache hits")
    case .ling, .qwen36:
        proofs.append("hybrid/path-dependent cache companion-state audit")
    case .deepseekV4:
        proofs.append("compressor/indexer cache-key and deviation proof")
    case .kimi:
        proofs.append("longer Kimi MLXPress cache-stack decode/coherency proof with enough tokens, plus thinking, tool-call, multi-turn, and warm-cache rows")
    default:
        break
    }
    if !facts.architecture.mediaCacheKinds.isEmpty && family != .qwenVL && family != .zaya {
        proofs.append("real media-cache proof for detected vision/audio config with media-salted cold/warm hits")
    }
    return proofs
}

private func readinessLoadMethod(for facts: MLXPressBundleFacts) -> String {
    if facts.format == .jangTQ && facts.isRouted {
        return "mmap safetensors + compression-first routed-weight residency + no default permanent prestack"
    }
    if facts.isRouted {
        return "mmap safetensors + bounded allocator/cache policy + family routed path"
    }
    return "mmap safetensors + bounded allocator/cache policy"
}
