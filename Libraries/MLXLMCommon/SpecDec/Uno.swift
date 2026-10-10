// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// IFM Uno self-speculation (arXiv 2609.04010, github.com/ifm-ai/uno): a conditional LoRA adds a
// discrete-diffusion pathway to the target's own weights. One Ψ-Spec "linear" cycle runs two causal
// multi-row forwards through the target:
//
//   draft   [seed, noise_1 … noise_{L-1}]   LoRA on the noise rows only. The seed row is an ordinary
//           AR step (its token is the pending `y` of the decode loop), so the "clean" token sampled from
//           it is an exact AR sample. Noise rows (uniform token ids) give proposals for the next L-1
//           positions. Their KV rows are discarded; the seed row's KV is kept.
//   verify  [clean, p_1 … p_{L-1}]          base weights. Proposals are accepted left to right with the
//           speculative rejection test (u < p/q, correction from max(0, p − q)), so the emitted
//           distribution is the target's own; greedy accepts on argmax equality and is byte-identical to
//           AR. If every proposal is accepted the last row adds a lookahead token.
//
// A cycle commits [clean, accepted…, final] — between 2 and L+1 tokens for two forwards. Afterwards the
// cache holds every committed token except `final`, which becomes the next cycle's seed: exactly the
// invariant ordinary decoding keeps (cache = prompt + returned tokens; the next token sampled, pending).

import CryptoKit
import Foundation
import MLX

/// The row mask the conditional LoRA reads during a forward: `(1, L, 1)` with 1 on rows that take the
/// diffusion pathway, or nil (base weights only). Set by the Uno cycle around its draft forward only.
public final class UnoRowMask: @unchecked Sendable {
    public var rows: MLXArray?
    public init() {}
}

/// A loaded Uno adapter: per target projection, Aᵀ `(in, r)` and scale·Bᵀ `(r, out)` in bf16.
public final class UnoAdapter: @unchecked Sendable {
    public struct Projection {
        public let aT: MLXArray
        public let bTScaled: MLXArray
    }

    public let directory: URL
    public let rank: Int
    public let scale: Float
    /// Keyed by the target module path, e.g. `model.layers.3.self_attn.q_proj`.
    public let projections: [String: Projection]

    init(directory: URL, rank: Int, scale: Float, projections: [String: Projection]) {
        self.directory = directory
        self.rank = rank
        self.scale = scale
        self.projections = projections
    }

    /// Whether `directory` holds a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors).
    public static func isAdapterDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: directory.appendingPathComponent("adapter_config.json").path)
            && fm.fileExists(
                atPath: directory.appendingPathComponent("adapter_model.safetensors").path)
    }

    public enum LoadError: Error, LocalizedError {
        case missingConfig(URL)
        case invalidConfig(String)
        case malformedTensor(String)

        public var errorDescription: String? {
            switch self {
            case .missingConfig(let url): return "Uno adapter config not found at \(url.path)"
            case .invalidConfig(let why): return "Uno adapter config is invalid: \(why)"
            case .malformedTensor(let why): return "Uno adapter tensor is malformed: \(why)"
            }
        }
    }

    /// Load a PEFT adapter directory. Every lora_A must pair with a lora_B of the configured rank;
    /// anything else fails the load (an incomplete adapter would silently draft from base weights).
    public static func load(from directory: URL) throws -> UnoAdapter {
        let configURL = directory.appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw LoadError.missingConfig(configURL) }
        guard let rank = json["r"] as? Int, rank > 0,
            let alpha = (json["lora_alpha"] as? NSNumber)?.doubleValue
        else { throw LoadError.invalidConfig("needs integer r > 0 and numeric lora_alpha") }
        let scale = Float(alpha / Double(rank))
        let arrays = try loadArrays(
            url: directory.appendingPathComponent("adapter_model.safetensors"))
        var a: [String: MLXArray] = [:]
        var b: [String: MLXArray] = [:]
        for (key, value) in arrays {
            let name = key.hasPrefix("base_model.model.")
                ? String(key.dropFirst("base_model.model.".count)) : key
            if let r = name.range(of: ".lora_A.") {
                a[String(name[..<r.lowerBound])] = value
            } else if let r = name.range(of: ".lora_B.") {
                b[String(name[..<r.lowerBound])] = value
            }
        }
        guard !a.isEmpty, Set(a.keys) == Set(b.keys) else {
            throw LoadError.malformedTensor("lora_A / lora_B sets differ")
        }
        let inputScales = try loadInputScales(directory)
        var projections: [String: Projection] = [:]
        for (path, la) in a {
            let lb = b[path]!
            guard la.ndim == 2, lb.ndim == 2, la.dim(0) == rank, lb.dim(1) == rank else {
                throw LoadError.malformedTensor(
                    "\(path): A \(la.shape) B \(lb.shape), rank \(rank)")
            }
            var aFolded = la
            if let inputScales {
                aFolded = try foldInputScale(la, path: path, scales: inputScales)
            }
            let aT = contiguous(aFolded.T.asType(.bfloat16))
            let bTScaled = contiguous((lb.T.asType(.float32) * scale).asType(.bfloat16))
            eval(aT, bTScaled)
            projections[path] = Projection(aT: aT, bTScaled: bTScaled)
        }
        return UnoAdapter(directory: directory, rank: rank, scale: scale, projections: projections)
    }

    /// AWQ input scales a converted target declares in `uno_config.json` (`input_scales`).
    ///
    /// An AWQ-quantized target folds a per-channel scale into the norm before q/k/v and gate/up, so
    /// those projections see the normalized hidden divided by that scale. The original adapter's
    /// LoRA A was trained on the unscaled hidden: `A · (x_awq ⊙ s)`, which is `(A ⊙ s) · x_awq`, so
    /// the scale folds into A's input columns once at load. o_proj and down_proj inputs are not
    /// rescaled. Without the fold the adapter drafts from the wrong coordinates (verification keeps
    /// the output exact, but acceptance drops). A declared file that is missing, altered, or short
    /// of a layer fails the load.
    static func loadInputScales(_ directory: URL) throws -> [String: MLXArray]? {
        let configURL = directory.appendingPathComponent("uno_config.json")
        guard let data = try? Data(contentsOf: configURL),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = json["input_scales"] as? String
        else { return nil }
        let url = directory.appendingPathComponent(name)
        guard let bytes = try? Data(contentsOf: url) else {
            throw LoadError.invalidConfig("uno_config.json declares input scales \(name), not found")
        }
        if let expected = json["input_scales_sha256"] as? String {
            let actual = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard actual == expected.lowercased() else {
                throw LoadError.invalidConfig("\(name) sha256 \(actual) != declared \(expected)")
            }
        }
        return try loadArrays(url: url)
    }

    /// `A ⊙ s` over A's input columns for the projections the AWQ fold rescales.
    static func foldInputScale(_ a: MLXArray, path: String, scales: [String: MLXArray]) throws
        -> MLXArray
    {
        let parts = path.split(separator: ".")
        guard let layerIndex = parts.firstIndex(of: "layers"), layerIndex + 1 < parts.count,
            let projection = parts.last
        else { throw LoadError.malformedTensor("\(path): no layer index") }
        let layer = parts[layerIndex + 1]
        let key: String
        switch projection {
        case "q_proj", "k_proj", "v_proj": key = "layers.\(layer).attn_in"
        case "gate_proj", "up_proj": key = "layers.\(layer).ffn_in"
        default: return a
        }
        guard let s = scales[key], s.ndim == 1, s.dim(0) == a.dim(1) else {
            throw LoadError.malformedTensor(
                "\(path): input scale \(key) missing or not (\(a.dim(1)))")
        }
        return a.asType(.float32) * s.asType(.float32).reshaped(1, -1)
    }
}

/// Process-wide cache of loaded adapters, so one adapter directory is read and converted once.
public final class UnoAdapterResolver: @unchecked Sendable {
    public static let shared = UnoAdapterResolver()
    private let lock = NSLock()
    private var adapters: [String: UnoAdapter] = [:]

    public func adapter(at directory: URL) throws -> UnoAdapter {
        let key = directory.standardizedFileURL.path
        lock.lock()
        if let cached = adapters[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let loaded = try UnoAdapter.load(from: directory)
        lock.lock()
        adapters[key] = loaded
        lock.unlock()
        return loaded
    }
}

/// A target that can host an Uno adapter's conditional LoRA.
public protocol UnoSelfSpeculativeModel: LanguageModel {
    /// Attach `adapter` (idempotent per adapter directory) and return the row mask its forward reads.
    /// Throws when the adapter does not cover every projection this target expects.
    func installUnoAdapter(_ adapter: UnoAdapter) throws -> UnoRowMask
    /// Exclusive upper bound of the uniform noise token ids (the diffusion-training mask token id).
    var unoNoiseTokenBound: Int { get }
}

/// How Uno picks its rows per cycle when the request does not pin one.
///
/// The best width depends on the text and on the context length. Greedy tool turns on Diplodocus
/// preview 3 (M5 Max) ran 178 tok/s at width 8 on one probe and 212 at width 16 on another (AR 86);
/// at 2.2k cached tokens sampled prose ran 88-93 at width 5, 66-77 at 12 and 55-65 at 16 (AR 79),
/// because a block over 8 rows leaves MLX's vector attention kernel and its cost climbs with the
/// context. So the width is chosen per cycle by `DFlash2WidthChooser` (expected tokens per measured
/// second, acceptance learned from every cycle), with measured costs kept per context bucket, and
/// width 1 standing for ordinary decoding: when no width beats it, decoding runs AR stretches.
/// Acceptance evidence carries from one request to the next on the same target: with a fresh 0.7
/// prior, one cycle that accepted nothing priced Uno below AR, and AR steps teach nothing about
/// acceptance, so a whole 600-token response (5.4 tokens/cycle in the request before it) ran AR.
enum UnoWidthPlan {
    static let candidates = [1, 5, 8, 12, 16]
    /// Ordinary decode steps per AR choice. AR steps are pipelined (the host reads token n while the
    /// GPU computes n+1), so one step is timed only as part of a stretch, from an idle GPU to the
    /// sync that waits for the stretch's last forward.
    static let arStretch = 8
    /// Longest AR stretch (consecutive AR choices double the stretch from `arStretch`). Each stretch
    /// ends in one host sync; at 8 steps that drained the pipelined AR loop once per 8 tokens and
    /// Uno-on-AR ran below plain AR (K2 long reasoning, 1-3 % speculative: 68-70 vs 74-80 tok/s).
    /// `VMLX_UNO_AR_STRETCH_MAX=<n>` overrides (A/B tool; 8 = the old fixed stretch).
    static let maxARStretch: Int = {
        if let raw = ProcessInfo.processInfo.environment["VMLX_UNO_AR_STRETCH_MAX"], let n = Int(raw) {
            return Swift.max(arStretch, n)
        }
        return 128
    }()

    /// Cost bucket for a cycle starting with `keys` cached tokens: one bucket below 1,024, then one
    /// per doubling.
    static func contextBucket(_ keys: Int) -> Int {
        keys < 1024 ? 0 : 1 + Int(log2(Double(keys) / 1024))
    }

    /// The width a converted bundle declares for its adapter (`uno_config.json` `runtime_block_size`,
    /// else the bundle's `jang_config.json` `speculative.block_size` for strategy `uno`).
    static func declaredWidth(adapterDirectory: URL) -> Int? {
        if let data = try? Data(contentsOf: adapterDirectory.appendingPathComponent("uno_config.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let width = (json["runtime_block_size"] as? NSNumber)?.intValue, width >= 2
        {
            return width
        }
        let jangURL = adapterDirectory.deletingLastPathComponent()
            .appendingPathComponent("jang_config.json")
        if let data = try? Data(contentsOf: jangURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let speculative = json["speculative"] as? [String: Any],
            (speculative["strategy"] as? String) == "uno",
            let width = (speculative["block_size"] as? NSNumber)?.intValue, width >= 2
        {
            return width
        }
        return nil
    }
}

/// Decode-loop counters for the Uno path.
public struct UnoGenerationStats: Sendable, Equatable {
    public var cycles = 0
    /// Uno cycles run at each width, and ordinary decode steps taken on the AR option.
    public var cyclesByWidth: [Int: Int] = [:]
    public var arSteps = 0
    public var committedTokens = 0
    public var acceptedProposals = 0
    public var proposedTokens = 0
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    /// Committed tokens per cycle (each cycle is two target forwards).
    public var tokensPerCycle: Double { cycles > 0 ? Double(committedTokens) / Double(cycles) : 0 }
}

/// Process-wide Uno counters (diagnostics: benches read deltas per request).
public enum UnoProcessCounters {
    nonisolated(unsafe) public static var cycles = 0
    nonisolated(unsafe) public static var committedTokens = 0
    nonisolated(unsafe) public static var arSteps = 0
}

/// One Ψ-Spec linear cycle at a time over a target cache. Owned by the token iterator.
struct UnoDecoder {
    static let traceEnabled = ProcessInfo.processInfo.environment["VMLX_UNO_TRACE"] == "1"

    /// Noise tokens are uniform ids in [1, maskTokenBound): the diffusion-training mask token bound
    /// (K2: vocabulary size 250,624; the reference passes `--mask-token-id 250624`).
    let model: any LanguageModel
    let mask: UnoRowMask
    let blockSize: Int
    let maskTokenBound: Int
    let temperature: Float
    let topP: Float
    let topK: Int
    let stopTokenIDs: Set<Int>
    /// Every random draw of a cycle (noise, samples, acceptance, corrections). Seeded from the
    /// request's `randomSeed` when it has one, so a seeded request reproduces under Uno too.
    var randomState = MLXRandom.RandomState()
    /// Per-cycle width choice; nil when the request pins `blockSize`.
    var chooser: DFlash2WidthChooser?
    var stats = UnoGenerationStats()

    /// Whether the next step should be ordinary decoding (the chooser's width-1 option).
    var wantsAR: Bool { chooser?.width == 1 }

    /// Pick the cost bucket for a cycle starting with `keys` cached tokens.
    mutating func prepare(keys: Int) {
        chooser?.setCostBucket(UnoWidthPlan.contextBucket(keys))
    }

    /// A finished AR stretch: `steps` ordinary tokens in `seconds`.
    mutating func observeAR(steps: Int, seconds: Double) {
        guard steps > 0, seconds > 0 else { return }
        stats.arSteps += steps
        UnoProcessCounters.arSteps += steps
        chooser?.observe(verifyWidth: 1, tokens: 1, seconds: seconds / Double(steps), count: steps)
        if Self.traceEnabled, let chooser {
            FileHandle.standardError.write(Data(String(
                format: "[Uno] ar_stretch steps=%d ms_per_step=%.2f chooser %@\n", steps,
                seconds * 1000 / Double(steps), chooser.traceDescription).utf8))
        }
    }

    /// Compile every speculative shape once, on a scratch cache, before the first request decodes.
    /// The first 8-row K2 forward compiles a kernel for ~1.8 s; without this it landed inside the first
    /// Uno response (a 2,000-token reply: 75 vs 82 tok/s for AR, the whole gap at low acceptance).
    static func prewarm(model: any LanguageModel, mask: UnoRowMask, widths: [Int], temperature: Float) {
        let cache = model.newCache(parameters: nil)
        let prefix = MLXArray((0 ..< 32).map { Int32(1000 + $0 * 7) }).reshaped(1, 32)
        eval(model(LMInput.Text(tokens: prefix), cache: cache, state: nil).logits)
        for L in widths where L > 1 {
            let tokens = MLXArray((0 ..< L).map { Int32(2000 + $0 * 13) }).reshaped(1, L)
            var rows = [Float](repeating: 1, count: L)
            rows[0] = 0
            mask.rows = MLXArray(rows).reshaped(1, L, 1).asType(.bfloat16)
            let draft = MultiRowDecodeAttention.withLinearChainMask {
                model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
            }.logits[0].asType(.float32)
            mask.rows = nil
            eval(draft)
            for layer in cache { _ = layer.trim(L) }
            let verify = MultiRowDecodeAttention.withLinearChainMask {
                model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
            }.logits[0].asType(.float32)
            if temperature > 0 {
                eval(DFlash2Sampling.probabilities(verify, temperature: temperature, topP: 0.95))
            } else {
                eval(argMax(verify, axis: -1))
            }
            for layer in cache { _ = layer.trim(L) }
        }
        Memory.clearCache()
    }

    private func forward(_ tokens: [Int], cache: [KVCache]) -> MLXArray {
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count))
        return model(input, cache: cache, state: nil).logits[0].asType(.float32)
    }

    /// Run one cycle from `seed` (the pending token, not yet in the cache). Commits at most
    /// `maxCommit` tokens (>= 3). Returns [clean, accepted…, final]; on return the cache holds the seed
    /// and every returned token except the last.
    ///
    /// One host sync per cycle: proposals stay on the GPU and feed the verify forward directly, the
    /// draft is dispatched while the host builds the verify graph, and every per-row outcome the host
    /// may need (acceptance flags, the correction for each row, the lookahead) comes back in one read.
    mutating func cycle(seed: Int, cache: [KVCache], maxCommit: Int) -> [Int] {
        let L = Swift.max(2, Swift.min(chooser?.width ?? blockSize, blockSize, maxCommit - 1))
        let greedy = temperature <= 0
        let rows = L - 1

        // Draft: seed row = base AR step, noise rows = diffusion pathway.
        let draftStart = Date.timeIntervalSinceReferenceDate
        // Noise from the MLX generator: drawn on the GPU, and reproducible under `MLXRandom.seed`
        // like the rest of sampling.
        let random = randomState
        let noise = withRandomState(random) {
            MLXRandom.randInt(low: 1, high: maskTokenBound, [L - 1], type: Int32.self)
        }
        let draftTokens = concatenated([MLXArray([Int32(seed)]), noise]).reshaped(1, L)
        var maskValues = [Float](repeating: 1, count: L)
        maskValues[0] = 0
        mask.rows = MLXArray(maskValues).reshaped(1, L, 1).asType(.bfloat16)
        let draftLogits = MultiRowDecodeAttention.withLinearChainMask {
            model(LMInput.Text(tokens: draftTokens), cache: cache, state: nil)
        }.logits[0].asType(.float32)
        mask.rows = nil
        let proposal: MLXArray  // (L,) int32: [clean, p_1 … p_{L-1}]
        var q: MLXArray? = nil
        if greedy {
            proposal = argMax(draftLogits, axis: -1).asType(.int32)
        } else {
            let probs = DFlash2Sampling.probabilities(
                draftLogits, temperature: temperature, topP: topP, topK: topK)
            proposal = withRandomState(random) {
                DFlash2Sampling.sample(probabilities: probs).reshaped(-1).asType(.int32)
            }
            q = probs[1...]
        }
        asyncEval(proposal)
        // Discard the noise rows; the seed row stays. (Offsets only: the dispatched draft graph
        // already holds the cache arrays it reads.)
        for layer in cache { _ = layer.trim(rows) }
        stats.draftSeconds += Date.timeIntervalSinceReferenceDate - draftStart

        // Verify with base weights, on-graph from the proposal.
        let verifyStart = Date.timeIntervalSinceReferenceDate
        let verifyLogits = MultiRowDecodeAttention.withLinearChainMask {
            model(LMInput.Text(tokens: proposal.reshaped(1, L)), cache: cache, state: nil)
        }.logits[0].asType(.float32)
        let outcome: MLXArray  // host read: [proposal (L) | flags (L-1) | finals (L)]
        if greedy {
            let targets = argMax(verifyLogits, axis: -1).asType(.int32)  // (L,)
            let flags = (proposal[1...] .== targets[..<rows]).asType(.int32)
            outcome = concatenated([proposal, flags, targets])
        } else {
            let p = DFlash2Sampling.probabilities(
                verifyLogits, temperature: temperature, topP: topP, topK: topK)
            let ids = proposal[1...].reshaped(rows, 1)
            let pAt = takeAlong(p[..<rows], ids, axis: -1).reshaped(-1)
            let qAt = takeAlong(q!, ids, axis: -1).reshaped(-1)
            // Correction for a rejection at each row, and the lookahead after a full accept.
            var residual = maximum(p[..<rows] - q!, MLXArray(Float(0)))
            let total = residual.sum(axis: -1, keepDims: true)
            residual = MLX.where(
                total .> 0, residual / maximum(total, MLXArray(Float(1e-30))), p[..<rows])
            let (u, corrections, lookahead) = withRandomState(random) {
                (
                    MLXRandom.uniform(low: 0, high: 1, [rows]),
                    DFlash2Sampling.sample(probabilities: residual).reshaped(-1),
                    DFlash2Sampling.sample(probabilities: p[rows ..< L]).reshaped(-1)
                )
            }
            let flags = ((u * qAt) .< pAt).asType(.int32)
            outcome = concatenated([
                proposal, flags, concatenated([corrections, lookahead]).asType(.int32),
            ])
        }
        eval(outcome)
        let host = outcome.asArray(Int32.self).map(Int.init)
        let proposals = Array(host[1 ..< L])
        let flags = Array(host[L ..< (L + rows)])
        let finals = Array(host[(L + rows)...])
        var committed = [host[0]]
        var accepted = 0
        while accepted < rows, flags[accepted] != 0 {
            committed.append(proposals[accepted])
            accepted += 1
        }
        committed.append(finals[accepted])
        // The verify forward appended [clean, p_1 … p_{L-1}]; keep clean and the accepted proposals.
        for layer in cache { _ = layer.trim(rows - accepted) }

        // A stop token ends the response: nothing after it may stay in the cache.
        if let stop = committed.firstIndex(where: { stopTokenIDs.contains($0) }),
            stop < committed.count - 1
        {
            let drop = committed.count - 1 - stop
            for layer in cache { _ = layer.trim(drop) }
            committed.removeLast(drop)
        }
        stats.verifySeconds += Date.timeIntervalSinceReferenceDate - verifyStart
        if Self.traceEnabled {
            FileHandle.standardError.write(Data(String(
                format: "[Uno] cycle=%d L=%d accepted=%d committed=%d ctx=%d draft_ms=%.2f verify_ms=%.2f\n",
                stats.cycles + 1, L, accepted, committed.count, cache.first?.offset ?? -1,
                (verifyStart - draftStart) * 1000,
                (Date.timeIntervalSinceReferenceDate - verifyStart) * 1000).utf8))
        }
        chooser?.observe(
            verifyWidth: L, tokens: accepted + 1,
            seconds: Date.timeIntervalSinceReferenceDate - draftStart)
        if Self.traceEnabled, let chooser {
            FileHandle.standardError.write(Data("[Uno] chooser \(chooser.traceDescription)\n".utf8))
        }
        stats.cycles += 1
        stats.cyclesByWidth[L, default: 0] += 1
        stats.committedTokens += committed.count
        UnoProcessCounters.cycles += 1
        UnoProcessCounters.committedTokens += committed.count
        stats.acceptedProposals += accepted
        stats.proposedTokens += proposals.count
        return committed
    }
}
