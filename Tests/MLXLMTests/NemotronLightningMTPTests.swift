import BenchmarkHelpers
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// NVIDIA-Nemotron-3.5-Lightning-30B-A3B: `nemotron_h`, 52 layers of
/// 23 mamba / 23 moe / 6 attention, 128 routed experts (6 per token) plus one
/// shared, and an MTP head the runtime does not implement yet.
///
/// The MTP weights in the bundle are:
///
///     mtp.layers.0   enorm | hnorm | eh_proj | norm | mixer.{q,k,v,o}_proj
///     mtp.layers.1   norm | mixer.gate(+e_score_correction_bias)
///                    | mixer.experts.{0..127}.{up,down}_proj
///                    | mixer.shared_experts.{up,down}_proj | final_layernorm
///
/// which is the main `NemotronHBlock` shape (`norm` + `mixer`) with a
/// DeepSeek-style `enorm`/`hnorm`/`eh_proj` fusion in front.
///
/// This suite establishes the ground truth before any MTP code is written:
/// does the bundle load, does it forward, and what does the loader currently do
/// with the `mtp.*` tensors. Enable with `NEMO_LIGHTNING_LIVE=1`.
@Suite("Nemotron Lightning MTP", .serialized)
struct NemotronLightningMTPTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["NEMO_LIGHTNING_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    /// Config-only: no weights, so this runs anywhere the bundle is present and
    /// pins the shape facts the MTP implementation depends on.
    @Test("config decodes the hybrid pattern and MoE shape", .enabled(if: enabled))
    func configDecodes() throws {
        let data = try Data(contentsOf: Self.bundle.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(NemotronHConfiguration.self, from: data)

        #expect(config.modelType == "nemotron_h")
        #expect(config.numHiddenLayers == 52)
        #expect(config.hybridOverridePattern.count == 52)
        #expect(config.nRoutedExperts == 128)
        #expect(config.numExpertsPerTok == 6)
        #expect(config.nSharedExperts == 1)

        // 23 mamba / 23 moe / 6 attention, whatever letters the pattern uses.
        let counts = Dictionary(
            grouping: config.hybridOverridePattern, by: { $0 }
        ).mapValues(\.count)
        print("[nemo] pattern letters: \(counts.sorted { $0.key < $1.key })")
        #expect(counts.values.reduce(0, +) == 52)
    }

    /// What the loader does with `mtp.*` today. Printed rather than asserted:
    /// this is the measurement that tells us whether the head is silently
    /// dropped, and it becomes an assertion once the module exists.
    @Test("MTP tensors are present in the bundle index", .enabled(if: enabled))
    func mtpTensorsPresent() throws {
        struct Index: Decodable { let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" } }
        let data = try Data(
            contentsOf: Self.bundle.appendingPathComponent("model.safetensors.index.json"))
        let keys = try JSONDecoder().decode(Index.self, from: data).weightMap.keys

        let mtp = keys.filter { $0.hasPrefix("mtp.") }
        let experts = Set(
            mtp.compactMap { key -> Int? in
                guard let range = key.range(of: "mixer.experts.") else { return nil }
                return Int(key[range.upperBound...].prefix { $0.isNumber })
            })

        print("[nemo] mtp tensors=\(mtp.count) experts=\(experts.count)")
        #expect(mtp.contains("mtp.layers.0.eh_proj.weight"))
        #expect(mtp.contains("mtp.layers.0.enorm.weight"))
        #expect(mtp.contains("mtp.layers.0.hnorm.weight"))
        #expect(mtp.contains("mtp.layers.1.mixer.gate.weight"))
        #expect(mtp.contains("mtp.layers.1.final_layernorm.weight"))
        #expect(experts.count == 128, "expected 128 MTP experts, saw \(experts.count)")
    }

    /// The base runtime has to work before MTP is worth writing. 61 GB BF16, so
    /// this is slow by construction.
    @Test("bundle loads and forwards", .enabled(if: enabled))
    func loadsAndForwards() async throws {
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: Self.bundle, using: NoOpTokenizerLoader())
        let model = context.model

        let cache = model.newCache(parameters: nil)
        let prompt = MLXArray((0 ..< 8).map { Int32(100 + $0) })[.newAxis, .ellipsis]
        let logits = model(prompt, cache: cache)
        // MLX's `eval` forces the lazy array graph — not code evaluation.
        eval(logits)

        print("[nemo] logits shape \(logits.shape)")
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(-1) > 0)

        let values = logits.asType(.float32).asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "non-finite logits from the base forward")

        // Does the loaded instance expose a native MTP head yet?
        let mtpModel = model as? any NativeMTPModel
        print("[nemo] conforms to NativeMTPModel: \(mtpModel != nil)")
        print("[nemo] nativeMTPAvailable: \(mtpModel?.nativeMTPAvailable ?? false)")
    }
    /// Default-off is a hard requirement, not a preference: the spec's measured
    /// numbers on this bundle are D2 0.84x (16 % SLOWER, 72.4 % accept) and D3
    /// 0.48x. A head that switched itself on would cost exactly the users with
    /// the least headroom.
    @Test("native MTP is off unless explicitly enabled", .enabled(if: enabled))
    func mtpDefaultsOff() async throws {
        guard ProcessInfo.processInfo.environment["VMLX_NEMOTRON_MTP"] == nil else {
            print("[nemo] VMLX_NEMOTRON_MTP is set — skipping the default-off check")
            return
        }
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: Self.bundle, using: NoOpTokenizerLoader())
        let mtpModel = context.model as? any NativeMTPModel
        #expect(mtpModel?.nativeMTPAvailable != true,
            "MTP head instantiated without VMLX_NEMOTRON_MTP")
    }

    /// With the flag on: every one of the 270 `mtp.*` tensors must bind, and the
    /// head must draft a usable distribution. `verify: .all` is the real check —
    /// a shape or key-path mistake fails the load rather than silently
    /// producing a plausible-but-wrong draft, which is the failure mode that
    /// reads as "MTP just isn't worth it".
    @Test("MTP head binds real weights and drafts",
        .enabled(if: enabled && ProcessInfo.processInfo.environment["VMLX_NEMOTRON_MTP"] != nil))
    func mtpHeadDrafts() async throws {
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: Self.bundle, using: NoOpTokenizerLoader())
        let model = try #require(context.model as? any NativeMTPModel)
        #expect(model.nativeMTPAvailable, "head absent with VMLX_NEMOTRON_MTP set")

        let cache = context.model.newCache(parameters: nil)
        let prompt = MLXArray((0 ..< 8).map { Int32(100 + $0) })[.newAxis, .ellipsis]

        // Backbone gives logits + PRE-final-norm hidden.
        let backbone = model.nativeBackboneForward(prompt, cache: cache)
        eval(backbone.logits, backbone.hiddenStates)
        print("[nemo] backbone logits \(backbone.logits.shape) hidden \(backbone.hiddenStates.shape)")
        #expect(backbone.hiddenStates.dim(-1) == 2688)

        // One MTP application = D2's single draft.
        let lastHidden = backbone.hiddenStates[0..., (backbone.hiddenStates.dim(1) - 1)..., 0...]
        let nextToken = MLXArray([Int32(200)])[.newAxis, .ellipsis]
        let mtpCache = model.makeNativeMTPCache()
        let draft = model.nativeMTPForward(
            hiddenStates: lastHidden, nextTokenIds: nextToken, cache: mtpCache)
        eval(draft.logits, draft.hiddenStates)

        print("[nemo] draft logits \(draft.logits.shape) hidden \(draft.hiddenStates.shape)")
        #expect(draft.logits.dim(-1) == backbone.logits.dim(-1), "draft head must share the vocabulary")
        #expect(draft.hiddenStates.dim(-1) == 2688)

        let values = draft.logits.asType(.float32).asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "non-finite MTP draft logits")

        // A head that bound garbage produces a flat or degenerate distribution.
        let maxV = values.max() ?? 0, minV = values.min() ?? 0
        print("[nemo] draft logit range \(minV) ... \(maxV)")
        #expect(maxV - minV > 1.0, "draft distribution is flat — the head likely bound wrong")
    }
}

/// Speculation must decline a bounded KV window. `RotatingKVCache` overwrites
/// evicted positions, so a rejected draft cannot be un-written once rotation
/// wraps — and on a hybrid stack the Mamba state has no offset to rewind at
/// all. Declining is the only sound answer; the alternative is subtly wrong
/// text that no test would catch.
@Suite("Native MTP requires an unbounded KV window")
struct NativeMTPWindowGuardTests {

    private func greedy(maxKVSize: Int?) -> GenerateParameters {
        GenerateParameters(maxTokens: 32, maxKVSize: maxKVSize, temperature: 0)
    }

    @Test("an unbounded window is eligible")
    func unboundedIsEligible() {
        #expect(greedy(maxKVSize: nil).canUseNativeMTP(for: LMInput(tokens: MLXArray([Int32(1)]))))
    }

    @Test(arguments: [512, 4096, 262_144])
    func boundedWindowDeclines(_ size: Int) {
        #expect(!greedy(maxKVSize: size).canUseNativeMTP(for: LMInput(tokens: MLXArray([Int32(1)]))),
            "maxKVSize \(size) must decline speculation")
    }
}

/// D1 vs D2 on the real bundle: accept rate, throughput, and whether the
/// speculative path stays token-identical to plain autoregressive.
///
/// The spec measured D2 at 0.84x (16 % slower, 72.4 % accept) and D3 at 0.48x
/// in Python and instructs both runtimes to re-measure rather than inherit the
/// number. This is that measurement for Swift. It decides whether the toggle is
/// worth anything on this family — it does NOT decide the default, which stays
/// off either way.
///
/// `NEMO_LIGHTNING_LIVE=1 VMLX_NEMOTRON_MTP=1`.
@Suite("Nemotron Lightning MTP throughput", .serialized)
struct NemotronLightningMTPBenchTests {

    static var enabled: Bool {
        NemotronLightningMTPTests.enabled
            && ProcessInfo.processInfo.environment["VMLX_NEMOTRON_MTP"] != nil
    }

    @Test("D1 vs D2: accept rate, tok/s, token identity", .enabled(if: enabled))
    func d1VersusD2() async throws {
        let context = try await MLXLLM.LLMModelFactory.shared.load(
            from: NemotronLightningMTPTests.bundle, using: NoOpTokenizerLoader())
        let model = try #require(context.model as? any NativeMTPModel)
        #expect(model.nativeMTPAvailable)

        let steps = Int(ProcessInfo.processInfo.environment["NEMO_BENCH_TOKENS"] ?? "80") ?? 80
        let prompt = MLXArray((0 ..< 32).map { Int32(1000 + $0) })[.newAxis, .ellipsis]

        // ---- D1: plain autoregressive, greedy ----
        func runD1() -> (tokens: [Int], seconds: Double) {
            let cache = context.model.newCache(parameters: nil)
            var out: [Int] = []
            var next = prompt
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< steps {
                let logits = context.model(next, cache: cache)
                let last = logits[0..., (logits.dim(1) - 1)..., 0...]
                let token = argMax(last, axis: -1)
                eval(token)
                let id = token.item(Int.self)
                out.append(id)
                next = MLXArray([Int32(id)])[.newAxis, .ellipsis]
            }
            return (out, CFAbsoluteTimeGetCurrent() - start)
        }

        // ---- D2: one MTP draft per step, verified in a 2-wide batch ----
        func runD2() -> (tokens: [Int], seconds: Double, proposed: Int, accepted: Int) {
            let cache = context.model.newCache(parameters: nil)
            let mtpCache = model.makeNativeMTPCache()
            var out: [Int] = []
            var proposed = 0, accepted = 0
            let start = CFAbsoluteTimeGetCurrent()

            var seed = model.nativeBackboneForward(prompt, cache: cache)
            var lastHidden = seed.hiddenStates[0..., (seed.hiddenStates.dim(1) - 1)..., 0...]
            var pending = argMax(
                seed.logits[0..., (seed.logits.dim(1) - 1)..., 0...], axis: -1)
            eval(pending)

            while out.count < steps {
                let committed = pending.item(Int.self)

                // draft t+2 from (h_i, t_{i+1})
                let draft = model.nativeMTPForward(
                    hiddenStates: lastHidden,
                    nextTokenIds: MLXArray([Int32(committed)])[.newAxis, .ellipsis],
                    cache: mtpCache)
                let draftToken = argMax(draft.logits[0..., 0..., 0...], axis: -1)
                eval(draftToken)
                let d1 = draftToken.item(Int.self)
                proposed += 1

                // Snapshot Mamba state BEFORE the verify forward. The
                // recurrence absorbs the draft token in place, so once the
                // verify runs there is no offset to rewind — the state is
                // simply gone. Attention slots are position-addressed and only
                // need their offset decremented.
                for c in cache {
                    if let mamba = c as? MambaCache {
                        mamba.recordPrefixCommitState(
                            length: 1, arrays: mamba.state, offset: mamba.offset)
                    }
                }

                // verify both positions in ONE backbone pass
                let verifyInput = MLXArray([Int32(committed), Int32(d1)])[.newAxis, .ellipsis]
                let verified = model.nativeBackboneMTPVerifyForward(verifyInput, cache: cache)
                let v0 = argMax(verified.logits[0..., 0..., 0...], axis: -1)
                eval(v0)
                let truth = v0.item(Int.self)

                out.append(committed)
                if truth == d1 {
                    accepted += 1
                    if out.count < steps { out.append(d1) }
                    let v1 = argMax(verified.logits[0..., 1..., 0...], axis: -1)
                    eval(v1)
                    for c in cache {
                        (c as? MambaCache)?.clearRecordedPrefixes()
                    }
                    pending = v1
                    lastHidden = verified.hiddenStates[0..., 1..., 0...]
                } else {
                    // Rejected: the caches now hold a bogus entry for d1.
                    // Rewind the attention slots and restore Mamba state.
                    for c in cache {
                        if let mamba = c as? MambaCache {
                            _ = mamba.commitRecordedPrefix(length: 1)
                        } else {
                            // `offset` is get-only on the protocol; `trim` is
                            // the supported rewind and reports what it could
                            // actually drop.
                            _ = c.trim(1)
                        }
                    }
                    pending = v0
                    lastHidden = verified.hiddenStates[0..., 0..., 0...]
                }
            }
            return (out, CFAbsoluteTimeGetCurrent() - start, proposed, accepted)
        }

        let d1 = runD1()
        let d2 = runD2()

        let d1Rate = Double(d1.tokens.count) / d1.seconds
        let d2Rate = Double(d2.tokens.count) / d2.seconds
        let acceptRate = d2.proposed > 0 ? Double(d2.accepted) / Double(d2.proposed) : 0

        print("[nemo/bench] D1  \(d1.tokens.count) tok in \(String(format: "%.2f", d1.seconds))s = \(String(format: "%.1f", d1Rate)) tok/s")
        print("[nemo/bench] D2  \(d2.tokens.count) tok in \(String(format: "%.2f", d2.seconds))s = \(String(format: "%.1f", d2Rate)) tok/s")
        print("[nemo/bench] accept \(d2.accepted)/\(d2.proposed) = \(String(format: "%.1f", acceptRate * 100))%")
        print("[nemo/bench] speedup \(String(format: "%.2f", d2Rate / max(d1Rate, 0.0001)))x")

        let common = min(d1.tokens.count, d2.tokens.count)
        let identical = Array(d1.tokens.prefix(common)) == Array(d2.tokens.prefix(common))
        print("[nemo/bench] token-identical to D1: \(identical)")

        // Correctness is the assertion. Speed is reported, never demanded —
        // the spec's own numbers say D2 loses on this model.
        #expect(identical, "D2 diverged from greedy D1 — speculation must be lossless")
        #expect(acceptRate > 0, "no drafts accepted at all — the head is likely mis-bound")
    }
}
