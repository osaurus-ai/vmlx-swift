// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3 prefix cache with turn cuts (v2): vMLX Python's structural rule (05-PY-STRUCTURAL-CACHE.md), ported.
///
/// A text prefill is cut at every multiple of C, at K, and at every non-assistant message end (lookbehind), so every
/// tool / user turn reuses the whole previous prompt up to its K instead of its last C multiple, and a hit stays
/// bit-identical to a cold prefill. Prompts follow the M3 message structure with synthetic marker ids (bos 1000,
/// eos 1001, roles `<id> \n`, assistant 1002); the model's markers are set directly (a stand-in tokenizer has no M3
/// vocabulary); marker resolution through a tokenizer is tested on its own.
@Suite("MiniMax-M3 structural prefix cache (turn cuts)", .serialized)
struct MiniMaxM3StructuralPrefixCacheTests {
    typealias A = MiniMaxM3AlignedPrefixCacheTests
    typealias DS = MiniMaxM3DSparkTests

    static let bos = 1000, eos = 1001, nl = 10
    static let ai = 1002, system = 1003, developer = 1004, user = 1005, tool = 1006
    static let markers = AlignedPrefillTurnMarkers(bos: bos, eos: eos, assistant: [ai, nl])
    static let gen = [bos, ai, nl]

    static func message(_ role: Int, _ content: [Int]) -> [Int] { [bos, role, nl] + content + [eos, nl] }
    static func content(_ n: Int, seed: UInt64) -> [Int] { A.ids(n, seed: seed).map { $0 % 990 } }
    /// K of a prompt that ends with the generation prompt: the `\n` after the last message's eos.
    static func k(_ prompt: [Int]) -> Int { prompt.count - gen.count - 1 }

    /// Python's 6-turn loop: system 301 + developer 707 + user 57 content tokens; tool outputs 517 / 333 / 1291 / 77 /
    /// 2113 (no K is a multiple of 64 or of either step).
    static let toolSizes = [517, 333, 1291, 77, 2113]
    static func opening() -> [Int] {
        message(system, content(301, seed: 1)) + message(developer, content(707, seed: 2))
            + message(user, content(57, seed: 3))
    }

    static func model() throws -> MiniMaxM3Model {
        let model = try A.model()
        model.setAlignedPrefillTurnMarkers(markers)
        return model
    }

    static func coordinator(_ dir: URL, step: Int) -> CacheCoordinator {
        var config = VMLXServerRuntimeSettings().cacheCoordinatorConfig(
            modelKey: "minimax-m3-structural-test-\(step)", diskCacheDirectory: dir)
        config.enableDiskCache = true
        let coordinator = CacheCoordinator(config: config)
        coordinator.setGenPromptSuffixTokens(gen)
        return coordinator
    }

    static func run(
        _ model: MiniMaxM3Model, _ prompt: [Int], coordinator: CacheCoordinator?, step: Int, recorder: A.Recorder,
        maxTokens: Int = A.maxTokens, drafter: URL? = nil
    ) throws -> A.Run {
        try A.run(
            model, prompt, coordinator: coordinator, step: step, recorder: recorder,
            boundaries: [prompt.count - gen.count], maxTokens: maxTokens, drafter: drafter)
    }

    // MARK: - The rule

    @Test("Turn cuts: non-assistant message ends only, lookbehind (cuts below r depend on tokens before r alone)")
    func turnCutsRule() {
        let a = Self.opening()
        let asst = Self.message(Self.ai, Self.content(9, seed: 4))
        let toolMsg = Self.message(Self.tool, Self.content(40, seed: 5))
        let prompt = a + asst + toolMsg + Self.gen
        let ends = [301 + 5, 301 + 5 + 707 + 5, 301 + 5 + 707 + 5 + 57 + 5]
        let systemEnd = ends[0] - 1, developerEnd = ends[1] - 1, userEnd = ends[2] - 1
        let assistantEnd = a.count + asst.count - 1
        let toolEnd = a.count + asst.count + toolMsg.count - 1
        let cuts = AlignedPrefillGeometry.turnCuts(prompt, markers: Self.markers)
        // The position right after each eos: system, developer, user, tool; never the assistant's.
        #expect(cuts == [systemEnd, developerEnd, userEnd, toolEnd])
        #expect(!cuts.contains(assistantEnd))
        #expect(toolEnd == Self.k(prompt), "K is the last tool message's cut")
        // Lookbehind: the cuts of any prefix are the full sequence's cuts up to it.
        for r in stride(from: 0, through: prompt.count, by: 7) {
            #expect(AlignedPrefillGeometry.turnCuts(Array(prompt[..<r]), markers: Self.markers)
                == cuts.filter { $0 <= r }, "prefix \(r)")
        }
        #expect(AlignedPrefillGeometry.turnCuts(prompt, markers: nil).isEmpty)
        // An eos with no open message counts as a non-assistant end.
        #expect(AlignedPrefillGeometry.turnCuts([5, Self.eos, 7], markers: Self.markers) == [2])
        // Pieces and reuse limits with cuts.
        #expect(AlignedPrefillGeometry.pieces(start: 0, end: 1000, step: 320, boundary: 990, turnCuts: [306, 990])
            .map(\.upperBound) == [306, 320, 640, 960, 990, 1000])
        #expect(AlignedPrefillGeometry.reuseLimit(
            matched: 3000, boundary: 2900, step: 2048, exactGeometry: false, turnCuts: [1018, 2741]) == 2741)
        #expect(AlignedPrefillGeometry.reuseLimit(
            matched: 2500, boundary: 2900, step: 2048, exactGeometry: false, turnCuts: [1018, 2741]) == 2048)
        // Tags: the v2 schema round-trips; v1 is unchanged.
        let tag = AlignedPrefillTag(step: 2048, boundary: 2741, turns: true)
        #expect(AlignedPrefillTag(metaState: tag.metaState) == tag && tag.metaState[0] == "aligned-prefill-v2-turns")
        #expect(AlignedPrefillTag(metaState: AlignedPrefillTag(step: 2048, boundary: 9).metaState)?.turns == false)
    }

    /// Encodes the M3 marker strings to fixed ids (and every other character to its byte), optionally breaking the joint.
    struct MarkerTokenizer: Tokenizer {
        var joinsWrong = false
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            var out: [Int] = []
            var rest = Substring(text)
            while !rest.isEmpty {
                if rest.hasPrefix("]~b]") { out.append(200019); rest = rest.dropFirst(4); continue }
                if rest.hasPrefix("[e~[") { out.append(200020); rest = rest.dropFirst(4); continue }
                if rest.hasPrefix("ai") { out.append(1361); rest = rest.dropFirst(2); continue }
                out.append(Int(rest.first!.asciiValue ?? 0))
                rest = rest.dropFirst()
            }
            if joinsWrong, out.count > 3 { out.swapAt(1, 2) }
            return out
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    @Test("Markers resolve through the tokenizer (the M3 ids); a tokenizer that encodes the joint otherwise gets none")
    func markerResolution() throws {
        let m = try #require(AlignedPrefillTurnMarkers.resolve(
            tokenizer: MarkerTokenizer(), bosText: "]~b]", eosText: "[e~[", assistantRoleText: "ai\n"))
        #expect(m == AlignedPrefillTurnMarkers(bos: 200019, eos: 200020, assistant: [1361, 10]))
        #expect(AlignedPrefillTurnMarkers.resolve(
            tokenizer: MarkerTokenizer(joinsWrong: true), bosText: "]~b]", eosText: "[e~[",
            assistantRoleText: "ai\n") == nil)
        #expect(AlignedPrefillTurnMarkers.resolve(
            tokenizer: DS.FixtureTokenizer(), bosText: "]~b]", eosText: "[e~[", assistantRoleText: "ai\n") == nil)
        // The model resolves through ModelContext and puts the markers in its identity (cache salt).
        let model = try A.model()
        // The routed rule of non-final pieces (`VMLX_M3_PREFILL_SORTED`, on by default) is part of the identity.
        let routed = MiniMaxM3Model.prefillSortedAll ? ":routed=sorted\(MiniMaxM3Model.prefillSortedMinPairs)" : ""
        #expect(model.alignedPrefillSchema == "minimax-m3:aligned-prefill-v1" + routed)
        _ = ModelContext(
            configuration: ModelConfiguration(id: "m3"), model: model,
            processor: MiniMaxM3ServingFixesTests.StandInProcessor(), tokenizer: MarkerTokenizer())
        #expect(model.alignedPrefillTurnMarkers == m)
        #expect(model.alignedPrefillSchema == "minimax-m3:aligned-prefill-v2-turns:turns=200019-200020-1361.10" + routed)
    }

    // MARK: - The tool loop

    struct TurnResult {
        let prompt: Int, k: Int, cached: Int
    }

    /// The 6-turn tool loop, warm (coordinator) vs cold (none): bit-identical every turn; returns per-turn counts.
    static func toolLoop(step: Int, recorder: A.Recorder, omitTurnCuts: Bool = false) throws
        -> (turns: [TurnResult], identical: [Bool], last: [Int], lastStore: [Int])
    {
        let dir = A.temporaryDirectory("structural-\(step)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try Self.model()
        let coordinator = Self.coordinator(dir, step: step)
        AlignedPrefill.hooks.controlOmitsTurnCuts = omitTurnCuts
        defer { AlignedPrefill.hooks.controlOmitsTurnCuts = false }
        var history = Self.opening()
        var turns: [TurnResult] = []
        var identical: [Bool] = []
        var prompt: [Int] = []
        for turn in 0 ..< 6 {
            prompt = history + Self.gen
            let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
            let cold = try Self.run(model, prompt, coordinator: nil, step: step, recorder: recorder)
            try #require(!cold.tokens.isEmpty && cold.cached == 0)
            identical.append(A.same(warm, cold))
            if !omitTurnCuts {
                let plan = try #require(warm.prefills.first?.plan)
                #expect(plan.turnCuts == AlignedPrefillGeometry.turnCuts(prompt, markers: Self.markers, end: Self.k(prompt)))
                let expectedPieces = AlignedPrefillGeometry.pieces(
                    start: warm.cached, end: prompt.count, step: step, boundary: Self.k(prompt),
                    turnCuts: plan.turnCuts ?? [])
                #expect(warm.prefills.first?.pieces == expectedPieces, "turn \(turn + 1): pieces")
                let kept = warm.restores.filter { $0.reuse > 0 }
                if warm.cached > 0 {
                    #expect(kept.count == 1 && kept[0].offsets.allSatisfy { $0 == warm.cached }
                        && kept[0].laneRows.allSatisfy { $0 == warm.cached }, "turn \(turn + 1): lanes")
                }
            }
            turns.append(TurnResult(prompt: prompt.count, k: Self.k(prompt), cached: warm.cached))
            if turn < 5 {
                history += Self.message(Self.ai, warm.tokens.map { $0 % 990 })
                    + Self.message(Self.tool, Self.content(Self.toolSizes[turn], seed: UInt64(10 + turn)))
            }
        }
        return (turns, identical, prompt, history)
    }

    @Test("6-turn tool loop: every turn reuses exactly the previous K (any size), cached == cold every step",
          arguments: [320, 2048])
    func toolLoopReusesPreviousK(step: Int) throws {
        try A.withRecorder { recorder in
            let (turns, identical, _, _) = try Self.toolLoop(step: step, recorder: recorder)
            #expect(identical.allSatisfy { $0 }, "cached != cold on turns \(identical.enumerated().filter { !$0.1 }.map { $0.0 + 1 })")
            #expect(turns[0].cached == 0)
            for t in 1 ..< turns.count {
                #expect(turns[t].cached == turns[t - 1].k, "turn \(t + 1): reused \(turns[t].cached), K_prev \(turns[t - 1].k)")
                #expect(turns[t - 1].k % step != 0)
            }
            let v1 = turns.indices.map { $0 == 0 ? 0 : (turns[$0 - 1].k / step) * step }
            print("[M3 structural] step \(step): (prompt / K / reused / v1 rule) "
                + zip(turns, v1).map { "\($0.prompt)/\($0.k)/\($0.cached)/\($1)" }.joined(separator: ", "))
        }
    }

    @Test("Partial hit (same prefix, different suffix inside the last tool output) and an exact repeat: cached == cold")
    func partialHitAndRepeat() throws {
        try A.withRecorder { recorder in
            let step = 2048
            let dir = A.temporaryDirectory("structural-partial")
            defer { try? FileManager.default.removeItem(at: dir) }
            let model = try Self.model()
            let coordinator = Self.coordinator(dir, step: step)
            // Three turns stored; the third's prompt then re-sent with its last tool output's tail replaced.
            var history = Self.opening()
            var prompts: [[Int]] = []
            for turn in 0 ..< 3 {
                let prompt = history + Self.gen
                prompts.append(prompt)
                let warm = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
                history += Self.message(Self.ai, warm.tokens.map { $0 % 990 })
                    + Self.message(Self.tool, Self.content(Self.toolSizes[turn], seed: UInt64(10 + turn)))
            }
            let third = prompts[2]
            var variant = third
            let divergence = Self.k(prompts[1]) + 150
            for i in divergence ..< (Self.k(third) - 2) { variant[i] = (variant[i] + 7) % 990 }
            let warm = try Self.run(model, variant, coordinator: coordinator, step: step, recorder: recorder)
            let cold = try Self.run(model, variant, coordinator: nil, step: step, recorder: recorder)
            // Longest stored prefix of the variant: the second turn's prompt; usable to its K, a turn cut of the variant.
            #expect(warm.cached == Self.k(prompts[1]), "partial: reused \(warm.cached), expected \(Self.k(prompts[1]))")
            #expect(A.same(warm, cold), "partial hit: cached != cold")
            let repeatWarm = try Self.run(model, third, coordinator: coordinator, step: step, recorder: recorder)
            let repeatCold = try Self.run(model, third, coordinator: nil, step: step, recorder: recorder)
            #expect(repeatWarm.cached == Self.k(third) && A.same(repeatWarm, repeatCold), "exact repeat")
            print("[M3 structural] partial: diverges at \(divergence), reused \(warm.cached); repeat reused \(repeatWarm.cached) of \(third.count)")
        }
    }

    @Test("DSpark lane: tool loop reuses the previous K, cached == cold (tokens, first logits, caches)")
    func dsparkLane() throws {
        let step = 320
        let dir = A.temporaryDirectory("structural-dspark")
        let drafter = A.temporaryDirectory("structural-dspark-drafter")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: drafter)
        }
        try DS.writeReleaseDrafter(drafter)
        try A.withRecorder { recorder in
            let model = try Self.model()
            #expect(try model.attachDSparkDraft(directory: drafter))
            let coordinator = Self.coordinator(dir, step: step)
            var history = Self.opening()
            var previousK = 0
            var specSteps = 0
            for turn in 0 ..< 4 {
                let prompt = history + Self.gen
                let warm = try Self.run(
                    model, prompt, coordinator: coordinator, step: step, recorder: recorder, maxTokens: 12,
                    drafter: drafter)
                let cold = try Self.run(
                    model, prompt, coordinator: nil, step: step, recorder: recorder, maxTokens: 12, drafter: drafter)
                specSteps += warm.dspark?.specSteps ?? 0
                #expect(warm.tokens == cold.tokens && A.equal(warm.logits[0], cold.logits[0]), "turn \(turn + 1)")
                let problems = DS.cacheProblems(warm.cache, cold.cache)
                #expect(problems.isEmpty, "turn \(turn + 1): caches differ \(problems.prefix(3))")
                if turn > 0 { #expect(warm.cached == previousK, "turn \(turn + 1): reused \(warm.cached), K_prev \(previousK)") }
                previousK = Self.k(prompt)
                history += Self.message(Self.ai, warm.tokens.map { $0 % 990 })
                    + Self.message(Self.tool, Self.content(Self.toolSizes[turn], seed: UInt64(10 + turn)))
            }
            #expect(specSteps > 0, "no DSpark verify ran")
        }
    }

    @Test("BatchEngine solo path (what Osaurus drives): restore counters show reuse == K_prev; cached == cold")
    func batchEngineSolo() async throws {
        let mlxLock = lockSerializedMLXTest()
        defer { mlxLock.unlock() }
        let step = 2048
        let dir = A.temporaryDirectory("structural-engine")
        defer { try? FileManager.default.removeItem(at: dir) }
        let recorder = A.Recorder()
        recorder.install()
        defer { A.Recorder.uninstall() }
        let model = try Self.model()
        nonisolated(unsafe) let shared = model
        let warmEngine = A.engine(shared, coordinator: Self.coordinator(dir, step: step))
        let coldEngine = A.engine(shared, coordinator: nil)
        // The engines' stand-in tokenizer has no M3 vocabulary (its context resolved no markers): set them again.
        model.setAlignedPrefillTurnMarkers(Self.markers)
        var history = Self.opening()
        var previousK = 0
        for turn in 0 ..< 4 {
            let prompt = history + Self.gen
            func arm(_ engine: BatchEngine) async -> ([MLXArray], AlignedPrefillPrefillRecord?, [AlignedPrefillRestoreRecord]) {
                let (l0, p0, r0) = (recorder.logits.count, recorder.prefills.count, recorder.restores.count)
                var parameters = GenerateParameters(maxTokens: A.maxTokens, temperature: 0)
                parameters.prefillStepSize = step
                let stream = await engine.generate(
                    input: A.input(prompt, boundaries: [prompt.count - Self.gen.count], stable: []),
                    parameters: parameters)
                for await _ in stream {}
                let logits = Array(recorder.logits[l0...])
                eval(logits)
                return (logits, recorder.prefills.count > p0 ? recorder.prefills[p0] : nil,
                        Array(recorder.restores[r0...]))
            }
            let warm = await arm(warmEngine)
            let cold = await arm(coldEngine)
            #expect(warm.0.count == cold.0.count && zip(warm.0, cold.0).allSatisfy(A.equal), "turn \(turn + 1): cached != cold")
            let reused = warm.1?.start ?? -1
            if turn > 0 {
                #expect(reused == previousK, "turn \(turn + 1): reused \(reused), K_prev \(previousK)")
                #expect(warm.2.contains { $0.reuse == previousK && $0.detail == "disk" }, "turn \(turn + 1): restores \(warm.2.map { "\($0.detail) \($0.reuse)" })")
            }
            previousK = Self.k(prompt)
            history += Self.message(Self.ai, (0 ..< 5).map { (turn * 31 + $0 * 7) % 990 })
                + Self.message(Self.tool, Self.content(Self.toolSizes[turn], seed: UInt64(10 + turn)))
        }
    }

    // MARK: - Controls (must fail)

    @Test("Control: rows produced without the turn cuts but tagged v2 are reused at K_prev and are NOT cold-identical")
    func controlWithoutCutsDiffers() throws {
        try A.withRecorder { recorder in
            let (turns, identical, _, _) = try Self.toolLoop(step: 2048, recorder: recorder, omitTurnCuts: true)
            // The control reuses what the rule allows (K_prev) ...
            #expect((1 ..< turns.count).allSatisfy { turns[$0].cached == turns[$0 - 1].k })
            // ... and the continuation from a boundary the prefill never cut at differs from the cold prefill.
            #expect(identical.dropFirst().contains(false), "omitting the turn cuts did not change any turn")
            print("[M3 structural] control (no cuts): identical per turn \(identical)")
        }
    }

    final class Lines: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func add(_ line: String) { lock.withLock { lines.append(line) } }
        var count: Int { lock.withLock { lines.count } }
        func since(_ start: Int) -> [String] { lock.withLock { Array(lines[start...]) } }
    }

    /// `tokens=<n>` of every `store ... result=written` receipt line.
    static func written(_ lines: [String]) -> [Int] {
        lines.filter { $0.hasPrefix("store ") && $0.contains("result=written") }.compactMap { line in
            line.split(separator: " ").first { $0.hasPrefix("tokens=") }.flatMap { Int($0.dropFirst("tokens=".count)) }
        }
    }

    @Test("One store per reusable boundary: K and the history boundaries' last cuts, never N / N-1 / N - gpl; reuse == K_prev, cached == cold; control: the old stores")
    func oneStorePerReusableBoundary() throws {
        try A.withRecorder { recorder in
            for single in [true, false] {
                let dir = A.temporaryDirectory("structural-single-store-\(single)")
                defer { try? FileManager.default.removeItem(at: dir) }
                let model = try Self.model()
                let coordinator = Self.coordinator(dir, step: 320)
                let lines = Lines()
                // The processor's message boundaries: system end (306), system + developer end (1018, the stable
                // prefix), the generation-stripped history. Their reusable rows are the turn cuts 305 / 1017 and K.
                let systemEnd = 306, developerEnd = 1018
                try TokenIterator.$alignedSingleStoreOverride.withValue(single) {
                    try RuntimeReceipts.$observer.withValue({ lines.add($0) }) {
                        var history = Self.opening()
                        var previousK = 0
                        for turn in 0 ..< 4 {
                            let prompt = history + Self.gen
                            let before = lines.count
                            let warm = try A.run(
                                model, prompt, coordinator: coordinator, step: 320, recorder: recorder,
                                boundaries: [systemEnd, developerEnd, prompt.count - Self.gen.count],
                                stable: [developerEnd])
                            let written = Self.written(lines.since(before))
                            let k = Self.k(prompt)
                            if single {
                                let expected = turn == 0 ? [systemEnd - 1, developerEnd - 1, k] : [k]
                                #expect(written.sorted() == expected, "turn \(turn + 1): stored \(written), expected \(expected)")
                            } else {
                                #expect(written.contains(prompt.count) && written.count > (turn == 0 ? 3 : 1),
                                        "control turn \(turn + 1): the old stores \(written)")
                            }
                            if turn > 0 {
                                #expect(warm.cached == previousK, "turn \(turn + 1): reused \(warm.cached), K_prev \(previousK)")
                            }
                            let cold = try A.run(
                                model, prompt, coordinator: nil, step: 320, recorder: recorder,
                                boundaries: [systemEnd, developerEnd, prompt.count - Self.gen.count], stable: [developerEnd])
                            #expect(A.same(warm, cold), "turn \(turn + 1): cached != cold (single store \(single))")
                            previousK = k
                            history += Self.message(Self.ai, warm.tokens.map { $0 % 990 })
                                + Self.message(Self.tool, Self.content(Self.toolSizes[turn], seed: UInt64(10 + turn)))
                        }
                    }
                }
                // A new conversation with the same system + developer messages restores the stable prefix's cut.
                if single {
                    let fresh = Self.message(Self.system, Self.content(301, seed: 1))
                        + Self.message(Self.developer, Self.content(707, seed: 2))
                        + Self.message(Self.user, Self.content(90, seed: 99)) + Self.gen
                    let warm = try A.run(
                        model, fresh, coordinator: coordinator, step: 320, recorder: recorder,
                        boundaries: [systemEnd, developerEnd, fresh.count - Self.gen.count], stable: [developerEnd])
                    let cold = try A.run(model, fresh, coordinator: nil, step: 320, recorder: recorder)
                    #expect(warm.cached == developerEnd - 1 && A.same(warm, cold), "new chat: reused \(warm.cached)")
                }
            }
        }
    }

    @Test("v1 entries never serve a v2 request (and vice versa): the salt carries the rule")
    func v1AndV2DoNotMix() throws {
        try A.withRecorder { recorder in
            let step = 320
            let dir = A.temporaryDirectory("structural-v1v2")
            defer { try? FileManager.default.removeItem(at: dir) }
            let model = try A.model()  // no markers: v1
            let coordinator = Self.coordinator(dir, step: step)
            let prompt = Self.opening() + Self.gen
            _ = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
            let v1Repeat = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
            #expect(v1Repeat.cached == Self.k(prompt), "v1 exact repeat should hit")
            model.setAlignedPrefillTurnMarkers(Self.markers)
            let v2 = try Self.run(model, prompt, coordinator: coordinator, step: step, recorder: recorder)
            #expect(v2.cached == 0, "a v1 entry served a v2 request (reused \(v2.cached))")
        }
    }
}
