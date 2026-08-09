// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// The MoE modules live on the shared model object — one weight set serves every
/// concurrent request — so a mutable field on one of them is as shared as a
/// global. `DeepseekV4MoE` used to park the current request's token ids in a
/// `currentInputIds` field that the decoder layer set immediately before the
/// call and cleared immediately after. On a hash-routed layer the gate ignores
/// the hidden states and picks experts purely by token id, so a concurrent write
/// routed one request's activations through another request's experts — finite,
/// in-range, and silently wrong. `num_hash_layers = 3` puts that in layers 0-2,
/// from where it propagates through all 43.
@Suite("DSV4 MoE token ids stay request-scoped", .serialized)
struct DeepseekV4MoERequestScopedIdsTests {

    /// Small enough to build in milliseconds, still hash-routed at layer 0.
    private static func tinyConfig() -> DeepseekV4Configuration {
        var c = DeepseekV4Configuration()
        c.vocabSize = 32
        c.hiddenSize = 8
        c.numHiddenLayers = 2
        c.nRoutedExperts = 8
        c.numExpertsPerTok = 2
        c.moeIntermediateSize = 8
        c.numHashLayers = 1
        return c
    }

    /// Every token id must land on a distinct expert pair, otherwise a swapped
    /// id would be indistinguishable from the right one and the test could not
    /// see the bug it exists to catch.
    private static func makeHashGate(_ c: DeepseekV4Configuration) -> DeepseekV4MoEGate {
        let gate = DeepseekV4MoEGate(config: c, layerIdx: 0)
        #expect(gate.isHashLayer, "layer 0 must be hash-routed for this test to mean anything")
        let table = (0..<c.vocabSize).flatMap { t in
            [Int32(t % c.nRoutedExperts), Int32((t * 3 + 1) % c.nRoutedExperts)]
        }
        gate.update(parameters: ModuleParameters.unflattened([
            "tid2eid": MLXArray(table, [c.vocabSize, c.numExpertsPerTok]),
            "weight": MLXRandom.normal([c.nRoutedExperts, c.hiddenSize], key: MLXRandom.key(7)),
        ]))
        return gate
    }

    private static func ints(_ a: MLXArray) -> [Int32] {
        a.asType(.int32).asArray(Int32.self)
    }

    private static func floats(_ a: MLXArray) -> [Float] {
        a.asType(.float32).asArray(Float.self)
    }

    @Test("hash routing is decided by the ids passed to the call, not by module state")
    func hashRoutingFollowsTheArgument() {
        let c = Self.tinyConfig()
        let gate = Self.makeHashGate(c)
        let x = MLXRandom.normal([1, 3, c.hiddenSize], key: MLXRandom.key(11))

        let idsA = MLXArray([Int32(0), 1, 2], [1, 3])
        let idsB = MLXArray([Int32(17), 23, 29], [1, 3])

        let (indA, _) = gate(x, inputIds: idsA)
        let (indB, _) = gate(x, inputIds: idsB)

        // Same activations, different ids -> different experts. This is what
        // makes a cross-request clobber corrupting rather than harmless.
        #expect(
            Self.ints(indA) != Self.ints(indB),
            "hash routing ignored the ids; the rest of this suite proves nothing")

        // A third party's call in between must not disturb the first caller.
        let (indNil, _) = gate(x, inputIds: nil)
        let (indAAgain, _) = gate(x, inputIds: idsA)
        #expect(Self.ints(indA) == Self.ints(indAAgain))

        // nil takes the non-hash softmax gate — the exact wrong-expert set a
        // premature clear used to produce.
        #expect(Self.ints(indNil) != Self.ints(indA))
    }

    @Test("MoE forward is a pure function of (x, inputIds) across interleaved calls")
    func moeForwardCarriesNoStateBetweenCalls() {
        let c = Self.tinyConfig()
        let moe = DeepseekV4MoE(config: c, layerIdx: 0)
        let table = (0..<c.vocabSize).flatMap { t in
            [Int32(t % c.nRoutedExperts), Int32((t * 3 + 1) % c.nRoutedExperts)]
        }
        moe.gate.update(parameters: ModuleParameters.unflattened([
            "tid2eid": MLXArray(table, [c.vocabSize, c.numExpertsPerTok])
        ]))

        // L > 1 keeps this on the eager path, which is where the field lived.
        let x = MLXRandom.normal([1, 3, c.hiddenSize], key: MLXRandom.key(13))
        let idsA = MLXArray([Int32(0), 1, 2], [1, 3])
        let idsB = MLXArray([Int32(17), 23, 29], [1, 3])

        let first = Self.floats(moe(x, inputIds: idsA))
        _ = moe(x, inputIds: idsB)
        _ = moe(x, inputIds: nil)
        _ = moe(x)
        let second = Self.floats(moe(x, inputIds: idsA))

        #expect(first == second)

        // The `UnaryLayer` shim must mean "no ids", i.e. the non-hash gate —
        // never "whatever ids were last seen".
        #expect(Self.floats(moe(x)) == Self.floats(moe(x, inputIds: nil)))
    }

    @Test("concurrent callers with different ids do not see each other's routing")
    func concurrentCallersAreIsolated() {
        let c = Self.tinyConfig()
        let gate = Self.makeHashGate(c)
        let x = MLXRandom.normal([1, 3, c.hiddenSize], key: MLXRandom.key(17))

        let lanes: [MLXArray] = (0..<8).map { (lane: Int) -> MLXArray in
            let ids: [Int32] = [Int32(lane), Int32(lane + 8), Int32(lane + 16)]
            return MLXArray(ids, [1, 3])
        }
        let reference = lanes.map { Self.ints(gate(x, inputIds: $0).0) }

        let lock = NSLock()
        nonisolated(unsafe) var mismatches = 0
        DispatchQueue.concurrentPerform(iterations: lanes.count * 8) { i in
            let lane = i % lanes.count
            let got = Self.ints(gate(x, inputIds: lanes[lane]).0)
            if got != reference[lane] {
                lock.lock()
                mismatches += 1
                lock.unlock()
            }
        }
        #expect(mismatches == 0)
    }

    // MARK: - Source invariants
    //
    // The behavioural tests above go red only when a race actually lands, and a
    // race that lands 20% of the time in production lands ~never in a unit test.
    // These pin the shape of the fix directly so reintroducing the pattern is a
    // deterministic failure.

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// Lines of a top-level `class`/`final class` body, each paired with the
    /// brace depth it sits at relative to the class. Depth 1 is a stored
    /// property; anything deeper is a local inside a member and irrelevant here.
    private static func classMembers(_ src: String, _ name: String) -> [(String, Int)] {
        let lines = src.components(separatedBy: "\n")
        // The delimiter matters: a bare prefix match on "DeepseekV4MoE" lands on
        // `DeepseekV4MoEGate`, which declares its weights via `@ParameterInfo`
        // on the same line and so passes this guard vacuously.
        func declares(_ line: String) -> Bool {
            for prefix in ["class \(name)", "final class \(name)"] {
                guard line.hasPrefix(prefix) else { continue }
                let rest = line.dropFirst(prefix.count).first
                if rest == nil || rest == ":" || rest == " " || rest == "<" { return true }
            }
            return false
        }
        guard let start = lines.firstIndex(where: declares) else {
            Issue.record("class \(name) not found — this guard is scanning nothing")
            return []
        }
        // Depth recorded is the depth *before* the line's own braces, so the
        // `class …  {` header itself is depth 0 and its properties depth 1.
        var members: [(String, Int)] = []
        var depth = 0
        for line in lines[start...] {
            let opens = line.filter { $0 == "{" }.count
            let closes = line.filter { $0 == "}" }.count
            members.append((line, depth))
            depth += opens - closes
            if depth == 0 && members.count > 1 { break }
        }
        return members
    }

    @Test("neither MoE module stores request-scoped MLXArray state")
    func moeModulesHoldNoMutableArrayState() throws {
        for (path, name) in [
            ("Libraries/MLXLLM/Models/DeepseekV4.swift", "DeepseekV4MoE"),
            ("Libraries/MLXLLM/Models/DeepseekV4JANGTQ.swift", "DeepseekV4MoEJANGTQ"),
        ] {
            let members = Self.classMembers(try Self.source(path), name)
            #expect(members.count > 10, "\(name) body did not parse")
            let body = members.map(\.0).joined(separator: "\n")
            let offenders = members.filter { line, depth in
                guard depth == 1 else { return false }
                let t = line.trimmingCharacters(in: .whitespaces)
                guard t.hasPrefix("var ") else { return false }
                return t.contains(": MLXArray") || t.contains(": MLX.MLXArray")
            }.map(\.0)
            #expect(
                offenders.isEmpty,
                """
                \(name) declares mutable MLXArray state: \(offenders).
                This module is shared by every concurrent request; request-scoped
                values must be call arguments. See the `currentInputIds` soup bug.
                """)
            #expect(body.contains("currentInputIds") == false)
        }
    }

    @Test("token ids reach both MoE modules as a call argument")
    func decoderLayersThreadIdsAsArguments() throws {
        let dsv4 = try Self.source("Libraries/MLXLLM/Models/DeepseekV4.swift")
        let jangtq = try Self.source("Libraries/MLXLLM/Models/DeepseekV4JANGTQ.swift")

        for src in [dsv4, jangtq] {
            #expect(src.contains("func callAsFunction(_ x: MLXArray, inputIds: MLXArray?)"))
            // The decoder layer must pass them down, not assign them onto `mlp`.
            #expect(src.contains("mlp(") && src.contains("inputIds: inputIds"))
            #expect(src.range(of: #"mlp\.\w+\s*=\s*inputIds"#, options: .regularExpression) == nil)
        }
    }
}
