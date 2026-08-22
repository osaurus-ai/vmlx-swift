// Copyright © 2026 Osaurus AI. All rights reserved.
//
// Depth sweep that produces the artifact `NativeMTPAutoDecodePolicy` demands.
//
// The launch gate is deliberately fail-closed: `usableBestDepth` returns nil
// unless the bundle's own `vmlx_mtp_tuning.json` carries `validated`,
// `output_equivalent`, `baseline_tok_s`, `best_tok_s` and a `speedup_vs_baseline`
// above 1. Five Qwen3.8-27B bundles ship a `best_depth: 1` stamp whose own note
// says "Conservative UNMEASURED default: 1 draft/step. Run a depth sweep." —
// so they do not launch. This is that sweep. It writes nothing on its own; it
// prints a JSON block to paste into the bundle, because a measurement that the
// runtime will trust must come from a run someone can point at.
//
// Method follows the rules those numbers are held to:
//
//   - ABAB interleaved, never A-then-B. Sequential A/B on a loaded box once
//     fabricated a 29% regression that did not exist.
//   - >= 3 hot rounds at a FIXED condition, MEDIAN reported. One sample is not
//     a finding, and the first round after load is discarded as cold.
//   - Free memory logged before EVERY leg. A timing claim taken while the box
//     is paging is a claim about the box, not the code.
//   - Greedy (temp 0) so `output_equivalent` means something.
//
// Usage:
//   VMLX_MTP_SWEEP_TARGET=~/models/JANGQ-AI/Qwen3.8-27B-JANG_2D \
//   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
//   swift test --filter QwenNativeMTPDepthSweepTests

import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers

@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class QwenNativeMTPDepthSweepTests: XCTestCase {

    private static var targetPath: String? {
        ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_TARGET"]
            .map { NSString(string: $0).expandingTildeInPath }
    }

    /// Depths to sweep. `1` is the shipping question; the others are context,
    /// because "depth 1 is best" is only meaningful against a neighbour.
    private static var depths: [Int] {
        (ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_DEPTHS"] ?? "1,2,3")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    private static var rounds: Int {
        Int(ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_ROUNDS"] ?? "4") ?? 4
    }

    /// Prose, deliberately. A counting/deterministic prompt once produced a
    /// 1.83x MTP figure that did not survive contact with real text — an
    /// artifact of a prompt the drafter could trivially predict.
    /// Prompt class is an axis, not a constant. The JANG_4D artifact records
    /// prose 1.43x and code 1.80x — acceptance differs enough by content that
    /// measuring one and generalising is how a 1.83x MTP number once got
    /// published off a counting prompt.
    private static var promptClass: String {
        ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_PROMPT"]?.lowercased() ?? "prose"
    }

    private static var prompt: String {
        switch promptClass {
        case "long":
            // A realistic long conversation, not a short probe. The 56-token
            // boundary measured earlier stored 124 MB in 27ms; DiskCache's own
            // note records a ~357MB boundary taking ~25s, so the store cost has
            // to be measured at depth before "SSD caching is free" can be said
            // about real chats rather than about short prompts.
            let para = """
                The write-ahead log appends every mutation to durable storage \
                before the corresponding page is modified in the buffer pool, so \
                a crash can always be reconciled by replaying the log forward \
                from the last checkpoint. Recovery proceeds in three phases: \
                analysis rebuilds the dirty page table, redo reapplies committed \
                work, and undo rolls back transactions that never reached a \
                commit record.
                """
            return "Read the following notes, then write a detailed design "
                + "review of the recovery protocol they describe.\n\n"
                + Array(repeating: para, count: Int(ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_LONG_REPEAT"] ?? "40") ?? 40).joined(separator: "\n\n")
                + "\n\nNow write the design review."
        case "code":
            return """
                Write a complete Swift implementation of a thread-safe LRU cache \
                with a fixed capacity. Include the node type, the doubly-linked \
                list operations, get and put, and a brief comment on why the lock \
                is held where it is.
                """
        default:
            return """
                Explain, in three well-developed paragraphs, how a write-ahead log \
                lets a database survive a crash without losing committed \
                transactions. Cover durability, recovery, and the trade-off \
                against write amplification.
                """
        }
    }

    private func freeMemoryGB() -> Double {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        let page = Double(sysconf(_SC_PAGESIZE))
        return Double(stats.free_count) * page / 1_073_741_824.0
    }

    private func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        return s.count % 2 == 1
            ? s[s.count / 2]
            : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    func testDepthSweep() async throws {
        guard let targetPath = Self.targetPath else {
            throw XCTSkip("Set VMLX_MTP_SWEEP_TARGET to a Qwen bundle with an MTP artifact")
        }
        let dir = URL(fileURLWithPath: targetPath)

        let status = try MTPBundleInspector.inspect(modelDirectory: dir)
        guard status.hasCompleteMTPArtifact else {
            throw XCTSkip("\(dir.lastPathComponent) has no complete MTP artifact")
        }

        // The MTP head must be BUILT at load, or every speculative arm dies
        // with "the loaded model has no active MTP head". In the product that
        // flag comes from `resolveNativeMTPLaunchPlan`, which consults the very
        // tuning file this sweep exists to produce — so measuring requires
        // forcing it on here. Measured 2026-08-22 on JANG_2D: without this the
        // baseline ran at 18.66 tok/s and all three MTP arms threw.
        var loadConfiguration = LoadConfiguration.default
        loadConfiguration.nativeMTP = true

        // Cache tier is an axis. A bare `ModelContext` has NO coordinator at
        // all -- `CacheCoordinatorConfig.enableDiskCache` defaults to false and
        // the coordinator normally lives on `ModelContainer`. So a sweep that
        // loads a context and calls `generate` measures the engine with ZERO
        // caching, which is NOT what the app runs. Passing a coordinator
        // explicitly makes "none / paged / disk" a controlled comparison rather
        // than an unstated condition -- and every number measured before this
        // axis existed was a no-cache number.
        // MLX's buffer cache regrows after every clearCache, and on a box that
        // is already sharing RAM with another large process that overhead is
        // what pushes free memory to ~1GB and makes timings untrustworthy
        // (a 16GB model was consuming ~29GB resident). Capping it applies
        // EQUALLY to every arm, so the speedup ratio stays a valid comparison
        // while the run actually fits.
        if let mb = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_CACHE_MB"],
            let bytes = Int(mb).map({ $0 * 1_048_576 })
        {
            MLX.GPU.set(cacheLimit: bytes)
            print("[mtpsweep] MLX cache limit = \(mb) MB")
        }

        let cacheTier = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_CACHE"]?
            .lowercased() ?? "none"
        let (context, _) = try await MLXLMCommon.loadModel(
            from: dir,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: loadConfiguration)
        nonisolated(unsafe) let ctx = context

        let coordinator: CacheCoordinator?
        switch cacheTier {
        case "paged":
            coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    enableDiskCache: false, modelKey: dir.lastPathComponent))
        case "disk":
            coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    enableDiskCache: true, modelKey: dir.lastPathComponent))
        default:
            coordinator = nil
        }
        nonisolated(unsafe) let coord = coordinator
        print("[mtpsweep] cache tier = \(cacheTier)")
        let maxTokens = 256

        // The verifier mode is part of the measurement, not a detail. The
        // shipped JANG_4D artifact records `input_capture_staged`; a sweep that
        // leaves it nil measures a DIFFERENT decode path and would stamp a
        // verifier it never ran.
        let verifier = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_VERIFIER"]
            .flatMap { $0.isEmpty ? nil : $0 }

        // DFlash 2 block sizes, so the same prompt-class and context axes that
        // decided the MTP depths can decide the block size too. The
        // checkpoint's own trained block_size is what ships when the user
        // leaves the setting blank, so "is the default the right value" is a
        // question the harness has to be able to answer.
        let dflashPath = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_DFLASH2"]
            .map { URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath) }
        let dflashBlocks = (ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_DFLASH2_BLOCKS"]
            ?? "4,6,8").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        struct Arm { let name: String; let depth: Int?; let dflashBlock: Int? }
        var arms: [Arm] = [Arm(name: "baseline", depth: nil, dflashBlock: nil)]
        arms += Self.depths.map { Arm(name: "mtp-d\($0)", depth: $0, dflashBlock: nil) }
        if dflashPath != nil {
            arms += dflashBlocks.map { Arm(name: "dflash-b\($0)", depth: nil, dflashBlock: $0) }
        }

        func run(_ arm: Arm) async throws
            -> (text: String, seconds: Double, tokens: Int, mtp: String)
        {
            // Reasoning is an axis too. With thinking on, a large share of the
            // generated tokens are hidden reasoning, and the acceptance the
            // controller reads is acceptance on THAT text — not on the answer
            // the user waits for.
            var context: [String: any Sendable]? = nil
            if let r = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_REASONING"] {
                context = ["enable_thinking": !(["0", "off", "false", "no"].contains(r.lowercased()))]
            }
            // Reasoning EFFORT is its own axis. A prior measurement found b15
            // best on code at effort `medium` while the default effort here is
            // xhigh -- so "which block size wins" may be conditioned on how
            // predictable the model's own output is.
            if let e = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_EFFORT"] {
                var c = context ?? [:]
                c["enable_thinking"] = true
                c["reasoning_effort"] = e
                context = c
            }
            let input = try await ctx.processor.prepare(
                input: UserInput(chat: [.user(Self.prompt)], additionalContext: context))
            var p = GenerateParameters(
                generationConfig: ctx.configuration.generationDefaults,
                fallback: GenerateParameters(maxTokens: maxTokens, prefillStepSize: 1024))
            p.maxTokens = maxTokens
            p.prefillStepSize = 1024
            p.temperature = 0
            p.topP = 1
            p.topK = 0
            p.minP = 0
            if let block = arm.dflashBlock, let dflashPath {
                p.draftStrategy = .dflash2(drafterPath: dflashPath, blockSize: block)
            } else {
                p.draftStrategy = arm.depth.map {
                    DraftStrategy.nativeMTP(depth: $0, verifierMode: verifier)
                }
            }

            var text = ""
            var tokens = 0
            // What actually RAN. Requesting depth 3 does not mean depth 3
            // executed: #283 added adaptive depth (it can downshift mid-run)
            // and #284 fixed a prose first turn locking MTP off for the whole
            // residency. A sweep that records only the REQUESTED depth would
            // attribute an adaptive downshift to the depth setting.
            var mtpNote = "-"
            let start = Date()
            for await item in try MLXLMCommon.generate(
                input: input, parameters: p, context: ctx, cacheCoordinator: coord)
            {
                switch item {
                case .chunk(let c): text += c
                case .reasoning(let c): text += c
                case .info(let info):
                    tokens = info.generationTokenCount
                    if let s = info.nativeMTPStats {
                        mtpNote = String(
                            format: "active=%d/%d commit=%.2f downshifts=%d reason=%@",
                            s.activeDepth, s.depth, s.avgCommittedPerVerify,
                            s.adaptiveDownshifts, s.adaptiveFallbackReason ?? "-")
                    }
                default: break
                }
            }
            return (text, Date().timeIntervalSince(start), tokens, mtpNote)
        }

        // ABAB: every round runs every arm, so drift in machine state hits all
        // arms equally instead of being attributed to whichever ran second.
        var samples: [String: [Double]] = [:]
        var lastText: [String: String] = [:]
        var lastMTP: [String: String] = [:]
        for round in 1...Self.rounds {
            for arm in arms {
                Memory.clearCache()
                let free = freeMemoryGB()
                let r = try await run(arm)
                let tps = Double(r.tokens) / Swift.max(r.seconds, 0.001)
                print(String(
                    format: "[mtpsweep r%d] %-10s free=%5.1fGB %4d tok %7.2fs %7.2f tok/s  %@",
                    round, (arm.name as NSString).utf8String!, free, r.tokens, r.seconds, tps,
                    r.mtp))
                // Round 1 is cold — discard it rather than letting load and
                // first-touch page faults masquerade as decode cost.
                if round > 1 { samples[arm.name, default: []].append(tps) }
                lastText[arm.name] = r.text
                lastMTP[arm.name] = r.mtp
            }
        }

        let baselineTPS = median(samples["baseline"] ?? [])
        XCTAssertGreaterThan(baselineTPS, 0, "baseline produced no samples")
        let baselineText = lastText["baseline"] ?? ""
        XCTAssertFalse(baselineText.isEmpty, "baseline produced no text — the sweep is vacuous")

        print(String(format: "\n[mtpsweep] baseline median %.2f tok/s (n=%d)",
                     baselineTPS, samples["baseline"]?.count ?? 0))

        var best: (depth: Int, tps: Double, equivalent: Bool)?
        for arm in arms where arm.depth != nil || arm.dflashBlock != nil {
            let tps = median(samples[arm.name] ?? [])
            let text = lastText[arm.name] ?? ""
            let equivalent = text == baselineText
            let shared = zip(text, baselineText).prefix { $0 == $1 }.count
            print(String(
                format: "[mtpsweep] %-10s median %.2f tok/s  speedup %.3fx  equivalent=%@ sharedPrefix=%d/%d  %@",
                (arm.name as NSString).utf8String!, tps, tps / Swift.max(baselineTPS, 0.001),
                equivalent ? "YES" : "no", shared, baselineText.count,
                lastMTP[arm.name] ?? "-"))

            // Different mechanisms promise different things, so assert what
            // each one actually guarantees.
            //
            // Native MTP verifies against the target's own logits and is
            // byte-identical on every bundle measured here, so any divergence
            // is a real defect.
            //
            // DFlash 2 does NOT promise byte-equality on a quantized target.
            // Its guarantee is that every emitted token is the verify forward's
            // own argmax; a quantized matmul's reduction order depends on row
            // count, so an M = 1+block verify produces logits a few ulp from
            // the M = 1 decode forward and a greedy near-tie can legitimately
            // flip — at ANY position, including an early one.
            //
            // Measured instance: on JANG_2D the b4 arm diverges at char 76 of
            // 967 because the model wrote "LRU cache with fixed capacity" where
            // the baseline wrote "LRU (Least Recently Used) cache with a fixed
            // capacity", then continued coherently for another ~900 chars
            // (981 vs 967 total). A shared-prefix threshold would call that a
            // broken forward. It is not one — so the DFlash arms are held to
            // the property that actually distinguishes a near-tie from a
            // broken forward: the completion keeps going and stays comparable
            // in length, rather than truncating or degenerating.
            if arm.depth != nil {
                XCTAssertEqual(
                    text, baselineText,
                    "\(arm.name) is native MTP and must be byte-identical to baseline")
            } else {
                XCTAssertFalse(text.isEmpty, "\(arm.name) produced no text")
                XCTAssertGreaterThan(
                    Double(text.count), Double(baselineText.count) * 0.6,
                    "\(arm.name) truncated or degenerated (\(text.count) vs \(baselineText.count) chars)")
            }

            if let dir = ProcessInfo.processInfo.environment["VMLX_MTP_SWEEP_DUMP"] {
                try? baselineText.write(toFile: dir + "/baseline.txt", atomically: true, encoding: .utf8)
                try? text.write(toFile: dir + "/\(arm.name).txt", atomically: true, encoding: .utf8)
            }
            if let d = arm.depth, best == nil || tps > best!.tps {
                best = (d, tps, equivalent)
            }
        }

        guard let best else { return }
        let speedup = best.tps / Swift.max(baselineTPS, 0.001)

        // Print, never write. The gate trusts this file; a sweep that silently
        // stamps its own result would let a bad run enable itself.
        print("""

            [mtpsweep] paste into \(dir.path)/vmlx_mtp_tuning.json —
            {
              "native_mtp": {
                "best_depth": \(best.depth),
                "validated": true,
                "output_equivalent": \(best.equivalent),
                "blocked": false,
                "baseline_tok_s": \(String(format: "%.2f", baselineTPS)),
                "best_tok_s": \(String(format: "%.2f", best.tps)),
                "speedup_vs_baseline": \(String(format: "%.3f", speedup)),
                "artifact": "\(dir.lastPathComponent)",
                "measured_at": "<date>",
                "note": "ABAB, \(Self.rounds) rounds, round 1 discarded as cold, median of the rest, prose prompt, temp 0"
              }
            }

            [mtpsweep] NOTE: speedup \(String(format: "%.3f", speedup))x — \
            the gate refuses anything <= 1.0, so this bundle \
            \(speedup > 1.0 ? "would launch" : "must NOT be stamped as launchable").
            """)
    }
}
