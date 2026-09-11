// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// A native-MTP model whose head the ANE emitter knows how to build.
public protocol ANEDraftableModel: NativeMTPModel {
    /// nil when this instance's head is not a supported shape.
    func aneHeadWeightSource(draftVocab: Int, window: Int) -> ANEHeadWeightSource?
}

/// Runtime switch for the ANE drafter. Experimental: off unless
/// `VMLX_ANE_MTP=1`. Greedy drafting only (the ANE returns argmax ids, not
/// draft distributions), so a sampled request keeps the GPU head.
public enum ANEMTPSettings {
    public static var enabled: Bool {
        ProcessInfo.processInfo.environment["VMLX_ANE_MTP"] == "1"
    }
    /// lm_head rows carried on the ANE (ids `0 ..< draftVocab`).
    public static var draftVocab: Int {
        Int(ProcessInfo.processInfo.environment["VMLX_ANE_MTP_VOCAB"] ?? "") ?? 32768
    }
    /// Head K/V window in positions.
    public static var window: Int {
        Int(ProcessInfo.processInfo.environment["VMLX_ANE_MTP_WINDOW"] ?? "") ?? 1024
    }
}

/// The `makeDrafts` seam served by the Neural Engine.
///
/// Owns an `ANEHeadRunner` and mirrors the iterator's head-cache lifecycle:
/// the aligned commit rows ride in the same tile eval as the first draft,
/// deeper levels append speculative rows, `trim(rows:)` drops them before
/// the next commit, `reset()` is the cache refresh.
public final class ANEMTPDrafter {
    public let runner: ANEHeadRunner
    public private(set) var forwardCount = 0
    public private(set) var buildSeconds: Double = 0

    public init(model: any ANEDraftableModel, draftVocab: Int = ANEMTPSettings.draftVocab,
                window: Int = ANEMTPSettings.window) throws {
        let start = Date.timeIntervalSinceReferenceDate
        guard let source = model.aneHeadWeightSource(draftVocab: draftVocab, window: window) else {
            throw ANEProgramError.create("this model's MTP head is not a shape the ANE emitter supports")
        }
        runner = try ANEHeadRunner(source: source)
        buildSeconds = Date.timeIntervalSinceReferenceDate - start
    }

    public var length: Int { runner.length }
    public func reset() { runner.reset() }
    public func trim(rows: Int) {
        guard rows > 0 else { return }
        runner.truncate(to: Swift.max(0, runner.length - rows))
    }

    /// Drafts `depth` tokens. `hidden` is `[1, n, H]` and `nextTokens`
    /// `[1, n]` (n ≥ 1): the first n−1 pairs are committed rows, the last
    /// pair is the chain's seed — exactly what the GPU `makeDrafts` gets.
    public func drafts(hidden: MLXArray, nextTokens: MLXArray, depth: Int) throws -> [Int32] {
        let n = nextTokens.dim(-1)
        precondition(hidden.dim(-2) == n, "hidden rows \(hidden.dim(-2)) != tokens \(n)")
        let h = hidden.asType(.float16).reshaped(n, -1)
        let H = h.dim(1)
        let flat = h.asArray(Float16.self)
        let ids = nextTokens.reshaped(-1).asArray(Int32.self)
        var pairs: [(hidden: [Float16], token: Int)] = []
        pairs.reserveCapacity(n)
        for r in 0 ..< n {
            pairs.append((Array(flat[(r * H) ..< ((r + 1) * H)]), Int(ids[r])))
        }
        var out: [Int32] = []
        out.reserveCapacity(depth)
        var step = try runner.draftChainStart(pairs: pairs)
        forwardCount += 1
        out.append(Int32(step.token))
        while out.count < depth {
            step = try runner.draftStep(hidden: step.hidden, token: step.token)
            forwardCount += 1
            out.append(Int32(step.token))
        }
        return out
    }
}
