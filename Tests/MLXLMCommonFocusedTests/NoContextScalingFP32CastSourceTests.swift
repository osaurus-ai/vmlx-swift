// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Source guard: no model runtime may cast cached keys/values (or any other
// full-context tensor consumed per decode step) to float32.
//
// The pattern this pins against is inherited from upstream mlx-swift /
// mlx-lm ports: an `asType(.float32)` on post-cache-update K/V materializes
// a full-context fp32 copy of the KV cache on EVERY decode token (~5x the
// native-dtype traffic) and was measured as the dominant long-context decode
// cost for every MLA family (Raptor 8B-A1B 173 -> 35 tok/s from 0.4k -> 26k
// ctx; fixed in #401 with a 2.4x recovery at 26k) plus the DSA/GLM5/QSA
// indexer variants of the same tax (#401/#405).
//
// The guard is an allowlist, not a ban: three sites keep fp32 deliberately,
// each with an in-code comment explaining why removing it is a CORRECTNESS
// regression, not an optimization:
//   - Phi.swift: fp16 score range (+-65504) protection, upstream-mirrored.
//   - Gemma4.swift (VLM): bf16 score STORAGE loses winner margins at its
//     d=512/scale-1.0 regime (measured maxDiff 0.27 vs fp32); needs an
//     mlx-fork sdpa_vector kernel, not a Swift-side dtype change.
//   - AttentionUtils.swift: the env-gated MLA fallback arm itself.
// A new legitimate exception must be added HERE with the same justification
// standard, not slipped past the guard.

import Foundation
import Testing

@Suite("No context-scaling fp32 casts in model runtimes")
struct NoContextScalingFP32CastSourceTests {

    private static func repoRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 { url.deleteLastPathComponent() }
        return url
    }

    /// Files allowed to contain `asType(.float32)` applied to a
    /// cache-consumption identifier, with the reason pinned above.
    private static let allowedFiles: Set<String> = [
        "Phi.swift",
        "Gemma4.swift",  // MLXVLM — global-layer upcast, load-bearing
        "AttentionUtils.swift",  // env-gated fp32 fallback arm
        "DeepseekV4Compressor.swift",  // VMLX_DSA_INDEX_FP32 fallback arm only
        "Glm5Next.swift",  // VMLX_GLM5_INDEX_FP32 fallback arm only
    ]

    /// Identifiers that name post-cache-update, context-length tensors in
    /// the model runtimes. A fp32 cast applied directly to one of these is
    /// the exact tax pattern this guard exists to block.
    private static let cacheIdentifiers = [
        "cachedKeys", "cachedValues",
        "cK", "cV",
        "allKeys",
        "pooledBroad", "poolKeys", "groupedKeys",
    ]

    @Test("model sources never cast cached K/V or pooled history to fp32")
    func noCacheFP32Casts() throws {
        let fm = FileManager.default
        let root = Self.repoRoot()
        var offenders: [String] = []
        for dir in ["Libraries/MLXLLM/Models", "Libraries/MLXVLM/Models"] {
            let base = root.appendingPathComponent(dir)
            guard let files = try? fm.contentsOfDirectory(atPath: base.path) else { continue }
            for file in files where file.hasSuffix(".swift") {
                if Self.allowedFiles.contains(file) { continue }
                let path = base.appendingPathComponent(file).path
                guard let source = try? String(contentsOfFile: path, encoding: .utf8) else {
                    continue
                }
                for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
                    guard line.contains("asType(.float32)") else { continue }
                    for ident in Self.cacheIdentifiers
                    where line.contains("\(ident).asType(.float32)") {
                        offenders.append("\(dir)/\(file): \(line.trimmingCharacters(in: .whitespaces))")
                    }
                }
            }
        }
        #expect(
            offenders.isEmpty,
            Comment(rawValue:
                "fp32 cast on a cache-consumption tensor (the long-context decode "
                + "tax fixed in #401/#405). Either run in the cache's native dtype "
                + "or document a correctness justification and add the file to the "
                + "allowlist in this guard:\n" + offenders.joined(separator: "\n")))
    }
}
