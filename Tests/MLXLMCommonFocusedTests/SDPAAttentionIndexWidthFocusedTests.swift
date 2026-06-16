// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("SDPA attention index width contracts")
struct SDPAAttentionIndexWidthFocusedTests {
    @Test("NAX attention keeps causal and mask sequence positions wider than int16")
    func naxAttentionSequencePositionsStayIntWidth() throws {
        let files = [
            "Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h",
            "Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention_nax.h",
            "Source/Cmlx/mlx-generated/steel_attention_nax.cpp",
        ]

        let forbiddenPatterns = [
            "const short row_pos = base_row",
            "const short col_pos = base_col",
            "const short lim_rows_q = params->qL_rem",
            "const short lim_rows_k = params->kL_rem",
        ]

        for file in files {
            let source = try Self.source(file)
            for pattern in forbiddenPatterns {
                #expect(!source.contains(pattern), "\(file) must not narrow long sequence positions via `\(pattern)`")
            }
        }
    }

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }
}
