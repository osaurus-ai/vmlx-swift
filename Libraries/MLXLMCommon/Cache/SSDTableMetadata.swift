// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Header-only accounting for tables the runtime reads from SSD rather than loading as weights.
/// This validates byte spans, not model compatibility or the complete safetensors schema.
public enum SSDTableMetadata {
    public static let ngramTensorMarker = ".ple.ngram_embedding."

    /// An invalid descriptor or overlapping span makes the discount unknown. Callers must then
    /// retain those bytes in the resident estimate. Packed formats are measured by their actual
    /// spans; neither quantization names nor nominal bits per parameter determine this count.
    public static func validatedNGramBytes(
        inHeader header: Data, payloadLength: UInt64? = nil
    ) -> UInt64? {
        guard header.count <= 64 << 20,
            let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any]
        else { return nil }
        var spans: [(start: UInt64, end: UInt64, table: Bool)] = []
        for (name, value) in object where name != "__metadata__" {
            guard let offsets = (value as? [String: Any])?["data_offsets"] as? [Int64],
                offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0]
            else { return nil }
            let start = UInt64(offsets[0])
            let end = UInt64(offsets[1])
            guard payloadLength.map({ end <= $0 }) ?? true else { return nil }
            spans.append((start, end, name.contains(ngramTensorMarker)))
        }
        var end: UInt64 = 0
        var total: UInt64 = 0
        for span in spans.sorted(by: { ($0.start, $0.end) < ($1.start, $1.end) }) {
            guard span.start >= end else { return nil }
            end = span.end
            if span.table {
                let sum = total.addingReportingOverflow(span.end - span.start)
                guard !sum.overflow else { return nil }
                total = sum.partialValue
            }
        }
        return total
    }
}
