// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Optional conditioning captured with an exact target-cache boundary.
/// Target KV remains independently usable when this acceleration is absent.
struct DFlash2PrefixContext {
    static let metadataKey = "__dflash2_context_descriptor__"
    static let hiddenKey = "__dflash2_context_hidden__"
    private static let version: Int64 = 1

    let hidden: MLXArray
    let end: Int
    let layerIDs: [Int]

    var start: Int { end - hidden.dim(1) }

    init?(hidden: MLXArray, end: Int, layerIDs: [Int]) {
        guard hidden.ndim == 3, hidden.dim(0) == 1,
            hidden.dim(1) > 0, hidden.dim(2) > 0,
            [.float16, .bfloat16, .float32].contains(hidden.dtype),
            end >= hidden.dim(1), !layerIDs.isEmpty,
            Set(layerIDs).count == layerIDs.count,
            layerIDs.allSatisfy({ Int32(exactly: $0) != nil })
        else { return nil }
        self.hidden = hidden
        self.end = end
        self.layerIDs = layerIDs
    }

    func trailing(_ limit: Int?) -> Self? {
        guard let limit else { return self }
        guard limit > 0 else { return nil }
        guard hidden.dim(1) > limit else { return self }
        return Self(
            hidden: hidden[0..., (hidden.dim(1) - limit)..., 0...],
            end: end, layerIDs: layerIDs)
    }

    /// Join only contiguous features from the same ordered capture taps.
    /// A missing earlier window must never be represented by synthetic rows.
    static func appending(
        _ suffix: MLXArray, endingAt end: Int, layerIDs: [Int],
        to previous: Self?, limit: Int?
    ) -> Self? {
        guard let next = Self(hidden: suffix, end: end, layerIDs: layerIDs) else { return nil }
        guard let previous else { return next.trailing(limit) }
        guard previous.end == next.start, previous.layerIDs == layerIDs,
            previous.hidden.dtype == suffix.dtype,
            previous.hidden.dim(2) == suffix.dim(2)
        else { return nil }
        return Self(
            hidden: concatenated([previous.hidden, suffix], axis: 1),
            end: end, layerIDs: layerIDs)?.trailing(limit)
    }

    private var hasFiniteValues: Bool {
        // Reject malformed auxiliary data rather than poisoning a valid KV hit.
        all(hidden .== hidden).item(Bool.self)
            && MLX.max(abs(hidden.asType(.float32))).item(Float.self).isFinite
    }

    /// Omit optional features if they cannot fit; callers still store base KV.
    func diskArrays(at boundary: Int, byteBudget: Int64) -> [String: MLXArray]? {
        guard end == boundary, Int64(exactly: end) != nil,
            Int64(exactly: start) != nil
        else { return nil }
        let descriptor = [Self.version, Int64(start), Int64(end), Int64(hidden.dim(2))]
            + layerIDs.map(Int64.init)
        let descriptorBytes = descriptor.count.multipliedReportingOverflow(by: 8)
        guard !descriptorBytes.overflow,
            let hiddenBytes = Int64(exactly: hidden.nbytes),
            let metadataBytes = Int64(exactly: descriptorBytes.partialValue)
        else { return nil }
        let bytes = hiddenBytes.addingReportingOverflow(metadataBytes)
        guard !bytes.overflow, byteBudget >= bytes.partialValue, hasFiniteValues else { return nil }
        return [Self.metadataKey: MLXArray(descriptor), Self.hiddenKey: hidden]
    }

    static func restore(
        from arrays: [String: MLXArray], boundary: Int,
        layerIDs: [Int], featureWidth: Int, limit: Int?
    ) -> Self? {
        guard let metadata = arrays[metadataKey], let hidden = arrays[hiddenKey],
            metadata.dtype == .int64, metadata.shape == [4 + layerIDs.count],
            !layerIDs.isEmpty, featureWidth > 0,
            hidden.ndim == 3, hidden.dim(0) == 1,
            hidden.dim(2) == featureWidth,
            [.float16, .bfloat16, .float32].contains(hidden.dtype)
        else { return nil }
        let descriptor = metadata.asArray(Int64.self)
        guard descriptor[0] == version,
            let start = Int(exactly: descriptor[1]), let end = Int(exactly: descriptor[2]),
            start >= 0, end == boundary, end > start,
            end - start == hidden.dim(1), descriptor[3] == Int64(featureWidth),
            descriptor.dropFirst(4).elementsEqual(layerIDs.map(Int64.init)),
            let context = Self(hidden: hidden, end: end, layerIDs: layerIDs),
            context.hasFiniteValues
        else { return nil }
        return context.trailing(limit)
    }
}
