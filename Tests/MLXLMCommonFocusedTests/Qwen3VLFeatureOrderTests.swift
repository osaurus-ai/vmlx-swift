// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pins the Qwen3VL vision-feature reordering: feature rows arrive
// image-batch-first (the order `prepare` concatenates pixels), while
// placeholders sit in conversation order. The permutation must map each
// placeholder slot to the next row of ITS kind — the pooled scatter and
// every per-layer deepstack application then land each block on its own
// medium's pads. Video-before-image across turns is the exact shape that
// silently swapped the two blocks (same defect fixed in Qwen35.swift).

import Testing

@testable import MLXVLM

struct Qwen3VLFeatureOrderTests {

    @Test("video placeholders before image placeholders pull video rows first")
    func videoFirstConversationOrder() {
        // Conversation: video turn (pads at 10,11,12) then image turn
        // (pads at 20,21). Rows: image batch = [0,1], video batch = [2,3,4].
        let perm = Qwen3VL.placeholderOrderPermutation(
            imagePositions: [20, 21],
            videoPositions: [10, 11, 12],
            imageRowCount: 2)
        // Slots 0-2 are the video pads -> video rows 2,3,4;
        // slots 3-4 are the image pads -> image rows 0,1.
        #expect(perm == [2, 3, 4, 0, 1])
    }

    @Test("image-first conversation is the identity permutation")
    func imageFirstIsIdentity() {
        let perm = Qwen3VL.placeholderOrderPermutation(
            imagePositions: [5, 6],
            videoPositions: [30, 31, 32],
            imageRowCount: 2)
        #expect(perm == [0, 1, 2, 3, 4])
    }

    @Test("interleaved kinds consume each queue in placeholder order")
    func interleavedKinds() {
        // img, vid, img, vid conversation order.
        let perm = Qwen3VL.placeholderOrderPermutation(
            imagePositions: [1, 8],
            videoPositions: [4, 12],
            imageRowCount: 2)
        #expect(perm == [0, 2, 1, 3])
    }
}
