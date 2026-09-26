// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

/// Waits for the work already queued on the stream this build computes on.
///
/// The cache paths call this before they read or write tensors on disk. Apple builds synchronize
/// the GPU stream, exactly as those call sites did before. A Linux build without a GPU backend has
/// no GPU stream, and asking for one aborts ("Cannot make gpu stream without gpu backend"), so
/// Linux synchronizes the default device's stream instead.
func synchronizeComputeStream() {
    #if os(Linux)
        Stream.defaultStream(Device.defaultDevice()).synchronize()
    #else
        // METAL-ONLY: case 3. `Stream.gpu` is the Metal stream on Apple platforms, which is why the
        // cache paths call this helper rather than `Stream.gpu` directly. Any other build must use
        // the default device's stream, as the Linux branch does: without a GPU backend `Stream.gpu`
        // aborts, and with a CPU default device it is not the stream the work ran on.
        Stream.gpu.synchronize()
    #endif
}
