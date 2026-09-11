// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import VMLXANEBridge

/// A host-mapped fp16 plane shared with the Neural Engine.
///
/// The plane is an IOSurface the ANE reads or writes directly; `pointer`
/// is its base address for the life of the plane. Width (innermost dim) of
/// an fp16 plane must be a multiple of 32 — see `vmlx_ane_bridge.h`.
public final class ANEPlane {
    let raw: OpaquePointer
    public let byteCount: Int

    public init?(byteCount: Int) {
        guard let raw = vmlx_ane_plane_create(byteCount) else { return nil }
        self.raw = raw
        self.byteCount = Int(vmlx_ane_plane_bytes(raw))
    }

    deinit { vmlx_ane_plane_free(raw) }

    public var pointer: UnsafeMutableRawPointer { vmlx_ane_plane_base(raw)! }

    public var fp16: UnsafeMutablePointer<Float16> {
        pointer.assumingMemoryBound(to: Float16.self)
    }

    public func zero() { memset(pointer, 0, byteCount) }
}

public enum ANEProgramError: Error, CustomStringConvertible {
    case unavailable
    case create(String)
    case eval(String)

    public var description: String {
        switch self {
        case .unavailable: return "the Neural Engine bridge is unavailable on this machine"
        case .create(let why): return "ANE program create failed: \(why)"
        case .eval(let why): return "ANE eval failed: \(why)"
        }
    }
}

/// One compiled-and-loaded ANE program with bound I/O planes.
///
/// Evals are synchronous and strictly serial per program; the caller owns
/// that serialization (the drafter runs them from one thread).
public final class ANEProgram {
    private let handle: OpaquePointer
    public let inputs: [ANEPlane]
    public let outputs: [ANEPlane]
    public let compileSeconds: Double
    public let cacheHit: Bool

    public static var isAvailable: Bool { vmlx_ane_available() != 0 }

    /// Default persistent compile cache. `nil` disables caching.
    public static let cacheDirectory: URL? = {
        if let env = ProcessInfo.processInfo.environment["VMLX_ANE_CACHE_DIR"] {
            return env.isEmpty ? nil : URL(fileURLWithPath: env)
        }
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        return base?.appendingPathComponent("vmlx/ane", isDirectory: true)
    }()

    public init(
        name: String,
        mil: String,
        weights: Data,
        inputs: [ANEPlane],
        outputs: [ANEPlane],
        procedureCount: Int = 1,
        cacheDirectory: URL? = ANEProgram.cacheDirectory
    ) throws {
        guard ANEProgram.isAvailable else { throw ANEProgramError.unavailable }
        if let cacheDirectory {
            try? FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        var error = [CChar](repeating: 0, count: 1024)
        var inRaw = inputs.map { Optional($0.raw) }
        var outRaw = outputs.map { Optional($0.raw) }
        let created: OpaquePointer? = weights.withUnsafeBytes { wb in
            inRaw.withUnsafeMutableBufferPointer { ib in
                outRaw.withUnsafeMutableBufferPointer { ob in
                    vmlx_ane_model_create(
                        name, mil, wb.baseAddress, wb.count,
                        ib.baseAddress, UInt32(ib.count),
                        ob.baseAddress, UInt32(ob.count),
                        UInt32(procedureCount),
                        cacheDirectory?.path,
                        &error, error.count)
                }
            }
        }
        guard let created else {
            throw ANEProgramError.create(String(cString: error))
        }
        self.handle = created
        self.inputs = inputs
        self.outputs = outputs
        self.compileSeconds = vmlx_ane_model_compile_seconds(created)
        self.cacheHit = vmlx_ane_model_cache_hit(created)
    }

    deinit { vmlx_ane_model_free(handle) }

    /// Runs the compiled program once (an ANE evaluation, not code
    /// execution); the bound planes are the inputs and outputs.
    public func eval(procedure: Int = 0) throws {
        var error = [CChar](repeating: 0, count: 1024)
        guard vmlx_ane_model_eval(handle, UInt32(procedure), &error, error.count) != 0 else {
            throw ANEProgramError.eval(String(cString: error))
        }
    }
}
