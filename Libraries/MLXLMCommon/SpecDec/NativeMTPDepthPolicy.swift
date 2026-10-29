import Foundation

/// Native-MTP draft-depth policy. A selected fixed depth is a ceiling, not permission to explore deeper.
/// Safety may use a shallower depth or AR under either policy.
///
/// No hard draft depth (Eric, 2026-10-19/20: "we never want hard set draft depth"): Adaptive governs the depth of
/// every request. A bundle stamp, tuning sidecar, family default or host default only chooses where Adaptive STARTS
/// (the requested depth). The top of the ladder is the runtime's verifier capability (``runtimeTopRung(model:environment:)``),
/// never a constant chosen by a host. `.adaptive(maximumDepth: n)` with a finite `n` is a diagnostic / benchmark
/// limit; `.fixed` is an explicit user depth choice only.
public enum NativeMTPDepthPolicy: Sendable, Equatable {
    case fixed
    case adaptive(maximumDepth: Int)

    /// Adaptive with no request-level ceiling: the runtime's top rung (``runtimeTopRung(model:environment:)``) is the
    /// only top. The library default and what an app's "On (Adaptive)" sends.
    public static let adaptiveToRuntimeCeiling: NativeMTPDepthPolicy = .adaptive(maximumDepth: .max)

    /// The deepest draft the native-MTP verifier checks row-exactly: every verify row reproduces the AR step bit for
    /// bit for windows of ``FlashVerificationScope/rowExactVerifyRows`` (2...8 rows = depth 1...7). The copy proposer
    /// already verifies 8-row windows on these models (`NativeMTPCopyProposer.maxWidth`).
    public static let verifierTopRung = FlashVerificationScope.rowExactVerifyRows.upperBound - 1

    /// The top rung of a recursive MTP head (a model that declares no other verifier capability): the verifier's
    /// exact window, ``verifierTopRung``. Until 2026-10-29 it was 5, the value the controller was given when it
    /// learned to climb past its start depth (e07b690fc, 2026-09-04; before: 3), a tuning number below what the
    /// verifier checks exactly. Measured fixed depths on Flash-Next 4S (NativeMTPDepthChooser.swift, 2026-10-06)
    /// were still rising at D5 on easy code (96.9 -> 100.1 tok/s); whether D6 / D7 pay is the controller's call.
    public static var recursiveHeadTopRung: Int { verifierTopRung }

    /// The runtime's top rung for `model`, in this order:
    /// 1. `VMLX_MTP_DEPTH_CAP=<n>`: a diagnostic lever (benchmarks); the app never sets it;
    /// 2. ``NativeMTPAdaptiveCeilingModel``: a model whose verifier cannot check deeper drafts exactly (correctness only);
    /// 3. a block drafter (``NativeMTPBlockDraftModel``, LiLiCorr): its deepest proposal (block - 1), within the
    ///    verifier's exact window;
    /// 4. a recursive head: ``recursiveHeadTopRung``.
    public static func runtimeTopRung(
        model: Any, environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let raw = environment["VMLX_MTP_DEPTH_CAP"], let cap = Int(raw) {
            return Swift.max(cap, 1)
        }
        if let declared = (model as? any NativeMTPAdaptiveCeilingModel)?.nativeMTPAdaptiveDepthCeiling {
            return Swift.max(declared, 1)
        }
        if let drafter = model as? any NativeMTPBlockDraftModel, drafter.nativeMTPUsesBlockDrafts {
            return Swift.max(1, Swift.min(drafter.nativeMTPBlockDraftDepthLimit, verifierTopRung))
        }
        return recursiveHeadTopRung
    }

    struct Resolution: Equatable {
        let initialDepth: Int
        let maximumDepth: Int
    }

    enum ValidationError: Error, Equatable {
        case invalidRequestedDepth
        case invalidMaximumDepth
        case invalidRuntimeCap
    }

    func resolve(requestedDepth: Int, runtimeCap: Int) throws -> Resolution {
        guard requestedDepth > 0 else { throw ValidationError.invalidRequestedDepth }
        guard runtimeCap > 0 else { throw ValidationError.invalidRuntimeCap }
        let maximum: Int
        switch self {
        case .fixed:
            maximum = min(requestedDepth, runtimeCap)
        case .adaptive(let ceiling):
            guard ceiling > 0 else { throw ValidationError.invalidMaximumDepth }
            maximum = min(ceiling, runtimeCap)
        }
        return Resolution(initialDepth: min(requestedDepth, maximum), maximumDepth: maximum)
    }
}
