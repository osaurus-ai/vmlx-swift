
/// A selected fixed depth is a ceiling, not permission to explore deeper.
/// Safety may use a shallower depth or AR under either policy.
public enum NativeMTPDepthPolicy: Sendable, Equatable {
    case fixed
    case adaptive(maximumDepth: Int)

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
