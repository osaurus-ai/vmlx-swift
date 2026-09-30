// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    /// The largest error, in units (see `unitsError`), of each of MLX's polynomial functions on its
    /// scalar CPU code, measured on Linux arm64 without Highway kernels. A Highway build may exceed
    /// it by 2 units, and no more. To measure it again, on Linux arm64, run
    /// `VMLX_NO_HIGHWAY=1 VMLX_ULP_BASELINE=1 swift test -c release -Xswiftc -enable-testing
    /// --scratch-path .build/no-hwy --filter PolynomialAccuracyTests`: each test prints
    /// `ULP_BASELINE <name> <units>` and fails, since it judges nothing.
    /// After `VMLX_NO_HIGHWAY=1 scripts/run-cpu-kernel-tests.sh` has built `.build/no-hwy`, add
    /// `--skip-build`: without it, swift test recompiles the Swift modules for testability.
    /// scripts/run-cpu-kernel-tests.sh unsets VMLX_ULP_BASELINE.
    enum ULPBaseline {
        static let units: [String: Double] = [
            "exp": 62.385592963546515,
            "sin": 1.4671259745955467,
            "cos": 1.4473254270851612,
            "sinWide": 1.45855319686234,
            "cosWide": 1.4345570486038923,
            "erf": 8.121505227230955,
            "erfInverse": 2.34520098939538,
            "sigmoid": 7.33725181221962,
            "logAddExp": 24.894253134727478,
        ]
    }
#endif
