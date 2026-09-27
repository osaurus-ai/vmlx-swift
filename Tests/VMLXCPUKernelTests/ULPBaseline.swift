// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    /// The largest error, in units (see `unitsError`), of each of MLX's polynomial functions on the
    /// scalar CPU code of osaurus-ai/mlx 866e78697 (arm64, no Highway), measured by plan 1's Task 24. A
    /// Highway build may exceed it by 2 units, and no more.
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
