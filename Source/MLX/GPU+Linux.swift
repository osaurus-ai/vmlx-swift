// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Linux has no Metal device, and GPU+Metal.swift is excluded there. Code that sizes its memory
// budget asks `GPU.maxRecommendedWorkingSetBytes()`; on Linux there is no working set to recommend,
// so it answers nil, as it does on a Metal device that reports none.

#if os(Linux)
    public enum GPU {
        public static func maxRecommendedWorkingSetBytes() -> Int? { nil }
    }
#endif
