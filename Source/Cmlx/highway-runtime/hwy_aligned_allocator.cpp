// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Google Highway's runtime for builds with Highway kernels: the hwy library's
// sources less its benchmarking and profiling tools (nanobenchmark,
// perf_counters, profiler). Cmlx excludes the submodule; these compile it.
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "hwy/aligned_allocator.cc"
#endif
