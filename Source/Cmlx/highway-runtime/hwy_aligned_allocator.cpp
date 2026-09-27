// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Google Highway's runtime, for builds with Highway kernels (Linux x86-64). The
// Highway submodule itself is excluded from Cmlx, so only this source compiles.
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "hwy/aligned_allocator.cc"
#endif
