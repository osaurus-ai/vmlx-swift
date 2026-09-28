// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// Google Highway's runtime, for builds with Highway kernels (Linux x86-64). The
// Highway submodule itself is excluded from Cmlx, so only this source compiles.
#if defined(MLX_USE_HIGHWAY_KERNELS)
// Package.swift's #if arch(x86_64) tests the host, so a cross-compilation from
// x86-64 could define the macro for a target mlx refuses Highway for.
#if !defined(__x86_64__) && !defined(_M_X64)
#error "MLX_USE_HIGHWAY_KERNELS requires an x86-64 target"
#endif
#include "hwy/targets.cc"
#endif
