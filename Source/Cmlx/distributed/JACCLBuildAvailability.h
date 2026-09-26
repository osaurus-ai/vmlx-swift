// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

// Keep older SDKs, iOS and non-Apple builds on MLX's upstream stub. The
// real implementation dynamically loads librdma; compiling it does not
// imply that RDMA devices, a connected peer or collectives are ready.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)
#define VMLX_BUILD_JACCL 1
#else
#define VMLX_BUILD_JACCL 0
#endif
#else
#define VMLX_BUILD_JACCL 0
#endif
