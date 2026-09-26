// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
#include "JACCLBuildAvailability.h"
#if VMLX_BUILD_JACCL
#include "../mlx/mlx/distributed/jaccl/jaccl.cpp"
#else
#include "../mlx/mlx/distributed/jaccl/no_jaccl.cpp"
#endif
