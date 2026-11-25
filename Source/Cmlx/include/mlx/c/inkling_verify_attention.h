// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
#ifndef VMLX_INKLING_VERIFY_ATTENTION_H
#define VMLX_INKLING_VERIFY_ATTENTION_H

#include <stdbool.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Build a lazy shared-KV verify operation. Returns false for unsupported layouts or devices.
bool vmlx_inkling_verify_attention(
    mlx_array* output, mlx_array query, mlx_array keys, mlx_array values,
    mlx_array bias, mlx_array sinks, float scale, bool sliding, mlx_stream stream);

#ifdef __cplusplus
}
#endif
#endif
