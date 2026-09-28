// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// The C++ core's CPU backend controls, for Swift: Highway targets, the int8
// quantized-matmul switch, the thread pool and per-family dispatch counters.
// Builds without Highway kernels (macOS; Linux without MLX_USE_HIGHWAY_KERNELS)
// answer as such a build: no targets, one thread, no counts. Their int8 switch
// is the core's own on Linux, where it changes nothing, and off on macOS.

#ifndef VMLX_CMLX_CPU_SHIM_H
#define VMLX_CMLX_CPU_SHIM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kernel families, in the order of mlx::core::cpu::highway_info::Family, which
// the shim checks enumerator by enumerator.
typedef enum {
  VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT = 0,
  VMLX_CPU_FAMILY_QMM_AFFINE_INT8,
  VMLX_CPU_FAMILY_QMM_FP,
  VMLX_CPU_FAMILY_RMS_NORM,
  VMLX_CPU_FAMILY_LAYER_NORM,
  VMLX_CPU_FAMILY_ROPE,
  VMLX_CPU_FAMILY_SDPA,
  VMLX_CPU_FAMILY_COUNT
} vmlx_cpu_family;

// Whether this build compiled the CPU backend's Highway kernels.
bool vmlx_cpu_highway_enabled(void);
// Highway target bits (HWY_* values) compiled into the dispatched kernels, and
// supported by this CPU as dispatch sees it (after a test restriction).
int64_t vmlx_cpu_highway_compiled_targets(void);
int64_t vmlx_cpu_highway_supported_targets(void);
// Restricts dispatch to `targets`; 0 restores the CPU's own set. Refuses a set
// with a target this CPU does not support, which would fault: returns false,
// and leaves dispatch unrestricted. For tests: process-wide, and never while
// kernels run.
bool vmlx_cpu_highway_set_targets_for_test(int64_t targets);
// Highway's name for one target bit, such as "AVX2"; "Unknown" for a bit it
// does not know, and "" in a build without Highway.
const char* vmlx_cpu_highway_target_name(int64_t target);

// The int8 quantized-matmul switch (off unless set).
bool vmlx_cpu_quantized_int8(void);
void vmlx_cpu_set_quantized_int8(bool enabled);

// The CPU pool: its size, why ("MLX_CPU_THREADS", "cgroup cpu.max", "physical
// cores", or "no pool"), an MLX_CPU_THREADS error ("" if none), and whether
// OpenBLAS was pinned to one thread.
int32_t vmlx_cpu_thread_count(void);
const char* vmlx_cpu_thread_count_reason(void);
const char* vmlx_cpu_thread_setting_error(void);
bool vmlx_cpu_openblas_pinned(void);

// Dispatch counters since the last reset; 0 for a family outside the enum.
int64_t vmlx_cpu_executed_targets(vmlx_cpu_family family);
uint64_t vmlx_cpu_highway_calls(vmlx_cpu_family family);
uint64_t vmlx_cpu_fallback_calls(vmlx_cpu_family family);
uint64_t vmlx_cpu_sgemm_column_splits(void);
void vmlx_cpu_reset_counters(void);

#ifdef __cplusplus
}
#endif

#endif
