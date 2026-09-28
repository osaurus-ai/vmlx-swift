// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "CmlxCPUShim.h"

#if defined(MLX_USE_HIGHWAY_KERNELS) || defined(__linux__)
#include "mlx/backend/cpu/precision.h"
#endif

#if defined(MLX_USE_HIGHWAY_KERNELS)

#include "mlx/backend/cpu/highway_info.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/backend/cpu/threading/config.h"

namespace cpu = mlx::core::cpu;
namespace hi = mlx::core::cpu::highway_info;

// vmlx_cpu_family lists hi::Family's enumerators, in order.
static constexpr bool same(vmlx_cpu_family family, hi::Family cpp) {
  return static_cast<int>(family) == static_cast<int>(cpp);
}
static_assert(
    same(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT, hi::Family::QmmAffineDequant));
static_assert(same(VMLX_CPU_FAMILY_QMM_AFFINE_INT8, hi::Family::QmmAffineInt8));
static_assert(same(VMLX_CPU_FAMILY_QMM_FP, hi::Family::QmmFp));
static_assert(same(VMLX_CPU_FAMILY_RMS_NORM, hi::Family::RmsNorm));
static_assert(same(VMLX_CPU_FAMILY_LAYER_NORM, hi::Family::LayerNorm));
static_assert(same(VMLX_CPU_FAMILY_ROPE, hi::Family::Rope));
static_assert(same(VMLX_CPU_FAMILY_SDPA, hi::Family::Sdpa));
static_assert(same(VMLX_CPU_FAMILY_COUNT, hi::Family::Count));

// The counters of `family`, and none for a value outside vmlx_cpu_family.
static hi::FamilyStats stats_of(vmlx_cpu_family family) {
  const auto index = static_cast<unsigned>(family);
  if (index >= static_cast<unsigned>(VMLX_CPU_FAMILY_COUNT)) {
    return {0, 0, 0};
  }
  return hi::stats(static_cast<hi::Family>(index));
}

extern "C" bool vmlx_cpu_highway_enabled(void) {
  return true;
}
extern "C" int64_t vmlx_cpu_highway_compiled_targets(void) {
  return hi::compiled_targets();
}
extern "C" int64_t vmlx_cpu_highway_supported_targets(void) {
  return hi::supported_targets();
}
extern "C" bool vmlx_cpu_highway_set_targets_for_test(int64_t targets) {
  // Dispatching a target this CPU lacks would fault (SIGILL). Clearing the
  // restriction first makes supported_targets() the CPU's own set.
  hi::set_targets_for_test(0);
  if ((targets & ~hi::supported_targets()) != 0) {
    return false;
  }
  hi::set_targets_for_test(targets);
  return true;
}
extern "C" const char* vmlx_cpu_highway_target_name(int64_t target) {
  return hi::target_name(target);
}
extern "C" bool vmlx_cpu_quantized_int8(void) {
  return cpu::quantized_int8();
}
extern "C" void vmlx_cpu_set_quantized_int8(bool enabled) {
  cpu::set_quantized_int8(enabled);
}
extern "C" int32_t vmlx_cpu_thread_count(void) {
  return cpu::ThreadPool::instance().max_threads();
}
extern "C" const char* vmlx_cpu_thread_count_reason(void) {
  return cpu::thread_config().reason;
}
extern "C" const char* vmlx_cpu_thread_setting_error(void) {
  return cpu::thread_config().error.c_str();
}
extern "C" bool vmlx_cpu_openblas_pinned(void) {
  return cpu::openblas_pinned();
}
extern "C" int64_t vmlx_cpu_executed_targets(vmlx_cpu_family family) {
  return stats_of(family).executed;
}
extern "C" uint64_t vmlx_cpu_highway_calls(vmlx_cpu_family family) {
  return stats_of(family).highway_calls;
}
extern "C" uint64_t vmlx_cpu_fallback_calls(vmlx_cpu_family family) {
  return stats_of(family).fallback_calls;
}
extern "C" uint64_t vmlx_cpu_sgemm_column_splits(void) {
  return hi::sgemm_column_splits();
}
extern "C" void vmlx_cpu_reset_counters(void) {
  hi::reset_stats();
}

#else

// A build without Highway kernels: no targets, no pool, no counts.
extern "C" bool vmlx_cpu_highway_enabled(void) {
  return false;
}
extern "C" int64_t vmlx_cpu_highway_compiled_targets(void) {
  return 0;
}
extern "C" int64_t vmlx_cpu_highway_supported_targets(void) {
  return 0;
}
extern "C" bool vmlx_cpu_highway_set_targets_for_test(int64_t targets) {
  return targets == 0;
}
extern "C" const char* vmlx_cpu_highway_target_name(int64_t) {
  return "";
}
#if defined(__linux__)
// Linux compiles precision.cpp with or without Highway kernels, so the shim
// reports the core's own switch, which changes nothing without them.
extern "C" bool vmlx_cpu_quantized_int8(void) {
  return mlx::core::cpu::quantized_int8();
}
extern "C" void vmlx_cpu_set_quantized_int8(bool enabled) {
  mlx::core::cpu::set_quantized_int8(enabled);
}
#else
extern "C" bool vmlx_cpu_quantized_int8(void) {
  return false;
}
extern "C" void vmlx_cpu_set_quantized_int8(bool) {}
#endif
extern "C" int32_t vmlx_cpu_thread_count(void) {
  return 1;
}
extern "C" const char* vmlx_cpu_thread_count_reason(void) {
  return "no pool";
}
extern "C" const char* vmlx_cpu_thread_setting_error(void) {
  return "";
}
extern "C" bool vmlx_cpu_openblas_pinned(void) {
  return false;
}
extern "C" int64_t vmlx_cpu_executed_targets(vmlx_cpu_family) {
  return 0;
}
extern "C" uint64_t vmlx_cpu_highway_calls(vmlx_cpu_family) {
  return 0;
}
extern "C" uint64_t vmlx_cpu_fallback_calls(vmlx_cpu_family) {
  return 0;
}
extern "C" uint64_t vmlx_cpu_sgemm_column_splits(void) {
  return 0;
}
extern "C" void vmlx_cpu_reset_counters(void) {}

#endif
