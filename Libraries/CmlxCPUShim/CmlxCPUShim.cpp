// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include "CmlxCPUShim.h"

#if defined(MLX_USE_HIGHWAY_KERNELS)

#include "mlx/backend/cpu/highway_info.h"
#include "mlx/backend/cpu/precision.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/backend/cpu/threading/config.h"

namespace cpu = mlx::core::cpu;
namespace hi = mlx::core::cpu::highway_info;

static_assert(
    static_cast<int>(hi::Family::Count) == VMLX_CPU_FAMILY_COUNT,
    "vmlx_cpu_family must list mlx::core::cpu::highway_info::Family in order");

static hi::Family family_of(vmlx_cpu_family family) {
  return static_cast<hi::Family>(family);
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
extern "C" void vmlx_cpu_highway_set_targets_for_test(int64_t targets) {
  hi::set_targets_for_test(targets);
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
  return hi::stats(family_of(family)).executed;
}
extern "C" uint64_t vmlx_cpu_highway_calls(vmlx_cpu_family family) {
  return hi::stats(family_of(family)).highway_calls;
}
extern "C" uint64_t vmlx_cpu_fallback_calls(vmlx_cpu_family family) {
  return hi::stats(family_of(family)).fallback_calls;
}
extern "C" uint64_t vmlx_cpu_sgemm_column_splits(void) {
  return hi::sgemm_column_splits();
}
extern "C" void vmlx_cpu_reset_counters(void) {
  hi::reset_stats();
}

#else

// A build without Highway kernels: no targets, no switch, no pool, no counts.
extern "C" bool vmlx_cpu_highway_enabled(void) {
  return false;
}
extern "C" int64_t vmlx_cpu_highway_compiled_targets(void) {
  return 0;
}
extern "C" int64_t vmlx_cpu_highway_supported_targets(void) {
  return 0;
}
extern "C" void vmlx_cpu_highway_set_targets_for_test(int64_t) {}
extern "C" const char* vmlx_cpu_highway_target_name(int64_t) {
  return "";
}
extern "C" bool vmlx_cpu_quantized_int8(void) {
  return false;
}
extern "C" void vmlx_cpu_set_quantized_int8(bool) {}
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
