// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
#include "include/mlx/c/inkling_verify_attention.h"

#ifdef _METAL_

#include "mlx/backend/metal/utils.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/primitives.h"

#include "mlx/utils.h"
#include <stdexcept>

namespace {
using namespace mlx::core;

class InklingVerifyAttention final : public UnaryPrimitive {
 public:
  InklingVerifyAttention(
      Stream stream, float scale, bool sliding, MTL::ComputePipelineState* first)
      : UnaryPrimitive(stream), scale_(scale), sliding_(sliding), first_(first) {}

  const char* name() const override { return "InklingVerifyAttention"; }

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling verifier attention requires Metal.");
  }

  static MTL::ComputePipelineState* kernel(metal::Device& device, bool sliding) {
    bool enabled = true, disabled = false;
    int blocks = 64;
    metal::MTLFCList constants = {
        {&enabled, MTL::DataType::DataTypeBool, 20},
        {&disabled, MTL::DataType::DataTypeBool, 21},
        {&disabled, MTL::DataType::DataTypeBool, 22},
        {&disabled, MTL::DataType::DataTypeBool, 23},
        {&enabled, MTL::DataType::DataTypeBool, 24},
        {&enabled, MTL::DataType::DataTypeBool, 25},
        {&blocks, MTL::DataType::DataTypeInt, 26},
        {&sliding, MTL::DataType::DataTypeBool, 27},
    };
    return device.get_kernel(
        "vmlx_inkling_verify_attention_1_bf16",
        sliding ? "vmlx_inkling_verify_attention_sliding" : "vmlx_inkling_verify_attention_full",
        constants);
  }

  void eval_gpu(const std::vector<array>& inputs, array& output) override {
    const auto& query = inputs[0];
    const auto& keys = inputs[1];
    const auto& values = inputs[2];
    const auto& bias = inputs[3];
    const auto& sinks = inputs[4];
    const int rows = query.shape(0);
    constexpr int keyHeads = 8, heads = 24, dim = 128, partitions = 64, window = 1024;
    const auto rowsHeads = rows * heads;
    const int baseLength = sliding_ ? window : keys.shape(2) - rows + 1;
    array partial({rowsHeads, partitions, dim}, bfloat16, nullptr, {});
    array sums({rowsHeads, partitions}, float32, nullptr, {});
    array maxima({rowsHeads, partitions}, float32, nullptr, {});
    for (auto* a : {&partial, &sums, &maxima, &output}) {
      a->set_data(allocator::malloc(a->nbytes()));
    }

    const auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    encoder.add_temporary(partial);
    encoder.add_temporary(sums);
    encoder.add_temporary(maxima);
    encoder.set_compute_pipeline_state(first_);
    encoder.set_input_array(query, 0);
    encoder.set_input_array(keys, 1);
    encoder.set_input_array(values, 2);
    encoder.set_output_array(partial, 3);
    encoder.set_output_array(sums, 4);
    encoder.set_output_array(maxima, 5);
    encoder.set_bytes(baseLength, 7);
    encoder.set_bytes(size_t(keys.strides()[1]), 8);
    encoder.set_bytes(size_t(keys.strides()[2]), 9);
    encoder.set_bytes(size_t(values.strides()[1]), 10);
    encoder.set_bytes(size_t(values.strides()[2]), 11);
    encoder.set_bytes(scale_, 12);
    encoder.set_input_array(bias, 14);
    encoder.set_bytes(1, 15);
    encoder.set_bytes(0, 16);
    encoder.set_bytes(int(bias.shape(2)), 17);
    encoder.set_input_array(sinks, 18);
    encoder.dispatch_threadgroups(MTL::Size(keyHeads, rows, partitions), MTL::Size(32, 3, 1));

    auto* second = device.get_kernel("sdpa_vector_2pass_2_bfloat16_t_128");
    encoder.set_compute_pipeline_state(second);
    encoder.set_input_array(partial, 0);
    encoder.set_input_array(sums, 1);
    encoder.set_input_array(maxima, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(partitions, 4);
    encoder.dispatch_threadgroups(MTL::Size(rowsHeads, 1, 1), MTL::Size(1024, 1, 1));
  }

 private:
  float scale_;
  bool sliding_;
  MTL::ComputePipelineState* first_;
};
} // namespace

extern "C" bool vmlx_inkling_verify_attention(
    mlx_array* output, mlx_array q, mlx_array keys, mlx_array values,
    mlx_array bias, mlx_array sinks, float scale, bool sliding, mlx_stream requested) {
  try {
    using namespace mlx::core;
    auto query = mlx_array_get_(q);
    auto k = mlx_array_get_(keys);
    auto v = mlx_array_get_(values);
    auto mask = mlx_array_get_(bias);
    auto sink = mlx_array_get_(sinks);
    const auto rows = query.shape(0);
    const auto length = k.shape(2);
    if (rows < 2 || rows > 8 ||
        query.shape() != Shape{rows, 3072} ||
        length < 1023 + rows || (sliding && length != 1023 + rows) ||
        k.shape() != Shape{1, 8, length, 128} || v.shape() != k.shape() ||
        mask.shape() != Shape{24, rows, length} ||
        sink.size() != rows * 24 || query.dtype() != bfloat16 ||
        k.dtype() != bfloat16 || v.dtype() != bfloat16 ||
        sink.dtype() != bfloat16 || mask.dtype() != bfloat16 ||
        k.strides()[3] != 1 || v.strides()[3] != 1) {
      return false;
    }
    const auto stream = mlx_stream_get_(requested);
    if (stream.device.type != Device::gpu || env::get_var("MLX_SDPA_BLOCKS", 0) > 0) {
      return false;
    }
    auto& device = metal::device(stream.device);
    if (device.get_architecture() != "applegpu_g17s") {
      return false;
    }
    auto* first = InklingVerifyAttention::kernel(device, sliding);
    auto primitive = std::make_shared<InklingVerifyAttention>(stream, scale, sliding, first);
    mlx_array_set_(*output, array(
        query.shape(), bfloat16, primitive,
        {contiguous(query, false, stream), k, v, contiguous(mask, false, stream),
         contiguous(sink, false, stream)}));
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

#else

extern "C" bool vmlx_inkling_verify_attention(
    mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, float, bool, mlx_stream) {
  return false;
}

#endif
