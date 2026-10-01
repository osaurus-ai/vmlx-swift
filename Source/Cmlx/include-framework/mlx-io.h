#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <Cmlx/mlx-api.h>
#include <Cmlx/mlx-array.h>
#include <Cmlx/mlx-io-load.h>
#include <Cmlx/mlx-stream.h>
#include <Cmlx/mlx-utils.h>

namespace mlx::core {
using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

/** Save array to out stream in .npy format */
MLX_API void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
MLX_API void save(std::string file, array a);

/** Load array from reader in .npy format */
MLX_API array
load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
MLX_API array load(std::string file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
MLX_API SafetensorsLoad
load_safetensors(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});
MLX_API SafetensorsLoad
load_safetensors(const std::string& file, StreamOrDevice s = {});
MLX_API SafetensorsLoad load_safetensors_excluding(
    const std::string& file,
    const std::unordered_set<std::string>& excluded_keys,
    StreamOrDevice s = {});
MLX_API SafetensorsLoad load_safetensors_excluding(
    const std::string& file,
    const std::unordered_set<std::string>& excluded_keys,
    bool exact_tensor_buffers,
    StreamOrDevice s = {});

MLX_API int64_t
safetensors_mmap_advise_routed(int32_t advice, int32_t cold_pct);
MLX_API int64_t safetensors_mmap_advise_experts(
    int32_t advice,
    const int32_t* layers,
    const int32_t* experts,
    int64_t count);
MLX_API int64_t safetensors_mmap_advise_layer(int32_t advice, int32_t layer);
MLX_API int64_t safetensors_mmap_tracked_buffer_bytes();
MLX_API array mmap_file_region(
    const std::string& file,
    uint64_t offset,
    size_t length,
    Shape shape,
    Dtype dtype);

// Named JANGH banks participate in canonical layer/expert cold-page advice.
// Valid names are
// model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.tq2_{packed,scales}. The
// array owns its mapping; registry entries hold only weak references. Advice
// uses the existing process-global layer/expert namespace, not a model ID.
MLX_API array mmap_file_region_named(
    const std::string& file,
    uint64_t offset,
    size_t length,
    Shape shape,
    Dtype dtype,
    const std::string& tensor_name);

MLX_API void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
MLX_API void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});

/** Load array map and metadata from .gguf file format */

MLX_API GGUFLoad load_gguf(const std::string& file, StreamOrDevice s = {});

MLX_API void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> meta_data = {});

} // namespace mlx::core
#endif
