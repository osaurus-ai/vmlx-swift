// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
#include "utils.h"
#include "sdpa_vector.h"

using namespace metal;

constant bool inkling_sliding [[function_constant(27)]];

// The first pass follows MLX's sdpa_vector_2pass_1 arithmetic and partition layout.
// Different query rows share the original KV storage and use their own key span and sink.
template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void vmlx_inkling_verify_attention_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& baseLength [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = kv_head_idx;
  const int window_start = inkling_sliding ? batch_idx : 0;
  const int N = baseLength + (inkling_sliding ? 0 : batch_idx);
  keys += kv_batch_head_idx * k_head_stride + (block_idx + window_start) * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + (block_idx + window_start) * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += (q_head_idx * tpg.y + batch_idx) * mask_head_stride +
        (block_idx + window_start) * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx]);
    sum_exp_score = 1;
  }

  // Pack the shared single-query boolean mask once per threadgroup. Keep
  // every selected key in its original strided partition and chronological
  // order; score/softmax/partial rounding and second-pass math stay unchanged.
  // Other layouts and very long per-partition spans retain the original loop.
  threadgroup uint packed_mask_words[32];
  const int mask_word_count = (N + blocks * 32 - 1) / (blocks * 32);
  bool use_packed_mask = false;
  if (bool_mask) {
    use_packed_mask = !do_causal && q_seq_len == 1 && mask_head_stride == 0 &&
        mask_kv_seq_stride == 1 && N > 32768 && mask_word_count <= 32;
    if (use_packed_mask) {
      for (int word = tidtg.y; word < mask_word_count; word += gqa_factor) {
        const int step = word * 32 + simd_lid;
        const int key = block_idx + step * blocks;
        uint bits = uint(
            simd_vote::vote_t(simd_ballot(key < N && bmask[step * blocks])));
        if (simd_lid == 0) {
          packed_mask_words[word] = bits;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  int next_mask_word = 0;
  uint active_mask_bits = 0;

  int i = block_idx;
  const int stride_k_tok = blocks * int(k_seq_stride);
  const int stride_v_tok = blocks * int(v_seq_stride);
  const int stride_mask_tok =
      (bool_mask || float_mask) ? blocks * mask_kv_seq_stride : 0;

  // For each key
  for (; !use_packed_mask && i + (TK - 1) * blocks < N; i += blocks * TK) {
    // Prefetch: issue all TK K/V loads first so the HW can overlap them.
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; t++) {
      const device T* kp = keys + t * stride_k_tok;
      for (int j = 0; j < qk_per_thread; j++) {
        k_reg[t][j] = kp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = values + t * stride_v_tok;
      for (int j = 0; j < v_per_thread; j++) {
        v_reg[t][j] = vp[j];
      }
    }

    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      bool use_key = true;
      if (do_causal) {
        use_key = ii <= (N - q_seq_len + int(q_seq_idx));
      } else if (bool_mask) {
        use_key = bmask[t * stride_mask_tok];
      } else if (float_mask) {
        use_key = (fmask[t * stride_mask_tok] >= Limits<T>::finite_min);
      }
      if (use_key) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        if (float_mask) {
          score += static_cast<U>(fmask[t * stride_mask_tok]);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
    }

    keys += TK * stride_k_tok;
    values += TK * stride_v_tok;
    if (bool_mask) {
      bmask += TK * stride_mask_tok;
    }
    if (float_mask) {
      fmask += TK * stride_mask_tok;
    }
  }

  // Retain the sparse packed-mask lane and handle the unrolled tail.
  for (; i < N; i += blocks) {
    if (use_packed_mask) {
      while (active_mask_bits == 0 && next_mask_word < mask_word_count) {
        active_mask_bits = packed_mask_words[next_mask_word++];
      }
      if (active_mask_bits == 0) {
        break;
      }
      const int next_key = block_idx +
          ((next_mask_word - 1) * 32 + int(ctz(active_mask_bits))) * blocks;
      active_mask_bits &= active_mask_bits - 1;
      const int skipped = next_key - i;
      keys += skipped * int(k_seq_stride);
      values += skipped * int(v_seq_stride);
      if (bool_mask) {
        bmask += skipped * mask_kv_seq_stride;
      }
      i = next_key;
    }
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries without mask or sinks only; the partials layout matches
// sdpa_vector_2pass_2.

instantiate_kernel("vmlx_inkling_verify_attention_1_bf16", vmlx_inkling_verify_attention_1, bfloat16_t, 128, 128)
