// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
import CmlxGraphShim
import Foundation
import MLX
import MLXNN

/// Affine q6/g64 decode gate/up projections and GELU, preserving projection
/// BF16 rounding and the existing FP32 per-lane accumulation/reduction order.
enum Spark25DualProjection {
    static func apply(_ x: MLXArray, gate: Linear, up: Linear) -> MLXArray? {
        #if canImport(Metal)
            guard !referenceOverride, x.shape == [1, 1, 2560], x.dtype == .bfloat16,
                Spark25Activation.usesMetalStream,
                type(of: gate) == QuantizedLinear.self, type(of: up) == QuantizedLinear.self,
                let g = gate as? QuantizedLinear, let u = up as? QuantizedLinear,
                g.bits == 6, u.bits == 6, g.groupSize == 64, u.groupSize == 64,
                g.mode == .affine, u.mode == .affine, g.bias == nil, u.bias == nil,
                let gb = g.biases, let ub = u.biases,
                g.weight.shape == [10240, 480], u.weight.shape == [10240, 480],
                g.weight.dtype == .uint32, u.weight.dtype == .uint32,
                g.scales.shape == [10240, 40], u.scales.shape == [10240, 40],
                gb.shape == [10240, 40], ub.shape == [10240, 40],
                g.scales.dtype == .bfloat16, u.scales.dtype == .bfloat16,
                gb.dtype == .bfloat16, ub.dtype == .bfloat16
            else { return nil }
            let inputs = [x, g.weight, g.scales, gb, u.weight, u.scales, ub]
            guard inputs.allSatisfy({ vmlx_graph_array_is_tracer($0.ctx.ctx) == 0 })
            else { return nil }
            return kernel(inputs, template: [("T", DType.bfloat16), ("K", 2560), ("N", 10240)],
                grid: (32, 2560, 1), threadGroup: (32, 2, 1),
                outputShapes: [[1, 1, 10240]], outputDTypes: [.bfloat16])[0]
        #else
            return nil
        #endif
    }

    private static let referenceOverride =
        ProcessInfo.processInfo.environment["VMLX_SPARK_DUAL_Q6_REFERENCE"] == "1"

    #if canImport(Metal)
    private static let kernel = MLXFast.metalKernel(
        name: "spark25_dual_q6_gelu", inputNames: ["x", "gw", "gs", "gb", "uw", "us", "ub"],
        outputNames: ["out"], source: """
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row0 = threadgroup_position_in_grid.y * 8 + sg * 4;
        const device uint8_t* w0 = (const device uint8_t*)gw + row0*(K*3/4) + lane*6;
        const device uint8_t* w1 = (const device uint8_t*)uw + row0*(K*3/4) + lane*6;
        const device T* s0 = gs+row0*(K/64)+lane/8;
        const device T* b0 = gb+row0*(K/64)+lane/8;
        const device T* s1 = us+row0*(K/64)+lane/8;
        const device T* b1 = ub+row0*(K/64)+lane/8;
        const device T* xp = x+lane*8;
        float r0[4]={0}, r1[4]={0};
        float xt[8];
        for(int k=0;k<K;k+=256) {
         float sum=load_vector<T,float,8,6>(xp,xt);
         for(int row=0;row<4;row++) {
          r0[row]+=qdot<float,8,6>(w0+row*(K*3/4),xt,float(s0[row*(K/64)]),float(b0[row*(K/64)]),sum);
          r1[row]+=qdot<float,8,6>(w1+row*(K*3/4),xt,float(s1[row*(K/64)]),float(b1[row*(K/64)]),sum);
         }
         w0+=192;w1+=192;s0+=4;b0+=4;s1+=4;b1+=4;xp+=256;
        }
        for(int row=0;row<4;row++) {
         float vg=simd_sum(r0[row]),vu=simd_sum(r1[row]);
         if(lane==0) {
          T g=T(vg), u=T(vu);
          T scaled=g/T(1.4142135623730951f);
          T e=T(spark_erf(float(scaled)));
          T plus=T(1)+e;
          T product=g*plus;
          T activation=product/T(2);
          out[row0+row]=activation*u;
         }
        }
        """, header: projectionHeader + Spark25Activation.geluMetalHeader,
        ensureRowContiguous: true)

    private static let projectionHeader = """
        // Copyright © 2023-2026 Apple Inc.
        template <int bits, int wsize = 8>
        inline constexpr short get_pack_factor() {
          return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
        }

        template <int bits, int wsize = 8>
        inline constexpr short get_bytes_per_pack() {
          constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
          return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector(const device T* x, thread U* x_thread) {
          static_assert(
              bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
                  bits == 6 || bits == 8,
              "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

          U sum = 0;

          if (bits == 1) {
            for (int i = 0; i < values_per_thread; i += 8) {
              for (int j = 0; j < 8; j++) {
                sum += x[i + j];
                x_thread[i + j] = x[i + j] / static_cast<U>(1 << j);
              }
            }
          }

          else if (bits == 2) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 4.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 64.0f;
            }
          }

          else if (bits == 3) {
            for (int i = 0; i < values_per_thread; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 8.0f;
              x_thread[i + 2] = x[i + 2] / 64.0f;
              x_thread[i + 3] = x[i + 3] / 2.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 128.0f;
              x_thread[i + 6] = x[i + 6] / 4.0f;
              x_thread[i + 7] = x[i + 7] / 32.0f;
            }
          }

          else if (bits == 4) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }

          else if (bits == 5) {
            for (int i = 0; i < values_per_thread; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 32.0f;
              x_thread[i + 2] = x[i + 2] / 4.0f;
              x_thread[i + 3] = x[i + 3] / 128.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 2.0f;
              x_thread[i + 6] = x[i + 6] / 64.0f;
              x_thread[i + 7] = x[i + 7] / 8.0f;
            }
          }

          else if (bits == 6) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 64.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 4.0f;
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }

          return sum;
        }

        template <typename U, int values_per_thread, int bits>
        inline U qdot(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          static_assert(
              bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
                  bits == 6 || bits == 8,
              "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

          U accum = 0;

          if (bits == 1) {
            for (int i = 0; i < (values_per_thread / 8); i++) {
              for (int j = 0; j < 8; j++) {
                accum += x_thread[8 * i + j] * (w[i] & (1 << j));
              }
            }
          }

          else if (bits == 2) {
            for (int i = 0; i < (values_per_thread / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (w[i] & 0x03) +
                   x_thread[4 * i + 1] * (w[i] & 0x0c) +
                   x_thread[4 * i + 2] * (w[i] & 0x30) +
                   x_thread[4 * i + 3] * (w[i] & 0xc0));
            }
          }

          else if (bits == 3) {
            for (int i = 0; i < (values_per_thread / 8); i++) {
              x_thread += 8 * i;
              w += 3 * i;

              accum += (w[0] & 0x07) * x_thread[0];
              accum += (w[0] & 0x38) * x_thread[1];
              accum += (w[0] & 0xc0) * x_thread[2];
              accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

              accum += (w[1] & 0x0e) * x_thread[3];
              accum += (w[1] & 0x70) * x_thread[4];
              accum += (w[1] & 0x80) * x_thread[5];
              accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

              accum += (w[2] & 0x1c) * x_thread[6];
              accum += (w[2] & 0xe0) * x_thread[7];
            }
          }

          else if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (values_per_thread / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (ws[i] & 0x000f) +
                   x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                   x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                   x_thread[4 * i + 3] * (ws[i] & 0xf000));
            }
          }

          else if (bits == 5) {
            for (int i = 0; i < (values_per_thread / 8); i++) {
              x_thread += 8 * i;
              w += 5 * i;

              accum += (w[0] & 0x1f) * x_thread[0];
              accum += (w[0] & 0xe0) * x_thread[1];
              accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
              accum += (w[1] & 0x7c) * x_thread[2];
              accum += (w[1] & 0x80) * x_thread[3];
              accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
              accum += (w[2] & 0xf0) * x_thread[4];
              accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
              accum += (w[3] & 0x3e) * x_thread[5];
              accum += (w[3] & 0xc0) * x_thread[6];
              accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
              accum += (w[4] & 0xf8) * x_thread[7];
            }
          }

          else if (bits == 6) {
            for (int i = 0; i < (values_per_thread / 4); i++) {
              x_thread += 4 * i;
              w += 3 * i;

              accum += (w[0] & 0x3f) * x_thread[0];

              accum += (w[0] & 0xc0) * x_thread[1];
              accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

              accum += (w[1] & 0xf0) * x_thread[2];
              accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

              accum += (w[2] & 0xfc) * x_thread[3];
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              accum += x_thread[i] * w[i];
            }
          }

          return scale * accum + sum * bias;
        }

        """
    #endif
}
