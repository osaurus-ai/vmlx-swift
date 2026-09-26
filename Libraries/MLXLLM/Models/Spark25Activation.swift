// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
import Cmlx
import CmlxGraphShim
import Foundation
import MLX
import MLXNN

/// Prefill erf GELU followed by multiplication, with the same BF16 rounding at
/// every intermediate as MLXNN.gelu(gate) * up. Other dtypes/devices retain the
/// reference path. This does not change projection quantization or residuals.
enum Spark25Activation {
    static func isLargePrefillShape(_ x: MLXArray) -> Bool {
        x.ndim >= 2 && x.dim(-2) >= 128
    }

    static func geluMultiply(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        #if canImport(Metal)
            // Keep single-token and short-chunk execution on the reference
            // path: full-model parsed decode regressed despite faster isolated
            // MLP timings. Fusion is qualified separately for large prefill.
            guard isLargePrefillShape(gate),
                !referenceOverride, usesMetalStream,
                gate.dtype == .bfloat16, up.dtype == .bfloat16,
                gate.shape == up.shape, gate.size > 0, gate.size <= Int(Int32.max),
                vmlx_graph_array_is_tracer(gate.ctx.ctx) == 0,
                vmlx_graph_array_is_tracer(up.ctx.ctx) == 0
            else { return gelu(gate) * up }
            return kernel(
                [gate, up], template: [("T", DType.bfloat16)],
                grid: (gate.size, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [gate.shape], outputDTypes: [.bfloat16]
            )[0]
        #else
            return gelu(gate) * up
        #endif
    }

    // Diagnostic A/B switch: the reference uses the same dtype, sampler and
    // model parameters. Read once, before the first forward.
    private static let referenceOverride =
        ProcessInfo.processInfo.environment["VMLX_SPARK_GELU_REFERENCE"] == "1"

    static var usesMetalStream: Bool {
        let stream = StreamOrDevice.default
        if stream == .gpu { return true }
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        return mlx_stream_get_device(&device, stream.ctx) == 0
            && mlx_device_get_type(&type, device) == 0 && type == MLX_GPU
    }

    // Traced transformations use the original expression. In particular, do
    // not put CustomTransforms/StopGradient nodes on ordinary inference:
    // their stream dependencies serialize otherwise asynchronous decode.

    // METAL-ONLY: case 2. Metal kernel, called only under `canImport(Metal)` and when
    // `usesMetalStream` holds; otherwise `geluMultiply` computes `gelu(gate) * up` with MLX ops.
    private static let kernel = MLXFast.metalKernel(
        name: "spark25_exact_bf16_gelu_multiply",
        inputNames: ["g", "u"], outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            T x = g[i];
            T scaled = x / T(1.4142135623730951f);
            T e = T(spark_erf(float(scaled)));
            T plus = T(1) + e;
            T product = x * plus;
            T activation = product / T(2);
            out[i] = activation * u[i];
            """,
        // Same polynomial and expm1 implementation as the pinned MLX Metal
        // erf.h/expm1f.h. Keep their coefficients, rounding and licenses intact.
        header: geluMetalHeader
    )

    static let geluMetalHeader = """
            // Copyright © 2023 Apple Inc.



            // Original license copied below:
            //  Copyright (c) 2015-2023 Norbert Juffa
            //  All rights reserved.
            //
            //  Redistribution and use in source and binary forms, with or without
            //  modification, are permitted provided that the following conditions
            //  are met:
            //
            //  1. Redistributions of source code must retain the above copyright
            //     notice, this list of conditions and the following disclaimer.
            //
            //  2. Redistributions in binary form must reproduce the above copyright
            //     notice, this list of conditions and the following disclaimer in the
            //     documentation and/or other materials provided with the distribution.
            //
            //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
            //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
            //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
            //  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
            //  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
            //  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
            //  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
            //  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
            //  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            //  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            //  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

            /* Compute exponential base e minus 1. Maximum ulp error = 0.997458

               i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
               Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
               With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
               when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

               NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
            */
            float spark_expm1f_scaled_unchecked(float a, float b) {
              float f, j, r, s, t, u, v, x, y;
              int i;

              // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
              j = fma(1.442695f, a, 12582912.f); // 0x1.715476p0, 0x1.8p23
              j = j - 12582912.0f; // 0x1.8p23
              i = (int)j;
              f = fma(j, -6.93145752e-1f, a);

              // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
              s = f * f;
              if (a == 0.0f)
                s = a; // ensure -0 is passed through
              // err = 0.997458  ulp1 = 11081805
              r = 1.97350979e-4f; // 0x1.9de000p-13
              r = fma(r, f, 1.39309070e-3f); // 0x1.6d30bcp-10
              r = fma(r, f, 8.33343994e-3f); // 0x1.1111f6p-7
              r = fma(r, f, 4.16668020e-2f); // 0x1.55559ep-5
              r = fma(r, f, 1.66666716e-1f); // 0x1.55555cp-3
              r = fma(r, f, 4.99999970e-1f); // 0x1.fffffep-2
              u = (j == 1) ? (f + 0.5f) : f;
              v = fma(r, s, u);
              s = 0.5f * b;
              t = ldexp(s, i);
              y = t - s;
              x = (t - y) - s; // double-float canonicalization of difference
              r = fma(v, t, x) + y;
              r = r + r;
              if (j == 0)
                r = v;
              if (j == 1)
                r = v + v;
              return r;
            }

            /* Compute exponential base e minus 1. max ulp err = 0.99746 */
            float spark_expm1f(float a) {
              float r;

              r = spark_expm1f_scaled_unchecked(a, 1.0f);
              /* handle severe overflow and underflow */
              if (abs(a - 1.0f) > 88.0f) {
                r = pow(2, a);
                r = fma(r, r, -1.0f);
              }
              return r;
            }
            // Copyright © 2023 Apple Inc.


            /*
             * Approximation to the error function.
             * Based on code from:
             * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
             */
            float spark_erf(float a) {
              float r, s, t, u;
              t = metal::abs(a);
              s = a * a;
              if (t > 0.927734375f) {
                // maximum error 0.99527 ulp
                r = metal::fma(
                    -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
                u = metal::fma(
                    -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
                r = metal::fma(r, s, u);
                r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
                r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
                r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
                r = metal::fma(r, t, -t);
                r = -spark_expm1f(r);
                r = metal::copysign(r, a);
              } else {
                // maximum error 0.98929 ulp
                r = -5.96761703e-4f; // -0x1.38e000p-11
                r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
                r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
                r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
                r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
                r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
                r = metal::fma(r, a, a);
              }
              return r;
            }
            """
}
