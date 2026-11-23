// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2-Mini multi-row step (B = 1, T rows): the speculative-verify / short-chunk path.
// Same arithmetic as `decodeStep` (Inkling2Decode.swift) generalized to T rows: the short convs
// walk the rows in order inside the kernels; attention takes an additive mask that carries the
// causal / window structure, the relative bias and each row's sink (as an extra zero key).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let f32 = DType.float32

enum Inkling2VerifyKernels {
    static let header = Inkling2DecodeKernels.header

    /// grid.x = head simdgroup (NQ q | NQ r | NKV k | NKV v), grid.y = row t (k / v groups run all
    /// rows themselves at t = 0). q / r: clipped per row, log-scaled by `tau[t]`, BF16-rounded as the
    /// reference. k / v: conv over rows in order from the packed state, rounded, clipped / normed.
    /// imeta = [q off, r off, k off, k dyn off, v off, v dyn off, qa row stride, ka row stride,
    /// va row stride]. Outputs q [T, NQ*D], r [T, NQ*DR], k / v [T, C], kvst [3, 2C] (state after
    /// the last row).
    static let qkvT = MLXFast.metalKernel(
        name: "inkling2_ver_qkv",
        inputNames: [
            "qa", "ka", "va", "kw", "vw", "packed", "qscale", "rscale", "tau", "fmeta", "imeta",
        ],
        outputNames: ["qo", "ro", "ko", "vo", "kvst"],
        source: """
            uint head = threadgroup_position_in_grid.x;
            uint t = threadgroup_position_in_grid.y;
            uint lane = thread_index_in_simdgroup;
            constexpr int C = NKV * D;
            constexpr int EL = D / 32;
            int P = packed_shape[1];
            int T = tau_shape[0];
            float qkc = fmeta[0], rc = fmeta[1], eps = fmeta[2];
            int qs = imeta[6], ks = imeta[7], vs = imeta[8];
            float tt = tau[t];
            if (head < (uint)NQ) {
                int h = head;
                float x[EL]; float ss = 0.f;
                for (int e = 0; e < EL; ++e) {
                    int j = lane + 32 * e;
                    x[e] = float(qa[(size_t)t * qs + imeta[0] + h * D + j]) * qscale[h * D + j];
                    ss += x[e] * x[e];
                }
                ss = simd_sum(ss);
                float f = min(qkc / (sqrt(ss) + 1e-6f), 1.f);
                for (int e = 0; e < EL; ++e) {
                    float y = ink_round_as(qo, x[e] * f);
                    if (tt != 1.f) y = ink_round_as(qo, y * tt);
                    ink_store(qo, (size_t)t * NQ * D + h * D + lane + 32 * e, y);
                }
            } else if (head < (uint)(2 * NQ)) {
                int h = head - NQ;
                float x = 0.f;
                if (lane < (uint)DR) x = float(qa[(size_t)t * qs + imeta[1] + h * DR + lane]) * rscale[h * DR + lane];
                float ss = simd_sum(x * x);
                float f = min(rc / (sqrt(ss) + 1e-6f), 1.f);
                float y = ink_round_as(ro, x * f);
                if (tt != 1.f) y = ink_round_as(ro, y * tt);
                if (lane < (uint)DR) ink_store(ro, (size_t)t * NQ * DR + h * DR + lane, y);
            } else if (t == 0) {
                bool isK = head < (uint)(2 * NQ + NKV);
                int h = isK ? head - 2 * NQ : head - 2 * NQ - NKV;
                const device auto* src = isK ? ka : va;
                int off = isK ? imeta[2] : imeta[4];
                int doff = isK ? imeta[3] : imeta[5];
                int rs = isK ? ks : vs;
                const device float* w = isK ? kw : vw;
                int col0 = isK ? 0 : C;
                float s0[EL], s1[EL], s2[EL];
                for (int e = 0; e < EL; ++e) {
                    int c = h * D + lane + 32 * e;
                    s0[e] = packed[col0 + c]; s1[e] = packed[P + col0 + c]; s2[e] = packed[2 * P + col0 + c];
                }
                for (int r = 0; r < T; ++r) {
                    float y[EL]; float ss = 0.f;
                    for (int e = 0; e < EL; ++e) {
                        int c = h * D + lane + 32 * e;
                        float xt = float(src[(size_t)r * rs + off + c]);
                        float o;
                        if (DYN) {
                            const device auto* dd = src + (size_t)r * rs + doff + h * 4;
                            float a = xt * (w[c] + float(dd[0]));
                            a = a + s2[e] * (w[C + c] + float(dd[1]));
                            a = a + s1[e] * (w[2 * C + c] + float(dd[2]));
                            a = a + s0[e] * (w[3 * C + c] + float(dd[3]));
                            o = (a + xt) / 3.0f;
                        } else {
                            float a = s0[e] * w[c];
                            a = a + s1[e] * w[C + c];
                            a = a + s2[e] * w[2 * C + c];
                            a = a + xt * w[3 * C + c];
                            o = (a + xt) / 2.2360679774997896f;
                        }
                        y[e] = ink_round_as(ko, o);
                        ss += y[e] * y[e];
                        s0[e] = s1[e]; s1[e] = s2[e]; s2[e] = xt;
                    }
                    ss = simd_sum(ss);
                    float f = isK ? min(qkc / (sqrt(ss) + 1e-6f), 1.f) : rsqrt(ss / float(D) + eps);
                    for (int e = 0; e < EL; ++e) {
                        int c = h * D + lane + 32 * e;
                        if (isK) ink_store(ko, (size_t)r * C + c, y[e] * f); else ink_store(vo, (size_t)r * C + c, y[e] * f);
                    }
                }
                for (int e = 0; e < EL; ++e) {
                    int c = h * D + lane + 32 * e;
                    kvst[col0 + c] = s0[e]; kvst[2 * C + col0 + c] = s1[e]; kvst[4 * C + col0 + c] = s2[e];
                }
            }
            """,
        header: header)

    /// mask [NQ, T, S]: key j at absolute position kStart + j: d = (seen + t) - (kStart + j);
    /// valid = d >= 0 (and d < W if W > 0); value = bf16(r[t, h] . proj[:, d]) for d < E, 0 beyond,
    /// minus row t's sink logit for head h (qa at meta[3]); -inf when invalid.
    ///
    /// The per-token sink is folded into the mask: attention runs with a zero native sink, and a
    /// row-constant shift of every key's logit by -sink gives exp(s_j) / (exp(sink) + sum exp(s_i)),
    /// the sink softmax, without a zero key concatenated in front of the whole cache each visit.
    /// meta = [S, seen, kStart, sink off, qa row stride, W].
    static let maskT = MLXFast.metalKernel(
        name: "inkling2_ver_mask",
        inputNames: ["r", "proj", "qa", "meta"],
        outputNames: ["out"],
        source: """
            uint j = thread_position_in_grid.x;
            uint t = thread_position_in_grid.y;
            uint h = thread_position_in_grid.z;
            int S = meta[0], seen = meta[1], kStart = meta[2], soff = meta[3], qs = meta[4], W = meta[5];
            int NQ = r_shape[1];
            int DR = proj_shape[0];
            int E = proj_shape[1];
            int T = r_shape[0];
            if (j >= (uint)S) return;
            size_t o = ((size_t)h * T + t) * S + j;
            int d = (seen + (int)t) - (kStart + (int)j);
            bool valid = d >= 0 && (W <= 0 || d < W);
            if (!valid) { ink_store(out, o, -INFINITY); return; }
            float b = 0.f;
            if (d < E) {
                for (int k = 0; k < DR; ++k) b += float(r[((size_t)t * NQ + h) * DR + k]) * float(proj[k * E + d]);
            }
            float sink = float(qa[(size_t)t * qs + soff + h]);
            ink_store(out, o, ink_round_as(out, b) - sink);
            """,
        header: header)

    /// Per channel, rows in order: m = dtype(sum_k down[t,k,j] * es[inds[t,k]] * w[t,k]); post-MLP
    /// conv (state at packed columns 2C + H + j); hout[t, j] = dtype(res[t, j] + fp32(dtype(conv))).
    /// Also copies kvst / ast into the packed state and writes `mout` [T, H] (the conv inputs, for
    /// the verify recorder). Grid = P threads. meta = [2C, H].
    static let finalT = MLXFast.metalKernel(
        name: "inkling2_ver_final",
        inputNames: ["down", "inds", "es", "w", "packed", "cw", "res", "kvst", "ast", "meta"],
        outputNames: ["hout", "np", "mout"],
        source: """
            uint c = thread_position_in_grid.x;
            int P = packed_shape[1];
            int C2 = meta[0];
            int H = meta[1];
            int T = res_shape[0];
            int K = inds_shape[1];
            if (c >= (uint)P) return;
            if ((int)c < C2) {
                np[c] = kvst[c]; np[P + c] = kvst[C2 + c]; np[2 * P + c] = kvst[2 * C2 + c];
                return;
            }
            if ((int)c < C2 + H) {
                int j = c - C2;
                np[c] = ast[j]; np[P + c] = ast[H + j]; np[2 * P + c] = ast[2 * H + j];
                return;
            }
            int j = c - C2 - H;
            float s0 = packed[c], s1 = packed[P + c], s2 = packed[2 * P + c];
            for (int t = 0; t < T; ++t) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {
                    size_t tk = (size_t)t * K + k;
                    acc += (down[tk * H + j] * es[inds[tk]]) * w[tk];
                }
                float xt = ink_round_as(hout, acc);
                mout[(size_t)t * H + j] = xt;
                float y = s0 * cw[j];
                y = y + s1 * cw[H + j];
                y = y + s2 * cw[2 * H + j];
                y = y + xt * cw[3 * H + j];
                float av = ink_round_as(hout, (y + xt) / 2.2360679774997896f);
                ink_store(hout, (size_t)t * H + j, res[(size_t)t * H + j] + av);
                s0 = s1; s1 = s2; s2 = xt;
            }
            np[c] = s0; np[P + c] = s1; np[2 * P + c] = s2;
            """,
        header: header)
}

extension Inkling2DecoderLayer {

    /// T rows (B = 1) through the fused path. `stack` `[1, T, S, H]`; `messages` `[1, T, 4L, H]`.
    /// With a recorder, every conv's pre-conv inputs are kept for rewinding.
    func multiRowStep(
        _ h: MLXArray, stack: MLXArray?, kv: any KVCache, conv: MambaCache,
        messages: MLXArray?, loopIndex: Int, ones: MLXArray, recorder: Inkling2ConvRecorder?
    ) -> (MLXArray, MLXArray?) {
        prepareDecodeWeights()
        let w = decodeWeights
        let a = selfAttn
        let H = hidden
        let T = h.dim(1)
        let dtype = h.dtype
        let (nq, nkv, hd, dRel) = (a.nq, a.nkv, a.hd, a.dRel)
        let C = nkv * hd
        let P = 2 * C + 2 * H
        let packed = (conv[0] ?? MLXArray.zeros([1, 3, P], dtype: f32)).reshaped([3, P])
        let hT = h.reshaped([T, H])
        let tg = 768
        let elems = (H + tg - 1) / tg
        let S0 = stack?.dim(2) ?? 0
        let st0 = stack?.reshaped([T, S0, H]) ?? w.zeroState!.reshaped([1, 1, H])
        let L4 = messages?.dim(2) ?? 0
        let msgs = messages?.reshaped([-1, H])
        let nqa = isSliding ? 1 : 3

        // 1. attention inputs: rowsQ / rowsK / rowsV [T, H] in the model dtype
        var rowsQ: MLXArray
        var rowsK: MLXArray? = nil
        var rowsV: MLXArray? = nil
        if attentionInput == nil {
            rowsQ = inputLN!(hT)
        } else if msgs == nil {
            let out = Inkling2DecodeKernels.depth(
                [
                    st0, hT, w.attnQuery!, w.attnNorm!,
                    w.normEps!,
                ],
                template: [
                    ("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 1), ("ELEMS", elems),
                    ("TG", tg),
                ],
                grid: (T * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[T, nqa, H]],
                outputDTypes: [dtype])[0]
            if nqa == 1 {
                rowsQ = out.reshaped([T, H])
            } else {
                rowsQ = out[0..., 0, 0...]
                rowsK = out[0..., 1, 0...]
                rowsV = out[0..., 2, 0...]
            }
        } else {
            let xr = Inkling2DecodeKernels.depth(
                [
                    st0, hT, w.attnQuery!, w.attnNorm!,
                    w.normEps!,
                ],
                template: [
                    ("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 0), ("ELEMS", elems),
                    ("TG", tg),
                ],
                grid: (T * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[T, nqa, H]],
                outputDTypes: [f32])[0].reshaped([T * nqa, H])
            let meta = MLXArray([Int32(4 * loopIndex), Int32(nqa), Int32(T), Int32(L4)])
            let g = Inkling2DecodeKernels.messageGates(
                x: xr, msgs: msgs!, down: w.qkvDown!, fg: w.qkvFG!, norm: w.gateNorm!,
                normEps: w.normEps!, gateEps: w.gateEps!, meta: meta, gates: 3, rows: T,
                dtype: dtype
            ).0.reshaped([3, T, H])
            rowsQ = g[0]
            rowsK = g[1]
            rowsV = g[2]
        }

        // 2. projections
        let nQ = nq * hd
        let nR = nq * dRel
        let qa: MLXArray
        let ka: MLXArray
        let va: MLXArray
        var imeta: [Int32]
        if let qp = w.quant {
            // Affine projections: qa = [q | r | gate | sink] (the plain sink appended, as the mask reads it at
            // 2nQ + nR), ka / va from the quantized k / v rows (+ the plain dynamic-tap rows after them).
            qa = MLX.concatenated([qp(rowsQ, qp.qrg), MLX.matmul(rowsQ, qp.sink)], axis: -1)
            if rowsK == nil {
                let kv = qp(rowsQ, qp.kv)  // [T, 2C]
                ka = kv
                va = kv
                imeta = [0, Int32(nQ), 0, 0, Int32(C), 0, Int32(qa.dim(1)), Int32(2 * C), Int32(2 * C)]
            } else {
                var kk = qp(rowsK!, qp.k)
                var vv = qp(rowsV!, qp.v)
                if !a.isSliding, let kt = qp.kTaps, let vt = qp.vTaps {
                    kk = MLX.concatenated([kk, MLX.matmul(rowsK!, kt)], axis: -1)
                    vv = MLX.concatenated([vv, MLX.matmul(rowsV!, vt)], axis: -1)
                }
                ka = kk
                va = vv
                imeta = [
                    0, Int32(nQ), 0, Int32(C), 0, Int32(C), Int32(qa.dim(1)), Int32(ka.dim(1)),
                    Int32(va.dim(1)),
                ]
            }
        } else if rowsK == nil {
            let y = MLX.matmul(rowsQ, w.xFused!.T)  // [T, N]
            qa = y
            ka = y
            va = y
            let base = 2 * nQ + nR + nq
            let N = Int32(y.dim(1))
            imeta = [0, Int32(nQ), Int32(base), 0, Int32(base + C), 0, N, N, N]
        } else {
            qa = MLX.matmul(rowsQ, w.qFused!.T)
            ka = MLX.matmul(rowsK!, w.kFused!.T)
            va = MLX.matmul(rowsV!, w.vFused!.T)
            imeta = [
                0, Int32(nQ), 0, Int32(C), 0, Int32(C), Int32(qa.dim(1)), Int32(ka.dim(1)),
                Int32(va.dim(1)),
            ]
        }
        let seen = kv.offset
        var taus = [Float](repeating: 1, count: T)
        if !a.isSliding {
            for t in 0 ..< T where Float(seen + t + 1) > a.logFloor {
                taus[t] = 1 + a.logAlpha * Foundation.log(max(Float(seen + t + 1) / a.logFloor, 1))
            }
        }
        let kwStatic: MLXArray
        let vwStatic: MLXArray
        if let ks = a.kSconv as? Inkling2ShortConv {
            kwStatic = ks.weight
            vwStatic = (a.vSconv as! Inkling2ShortConv).weight
        } else {
            kwStatic = (a.kSconv as! Inkling2DynamicShortConv).staticWeight
            vwStatic = (a.vSconv as! Inkling2DynamicShortConv).staticWeight
        }
        let outs = Inkling2VerifyKernels.qkvT(
            [
                qa, ka, va, kwStatic, vwStatic, packed, a.qL2Clip.scale, a.relL2Clip.scale,
                MLXArray(taus),
                MLXArray([a.qkClip, a.relClip, a.eps]), MLXArray(imeta),
            ],
            template: [
                ("NQ", nq), ("NKV", nkv), ("D", hd), ("DR", dRel), ("DYN", a.isSliding ? 0 : 1),
            ],
            grid: ((2 * nq + 2 * nkv) * 32, T, 1), threadGroup: (32, 1, 1),
            outputShapes: [[T, nQ], [T, nR], [T, C], [T, C], [3, 2 * C]],
            outputDTypes: [dtype, dtype, dtype, dtype, f32])

        // 3. attention: keys [visible window], mask with sink column
        var (keys, values) = kv.update(
            keys: outs[2].reshaped([1, T, nkv, hd]).transposed(0, 2, 1, 3),
            values: outs[3].reshaped([1, T, nkv, hd]).transposed(0, 2, 1, 3))
        var S = keys.dim(2)
        var kStart = seen + T - S
        if let window = a.window {
            let visible = min(S, window - 1 + T)
            if visible < S {
                keys = keys[0..., 0..., (S - visible)..., 0...]
                values = values[0..., 0..., (S - visible)..., 0...]
                kStart += S - visible
                S = visible
            }
        }
        let sinkOff = 2 * nQ + nR
        let mask = Inkling2VerifyKernels.maskT(
            [
                outs[1].reshaped([T, nq, dRel]), a.relLogits.proj, qa,
                MLXArray([
                    Int32(S), Int32(seen), Int32(kStart), Int32(sinkOff), Int32(qa.dim(1)),
                    Int32(a.window ?? 0),
                ]),
            ],
            grid: (S, T, nq), threadGroup: (min(256, S), 1, 1),
            outputShapes: [[nq, T, S]], outputDTypes: [dtype])[0]
        let attn = MLXFast.scaledDotProductAttention(
            queries: outs[0].reshaped([1, T, nq, hd]).transposed(0, 2, 1, 3),
            keys: keys, values: values, scale: a.scale,
            mask: .array(mask.reshaped([1, nq, T, S])),
            sinks: w.zeroSinks(nq: nq, dtype: dtype))
        let merged = attn.transposed(0, 2, 1, 3).reshaped([T, nQ])
        let gated = Inkling2DecodeKernels.sigmoidGate(
            [merged, qa[0..., (nQ + nR) ..< (2 * nQ + nR)], MLXArray([Int32(0)])],
            grid: (T * nQ, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[T, nQ]],
            outputDTypes: [dtype])[0]
        let o = a.oProj(gated)  // [T, H]

        // 4. residual + MLP input
        let states = isBlockBoundary ? inklingAppendState(stack, h) : stack
        let (residual, newAttn) = postAttnConv.residual(
            o.reshaped([1, T, H]), packed: packed.reshaped([1, 3, P]), offset: 2 * C, res: h,
            mode: isBlockBoundary ? 0 : 1, outDType: f32)
        let S1 = states?.dim(2) ?? 0
        let st1 = states?.reshaped([T, S1, H]) ?? w.zeroState!.reshaped([1, 1, H])
        let res2 = residual.reshaped([T, H])
        var xm32: MLXArray
        if let msgs {
            let mraw = Inkling2DecodeKernels.depth(
                [st1, res2, w.mlpQuery!, w.mlpNorm!, w.normEps!],
                template: [
                    ("NS", S1 + 1), ("HASP", 1), ("NQ", 1), ("NORM", 0), ("ELEMS", elems),
                    ("TG", tg),
                ],
                grid: (T * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[T, 1, H]],
                outputDTypes: [f32])[0].reshaped([T, H])
            let meta = MLXArray([Int32(4 * loopIndex + 3), Int32(1), Int32(T), Int32(L4)])
            xm32 =
                Inkling2DecodeKernels.messageGates(
                    x: mraw, msgs: msgs, down: w.mlpDown!, fg: w.mlpFG!, norm: w.mlpNorm!,
                    normEps: w.normEps!, gateEps: w.gateEps!, meta: meta, gates: 1, rows: T,
                    dtype: dtype
                ).1
        } else {
            let xm = Inkling2DecodeKernels.depth(
                [st1, res2, w.mlpQuery!, w.mlpNorm!, w.normEps!],
                template: [
                    ("NS", S1 + 1), ("HASP", 1), ("NQ", 1), ("NORM", 1), ("ELEMS", elems),
                    ("TG", tg),
                ],
                grid: (T * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[T, 1, H]],
                outputDTypes: [dtype])[0].reshaped([T, H])
            xm32 = xm.asType(f32)
        }

        // 5. MoE
        let router = mlp.gate
        let logits = MLX.matmul(xm32, w.routerT!)
        let sel = Inkling2Kernels.routerSelect(
            [logits, router.bias, router.meta, router.scaleArray], grid: (T * 128, 1, 1),
            threadGroup: (128, 1, 1), outputShapes: [[T, router.topK], [T, router.topK]],
            outputDTypes: [.uint32, f32])
        let inds = sel[0]
        let k = router.topK
        if let jx = mlp.janghExperts {
            // JANGH: the router-weighted sum per row (decode kernels, F32), combined by the final kernel as one slot.
            let perPick = w.jangh.map {
                Inkling2DecodeKernels.janghExperts(x32: xm32, inds: inds, banks: $0, topK: k).down  // [T * k, H]
            }
            let sum = perPick ?? jx.decodeWeightedSum(xm32, indices: inds, scores: sel[1])  // [T, H]
            let fin = Inkling2VerifyKernels.finalT(
                [
                    sum, perPick != nil ? inds : MLXArray.zeros([T, 1], dtype: .uint32), w.ones!,
                    perPick != nil ? sel[1] : MLXArray.ones([T, 1], dtype: f32),
                    packed, postMLPConv.weight, res2, outs[4], newAttn.reshaped([3, H]),
                    MLXArray([Int32(2 * C), Int32(H)]),
                ],
                grid: (P, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[T, H], [3, P], [T, H]],
                outputDTypes: [dtype, f32, f32])
            if let recorder {
                let kRaw = ka[0..., Int(imeta[2]) ..< (Int(imeta[2]) + C)].asType(f32)
                let vRaw = va[0..., Int(imeta[4]) ..< (Int(imeta[4]) + C)].asType(f32)
                recorder.record(
                    conv: conv, oldState: packed.reshaped([1, 3, P]),
                    inputs: MLX.concatenated([kRaw, vRaw, o.asType(f32), fin[2]], axis: -1).reshaped([
                        1, T, P,
                    ]))
            }
            conv[0] = fin[1].reshaped([1, 3, P])
            conv.offset += T
            return (fin[0].reshaped([1, T, H]), states)
        }
        let sw = mlp.switchMLP!
        // Per-row expert gathers: measured fastest at T = 8 on M5 Max (gate/up 326 us vs 636 for a
        // grouped one-read-per-expert kernel and ~800 sorted): rows in a verify block rarely share
        // experts, so the gather is already at bandwidth.
        let down: MLXArray
        if w.nativeFP4Experts {
            down = Inkling2DecodeKernels.nvfp4Experts(x32: xm32, inds: inds, mlp: sw, topK: k).1
        } else {
            let gu = sw.gateUp(xm32.reshaped([T, 1, 1, H]), inds, sorted: false)  // [T, k, 1, 2I]
            let act = Inkling2Kernels.moeActivation(
                [
                    gu.reshaped([T * k, 2 * sw.inter]), inds.reshaped([T * k]),
                    sw.gateUp.isNVFP4 ? sw.gateUp.expertScale : w.ones!, sw.caps,
                ],
                grid: (sw.inter, T * k, 1), threadGroup: (min(256, sw.inter), 1, 1),
                outputShapes: [[T * k, sw.inter]], outputDTypes: [f32])[0]
            down = sw.down(act.reshaped([T, k, 1, sw.inter]), inds, sorted: false)  // [T, k, 1, H]
        }
        let fin = Inkling2VerifyKernels.finalT(
            [
                down.reshaped([T * k, H]), inds, sw.down.isNVFP4 ? sw.down.expertScale : w.ones!,
                sel[1],
                packed, postMLPConv.weight, res2, outs[4], newAttn.reshaped([3, H]),
                MLXArray([Int32(2 * C), Int32(H)]),
            ],
            grid: (P, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[T, H], [3, P], [T, H]],
            outputDTypes: [dtype, f32, f32])
        if let recorder {
            let kRaw = ka[0..., Int(imeta[2]) ..< (Int(imeta[2]) + C)].asType(f32)
            let vRaw = va[0..., Int(imeta[4]) ..< (Int(imeta[4]) + C)].asType(f32)
            recorder.record(
                conv: conv, oldState: packed.reshaped([1, 3, P]),
                inputs: MLX.concatenated([kRaw, vRaw, o.asType(f32), fin[2]], axis: -1).reshaped([
                    1, T, P,
                ]))
        }
        conv[0] = fin[1].reshaped([1, 3, P])
        conv.offset += T
        return (fin[0].reshaped([1, T, H]), states)
    }
}
