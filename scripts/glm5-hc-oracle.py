#!/usr/bin/env python3
"""Differential oracle for GLM-5.3's manifold-constrained hyper-connections (mHC).

This is a LINE-BY-LINE transcription of the reference implementation in
huggingface/transformers, `src/transformers/models/glm5_next/modeling_glm5_next.py`
(classes `Glm5NextTextUnweightedRMSNorm`, `Glm5NextTextHyperConnection`,
`Glm5NextTextHyperHead`, and the hyper-connection bracketing inside
`Glm5NextTextDecoderLayer.forward`).  It exists so the Swift implementation can be
checked against something other than my reading of that file: the Swift side reuses
`DeepseekV4HyperConnection`, and "DeepSeek's mechanism is also GLM's mechanism" is
exactly the kind of claim that deserves numbers rather than a comment.

Pure Python on purpose — no numpy, no torch.  The tensors are deliberately tiny, and
a dependency-free transcription is one fewer place for broadcasting semantics I did
not intend to creep in.

Run it to regenerate the fixture embedded in Glm5NextHyperConnectionOracleTests.swift:

    python3 scripts/glm5-hc-oracle.py

The sublayer inside the bracketing is the IDENTITY.  That is not a simplification of
the thing under test — the sublayer (attention or MoE) is tested elsewhere, and
replacing it with the identity is what isolates the hyper-connection wiring: which
tensor is the residual, whether `comb` is transposed before the matmul, and how the
sublayer's output is placed back across the streams.
"""

import json
import math

B, L, HC, D = 1, 2, 4, 4
MIX = (2 + HC) * HC          # 24
RMS_EPS = 1e-5               # config.rms_norm_eps — the INPUT NORM's epsilon
HC_EPS = 1e-6                # config.hc_eps — the sigmoid/softmax/Sinkhorn epsilon
ITERS = 20                   # config.hc_sinkhorn_iters


def lcg(seed):
    """A tiny deterministic generator, so the fixture is reproducible from this file."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2 ** 31)
        yield (state / (2 ** 31)) * 2.0 - 1.0


def make(shape, gen, scale=1.0):
    if not shape:
        return next(gen) * scale
    return [make(shape[1:], gen, scale) for _ in range(shape[0])]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def unweighted_rms_norm(vec, eps):
    """`x * rsqrt(x.square().mean(-1, keepdim=True) + eps)` — no learned gain."""
    mean_sq = sum(v * v for v in vec) / len(vec)
    inv = 1.0 / math.sqrt(mean_sq + eps)
    return [v * inv for v in vec]


def hyper_connection(streams, fn, base, scale):
    """`Glm5NextTextHyperConnection.forward` — returns (post, comb, collapsed).

    `streams` is [B][L][HC][D]; `fn` is [MIX][HC*D]; `base` is [MIX]; `scale` is [3].
    """
    pre_scale, post_scale, comb_scale = scale
    pre_b = base[0:HC]
    post_b = base[HC:2 * HC]
    comb_b = base[2 * HC:]

    post_out, comb_out, collapsed_out = [], [], []
    for b in range(B):
        post_row, comb_row, collapsed_row = [], [], []
        for t in range(L):
            streams_bt = streams[b][t]                       # [HC][D]
            flat = [v for stream in streams_bt for v in stream]   # flatten(start_dim=2)
            normed = unweighted_rms_norm(flat, RMS_EPS)

            mix = [sum(normed[k] * fn[m][k] for k in range(HC * D)) for m in range(MIX)]
            pre_w = mix[0:HC]
            post_w = mix[HC:2 * HC]
            comb_w = mix[2 * HC:]

            pre = [sigmoid(pre_w[i] * pre_scale + pre_b[i]) + HC_EPS for i in range(HC)]
            post = [2.0 * sigmoid(post_w[i] * post_scale + post_b[i]) for i in range(HC)]

            logits = [[comb_w[i * HC + j] * comb_scale + comb_b[i * HC + j]
                       for j in range(HC)] for i in range(HC)]
            comb = [[v + HC_EPS for v in softmax(row)] for row in logits]

            # The reference normalises COLUMNS first, then runs (iters - 1) rounds of
            # {rows, columns}.  The leading column pass is easy to lose when reading
            # the loop alone, and it changes the result.
            comb = col_normalize(comb)
            for _ in range(ITERS - 1):
                comb = row_normalize(comb)
                comb = col_normalize(comb)

            # `(pre.unsqueeze(-1) * hidden_streams).sum(dim=2)` — weights the ORIGINAL
            # streams, not the normalised flat vector.
            collapsed = [sum(pre[h] * streams_bt[h][d] for h in range(HC)) for d in range(D)]

            post_row.append(post)
            comb_row.append(comb)
            collapsed_row.append(collapsed)
        post_out.append(post_row)
        comb_out.append(comb_row)
        collapsed_out.append(collapsed_row)
    return post_out, comb_out, collapsed_out


def row_normalize(m):
    out = []
    for row in m:
        s = sum(row) + HC_EPS
        out.append([v / s for v in row])
    return out


def col_normalize(m):
    sums = [sum(m[i][j] for i in range(len(m))) + HC_EPS for j in range(len(m[0]))]
    return [[m[i][j] / sums[j] for j in range(len(m[0]))] for i in range(len(m))]


def expand(block_out, residual, post, comb):
    """`post.unsqueeze(-1) * out.unsqueeze(-2) + matmul(comb.transpose(-1, -2), residual)`.

    Note the TRANSPOSE on `comb`: stream `i` of the result gathers column `i` of the
    mixing matrix, not row `i`.  Sinkhorn leaves `comb` doubly stochastic but NOT
    symmetric, so dropping the transpose is a silent, plausible-looking error.
    """
    out = []
    for b in range(B):
        row = []
        for t in range(L):
            combT = [[comb[b][t][j][i] for j in range(HC)] for i in range(HC)]
            mixed = [[sum(combT[i][j] * residual[b][t][j][d] for j in range(HC))
                      for d in range(D)] for i in range(HC)]
            row.append([[post[b][t][i] * block_out[b][t][d] + mixed[i][d]
                         for d in range(D)] for i in range(HC)])
        out.append(row)
    return out


def hyper_head(streams):
    """`Glm5NextTextHyperHead.forward` — an UNWEIGHTED MEAN over the stream axis.

    The reference's own docstring: "Unlike DeepSeek-V4, this is an unweighted mean."
    DeepSeek-V4 reduces with a learned head; GLM ships no such weights, which is what
    made a parameter-free reduce inferable — but mean and sum are both parameter-free,
    and only this line settles which.
    """
    return [[[sum(streams[b][t][h][d] for h in range(HC)) / HC for d in range(D)]
             for t in range(L)] for b in range(B)]


def main():
    gen = lcg(20260830)
    streams = make([B, L, HC, D], gen)
    fn = make([MIX, HC * D], gen, scale=0.5)
    base = make([MIX], gen, scale=0.3)
    scale = [0.8, 1.2, 0.9]

    # One full attention-site bracketing with the sublayer as the identity.
    post, comb, collapsed = hyper_connection(streams, fn, base, scale)
    after_block = expand(collapsed, streams, post, comb)
    reduced = hyper_head(after_block)

    fixture = {
        "shape": {"B": B, "L": L, "hc": HC, "hidden": D},
        "eps": {"rms": RMS_EPS, "hc": HC_EPS, "iters": ITERS},
        "in": {"streams": streams, "fn": fn, "base": base, "scale": scale},
        "out": {
            "collapsed": collapsed,
            "post": post,
            "comb": comb,
            "afterBlock": after_block,
            "reducedMean": reduced,
        },
    }
    print(json.dumps(fixture, separators=(",", ":")))


if __name__ == "__main__":
    main()
