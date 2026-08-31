#!/usr/bin/env python3
"""Differential oracle for GLM-5.3's DSA indexer with k-pool compression.

A line-by-line transcription of `Glm5NextTextIndexer` in huggingface/transformers,
`src/transformers/models/glm5_next/modeling_glm5_next.py` — `get_pooled_states`,
the scoring and top-k selection in `forward`, `append_visible_tail`, and the mask
built by `build_attention_mask_from_topk`.

Pure Python, no numpy: the tensors are tiny and a dependency-free transcription is
one fewer place for broadcasting I did not intend.

The fixture deliberately sets `index_topk` far below the sequence length so that
selection actually BITES. At the shipped 2048 the selection is a no-op for any
sequence that fits, which is true and useful — and useless as a test.

    python3 scripts/glm5-indexer-oracle.py
"""

import json
import math

B, S, N = 1, 6, 6        # batch, query length, kv length (single prefill pass)
D = 4                    # index_head_dim
H = 2                    # index_n_heads
K = 2                    # index_kpool
TOPK = 4                 # index_topk — small on purpose, so select_k < pool count
ALWAYS_TAIL = True       # index_kpool_always_select_tail


def lcg(seed):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2 ** 31)
        yield (state / (2 ** 31)) * 2.0 - 1.0


def make(shape, gen, scale=1.0):
    if not shape:
        return next(gen) * scale
    return [make(shape[1:], gen, scale) for _ in range(shape[0])]


def softmax(row):
    finite = [v for v in row if v != float("-inf")]
    if not finite:
        return [float("nan")] * len(row)
    m = max(finite)
    exps = [0.0 if v == float("-inf") else math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def pooled_states(keys, gate, valid, ape):
    """`get_pooled_states`: pools start at the FIRST VALID token, not slot 0."""
    pool_count = (N + K - 1) // K
    first_key = next((i for i, v in enumerate(valid[0]) if v), N)

    pool_indices = [[first_key + p * K + j for j in range(K)] for p in range(pool_count)]
    pool_keys, pool_valid, masked_indices = [], [], []
    for p in range(pool_count):
        raw = pool_indices[p]
        safe = [min(max(i, 0), N - 1) for i in raw]
        grouped_valid = [valid[0][safe[j]] and raw[j] < N for j in range(K)]
        pool_valid.append(all(grouped_valid))
        masked_indices.append([raw[j] if grouped_valid[j] else -1 for j in range(K)])

        # A LEARNED weighted average inside the pool: softmax over the POOL-MEMBER
        # axis, independently per feature channel.
        probs = []
        for d in range(D):
            column = [
                gate[0][safe[j]][d] + ape[j][d] if grouped_valid[j] else float("-inf")
                for j in range(K)
            ]
            sm = softmax(column)
            probs.append([0.0 if v != v else v for v in sm])   # nan_to_num
        pool_keys.append([
            sum(probs[d][j] * keys[0][safe[j]][d] for j in range(K)) for d in range(D)
        ])
    return pool_keys, masked_indices, pool_valid


def select(hidden, q, keys, gate, valid, ape, weights_proj):
    pool_keys, pool_indices, pool_valid = pooled_states(keys, gate, valid, ape)
    pool_count = len(pool_keys)

    visible = [[[kv <= s and valid[0][kv] for kv in range(N)] for s in range(S)]]

    scores = [[[[
        max(0.0, sum(q[0][s][h][d] * pool_keys[p][d] for d in range(D)) * D ** -0.5)
        for p in range(pool_count)] for h in range(H)] for s in range(S)]]

    weights = [[
        [sum(hidden[0][s][c] * weights_proj[h][c] for c in range(len(weights_proj[0])))
         * H ** -0.5 for h in range(H)]
        for s in range(S)]]

    index_scores = [[[
        sum(weights[0][s][h] * scores[0][s][h][p] for h in range(H))
        for p in range(pool_count)] for s in range(S)]]

    pool_end = [min(max(pool_indices[p][-1], 0), N - 1) for p in range(pool_count)]
    candidates = [[[visible[0][s][pool_end[p]] and pool_valid[p]
                    for p in range(pool_count)] for s in range(S)]]

    NEG = -3.4028234663852886e38
    for s in range(S):
        for p in range(pool_count):
            if not candidates[0][s][p]:
                index_scores[0][s][p] = NEG

    select_k = min(TOPK // K, pool_count)
    topk = []
    for s in range(S):
        order = sorted(range(pool_count), key=lambda p: -index_scores[0][s][p])
        chosen = order[:select_k]
        row = []
        for p in chosen:
            for j in range(K):
                row.append(pool_indices[p][j] if candidates[0][s][p] else -1)
        topk.append(row)

    width = TOPK
    if ALWAYS_TAIL and K > 1:
        first_key = next((i for i, v in enumerate(valid[0]) if v), N)
        for s in range(S):
            visible_count = sum(1 for kv in range(N) if visible[0][s][kv])
            tail_count = visible_count % K
            tail_start = first_key + visible_count - tail_count
            tail = []
            for j in range(K - 1):
                idx = tail_start + j
                ok = j < tail_count and idx < N and visible[0][s][min(max(idx, 0), N - 1)]
                tail.append(idx if ok else -1)
            topk[s] = topk[s] + tail
        width += K - 1

    for s in range(S):
        topk[s] = (topk[s] + [-1] * width)[:width]
    return [topk]


def mask_from(topk):
    """`build_attention_mask_from_topk`, boolean form."""
    return [[[any(0 <= i < N and i == kv for i in topk[0][s]) for kv in range(N)]
             for s in range(S)]]


def main():
    gen = lcg(20260831)
    hidden_size = 4
    hidden = make([B, S, hidden_size], gen)
    q = make([B, S, H, D], gen)
    keys = make([B, N, D], gen)
    gate = make([B, N, D], gen)
    ape = make([K, D], gen, scale=0.5)
    weights_proj = make([H, hidden_size], gen, scale=0.7)
    valid = [[True] * N]

    pool_keys, pool_indices, pool_valid = pooled_states(keys, gate, valid, ape)
    first = next((i for i, v in enumerate(valid[0]) if v), N)
    topk = select(hidden, q, keys, gate, valid, ape, weights_proj)

    print(json.dumps({
        "shape": {"B": B, "S": S, "N": N, "D": D, "H": H, "K": K,
                  "topk": TOPK, "hidden": hidden_size},
        "in": {"hidden": hidden, "q": q, "keys": keys, "gate": gate,
               "ape": ape, "weightsProj": weights_proj},
        "out": {"poolKeys": pool_keys, "poolIndices": pool_indices,
                "poolLayout": [[first + p * K + j for j in range(K)] for p in range(len(pool_keys))],
                "poolValid": pool_valid, "topk": topk, "mask": mask_from(topk)},
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
