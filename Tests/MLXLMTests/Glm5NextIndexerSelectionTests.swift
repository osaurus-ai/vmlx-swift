// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// GLM-5.3's DSA indexer: the k-pool selection math, against the reference.
//
// Below `index_topk` the model skips selection entirely, because selecting the top 2048 keys out of
// at most 2048 IS selecting all of them. That equivalence is asserted here rather than argued, and
// it is also why the rest of this suite runs at a deliberately tiny `index_topk`: at the shipped
// value nothing is ever discarded, and a test where the mechanism cannot bite proves only that it
// compiles.
//
// The fixture comes from `scripts/glm5-indexer-oracle.py`, a transcription of
// `Glm5NextTextIndexer` in huggingface/transformers `models/glm5_next/modeling_glm5_next.py`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("GLM-5.3 indexer k-pool selection")
struct Glm5NextIndexerSelectionTests {

    struct Fixture: Codable {
        struct Shape: Codable {
            let B: Int; let S: Int; let N: Int; let D: Int; let H: Int; let K: Int
            let topk: Int; let hidden: Int
        }
        struct Inputs: Codable {
            let hidden: [[[Float]]]
            let q: [[[[Float]]]]
            let keys: [[[Float]]]
            let gate: [[[Float]]]
            let ape: [[Float]]
            let weightsProj: [[Float]]
        }
        struct Outputs: Codable {
            let poolKeys: [[Float]]
            let poolIndices: [[Int]]
            let poolLayout: [[Int]]
            let poolValid: [Bool]
            let topk: [[[Int]]]
            let mask: [[[Bool]]]
        }
        let shape: Shape
        let `in`: Inputs
        let out: Outputs
    }

    static let fixture: Fixture = {
        // swift-format-ignore
        let json = #"""
{"shape":{"B":1,"S":6,"N":6,"D":4,"H":2,"K":2,"topk":4,"hidden":4},"in":{"hidden":[[[-0.3240899629890919,-0.909937153570354,-0.9567803051322699,-0.8292000340297818],[-0.7063715234398842,-0.7497755428776145,0.1064127292484045,0.9876832580193877],[0.45567435398697853,0.3801688374951482,-0.15016487054526806,0.08985673170536757],[-0.6972405463457108,-0.32460936065763235,-0.15538902767002583,-0.939588830806315],[-0.8264826871454716,-0.993581916205585,0.3108358923345804,-0.11560036707669497],[0.603282518684864,0.4107563206925988,-0.1355967465788126,0.9778901999816298]]],"q":[[[[-0.38416124507784843,-0.4815754620358348,0.02554907090961933,-0.7556375535205007],[-0.00436440110206604,0.8485868209972978,-0.32338431663811207,0.5959476819261909]],[[0.2279740683734417,-0.08522323798388243,-0.3434658255428076,-0.6229870663955808],[0.7946609184145927,0.07621581945568323,0.6795255448669195,0.12758868094533682]],[[0.5126316882669926,-0.9272745726630092,0.7655206341296434,0.12414033990353346],[-0.39695750176906586,-0.819267145358026,-0.6302012074738741,0.1351834973320365]],[[0.17833059653639793,-0.07212918531149626,0.39934530295431614,-0.17075710464268923],[-0.1652563437819481,-0.6963291754946113,-0.6965724471956491,-0.7273448342457414]],[[-0.9621621780097485,0.40385031420737505,0.42588995583355427,0.9547153143212199],[-0.011555492877960205,-0.5543065136298537,-0.1933323759585619,-0.22233304474502802]],[[-0.34339405968785286,0.0920259254053235,-0.3799812104552984,-0.5509637044742703],[0.6709794625639915,0.021282956935465336,-0.5630240235477686,0.7138098692521453]]]],"keys":[[[0.7512105740606785,0.6811717571690679,0.4995159227401018,-0.13604397792369127],[0.3707747310400009,0.16342721227556467,0.19394825212657452,0.9627901455387473],[-0.6622121073305607,0.13716152776032686,0.9110228959470987,0.22168366145342588],[-0.01871418207883835,0.778307587839663,0.4802562650293112,-0.03338324371725321],[-0.3695278875529766,0.6326560648158193,0.3659761715680361,0.6320748655125499],[0.0744350254535675,-0.6500137215480208,0.8125855010002851,0.21978890802711248]]],"gate":[[[0.6898329965770245,0.2267908053472638,-0.8734553661197424,-0.34018072951585054],[0.9240489527583122,0.4950938383117318,0.28257261775434017,-0.48851645831018686],[0.8213133551180363,0.29486329574137926,0.04156708903610706,0.44162799697369337],[-0.7207040041685104,0.26751668099313974,-0.23225705511868,0.9177428325638175],[0.7236665450036526,0.7080206917598844,-0.8675102349370718,0.5534211499616504],[-0.11187615245580673,0.2130845794454217,-0.10755201615393162,0.5436616903170943]]],"ape":[[-0.30630301497876644,0.3814736292697489,0.4646519059315324,-0.1862423405982554],[-0.11465150117874146,-0.4128709170036018,-0.13059854600578547,-0.4922123705036938]],"weightsProj":[[0.5202278066426516,-0.09691369058564304,0.3895379608497023,0.503867754060775],[-0.5300160370767116,0.5913715026341378,0.21533684376627205,0.5062724919058382]]},"out":{"poolKeys":[[0.521090095937176,0.48886040527862024,0.3049816349619707,0.2906745430887113],[-0.5297734344858451,0.3329706064413511,0.7837542320353518,0.0833267299950643],[-0.21664150576700897,0.3556319814119717,0.6076293804196069,0.45820692813185127]],"poolIndices":[[0,1],[2,3],[4,5]],"poolLayout":[[0,1],[2,3],[4,5]],"poolValid":[true,true,true],"topk":[[[-1,-1,-1,-1,0],[0,1,-1,-1,-1],[0,1,-1,-1,2],[0,1,2,3,-1],[0,1,2,3,4],[0,1,2,3,-1]]],"mask":[[[true,false,false,false,false,false],[true,true,false,false,false,false],[true,true,true,false,false,false],[true,true,true,true,false,false],[true,true,true,true,true,false],[true,true,true,true,false,false]]]}}
"""#
        return try! JSONDecoder().decode(Fixture.self, from: Data(json.utf8))
    }()

    static func flat(_ x: [[Float]]) -> [Float] { x.flatMap { $0 } }
    static func flat(_ x: [[[Float]]]) -> [Float] { x.flatMap { flat($0) } }
    static func flat(_ x: [[[[Float]]]]) -> [Float] { x.flatMap { flat($0) } }

    static func makeIndexer(_ f: Fixture, topK: Int? = nil) -> Glm5NextIndexer {
        let ix = Glm5NextIndexer(
            numHeads: f.shape.H, headDim: f.shape.D, topK: topK ?? f.shape.topk,
            poolSize: f.shape.K, poolCompressed: true, alwaysSelectTail: true,
            hiddenSize: f.shape.hidden, qLoraRank: f.shape.hidden)
        ix.update(parameters: ModuleParameters.unflattened([
            "index_kpool_compress_ape": MLXArray(
                flat(f.in.ape), [f.shape.K, f.shape.D]),
            "weights_proj.weight": MLXArray(
                flat(f.in.weightsProj), [f.shape.H, f.shape.hidden]),
        ]))
        return ix
    }

    /// `[k | gate | valid]`, which is exactly what the indexer caches per position.
    static func packed(_ f: Fixture) -> MLXArray {
        let k = MLXArray(flat(f.in.keys), [f.shape.B, f.shape.N, f.shape.D])
        let g = MLXArray(flat(f.in.gate), [f.shape.B, f.shape.N, f.shape.D])
        let v = MLXArray.ones([f.shape.B, f.shape.N, 1], dtype: .float32)
        return concatenated([k, g, v], axis: -1)
    }

    /// The pooled key is a LEARNED weighted average within each pool — a softmax over the
    /// pool-member axis, per feature channel, with the positional code added to the gate scores.
    /// Getting the softmax axis wrong (over channels, say) produces the right shape and wrong
    /// numbers, which is the reason this is checked elementwise against the reference.
    @Test("pooled keys, indices and validity match the reference")
    func pooledStatesMatchReference() throws {
        let f = Self.fixture
        try MLXMetalTestLock.withLock {
            let ix = Self.makeIndexer(f)
            let (keys, indices, valid) = try ix.pooledStates(packed: Self.packed(f))
            eval(keys, indices, valid)

            let poolCount = f.out.poolValid.count
            #expect(keys.shape == [f.shape.B, poolCount, f.shape.D])

            let gotKeys = keys.asType(.float32).asArray(Float.self)
            let wantKeys = Self.flat(f.out.poolKeys)
            let dKeys = zip(gotKeys, wantKeys).map { abs($0 - $1) }.max() ?? 0
            #expect(dKeys < 1e-6, "pooled keys differ from the reference by \(dKeys)")

            // The SHARED (P, K) layout, not the reference's per-batch masked copy — see
            // `pooledStates`. Shape is asserted too: a stray batch axis carries the same values and
            // makes every downstream gather read the wrong dimension.
            #expect(indices.shape == [f.out.poolLayout.count, f.shape.K])
            #expect(indices.asArray(Int32.self) == f.out.poolLayout.flatMap { $0 }.map(Int32.init))
            #expect(valid.asArray(Bool.self) == f.out.poolValid)
        }
    }

    /// The whole selection: score, top-k, expand pools back to raw tokens, append the tail.
    @Test("selected token indices match the reference")
    func selectionMatchesReference() throws {
        let f = Self.fixture
        try MLXMetalTestLock.withLock {
            let ix = Self.makeIndexer(f)
            let q = MLXArray(
                Self.flat(f.in.q), [f.shape.B, f.shape.S, f.shape.H, f.shape.D])
            let hidden = MLXArray(
                Self.flat(f.in.hidden), [f.shape.B, f.shape.S, f.shape.hidden])
            let selected = try ix.selectTopK(
                packed: Self.packed(f), queries: q, hidden: hidden, queryOffset: 0)
            eval(selected)

            let want = f.out.topk.flatMap { $0.flatMap { $0 } }.map(Int32.init)
            #expect(selected.shape == [f.shape.B, f.shape.S, want.count / f.shape.S])
            #expect(selected.asArray(Int32.self) == want)

            // The mask the attention actually uses.
            let mask = ix.maskFromIndices(selected, kvLength: f.shape.N)
            eval(mask)
            #expect(mask.shape == [f.shape.B, 1, f.shape.S, f.shape.N])
            #expect(mask.asArray(Bool.self) == f.out.mask.flatMap { $0.flatMap { $0 } })
        }
    }

    /// Selection must never look forward, at any budget.
    ///
    /// The causal guarantee is structural — a pool is a candidate only when its LAST token is
    /// visible — so it is worth asserting separately from the reference comparison: a transcription
    /// error shared by both would satisfy the equality above and still attend to the future.
    @Test("no query ever selects a token ahead of itself")
    func selectionIsCausal() throws {
        let f = Self.fixture
        try MLXMetalTestLock.withLock {
            let ix = Self.makeIndexer(f)
            let q = MLXArray(
                Self.flat(f.in.q), [f.shape.B, f.shape.S, f.shape.H, f.shape.D])
            let hidden = MLXArray(
                Self.flat(f.in.hidden), [f.shape.B, f.shape.S, f.shape.hidden])
            let selected = try ix.selectTopK(
                packed: Self.packed(f), queries: q, hidden: hidden, queryOffset: 0)
            eval(selected)

            let width = selected.dim(-1)
            let values = selected.asArray(Int32.self)
            for s in 0 ..< f.shape.S {
                for w in 0 ..< width {
                    let v = Int(values[s * width + w])
                    #expect(v <= s, "query \(s) selected future token \(v)")
                }
            }
        }
    }

    /// The claim the fast path rests on: with a sequence that fits inside `index_topk`, selection
    /// reproduces the CAUSAL mask exactly — so skipping it is an equivalence, not an approximation.
    ///
    /// This is the assertion behind `Glm5NextSparseAttention` running full attention below the
    /// threshold. It is also where the dependence on `index_kpool_always_select_tail` shows: without
    /// the tail, the tokens of the incomplete trailing pool are unreachable and the mask is strictly
    /// smaller than causal.
    @Test("below index_topk the selection is exactly the causal mask")
    func selectionEqualsCausalBelowTopK() throws {
        let f = Self.fixture
        try MLXMetalTestLock.withLock {
            // Generous budget: pool count (3) <= topK / poolSize, so nothing is discarded.
            let ix = Self.makeIndexer(f, topK: 64)
            let q = MLXArray(
                Self.flat(f.in.q), [f.shape.B, f.shape.S, f.shape.H, f.shape.D])
            let hidden = MLXArray(
                Self.flat(f.in.hidden), [f.shape.B, f.shape.S, f.shape.hidden])
            let selected = try ix.selectTopK(
                packed: Self.packed(f), queries: q, hidden: hidden, queryOffset: 0)
            let mask = ix.maskFromIndices(selected, kvLength: f.shape.N)
            eval(mask)

            let got = mask.asArray(Bool.self)
            var causal = [Bool]()
            for s in 0 ..< f.shape.S {
                for kv in 0 ..< f.shape.N { causal.append(kv <= s) }
            }
            #expect(got == causal, "a full-budget selection is not the causal mask")

            // And confirm the small-budget fixture really is different, or the check above would
            // hold no matter what the selection did.
            let tight = Self.makeIndexer(f)
            let tightMask = tight.maskFromIndices(
                try tight.selectTopK(
                    packed: Self.packed(f), queries: q, hidden: hidden, queryOffset: 0),
                kvLength: f.shape.N)
            eval(tightMask)
            #expect(
                tightMask.asArray(Bool.self) != causal,
                "the tight-budget selection also equals causal — the fixture does not sparsify")
        }
    }
}
