import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

/// These test PROPERTIES, not outputs.
///
/// A shape/smoke test on a conv module passes whether or not it is causal, and
/// passes whether or not chunked inference matches whole-sequence inference.
/// Both of those are wrong only on a live mic — which is precisely the class of
/// defect an offline test cannot see. So each test here perturbs the input and
/// asserts what must NOT change.
public class VoiceChatStreamingConformerTests: XCTestCase {

    private func randomWeights(_ conv: VoiceChatCausalDepthwiseConv, dim: Int, k: Int) {
        MLXRandom.seed(0)
        conv.update(parameters: ModuleParameters.unflattened([
            "depthwise_conv_weight": MLXRandom.normal([dim, 1, k]) * 0.5
        ]))
    }

    /// CAUSALITY: changing a LATE input frame must not change EARLY outputs.
    ///
    /// The symmetric padding in `NemotronHOmni/Parakeet.swift` fails this — it
    /// reads (K-1)/2 frames of future — which is exactly why VoiceChat needs its
    /// own conv rather than a flag on that one.
    func testCausalConvDoesNotLeakFuture() {
        let (B, T, D, K) = (1, 24, 8, 9)
        let conv = VoiceChatCausalDepthwiseConv(dim: D, kernelSize: K)
        randomWeights(conv, dim: D, k: K)

        MLXRandom.seed(1)
        let x = MLXRandom.normal([B, T, D])
        let base = conv(x)

        // Perturb the LAST frame only.
        var perturbed = x
        perturbed[0..., (T - 1) ..< T, 0...] = MLXRandom.normal([B, 1, D]) * 10.0
        let after = conv(perturbed)

        // Every output strictly before the perturbed frame must be untouched.
        let head = T - 1
        let delta = MLX.abs(base[0..., 0 ..< head, 0...] - after[0..., 0 ..< head, 0...])
        let maxDelta = delta.max().item(Float.self)
        XCTAssertLessThan(maxDelta, 1e-5,
            "future frame leaked into past outputs — conv is not causal")

        // Sanity: the perturbation must actually reach the LAST output, or the
        // test above is vacuous (it would also pass on a conv that ignores input).
        let lastDelta = MLX.abs(base[0..., head..., 0...] - after[0..., head..., 0...])
            .max().item(Float.self)
        XCTAssertGreaterThan(lastDelta, 1e-3, "perturbation had no effect at all")
    }

    /// CHUNK INVARIANCE: streaming in chunks with carried state must equal
    /// running the whole sequence at once.
    ///
    /// Without cross-chunk state each boundary zero-pads and injects an artifact
    /// once per chunk — periodic glitching that a whole-sequence test never sees.
    func testStreamingChunksMatchWholeSequence() {
        let (B, T, D, K) = (1, 30, 6, 9)
        let conv = VoiceChatCausalDepthwiseConv(dim: D, kernelSize: K)
        randomWeights(conv, dim: D, k: K)

        MLXRandom.seed(2)
        let x = MLXRandom.normal([B, T, D])

        conv.resetState()
        let whole = conv(x, streaming: false)

        // Same input, fed as uneven chunks with state carried across.
        conv.resetState()
        var pieces: [MLXArray] = []
        var cursor = 0
        for size in [7, 5, 11, 4, 3] {
            let end = min(cursor + size, T)
            pieces.append(conv(x[0..., cursor ..< end, 0...], streaming: true))
            cursor = end
        }
        let streamed = MLX.concatenated(pieces, axis: 1)

        XCTAssertEqual(streamed.dim(1), T)
        let maxDelta = MLX.abs(whole - streamed).max().item(Float.self)
        XCTAssertLessThan(maxDelta, 1e-4,
            "chunked streaming diverged from whole-sequence — cross-chunk state is wrong")
    }

    /// Without carried state, chunking MUST differ — proves the state is doing
    /// real work and the test above is not passing for a trivial reason.
    func testWithoutStateChunkingIsWrong() {
        let (B, T, D, K) = (1, 24, 6, 9)
        let conv = VoiceChatCausalDepthwiseConv(dim: D, kernelSize: K)
        randomWeights(conv, dim: D, k: K)

        MLXRandom.seed(3)
        let x = MLXRandom.normal([B, T, D])
        conv.resetState()
        let whole = conv(x, streaming: false)

        // streaming:false on every chunk => each boundary zero-pads.
        conv.resetState()
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: T, by: 8) {
            let end = min(start + 8, T)
            pieces.append(conv(x[0..., start ..< end, 0...], streaming: false))
        }
        let naive = MLX.concatenated(pieces, axis: 1)

        let maxDelta = MLX.abs(whole - naive).max().item(Float.self)
        XCTAssertGreaterThan(maxDelta, 1e-3,
            "stateless chunking should differ — if it matches, the state carry is a no-op")
    }

    /// The `[[70, 0]]` mask: 70 frames of history, ZERO lookahead.
    func testChunkedLimitedMaskMatchesShippedContext() {
        let T = 100
        let mask = VoiceChatAttentionMask.chunkedLimited(
            queryCount: T, leftContext: 70, rightContext: 0, dtype: .float32)
        XCTAssertEqual(mask.shape, [T, T])

        let neg = -Float.greatestFiniteMagnitude
        func allowed(_ q: Int, _ k: Int) -> Bool {
            mask[q, k].item(Float.self) != neg
        }

        // Self always allowed.
        XCTAssertTrue(allowed(80, 80))
        // Any future frame blocked — right context is 0.
        XCTAssertFalse(allowed(80, 81), "right context 0 must block ALL lookahead")
        XCTAssertFalse(allowed(80, 99))
        // Exactly 70 frames of history allowed, 71 back is not.
        XCTAssertTrue(allowed(80, 80 - 70))
        XCTAssertFalse(allowed(80, 80 - 71), "left context must be bounded at 70")
    }

    /// LayerNorm conv module runs and is shape-preserving.
    func testConvModuleShapeAndReset() {
        let (B, T, D) = (2, 16, 32)
        let m = VoiceChatConformerConvModule(dim: D, kernelSize: 9)
        MLXRandom.seed(4)
        let x = MLXRandom.normal([B, T, D])
        let y = m(x, streaming: false)
        XCTAssertEqual(y.shape, [B, T, D])
        m.resetState()
    }
}
