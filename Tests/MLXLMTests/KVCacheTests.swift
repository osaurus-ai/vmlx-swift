import Foundation
import MLX
import MLXLMCommon
import Testing

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
]

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

private func makeQuantizedAttentionInputs(dtype: DType = .float16) -> (
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?)
) {
    let headDimension = 32
    let sequenceLength = 4
    let queries = MLXArray.zeros([1, 1, sequenceLength, headDimension], dtype: dtype)
    let keys = MLXArray(0 ..< sequenceLength * headDimension)
        .reshaped([1, 1, sequenceLength, headDimension]).asType(dtype)
    var valueData = [Float](repeating: 0, count: sequenceLength * headDimension)
    for index in 0 ..< sequenceLength {
        valueData[index * headDimension + index] = 1
    }
    let values = MLXArray(valueData, [1, 1, sequenceLength, headDimension]).asType(dtype)
    let cache = QuantizedKVCache(groupSize: 32, bits: 8)
    let (quantizedKeys, quantizedValues) = cache.updateQuantized(keys: keys, values: values)
    return (
        queries, keys, values,
        quantizedKeys,
        quantizedValues
    )
}

private func additiveMask(_ mask: MLXArray, dtype: DType) -> MLXArray {
    MLX.where(
        mask, MLXArray(0, dtype: dtype),
        MLXArray(dtype.finfo!.min, dtype: dtype))
}

@Test(.serialized)
func testQuantizedAttentionCausalMaskMatchesReference() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let input = makeQuantizedAttentionInputs()
    let scale = 1 / sqrt(Float(input.queries.dim(-1)))
    let output = quantizedScaledDotProductAttention(
        queries: input.queries,
        quantizedKeys: input.quantizedKeys,
        quantizedValues: input.quantizedValues,
        scale: scale,
        mask: .causal,
        groupSize: 32,
        bits: 8)
    let reference = MLXFast.scaledDotProductAttention(
        queries: input.queries,
        keys: input.keys,
        values: input.values,
        scale: scale,
        mask: .causal)
    eval(output, reference)

    #expect(output[0, 0, 0, 1 ..< 4].abs().max().item(Float.self) < 1e-3)
    #expect(allClose(output, reference, rtol: 0.05, atol: 0.01).item(Bool.self))
}

@Test(.serialized, arguments: [DType.float16, DType.bfloat16])
func testQuantizedAttentionBooleanArrayMaskMatchesReference(dtype: DType) {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let input = makeQuantizedAttentionInputs(dtype: dtype)
    let scale = 1 / sqrt(Float(input.queries.dim(-1)))
    let mask = MLXArray([
        true, false, false, false,
        true, true, false, false,
        true, true, true, false,
        true, true, true, true,
    ], [1, 1, 4, 4])
    let output = quantizedScaledDotProductAttention(
        queries: input.queries,
        quantizedKeys: input.quantizedKeys,
        quantizedValues: input.quantizedValues,
        scale: scale,
        mask: .array(mask),
        groupSize: 32,
        bits: 8)
    let reference = MLXFast.scaledDotProductAttention(
        queries: input.queries,
        keys: input.keys,
        values: input.values,
        scale: scale,
        mask: .array(additiveMask(mask, dtype: input.queries.dtype)))
    eval(output, reference)

    // Masked-position weight must sit at the 8-bit dequantization noise floor
    // (one affine quantum ~= 1/255), not at the ~0.25 softmax mass the old
    // positive fill produced; bf16 values carry a full quantum of noise.
    let maskedTolerance: Float = dtype == .bfloat16 ? 5e-3 : 1e-3
    #expect(output[0, 0, 0, 1 ..< 4].abs().max().item(Float.self) < maskedTolerance)
    #expect(allClose(output, reference, rtol: 0.05, atol: 0.01).item(Bool.self))
}

@Test(.serialized)
func testQuantizedAttentionFullyMaskedRowIsFinite() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let input = makeQuantizedAttentionInputs()
    let output = quantizedScaledDotProductAttention(
        queries: input.queries,
        quantizedKeys: input.quantizedKeys,
        quantizedValues: input.quantizedValues,
        scale: 1 / sqrt(Float(input.queries.dim(-1))),
        mask: .array(MLXArray.zeros([1, 1, 4, 4], dtype: .bool)),
        groupSize: 32,
        bits: 8)
    eval(output)

    #expect(isFinite(output).all().item(Bool.self))
}

@Test(.serialized)
func testQuantizedAttentionAdditiveMaskMatchesReference() {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let input = makeQuantizedAttentionInputs()
    let scale = 1 / sqrt(Float(input.queries.dim(-1)))
    let booleanMask = MLXArray([
        true, false, false, false,
        true, true, false, false,
        true, true, true, false,
        true, true, true, true,
    ], [1, 1, 4, 4])
    let mask = additiveMask(booleanMask, dtype: input.queries.dtype)
    let output = quantizedScaledDotProductAttention(
        queries: input.queries,
        quantizedKeys: input.quantizedKeys,
        quantizedValues: input.quantizedValues,
        scale: scale,
        mask: .array(mask),
        groupSize: 32,
        bits: 8)
    let reference = MLXFast.scaledDotProductAttention(
        queries: input.queries,
        keys: input.keys,
        values: input.values,
        scale: scale,
        mask: .array(mask))
    eval(output, reference)

    #expect(allClose(output, reference, rtol: 0.05, atol: 0.01).item(Bool.self))
}
