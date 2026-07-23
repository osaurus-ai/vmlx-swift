// Copyright © 2026 Osaurus AI
//
// Media salt: a stable fingerprint of VLM image/video/audio inputs for cache keying.
//
// Problem
// -------
// The multi-tier cache coordinator keys its lookups by the flat text token list.
// For VLMs, the model replaces placeholder `<image>` tokens with the vision
// tower's output embeddings during forward pass — so the same token prefix
// with a different image produces different KV state. Naive text-only keying
// would yield false-positive cache hits with the wrong KV for image positions,
// which is why `TokenIterator.init` historically bypassed the cache entirely
// whenever `input.image != nil || input.video != nil`.
//
// Bypassing the cache is extremely expensive: every turn of a multi-turn VLM
// chat re-runs the full vision tower AND re-prefills every prior text token,
// even though neither has changed. On Gemma 4 VLM at 2K context that's
// hundreds of milliseconds to seconds of wasted work per turn.
//
// Fix
// ---
// Compute a SHA256 fingerprint of the raw pixel/audio bytes + shape + dtype of
// any image/video/audio input, and pass it alongside the token list to the cache tiers.
// Each tier mixes the salt into its internal hash (same way modelKey is
// mixed), so "same text prefix + same image" cache-hits while "same text +
// different image" misses, as required.
//
// The salt is computed once per TokenIterator init, stored on the iterator,
// and reused at store time. Cost: one SHA256 pass over the pixel tensor
// (~a few milliseconds for a typical 448x448 image), negligible vs vision
// tower forward (~100 ms).

import CryptoKit
import Foundation
import MLX

/// Computes a stable fingerprint for the media portion of an ``LMInput``.
///
/// Returns `nil` when the input has no image, video, or audio — callers can
/// then fall through the cache coordinator exactly as they did for text-only
/// inputs, preserving byte-for-byte behavior on text-only paths.
///
/// The fingerprint is a lowercase hex SHA256 of:
/// - The literal UTF-8 tag `"image:"` (if an image is present) followed by
///   shape, dtype, and raw contiguous pixel bytes
/// - The literal UTF-8 tag `"video:"` (if a video is present) followed by
///   shape, dtype, and raw contiguous pixel bytes
/// - The literal UTF-8 tag `"audio:"` (if audio is present) followed by
///   sample rate, shape, dtype, and raw contiguous waveform bytes
///
/// Shape + dtype are hashed before bytes so different-shaped tensors with
/// identical bytes never collide. Ordering is deterministic: image before
/// video. The `frames: [THW]?` field is ignored because it is recoverable
/// from the pixel tensor shape.
///
/// Thread-safety: safe to call concurrently; CryptoKit's SHA256 is
/// value-typed and the underlying MLXArray reads are via `asData` which
/// takes a read-only snapshot.
public func computeMediaSalt(for input: LMInput) -> String? {
    let hasImage = input.image != nil
    let hasVideo = input.video != nil
    let hasAudio = input.audio != nil
    guard hasImage || hasVideo || hasAudio else { return nil }

    var hasher = SHA256()

    if let image = input.image {
        hasher.update(data: Data("image:".utf8))
        hashMLXArray(image.pixels, into: &hasher)
    }
    if let video = input.video {
        hasher.update(data: Data("video:".utf8))
        hashMLXArray(video.pixels, into: &hasher)
    }
    if let audio = input.audio {
        // Same shape+dtype+bytes treatment as image/video. Hash the
        // waveform — NOT the (optional) pre-encoded embedding — so
        // logically-identical inputs produced via different encoder
        // paths still cache-collide correctly. SR is mixed in too:
        // identical bytes at different SR are different inputs.
        hasher.update(data: Data("audio:".utf8))
        var sr = Int64(audio.sampleRate)
        withUnsafeBytes(of: &sr) { hasher.update(bufferPointer: $0) }
        hashMLXArray(audio.waveform, into: &hasher)
    }

    let digest = hasher.finalize()
    return digest.map { String(format: "%02x", $0) }.joined()
}

/// Feeds an MLXArray's shape, dtype, and raw contiguous bytes into an
/// in-progress SHA256 hasher.
///
/// Uses `asData(access: .noCopyIfContiguous)` so the common case of a
/// contiguous pixel tensor avoids any allocation. The data accessor
/// materializes any pending lazy computation before reading bytes — this is
/// safe for already-materialized pixel tensors and a necessary cost for
/// lazy ones.
private func hashMLXArray(_ array: MLXArray, into hasher: inout SHA256) {
    // Shape: fixed-count Int header followed by each dim as an Int64. Hashing
    // the dim count first makes rank-2 [3, 448*448] distinguishable from
    // rank-4 [1, 3, 448, 448] even if the byte blob would otherwise match.
    var dimCount = Int64(array.shape.count)
    withUnsafeBytes(of: &dimCount) { hasher.update(bufferPointer: $0) }
    for dim in array.shape {
        var d = Int64(dim)
        withUnsafeBytes(of: &d) { hasher.update(bufferPointer: $0) }
    }

    // Dtype: hash its string representation (stable across runs).
    hasher.update(data: Data(String(describing: array.dtype).utf8))

    // Pixel bytes: noCopy when contiguous (zero allocation on the hot path).
    let data = array.asData(access: .noCopyIfContiguous)
    hasher.update(data: data.data)
}

/// Derive a cache-scope salt string from a request's `additionalContext`.
///
/// Recognized scopes are intentionally narrow: only context keys that can
/// change KV state without being represented by the concrete token prefix are
/// folded into the cache key.
///
/// - Returns:
///   - `"reasoning=off"` when `enable_thinking == false`
///   - `"reasoning=on"`  when `enable_thinking == true`
///   - `"effort=<value>"` when `reasoning_effort` is present
///   - a `|`-joined composition when multiple keys are present
///   - `nil` when no recognized semantic key is present (so unrelated metadata
///     in `additionalContext` does not fragment cache keys)
///
/// VLM/LM processors call this once when constructing `LMInput`, so the
/// cache coordinator sees the same salt across prefill and every decode
/// step within the request without re-deriving it.
public func cacheScopeSalt(from additionalContext: [String: any Sendable]?) -> String? {
    guard let additionalContext else { return nil }
    var parts: [String] = []
    if let thinking = additionalContext["enable_thinking"] as? Bool {
        parts.append(thinking ? "reasoning=on" : "reasoning=off")
    }
    if let effort = additionalContext["reasoning_effort"] as? String {
        let normalized = effort.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !normalized.isEmpty {
            parts.append("effort=\(normalized)")
        }
    }
    return parts.isEmpty ? nil : parts.joined(separator: "|")
}

/// Combine the media-bytes fingerprint and the request-scope salt into a
/// single cache-coordinator salt.
///
/// Returns `nil` only when both components are absent — pure text-only
/// prompts with no scope flags fall through to the legacy text-keyed path
/// without paying any extra hashing cost.
///
/// When either side is present the result is a stable hex SHA256 over a
/// tag-prefixed concatenation, so:
/// - identical media + identical reasoning scope → identical salt (cache hit),
/// - identical media + different reasoning scope → different salt (key isolation),
/// - identical scope + different media → different salt (false-positive
///   protection that already existed via `computeMediaSalt`).
public func computeCacheSalt(for input: LMInput) -> String? {
    let media = computeMediaSalt(for: input)
    let scope = input.cacheScopeSalt
    if media == nil && scope == nil { return nil }

    var hasher = SHA256()
    if let media {
        hasher.update(data: Data("media:".utf8))
        hasher.update(data: Data(media.utf8))
    }
    if let scope {
        hasher.update(data: Data("scope:".utf8))
        hasher.update(data: Data(scope.utf8))
    }
    let digest = hasher.finalize()
    return digest.map { String(format: "%02x", $0) }.joined()
}

/// Compute the cache-coordinator salt for a concrete request.
///
/// The media/scope salt isolates VLM bytes and chat-template flags such as
/// reasoning mode. The KV-policy salt below isolates cache representations:
/// disk/paged entries created with TurboQuant KV, affine KV, rotating-window
/// caps, or a previous serializer contract must not be restored into a request
/// using different cache semantics.
public func computeCacheSalt(for input: LMInput, parameters: GenerateParameters) -> String? {
    let base = computeCacheSalt(for: input)
    let policy = cachePolicySalt(for: parameters)

    var hasher = SHA256()
    if let base {
        hasher.update(data: Data("base:".utf8))
        hasher.update(data: Data(base.utf8))
    }
    hasher.update(data: Data("policy:".utf8))
    hasher.update(data: Data(policy.utf8))
    let digest = hasher.finalize()
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func cachePolicySalt(for parameters: GenerateParameters) -> String {
    let kvBits = parameters.kvBits.map(String.init) ?? "none"
    let maxKV = parameters.maxKVSize.map(String.init) ?? "none"
    return [
        "cache-policy-v4",
        "promptBoundaryDisk=raw-kv|zaya-typed-tq-min44-v1",
        "kvMode=\(cachePolicyDescription(parameters.kvMode))",
        "kvBits=\(kvBits)",
        "kvGroup=\(parameters.kvGroupSize)",
        "qStart=\(parameters.quantizedKVStart)",
        "maxKV=\(maxKV)",
    ].joined(separator: "|")
}

private func cachePolicyDescription(_ mode: KVQuantizationMode) -> String {
    switch mode {
    case .none:
        return "none"
    case .affine(let bits, let groupSize):
        return "affine(bits:\(bits),group:\(groupSize))"
    case .turboQuant(let keyBits, let valueBits):
        return "turboQuant(k:\(keyBits),v:\(valueBits))"
    }
}
