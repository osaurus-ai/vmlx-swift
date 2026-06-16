// Copyright © 2026 Jinho Jang. All rights reserved.
//
// ShardingPlan — declarative description of which `Linear`s in a model
// graph become tensor-parallel `AllToShardedLinear` / `ShardedToAllLinear`
// variants when distributed across a `Group`.
//
// Mirrors the shape of `MLXLMCommon.BaseConfiguration.PerLayerQuantization`
// (which keys quantization decisions by module path with a global default).
// The same module-path keys are used here, so a future config-driven
// per-layer sharding override is a one-line `LinearShardingDirective` map
// addition.
//
// **Why declarative**: every model family (Llama, Mistral, Gemma, Qwen3.x,
// MoE families) needs the same conceptual mapping ("output-shard the
// q/k/v/gate/up; input-shard with all-reduce the o/down"), but with
// different module-path keys per family. Declarative lets us add new
// families with a single `ShardingPlans+<Family>.swift` file containing
// nothing but a static factory — no traversal logic per family.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Directives

/// What to do with one named `Linear` in a sharded forward pass.
public enum LinearShardingDirective: Sendable, Equatable {
    /// Replace with `AllToShardedLinear` — each rank holds a row-shard of
    /// the weight, output is sharded across the group along the output
    /// axis. Use for projections that produce sharded heads (q/k/v) or
    /// sharded MLP intermediate channels (gate/up).
    case allToSharded(segments: Int)

    /// Replace with `ShardedToAllLinear` — each rank holds a column-shard
    /// of the weight, partial outputs are summed via all-reduce. Use for
    /// projections that consume sharded inputs and produce a full-width
    /// output (o, down).
    case shardedToAll(segments: Int)

    /// Leave unsharded. Default for any path not in the directive map.
    case replicated
}

public enum ParameterShardingDirective: Sendable, Equatable {
    /// Slice a tensor along `axis`, optionally preserving logical segments
    /// before taking the rank-local slice within each segment.
    case slice(axis: Int, segments: Int)
}

// MARK: - Plan

/// Sharding plan for one model family. Keys are MLXNN module-path
/// suffixes (the same path Module.modules() returns for nested children).
/// The walker matches a directive against any path that **ends with** the
/// directive key, so a single key like `"self_attn.q_proj"` covers every
/// transformer layer's query projection without enumerating layer indices.
///
/// To create a plan for a new family, write a static factory in a sibling
/// `ShardingPlans+<Family>.swift` file that returns a `ShardingPlan`
/// instance. The walker handles the rest.
public struct ShardingPlan: Sendable, Equatable {

    /// Map of module-path suffix → directive. The walker matches a
    /// module's full path against each key by `hasSuffix`, picking the
    /// longest match if multiple directives match. Paths not in the map
    /// receive `.replicated` (no transform).
    public let directives: [String: LinearShardingDirective]
    public let parameterDirectives: [String: ParameterShardingDirective]

    public init(
        directives: [String: LinearShardingDirective],
        parameterDirectives: [String: ParameterShardingDirective] = [:]
    ) {
        self.directives = directives
        self.parameterDirectives = parameterDirectives
    }

    /// Total number of directives — useful for tests asserting that
    /// `apply` actually rewrote the expected number of Linears.
    public var directiveCount: Int { directives.count }

    // MARK: - Walker

    /// Walk `root.leafModules()`, replace every `Linear` whose
    /// dot-path ends with a directive key, and call
    /// `Module.update(modules:)` to commit the replacements.
    ///
    /// Returns the set of full module paths that were actually replaced.
    /// Callers (tests / load-time mutators) can compare this against the
    /// expected count for the model family to surface plan/model drift.
    ///
    /// - Note: the walker is a no-op when `group.size == 1` because the
    ///   TP variants on a size-1 group produce bit-identical outputs to
    ///   the dense `Linear`. Skipping the rewrite avoids paying the
    ///   weight-slice + concat overhead on the unsharded path.
    ///
    /// Implementation pattern matches `QuantizedLinear.quantize(...)` in
    /// MLXNN: flatten leaves → compactMap to replacement modules →
    /// `update(modules:)`. The dotted path passed to the directive
    /// matcher is exactly the flattened key (e.g.
    /// `"model.layers.0.self_attn.q_proj"`).
    @discardableResult
    public func apply(to root: Module, group: Group) -> Set<String> {
        guard group.isMultiRank else { return [] }
        var replaced: Set<String> = []
        // We need the full dotted path for suffix matching, which
        // `compactMapValues` on NestedDictionary doesn't surface — use
        // `flattened()` to get `[(path, Module)]`, build a list of
        // replacements, then `unflattened()` back into the structure
        // `update(modules:)` expects.
        let flat: [(String, Module)] = root.leafModules().flattened()
        var rewrites: [(String, Module)] = []
        for (path, m) in flat {
            guard let directive = bestDirectiveFor(path: path) else { continue }
            if let linear = m as? Linear,
               let replacement = Self.makeReplacement(for: linear, directive: directive, group: group)
            {
                rewrites.append((path, replacement))
                replaced.insert(path)
            } else if let quantizedSwitchLinear = m as? QuantizedSwitchLinear,
                      let replacement = Self.makeReplacement(
                        for: quantizedSwitchLinear, directive: directive, group: group)
            {
                rewrites.append((path, replacement))
                replaced.insert(path)
            } else if let switchLinear = m as? SwitchLinear,
                      let replacement = Self.makeReplacement(
                        for: switchLinear, directive: directive, group: group)
            {
                rewrites.append((path, replacement))
                replaced.insert(path)
            }
        }
        if !rewrites.isEmpty {
            root.update(modules: ModuleChildren.unflattened(rewrites))
        }
        let parameterRewrites = parameterReplacements(on: root, group: group)
        if !parameterRewrites.isEmpty {
            root.update(parameters: ModuleParameters.unflattened(parameterRewrites))
        }
        return replaced
    }

    /// Find the directive whose key has the longest `hasSuffix` match
    /// against `path`. Longer-suffix-wins lets a per-layer override
    /// (`"model.layers.0.mlp.down_proj"`) take precedence over the
    /// family-default (`"mlp.down_proj"`).
    private func bestDirectiveFor(path: String)
        -> LinearShardingDirective?
    {
        var best: (key: String, directive: LinearShardingDirective)?
        for (key, directive) in directives {
            if path.hasSuffix(key) {
                if best == nil || key.count > best!.key.count {
                    best = (key, directive)
                }
            }
        }
        return best?.directive
    }

    /// Construct the replacement TP variant from a dense `Linear` per
    /// the directive. Returns nil for `.replicated` (caller skips).
    private static func makeReplacement(
        for linear: Linear,
        directive: LinearShardingDirective,
        group: Group
    ) -> Module? {
        switch directive {
        case .allToSharded(let segments):
            return AllToShardedLinear.from(
                linear, group: group, segments: segments)
        case .shardedToAll(let segments):
            return ShardedToAllLinear.from(
                linear, group: group, segments: segments)
        case .replicated:
            return nil
        }
    }

    private static func makeReplacement(
        for linear: QuantizedSwitchLinear,
        directive: LinearShardingDirective,
        group: Group
    ) -> Module? {
        switch directive {
        case .allToSharded(let segments):
            return QuantizedAllToShardedSwitchLinear.from(linear, group: group, segments: segments)
        case .shardedToAll(let segments):
            return QuantizedShardedToAllSwitchLinear.from(linear, group: group, segments: segments)
        case .replicated:
            return nil
        }
    }

    private static func makeReplacement(
        for linear: SwitchLinear,
        directive: LinearShardingDirective,
        group: Group
    ) -> Module? {
        if ProcessInfo.processInfo.environment["TP_QUANTIZE_EXPERTS"] == "1" {  // [quantexp-plan2]
            let env = ProcessInfo.processInfo.environment
            let groupSize = Int(env["TP_MIMO_ROUTED_EXPERT_GROUP_SIZE"] ?? "") ?? 64
            let bits = Int(env["TP_MIMO_ROUTED_EXPERT_BITS"] ?? "") ?? 8
            let quantized = QuantizedSwitchLinear(
                linear, groupSize: groupSize, bits: bits, mode: .affine)
            switch directive {
            case .allToSharded(let segments):
                return QuantizedAllToShardedSwitchLinear.from(
                    quantized, group: group, segments: segments)
            case .shardedToAll(let segments):
                return QuantizedShardedToAllSwitchLinear.from(
                    quantized, group: group, segments: segments)
            case .replicated:
                return nil
            }
        }
        switch directive {
        case .allToSharded(let segments):
            return AllToShardedSwitchLinear.from(linear, group: group, segments: segments)
        case .shardedToAll(let segments):
            return ShardedToAllSwitchLinear.from(linear, group: group, segments: segments)
        case .replicated:
            return nil
        }
    }

    private func parameterReplacements(on root: Module, group: Group) -> [(String, MLXArray)] {
        guard group.isMultiRank else { return [] }
        var rewrites: [(String, MLXArray)] = []
        for (path, value) in root.parameters().flattened() {
            guard let directive = bestParameterDirectiveFor(path: path) else { continue }
            rewrites.append((path, Self.slice(value, directive: directive, group: group)))
        }
        return rewrites
    }

    private func bestParameterDirectiveFor(path: String)
        -> ParameterShardingDirective?
    {
        var best: (key: String, directive: ParameterShardingDirective)?
        for (key, directive) in parameterDirectives {
            if path.hasSuffix(key) {
                if best == nil || key.count > best!.key.count {
                    best = (key, directive)
                }
            }
        }
        return best?.directive
    }

    private static func slice(
        _ value: MLXArray,
        directive: ParameterShardingDirective,
        group: Group
    ) -> MLXArray {
        switch directive {
        case .slice(let axis, let segments):
            precondition(axis >= 0 && axis < value.shape.count, "parameter slice axis out of range")
            precondition(segments >= 1, "segments must be >= 1")
            precondition(value.dim(axis) % segments == 0, "parameter dim must be divisible by segments")

            let perSegment = value.dim(axis) / segments
            precondition(perSegment % group.size == 0, "(parameter dim / segments) must divide group size")
            let perRank = perSegment / group.size

            var chunks: [MLXArray] = []
            for segment in 0 ..< segments {
                let segmentStart = segment * perSegment
                let rankStart = segmentStart + group.rank * perRank
                let rankEnd = rankStart + perRank
                chunks.append(sliceAxis(value, axis: axis, start: rankStart, end: rankEnd))
            }
            return concatenated(chunks, axis: axis)
        }
    }

    private static func sliceAxis(_ value: MLXArray, axis: Int, start: Int, end: Int) -> MLXArray {
        switch axis {
        case 0:
            return value[start ..< end]
        case 1:
            return value[0..., start ..< end]
        case 2:
            return value[0..., 0..., start ..< end]
        default:
            preconditionFailure("parameter slicing currently supports axes 0...2")
        }
    }
}
