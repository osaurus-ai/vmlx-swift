import Foundation
import MLX
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif

/// Explicit selection context for the canonical namespace only. Ordinary
/// reasoning/media/request salts remain unchanged. Unknown context is ineligible.
public struct CanonicalRequiredToolContext: Sendable, Equatable {
    public let selectedName: String?
    private let catalogDigest: String

    public init?(additionalContext: [String: any Sendable]?, tools: [ToolSpec]?) {
        guard additionalContext?["tool_choice"] as? String == "required",
              let tools, let digest = Self.digest(tools) else { return nil }
        if let raw = additionalContext?["tool_choice_name"] {
            guard let name = raw as? String, !name.isEmpty,
                  tools.count == 1,
                  (tools[0]["function"] as? [String: any Sendable])?["name"] as? String == name
            else { return nil }
            selectedName = name
        } else { selectedName = nil }
        catalogDigest = digest
    }

    public func matchesCatalog(_ tools: [ToolSpec]?) -> Bool {
        guard let tools else { return false }
        return Self.digest(tools) == catalogDigest
    }

    private static func digest(_ tools: [ToolSpec]) -> String? {
        guard !tools.isEmpty else { return nil }
        var names = Set<String>()
        for tool in tools {
            guard tool["type"] as? String == "function",
                  let function = tool["function"] as? [String: any Sendable],
                  let name = function["name"] as? String, !name.isEmpty,
                  names.insert(name).inserted else { return nil }
        }
        // Dictionary order is canonical; catalog/parameter array order is preserved.
        let object = tools.map { $0.mapValues { $0 as Any } }
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func requestSalt(ordinarySalt: String?, modelIdentity: String) -> String {
        let scope = selectedName.map { "named:\($0.utf8.count):\($0)" } ?? "required"
        let ordinary = ordinarySalt.map { "some:\($0.utf8.count):\($0)" } ?? "none"
        let value = "canonical-required-tool-v1:\(modelIdentity):\(scope):\(catalogDigest):\(ordinary)"
        return SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Adopted only by a model that owns a qualified solo cold chunk schedule and
/// complete typed state. Does not enable ordinary, batched, MTP or media restore.
public protocol CanonicalRequiredToolCacheModel: LanguageModel {
    var canonicalRequiredToolCacheIdentity: String { get }
    /// Opt-in to request-local ordinary stable N-1 reconstruction only.
    /// This never admits an ordinary canonical disk restore.
    var supportsOrdinaryStablePrefixRederive: Bool { get }
    func canonicalRequiredToolChunkSize(parameters: GenerateParameters) -> Int?
    func validateCanonicalRequiredToolCache(_ cache: [KVCache], boundary: Int) -> Bool
}

public extension CanonicalRequiredToolCacheModel {
    var supportsOrdinaryStablePrefixRederive: Bool { false }
}

/// Select one missing processor-stable N-1 target without changing cold prepare
/// geometry. Required selection and warm/unproven origins keep their old paths.
func canonicalOrdinaryStableRederiveTarget(
    input: LMInput, inputForPrepare: LMInput, promptTokens: [Int], cache: [KVCache],
    model: any CanonicalRequiredToolCacheModel, parameters: GenerateParameters,
    structuralBoundary: Int?, missingTargets: [Int]
) -> Int? {
    guard !Task.isCancelled, model.supportsOrdinaryStablePrefixRederive,
          input.cacheRestorePolicy == .standard, input.cachePromptIntent == .generation,
          input.canonicalRequiredToolContext == nil,
          !input.hasMediaContent, !input.requiresPostPrepareCacheKey,
          input.text.mask == nil, inputForPrepare.text.mask == nil,
          !inputForPrepare.hasMediaContent, !inputForPrepare.requiresPostPrepareCacheKey,
          input.text.tokenIds == promptTokens, input.text.tokens.size == promptTokens.count,
          inputForPrepare.text.tokenIds == promptTokens,
          inputForPrepare.text.tokens.size == promptTokens.count,
          let chunk = model.canonicalRequiredToolChunkSize(parameters: parameters),
          chunk == parameters.prefillStepSize,
          model.validateCanonicalRequiredToolCache(cache, boundary: 0),
          let structuralBoundary, structuralBoundary > chunk,
          structuralBoundary < promptTokens.count
    else { return nil }
    return missingTargets.filter {
        $0 >= chunk && $0 < structuralBoundary
            && input.cacheStablePrefixTokenCounts.contains($0 + 1)
    }.min()
}

func canonicalRequiredToolSalt(input: LMInput, model: any CanonicalRequiredToolCacheModel,
                               parameters: GenerateParameters, cache: [KVCache], ordinarySalt: String?) -> String? {
    guard input.cacheRestorePolicy == .freshRequiredToolSelection,
          input.cachePromptIntent == .generation,
          !input.hasMediaContent, !input.requiresPostPrepareCacheKey,
          let ids = input.text.tokenIds, ids.count == input.text.tokens.size,
          input.text.mask == nil,
          let scope = input.canonicalRequiredToolContext, scope.matchesCatalog(input.toolSchemas),
          model.canonicalRequiredToolChunkSize(parameters: parameters) != nil,
          model.validateCanonicalRequiredToolCache(cache, boundary: cache.first?.offset ?? 0)
    else { return nil }
    return scope.requestSalt(ordinarySalt: ordinarySalt, modelIdentity: model.canonicalRequiredToolCacheIdentity)
}
