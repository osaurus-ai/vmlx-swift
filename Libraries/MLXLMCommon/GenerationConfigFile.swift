// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
public struct GenerationConfigFile: Codable, Equatable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var maxNewTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?
    public var doSample: Bool?
    public var suppressTokens: [Int]?

    public init(
        eosTokenIds: IntOrIntArray? = nil,
        maxNewTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        doSample: Bool? = nil,
        suppressTokens: [Int]? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.doSample = doSample
        self.suppressTokens = suppressTokens
    }

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case maxNewTokens = "max_new_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case doSample = "do_sample"
        case suppressTokens = "suppress_tokens"
    }
}
