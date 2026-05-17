// Copyright © 2024 Apple Inc.

import Foundation

/// Configuration for a given model name with overrides for prompts and tokens.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
public struct ModelConfiguration: Sendable {

    public enum DirectoryError: LocalizedError, Equatable {
        case unresolvedModelDirectory(String)
        case unresolvedTokenizerDirectory(String)

        public var errorDescription: String? {
            switch self {
            case .unresolvedModelDirectory(let id):
                return "Model configuration '\(id)' has not been resolved to a local directory."
            case .unresolvedTokenizerDirectory(let id):
                return "Tokenizer source '\(id)' has not been resolved to a local directory."
            }
        }
    }

    public enum Identifier: Sendable {
        case id(String, revision: String = "main")
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let id, _):
            id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// The resolved local directory containing model files.
    ///
    /// - Throws: ``DirectoryError/unresolvedModelDirectory(_:)`` if this configuration still
    ///   identifies a remote model by ID rather than a local directory.
    package var modelDirectory: URL {
        get throws {
            switch id {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedModelDirectory(id)
            }
        }
    }

    /// The resolved local directory containing tokenizer files.
    ///
    /// If ``tokenizerSource`` is `nil`, this falls back to ``modelDirectory``.
    ///
    /// - Throws: ``DirectoryError/unresolvedTokenizerDirectory(_:)`` if the tokenizer still
    ///   points to a remote source by ID, or ``DirectoryError/unresolvedModelDirectory(_:)``
    ///   if no separate tokenizer source is set and the model itself is unresolved.
    package var tokenizerDirectory: URL {
        get throws {
            switch tokenizerSource {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedTokenizerDirectory(id)
            case nil:
                return try modelDirectory
            }
        }
    }

    /// Where to load the tokenizer from when it differs from the model directory.
    ///
    /// - `.id`: download from a remote provider (requires a ``Downloader``)
    /// - `.directory`: load from a local path
    /// - `nil`: use the same directory as the model
    public let tokenizerSource: TokenizerSource?

    /// A reasonable default prompt for the model
    public var defaultPrompt: String

    /// Additional tokens to use for end of string (specified as strings, converted to IDs at runtime)
    public var extraEOSTokens: Set<String>

    /// EOS token IDs loaded from config.json/generation_config.json
    public var eosTokenIds: Set<Int> = []

    /// Tool call format for this model (nil = default JSON format)
    public var toolCallFormat: ToolCallFormat?

    /// Reasoning-parser capability stamp for this model.
    ///
    /// When set, `TextToolTokenLoopHandler` / `BatchEngine.generate()` construct
    /// a `ReasoningParser` and pipeline each decoded chunk through it *before*
    /// the tool-call processor so `<think>...</think>` content is stripped
    /// from user-visible `.chunk(String)` events.
    ///
    /// Accepted values match `ReasoningParser.fromCapabilityName(_:)`:
    /// - `"think_xml"`, `"qwen3"`, `"qwen3_5"`, `"qwen3_6"`, `"deepseek_r1"`,
    ///   `"glm4"`, `"glm5"`, `"nemotron"`, `"minimax"`, `"bailing*"`,
    ///   `"ling*"`, `"hy3*"` / `"hunyuan*"`, `"laguna*"`, `"zaya*"`, and
    ///   suffixed direct capability aliases such as `"glm-5.1-flash"` /
    ///   `"laguna_glm_thinking_v5"` → strip `<think>...</think>`.
    ///   These stamps use `startInReasoning=true` because the Qwen 3.x
    ///   chat templates prefill `<think>\n` at prompt tail when
    ///   `enable_thinking=true` (the template default).
    /// - `"harmony"` / `"harmony_channel"` / `"gemma4_channel"` /
    ///   `"gemma4*"` / `"gemma-4*"` / `"gpt_oss*"` / `"gpt-oss*"` →
    ///   strip Harmony channel envelopes
    ///   (`<|channel>thought\n...<channel|>` for Gemma 4 and
    ///   `<|channel|>analysis/final<|message|>...` for GPT-OSS).
    /// - explicit `"mistral4*"` capability stamps → strip
    ///   `[THINK]...[/THINK]`. The `mistral4` model-type fallback remains
    ///   `"none"` unless a bundle explicitly stamps this parser.
    /// - `"none"`, `"mistral"`, `"gemma"` → no stripping.
    /// - `nil` → no stripping (byte-compatible with upstream default).
    public var reasoningParserName: String?

    /// Optional defaults loaded from `generation_config.json`.
    public var generationDefaults: GenerationConfigFile?

    /// No-load MTP/VL status inferred from bundle metadata and tensor names.
    ///
    /// This is a capability/status surface only. `preserved_enabled` means the
    /// artifact has MTP tensors preserved; it does not enable speculative MTP
    /// decode unless ``MTPBundleStatus/speculativeDecodeEnabled`` is true.
    public var mtpStatus: MTPBundleStatus?

    public init(
        id: String, revision: String = "main",
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        toolCallFormat: ToolCallFormat? = nil,
        reasoningParserName: String? = nil,
        generationDefaults: GenerationConfigFile? = nil,
        mtpStatus: MTPBundleStatus? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.toolCallFormat = toolCallFormat
        self.reasoningParserName = reasoningParserName
        self.generationDefaults = generationDefaults
        self.mtpStatus = mtpStatus
    }

    public init(
        directory: URL,
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        reasoningParserName: String? = nil,
        generationDefaults: GenerationConfigFile? = nil,
        mtpStatus: MTPBundleStatus? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.reasoningParserName = reasoningParserName
        self.generationDefaults = generationDefaults
        self.mtpStatus = mtpStatus
    }

    /// Maps this configuration's behavioral properties into a
    /// ``ResolvedModelConfiguration`` with the given directories.
    ///
    /// This is a pure data mapping with no I/O. The directories should
    /// already be resolved (downloaded or local) before calling this method.
    public func resolved(
        modelDirectory: URL, tokenizerDirectory: URL
    ) -> ResolvedModelConfiguration {
        ResolvedModelConfiguration(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            name: name,
            defaultPrompt: defaultPrompt,
            extraEOSTokens: extraEOSTokens,
            eosTokenIds: eosTokenIds,
            toolCallFormat: toolCallFormat,
            reasoningParserName: reasoningParserName,
            generationDefaults: generationDefaults,
            mtpStatus: mtpStatus)
    }

}

extension ModelConfiguration: Equatable {

}

extension ModelConfiguration.Identifier: Equatable {

    public static func == (lhs: ModelConfiguration.Identifier, rhs: ModelConfiguration.Identifier)
        -> Bool
    {
        switch (lhs, rhs) {
        case (.id(let lhsID, let lhsRevision), .id(let rhsID, let rhsRevision)):
            lhsID == rhsID && lhsRevision == rhsRevision
        case (.directory(let lhsURL), .directory(let rhsURL)):
            lhsURL == rhsURL
        default:
            false
        }
    }
}
