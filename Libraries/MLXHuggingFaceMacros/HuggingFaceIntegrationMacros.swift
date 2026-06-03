import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

@main
struct Macros: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        DownloaderMacro.self,
        TokenizerAdaptorMacro.self,
        TokenizerLoaderMacro.self,
        LoadContainerMacro.self,
        LoadContextMacro.self,
    ]
}

public struct DownloaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let argument = node.arguments.first?.expression.description ?? "HubClient()"

        return
            """
            // make sure you:
            //
            // import VMLXHuggingFace
            //
            { (hubApi: HubClient) -> MLXLMCommon.Downloader in
                struct HubBridge: MLXLMCommon.Downloader {
                    private let upstream: HubClient

                    init(_ upstream: HubClient) {
                        self.upstream = upstream
                    }

                    public func download(
                        id: String,
                        revision: String?,
                        matching patterns: [String],
                        useLatest: Bool,
                        progressHandler: @Sendable @escaping (Progress) -> Void
                    ) async throws -> URL {
                        guard let repoID = VMLXHuggingFace.Repo.ID(rawValue: id) else {
                            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
                        }
                        let revision = revision ?? "main"

                        return try await upstream.downloadSnapshot(
                            of: repoID,
                            revision: revision,
                            matching: patterns,
                            progressHandler: { @MainActor progress in
                                progressHandler(progress)
                            }
                        )
                    }
                }

                return HubBridge(hubApi)
            }(\(raw: argument))
            """
    }
}

public struct TokenizerAdaptorMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let argument = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#adaptHuggingFaceTokenizer requires an argument")
        }

        return
            """
            // make sure you:
            //
            // import VMLXTokenizers
            //
            { (huggingFaceTokenizer: VMLXTokenizers.Tokenizer) -> MLXLMCommon.Tokenizer in
                struct TokenizerBridge: MLXLMCommon.GenerationPromptControllableTokenizer {
                    private let upstream: any VMLXTokenizers.Tokenizer

                    init(_ upstream: any VMLXTokenizers.Tokenizer) {
                        self.upstream = upstream
                    }

                    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
                    }

                    // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
                    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
                    }

                    func convertTokenToId(_ token: String) -> Int? {
                        upstream.convertTokenToId(token)
                    }

                    func convertIdToToken(_ id: Int) -> String? {
                        upstream.convertIdToToken(id)
                    }

                    var bosToken: String? { upstream.bosToken }
                    var eosToken: String? { upstream.eosToken }
                    var unknownToken: String? { upstream.unknownToken }

                    func applyChatTemplate(
                        messages: [[String: any Sendable]],
                        tools: [[String: any Sendable]]?,
                        additionalContext: [String: any Sendable]?
                    ) throws -> [Int] {
                        let chatTemplateTools = MLXLMCommon.normalizedToolsForChatTemplate(tools)
                        // Iter 50 escape hatch: `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja`
                        // bypasses the tokenizer's shipped chat template. Motivation: Gemma-4's
                        // native template trips a swift-jinja 1.3.0 interaction bug — all
                        // constructs parse individually (see Gemma4ChatTemplateProbeTests)
                        // but the full assembly fails. The override lets callers ship
                        // `Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja` (or any
                        // other compatible template) for models blocked by upstream gaps.
                        // Default behaviour (no env var) is unchanged.
                        let env = ProcessInfo.processInfo.environment
                        if let path = env["VMLX_CHAT_TEMPLATE_OVERRIDE"], !path.isEmpty,
                           let src = try? String(contentsOfFile: path, encoding: .utf8) {
                            do {
                                return try upstream.applyChatTemplate(
                                    messages: messages,
                                    chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(src),
                                    addGenerationPrompt: true,
                                    truncation: false,
                                    maxLength: nil,
                                    tools: chatTemplateTools,
                                    additionalContext: additionalContext)
                            } catch VMLXTokenizers.TokenizerError.missingChatTemplate {
                                throw MLXLMCommon.TokenizerError.missingChatTemplate
                            }
                        }
                        let lagunaEos =
                            String(UnicodeScalar(0x3008)!)
                            + "|EOS|"
                            + String(UnicodeScalar(0x3009)!)
                        let hasLagunaSentinel =
                            upstream.bosToken == lagunaEos
                            && upstream.eosToken == lagunaEos
                            && upstream.convertTokenToId("<assistant>") != nil
                            && upstream.convertTokenToId("</assistant>") != nil
                            && upstream.convertTokenToId("<think>") != nil
                            && upstream.convertTokenToId("</think>") != nil
                        if hasLagunaSentinel
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template auto-correction engaged: LagunaMinimal\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        let hasLFM2NativeToolSentinels =
                            upstream.bosToken == "<|startoftext|>"
                            && upstream.eosToken == "<|im_end|>"
                            && upstream.convertTokenToId("<|tool_call_start|>") != nil
                            && upstream.convertTokenToId("<|tool_call_end|>") != nil
                        if !(tools?.isEmpty ?? true),
                           hasLFM2NativeToolSentinels,
                           (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template tools -> LFM2ToolMinimal fallback engaged\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.lfm2ToolMinimal),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                    additionalContext: additionalContext)
                        }
                        let hasGemma4NativeToolSentinels =
                            upstream.convertTokenToId("<|tool_call>") != nil
                            && upstream.convertTokenToId("<|turn>") != nil
                        let gemmaRequiredToolChoice =
                            !(chatTemplateTools?.isEmpty ?? true)
                            && (upstream.bosToken == "<bos>" || hasGemma4NativeToolSentinels)
                            && (additionalContext?["tool_choice"] as? String) == "required"
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1"
                        if gemmaRequiredToolChoice {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template required tools -> Gemma4WithTools fallback engaged\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        let step37Bos =
                            "<" + String(UnicodeScalar(0xFF5C)!)
                            + "begin" + String(UnicodeScalar(0x2581)!) + "of"
                            + String(UnicodeScalar(0x2581)!) + "sentence"
                            + String(UnicodeScalar(0xFF5C)!) + ">"
                        let hasStep37Sentinels =
                            upstream.bosToken == step37Bos
                            && upstream.eosToken == "<|im_end|>"
                            && upstream.convertTokenToId("<|im_start|>") != nil
                            && upstream.convertTokenToId("<tool_call>") != nil
                            && upstream.convertTokenToId("<im_patch>") != nil
                        let step37NeedsFallback =
                            hasStep37Sentinels
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1"
                            && (
                                !(tools?.isEmpty ?? true)
                                || (additionalContext?["enable_thinking"] as? Bool) == false
                            )
                        if step37NeedsFallback {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template auto-correction engaged: Step37Minimal\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.step37Minimal),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        // MiniMax-M2 native template auto-correction: every shipping
                        // MiniMax-M2 / M2.7 chat_template.jinja unconditionally prefills
                        // <think> at the assistant tail and ignores enable_thinking.
                        // Direct-answer (thinking-off) callers therefore see all output
                        // trapped in Generation.reasoning. When (a) additionalContext sets
                        // enable_thinking=false, (b) the tokenizer carries the MiniMax-
                        // specific ]~!b[ / [e~[ BOS/EOS pair, and (c) no env override
                        // is set, force the corrected MiniMaxM2Minimal fallback first.
                        // Do not use convertTokenToId here: some tokenizers return an
                        // unknown-token id for arbitrary strings, which can misroute
                        // Gemma/Laguna/Nemotron into the MiniMax fallback.
                        // Auto-engage is one-way: thinking-on requests fall through to
                        // the native template untouched.
                        if let ctx = additionalContext,
                           let enableThinking = ctx["enable_thinking"] as? Bool,
                           enableThinking == false,
                           upstream.bosToken == "]~!b[",
                           upstream.eosToken == "[e~[" {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template auto-correction engaged: MiniMaxM2Minimal (enable_thinking=false)\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.minimaxM2Minimal),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        // Mistral 4 effort coercion (mirrors Python
                        // `vmlx serve` server.py:3216-3225). The Mistral 4
                        // chat template renders a [MODEL_SETTINGS]
                        // {"reasoning_effort":"high"|"none"|"max"} block
                        // gated on this field. Python auto-maps
                        // enable_thinking → reasoning_effort so callers
                        // don't have to pass both. Mirror that mapping here
                        // when (a) tokenizer carries the [MODEL_SETTINGS]
                        // sentinel that all Mistral 4 variants ship, and
                        // (b) the caller did NOT already provide an
                        // explicit reasoning_effort. Without this Swift
                        // emits "none" even when enable_thinking=True,
                        // breaking inference-cost + token-count parity vs
                        // Python (~/vmlx/docs/AUDIT-RELEASE-READINESS.md:
                        // "Mistral 4 effort not normalized").
                        var mistral4AdjustedContext = additionalContext
                        if mistral4AdjustedContext?["reasoning_effort"] == nil,
                           upstream.convertTokenToId("[MODEL_SETTINGS]") != nil,
                           let enableThinking = mistral4AdjustedContext?["enable_thinking"] as? Bool {
                            var ctx = mistral4AdjustedContext ?? [:]
                            ctx["reasoning_effort"] = enableThinking ? "high" : "none"
                            mistral4AdjustedContext = ctx
                        }
                        let dsv4Bos =
                            "<" + String(UnicodeScalar(0xFF5C)!)
                            + "begin" + String(UnicodeScalar(0x2581)!) + "of"
                            + String(UnicodeScalar(0x2581)!) + "sentence"
                            + String(UnicodeScalar(0xFF5C)!) + ">"
                        var adjustedContext = mistral4AdjustedContext
                        if upstream.bosToken == dsv4Bos,
                           let enableThinking = adjustedContext?["enable_thinking"] as? Bool,
                           enableThinking == false,
                           adjustedContext?["reasoning_effort"] != nil {
                            adjustedContext?.removeValue(forKey: "reasoning_effort")
                        }
                        if !(tools?.isEmpty ?? true),
                           upstream.bosToken == "<s>",
                           upstream.convertTokenToId("<|im_end|>") != nil,
                           (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template tools -> NemotronMinimal fallback engaged\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                addGenerationPrompt: true,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: adjustedContext)
                        }
                        do {
                            return try upstream.applyChatTemplate(
                                messages: messages, tools: chatTemplateTools, additionalContext: adjustedContext)
                        } catch VMLXTokenizers.TokenizerError.missingChatTemplate {
                            // Missing-template fallbacks for bundles that ship
                            // tokenizer special tokens but no tokenizer_config
                            // chat_template field. VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1
                            // opts out.
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                                if hasLagunaSentinel {
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template missing -> LagunaMinimal fallback engaged\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: adjustedContext)
                                }
                                if upstream.bosToken == dsv4Bos {
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template missing -> DSV4Minimal fallback engaged\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.dsv4Minimal),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: adjustedContext)
                                }
                                if upstream.bosToken == "]~!b[",
                                   upstream.eosToken == "[e~[" {
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template missing -> MiniMaxM2Minimal fallback engaged\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.minimaxM2Minimal),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                }
                                if upstream.bosToken == "<bos>" {
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template missing -> Gemma4 fallback engaged\\n"
                                                .data(using: .utf8)!)
                                    }
                                    let template = (tools?.isEmpty ?? true)
                                        ? MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal
                                        : MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(template),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                }
                                if upstream.bosToken == "<s>",
                                   upstream.convertTokenToId("<|im_end|>") != nil {
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template missing -> NemotronMinimal fallback engaged\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                }
                            }
                            throw MLXLMCommon.TokenizerError.missingChatTemplate
                        } catch {
                            // Upstream threw on a template the swift-jinja runtime
                            // can't evaluate (Gemma-4 `multiplicativeBinaryOperator`
                            // parse, Nemotron `not in` on ArrayValue tuples, …).
                            // Try built-in fallbacks, picking the family that
                            // matches the tokenizer's special-token vocabulary so
                            // the emitted prompt shape stays model-native.
                            // `VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1` opts out.
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") == "1" {
                                throw error
                            }
                            // Family sniff. Gemma-4 is the only widely-used
                            // family whose bos_token is literally "<bos>";
                            // ChatML-family models (Nemotron-Cascade-2 + all
                            // Mistral/Qwen 3.x descendants) use "<s>" or no
                            // bos. That single check lets us pick the right
                            // fallback ordering without needing the model
                            // config parsed separately. `convertTokenToId`
                            // protects us from applying a fallback whose
                            // sentinel tokens are not in vocab.
                            let isGemma = upstream.bosToken == "<bos>"
                            let hasNemotronSentinel =
                                upstream.convertTokenToId("<|im_start|>") != nil
                                || upstream.convertTokenToId("<|im_end|>") != nil
                            let ordered: [(label: String, template: String)]
                            if hasLagunaSentinel {
                                ordered = [
                                    ("LagunaMinimal", MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                ]
                            } else if isGemma {
                                ordered = [
                                    ("Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                    ("Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal),
                                ]
                            } else if hasNemotronSentinel {
                                ordered = [
                                    ("NemotronMinimal", MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                    ("Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                    ("Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal),
                                ]
                            } else {
                                ordered = MLXLMCommon.ChatTemplateFallbacks.orderedFallbacks
                            }
                            for candidate in ordered {
                                do {
                                    let ids = try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(candidate.template),
                                        addGenerationPrompt: true,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template fallback engaged: \\(candidate.label)\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return ids
                                } catch {
                                    continue
                                }
                            }
                            throw error
                        }
                    }

                    func applyChatTemplate(
                        messages: [[String: any Sendable]],
                        tools: [[String: any Sendable]]?,
                        additionalContext: [String: any Sendable]?,
                        addGenerationPrompt: Bool
                    ) throws -> [Int] {
                        let chatTemplateTools = MLXLMCommon.normalizedToolsForChatTemplate(tools)
                        let env = ProcessInfo.processInfo.environment
                        if let path = env["VMLX_CHAT_TEMPLATE_OVERRIDE"], !path.isEmpty,
                           let src = try? String(contentsOfFile: path, encoding: .utf8) {
                            do {
                                return try upstream.applyChatTemplate(
                                    messages: messages,
                                    chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(src),
                                    addGenerationPrompt: addGenerationPrompt,
                                    truncation: false,
                                    maxLength: nil,
                                    tools: chatTemplateTools,
                                    additionalContext: additionalContext)
                            } catch VMLXTokenizers.TokenizerError.missingChatTemplate {
                                throw MLXLMCommon.TokenizerError.missingChatTemplate
                            }
                        }
                        let lagunaEos =
                            String(UnicodeScalar(0x3008)!)
                            + "|EOS|"
                            + String(UnicodeScalar(0x3009)!)
                        let hasLagunaSentinel =
                            upstream.bosToken == lagunaEos
                            && upstream.eosToken == lagunaEos
                            && upstream.convertTokenToId("<assistant>") != nil
                            && upstream.convertTokenToId("</assistant>") != nil
                            && upstream.convertTokenToId("<think>") != nil
                            && upstream.convertTokenToId("</think>") != nil
                        if hasLagunaSentinel
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        let hasLFM2NativeToolSentinels =
                            upstream.bosToken == "<|startoftext|>"
                            && upstream.eosToken == "<|im_end|>"
                            && upstream.convertTokenToId("<|tool_call_start|>") != nil
                            && upstream.convertTokenToId("<|tool_call_end|>") != nil
                        if !(tools?.isEmpty ?? true),
                           hasLFM2NativeToolSentinels,
                           (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template tools -> LFM2ToolMinimal fallback engaged\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.lfm2ToolMinimal),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                    additionalContext: additionalContext)
                        }
                        let hasGemma4NativeToolSentinels =
                            upstream.convertTokenToId("<|tool_call>") != nil
                            && upstream.convertTokenToId("<|turn>") != nil
                        let gemmaRequiredToolChoice =
                            !(chatTemplateTools?.isEmpty ?? true)
                            && (upstream.bosToken == "<bos>" || hasGemma4NativeToolSentinels)
                            && (additionalContext?["tool_choice"] as? String) == "required"
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1"
                        if gemmaRequiredToolChoice {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template required tools -> Gemma4WithTools fallback engaged\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        let step37Bos =
                            "<" + String(UnicodeScalar(0xFF5C)!)
                            + "begin" + String(UnicodeScalar(0x2581)!) + "of"
                            + String(UnicodeScalar(0x2581)!) + "sentence"
                            + String(UnicodeScalar(0xFF5C)!) + ">"
                        let hasStep37Sentinels =
                            upstream.bosToken == step37Bos
                            && upstream.eosToken == "<|im_end|>"
                            && upstream.convertTokenToId("<|im_start|>") != nil
                            && upstream.convertTokenToId("<tool_call>") != nil
                            && upstream.convertTokenToId("<im_patch>") != nil
                        let step37NeedsFallback =
                            hasStep37Sentinels
                            && (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1"
                            && (
                                !(tools?.isEmpty ?? true)
                                || (additionalContext?["enable_thinking"] as? Bool) == false
                            )
                        if step37NeedsFallback {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                FileHandle.standardError.write(
                                    "[vmlx] chat-template auto-correction engaged: Step37Minimal\\n"
                                        .data(using: .utf8)!)
                            }
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.step37Minimal),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        if let ctx = additionalContext,
                           let enableThinking = ctx["enable_thinking"] as? Bool,
                           enableThinking == false,
                           upstream.bosToken == "]~!b[",
                           upstream.eosToken == "[e~[" {
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.minimaxM2Minimal),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: additionalContext)
                        }
                        var mistral4AdjustedContext = additionalContext
                        if mistral4AdjustedContext?["reasoning_effort"] == nil,
                           upstream.convertTokenToId("[MODEL_SETTINGS]") != nil,
                           let enableThinking = mistral4AdjustedContext?["enable_thinking"] as? Bool {
                            var ctx = mistral4AdjustedContext ?? [:]
                            ctx["reasoning_effort"] = enableThinking ? "high" : "none"
                            mistral4AdjustedContext = ctx
                        }
                        let dsv4Bos =
                            "<" + String(UnicodeScalar(0xFF5C)!)
                            + "begin" + String(UnicodeScalar(0x2581)!) + "of"
                            + String(UnicodeScalar(0x2581)!) + "sentence"
                            + String(UnicodeScalar(0xFF5C)!) + ">"
                        var adjustedContext = mistral4AdjustedContext
                        if upstream.bosToken == dsv4Bos,
                           let enableThinking = adjustedContext?["enable_thinking"] as? Bool,
                           enableThinking == false,
                           adjustedContext?["reasoning_effort"] != nil {
                            adjustedContext?.removeValue(forKey: "reasoning_effort")
                        }
                        if !(tools?.isEmpty ?? true),
                           upstream.bosToken == "<s>",
                           upstream.convertTokenToId("<|im_end|>") != nil,
                           (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                    MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: adjustedContext)
                        }
                        do {
                            return try upstream.applyChatTemplate(
                                messages: messages,
                                chatTemplate: nil,
                                addGenerationPrompt: addGenerationPrompt,
                                truncation: false,
                                maxLength: nil,
                                tools: chatTemplateTools,
                                additionalContext: adjustedContext)
                        } catch VMLXTokenizers.TokenizerError.missingChatTemplate {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") != "1" {
                                if hasLagunaSentinel {
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                        addGenerationPrompt: addGenerationPrompt,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: adjustedContext)
                                }
                                if upstream.bosToken == dsv4Bos {
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.dsv4Minimal),
                                        addGenerationPrompt: addGenerationPrompt,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: adjustedContext)
                                }
                                if upstream.bosToken == "<bos>" {
                                    let template = (tools?.isEmpty ?? true)
                                        ? MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal
                                        : MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(template),
                                        addGenerationPrompt: addGenerationPrompt,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                }
                                if upstream.bosToken == "<s>",
                                   upstream.convertTokenToId("<|im_end|>") != nil {
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                            MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                        addGenerationPrompt: addGenerationPrompt,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                }
                            }
                            if upstream.bosToken == "]~!b[",
                               upstream.eosToken == "[e~[" {
                                return try upstream.applyChatTemplate(
                                    messages: messages,
                                    chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(
                                        MLXLMCommon.ChatTemplateFallbacks.minimaxM2Minimal),
                                    addGenerationPrompt: addGenerationPrompt,
                                    truncation: false,
                                    maxLength: nil,
                                    tools: chatTemplateTools,
                                    additionalContext: additionalContext)
                            }
                            throw MLXLMCommon.TokenizerError.missingChatTemplate
                        } catch {
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") == "1" {
                                throw error
                            }
                            let isGemma = upstream.bosToken == "<bos>"
                            let hasNemotronSentinel =
                                upstream.convertTokenToId("<|im_start|>") != nil
                                || upstream.convertTokenToId("<|im_end|>") != nil
                            let ordered: [(label: String, template: String)]
                            if hasLagunaSentinel {
                                ordered = [
                                    ("LagunaMinimal", MLXLMCommon.ChatTemplateFallbacks.lagunaMinimal),
                                ]
                            } else if isGemma {
                                ordered = [
                                    ("Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                    ("Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal),
                                ]
                            } else if hasNemotronSentinel {
                                ordered = [
                                    ("NemotronMinimal", MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                    ("Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                    ("Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal),
                                ]
                            } else {
                                ordered = MLXLMCommon.ChatTemplateFallbacks.orderedFallbacks
                            }
                            for candidate in ordered {
                                do {
                                    return try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: VMLXTokenizers.ChatTemplateArgument.literal(candidate.template),
                                        addGenerationPrompt: addGenerationPrompt,
                                        truncation: false,
                                        maxLength: nil,
                                        tools: chatTemplateTools,
                                        additionalContext: additionalContext)
                                } catch {
                                    continue
                                }
                            }
                            throw error
                        }
                    }
                }

                return TokenizerBridge(huggingFaceTokenizer)
            }(\(raw: argument))
            """
    }
}

public struct TokenizerLoaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        return
            """
            { () -> MLXLMCommon.TokenizerLoader in
                struct TransformersLoader: MLXLMCommon.TokenizerLoader {
                    public init() {}

                    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
                        // make sure you:
                        //
                        // import VMLXTokenizers
                        //
                        let upstream = try await VMLXTokenizers.AutoTokenizer.from(modelFolder: directory)
                        return #adaptHuggingFaceTokenizer(upstream)
                    }
                }

                return TransformersLoader()
            }()
            """
    }
}

public struct LoadContainerMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message(
                "#huggingFaceLoadModelContainer requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

public struct LoadContextMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#huggingFaceLoadModel requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModel(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

enum MacroExpansionError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}
