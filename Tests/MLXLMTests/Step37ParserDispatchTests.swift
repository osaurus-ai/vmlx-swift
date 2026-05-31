import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing
import VMLXJinja

private extension Template {
    func renderStep37(_ context: [String: any Sendable]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

@Suite("Step 3.7 parser and capability dispatch")
struct Step37ParserDispatchTests {
    @Test("Step parser aliases resolve to Step parser")
    func stepParserAliasesResolveToStepParser() throws {
        for stamp in ["step", "stepfun", "step3p5", "step3p7", "step3_5", "step3_7"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .step)
        }
        for modelType in ["step3p5", "step3p7", "stepfun"] {
            #expect(ToolCallFormat.infer(from: modelType) == .step)
        }
        #expect(ToolCallFormat.step.createParser() is StepToolCallParser)
    }

    @Test("Step reasoning aliases resolve to Qwen-style think XML")
    func stepReasoningAliasesResolveToThinkXML() throws {
        for stamp in ["step", "stepfun", "step3p5", "step3p7", "qwen3"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
        }
        for modelType in ["step3p5", "step3p7", "Step3p7"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
        }
    }

    @Test("Step XML function parser extracts multiline arguments without leaks")
    func stepXMLFunctionParserExtractsMultilineArguments() throws {
        let processor = ToolCallProcessor(format: .step)
        let stream = """
            <think>Need one tool call.</think>
            <tool_call><function=line_count><parameter=text>red
            green
            blue</parameter></function></tool_call>
            """
        var reasoning = ReasoningParser.fromCapabilityName("qwen3")
        var visible = ""
        var capturedReasoning = ""

        for scalar in stream {
            if var parser = reasoning {
                for segment in parser.feed(String(scalar)) {
                    switch segment {
                    case .reasoning(let text):
                        capturedReasoning += text
                    case .content(let text):
                        if let chunk = processor.processChunk(text) {
                            visible += chunk
                        }
                    }
                }
                reasoning = parser
            }
        }
        if var parser = reasoning {
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let text):
                    capturedReasoning += text
                case .content(let text):
                    if let chunk = processor.processChunk(text) {
                        visible += chunk
                    }
                }
            }
        }
        processor.processEOS()

        #expect(capturedReasoning.contains("Need one tool call."))
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls[0].function.name == "line_count")
        #expect(processor.toolCalls[0].function.arguments["text"] == .string("red\ngreen\nblue"))
        #expect(!visible.contains("<think>"))
        #expect(!visible.contains("</think>"))
        #expect(!visible.contains("<tool_call>"))
        #expect(!visible.contains("<function=line_count>"))
        #expect(!visible.contains("<parameter=text>"))
    }

    @Test("Step JANG capabilities route tool and reasoning parsers")
    func stepJangCapabilitiesRouteParsers() throws {
        let cfg = try JangLoader.parseConfig(from: [
            "format": "jang",
            "format_version": "2.0",
            "model_family": "step3p7",
            "capabilities": [
                "family": "step3p7",
                "reasoning_parser": "qwen3",
                "tool_parser": "step3p5",
                "supports_tools": true,
                "supports_thinking": true,
                "think_in_template": true,
                "modality": "vision",
                "cache_type": "kv",
            ] as [String: Any],
        ])

        #expect(cfg.modelFamily == "step3p7")
        #expect(cfg.capabilities?.supportsTools == true)
        #expect(cfg.capabilities?.supportsThinking == true)
        #expect(cfg.capabilities?.thinkInTemplate == true)
        #expect(cfg.capabilities?.modality == "vision")
        #expect(cfg.capabilities?.cacheType == "kv")
        #expect(ToolCallFormat.fromCapabilityName(cfg.capabilities?.toolParser) == .step)
        #expect(ReasoningParser.fromCapabilityName(cfg.capabilities?.reasoningParser) != nil)
    }

    @Test("Step fallback honors enable_thinking at assistant tail")
    func stepFallbackHonorsEnableThinkingAtAssistantTail() throws {
        let template = try Template(ChatTemplateFallbacks.step37Minimal)
        let base: [String: any Sendable] = [
            "bos_token": "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>",
            "messages": [
                ["role": "user", "content": "Answer directly."],
            ],
            "add_generation_prompt": true,
        ]
        var off = base
        off["enable_thinking"] = false
        let renderedOff = try template.renderStep37(off)
        #expect(renderedOff.hasSuffix("<|im_start|>assistant\n<think>\n</think>\n\n"))

        var on = base
        on["enable_thinking"] = true
        let renderedOn = try template.renderStep37(on)
        #expect(renderedOn.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    @Test("Step fallback renders native XML required tool contract")
    func stepFallbackRendersNativeXMLRequiredToolContract() throws {
        let template = try Template(ChatTemplateFallbacks.step37Minimal)
        let rendered = try template.renderStep37([
            "bos_token": "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>",
            "messages": [
                ["role": "user", "content": "Use line_count on red\ngreen\nblue."],
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count lines.",
                        "parameters": [
                            "type": "object",
                            "properties": [
                                "text": ["type": "string", "description": "Input text"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "required": ["text"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            "tool_choice": "required",
            "tool_choice_name": "line_count",
            "enable_thinking": true,
            "add_generation_prompt": true,
        ])
        #expect(rendered.contains("The active API tool_choice is required"))
        #expect(rendered.contains("Use the `line_count` function."))
        #expect(rendered.contains("<tool_call>\n<function=FUNCTION_NAME>"))
        #expect(rendered.contains("<function=example_function_name>"))
        #expect(rendered.contains("<parameter=ARGUMENT_NAME>"))
        #expect(
            rendered.hasSuffix("<|im_start|>assistant\n<think>\n</think>\n\n"),
            "Explicit required tool_choice must close Step's thinking rail even if thinking is otherwise enabled."
        )
        #expect(!rendered.contains("enable_thinking"))
        #expect(!rendered.contains("tool_choice_name"))
    }

    @Test("Step3p7 wrapper config decodes the Step3p5 text runtime shape")
    func step3p7WrapperConfigDecodesTextRuntimeShape() throws {
        let json = Data(Self.stepWrapperConfigJSON.utf8)
        let cfg = try JSONDecoder.json5().decode(Step3p5Configuration.self, from: json)
        #expect(cfg.modelType == "step3p5")
        #expect(cfg.numHiddenLayers == 45)
        #expect(cfg.layerTypes.prefix(5) == [
            "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
        ])
        #expect(cfg.isMoELayer(3))
        #expect(!cfg.isMoELayer(0))
        #expect(cfg.slidingNumAttentionHeads == 96)
        #expect(cfg.slidingNumAttentionGroups == 8)
    }

    @Test("Step cache topology keeps full attention TQ-compatible and sliding disk-backed")
    func stepCacheTopologyKeepsFullAttentionTQCompatible() throws {
        let cfg = try JSONDecoder.json5().decode(
            Step3p5Configuration.self,
            from: Data(Self.tinyStepWrapperConfigJSON.utf8))

        let defaultCache = Step3p5Model.makeCache(
            layerTypes: cfg.layerTypes,
            slidingWindow: cfg.slidingWindow,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings,
            parameters: nil)
        #expect(defaultCache.count == 4)
        #expect(defaultCache[0] is KVCacheSimple)
        #expect(defaultCache[1] is RotatingKVCache)
        #expect(defaultCache[2] is KVCacheSimple)
        #expect(defaultCache[3] is RotatingKVCache)

        let defaultTopology = ModelCacheTopologySnapshot(cache: defaultCache)
        #expect(defaultTopology.kvLayerCount == 2)
        #expect(defaultTopology.rotatingKVLayerCount == 2)
        #expect(defaultTopology.turboQuantKVLayerCount == 0)
        #expect(defaultTopology.requiresDiskBackedCoordinatorRestore)

        var params = GenerateParameters()
        params.maxKVSize = 2048
        let boundedCache = Step3p5Model.makeCache(
            layerTypes: cfg.layerTypes,
            slidingWindow: cfg.slidingWindow,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings,
            parameters: params)
        #expect(boundedCache.allSatisfy { $0 is RotatingKVCache })
        let boundedTopology = ModelCacheTopologySnapshot(cache: boundedCache)
        #expect(boundedTopology.kvLayerCount == 0)
        #expect(boundedTopology.rotatingKVLayerCount == 4)
        #expect(boundedTopology.requiresDiskBackedCoordinatorRestore)
    }

    @Test("Step TurboQuant KV contract covers only full attention layers")
    func stepTurboQuantKVContractCoversOnlyFullAttentionLayers() throws {
        let batchQuantize = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchQuantize.swift",
            encoding: .utf8)
        #expect(batchQuantize.contains("Preserves `RotatingKVCache`, `DeepseekV4Cache`, `MambaCache`,"))
        #expect(batchQuantize.contains("ordinary `KVCacheSimple` layers compress"))
        #expect(batchQuantize.contains("maybeQuantizeKVCache("))

        let topology = ModelCacheTopologySnapshot(cache: [
            TurboQuantKVCache(),
            RotatingKVCache(maxSize: 512),
            TurboQuantKVCache(),
            RotatingKVCache(maxSize: 512),
        ])
        #expect(topology.kvLayerCount == 0)
        #expect(topology.turboQuantKVLayerCount == 2)
        #expect(topology.rotatingKVLayerCount == 2)
        #expect(topology.requiresDiskBackedCoordinatorRestore)
        #expect(topology.topologyTags.contains("turboQuantKVLayers=2"))
        #expect(topology.topologyTags.contains("rotatingLayers=2"))
        #expect(topology.topologyTags.contains("restore=disk-backed"))
    }

    @Test("Step JANGTQ per-layer quantization inherits top-level group size")
    func stepJANGTQPerLayerQuantizationInheritsTopLevelGroupSize() throws {
        let json = """
            {
              "model_type": "step3p7",
              "quantization": {
                "bits": 2,
                "group_size": 128,
                "model.layers.22.mlp.switch_mlp.up_proj": {"bits": 2},
                "model.layers.22.mlp.switch_mlp.down_proj": {"bits": 4, "group_size": 128}
              }
            }
            """
        let cfg = try JSONDecoder.json5().decode(
            BaseConfiguration.self,
            from: Data(json.utf8))
        let perLayer = try #require(cfg.quantizationContainer?.perLayerQuantization)
        let up = try #require(
            perLayer.quantization(layer: "model.layers.22.mlp.switch_mlp.up_proj"))
        let down = try #require(
            perLayer.quantization(layer: "model.layers.22.mlp.switch_mlp.down_proj"))
        #expect(up.groupSize == 128)
        #expect(up.bits == 2)
        #expect(down.groupSize == 128)
        #expect(down.bits == 4)
    }

    @Test("Step sanitizer drops NVFP4 attention scale side tensors")
    func stepSanitizerDropsNVFP4AttentionScaleSideTensors() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/Step3p5.swift",
            encoding: .utf8)
        #expect(source.contains("self_attn.k_proj.k_scale"))
        #expect(source.contains("self_attn.v_proj.v_scale"))
        #expect(source.contains("hasSuffix(\".tq_bits\")"))
    }

    private static let stepWrapperConfigJSON = """
        {
          "model_type": "step3p7",
          "weight_format": "mxtq",
          "mxtq_bits": 2,
          "mxtq_gate_up_bits": 2,
          "mxtq_down_bits": 4,
          "mxtq_seed": 42,
          "text_config": {
            "model_type": "step3p5",
            "hidden_size": 4096,
            "num_hidden_layers": 45,
            "vocab_size": 128896,
            "num_attention_heads": 64,
            "num_attention_groups": 8,
            "head_dim": 128,
            "intermediate_size": 12288,
            "moe_num_experts": 288,
            "moe_top_k": 8,
            "moe_intermediate_size": 1280,
            "share_expert_dim": 1280,
            "moe_layers_enum": "3,4,5",
            "layer_types": [
              "full_attention", "sliding_attention", "sliding_attention",
              "sliding_attention", "full_attention"
            ],
            "rope_theta": [5000000, 10000, 10000, 10000, 5000000],
            "partial_rotary_factors": [0.5, 1.0, 1.0, 1.0, 0.5],
            "attention_other_setting": {
              "num_attention_heads": 96,
              "num_attention_groups": 8
            }
          }
        }
        """

    private static let tinyStepWrapperConfigJSON = """
        {
          "model_type": "step3p7",
          "text_config": {
            "model_type": "step3p5",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "vocab_size": 128,
            "num_attention_heads": 2,
            "num_attention_groups": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 16,
            "share_expert_dim": 16,
            "moe_layers_enum": "1,2,3",
            "layer_types": [
              "full_attention", "sliding_attention",
              "full_attention", "sliding_attention"
            ],
            "rope_theta": [5000000, 10000, 5000000, 10000],
            "partial_rotary_factors": [0.5, 1.0, 0.5, 1.0],
            "sliding_window": 64,
            "attention_other_setting": {
              "num_attention_heads": 2,
              "num_attention_groups": 1
            }
          }
        }
        """
}
