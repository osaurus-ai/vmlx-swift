// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import Testing

@Suite("MLXPress low-RAM policy source coverage")
struct MLXPressLowRamPolicySourceTests {
    @Test("MLXPress defaults to compression-first canonical mmap residency")
    func mlxpressDefaultsToCompressionFirstResidency() throws {
        let mlxpress = try String(
            contentsOfFile: "Sources/MLXPress/MLXPress.swift",
            encoding: .utf8)
        let prestacker = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JangPressPrestacker.swift",
            encoding: .utf8)
        let factory = try String(
            contentsOfFile: "Libraries/MLXLMCommon/ModelFactory.swift",
            encoding: .utf8)
        let streamingExperts = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQStreamingExperts.swift",
            encoding: .utf8)
        let coldSweep = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/MLXPressMmapColdSweep.swift",
            encoding: .utf8)
        let machCache = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JangPressMachCache.swift",
            encoding: .utf8)
        let aliases = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/MLXPressAliases.swift",
            encoding: .utf8)
        let load = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Load.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let loadTimeStacking = try String(
            contentsOfFile: "Libraries/MLXLMCommon/LoadTimeStacking.swift",
            encoding: .utf8)
        let kernels = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQKernels.swift",
            encoding: .utf8)
        let deepseekJANGTQ = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/DeepseekV3JANGTQ.swift",
            encoding: .utf8)
        let minimaxJANGTQ = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift",
            encoding: .utf8)
        let deepseekV3 = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/DeepseekV3.swift",
            encoding: .utf8)
        let attentionUtils = try String(
            contentsOfFile: "Libraries/MLXLMCommon/AttentionUtils.swift",
            encoding: .utf8)
        let hy3 = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/Hy3.swift",
            encoding: .utf8)
        let safetensors = try String(
            contentsOfFile: "Source/Cmlx/mlx/mlx/io/safetensors.cpp",
            encoding: .utf8)
        let cli = try String(
            contentsOfFile: "Sources/MLXPressCLI/main.swift",
            encoding: .utf8)
        let metricsSummary = try String(
            contentsOfFile: "scripts/summarize-mlxpress-metrics.py",
            encoding: .utf8)

        #expect(mlxpress.contains("enableActiveExpertStreaming: Bool = false"))
        #expect(mlxpress.contains("enum MLXPressPrestackPolicy"))
        #expect(mlxpress.contains("ephemeralTemporaryOverlay"))
        #expect(mlxpress.contains("compression-first"))
        #expect(mlxpress.contains("mmap-backed safetensors"))
        #expect(mlxpress.contains("size-aware `pread` policy"))
        #expect(mlxpress.contains("JANGTQStreamingExperts.hasStreamableExperts"))
        #expect(mlxpress.contains("shouldUseEphemeralPrestack"))
        #expect(mlxpress.contains("MLXPRESS_PRESTACK_EPHEMERAL"))
        #expect(mlxpress.contains("MLXPRESS_PRESTACK_STRICT"))
        #expect(mlxpress.contains("temporary routed overlay will be removed"))
        #expect(mlxpress.contains(#"setenv("MLXPRESS_PRESTACK", "0", 1)"#))
        #expect(mlxpress.contains(#"setenv("JANGPRESS_PRESTACK", "0", 1)"#))
        #expect(
            mlxpress.contains(
                #"if getenv("MLXPRESS_STREAMING_EVAL_EACH_LAYER") == nil"#))
        #expect(mlxpress.contains("allocatorCacheLimit: configuration.allocatorCacheLimit"))
        #expect(mlxpress.contains("decodeCacheLimitOverride"))
        #expect(mlxpress.contains("MLXPRESS_DECODE_CACHE_LIMIT_MB"))
        #expect(prestacker.contains("prestackExplicitlyEnabled = isEnabledFlag(prestackRaw)"))
        #expect(prestacker.contains("if hasJANGTQSidecar && prestackExplicitlyEnabled"))
        #expect(prestacker.contains("MLXPRESS_PRESTACK_EPHEMERAL"))
        #expect(prestacker.contains("cleanupEphemeralPrestackDirectories"))
        #expect(prestacker.contains("cleanupEphemeralPrestackDirectory(_ url: URL)"))
        #expect(prestacker.contains("atexit(jangPressPrestackerCleanupAtExit)"))
        #expect(factory.contains("cleanupEphemeralPrestackDirectory(loadDirectory)"))
        #expect(factory.contains("context.configuration.id = .directory(directory)"))
        #expect(factory.contains("MLXPRESS_PRESTACK=1 / JANGPRESS_PRESTACK=1"))
        #expect(factory.contains("withMmapSafetensorsEnv"))
        #expect(factory.contains("adviseCanonicalMmapRoutedExpertsIfAvailable"))
        #expect(factory.contains("MLX_SAFETENSORS_MMAP"))
        #expect(factory.contains("MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS"))
        #expect(factory.contains("MLXPRESS_MMAP_TENSOR_BUFFERS"))
        #expect(machCache.contains("VM_FLAGS_PURGABLE"))
        #expect(machCache.contains("releaseColdTiles"))
        #expect(machCache.contains("VM_PURGABLE_VOLATILE"))
        #expect(machCache.contains("component: String"))
        #expect(machCache.contains("registerFilled"))
        #expect(machCache.contains("public func remove(layer: Int"))
        #expect(machCache.contains("MLXPRESS_MACH_RELEASE_POLICY"))
        #expect(machCache.contains("JANGPRESS_MACH_RELEASE_POLICY"))
        #expect(machCache.contains("MADV_PAGEOUT"))
        #expect(machCache.contains("VM_VOLATILE_GROUP_7"))
        #expect(machCache.contains("VM_PURGABLE_BEHAVIOR_LIFO"))
        #expect(machCache.contains("VM_PURGABLE_NO_AGING"))
        #expect(machCache.contains("let readCount = min(tile.byteCount - readBytes, 64 * 1024 * 1024)"))
        #expect(machCache.contains("pread("))
        #expect(machCache.contains("mlx_array_new_data_managed_payload"))
        #expect(aliases.contains("MLXPressMachCache"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS"))
        #expect(streamingExperts.contains("tensor.mach_register"))
        #expect(streamingExperts.contains("releaseMachColdTiles"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_OFFSET_SPANS"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_MACH_OFFSET_SPANS"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_FULL_OFFSET_SPANS"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_PREWARM_FULL_OFFSET_SPANS"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_FULL_OFFSET_LAZY"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_PREWARM_MAX_LAYERS"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MACH_PREWARM_MAX_GB"))
        #expect(streamingExperts.contains("machOffsetSpansEnabled"))
        #expect(streamingExperts.contains("machFullOffsetSpansEnabled"))
        #expect(streamingExperts.contains("machPrewarmFullOffsetSpansEnabled"))
        #expect(streamingExperts.contains("machFullOffsetLazyEnabled"))
        #expect(streamingExperts.contains("machOffsetSpanBudgetBytes"))
        #expect(streamingExperts.contains("machOffsetSpanRegisteredBytes"))
        #expect(streamingExperts.contains("StreamingMachTensorEntry"))
        #expect(streamingExperts.contains("loadMachOffsetSpan"))
        #expect(streamingExperts.contains("makeMachFullOffsetSpanArray"))
        #expect(streamingExperts.contains("prewarmFullOffsetSpansIfNeeded"))
        #expect(streamingExperts.contains("tensor.mach_full_offset_read"))
        #expect(streamingExperts.contains("tensor.mach_full_offset_hit"))
        #expect(streamingExperts.contains("tensor.mach_full_offset_prewarm"))
        #expect(streamingExperts.contains("registerMachBackedTensor"))
        #expect(streamingExperts.contains("useNoCacheForRegistration"))
        #expect(
            streamingExperts.contains(
                "cachedDarwinFileDescriptor(for: ref.fileURL, useNoCache: useNoCache)"))
        #expect(streamingExperts.contains("cachedDarwinFileDescriptor(for url: URL, useNoCache: Bool)"))
        #expect(streamingExperts.contains(#"let cacheKey = "\(useNoCache ? "nocache" : "cache"):\(key)""#))
        #expect(streamingExperts.contains("let readCount = min(count - total, 64 * 1024 * 1024)"))
        #expect(streamingExperts.contains("machCache.registerFilled"))
        #expect(streamingExperts.contains("tensor.mach_offset_read"))
        #expect(streamingExperts.contains("tensor.mach_offset_array"))
        #expect(streamingExperts.contains("tensor.mach_offset_evict"))
        #expect(streamingExperts.contains("tensor.mach_offset_budget_skip"))
        #expect(streamingExperts.contains("machTensorEvictions"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_OFFSET_SPAN_CACHE_MB"))
        #expect(streamingExperts.contains("tensor.offset_span_cache_hit"))
        #expect(streamingExperts.contains("tensor.offset_span_cache_store"))
        #expect(streamingExperts.contains("cacheKey = StreamingOffsetSpanCacheKey"))
        #expect(!streamingExperts.contains("cacheEligible = spanOffset == descriptor.spanOffset"))
        #expect(
            cli.contains(
                "--active-expert-streaming defaults to off"))
        #expect(cli.contains("--ephemeral-prestack"))
        #expect(cli.contains("MiniMax-class resident-compute diagnostic"))
        #expect(cli.contains("prestack: prestack"))
        #expect(cli.contains("--compiled-decode"))
        #expect(cli.contains("--allow-minimax-compiled-decode"))
        #expect(cli.contains("allowMiniMaxCompiledDecode"))
        #expect(cli.contains("MLXPRESS_COMPILED_DECODE_ALLOW_MINIMAX"))
        #expect(evaluate.contains("compiledDecodeAllowsMiniMax"))
        #expect(evaluate.contains("MLXPRESS_COMPILED_DECODE_ALLOW_MINIMAX"))
        #expect(evaluate.contains("CompilableKVCache(from: layer, maxLength: maxCacheLength)"))
        #expect(evaluate.contains("CompilableTurboQuantKVCache(from: layer as! TurboQuantKVCache)"))
        #expect(evaluate.contains("self.cache = promoted"))
        #expect(cli.contains(#"context["enable_thinking"] = enableThinking"#))
        #expect(cli.contains(#"context["thinking"] = enableThinking"#))
        #expect(cli.contains("--metrics-jsonl"))
        #expect(cli.contains("MetricsJSONLWriter"))
        #expect(!cli.contains("gateAwareMemoryLimit"))
        #expect(cli.contains("memoryLimit: .default"))
        #expect(cli.contains(#""tokens_per_second""#))
        #expect(cli.contains(#""activity_gate""#))
        #expect(cli.contains(#""system_pressure""#))
        #expect(cli.contains(#""mlx_active_memory_bytes""#))
        #expect(cli.contains(#""mlx_peak_memory_bytes""#))
        #expect(cli.contains(#""mlx_active_delta_bytes""#))
        #expect(cli.contains(#""swapouts_delta_bytes""#))
        #expect(cli.contains(#""file_read_pressure""#))
        #expect(cli.contains(#""effective_read_pressure""#))
        #expect(cli.contains(#""read_mb_per_generated_token""#))
        #expect(cli.contains(#""effective_read_mb_per_generated_token""#))
        #expect(cli.contains(#""tensor.mach_offset_read""#))
        #expect(cli.contains(#""active_expert_trace""#))
        #expect(cli.contains(#""consecutive_reuse_rate""#))
        #expect(cli.contains(#""active_slice_residency""#))
        #expect(cli.contains(#""byte_hit_rate""#))
        #expect(cli.contains(#""tensor_resident_bytes""#))
        #expect(cli.contains(#""slice_resident_bytes""#))
        #expect(cli.contains(#""bank_resident_bytes""#))
        #expect(cli.contains("--file-read-gate-mb-per-token"))
        #expect(cli.contains("--active-expert-trace"))
        #expect(cli.contains("writeFileReadPressure"))
        #expect(cli.contains("writeEffectiveReadPressure"))
        #expect(cli.contains("writeActiveExpertTrace"))
        #expect(cli.contains("writeActiveSliceResidency"))
        #expect(cli.contains(#""streaming_profile""#))
        #expect(cli.contains("writeStreamingProfile"))
        #expect(cli.contains("writeSystemPressure"))
        #expect(cli.contains("MLXPressStreamingProfile.snapshot()"))
        #expect(cli.contains("reasoning-preview="))
        #expect(metricsSummary.contains("decode_tokens_per_second"))
        #expect(metricsSummary.contains("peak_footprint_delta_percent_of_model"))
        #expect(metricsSummary.contains("peak_mlx_active_mb"))
        #expect(metricsSummary.contains("mlx_active_mb="))
        #expect(metricsSummary.contains("top_profile_rows"))
        #expect(metricsSummary.contains("bandwidth_mb_per_second"))
        #expect(metricsSummary.contains("profile_read_mb_per_generated_token"))
        #expect(metricsSummary.contains("profile_read_mb_per_gen_token"))
        #expect(metricsSummary.contains("tensor.mach_offset_read"))
        #expect(metricsSummary.contains("file_read_pressure_passed"))
        #expect(metricsSummary.contains("effective_read_pressure_passed"))
        #expect(metricsSummary.contains("effective_read_mb_per_gen_token"))
        #expect(metricsSummary.contains("file_read_gate"))
        #expect(metricsSummary.contains("effective_read_gate"))
        #expect(metricsSummary.contains("active_expert_trace_calls"))
        #expect(metricsSummary.contains("trace_reuse"))
        #expect(metricsSummary.contains("active_slice_residency_hit_rate"))
        #expect(metricsSummary.contains("active_slice_residency_tensor_resident_mb"))
        #expect(metricsSummary.contains("active_slice_residency_bank_resident_mb"))
        #expect(metricsSummary.contains("cache_byte_hit_rate"))
        #expect(metricsSummary.contains("tensor_resident_mb"))
        #expect(metricsSummary.contains("bank_resident_mb"))
        #expect(metricsSummary.contains("system_swapout_mb"))
        #expect(metricsSummary.contains("swapout_mb"))
        #expect(streamingExperts.contains("F_NOCACHE"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_F_NOCACHE"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_F_NOCACHE_THRESHOLD_GB"))
        #expect(streamingExperts.contains("active-expert read cache policy"))
        #expect(streamingExperts.contains("totalTensorBytes"))
        #expect(streamingExperts.contains("mlXPressStreamingUseFNoCache"))
        #expect(streamingExperts.contains("Darwin.pread"))
        #expect(streamingExperts.contains("cachedFileDescriptors"))
        #expect(streamingExperts.contains("cachedDarwinFileDescriptor"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_MMAP_ACTIVE_TENSORS"))
        #expect(streamingExperts.contains("tensor.mmap_array"))
        #expect(streamingExperts.contains("tensor.stacked_mmap_array"))
        #expect(streamingExperts.contains("mlx_array_new_mmap_file_region"))
        #expect(streamingExperts.contains("JANGTQStackedOffsetDescriptor"))
        #expect(streamingExperts.contains("stackedOffsetDescriptor"))
        #expect(streamingExperts.contains("expertByteOffsets"))
        #expect(streamingExperts.contains("expert-major-single-file-offsets"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_OFFSET_KERNELS"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER"))
        #expect(streamingExperts.contains("canUseOffsetDispatch"))
        #expect(streamingExperts.contains("hasAlignedOffsetShardGroups"))
        #expect(streamingExperts.contains("hasOffsetDispatchCoverage"))
        #expect(streamingExperts.contains("offset_dispatch_unaligned_shards"))
        #expect(streamingExperts.contains("offset_dispatch_flexible_shard_groups"))
        #expect(streamingExperts.contains("loadOffsetSpan"))
        #expect(streamingExperts.contains("presentExperts"))
        #expect(streamingExperts.contains("tensor.offset_span_active_filter"))
        #expect(streamingExperts.contains("tensor.offset_span_active_window"))
        #expect(streamingExperts.contains("tensor.offset_span_active_segments"))
        #expect(streamingExperts.contains("JANGTQActiveOffsetWindow"))
        #expect(streamingExperts.contains("activeExpertByteWindows"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB"))
        #expect(streamingExperts.contains("tensor.offset_span_active_coalesced_experts"))
        #expect(streamingExperts.contains("activeExpertWindows"))
        #expect(streamingExperts.contains("activeExperts: Set<Int>? = nil"))
        #expect(streamingExperts.contains("mlXPressStreamingGateUpOffsetSpanGroups"))
        #expect(streamingExperts.contains("mlXPressStreamingDownOffsetSpanGroups"))
        #expect(streamingExperts.contains("fusedGateUpSwiGLUOffsets"))
        #expect(streamingExperts.contains("gatherTQOffsets"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_SCORED_DOWN_KERNELS"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_SCORED_DOWN_KERNELS"))
        #expect(streamingExperts.contains("callChunkScoredOffsets"))
        #expect(streamingExperts.contains("callChunkScoredSlotOffsets"))
        #expect(streamingExperts.contains("exactSlotSpans"))
        #expect(streamingExperts.contains("slot_offsets_decode_fast_path"))
        #expect(streamingExperts.contains("slot_offsets_gateup_missing_exact_span"))
        #expect(streamingExperts.contains("slot_offsets_down_missing_exact_span"))
        #expect(streamingExperts.contains("gateup.slot_offsets_build"))
        #expect(streamingExperts.contains("down.slot_offsets_scored_build"))
        #expect(streamingExperts.contains("gatherTQTopKOffsetsScored"))
        #expect(streamingExperts.contains("gatherTQTopKSlots8Scored"))
        #expect(kernels.contains("fusedGateUpSwiGLUSlots8"))
        #expect(kernels.contains("gatherTQSlots8Scored"))
        #expect(kernels.contains("kFusedSwiGLUSlots8Source"))
        #expect(kernels.contains("kGatherTQSlots8ScoredSource"))
        #expect(streamingExperts.contains("gateup.offset_build"))
        #expect(streamingExperts.contains("down.offset_build"))
        #expect(streamingExperts.contains("down.offset_scored_build"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_EXPERT_CACHE_MB"))
        #expect(streamingExperts.contains("StreamingTensorCacheKey"))
        #expect(streamingExperts.contains("MLXPressActiveSliceResidency"))
        #expect(streamingExperts.contains("MLXPressActiveSliceResidencySnapshot"))
        #expect(streamingExperts.contains("StreamingSliceCacheEntry"))
        #expect(streamingExperts.contains("tensorCacheBudgetBytes"))
        #expect(streamingExperts.contains("tensorCacheHits"))
        #expect(streamingExperts.contains("tensorResidentBytes"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_BANK_CACHE_MB"))
        #expect(streamingExperts.contains("StreamingActiveBankCacheKey"))
        #expect(streamingExperts.contains("activeBankCacheHits"))
        #expect(streamingExperts.contains("storeActiveBank"))
        #expect(streamingExperts.contains("storeTensor"))
        #expect(streamingExperts.contains("storeSliceData"))
        #expect(streamingExperts.contains("cachedSliceData"))
        #expect(streamingExperts.contains("tensor.active_slice_cache_hit"))
        #expect(streamingExperts.contains("MLXArray(data, shape, dtype: dtype)"))
        #expect(streamingExperts.contains("VMLX_JANGTQ_PREFILL_STEP"))
        #expect(streamingExperts.contains("ref.byteCount % expertCount == 0"))
        #expect(
            streamingExperts.contains(
                "let expertOffset = ref.offset + UInt64(expertIdx * expertByteCount)"))
        #expect(streamingExperts.contains("isStreamableRoutedTensorKey"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_EXPERTS"))
        #expect(streamingExperts.contains("configureModelDirectory"))
        #expect(streamingExperts.contains("configuredModelDirectory"))
        #expect(
            streamingExperts.contains(
                #"switch_mlp\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)"#))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_PROFILE"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_PROFILE_EVERY"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_PROFILE_TOP"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_EVAL_LAYER_STRIDE"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_EVAL_LAYER_STRIDE"))
        #expect(streamingExperts.contains("mlXPressStreamingShouldEvaluateLayer"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_EVAL_MLX_PEAK_MB"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_EVAL_MLX_PEAK_MB"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_EVAL_MLX_ACTIVE_MB"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_EVAL_RESET_PEAK"))
        #expect(streamingExperts.contains("MLXPressStreamingAllocatorEvalScheduler"))
        #expect(streamingExperts.contains("scheduler.allocator_eval"))
        #expect(streamingExperts.contains("mlXPressStreamingMaterializeIfNeeded"))
        #expect(streamingExperts.contains("MLXPressStreamingProfileSnapshot"))
        #expect(streamingExperts.contains("MLXPressStreamingProfileRow"))
        #expect(streamingExperts.contains("snapshot(maxRows: Int? = nil)"))
        #expect(streamingExperts.contains("MLXPressActiveExpertTrace"))
        #expect(streamingExperts.contains("MLXPRESS_ACTIVE_EXPERT_TRACE"))
        #expect(streamingExperts.contains("consecutiveReuseTouches"))
        #expect(streamingExperts.contains("topExperts"))
        #expect(streamingExperts.contains("bytes/call=%.2fMB"))
        #expect(streamingExperts.contains("bw=%.2fMB/s"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_BANK_LOAD"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_BANK_LOAD"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_DIRECT_STACKED"))
        #expect(streamingExperts.contains("JANGPRESS_STREAMING_DIRECT_STACKED"))
        #expect(streamingExperts.contains("loadDirectStack"))
        #expect(streamingExperts.contains("canUseDirectStacked"))
        #expect(streamingExperts.contains("residentExpertsEnabled"))
        #expect(streamingExperts.contains("MLXPRESS_RESIDENT_EXPERTS"))
        #expect(streamingExperts.contains("registerResidentTensor"))
        #expect(streamingExperts.contains("tensor.direct_stacked_mmap_load"))
        #expect(streamingExperts.contains("withMmapSafetensorsEnvForDirectStackedLoad"))
        #expect(streamingExperts.contains("directStackedShardCache"))
        #expect(
            streamingExperts.contains(
                #"MLXPressStreamingProfile.time("router.indices_readback")"#))
        #expect(
            streamingExperts.contains(#"MLXPressStreamingProfile.time("tensor.stacked_read""#))
        #expect(
            streamingExperts.contains(#""tensor.stacked_bank_read""#)
        )
        #expect(
            streamingExperts.contains(#""tensor.stacked_bank_cached_assemble""#)
        )
        #expect(
            streamingExperts.contains(#""tensor.stacked_bank_array""#)
        )
        #expect(
            streamingExperts.contains(#"profileName: "gateup.eval""#))
        #expect(
            streamingExperts.contains(#"profileName: "down.eval""#))
        #expect(
            streamingExperts.contains(
                """
                let indexValues = allIndexValues.map {
                                Array($0[valueStart ..< valueEnd])
                            }
                """)
        )
        #expect(
            streamingExperts.contains(
                "slotIndexValues.append(allIndexValues[token * kSlots + slot])"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS"))
        #expect(streamingExperts.contains(#"?? "4""#))
        #expect(streamingExperts.contains("reduceChunkBatchedTopK"))
        #expect(
            streamingExperts.contains(
                "if fastReduceMaxTokens > 0 && tokenCount <= fastReduceMaxTokens"))
        #expect(streamingExperts.contains("let scoresFlat = stopGradient(scores.reshaped"))
        #expect(streamingExperts.contains("if arrays.count == 1"))
        #expect(streamingExperts.contains("arrays[0].expandedDimensions(axis: 0)"))
        #expect(streamingExperts.contains("func loadStack("))
        #expect(streamingExperts.contains("readStackedSlicesNoCacheDarwin"))
        #expect(streamingExperts.contains("MLXPRESS_STREAMING_SOURCE_ORDER_READS"))
        #expect(streamingExperts.contains("StreamingStackedSliceReadPlan"))
        #expect(streamingExperts.contains("makeStackedSliceReadPlan"))
        #expect(streamingExperts.contains(#""tensor.stacked_bank_read_range""#))
        #expect(streamingExperts.contains("materializeBeforeRouterReadback"))
        #expect(load.contains("declaresJANGTQNative"))
        #expect(load.contains("JANGTQStreamingExperts.configureModelDirectory(modelDirectory)"))
        #expect(load.contains("JANGTQStreamingExperts.isStreamableRoutedTensorKey(key)"))
        #expect(load.contains("generating deterministic signs/codebooks on demand"))
        #expect(minimaxJANGTQ.contains("JANGTQStreamingExperts.residentExpertsEnabled"))
        #expect(minimaxJANGTQ.contains("JANGTQStreamingExperts.registerResidentTensor"))
        #expect(loadTimeStacking.contains("MLXPRESS_LOADTIME_MATERIALIZE_STACKS"))
        #expect(loadTimeStacking.contains("JANGPRESS_LOADTIME_MATERIALIZE_STACKS"))
        #expect(loadTimeStacking.contains("MLXPRESS_LOADTIME_MACH_STACKS"))
        #expect(loadTimeStacking.contains("JANGPRESS_LOADTIME_MACH_STACKS"))
        #expect(loadTimeStacking.contains("loadTimeMachBackedStacked"))
        #expect(minimaxJANGTQ.contains("loadTimeMachBackedStacked(stacked, label: target)"))
        #expect(kernels.contains("hadamardShuffleLE1024"))
        #expect(kernels.contains("simd_shuffle_xor"))
        #expect(kernels.contains("if maxBlock <= 1024"))
        #expect(kernels.contains("fusedGateUpSwiGLUOffsets"))
        #expect(kernels.contains("gatherTQTopKOffsets"))
        #expect(kernels.contains("gatherTQTopKOffsetsScored"))
        #expect(kernels.contains("gatherTQOffsetsScored"))
        #expect(kernels.contains("packed_offsets[expert]"))
        #expect(kernels.contains("norm_offsets[expert]"))
        #expect(kernels.contains("scores[row_idx]"))
        #expect(kernels.contains("NumPyPCG64.generateRandomSigns"))
        #expect(kernels.contains("TQCodebook.computeCodebook"))
        #expect(deepseekJANGTQ.contains("MLXPRESS_STREAMING_MOE_TOKEN_CHUNK_SIZE"))
        #expect(deepseekJANGTQ.contains("VMLX_JANGTQ_PREFILL_STEP"))
        #expect(deepseekJANGTQ.contains("VMLX_JANGTQ_PREFILL_SYNC"))
        #expect(deepseekJANGTQ.contains("mxtqDownBits"))
        #expect(deepseekJANGTQ.contains("downBits: jangtq.mxtqDownBits ?? jangtq.mxtqBits"))
        #expect(deepseekJANGTQ.contains("callStreamingChunks"))
        #expect(deepseekJANGTQ.contains("MLXPRESS_STREAMING_EVAL_DECODER_LAYER"))
        #expect(deepseekJANGTQ.contains("evalStreamingLayerOutput"))
        #expect(deepseekJANGTQ.contains("JANGTQStreamingExperts.materializeBeforeRouterReadback"))
        #expect(deepseekJANGTQ.contains("routeInput = stopGradient(x)"))
        #expect(deepseekJANGTQ.contains(#"MLXPressStreamingProfile.time("kimi.router_gate""#))
        #expect(deepseekJANGTQ.contains(#"MLXPressStreamingProfile.time("kimi.streaming_reduce""#))
        #expect(deepseekJANGTQ.contains(#"MLXPressStreamingProfile.time("kimi.layer_eval""#))
        #expect(cli.contains(#"MLXPressStreamingProfile.dump(reason: "turn=\(turnIndex + 1)-end")"#))
        #expect(deepseekV3.contains("let originalScores = sigmoid(hiddenStates.asType(.float32))"))
        #expect(deepseekV3.contains("if nGroup > 1"))
        #expect(deepseekV3.contains("let inds = argPartition(-scoresForChoice"))
        #expect(deepseekV3.contains("var scores = takeAlong(originalScores, inds, axis: -1)"))
        #expect(deepseekV3.contains("let output = mlaScaledDotProductAttention"))
        #expect(deepseekV3.contains("decode"))
        #expect(attentionUtils.contains("public func mlaScaledDotProductAttention"))
        #expect(attentionUtils.contains("let queryLength = queries.dim(2)"))
        #expect(attentionUtils.contains("if queryLength == 1 && originalDtype != .float32"))
        #expect(attentionUtils.contains("queries.asType(.float32)"))
        #expect(attentionUtils.contains("return out.asType(originalDtype)"))
        #expect(coldSweep.contains("mlx_safetensors_mmap_advise_layer"))
        #expect(deepseekJANGTQ.contains("MLXPressMmapColdSweep.afterLayer"))
        #expect(hy3.contains("MLXPressMmapColdSweep.afterLayer"))
        #expect(safetensors.contains("safetensors_mmap_advise_layer"))
        #expect(safetensors.contains("load_safetensors_mmap"))
        #expect(safetensors.contains("MLX_SAFETENSORS_MMAP_COLD_ADVICE"))
        #expect(safetensors.contains("MADV_PAGEOUT"))
        #expect(safetensors.contains("MADV_DONTNEED"))
        #expect(safetensors.contains("MADV_WILLNEED"))
        #expect(safetensors.contains("MS_INVALIDATE"))
        #expect(safetensors.contains("msync"))
    }

    @Test("readiness checklist tracks attention, position, media, and companion cache axes")
    func readinessChecklistTracksArchitectureAxes() throws {
        let readiness = try String(
            contentsOfFile: "Sources/MLXPress/MLXPressModelReadiness.swift",
            encoding: .utf8)

        #expect(readiness.contains("attention-architecture-classified"))
        #expect(readiness.contains("hadamard-tq-matmul-contract"))
        #expect(readiness.contains("rope-mrope-position-contract"))
        #expect(readiness.contains("resident compute with macOS cold-page reclaim/compression"))
        #expect(readiness.contains("a coherent 1 tok/s streaming row is not the target methodology"))
        #expect(readiness.contains("usable decode token/s"))
        #expect(readiness.contains("cache-block-storage-encode"))
        #expect(readiness.contains("hybrid-companion-state-split"))
        #expect(readiness.contains("vl-vector-media-cache-proof"))
        #expect(readiness.contains("parser-autodetect-stack"))
        #expect(readiness.contains("per-turn-ram-speed-artifact"))
        #expect(readiness.contains("cold-warm-deviation-proof"))
        #expect(readiness.contains("synchronous-prompt-boundary-rederive"))
        #expect(!readiness.contains("async-rederive-warm-pass"))
        let batchEngineTests = try String(
            contentsOfFile: "Tests/MLXLMTests/BatchEngineTurboQuantTests.swift",
            encoding: .utf8)
        let ssmReDerive = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/SSMReDerive.swift",
            encoding: .utf8)
        #expect(!batchEngineTests.contains("pending `SSMReDeriver` port"))
        #expect(batchEngineTests.contains("Detached async SSM re-derive — intentionally not a production path"))
        #expect(ssmReDerive.contains("This is currently a parity hook, not the production store path."))
        #expect(ssmReDerive.contains("Do not describe Ultra or other hybrid\n/// rows as inline-capture or detached-async ready"))
        #expect(readiness.contains("attentionSummary"))
        #expect(readiness.contains("positionSummary"))
        #expect(readiness.contains("cacheStorageSummary"))
        #expect(readiness.contains("matmulSummary"))
        #expect(readiness.contains("positionVectorSummary"))
        #expect(readiness.contains("cacheEncodingSummary"))
        #expect(readiness.contains("hybridSplitSummary"))
        #expect(readiness.contains("2D/3D MRoPE"))
        #expect(readiness.contains("ZayaCCACache"))
        #expect(readiness.contains("HybridPoolCache"))

        let bundleFacts = try String(
            contentsOfFile: "Sources/MLXPress/MLXPressBundleFacts.swift",
            encoding: .utf8)
        #expect(bundleFacts.contains("MLXPressArchitectureFacts"))
        #expect(bundleFacts.contains("matmulKinds"))
        #expect(bundleFacts.contains("jangtq-hadamard-rotation"))
        #expect(bundleFacts.contains("randomized-hadamard-pow2-blocks"))
        #expect(bundleFacts.contains("mla-partial-rope-attention"))
        #expect(bundleFacts.contains("2d-3d-mrope-grid"))
        #expect(bundleFacts.contains("qwen-vl-image-grid-thw"))
        #expect(bundleFacts.contains("qwen-vl-video-grid-thw"))
        #expect(bundleFacts.contains("qwen-vl-position-ids[3,batch,seq]"))
        #expect(bundleFacts.contains("qwen-vl-mrope-delta-vector"))
        #expect(bundleFacts.contains("vision_config"))
        #expect(bundleFacts.contains("has_vision"))
        #expect(bundleFacts.contains("boolValue"))
        #expect(bundleFacts.contains("declaresJangTQ"))
        #expect(bundleFacts.contains(#"weightFormat == "mxtq""#))
        #expect(bundleFacts.contains("profile.contains(\"jangtq\")"))
        #expect(bundleFacts.contains("disk-l2-sha256(modelKey,mediaSalt,tokenJson)"))
        #expect(
            bundleFacts.contains(
                "cache-policy-salt(kvMode,kvBits,kvGroup,maxKV,promptBoundaryRawKV)"))
        #expect(bundleFacts.contains("tqdiskserializer-v2-layer-kind-tags"))
        #expect(bundleFacts.contains("vl-partial-hit-rejected-for-media-token-region"))
        #expect(bundleFacts.contains("media-salted-cache-keys"))
        #expect(bundleFacts.contains("zaya-cca(conv_state,prev_hs)"))
        #expect(bundleFacts.contains("zaya-cca-kv-plus-conv_state-prev_hs"))
        #expect(readiness.contains("real media-cache proof for detected vision/audio config"))
    }

    @Test("Nemotron Ultra status does not overstate saved JPREG JangPress proof")
    func nemotronUltraStatusKeepsJPREGProofHonest() throws {
        let status = try String(
            contentsOfFile: "docs/NEMOTRON_ULTRA_RUNTIME_STATUS_2026_06_06.md",
            encoding: .utf8)

        #expect(status.contains("JangPress: enabled=false"))
        #expect(status.contains("RouterAdvice: enabled=false"))
        #expect(status.contains("mmap tracked Metal buffers: `94.4 GB`"))
        #expect(status.contains("DOT primitive count: `QuantizedMatmul=192`, `AsType=1152`."))
        #expect(status.contains("it is the expected Mamba/shared-expert affine-8 surface"))
        #expect(status.contains("Runtime speed gate: `PARTIAL`."))
        #expect(status.contains("Best saved live speed: `8.335 tok/s`"))
        #expect(status.contains("MoE remains a major bucket: `65.773 ms` across 48 layers."))
        #expect(status.contains("Mamba remains a major bucket: `64.157 ms` across 48 layers."))
        #expect(status.contains("Swift active-expert streaming for Nemotron Ultra remains explicit-only"))
        #expect(status.contains("The next aligned runtime work is not Osaurus wiring"))
        #expect(
            status.contains(
                """
                The saved JPREG row is a low-footprint mmap row, not an enabled JangPress
                  advisory row.
                """))
    }

    @Test("README keeps Nemotron Ultra resident speed separate from mmap speed")
    func readmeKeepsNemotronUltraReleaseBoundaryHonest() throws {
        let readme = try String(contentsOfFile: "README.md", encoding: .utf8)

        #expect(readme.contains("resident Swift decode | 8.1 tok/s"))
        #expect(readme.contains("low-footprint mmap decode | 7.0 tok/s full-run, 9.8-9.9 tok/s tail"))
        #expect(readme.contains("The current Swift runtime proof is intentionally split by runtime path"))
        #expect(readme.contains("It uses about 100 GB physical footprint"))
        #expect(readme.contains("first-decode cost remains the open low-footprint speed gap"))
        #expect(readme.contains("hybrid SSM disk-backed prefix-cache"))
        #expect(readme.contains("SSM companion hits and cache-salt isolation"))
    }

    @Test("Nemotron-H JANGTQ docs do not regress to obsolete wrapper-blocked status")
    func nemotronHJANGTQDocsTrackCurrentWrapperStatus() throws {
        let modelSource = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronH.swift",
            encoding: .utf8)
        let jangtqSource = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronHJANGTQ.swift",
            encoding: .utf8)
        let omniSource = try String(
            contentsOfFile: "Libraries/MLXVLM/Models/NemotronHOmni/NemotronHOmni.swift",
            encoding: .utf8)
        let production = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JANGPRESS-PRODUCTION.md",
            encoding: .utf8)
        let perModel = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JANGPRESS-PER-MODEL-RESULTS.md",
            encoding: .utf8)
        let deepTrace = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JANGPRESS-DEEP-TRACE.md",
            encoding: .utf8)
        let status = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/JANGPRESS-STATUS.md",
            encoding: .utf8)

        #expect(modelSource.contains("NemotronHJANGTQContext"))
        #expect(modelSource.contains("NemotronHJANGTQSwitchMLP("))
        #expect(modelSource.contains("weightedDecode(expertInput, inds, scores: scores)"))
        #expect(jangtqSource.contains("final class NemotronHJANGTQSwitchMLP"))
        #expect(jangtqSource.contains("func weightedDecode(_ x: MLXArray, _ indices: MLXArray, scores: MLXArray) -> MLXArray?"))
        #expect(omniSource.contains("public let jangtqContext: NemotronHJANGTQContext?"))

        for doc in [production, perModel, deepTrace, status] {
            #expect(!doc.contains("Nemotron-H JANGTQ wrapper missing"))
            #expect(!doc.contains("NemotronHJANGTQModel wrapper isn't wired"))
            #expect(!doc.contains("NemotronHJANGTQModel Swift wrapper"))
            #expect(!doc.contains("Nemotron is RSS-only path"))
            #expect(!doc.contains("Omni multimodal keys"))
        }
        #expect(production.contains("Current correction (2026-06-06)"))
        #expect(perModel.contains("Status superseded (2026-06-06)"))
        #expect(deepTrace.contains("2026-06-06 update"))
    }
}
