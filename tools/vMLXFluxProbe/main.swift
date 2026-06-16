import Foundation
import CryptoKit
import ImageIO
@preconcurrency import MLX
import vMLXFlux
import vMLXFluxKit

@main
struct VMLXFluxProbe {
    static func main() async {
        do {
            let options = try ProbeOptions(arguments: Array(CommandLine.arguments.dropFirst()))
            VMLXFluxModels.registerAll()
            VMLXFluxVideo.registerAll()

            let store = MLXStudioModelStore(root: options.root)
            let models = try store.scan()
            try FileManager.default.createDirectory(
                at: options.artifactDirectory,
                withIntermediateDirectories: true)

            try writeScanArtifacts(
                models: models,
                artifactDirectory: options.artifactDirectory,
                jsonOutput: options.json)

            if options.matrix {
                try await runMatrixProbe(
                    models: models,
                    options: options,
                    artifactDirectory: options.artifactDirectory)
            } else if let requestedModel = options.model {
                guard let local = try store.resolve(name: requestedModel) else {
                    throw ProbeError("model \(requestedModel) not found under \(options.root.path)")
                }
                try writeLocalModelFacts(local, artifactDirectory: options.artifactDirectory)
                if options.load || options.generate {
                    try await runLoadProbe(
                        local: local,
                        options: options,
                        artifactDirectory: options.artifactDirectory)
                }
            }
        } catch {
            fputs("vmlxflux-probe error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func runMatrixProbe(
        models: [LocalFluxModel],
        options: ProbeOptions,
        artifactDirectory: URL
    ) async throws {
        var rows: [[String: Any]] = []
        for local in models {
            try writeLocalModelFacts(local, artifactDirectory: artifactDirectory)
            let payload = try await runLoadProbe(
                local: local,
                options: options,
                artifactDirectory: artifactDirectory)
            rows.append(matrixRow(local: local, payload: payload))
        }

        let matrixPayload: [String: Any] = [
            "started_at": isoTimestamp(options.startedAt),
            "finished_at": isoTimestamp(),
            "root": options.root.path,
            "model_count": models.count,
            "generate_requested": options.generate,
            "width": options.width,
            "height": options.height,
            "steps": options.steps,
            "turns": options.turns,
            "rows": rows,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: matrixPayload,
            options: [.prettyPrinted, .sortedKeys])
        let matrixURL = artifactDirectory.appendingPathComponent("compatibility-matrix.json")
        try data.write(to: matrixURL)
        try writeMatrixMarkdown(rows: rows, artifactDirectory: artifactDirectory)
        print("compatibility matrix artifact: \(matrixURL.path)")
    }

    private static func writeScanArtifacts(
        models: [LocalFluxModel],
        artifactDirectory: URL,
        jsonOutput: Bool
    ) throws {
        let rows = models.map(modelJSON)
        let data = try JSONSerialization.data(
            withJSONObject: rows,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: artifactDirectory.appendingPathComponent("scan.json"))

        var lines: [String] = [
            "# vMLX Flux Local Model Scan",
            "",
            "| Directory | Canonical | Kind | Quant | Safetensors | Bytes | Readiness | Reasons |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
        for model in models {
            lines.append(
                "| \(model.directoryName) | \(model.canonicalName ?? "unknown") | \(model.kind?.rawValue ?? "unknown") | \(model.quantizationBits.map(String.init) ?? "-") | \(model.safetensorCount) | \(model.totalBytes) | \(model.readiness.rawValue) | \(model.blockedReasons.joined(separator: "; ")) |")
        }
        try lines.joined(separator: "\n")
            .write(
                to: artifactDirectory.appendingPathComponent("scan.md"),
                atomically: true,
                encoding: .utf8)

        if jsonOutput {
            FileHandle.standardOutput.write(data)
            print("")
        } else {
            print("scanned \(models.count) local image models")
            print("artifacts: \(artifactDirectory.path)")
            for model in models {
                print("\(model.directoryName): canonical=\(model.canonicalName ?? "unknown") readiness=\(model.readiness.rawValue) safetensors=\(model.safetensorCount) bytes=\(model.totalBytes)")
            }
        }
    }

    private static func writeLocalModelFacts(
        _ model: LocalFluxModel,
        artifactDirectory: URL
    ) throws {
        let factsURL = artifactDirectory.appendingPathComponent("\(model.directoryName)-facts.json")
        let data = try JSONSerialization.data(
            withJSONObject: modelJSON(model),
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: factsURL)
    }

    @discardableResult
    private static func runLoadProbe(
        local: LocalFluxModel,
        options: ProbeOptions,
        artifactDirectory: URL
    ) async throws -> [String: Any] {
        let logURL = artifactDirectory.appendingPathComponent("\(local.directoryName)-load.json")
        let startedAt = Date()
        var payload: [String: Any] = [
            "model": modelJSON(local),
            "started_at": isoTimestamp(startedAt),
            "generate_requested": options.generate,
            "edit_requested": options.edit,
            "qwen_edit_prompt_requested": options.qwenEditPrompt,
            "qwen_edit_conditioning_requested": options.qwenEditConditioning,
            "qwen_edit_vision_requested": options.qwenEditVision,
            "qwen_edit_denoise_requested": options.qwenEditDenoise,
            "turns": options.turns,
            "width": options.width,
            "height": options.height,
            "width_explicit": options.widthExplicit,
            "height_explicit": options.heightExplicit,
            "source_image": options.sourceImage?.path ?? NSNull(),
            "source_images": options.sourceImages.map(\.path),
            "mask_image": options.maskImage?.path ?? NSNull(),
            "steps": options.steps,
            "seed": options.seed.map { $0 as Any } ?? NSNull(),
        ]

        do {
            let engine = FluxEngine()
            let loaded = try await engine.load(name: local.directoryName,
                                               from: MLXStudioModelStore(root: options.root))
            payload["load_status"] = "loaded"
            payload["loaded_model"] = modelJSON(loaded)
            payload["load_elapsed_seconds"] = Date().timeIntervalSince(startedAt)

            if options.generate {
                var turnRecords: [[String: Any]] = []
                for (index, prompt) in options.turns.enumerated() {
                    let request = ImageGenRequest(
                        prompt: prompt,
                        negativePrompt: options.negativePrompt,
                        width: options.width,
                        height: options.height,
                        steps: options.steps,
                        guidance: options.guidance ?? defaultGuidance(for: loaded),
                        seed: options.seed ?? UInt64(index + 1),
                        outputDir: options.outputDirectory)
                    let turnStart = Date()
                    var record: [String: Any] = [
                        "turn": index + 1,
                        "prompt": prompt,
                        "started_at": isoTimestamp(turnStart),
                    ]
                    do {
                        let stream = await engine.generate(request)
                        var steps: [[String: Any]] = []
                        var completedURL: String?
                        for try await event in stream {
                            switch event {
                            case .step(let step, let total, let eta):
                                steps.append([
                                    "step": step,
                                    "total": total,
                                    "eta_seconds": eta.map { $0 as Any } ?? NSNull(),
                                ])
                            case .preview(let data, let step):
                                steps.append([
                                    "preview_step": step,
                                    "preview_bytes": data.count,
                                ])
                            case .completed(let url, let seed):
                                completedURL = url.path
                                record["seed"] = seed
                            case .failed(let message, let hfAuth):
                                record["status"] = "failed_event"
                                record["message"] = message
                                record["hf_auth"] = hfAuth
                            case .cancelled:
                                record["status"] = "cancelled"
                            }
                        }
                        record["steps"] = steps
                        if let completedURL {
                            record["status"] = "completed"
                            record["output"] = completedURL
                            record["image_diagnostics"] = imageDiagnostics(
                                for: URL(fileURLWithPath: completedURL))
                        } else if record["status"] == nil {
                            record["status"] = "no_completed_event"
                        }
                    } catch {
                        record["status"] = "threw"
                        record["error"] = String(describing: error)
                    }
                    record["elapsed_seconds"] = Date().timeIntervalSince(turnStart)
                    turnRecords.append(record)
                }
                payload["generation_turns"] = turnRecords
            }

            if options.edit {
                guard !options.sourceImages.isEmpty else {
                    throw ProbeError("--edit requires --source-image")
                }
                var turnRecords: [[String: Any]] = []
                for (index, prompt) in options.turns.enumerated() {
                    let request = try ImageEditRequest(
                        prompt: prompt,
                        sourceImages: options.sourceImages,
                        mask: options.maskImage,
                        strength: options.strength,
                        width: options.widthExplicit ? options.width : nil,
                        height: options.heightExplicit ? options.height : nil,
                        steps: options.steps,
                        guidance: options.guidance ?? 4.0,
                        seed: options.seed ?? UInt64(index + 1),
                        outputDir: options.outputDirectory)
                    let turnStart = Date()
                    var record: [String: Any] = [
                        "turn": index + 1,
                        "prompt": prompt,
                        "source_image": options.sourceImage?.path ?? NSNull(),
                        "source_images": options.sourceImages.map(\.path),
                        "mask_image": options.maskImage?.path ?? NSNull(),
                        "started_at": isoTimestamp(turnStart),
                    ]
                    do {
                        let stream = await engine.edit(request)
                        var steps: [[String: Any]] = []
                        var completedURL: String?
                        for try await event in stream {
                            switch event {
                            case .step(let step, let total, let eta):
                                steps.append([
                                    "step": step,
                                    "total": total,
                                    "eta_seconds": eta.map { $0 as Any } ?? NSNull(),
                                ])
                            case .preview(let data, let step):
                                steps.append([
                                    "preview_step": step,
                                    "preview_bytes": data.count,
                                ])
                            case .completed(let url, let seed):
                                completedURL = url.path
                                record["seed"] = seed
                            case .failed(let message, let hfAuth):
                                record["status"] = "failed_event"
                                record["message"] = message
                                record["hf_auth"] = hfAuth
                            case .cancelled:
                                record["status"] = "cancelled"
                            }
                        }
                        record["steps"] = steps
                        if let completedURL {
                            record["status"] = "completed"
                            record["output"] = completedURL
                            record["image_diagnostics"] = imageDiagnostics(
                                for: URL(fileURLWithPath: completedURL))
                        } else if record["status"] == nil {
                            record["status"] = "no_completed_event"
                        }
                    } catch {
                        record["status"] = "threw"
                        record["error"] = String(describing: error)
                    }
                    record["elapsed_seconds"] = Date().timeIntervalSince(turnStart)
                    turnRecords.append(record)
                }
                payload["edit_turns"] = turnRecords
            }

            if options.qwenEditConditioning {
                guard !options.sourceImages.isEmpty else {
                    throw ProbeError("--qwen-edit-conditioning requires --source-image")
                }
                guard local.canonicalName == "qwen-image-edit" else {
                    throw ProbeError("--qwen-edit-conditioning requires a qwen-image-edit model")
                }
                let plan = try QwenImageEditPreprocessPlan(
                    sourceImages: options.sourceImages,
                    requestedWidth: options.widthExplicit ? options.width : nil,
                    requestedHeight: options.heightExplicit ? options.height : nil,
                    steps: options.steps,
                    guidance: options.guidance ?? 4.0)
                let conditioningStart = Date()
                let conditioning = try QwenImageEditConditioner.encode(
                    modelPath: local.directory,
                    sourceImages: options.sourceImages,
                    plan: plan)
                payload["qwen_edit_conditioning"] = [
                    "status": "encoded",
                    "elapsed_seconds": Date().timeIntervalSince(conditioningStart),
                    "output_width": plan.outputWidth,
                    "output_height": plan.outputHeight,
                    "vae_width": plan.vaeWidth,
                    "vae_height": plan.vaeHeight,
                    "conditioning_width": plan.vlWidth,
                    "conditioning_height": plan.vlHeight,
                    "patch_rows": conditioning.patchRows,
                    "patch_columns": conditioning.patchColumns,
                    "image_count": conditioning.imageCount,
                    "latents_shape": conditioning.latents.shape,
                    "image_ids_shape": conditioning.imageIDs.shape,
                    "latents_stats": mlxStats(conditioning.latents),
                    "image_ids_stats": mlxStats(conditioning.imageIDs),
                ]
            }

            if options.qwenEditPrompt {
                guard !options.sourceImages.isEmpty else {
                    throw ProbeError("--qwen-edit-prompt requires --source-image")
                }
                guard local.canonicalName == "qwen-image-edit" else {
                    throw ProbeError("--qwen-edit-prompt requires a qwen-image-edit model")
                }
                let plan = try QwenImageEditPreprocessPlan(
                    sourceImages: options.sourceImages,
                    requestedWidth: options.widthExplicit ? options.width : nil,
                    requestedHeight: options.heightExplicit ? options.height : nil,
                    steps: options.steps,
                    guidance: options.guidance ?? 4.0)
                let visionInputs = try options.sourceImages.map {
                    try QwenImageEditPreprocessor.visionInput(sourceImage: $0, plan: plan)
                }
                let imageTokenCounts = visionInputs.map(\.imageTokenCount)
                let totalImageTokenCount = imageTokenCounts.reduce(0, +)
                let tokenizer = try await QwenImageEditPromptTokenizer(modelPath: local.directory)
                var promptRecords: [[String: Any]] = []
                for prompt in options.turns {
                    let promptInput = try QwenImageEditPreprocessor.visionLanguagePrompt(
                        prompt: prompt,
                        imageTokenCounts: imageTokenCounts)
                    let tokens = tokenizer.tokenize(promptInput)
                    guard tokens.imageTokenCount == totalImageTokenCount else {
                        throw ProbeError(
                            "qwen edit prompt image token count mismatch: got \(tokens.imageTokenCount), expected \(totalImageTokenCount)")
                    }
                    promptRecords.append([
                        "status": "tokenized",
                        "prompt": prompt,
                        "sequence_length": tokens.sequenceLength,
                        "input_ids_shape": tokens.inputIDs.shape,
                        "attention_mask_shape": tokens.attentionMask.shape,
                        "image_token_id": QwenImageEditPreprocessor.imageTokenID,
                        "image_token_count": tokens.imageTokenCount,
                        "expected_image_token_count": totalImageTokenCount,
                        "image_token_counts": imageTokenCounts,
                        "template_drop_index": tokens.templateDropIndex,
                        "image_grid_thw": visionInputs.map(\.imageGridTHW),
                    ])
                }
                payload["qwen_edit_prompt_tokens"] = promptRecords
            }

            if options.qwenEditVision {
                guard !options.sourceImages.isEmpty else {
                    throw ProbeError("--qwen-edit-vision requires --source-image")
                }
                guard local.canonicalName == "qwen-image-edit" else {
                    throw ProbeError("--qwen-edit-vision requires a qwen-image-edit model")
                }
                let plan = try QwenImageEditPreprocessPlan(
                    sourceImages: options.sourceImages,
                    requestedWidth: options.widthExplicit ? options.width : nil,
                    requestedHeight: options.heightExplicit ? options.height : nil,
                    steps: options.steps,
                    guidance: options.guidance ?? 4.0)
                var encodingRecords: [[String: Any]] = []
                for prompt in options.turns {
                    let visionStart = Date()
                    let encoding = try await QwenImageEditPromptImageEncoder.encode(
                        modelPath: local.directory,
                        sourceImages: options.sourceImages,
                        prompt: prompt,
                        plan: plan)
                    encodingRecords.append([
                        "status": "encoded",
                        "elapsed_seconds": Date().timeIntervalSince(visionStart),
                        "prompt": prompt,
                        "image_grid_thw": encoding.features.imageGridTHWs,
                        "feature_shape": encoding.features.imageFeatures.shape,
                        "feature_stats": mlxStats(encoding.features.imageFeatures),
                        "image_token_count": encoding.features.imageTokenCount,
                        "token_sequence_length": encoding.tokens.sequenceLength,
                        "token_image_count": encoding.tokens.imageTokenCount,
                        "template_drop_index": encoding.tokens.templateDropIndex,
                        "prompt_embeds_shape": encoding.promptEmbeddings.promptEmbeds.shape,
                        "prompt_mask_shape": encoding.promptEmbeddings.attentionMask.shape,
                        "prompt_embeds_stats": mlxStats(encoding.promptEmbeddings.promptEmbeds),
                        "prompt_mask_stats": mlxStats(encoding.promptEmbeddings.attentionMask),
                        "matches_features": encoding.tokens.imageTokenCount == encoding.features.imageTokenCount,
                    ])
                }
                payload["qwen_edit_vision_language"] = encodingRecords
            }

            if options.qwenEditDenoise {
                guard !options.sourceImages.isEmpty else {
                    throw ProbeError("--qwen-edit-denoise requires --source-image")
                }
                guard local.canonicalName == "qwen-image-edit" else {
                    throw ProbeError("--qwen-edit-denoise requires a qwen-image-edit model")
                }
                let plan = try QwenImageEditPreprocessPlan(
                    sourceImages: options.sourceImages,
                    requestedWidth: options.widthExplicit ? options.width : nil,
                    requestedHeight: options.heightExplicit ? options.height : nil,
                    steps: options.steps,
                    guidance: options.guidance ?? 4.0)
                var denoiseRecords: [[String: Any]] = []
                for (index, prompt) in options.turns.enumerated() {
                    let seed = options.seed ?? UInt64(index + 1)
                    let denoiseStart = Date()
                    let result = try await QwenImageEditDenoiseProbe.predictVelocity(
                        modelPath: local.directory,
                        sourceImages: options.sourceImages,
                        prompt: prompt,
                        plan: plan,
                        seed: seed)
                    denoiseRecords.append([
                        "status": "predicted",
                        "elapsed_seconds": Date().timeIntervalSince(denoiseStart),
                        "prompt": prompt,
                        "seed": seed,
                        "output_width": plan.outputWidth,
                        "output_height": plan.outputHeight,
                        "vae_width": plan.vaeWidth,
                        "vae_height": plan.vaeHeight,
                        "conditioning_width": plan.vlWidth,
                        "conditioning_height": plan.vlHeight,
                        "steps": plan.steps,
                        "guidance": plan.guidance,
                        "target_latent_count": result.targetLatentCount,
                        "conditioning_latent_count": result.conditioningLatentCount,
                        "image_shapes": result.imageShapes.map { [$0.frame, $0.height, $0.width] },
                        "combined_velocity_shape": result.combinedVelocity.shape,
                        "target_velocity_shape": result.targetVelocity.shape,
                        "combined_velocity_stats": mlxStats(result.combinedVelocity),
                        "target_velocity_stats": mlxStats(result.targetVelocity),
                    ])
                }
                payload["qwen_edit_denoise"] = denoiseRecords
            }
        } catch {
            payload["load_status"] = "failed"
            payload["error"] = String(describing: error)
            payload["load_elapsed_seconds"] = Date().timeIntervalSince(startedAt)
        }

        payload["finished_at"] = isoTimestamp()
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: logURL)
        print("load probe artifact: \(logURL.path)")
        return payload
    }

    private static func defaultGuidance(for model: LocalFluxModel) -> Float {
        switch model.canonicalName {
        case "z-image-turbo":
            return 0
        case "ideogram":
            return 7
        default:
            return 3.5
        }
    }

    private static func matrixRow(
        local: LocalFluxModel,
        payload: [String: Any]
    ) -> [String: Any] {
        let turns = payload["generation_turns"] as? [[String: Any]] ?? []
        let completed = turns.filter { ($0["status"] as? String) == "completed" }.count
        let threw = turns.filter { ($0["status"] as? String) == "threw" }.count
        let failed = turns.filter {
            guard let status = $0["status"] as? String else { return true }
            return status != "completed"
        }.count
        let nativeStatus = runtimeStatus(for: local)
        let loadStatus = payload["load_status"] as? String ?? "not_requested"
        let gateStatus: String
        var gateReasons = runtimeBlockers(for: local)
        if local.readiness != .loadableScaffold {
            gateReasons.append(contentsOf: local.blockedReasons)
        }
        if loadStatus != "loaded" {
            gateReasons.append("native load did not complete")
        }
        if payload["generate_requested"] as? Bool == true {
            if failed > 0 {
                gateReasons.append("\(failed) generation turn(s) did not complete")
            }
            if nativeStatus != "production_ready" {
                gateReasons.append("native runtime status is \(nativeStatus)")
            }
        }
        if gateReasons.isEmpty {
            gateStatus = "production_candidate"
        } else if loadStatus == "loaded" {
            gateStatus = "blocked_after_load"
        } else {
            gateStatus = "blocked_before_load"
        }

        return [
            "directory_name": local.directoryName,
            "canonical_name": local.canonicalName ?? NSNull(),
            "kind": local.kind?.rawValue ?? NSNull(),
            "quantization_bits": local.quantizationBits.map { $0 as Any } ?? NSNull(),
            "components": local.components.map(\.rawValue).sorted(),
            "safetensor_count": local.safetensorCount,
            "total_bytes": local.totalBytes,
            "readiness": local.readiness.rawValue,
            "native_runtime_status": nativeStatus,
            "load_status": loadStatus,
            "generation_turns": turns.count,
            "generation_completed": completed,
            "generation_threw": threw,
            "generation_failed_or_missing": failed,
            "gate_status": gateStatus,
            "gate_reasons": Array(NSOrderedSet(array: gateReasons)) as? [String] ?? gateReasons,
            "artifact": "\(local.directoryName)-load.json",
        ]
    }

    private static func writeMatrixMarkdown(
        rows: [[String: Any]],
        artifactDirectory: URL
    ) throws {
        var lines: [String] = [
            "# vMLX Flux Native Compatibility Matrix",
            "",
            "| Directory | Canonical | Load | Generation | Native status | Gate | Reasons |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows {
            let directory = row["directory_name"] as? String ?? "unknown"
            let canonical = row["canonical_name"] as? String ?? "unknown"
            let load = row["load_status"] as? String ?? "unknown"
            let completed = row["generation_completed"] as? Int ?? 0
            let turns = row["generation_turns"] as? Int ?? 0
            let native = row["native_runtime_status"] as? String ?? "unknown"
            let gate = row["gate_status"] as? String ?? "unknown"
            let reasons = (row["gate_reasons"] as? [String] ?? [])
                .joined(separator: "; ")
                .replacingOccurrences(of: "\n", with: " ")
            lines.append("| \(directory) | \(canonical) | \(load) | \(completed)/\(turns) | \(native) | \(gate) | \(reasons) |")
        }
        try lines.joined(separator: "\n").write(
            to: artifactDirectory.appendingPathComponent("compatibility-matrix.md"),
            atomically: true,
            encoding: .utf8)
    }

    private static func modelJSON(_ model: LocalFluxModel) -> [String: Any] {
        [
            "directory": model.directory.path,
            "directory_name": model.directoryName,
            "canonical_name": model.canonicalName ?? NSNull(),
            "display_name": model.displayName,
            "kind": model.kind?.rawValue ?? NSNull(),
            "quantization_bits": model.quantizationBits.map { $0 as Any } ?? NSNull(),
            "components": model.components.map(\.rawValue).sorted(),
            "safetensor_count": model.safetensorCount,
            "total_bytes": model.totalBytes,
            "has_model_index": model.hasModelIndex,
            "readiness": model.readiness.rawValue,
            "blocked_reasons": model.blockedReasons,
            "native_runtime_status": runtimeStatus(for: model),
            "native_runtime_blockers": runtimeBlockers(for: model),
        ]
    }

    private static func runtimeStatus(for model: LocalFluxModel) -> String {
        switch model.canonicalName {
        case "z-image-turbo", "flux1-schnell", "qwen-image":
            return "native_pipeline_implemented"
        case "qwen-image-edit":
            return [4, 5, 6, 8].contains(model.quantizationBits ?? -1)
                && model.readiness == .loadableScaffold
                ? "native_pipeline_implemented"
                : "native_pipeline_partial"
        case "ideogram":
            return model.readiness == .loadableScaffold
                ? "native_pipeline_implemented"
                : "not_implemented"
        case "flux1-dev", "flux1-kontext", "flux1-fill",
             "flux2-klein", "flux2-klein-edit",
             "fibo", "seedvr2":
            return "not_implemented"
        case "wan-2.1", "wan-2.2":
            return "video_scaffold_only"
        case .some:
            return "unknown_model_runtime"
        case .none:
            return "unknown"
        }
    }

    private static func runtimeBlockers(for model: LocalFluxModel) -> [String] {
        switch model.canonicalName {
        case "z-image-turbo", "flux1-schnell":
            return [
                "current 103be437 main 4/8-bit live probes completed three turns with same-seed repeated-prompt SHA match, different-prompt SHA change, finite diagnostics where emitted, and viewed coherent apple/mountain images on 2026-06-16",
                "run a broader Osaurus-side production matrix before release promotion",
            ]
        case "qwen-image":
            if model.quantizationBits == 4
                && model.readiness == .loadableScaffold
            {
                return [
                    "current 103be437 main qwen-image 4-bit text-to-image path completed a 20-step three-turn live probe with same-seed prompt sensitivity, deterministic repeat, finite diagnostics, and viewed coherent apple/mountain images on 2026-06-16",
                    "current da782e8b main qwen-image 8-bit text-to-image path completed a 20-step three-turn live probe with same-seed prompt sensitivity, deterministic repeat, finite diagnostics, and viewed coherent apple/mountain images on 2026-06-16",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.quantizationBits == 6
                && model.readiness == .loadableScaffold
            {
                return [
                    "current 103be437 main qwen-image 6-bit text-to-image path completed a 20-step three-turn live probe with same-seed prompt sensitivity, deterministic repeat, finite diagnostics, and viewed coherent apple/mountain images on 2026-06-16",
                    "current da782e8b main qwen-image 8-bit text-to-image path completed a 20-step three-turn live probe with same-seed prompt sensitivity, deterministic repeat, finite diagnostics, and viewed coherent apple/mountain images on 2026-06-16",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.quantizationBits == 8
                && model.readiness == .loadableScaffold
            {
                return [
                    "current da782e8b main qwen-image 8-bit text-to-image path completed a 20-step three-turn live probe with same-seed prompt sensitivity, deterministic repeat, finite diagnostics, and viewed coherent apple/mountain images on 2026-06-16",
                    "source bundle: AbstractFramework/qwen-image-8bit, 29.5GB dry-run, staged as qwen-image-mflux-8bit with quantization_level=8 index metadata",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            return [
                "qwen-image 4-bit and 6-bit have current-103be live proof and qwen-image 8-bit has current-da782 live proof; this quant variant has not completed live generation",
                "live coherent text-to-image proof is missing for this quant variant",
            ]
        case "qwen-image-edit":
            if model.quantizationBits == 8
                && model.readiness == .loadableScaffold
            {
                return [
                    "current ed84bac6 main qwen-image-edit q8 text-image edit path completed a 20-step three-turn live probe with same-seed deterministic repeat and prompt-sensitive SHA changes on 2026-06-16",
                    "visual proof: q8 cleanly edits the source apple into a centered blue apple and a green pear on a plain white background",
                    "mask/inpaint edit fields are not wired yet",
                    "q3 is loadable after staging the missing shard but remains hidden because the viewed output is high-noise and not a clean prompt-following edit",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.quantizationBits == 6
                && model.readiness == .loadableScaffold
            {
                return [
                    "current ed84bac6 main qwen-image-edit q6 text-image edit path completed a 20-step three-turn live probe with same-seed deterministic repeat and prompt-sensitive SHA changes on 2026-06-16",
                    "visual proof: q6 cleanly edits the source apple into a centered blue apple and a green pear on a plain white background",
                    "mask/inpaint edit fields are not wired yet",
                    "q3 is loadable after staging the missing shard but remains hidden because the viewed output is high-noise and not a clean prompt-following edit",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.quantizationBits == 5
                && model.readiness == .loadableScaffold
            {
                return [
                    "current 103be437 main qwen-image-edit q5 text-image edit path completed a 20-step three-turn live probe with same-seed deterministic repeat and prompt-sensitive SHA changes on 2026-06-16",
                    "visual boundary: q5 cleanly edits blue apple and green pear; q4 has current-103be proof but remains noisier/weaker on shape-changing green-pear prompts; q6/q8 now have current-ed84 clean proof",
                    "mask/inpaint edit fields are not wired yet",
                    "q3 is loadable after staging the missing shard but remains hidden because the viewed output is high-noise and not a clean prompt-following edit",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.quantizationBits == 4
                && model.readiness == .loadableScaffold
            {
                return [
                    "current 103be437 main qwen-image-edit q4 text-image edit path completed a 20-step three-turn live probe with same-seed deterministic repeat and prompt-sensitive SHA changes on 2026-06-16",
                    "visual boundary: q4 changes color and shape but remains noisier/weaker on shape-changing green-pear prompts; q5/q6/q8 are cleaner current-head edit rows",
                    "mask/inpaint edit fields are not wired yet",
                    "q3 is loadable after staging the missing shard but remains hidden because the viewed output is high-noise and not a clean prompt-following edit",
                    "run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            if model.readiness != .loadableScaffold {
                return [
                    "local qwen-image-edit bundle is incomplete and cannot enter the native load path",
                    "qwen-image-edit q4/q5 have current-103be live proof and q6/q8 have current-ed84 live proof; this quant variant has not completed live generation",
                    "mask/inpaint edit fields are not wired yet",
                ]
            }
            return [
                "qwen-image-edit q4/q5 have current-103be live proof and q6/q8 have current-ed84 live proof; q3 loads but viewed q3 output is high-noise and not a clean prompt-following edit",
                "mask/inpaint edit fields are not wired yet",
                "live coherent edited-image proof is missing for this quant variant",
            ]
        case "flux1-dev", "flux1-kontext", "flux1-fill",
             "flux2-klein", "flux2-klein-edit", "fibo", "seedvr2":
            return [
                "model generate/edit/upscale body throws FluxError.notImplemented",
                "text encoder ports are missing",
                "safetensors-to-module key mapping is missing",
            ]
        case "ideogram":
            if model.readiness == .loadableScaffold {
                if model.directoryName.localizedCaseInsensitiveContains("nf4") {
                    return [
                        "Ideogram NF4 source path is wired through Qwen3 text encoder, conditional/unconditional 34-layer DiT, bitsandbytes NF4 linear dequantization, VAE decode, and PNG output",
                        "current 103be437 NF4 exact strict 512px object-icon probe completed three 20-step turns; apple and mountain prompts were coherent, prompt-sensitive, and repeated apple had identical SHA",
                        "NF4 proof artifact: docs/local/vmlx-flux-probes/2026-06-16-current-103be-ideogram-nf4-bb4-exact/ideogram-4-nf4-load.json",
                        "visual boundary: 103be seed/order variants with adjacent repeated prompts generated safety-filter-like text cards, so expose Ideogram as staged/testable with prompt-pattern caveats rather than a general clean object renderer",
                        "official ideogram-ai/ideogram-4-fp8 and ideogram-ai/ideogram-4-nf4 dry-runs still return access denied for the current HF account; current NF4 live proof uses the staged cocktailpeanut/ideogram-4-nf4 mirror",
                        "run a broader Osaurus-side production matrix before release promotion",
                    ]
                }
                return [
                    "Ideogram fp8 source path is wired through Qwen3 text encoder, conditional/unconditional 34-layer DiT, VAE decode, and PNG output",
                    "live 20-step fp8 typography probe completed after the rotary-half correction; HELLO/BANANA outputs were prompt-sensitive and repeated HELLO had identical SHA",
                    "current 103be437 main exact strict 512px object-icon probe completed three turns; apple and mountain prompts were coherent, prompt-sensitive, and repeated apple had identical SHA",
                    "visual boundary: 103be seed/order variants with adjacent repeated prompts generated safety-filter-like text cards, so expose Ideogram fp8 as staged/testable with prompt-pattern caveats rather than a general clean object renderer",
                    "official ideogram-ai/ideogram-4-fp8 and ideogram-ai/ideogram-4-nf4 dry-runs still return access denied for the current HF account; fp8 live proof uses the staged cocktailpeanut/ideogram-4-fp8 mirror",
                    "NF4 support has current-103be staged mirror proof; run a broader Osaurus-side production matrix before release promotion",
                ]
            }
            return [
                "local Ideogram bundle is incomplete and cannot enter the native load path",
                "official ideogram-ai/ideogram-4-fp8 and ideogram-ai/ideogram-4-nf4 dry-runs still return access denied for the current HF account",
            ]
        case "wan-2.1", "wan-2.2":
            return [
                "video path is scaffolded",
                "real Wan safetensors key mapping and scalable attention are missing",
            ]
        default:
            return []
        }
    }

    private static func imageDiagnostics(for url: URL) -> [String: Any] {
        do {
            let data = try Data(contentsOf: url)
            var result: [String: Any] = [
                "path": url.path,
                "bytes": data.count,
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            ]
            if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
               let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] {
                result["pixel_width"] = properties[kCGImagePropertyPixelWidth]
                result["pixel_height"] = properties[kCGImagePropertyPixelHeight]
                result["color_model"] = properties[kCGImagePropertyColorModel]
                result["has_alpha"] = properties[kCGImagePropertyHasAlpha]
            }
            return result
        } catch {
            return [
                "path": url.path,
                "error": String(describing: error),
            ]
        }
    }

    private static func mlxStats(_ array: MLXArray) -> [String: Any] {
        eval(array)
        let f = array.asType(.float32)
        let meanValue = mean(f).item(Float.self)
        let maxValue = MLX.max(f).item(Float.self)
        let minValue = (-MLX.max(-f)).item(Float.self)
        return [
            "shape": array.shape,
            "mean": meanValue,
            "min": minValue,
            "max": maxValue,
            "finite": meanValue.isFinite && minValue.isFinite && maxValue.isFinite,
        ]
    }

    private static func isoTimestamp(_ date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}

struct ProbeOptions {
    static let defaultTurns = [
        "a small red cube on a white table",
        "the same cube with blue lighting",
        "the same scene as a pencil sketch",
    ]

    var root = MLXStudioModelStore.defaultImageRoot
    let startedAt = Date()
    var artifactDirectory = URL(fileURLWithPath: "docs/local/vmlx-flux-probes")
        .appendingPathComponent(Self.timestamp())
    var outputDirectory = URL(fileURLWithPath: "docs/local/vmlx-flux-outputs", isDirectory: true)
    var model: String?
    var matrix = false
    var load = false
    var generate = false
    var edit = false
    var qwenEditPrompt = false
    var qwenEditConditioning = false
    var qwenEditVision = false
    var qwenEditDenoise = false
    var json = false
    var width = 256
    var height = 256
    var widthExplicit = false
    var heightExplicit = false
    var steps = 1
    var seed: UInt64?
    var guidance: Float?
    var negativePrompt: String?
    var sourceImages: [URL] = []
    var sourceImage: URL? { sourceImages.first }
    var maskImage: URL?
    var strength: Float = 0.75
    var turns = Self.defaultTurns

    init(arguments: [String]) throws {
        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--root":
                root = URL(fileURLWithPath: try Self.value(after: arg, in: arguments, index: &index), isDirectory: true)
            case "--artifacts":
                artifactDirectory = URL(fileURLWithPath: try Self.value(after: arg, in: arguments, index: &index), isDirectory: true)
            case "--output-dir":
                outputDirectory = URL(fileURLWithPath: try Self.value(after: arg, in: arguments, index: &index), isDirectory: true)
            case "--model":
                model = try Self.value(after: arg, in: arguments, index: &index)
            case "--matrix", "--all":
                matrix = true
                load = true
                generate = true
            case "--load":
                load = true
            case "--generate":
                generate = true
                load = true
            case "--edit":
                edit = true
                load = true
            case "--qwen-edit-prompt":
                qwenEditPrompt = true
                load = true
            case "--qwen-edit-conditioning":
                qwenEditConditioning = true
                load = true
            case "--qwen-edit-vision":
                qwenEditVision = true
                load = true
            case "--qwen-edit-denoise":
                qwenEditDenoise = true
                load = true
            case "--no-generate":
                generate = false
            case "--json":
                json = true
            case "--width":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Int(value) else { throw ProbeError("invalid --width") }
                width = parsed
                widthExplicit = true
            case "--height":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Int(value) else { throw ProbeError("invalid --height") }
                height = parsed
                heightExplicit = true
            case "--steps":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Int(value) else { throw ProbeError("invalid --steps") }
                steps = parsed
            case "--seed":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = UInt64(value) else { throw ProbeError("invalid --seed") }
                seed = parsed
            case "--guidance":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Float(value) else { throw ProbeError("invalid --guidance") }
                guidance = parsed
            case "--negative":
                negativePrompt = try Self.value(after: arg, in: arguments, index: &index)
            case "--source-image":
                sourceImages.append(URL(fileURLWithPath: try Self.value(after: arg, in: arguments, index: &index)))
            case "--mask-image":
                maskImage = URL(fileURLWithPath: try Self.value(after: arg, in: arguments, index: &index))
            case "--strength":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Float(value) else { throw ProbeError("invalid --strength") }
                strength = parsed
            case "--turn":
                let turn = try Self.value(after: arg, in: arguments, index: &index)
                if turns == Self.defaultTurns {
                    turns = []
                }
                turns.append(turn)
            default:
                throw ProbeError("unknown argument \(arg)")
            }
            index += 1
        }
    }

    private static func value(after flag: String, in arguments: [String], index: inout Int) throws -> String {
        let next = index + 1
        guard next < arguments.count else {
            throw ProbeError("missing value after \(flag)")
        }
        index = next
        return arguments[next]
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: Date())
    }
}

struct ProbeError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}
