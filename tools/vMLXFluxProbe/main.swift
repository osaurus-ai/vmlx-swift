import Foundation
import CryptoKit
import ImageIO
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
            "turns": options.turns,
            "width": options.width,
            "height": options.height,
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
                        guidance: options.guidance ?? (loaded.canonicalName == "z-image-turbo" ? 0 : 3.5),
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
        let nativeStatus = runtimeStatus(for: local.canonicalName)
        let loadStatus = payload["load_status"] as? String ?? "not_requested"
        let gateStatus: String
        var gateReasons = runtimeBlockers(for: local.canonicalName)
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
            "native_runtime_status": runtimeStatus(for: model.canonicalName),
            "native_runtime_blockers": runtimeBlockers(for: model.canonicalName),
        ]
    }

    private static func runtimeStatus(for canonicalName: String?) -> String {
        switch canonicalName {
        case "z-image-turbo":
            return "native_pipeline_implemented"
        case "flux1-schnell", "flux1-dev", "flux1-kontext", "flux1-fill",
             "flux2-klein", "flux2-klein-edit", "qwen-image", "qwen-image-edit",
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

    private static func runtimeBlockers(for canonicalName: String?) -> [String] {
        switch canonicalName {
        case "z-image-turbo":
            return [
                "requires live same-seed prompt-sensitivity and multi-turn matrix before production promotion",
            ]
        case "flux1-schnell", "flux1-dev", "flux1-kontext", "flux1-fill",
             "flux2-klein", "flux2-klein-edit", "qwen-image", "qwen-image-edit",
             "fibo", "seedvr2":
            return [
                "model generate/edit/upscale body throws FluxError.notImplemented",
                "text encoder ports are missing",
                "safetensors-to-module key mapping is missing",
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
    var json = false
    var width = 256
    var height = 256
    var steps = 1
    var seed: UInt64?
    var guidance: Float?
    var negativePrompt: String?
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
            case "--no-generate":
                generate = false
            case "--json":
                json = true
            case "--width":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Int(value) else { throw ProbeError("invalid --width") }
                width = parsed
            case "--height":
                let value = try Self.value(after: arg, in: arguments, index: &index)
                guard let parsed = Int(value) else { throw ProbeError("invalid --height") }
                height = parsed
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
