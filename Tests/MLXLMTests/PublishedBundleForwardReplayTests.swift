import CryptoKit
import Dispatch
import Foundation
import MLX
import MLXNN
@preconcurrency import VMLXTokenizers
import XCTest

@testable import MLXHuggingFace
@testable import MLXLLM
@testable import MLXLMCommon

#if canImport(Metal)
    import Metal
#endif

/// Component equivalence/performance only. No sampling, generated answer,
/// speculation, prefix coordinator or generation-settings override is used.
final class PublishedBundleForwardReplayTests: XCTestCase {
    private enum Failure: Error { case invalid(String) }

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func fileSHA(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while let data = try file.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func json(_ url: URL) throws -> [String: Any] {
        guard
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any]
        else { throw Failure.invalid("Expected JSON object: \(url.path)") }
        return object
    }

    private func tokenSHA(_ ids: [Int]) -> String {
        var data = Data()
        for id in ids {
            var word = UInt32(id).littleEndian
            withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
        }
        return sha(data)
    }

    /// Force synchronous overload selection before entering the async test's
    /// context. All Metal work and receipt mutation stay in one serialized,
    /// nonsuspending scope; no Sendable escape or unsafe capture is required.
    private func withSynchronousGPU(_ body: () throws -> Void) rethrows {
        try MLXMetalTestLock.withLock {
            try MLX.Stream.withNewDefaultStream(device: .gpu, body)
        }
    }

    private func cacheIdentity(_ cache: [KVCache]) -> [[String: Any]] {
        cache.enumerated().map { index, layer in
            [
                "layer": index, "type": String(reflecting: type(of: layer)),
                "offset": layer.offset,
                "state": layer.state.map {
                    ["shape": $0.shape, "dtype": String(describing: $0.dtype)] as [String: Any]
                },
            ] as [String: Any]
        }
    }

    private func financeDocument() -> String {
        var text = """
            Review this fictional finance record for Meridian Workshop. Reconcile revenue,
            cash receipts, expenses, inventory and tax-basis differences using only this record.
            Identify arithmetic discrepancies and distinguish recorded facts from missing inputs.
            No jurisdiction or tax rate is supplied; do not invent one. The company starts with
            cash 250000 dollars, inventory cost 74000 dollars and accounts receivable 32000 dollars.

            """
        for month in 1 ... 96 {
            text += """
                Period \(month): invoices total \(81000 + month * 137) dollars, cash collected
                \(76000 + month * 119) dollars, cash operating expenses \(42000 + month * 71)
                dollars, inventory purchases \(14000 + month * 23) dollars, inventory cost sold
                \(13000 + month * 19) dollars. Recorded depreciation is 3100 dollars and tax-basis
                depreciation is 3900 dollars. The ledger records a customer refund of 675 dollars,
                a supplier credit of 430 dollars and loan principal repayment of 2100 dollars.
                Separate operating cash flow from revenue, principal payments from expenses,
                and temporary basis differences from permanent differences. Attach a source
                period to every figure and list any supporting schedule needed for a tax return.

                """
        }
        return text
    }

    private func verifyBundle(
        directory: URL, manifestURL: URL, fullWeights: Bool
    ) throws -> [String: Any] {
        let manifest = try json(manifestURL)
        guard let path = manifest["path"] as? String,
            URL(fileURLWithPath: path).resolvingSymlinksInPath()
                == directory.resolvingSymlinksInPath(),
            let revision = manifest["revision"] as? String, revision.count == 40,
            let files = manifest["files"] as? [[String: Any]], !files.isEmpty
        else { throw Failure.invalid("Manifest path/revision/files do not identify this bundle") }
        var checked = [[String: Any]]()
        for entry in files {
            guard let name = entry["file"] as? String, !name.hasPrefix("/"),
                !name.split(separator: "/").contains(".."), let bytes = entry["bytes"] as? NSNumber,
                let expected = entry["sha256"] as? String, expected.count == 64
            else { throw Failure.invalid("Invalid staged-bundle file entry") }
            let file = directory.appendingPathComponent(name)
            let attributes = try FileManager.default.attributesOfItem(
                atPath: file.resolvingSymlinksInPath().path)
            guard (attributes[.size] as? NSNumber)?.uint64Value == bytes.uint64Value else {
                throw Failure.invalid("Bundle file size changed: \(name)")
            }
            let isWeight = name.hasSuffix(".safetensors")
            let freshlyHashed = !isWeight || fullWeights
            if freshlyHashed, try fileSHA(file) != expected {
                throw Failure.invalid("Bundle SHA changed: \(name)")
            }
            checked.append([
                "file": name, "bytes": bytes, "expected_sha256": expected,
                "freshly_hashed": freshlyHashed,
                "resolved_path": file.resolvingSymlinksInPath().path,
            ])
        }
        guard files.contains(where: { $0["file"] as? String == "config.json" }) else {
            throw Failure.invalid("Manifest must identify config.json")
        }
        return [
            "manifest_path": manifestURL.path, "manifest_sha256": try fileSHA(manifestURL),
            "model": manifest["model"] ?? NSNull(), "revision": revision,
            "directory": directory.path, "files": checked,
            "weight_identity_scope": fullWeights
                ? "fresh full SHA256" : "external verified staged manifest; fresh size/path only",
        ]
    }

    private func verifyBuild(_ url: URL) throws -> [String: Any] {
        let binding = try json(url)
        guard let expected = binding["binary_sha256"] as? String,
            binding["compiled_source"] != nil,
            let binary = Bundle(for: PublishedBundleForwardReplayTests.self).executableURL,
            try fileSHA(binary) == expected
        else {
            throw Failure.invalid(
                "Build receipt must bind executing test binary and compiled_source")
        }
        // The supplied build-time snapshot is authoritative. Reading current
        // worktree files after compilation would not establish binary identity.
        return [
            "receipt_path": url.path, "receipt_sha256": try fileSHA(url),
            "binding": binding, "executing_binary": binary.path, "binary_sha256": expected,
        ]
    }

    private func quantizationIdentity(
        _ model: any LanguageModel, config: [String: Any]
    ) throws -> [[String: Any]] {
        let declared = config["quantization"] as? [String: Any] ?? [:]
        var records = [[String: Any]]()
        var found = Set<String>()
        for (name, module) in model.namedModules() {
            guard let quantized = module as? any Quantized else { continue }
            let expected = declared[name] as? [String: Any] ?? declared
            guard (expected["bits"] as? Int) == quantized.bits,
                (expected["group_size"] as? Int) == quantized.groupSize,
                quantized.mode == .affine
            else { throw Failure.invalid("Loaded quantization differs from bundle at \(name)") }
            found.insert(name)
            records.append([
                "path": name, "type": String(reflecting: type(of: module)),
                "bits": quantized.bits, "group_size": quantized.groupSize, "mode": "affine",
                "parameters": module.parameters().flattened().map {
                    ["name": $0.0, "shape": $0.1.shape, "dtype": String(describing: $0.1.dtype)]
                        as [String: Any]
                }.sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") },
            ])
        }
        for (name, item) in declared where item is [String: Any] {
            guard found.contains(name) else {
                throw Failure.invalid("Declared packed module not loaded: \(name)")
            }
        }
        guard !records.isEmpty else { throw Failure.invalid("No native quantized modules loaded") }
        return records.sorted { ($0["path"] as? String ?? "") < ($1["path"] as? String ?? "") }
    }

    func testOptInExactPublishedBundleForwardReplay() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VMLX_BUNDLE_FORWARD_REPLAY"] == "1" else {
            throw XCTSkip("Set VMLX_BUNDLE_FORWARD_REPLAY=1 for published-bundle component replay")
        }
        func required(_ key: String) throws -> String {
            guard let value = env[key], !value.isEmpty else {
                throw Failure.invalid("Missing \(key)")
            }
            return value
        }
        let directory = URL(fileURLWithPath: try required("VMLX_BUNDLE_REPLAY_MODEL_PATH"))
        let manifestURL = URL(fileURLWithPath: try required("VMLX_BUNDLE_REPLAY_MANIFEST"))
        let buildURL = URL(fileURLWithPath: try required("VMLX_BUNDLE_REPLAY_BUILD_RECEIPT"))
        let output = URL(fileURLWithPath: try required("VMLX_BUNDLE_REPLAY_OUTPUT"))
        let arm = try required("VMLX_BUNDLE_REPLAY_ARM")
        guard ["reference", "candidate"].contains(arm),
            let round = Int(try required("VMLX_BUNDLE_REPLAY_ROUND")), (1 ... 3).contains(round),
            !FileManager.default.fileExists(atPath: output.path)
        else { throw Failure.invalid("Arm/round invalid or output already exists") }
        var receipt: [String: Any] = [
            "schema": 1, "arm": arm, "round": round, "status": "running",
            "started_utc": ISO8601DateFormatter().string(from: Date()),
            "host": ProcessInfo.processInfo.hostName,
            "scope":
                "empty-native-KV prefill and teacher-forced decode; no answer generation or quality proof",
            "cache_scope": "newCache(parameters:nil), no prefix/paged/disk coordinator",
            "cold_scope":
                "empty KV with loaded weights; not cold model load, pristine JIT or SSD miss",
            "timer_scope":
                "input graph already prepared; forward+eval of logits/cache+scoped GPU completion",
            "hash_scope": "all raw contiguous logit bytes, native dtype, outside timer",
            "physical_footprint":
                "external monitor required; allocator statistics are not physical footprint",
        ]
        var cases = [[String: Any]]()
        defer {
            receipt["cases"] = cases
            receipt["finished_utc"] = ISO8601DateFormatter().string(from: Date())
            do {
                try FileManager.default.createDirectory(
                    at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                let data = try JSONSerialization.data(
                    withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: output, options: .withoutOverwriting)
                print("PUBLISHED_FORWARD_REPLAY_RECEIPT \(output.path)")
            } catch { XCTFail("Could not preserve replay receipt: \(error)") }
        }
        do {
            receipt["build"] = try verifyBuild(buildURL)
            receipt["bundle"] = try verifyBundle(
                directory: directory, manifestURL: manifestURL,
                fullWeights: env["VMLX_BUNDLE_REPLAY_VERIFY_WEIGHTS"] == "1")
            let config = try json(directory.appendingPathComponent("config.json"))
            guard let family = config["model_type"] as? String,
                ["k2_horizon", "spark2_5"].contains(family)
            else { throw Failure.invalid("Only exact K2/Spark published bundles supported") }
            let switchName =
                family == "k2_horizon"
                ? "VMLX_K2_GROUPED_NORM_REFERENCE" : "VMLX_SPARK_MLP_INPUT_CAST_REFERENCE"
            guard (env[switchName] == "1") == (arm == "reference") else {
                throw Failure.invalid("Separate process arm must agree with \(switchName)")
            }
            receipt["family"] = family
            receipt["reference_switch"] = ["name": switchName, "value": env[switchName] ?? "unset"]
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let context = try await MLXLMCommon.loadModel(
                from: directory, using: #huggingFaceTokenizerLoader())
            nonisolated(unsafe) let ctx = context
            MLX.Stream().synchronize()
            receipt["load_elapsed_ns"] = DispatchTime.now().uptimeNanoseconds - loadStart
            guard
                (family == "k2_horizon" && ctx.model is K2HorizonModel)
                    || (family == "spark2_5" && ctx.model is Spark25Model)
            else { throw Failure.invalid("Production factory selected unexpected model") }
            receipt["loaded_type"] = String(reflecting: type(of: ctx.model))
            receipt["quantization"] = try quantizationIdentity(ctx.model, config: config)
            let document = financeDocument()
            let prompt = try ctx.tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": document]], tools: nil,
                additionalContext: nil)
            let continuation =
                "The ledger distinguishes revenue from cash receipts. Principal repayment reduces the loan balance rather than operating profit. Inventory purchases and cost sold must be reconciled before any tax adjustment. Missing jurisdiction and tax-rate inputs remain unresolved."
            let continuationIDs = Array(
                ctx.tokenizer.encode(text: continuation, addSpecialTokens: false).prefix(32))
            guard prompt.count >= 2048, continuationIDs.count == 32,
                prompt.allSatisfy({ $0 >= 0 }), continuationIDs.allSatisfy({ $0 >= 0 })
            else { throw Failure.invalid("Tokenizer did not supply required fixed replay lengths") }
            receipt["input"] = [
                "document": document, "document_sha256": sha(Data(document.utf8)),
                "native_chat_template_kwargs": "none; bundle template defaults",
                "full_chat_tokens": prompt.count, "full_chat_token_sha256_le_u32": tokenSHA(prompt),
                "case_policy":
                    "first128/512/2048 tokens of native rendered finance document; slices may end inside document, not complete conversations",
                "teacher_text": continuation, "teacher_ids": continuationIDs,
                "teacher_ids_sha256_le_u32": tokenSHA(continuationIDs),
            ]
            try self.withSynchronousGPU {
                let stream = MLX.Stream()
                let warmPrefill = env["VMLX_BUNDLE_REPLAY_WARM_PREFILL"] == "1"
                receipt["prefill_timing_protocol"] = warmPrefill
                    ? "two untimed same-shape forwards with separate fresh KV caches; timed KV remains fresh"
                    : "first same-shape forward in this process; loaded weights, no shader warmup"
                for length in [128, 512, 2048] {
                    let ids = Array(prompt.prefix(length))
                    if warmPrefill {
                        for _ in 0 ..< 2 {
                            let warmCache = ctx.model.newCache(parameters: nil)
                            let warmInput = MLXArray(ids.map(Int32.init), [1, ids.count])
                            let warmLogits = ctx.model(warmInput, cache: warmCache)
                            eval([warmLogits] + warmCache.flatMap { $0.state })
                            stream.synchronize()
                        }
                    }
                    let cache = ctx.model.newCache(parameters: nil)
                    guard
                        cache.allSatisfy({
                            $0.offset == 0 && ($0 is KVCacheSimple || $0 is RotatingKVCache)
                        })
                    else {
                        throw Failure.invalid("Replay requires fresh native KV caches")
                    }
                    var row: [String: Any] = [
                        "prefill_tokens": length, "prompt_ids": ids,
                        "prompt_ids_sha256_le_u32": self.tokenSHA(ids),
                        "initial_cache": self.cacheIdentity(cache),
                        "steps": [[String: Any]](),
                    ]
                    cases.append(row)
                    var steps = [[String: Any]]()
                    func forward(_ inputIDs: [Int], phase: String, index: Int) throws {
                        let input = MLXArray(inputIDs.map(Int32.init), [1, inputIDs.count])
                        eval(input)
                        stream.synchronize()
                        let start = DispatchTime.now().uptimeNanoseconds
                        let logits = ctx.model(input, cache: cache)
                        eval([logits] + cache.flatMap { $0.state })
                        stream.synchronize()
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        // Copy/hash outside the forward timer. No cast,
                        // top-k selection or last-row truncation is used.
                        let raw = logits.asData(access: .copy)
                        steps.append([
                            "phase": phase, "index": index, "input_ids": inputIDs,
                            "elapsed_ns": elapsed, "shape": raw.shape,
                            "dtype": String(describing: raw.dType), "bytes": raw.data.count,
                            "logits_sha256": self.sha(raw.data),
                            "cache_offsets": cache.map(\.offset),
                        ])
                        row["steps"] = steps
                        cases[cases.count - 1] = row
                    }
                    try forward(ids, phase: "empty_kv_prefill", index: 0)
                    for (index, id) in continuationIDs.enumerated() {
                        try forward([id], phase: "teacher_forced_decode", index: index)
                    }
                    row["final_cache"] = self.cacheIdentity(cache)
                    row["status"] = "completed; cross-process exact comparison pending"
                    cases[cases.count - 1] = row
                }
                let probeInput = MLXArray([Int32(prompt[0])], [1, 1])
                if let spark = ctx.model as? Spark25Model, let layer = spark.model.layers.first {
                    let x = spark.model.embedding(probeInput).asType(.float32)
                    let normalized = layer.inputNorm(x).asType(layer.inputNorm.weight.dtype)
                    let h = x + layer.attention(normalized, mask: .causal, cache: nil)
                    let post = layer.postNorm(h).asType(layer.postNorm.weight.dtype)
                    let output = layer(x, mask: .causal, cache: nil)
                    eval(x, normalized, h, post, output)
                    stream.synchronize()
                    guard x.dtype == .float32, h.dtype == .float32, output.dtype == .float32,
                        normalized.dtype == layer.inputNorm.weight.dtype,
                        post.dtype == layer.postNorm.weight.dtype
                    else {
                        throw Failure.invalid(
                            "Loaded Spark first-layer residual/cast contract changed")
                    }
                    receipt["loaded_first_layer_probe"] = [
                        "scope":
                            "off-timer loaded first-layer arithmetic probe; no whole-model dispatch count",
                        "input_dtype": String(describing: x.dtype),
                        "attention_residual_dtype": String(describing: h.dtype),
                        "output_residual_dtype": String(describing: output.dtype),
                        "projection_input_dtype": String(describing: normalized.dtype),
                        "post_projection_input_dtype": String(describing: post.dtype),
                        "output_sha256": self.sha(output.asData(access: .copy).data),
                    ]
                }
                if let path = env["VMLX_BUNDLE_REPLAY_CAPTURE"] {
                    do {
                        receipt["capture"] = try self.captureDecode(
                            path: URL(fileURLWithPath: path), model: ctx.model,
                            prompt: Array(prompt.prefix(128)), teacherToken: continuationIDs[0],
                            stream: stream)
                    } catch {
                        receipt["capture"] = [
                            "status": "failed", "error": String(describing: error),
                            "scope": "optional dispatch proof remains missing",
                        ]
                    }
                }
            }
            receipt["status"] = "completed; external exact-logit comparison required"
        } catch {
            receipt["status"] = "failed"
            receipt["failure"] = String(describing: error)
            throw error
        }
    }

    private func captureDecode(
        path: URL, model: any LanguageModel, prompt: [Int], teacherToken: Int, stream: MLX.Stream
    ) throws -> [String: Any] {
        #if canImport(Metal)
            guard !FileManager.default.fileExists(atPath: path.path) else {
                throw Failure.invalid("Refusing to replace existing capture")
            }
            let manager = MTLCaptureManager.shared()
            guard !manager.isCapturing, manager.supportsDestination(.gpuTraceDocument),
                let device = MTLCreateSystemDefaultDevice()
            else { return ["status": "unavailable", "scope": "no actual dispatch proof"] }
            let cache = model.newCache(parameters: nil)
            eval(model(MLXArray(prompt.map(Int32.init), [1, prompt.count]), cache: cache))
            stream.synchronize()
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = path
            try manager.startCapture(with: descriptor)
            defer { manager.stopCapture() }
            eval(model(MLXArray([Int32(teacherToken)], [1, 1]), cache: cache))
            stream.synchronize()
            return [
                "status": "captured; trace inspection pending", "path": path.path,
                "scope":
                    "one ordinary loaded-model teacher-forced decode after128-token native-KV prefill; outside all timers",
            ]
        #else
            return ["status": "unavailable", "scope": "Metal unavailable"]
        #endif
    }
}
