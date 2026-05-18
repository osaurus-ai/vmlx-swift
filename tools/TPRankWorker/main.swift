// Copyright © 2026 Jinho Jang. All rights reserved.
//
// TPRankWorker — minimal rank entry point for the size-2 tensor-parallel
// loopback proof. One process per rank; env vars (`MLX_RANK`,
// `MLX_WORLD_SIZE`, `MLX_JACCL_COORDINATOR`, optionally
// `MLX_IBV_DEVICES`) are read by `mlx_distributed_init` at startup.
// The launcher (`Tools/tp-launch.sh`) sets them before exec.
//
// Behavior per invocation:
//   1. `Group(strict: env["TP_STRICT"] != "0", backend: env["MLX_DIST_BACKEND"])`.
//      Default backend is `"ring"` (TCP) for first-pass validation; flip
//      to `"jaccl"` (RDMA) once ring proves the TP math correct.
//   2. Load the model bundle at `TP_MODEL_PATH`.
//   3. Apply `ShardingPlan.llama` if `world_size > 1` (no-op on size-1).
//   4. Run one prefill forward pass on a fixed token sequence
//      (`TP_PROMPT_TOKEN_IDS` env, comma-separated).
//   5. Materialize the logits, write them to `TP_OUTPUT_PATH` as a
//      raw `.f32` blob (shape header + row-major Float32 LE).
//   6. Exit 0.
//
// Bit-identity test: run once with `MLX_WORLD_SIZE=1` (single-rank
// baseline) producing `baseline.f32`, then run twice with
// `MLX_RANK=0/1 MLX_WORLD_SIZE=2` (size-2 TP) producing `tp_rank0.f32`.
// Compare baseline vs rank0 to within 1e-3 tolerance.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXDistributedJACCL
import MLXDistributedTP
import MLXHuggingFace
import MLXNN
@preconcurrency import VMLXTokenizers

@main
struct TPRankWorker {
    static func main() async {
        setvbuf(stdout, nil, _IONBF, 0)

        let env = ProcessInfo.processInfo.environment
        let rank = Int(env["MLX_RANK"] ?? "0") ?? 0
        let worldSize = Int(env["MLX_WORLD_SIZE"] ?? "1") ?? 1
        let backend = env["MLX_DIST_BACKEND"] ?? "ring"
        let strict = (env["TP_STRICT"] ?? "1") != "0"
        let modelPath = env["TP_MODEL_PATH"] ?? ""
        let outputPath = env["TP_OUTPUT_PATH"]
            ?? "/tmp/tp_rank\(rank).f32"
        let promptTokens: [Int] = (env["TP_PROMPT_TOKEN_IDS"] ?? "1,2,3,4,5,6,7,8")
            .split(separator: ",")
            .compactMap { Int($0) }
        let smokeMode = env["TP_SMOKE"] == "1"

        log("rank=\(rank) world_size=\(worldSize) backend=\(backend) strict=\(strict)")
        log("model_path=\(modelPath)")
        log("output_path=\(outputPath)")
        log("prompt_tokens=\(promptTokens)")
        log("smoke=\(smokeMode)")

        if smokeMode {
            // Minimal allSum smoke — no model, no sharding. If this
            // works, the issue is upstream of collectives. If this
            // hangs, the issue IS the collective itself.
            log("[smoke] init Group")
            let g = Group(strict: strict, backend: backend)
            log("[smoke] Group rank=\(g.rank) size=\(g.size)")
            let payload: [Float] = (0..<8).map { Float($0 + rank * 100) }
            let x = MLXArray(payload)
            log("[smoke] before allSum local=\(payload)")
            let summed = Collectives.allSum(x, group: g)
            MLX.eval(summed)
            let arr = summed.asArray(Float.self)
            log("[smoke] after  allSum sum=\(arr)")
            log("[smoke] DONE")
            exit(0)
        }

        guard !modelPath.isEmpty else {
            err("TP_MODEL_PATH not set")
            exit(2)
        }

        // 1. Init distributed group.
        log("initializing Group strict=\(strict) backend=\(backend)")
        if backend == "jaccl" {
            log("JACCL.isAvailable() = \(JACCL.isAvailable())")
        }
        let group = Group(strict: strict, backend: backend)
        log("Group rank=\(group.rank) size=\(group.size) isMultiRank=\(group.isMultiRank)")

        if worldSize > 1 && group.size != worldSize {
            err("declared MLX_WORLD_SIZE=\(worldSize) but mlx_distributed_init returned size=\(group.size)")
        }

        // 2. Load model bundle.
        let modelDir = URL(fileURLWithPath: modelPath)
        log("loading model from \(modelDir.path)")
        let context: ModelContext
        do {
            context = try await loadModel(
                from: modelDir, using: #huggingFaceTokenizerLoader())
        } catch {
            err("loadModel failed: \(error)")
            exit(3)
        }
        log("model loaded: \(type(of: context.model))")

        // 3. Apply sharding plan iff multi-rank.
        if group.isMultiRank {
            log("applying ShardingPlan.llama (world_size=\(group.size))")
            let replaced = ShardingPlan.llama.apply(to: context.model, group: group)
            log("sharding replaced \(replaced.count) Linears")
            if replaced.isEmpty {
                err("WARNING: zero Linears were replaced — directive keys may not match this model family")
            } else {
                let sample = Array(replaced.sorted().prefix(5))
                for path in sample {
                    log("  replaced: \(path)")
                }
            }
            MLX.eval(context.model)
        } else {
            log("world_size=1 — sharding plan is a no-op (baseline mode)")
        }

        // 4. Run one forward pass on the fixed token sequence.
        log("running forward pass on \(promptTokens.count) tokens")
        let tokens = MLXArray(promptTokens.map { Int32($0) })
            .reshaped([1, promptTokens.count])
        let logits = context.model(tokens, cache: nil)
        MLX.eval(logits)
        log("logits shape=\(logits.shape) dtype=\(logits.dtype)")

        // 5. Cast to fp32 and dump.
        let logitsFP32 = logits.asType(.float32)
        MLX.eval(logitsFP32)

        let shape = logitsFP32.shape
        let count = shape.reduce(1, *)
        let floats: [Float] = logitsFP32.asArray(Float.self)
        let data = floats.withUnsafeBytes { Data($0) }
        let expectedBytes = count * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            err("logits byte count mismatch: got \(data.count), expected \(expectedBytes)")
            exit(4)
        }

        // Write [ndim u32][dim u32 × ndim][count u64][raw f32 bytes].
        var header = Data()
        var ndim = UInt32(shape.count).littleEndian
        header.append(Data(bytes: &ndim, count: MemoryLayout<UInt32>.size))
        for d in shape {
            var dimVal = UInt32(d).littleEndian
            header.append(Data(bytes: &dimVal, count: MemoryLayout<UInt32>.size))
        }
        var n = UInt64(count).littleEndian
        header.append(Data(bytes: &n, count: MemoryLayout<UInt64>.size))

        let outURL = URL(fileURLWithPath: outputPath)
        do {
            try (header + data).write(to: outURL)
        } catch {
            err("failed to write \(outputPath): \(error)")
            exit(5)
        }
        log("wrote \(expectedBytes) bytes (\(count) Float32) to \(outputPath)")
        log("DONE rank=\(rank) — exiting 0")
    }
}

// MARK: - Logging helpers

@inline(__always) private func log(_ msg: String) {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    print("[TPRankWorker rank=\(rank)] \(msg)")
}

@inline(__always) private func err(_ msg: String) {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    FileHandle.standardError.write(Data(
        "[TPRankWorker rank=\(rank)] ERROR: \(msg)\n".utf8))
}
