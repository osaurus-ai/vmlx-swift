// DeepSeek-V4.1 Engram: per-token n-gram-hash gated residual applied at engram_layer_ids (1, 14).
//
// This is NOT a long-context feature -- the reference adds `h = layer.engram(h, rows, mask)` to the
// FULL hidden state at those layers for EVERY token, so without it the model emits garbage even on a
// short prompt. Port of jang_tools/dsv41/{engram.py,model.py} (the working vmlx-py path).
//
// Split of responsibilities (mirrors the Python):
//   * EngramAux         -- the hashing tables exported by export_engram_aux.py (token_map /
//                          multipliers / primes / offsets + scalars), so Swift does not reimplement
//                          HF-tokenizer normalization. Loaded from `engram_aux.safetensors`.
//   * EngramNgramHasher -- pure, stateless CPU Int64 n-gram hashing (exactly numpy's arithmetic).
//                          The per-sequence compressed-token history is NOT kept here: it travels
//                          with the sequence's KV cache (`HybridPoolCache.auxiliaryTokenHistory`),
//                          the Swift analog of the reference `Session.history`, so prefix-cache
//                          restores, copies, trims and checkpoints hash exactly like an
//                          uninterrupted run.
//   * EngramTable       -- one engram layer's FP8 table, read in place from the bundle's engram/
//                          shard with concurrent `pread` (never mapped, never wired; the OS page
//                          cache may keep hot n-grams).
//   * EngramModule      -- the wkv (affine-quant, in the BACKBONE) + q/k_weight gated residual.
//   * EngramRuntime     -- immutable aux + tables, resolved from the bundle directory at load.
//
// The wkv/q_weight/k_weight module weights live in the backbone shards (affine quant) and load through
// the normal path; only the big embed table is disk-resident here.

import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Errors

enum EngramError: Error, CustomStringConvertible {
    case missingFile(URL)
    case malformed(URL, String)
    case missingTensor(String)
    case mismatch(String)
    case openFailed(URL, Int32)

    var description: String {
        switch self {
        case .missingFile(let url):
            return "DeepSeek-V4.1 engram: required file not found: \(url.path)"
        case .malformed(let url, let why):
            return "DeepSeek-V4.1 engram: malformed \(url.lastPathComponent): \(why)"
        case .missingTensor(let key):
            return "DeepSeek-V4.1 engram: tensor \(key) not found in the bundle's engram shards"
        case .mismatch(let why):
            return "DeepSeek-V4.1 engram: aux tables do not match config.json: \(why)"
        case .openFailed(let url, let err):
            return "DeepSeek-V4.1 engram: cannot open \(url.path): errno \(err)"
        }
    }
}

// MARK: - Minimal safetensors header reader

enum EngramSafetensors {
    struct Entry {
        let dtype: String
        let shape: [Int]
        /// Absolute byte offsets in the file.
        let start: Int
        let end: Int
    }

    static func header(_ url: URL) throws -> (entries: [String: Entry], metadata: [String: String]) {
        guard let fh = try? FileHandle(forReadingFrom: url) else { throw EngramError.missingFile(url) }
        defer { try? fh.close() }
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw EngramError.malformed(url, "short header length")
        }
        let hlen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard hlen > 0, hlen < 512 * 1024 * 1024,
            let hdrData = try fh.read(upToCount: Int(hlen)), hdrData.count == Int(hlen),
            let obj = try JSONSerialization.jsonObject(with: hdrData) as? [String: Any]
        else { throw EngramError.malformed(url, "unreadable JSON header") }
        let base = 8 + Int(hlen)
        var entries: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        for (key, raw) in obj {
            if key == "__metadata__" {
                metadata = (raw as? [String: String]) ?? [:]
                continue
            }
            guard let t = raw as? [String: Any],
                let dtype = t["dtype"] as? String,
                let shape = (t["shape"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                let off = (t["data_offsets"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                off.count == 2
            else { throw EngramError.malformed(url, "bad entry \(key)") }
            entries[key] = Entry(dtype: dtype, shape: shape, start: base + off[0], end: base + off[1])
        }
        return (entries, metadata)
    }
}

// MARK: - Aux tables (exported sidecar)

/// Parsed `engram_aux.safetensors`: the n-gram hashing constants + scalar layout.
struct EngramAux {
    let tokenMap: [Int32]          // [vocab] -> compressed token id
    let multipliers: [[Int64]]     // [nLayers][maxNgram]
    let primes: [[[Int64]]]        // [nLayers][maxNgram-1][nHeads]
    let offsets: [[Int64]]         // [nLayers][cols]
    let padId: Int64
    let nHeads: Int
    let headDim: Int
    let maxNgram: Int
    let layerIds: [Int]
    let compressedVocab: Int

    var cols: Int { (maxNgram - 1) * nHeads }

    static func load(_ url: URL) throws -> EngramAux {
        let (entries, meta) = try EngramSafetensors.header(url)
        let data = try Data(contentsOf: url, options: .mappedIfSafe)

        func raw(_ name: String, _ dtype: String) throws -> (bytes: Data, shape: [Int]) {
            guard let e = entries[name] else { throw EngramError.malformed(url, "missing \(name)") }
            guard e.dtype == dtype, e.end <= data.count, e.start <= e.end else {
                throw EngramError.malformed(url, "\(name) is \(e.dtype), expected \(dtype)")
            }
            return (data.subdata(in: e.start ..< e.end), e.shape)
        }
        func i32(_ d: Data) -> [Int32] {
            d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        func i64(_ d: Data) -> [Int64] {
            d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
        }

        let tm = try raw("token_map", "I32")
        let mu = try raw("multipliers", "I64")
        let pr = try raw("primes", "I64")
        let of = try raw("offsets", "I64")
        let pad = try raw("pad_id", "I64")
        guard mu.shape.count == 2, pr.shape.count == 3, of.shape.count == 2,
            pr.shape[0] == mu.shape[0], of.shape[0] == mu.shape[0],
            pr.shape[1] == mu.shape[1] - 1, of.shape[1] == pr.shape[1] * pr.shape[2]
        else { throw EngramError.malformed(url, "inconsistent table shapes") }

        let tokenMap = i32(tm.bytes)
        let muFlat = i64(mu.bytes), prFlat = i64(pr.bytes), ofFlat = i64(of.bytes)
        let nLayers = mu.shape[0], maxNgram = mu.shape[1]
        let ngM1 = pr.shape[1], nHeads = pr.shape[2], cols = of.shape[1]

        let multipliers = (0..<nLayers).map { Array(muFlat[($0 * maxNgram)..<(($0 + 1) * maxNgram)]) }
        let offsets = (0..<nLayers).map { Array(ofFlat[($0 * cols)..<(($0 + 1) * cols)]) }
        let primes = (0..<nLayers).map { l in
            (0..<ngM1).map { g -> [Int64] in
                let start = (l * ngM1 + g) * nHeads
                return Array(prFlat[start..<(start + nHeads)])
            }
        }
        guard let headDim = meta["head_dim"].flatMap({ Int($0) }),
            let layerIdsRaw = meta["layer_ids"],
            let compressedVocab = meta["compressed_vocab_size"].flatMap({ Int($0) })
        else { throw EngramError.malformed(url, "missing __metadata__ fields") }
        let layerIds = layerIdsRaw.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        guard layerIds.count == nLayers, let padId = i64(pad.bytes).first else {
            throw EngramError.malformed(url, "layer_ids / pad_id")
        }
        return EngramAux(
            tokenMap: tokenMap, multipliers: multipliers, primes: primes, offsets: offsets,
            padId: padId, nHeads: nHeads, headDim: headDim, maxNgram: maxNgram,
            layerIds: layerIds, compressedVocab: compressedVocab)
    }
}

// MARK: - N-gram hashing (CPU, stateless)

/// Mirrors `NgramHasher.hash` (engram.py) exactly, but stateless: the caller supplies the
/// compressed tokens immediately preceding `startPos` (at most `maxNgram - 1` are used) and keeps
/// the returned compressed tokens as the sequence's history.
struct EngramNgramHasher {
    static let DEAD: Int64 = -1
    let aux: EngramAux

    init(_ aux: EngramAux) { self.aux = aux }

    /// Number of preceding compressed tokens an n-gram can reach.
    var historyReach: Int { aux.maxNgram - 1 }

    /// Compressed ids for `inputIds` (image tokens -> DEAD when `keep[i] == false`).
    func compress(_ inputIds: [Int32], keep: [Bool]? = nil) -> [Int64] {
        let vocab = aux.tokenMap.count
        return inputIds.enumerated().map { i, raw in
            if keep?[i] == false { return Self.DEAD }
            let id = Int(raw)
            precondition(
                id >= 0 && id < vocab,
                "DeepSeek-V4.1 engram: token id \(id) outside the aux token_map (\(vocab))")
            return Int64(aux.tokenMap[id])
        }
    }

    /// Hash ids, laid out `[L][nLayers][cols]` flattened.
    ///
    /// - Parameters:
    ///   - comp: compressed ids of the chunk (positions `startPos ..< startPos + L`).
    ///   - startPos: absolute position of `comp[0]`.
    ///   - previous: compressed ids at positions `startPos - previous.count ..< startPos`. Must
    ///     hold `min(historyReach, startPos)` entries for an exact result; the runtime guarantees
    ///     that from the cache-carried history.
    func hashes(_ comp: [Int64], startPos: Int, previous: [Int64]) -> [Int64] {
        let L = comp.count
        let n = aux.maxNgram
        let nLayers = aux.multipliers.count
        let cols = aux.cols
        let nHeads = aux.nHeads
        let histStart = startPos - previous.count
        // history[pos] for pos in [histStart, startPos + L)
        let history = previous + comp
        @inline(__always) func at(_ pos: Int) -> Int64 {
            let clipped = max(pos, 0)
            let idx = clipped - histStart
            // Only reachable when `previous` is shorter than required (unknown history).
            return (idx >= 0 && idx < history.count) ? history[idx] : aux.padId
        }

        var toks = [Int64](repeating: 0, count: n)
        var out = [Int64](repeating: 0, count: L * nLayers * cols)
        for i in 0..<L {
            let pos = startPos + i
            var blocked = false
            for shift in 0..<n {
                let src = at(pos - shift)
                if pos < shift || src == Self.DEAD { blocked = true }
                toks[shift] = blocked ? aux.padId : src
            }
            for l in 0..<nLayers {
                let mult = aux.multipliers[l]
                var rolling = toks[0] &* mult[0]
                var col = 0
                let base = (i * nLayers + l) * cols
                for k in 1..<n {
                    rolling ^= toks[k] &* mult[k]
                    let primesK = aux.primes[l][k - 1]
                    for hd in 0..<nHeads {
                        var m = rolling % primesK[hd]  // numpy % is floor-mod for a positive divisor
                        if m < 0 { m += primesK[hd] }
                        out[base + col] = m + aux.offsets[l][col]
                        col += 1
                    }
                }
            }
        }
        return out
    }
}

// MARK: - SSD-resident FP8 embedding table (concurrent pread)

/// One engram layer's FP8 table, read in place from its safetensors shard.
///
/// The tables are ~189 GB; mapping them would page-fault one row at a time and make throughput
/// hostage to fault servicing. Rows are therefore read with concurrent `pread` (64 in flight, 256-row
/// tasks — measured ~130k cold rows/s vs ~8-11k serial in the reference), row bytes and scale bytes in
/// the same task. Nothing is wired or resident beyond the OS page cache, which is allowed to keep hot
/// n-grams (no `F_NOCACHE`). Row r = E4M3 bytes [dim], scaled per block of `block` by E8M0 bytes [sdim].
final class EngramTable: @unchecked Sendable {
    private let wFd: Int32
    private let sFd: Int32
    private let ownsScaleFd: Bool
    private let wOff: Int
    private let sOff: Int
    let rowCount: Int
    let dim: Int                     // 256
    let sdim: Int                    // 8
    let block: Int                   // dim/sdim = 32

    /// Concurrent read lanes and rows per task (engram.py `_IO_POOL` / `_IO_TASK`).
    static let ioLanes = 64
    static let ioTaskRows = 256
    private static let ioQueue = DispatchQueue(
        label: "ai.vmlx.dsv41.engram-io", qos: .userInitiated, attributes: .concurrent)
    private static let readFailures = ManagedAtomicCounter()

    init(weightFile: URL, weightOffset: Int, weightShape: [Int],
         scaleFile: URL, scaleOffset: Int, scaleShape: [Int]) throws {
        func open(_ url: URL) throws -> Int32 {
            let fd = Darwin.open(url.path, O_RDONLY)
            guard fd >= 0 else { throw EngramError.openFailed(url, errno) }
            return fd
        }
        guard weightShape.count == 2, scaleShape.count == 2, weightShape[0] == scaleShape[0],
            scaleShape[1] > 0, weightShape[1] % scaleShape[1] == 0
        else { throw EngramError.malformed(weightFile, "engram table shapes \(weightShape) / \(scaleShape)") }
        self.wFd = try open(weightFile)
        if scaleFile == weightFile {
            self.sFd = self.wFd
            self.ownsScaleFd = false
        } else {
            do {
                self.sFd = try open(scaleFile)
            } catch {
                Darwin.close(self.wFd)
                throw error
            }
            self.ownsScaleFd = true
        }
        self.wOff = weightOffset
        self.sOff = scaleOffset
        self.rowCount = weightShape[0]
        self.dim = weightShape[1]
        self.sdim = scaleShape[1]
        self.block = weightShape[1] / scaleShape[1]
    }

    deinit {
        if ownsScaleFd { Darwin.close(sFd) }
        Darwin.close(wFd)
    }

    /// pread exactly `count` bytes (handles short reads). Returns false on EOF/error.
    private static func preadExactly(
        _ fd: Int32, _ dst: UnsafeMutableRawPointer, _ count: Int, _ offset: Int
    ) -> Bool {
        var done = 0
        while done < count {
            let n = Darwin.pread(fd, dst.advanced(by: done), count - done, off_t(offset + done))
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return false }
            done += n
        }
        return true
    }

    /// Read rows `[lo, hi)` of `uniq` into the shared buffers.
    private func readRange(
        _ uniq: UnsafeBufferPointer<Int64>, _ lo: Int, _ hi: Int,
        _ rbase: UnsafeMutableRawPointer, _ sbase: UnsafeMutableRawPointer
    ) {
        for j in lo..<hi {
            let r = Int(uniq[j])
            precondition(r >= 0 && r < rowCount, "DeepSeek-V4.1 engram row \(r) out of range")
            let okW = Self.preadExactly(wFd, rbase.advanced(by: j * dim), dim, wOff + r * dim)
            let okS = Self.preadExactly(sFd, sbase.advanced(by: j * sdim), sdim, sOff + r * sdim)
            if !(okW && okS) && Self.readFailures.increment() == 1 {
                NSLog("VMLX DSV4.1 engram: short read on table row \(r) (errno \(errno)); "
                    + "row left zeroed. Check the bundle's engram/ shards.")
            }
        }
    }

    /// Raw E4M3 row bytes [U*dim] and E8M0 scale bytes [U*sdim] for `uniq`, read with up to
    /// `lanes` concurrent tasks of at most `taskRows` rows. `lanes == 1` is the serial reference.
    func readRaw(_ uniq: [Int64], lanes: Int = EngramTable.ioLanes,
                 taskRows: Int = EngramTable.ioTaskRows) -> (rows: [UInt8], scales: [UInt8]) {
        let U = uniq.count
        var rowBytes = [UInt8](repeating: 0, count: U * dim)
        var scaleBytes = [UInt8](repeating: 0, count: U * sdim)
        guard U > 0 else { return (rowBytes, scaleBytes) }
        // Small batches (decode) still fan out so their few cold reads overlap.
        let chunk = max(1, min(max(1, taskRows), (U + max(1, lanes) - 1) / max(1, lanes)))
        let tasks = (U + chunk - 1) / chunk
        uniq.withUnsafeBufferPointer { ub in
            rowBytes.withUnsafeMutableBytes { rdst in
                scaleBytes.withUnsafeMutableBytes { sdst in
                    let rbase = rdst.baseAddress!
                    let sbase = sdst.baseAddress!
                    if tasks == 1 || lanes <= 1 {
                        readRange(ub, 0, U, rbase, sbase)
                        return
                    }
                    // Bounded fan-out: `lanes` workers pull task indices; each task is one chunk.
                    let group = DispatchGroup()
                    let next = ManagedAtomicCounter()
                    let workers = min(lanes, tasks)
                    for _ in 0..<workers {
                        Self.ioQueue.async(group: group) {
                            while true {
                                let t = next.increment() - 1
                                if t >= tasks { break }
                                let lo = t * chunk
                                self.readRange(ub, lo, min(lo + chunk, U), rbase, sbase)
                            }
                        }
                    }
                    group.wait()
                }
            }
        }
        return (rowBytes, scaleBytes)
    }

    /// Dedup `ids` (first-occurrence order) -> (unique ids, inverse index per id).
    static func dedup(_ ids: [Int64]) -> (uniq: [Int64], inv: [Int32]) {
        var uniqIndex = [Int64: Int32](minimumCapacity: ids.count)
        var uniq: [Int64] = []
        uniq.reserveCapacity(ids.count)
        var inv = [Int32](repeating: 0, count: ids.count)
        for (i, id) in ids.enumerated() {
            if let u = uniqIndex[id] {
                inv[i] = u
            } else {
                let u = Int32(uniq.count)
                uniqIndex[id] = u
                uniq.append(id)
                inv[i] = u
            }
        }
        return (uniq, inv)
    }

    /// Host-side result of one lookup, ready to become an MLX tensor on the forward thread.
    struct RawLookup {
        let uniqCount: Int
        let rows: [UInt8]
        let scales: [UInt8]
        let inv: [Int32]
    }

    func readLookup(_ ids: [Int64]) -> RawLookup {
        let (uniq, inv) = Self.dedup(ids)
        let (rows, scales) = readRaw(uniq)
        return RawLookup(uniqCount: uniq.count, rows: rows, scales: scales, inv: inv)
    }

    /// bf16 rows `[ids.count, dim]` from a host lookup (dequant math unchanged).
    func materialize(_ raw: RawLookup) -> MLXArray {
        let U = raw.uniqCount
        let r = MLXArray(raw.rows, [U, dim])                  // uint8 E4M3
        let s = MLXArray(raw.scales, [U, sdim])               // uint8 E8M0
        let vals = fromFP8(r, dtype: .float32).reshaped(U, sdim, block)
            * EngramMath.e8m0ToFloat(s).expandedDimensions(axis: -1)
        let dequant = vals.reshaped(U, dim).asType(.bfloat16)
        return dequant[MLXArray(raw.inv)]
    }

    /// ids [...] (flattened) -> bf16 [count, dim]. Duplicate ids are read once.
    func lookup(_ ids: [Int64]) -> MLXArray {
        materialize(readLookup(ids))
    }
}

/// Tiny lock-based counter (no swift-atomics dependency in MLXLLM).
final class ManagedAtomicCounter: @unchecked Sendable {
    private var value = 0
    private let lock = NSLock()
    /// Increment and return the NEW value.
    func increment() -> Int {
        lock.lock()
        defer { lock.unlock() }
        value += 1
        return value
    }
}

enum EngramMath {
    /// UE8M0 byte -> 2**(byte-127), built from the float32 exponent bits (byte 0 is the 2**-127 subnormal).
    static func e8m0ToFloat(_ s: MLXArray) -> MLXArray {
        let b = s.asType(.uint32)
        let bits = MLX.where(b .== MLXArray(UInt32(0)),
                             MLXArray(UInt32(0x0040_0000)), b << 23)
        return bits.view(dtype: .float32)
    }
}

// MARK: - Engram module (gated residual)

/// `h = h + gate * value`, gate from the alignment of h with a key derived from the looked-up n-gram
/// rows through `wkv`. Mirrors Python `Engram.__call__` (model.py). q_weight/k_weight are learned
/// BF16 params in the backbone (NOT ones); wkv is a backbone affine QuantizedLinear.
final class EngramModule: Module {
    let dim: Int
    let hc: Int
    let eps: Float
    @ModuleInfo(key: "wkv") var wkv: Linear            // QuantizedLinear at load (affine 8-bit/g64)
    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray

    init(config: DeepseekV4Configuration) {
        self.dim = config.hiddenSize
        self.hc = config.hcMult
        self.eps = config.rmsNormEps
        let nCols = (config.engramMaxNgramSize - 1) * config.engramNHeads
        self._wkv.wrappedValue = Linear(nCols * config.engramHeadDim, config.hiddenSize * (config.hcMult + 1), bias: false)
        self._qWeight.wrappedValue = MLXArray.ones([config.hcMult, config.hiddenSize])
        self._kWeight.wrappedValue = MLXArray.ones([config.hcMult, config.hiddenSize])
    }

    /// h [1,S,hc,D]; rows [1,S,cols,headDim] bf16. textMask [1,S] optional (0 on image tokens).
    func callAsFunction(_ h: MLXArray, rows: MLXArray, textMask: MLXArray? = nil) -> MLXArray {
        let S = rows.dim(1)
        let kv = wkv(rows.reshaped(1, S, -1))                       // [1,S, D*(hc+1)]
        let key = kv[0..., 0..., 0 ..< (hc * dim)].asType(.float32).reshaped(1, S, hc, dim)
        let value = kv[0..., 0..., (hc * dim)...].asType(.float32)  // [1,S,D]
        let weight = qWeight.asType(.float32) * kWeight.asType(.float32)   // [hc,D]
        let hf = h.asType(.float32)
        let rstd = rsqrt(mean(hf * hf, axis: -1) + eps) * rsqrt(mean(key * key, axis: -1) + eps)  // [1,S,hc]
        let dot = (hf * weight * key).sum(axis: -1) * rstd * pow(Float(dim), -0.5)                // [1,S,hc]
        var gate = sigmoid(sign(dot) * sqrt(maximum(abs(dot), 1e-6)))                             // [1,S,hc]
        if let textMask { gate = gate * textMask.expandedDimensions(axis: -1) }
        let add = gate.expandedDimensions(axis: -1) * value.expandedDimensions(axis: 2)           // [1,S,hc,D]
        return (hf + add).asType(h.dtype)
    }
}

// MARK: - Runtime (aux + tables), resolved from the bundle

/// Immutable engram state for one loaded DeepSeek-V4.1 model: aux hashing tables plus the
/// disk-resident FP8 tables. Holds NO per-sequence state (history travels with the KV cache), so
/// it is safe to share across concurrent forwards.
final class EngramRuntime: @unchecked Sendable {
    let aux: EngramAux
    let hasher: EngramNgramHasher
    let layerIds: [Int]
    private let tables: [Int: EngramTable]

    /// Background lane for overlapping deeper layers' table reads with the forward.
    private static let prefetchQueue = DispatchQueue(
        label: "ai.vmlx.dsv41.engram-prefetch", qos: .userInitiated, attributes: .concurrent)

    /// Resolve the engram tables of a V4.1 bundle. Order: `VMLX_DSV41_ENGRAM_DIR` (a directory
    /// holding `engram_aux.safetensors` + `engram/`, diagnostic override only), then the bundle:
    /// `config.json` / `jang_config.json` `runtime.engram_dir` (default `engram`), with
    /// `engram_aux.safetensors` at the bundle root (or inside the engram dir).
    static func resolve(
        modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> (aux: URL, engramDir: URL) {
        if let override = environment["VMLX_DSV41_ENGRAM_DIR"], !override.isEmpty {
            let dir = URL(fileURLWithPath: override)
            return (dir.appendingPathComponent("engram_aux.safetensors"),
                    dir.appendingPathComponent("engram"))
        }
        func runtimeEngramDir(_ name: String) -> String? {
            guard let data = try? Data(contentsOf: modelDirectory.appendingPathComponent(name)),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let runtime = obj["runtime"] as? [String: Any],
                let dir = runtime["engram_dir"] as? String, !dir.isEmpty
            else { return nil }
            return dir
        }
        let rel = runtimeEngramDir("config.json") ?? runtimeEngramDir("jang_config.json") ?? "engram"
        let engramDir = rel.hasPrefix("/")
            ? URL(fileURLWithPath: rel)
            : modelDirectory.appendingPathComponent(rel)
        let fm = FileManager.default
        var aux = modelDirectory.appendingPathComponent("engram_aux.safetensors")
        if !fm.fileExists(atPath: aux.path) {
            let inner = engramDir.appendingPathComponent("engram_aux.safetensors")
            if fm.fileExists(atPath: inner.path) { aux = inner }
        }
        guard fm.fileExists(atPath: aux.path) else { throw EngramError.missingFile(aux) }
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: engramDir.path, isDirectory: &isDir), isDir.boolValue else {
            throw EngramError.missingFile(engramDir)
        }
        return (aux, engramDir)
    }

    /// Load and validate against `config`. Throws (never returns a partial runtime) when any table
    /// or aux field is missing or inconsistent, so a broken bundle fails at load instead of
    /// producing silent garbage.
    convenience init(modelDirectory: URL, config: DeepseekV4Configuration) throws {
        let (auxURL, engramDir) = try Self.resolve(modelDirectory: modelDirectory)
        try self.init(auxURL: auxURL, engramDir: engramDir, config: config)
    }

    init(auxURL: URL, engramDir: URL, config: DeepseekV4Configuration?) throws {
        let aux = try EngramAux.load(auxURL)
        if let config {
            guard aux.layerIds == config.engramLayerIds else {
                throw EngramError.mismatch(
                    "layer_ids \(aux.layerIds) vs engram_layer_ids \(config.engramLayerIds)")
            }
            if config.engramNHeads > 0, aux.nHeads != config.engramNHeads {
                throw EngramError.mismatch("n_heads \(aux.nHeads) vs \(config.engramNHeads)")
            }
            if config.engramHeadDim > 0, aux.headDim != config.engramHeadDim {
                throw EngramError.mismatch("head_dim \(aux.headDim) vs \(config.engramHeadDim)")
            }
            if config.engramMaxNgramSize > 1, aux.maxNgram != config.engramMaxNgramSize {
                throw EngramError.mismatch(
                    "max_ngram \(aux.maxNgram) vs \(config.engramMaxNgramSize)")
            }
            guard aux.tokenMap.count >= config.vocabSize else {
                throw EngramError.mismatch("token_map \(aux.tokenMap.count) < vocab \(config.vocabSize)")
            }
        }
        self.aux = aux
        self.hasher = EngramNgramHasher(aux)
        self.layerIds = aux.layerIds

        // Resolve shard key -> file: index.json when present, otherwise scan every shard header.
        var keyToFile: [String: URL] = [:]
        var headers: [URL: [String: EngramSafetensors.Entry]] = [:]
        let idxURL = engramDir.appendingPathComponent("engram.safetensors.index.json")
        if let idxData = try? Data(contentsOf: idxURL),
            let idx = try? JSONSerialization.jsonObject(with: idxData) as? [String: Any],
            let wm = idx["weight_map"] as? [String: String]
        {
            for (k, f) in wm { keyToFile[k] = engramDir.appendingPathComponent(f) }
        } else {
            let files = (try? FileManager.default.contentsOfDirectory(
                at: engramDir, includingPropertiesForKeys: nil)) ?? []
            for url in files.sorted(by: { $0.path < $1.path }) where url.pathExtension == "safetensors" {
                let entries = try EngramSafetensors.header(url).entries
                headers[url] = entries
                for k in entries.keys where k.contains(".engram.embed.") { keyToFile[k] = url }
            }
        }
        func entry(_ key: String) throws -> (URL, EngramSafetensors.Entry) {
            guard let file = keyToFile[key] else { throw EngramError.missingTensor(key) }
            let entries: [String: EngramSafetensors.Entry]
            if let cached = headers[file] {
                entries = cached
            } else {
                entries = try EngramSafetensors.header(file).entries
                headers[file] = entries
            }
            guard let e = entries[key] else { throw EngramError.missingTensor(key) }
            return (file, e)
        }
        var tbl: [Int: EngramTable] = [:]
        for l in aux.layerIds {
            let (wf, w) = try entry("layers.\(l).engram.embed.weight")
            let (sf, sc) = try entry("layers.\(l).engram.embed.scale")
            guard w.dtype == "F8_E4M3", sc.dtype == "F8_E8M0" else {
                throw EngramError.malformed(wf, "layer \(l) table dtypes \(w.dtype)/\(sc.dtype)")
            }
            if let size = (try? FileManager.default.attributesOfItem(atPath: wf.path))?[.size]
                as? NSNumber, size.intValue < w.end
            {
                throw EngramError.malformed(wf, "file shorter than layer \(l) table")
            }
            let table = try EngramTable(
                weightFile: wf, weightOffset: w.start, weightShape: w.shape,
                scaleFile: sf, scaleOffset: sc.start, scaleShape: sc.shape)
            guard table.dim == aux.headDim else {
                throw EngramError.mismatch("layer \(l) row width \(table.dim) vs head_dim \(aux.headDim)")
            }
            tbl[l] = table
        }
        self.tables = tbl
    }

    /// A chunk's engram rows. The first engram layer is read synchronously; deeper layers are read
    /// on a background lane so their SSD reads overlap the forward of the layers before them. Rows
    /// become MLX tensors only on the calling (forward) thread.
    final class Chunk: @unchecked Sendable {
        private let tokens: Int
        private let cols: Int
        private var pending: [Int: (table: EngramTable, group: DispatchGroup, box: Box)] = [:]

        final class Box: @unchecked Sendable { var raw: EngramTable.RawLookup? }

        init(tokens: Int, cols: Int) {
            self.tokens = tokens
            self.cols = cols
        }

        func add(layer: Int, table: EngramTable, group: DispatchGroup, box: Box) {
            pending[layer] = (table, group, box)
        }

        /// `[1, L, cols, headDim]` bf16 rows for `layer`, or nil if it is not an engram layer.
        func rows(layer: Int) -> MLXArray? {
            guard let p = pending[layer] else { return nil }
            p.group.wait()
            guard let raw = p.box.raw else { return nil }
            return p.table.materialize(raw).reshaped(1, tokens, cols, p.table.dim)
        }
    }

    /// The next forward's rows, requested while the current chunk computes (reference
    /// `DSV41._start_lookahead`). Used only by a forward with exactly this start position,
    /// compressed ids and history; anything else discards it.
    final class Lookahead: @unchecked Sendable {
        let startPos: Int
        let comp: [Int64]
        let previous: [Int64]
        let chunk: Chunk

        init(startPos: Int, comp: [Int64], previous: [Int64], chunk: Chunk) {
            self.startPos = startPos
            self.comp = comp
            self.previous = previous
            self.chunk = chunk
        }

        func matches(startPos: Int, comp: [Int64], previous: [Int64]) -> Bool {
            self.startPos == startPos && self.comp == comp && self.previous == previous
        }
    }

    /// Hash the chunk and start its table reads.
    /// - Parameters:
    ///   - comp: compressed ids of the chunk (see `EngramNgramHasher.compress`).
    ///   - previous: compressed ids immediately before `startPos` (cache-carried history).
    ///   - background: read every layer on the background lane, one layer after another so
    ///     the first layer's rows (needed first) get every I/O lane (lookahead requests).
    ///     Otherwise the first layer is read before returning.
    func prepare(comp: [Int64], startPos: Int, previous: [Int64], background: Bool = false) -> Chunk {
        let h = hasher.hashes(comp, startPos: startPos, previous: previous)
        let L = comp.count
        let cols = aux.cols
        let nLayers = aux.layerIds.count
        let chunk = Chunk(tokens: L, cols: cols)
        var chained: [(table: EngramTable, ids: [Int64], group: DispatchGroup, box: Chunk.Box)] = []
        for (j, lid) in aux.layerIds.enumerated() {
            guard let table = tables[lid] else { continue }
            var ids = [Int64](repeating: 0, count: L * cols)
            for i in 0..<L {
                let src = (i * nLayers + j) * cols
                for c in 0..<cols { ids[i * cols + c] = h[src + c] }
            }
            let group = DispatchGroup()
            let box = Chunk.Box()
            if background {
                group.enter()
                chained.append((table, ids, group, box))
            } else if j == 0 {
                box.raw = table.readLookup(ids)
            } else {
                Self.prefetchQueue.async(group: group) { box.raw = table.readLookup(ids) }
            }
            chunk.add(layer: lid, table: table, group: group, box: box)
        }
        if !chained.isEmpty {
            Self.prefetchQueue.async {
                for read in chained {
                    read.box.raw = read.table.readLookup(read.ids)
                    read.group.leave()
                }
            }
        }
        return chunk
    }
}
