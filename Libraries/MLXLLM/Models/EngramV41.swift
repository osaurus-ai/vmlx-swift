// DeepSeek-V4.1 Engram: per-token n-gram-hash gated residual applied at engram_layer_ids (1, 14).
//
// This is NOT a long-context feature -- the reference adds `h = layer.engram(h, rows, mask)` to the
// FULL hidden state at those layers for EVERY token, so without it the model emits garbage even on a
// short prompt. Port of jang_tools/dsv41/{engram.py,model.py} (the working vmlx-py path).
//
// Split of responsibilities (mirrors the Python):
//   * EngramAux        -- the hashing tables exported by scratchpad/export_engram_aux.py
//                         (token_map / multipliers / primes / offsets + scalars), so Swift does not
//                         reimplement HF-tokenizer normalization. Loaded from `engram_aux.safetensors`.
//   * EngramNgramHasher-- pure-CPU Int64 n-gram hashing with per-sequence compressed-token history
//                         (exactly numpy's arithmetic; MLX int64 bitops are avoided on purpose).
//   * EngramTable      -- one engram layer's disk-resident FP8 embedding table, memory-mapped from the
//                         bundle's engram/ shards and looked up per hash id (E4M3 rows x E8M0 blocks).
//   * EngramModule     -- the wkv (affine-quant, in the BACKBONE) + q/k_weight gated residual.
//
// The wkv/q_weight/k_weight module weights live in the backbone shards (affine quant) and load through
// the normal path; only the big embed table is disk-resident here.

import Darwin
import Foundation
import MLX
import MLXNN

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

    /// Minimal safetensors reader (u64 header length, JSON header, then raw little-endian tensor data).
    static func load(_ url: URL) throws -> EngramAux {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let hlen = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt64.self).littleEndian }
        let headerData = data.subdata(in: 8 ..< (8 + Int(hlen)))
        let base = 8 + Int(hlen)
        let header = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        let meta = header["__metadata__"] as! [String: String]

        func raw(_ name: String) -> (bytes: Data, shape: [Int], dtype: String) {
            let t = header[name] as! [String: Any]
            let shape = (t["shape"] as! [Any]).map { ($0 as! NSNumber).intValue }
            let off = (t["data_offsets"] as! [Any]).map { ($0 as! NSNumber).intValue }
            return (data.subdata(in: (base + off[0]) ..< (base + off[1])), shape, t["dtype"] as! String)
        }
        func i32(_ d: Data) -> [Int32] {
            d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        func i64(_ d: Data) -> [Int64] {
            d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
        }

        let tm = raw("token_map"); let tokenMap = i32(tm.bytes)
        let mu = raw("multipliers"); let muFlat = i64(mu.bytes)
        let pr = raw("primes"); let prFlat = i64(pr.bytes)
        let of = raw("offsets"); let ofFlat = i64(of.bytes)
        let padId = i64(raw("pad_id").bytes)[0]

        let nLayers = mu.shape[0], maxNgram = mu.shape[1]
        let nHeads = pr.shape[2]
        let cols = of.shape[1]

        func reshape2(_ flat: [Int64], _ a: Int, _ b: Int) -> [[Int64]] {
            (0..<a).map { i in Array(flat[(i * b)..<((i + 1) * b)]) }
        }
        let multipliers = reshape2(muFlat, nLayers, maxNgram)
        let offsets = reshape2(ofFlat, nLayers, cols)
        let ngM1 = pr.shape[1]
        var primes = [[[Int64]]]()
        for l in 0..<nLayers {
            var perNg = [[Int64]]()
            for g in 0..<ngM1 {
                let start = (l * ngM1 + g) * nHeads
                perNg.append(Array(prFlat[start..<(start + nHeads)]))
            }
            primes.append(perNg)
        }

        return EngramAux(
            tokenMap: tokenMap, multipliers: multipliers, primes: primes, offsets: offsets,
            padId: padId, nHeads: nHeads, headDim: Int(meta["head_dim"]!)!,
            maxNgram: maxNgram, layerIds: meta["layer_ids"]!.split(separator: ",").map { Int($0)! },
            compressedVocab: Int(meta["compressed_vocab_size"]!)!)
    }
}

// MARK: - N-gram hashing (CPU, per sequence)

/// Per-sequence n-gram hash ids, with the compressed-token history kept across decode steps.
/// Batch size 1 per sequence (matches the runtime session). Mirrors `NgramHasher.hash` exactly.
final class EngramNgramHasher {
    static let DEAD: Int64 = -1
    let aux: EngramAux
    private var history: [Int64] = []   // compressed token ids seen so far

    init(_ aux: EngramAux) { self.aux = aux }

    func reset() { history = [] }

    /// inputIds [L] (absolute positions startPos..startPos+L-1). Returns hash ids [L][nLayers][cols].
    /// `keep` (optional, [L], true=real text) makes image tokens DEAD so they join no n-gram.
    func hashes(_ inputIds: [Int32], startPos: Int, keep: [Bool]? = nil) -> [[[Int64]]] {
        let L = inputIds.count
        let n = aux.maxNgram
        let nLayers = aux.multipliers.count
        let nHeads = aux.nHeads

        // comp = token_map[input_ids], image tokens -> DEAD
        var comp = [Int64](repeating: 0, count: L)
        for i in 0..<L {
            let c = Int64(aux.tokenMap[Int(inputIds[i])])
            comp[i] = (keep?[i] == false) ? Self.DEAD : c
        }
        // history = history[:startPos] + comp  (startPos is authoritative; trims any stale tail)
        if history.count > startPos { history.removeLast(history.count - startPos) }
        while history.count < startPos { history.append(aux.padId) }  // defensive; shouldn't trigger
        history.append(contentsOf: comp)

        // toks[pos][shift] = history[clip(pos-shift,0,..)] with blocking (pos<shift or DEAD) -> pad
        // toks shape [L][n]
        var toks = [[Int64]](repeating: [Int64](repeating: 0, count: n), count: L)
        for i in 0..<L {
            let pos = startPos + i
            var blocked = false
            for shift in 0..<n {
                let srcIdx = max(pos - shift, 0)
                let src = history[srcIdx]
                if pos < shift || src == Self.DEAD { blocked = true }
                toks[i][shift] = blocked ? aux.padId : src
            }
        }

        // prod[pos][layer][k] = toks[pos][k] * multipliers[layer][k]; rolling XOR; % primes + offsets
        var out = [[[Int64]]](repeating:
            [[Int64]](repeating: [Int64](repeating: 0, count: aux.cols), count: nLayers), count: L)
        for i in 0..<L {
            for l in 0..<nLayers {
                var rolling = toks[i][0] &* aux.multipliers[l][0]
                var col = 0
                for k in 1..<n {
                    rolling ^= toks[i][k] &* aux.multipliers[l][k]
                    let primesK = aux.primes[l][k - 1]   // [nHeads]
                    for hd in 0..<nHeads {
                        // nonnegative modulo (Python % on positive primes of possibly-negative rolling
                        // never happens here: rolling of unsigned-ish products stays >= 0 in practice,
                        // but guard anyway)
                        var m = rolling % primesK[hd]
                        if m < 0 { m += primesK[hd] }
                        out[i][l][col] = m + aux.offsets[l][col]
                        col += 1
                    }
                }
            }
        }
        return out
    }
}

// MARK: - SSD-resident FP8 embedding table (no mmap)

/// One engram layer's FP8 table, kept on SSD and read per touched row — NOT memory-mapped.
///
/// The engram embed tables are ~189 GB across the bundle's `engram/` shards; mapping them would
/// either fault hundreds of MiB per gather into RSS or make throughput hostage to the filesystem
/// page cache. Instead this mirrors `Qwen4ExpNGramTable`: each shard is held open `O_RDONLY` with
/// `F_NOCACHE` and only the bytes of the rows a token actually hashes to are `pread` off disk. The
/// model weights stay the only resident tensors (Activity Monitor RSS ≈ backbone size); the n-gram
/// table is a sparse SSD lookup. Row r = E4M3 bytes [dim], scaled per block of `block` by E8M0 scale
/// bytes [sdim]. `lookup()` returns bf16 rows [count, dim]; the dequant math is unchanged.
final class EngramTable {
    private let wFd: Int32           // weight shard fd (O_RDONLY, F_NOCACHE)
    private let sFd: Int32           // scale shard fd (== wFd when same file)
    private let ownsScaleFd: Bool
    private let wOff: Int            // byte offset of this tensor's data in the file
    private let sOff: Int
    let dim: Int                     // 256
    let sdim: Int                    // 8
    let block: Int                   // dim/sdim = 32

    init(weightFile: URL, weightOffset: Int, weightShape: [Int],
         scaleFile: URL, scaleOffset: Int, scaleShape: [Int]) throws {
        func openNoCache(_ url: URL) throws -> Int32 {
            let fd = Darwin.open(url.path, O_RDONLY)
            guard fd >= 0 else { throw EngramTableError.openFailed(url, errno) }
            _ = fcntl(fd, F_NOCACHE, 1)  // best-effort: keep the sparse table off the page cache
            return fd
        }
        self.wFd = try openNoCache(weightFile)
        if scaleFile == weightFile {
            self.sFd = self.wFd
            self.ownsScaleFd = false
        } else {
            do {
                self.sFd = try openNoCache(scaleFile)
            } catch {
                Darwin.close(self.wFd)
                throw error
            }
            self.ownsScaleFd = true
        }
        self.wOff = weightOffset
        self.sOff = scaleOffset
        self.dim = weightShape[1]
        self.sdim = scaleShape[1]
        self.block = weightShape[1] / scaleShape[1]
    }

    deinit {
        if ownsScaleFd { Darwin.close(sFd) }
        Darwin.close(wFd)
    }

    /// pread exactly `count` bytes from `fd` at `offset` into `dst` (handles short reads).
    private static func preadExactly(_ fd: Int32, _ dst: UnsafeMutableRawPointer, _ count: Int, _ offset: Int) {
        var done = 0
        while done < count {
            let n = Darwin.pread(fd, dst.advanced(by: done), count - done, off_t(offset + done))
            if n <= 0 { break }  // short/failed read leaves zeros; lookup degrades, never crashes
            done += n
        }
    }

    /// ids [...] (flattened externally) -> bf16 [count, dim]. Duplicate ids are read once.
    func lookup(_ ids: [Int64]) -> MLXArray {
        let count = ids.count
        // dedup
        var uniqIndex = [Int64: Int]()
        var uniq = [Int64]()
        var inv = [Int32](repeating: 0, count: count)
        for (i, id) in ids.enumerated() {
            if let u = uniqIndex[id] { inv[i] = Int32(u) }
            else { let u = uniq.count; uniqIndex[id] = u; uniq.append(id); inv[i] = Int32(u) }
        }
        let U = uniq.count
        var rowBytes = [UInt8](repeating: 0, count: U * dim)
        var scaleBytes = [UInt8](repeating: 0, count: U * sdim)
        rowBytes.withUnsafeMutableBytes { rdst in
            scaleBytes.withUnsafeMutableBytes { sdst in
                let rbase = rdst.baseAddress!
                let sbase = sdst.baseAddress!
                for (j, id) in uniq.enumerated() {
                    let r = Int(id)
                    // Only the touched rows are read off SSD (no whole-file mapping).
                    Self.preadExactly(wFd, rbase.advanced(by: j * dim), dim, wOff + r * dim)
                    Self.preadExactly(sFd, sbase.advanced(by: j * sdim), sdim, sOff + r * sdim)
                }
            }
        }
        let r = MLXArray(rowBytes, [U, dim])                  // uint8 E4M3
        let s = MLXArray(scaleBytes, [U, sdim])               // uint8 E8M0
        let vals = fromFP8(r, dtype: .float32).reshaped(U, sdim, block)
            * EngramMath.e8m0ToFloat(s).expandedDimensions(axis: -1)
        let dequant = vals.reshaped(U, dim).asType(.bfloat16)
        let invArr = MLXArray(inv)
        return dequant[invArr]                                // [count, dim]
    }
}

enum EngramTableError: Error, CustomStringConvertible {
    case openFailed(URL, Int32)
    var description: String {
        switch self {
        case let .openFailed(url, err): return "engram table open failed for \(url.lastPathComponent): errno \(err)"
        }
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

// MARK: - Runtime (aux + tables + per-sequence hashers)

/// Owns the engram aux tables, the disk-resident FP8 tables, and one n-gram hasher per live
/// sequence (keyed by its KV-cache identity). Produces the per-layer row tensors a forward needs.
/// Located lazily from `VMLX_DSV41_ENGRAM_DIR` (the bundle dir holding engram_aux.safetensors + engram/).
final class EngramRuntime {
    let aux: EngramAux
    let layerIds: [Int]
    private let tables: [Int: EngramTable]
    private var hashers: [ObjectIdentifier: EngramNgramHasher] = [:]

    static func locate() -> EngramRuntime? {
        guard let dir = ProcessInfo.processInfo.environment["VMLX_DSV41_ENGRAM_DIR"] else { return nil }
        return try? EngramRuntime(bundleDir: URL(fileURLWithPath: dir))
    }

    init(bundleDir: URL) throws {
        self.aux = try EngramAux.load(bundleDir.appendingPathComponent("engram_aux.safetensors"))
        self.layerIds = aux.layerIds
        let engDir = bundleDir.appendingPathComponent("engram")
        // Cache each shard's parsed header (dataBase = 8 + headerLen; data_offsets relative to that).
        var headerCache: [URL: (base: Int, header: [String: Any])] = [:]
        func header(_ url: URL) throws -> (base: Int, header: [String: Any]) {
            if let h = headerCache[url] { return h }
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            let lenData = fh.readData(ofLength: 8)
            let hlen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
            let hdrData = fh.readData(ofLength: Int(hlen))
            let hdr = try JSONSerialization.jsonObject(with: hdrData) as! [String: Any]
            let v = (base: 8 + Int(hlen), header: hdr)
            headerCache[url] = v
            return v
        }
        // Resolve each shard key -> file. Prefer an index.json if present; otherwise SCAN every
        // *.safetensors in engram/ and read its header keys. The JANG bundle ships the engram tables as
        // plain `model-000NN-of-000MM.safetensors` shards with NO index (just the two big FP8 tables),
        // which the index/single-file lookup missed -> runtime silently failed to load -> engram never
        // ran (the model needs it, so output was garbage). This scans the real shards.
        var keyToFile: [String: URL] = [:]
        let idxURL = engDir.appendingPathComponent("engram.safetensors.index.json")
        if let idxData = try? Data(contentsOf: idxURL),
           let idx = try? JSONSerialization.jsonObject(with: idxData) as? [String: Any],
           let wm = idx["weight_map"] as? [String: String] {
            for (k, f) in wm { keyToFile[k] = engDir.appendingPathComponent(f) }
        } else {
            let files = (try? FileManager.default.contentsOfDirectory(at: engDir, includingPropertiesForKeys: nil)) ?? []
            for url in files where url.pathExtension == "safetensors" {
                if let (_, hdr) = try? header(url) {
                    for k in hdr.keys where k.hasPrefix("layers.") && k.contains(".engram.embed.") {
                        keyToFile[k] = url
                    }
                }
            }
        }
        func tensorInfo(_ key: String) throws -> (file: URL, offset: Int, shape: [Int]) {
            guard let file = keyToFile[key] else { throw EngramError.missing(key) }
            let (base, hdr) = try header(file)
            let t = hdr[key] as! [String: Any]
            let shape = (t["shape"] as! [Any]).map { ($0 as! NSNumber).intValue }
            let off = (t["data_offsets"] as! [Any]).map { ($0 as! NSNumber).intValue }
            return (file, base + off[0], shape)
        }
        var tbl: [Int: EngramTable] = [:]
        for l in aux.layerIds {
            let w = try tensorInfo("layers.\(l).engram.embed.weight")
            let s = try tensorInfo("layers.\(l).engram.embed.scale")
            tbl[l] = try EngramTable(
                weightFile: w.file, weightOffset: w.offset, weightShape: w.shape,
                scaleFile: s.file, scaleOffset: s.offset, scaleShape: s.shape)
        }
        self.tables = tbl
    }

    enum EngramError: Error { case missing(String) }

    /// Rows for every engram layer for this chunk. inputIds [L] (Int32). `cacheKey` identifies the
    /// sequence; `reset` true at start_pos 0. Returns {layerId: [1, L, cols, headDim] bf16}.
    func rows(inputIds: [Int32], startPos: Int, cacheKey: ObjectIdentifier, reset: Bool) -> [Int: MLXArray] {
        let hasher: EngramNgramHasher
        if let h = hashers[cacheKey] { hasher = h }
        else { hasher = EngramNgramHasher(aux); hashers[cacheKey] = hasher }
        if reset { hasher.reset() }
        let h = hasher.hashes(inputIds, startPos: startPos)   // [L][nLayers][cols]
        let L = inputIds.count
        let cols = aux.cols
        var result: [Int: MLXArray] = [:]
        for (j, lid) in aux.layerIds.enumerated() {
            guard let table = tables[lid] else { continue }
            var ids = [Int64](repeating: 0, count: L * cols)
            var p = 0
            for i in 0..<L { for c in 0..<cols { ids[p] = h[i][j][c]; p += 1 } }
            let looked = table.lookup(ids)                     // [L*cols, headDim]
            result[lid] = looked.reshaped(1, L, cols, table.dim)
        }
        return result
    }

    func drop(cacheKey: ObjectIdentifier) { hashers[cacheKey] = nil }
}
