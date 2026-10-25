// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 Engram runtime contracts (CPU only, no model weights): stateless n-gram hashing is
// restore-safe (chunked == one-shot given the cache-carried history), the history travels with
// DeepseekV4Cache through copy/trim/metaState, concurrent table reads equal serial reads, and a
// bundle without engram tables fails at resolve time.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 engram runtime")
struct EngramV41FocusedTests {

    /// Small synthetic aux with the real layout (2 layers, max_ngram 4, 3 heads).
    static func syntheticAux() -> EngramAux {
        let nHeads = 3, maxNgram = 4, nLayers = 2
        var rng = SplitMix64(seed: 7)
        let tokenMap = (0..<500).map { _ in Int32(rng.next() % 300) }
        let multipliers = (0..<nLayers).map { _ in (0..<maxNgram).map { _ in Int64(rng.next() % 1_000_003) * 2 + 1 } }
        let primes: [[[Int64]]] = (0..<nLayers).map { l in
            (0..<(maxNgram - 1)).map { g in (0..<nHeads).map { h in [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051][(l * 3 + g + h) % 9] } }
        }
        let cols = (maxNgram - 1) * nHeads
        let offsets = (0..<nLayers).map { l in (0..<cols).map { Int64($0 * 1100 + l * 50_000) } }
        return EngramAux(
            tokenMap: tokenMap, multipliers: multipliers, primes: primes, offsets: offsets,
            padId: 299, nHeads: nHeads, headDim: 16, maxNgram: maxNgram, layerIds: [1, 14],
            compressedVocab: 300)
    }

    @Test("chunked hashing with carried history equals one-shot hashing")
    func chunkedEqualsOneShot() {
        let hasher = EngramNgramHasher(Self.syntheticAux())
        var rng = SplitMix64(seed: 11)
        let ids = (0..<97).map { _ in Int32(rng.next() % 500) }
        let comp = hasher.compress(ids)
        let full = hasher.hashes(comp, startPos: 0, previous: [])
        let per = hasher.aux.multipliers.count * hasher.aux.cols
        for split in [1, 2, 3, 4, 5, 50, 96] {
            let head = hasher.hashes(Array(comp[..<split]), startPos: 0, previous: [])
            let reach = min(hasher.historyReach, split)
            let tail = hasher.hashes(
                Array(comp[split...]), startPos: split, previous: Array(comp[(split - reach)..<split]))
            #expect(head + tail == full, "split \(split)")
            #expect(head.count == split * per)
        }
    }

    @Test("history-less suffix differs from the true hash (the defect being guarded)")
    func missingHistoryIsDetectable() {
        let hasher = EngramNgramHasher(Self.syntheticAux())
        let comp = hasher.compress((0..<20).map { Int32($0 * 7) })
        let full = hasher.hashes(comp, startPos: 0, previous: [])
        let padded = hasher.hashes(Array(comp[10...]), startPos: 10, previous: [])
        let per = hasher.aux.multipliers.count * hasher.aux.cols
        #expect(Array(full[(10 * per)...]) != padded)
    }

    @Test("matches the reference NgramHasher on the real bundle aux tables")
    func matchesReferenceOnBundleAux() throws {
        let aux = URL(fileURLWithPath:
            "/Users/eric/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX/engram_aux.safetensors")
        guard FileManager.default.fileExists(atPath: aux.path) else { return }
        let hasher = EngramNgramHasher(try EngramAux.load(aux))
        let ids: [Int32] = [0, 3923, 128803, 1, 100, 100, 100, 7, 129000, 55555, 42, 9, 12345]
        // Pure-python replica of engram.py NgramHasher.hash (numpy int64 semantics).
        let expected: [Int64] = [5702652, 24393184, 39183301, 61503783, 78984481, 90144770, 103724983, 121476532, 131476717, 152385926, 164143652, 191599181, 198145581, 217419202, 238329687, 247482304, 257520185, 283540304, 288689443, 315264704, 322314652, 348890037, 367865216, 380066193, 14361964, 29977809, 32716444, 61209771, 67456205, 84275051, 108494439, 120604547, 132225184, 150270452, 174805952, 178895254, 193207923, 211610196, 226280582, 252505246, 261362261, 279862797, 301364059, 312864848, 328367036, 349372619, 363873742, 372375971, 12965746, 26944404, 32626337, 62366664, 67926688, 91366796, 103134342, 124898803, 139769478, 145455487, 175250897, 186009567, 207438529, 218580282, 227826845, 255017647, 264129995, 282398210, 288258236, 307295400, 325359853, 344515647, 366152524, 376116836, 13642165, 16532309, 45832057, 54838103, 75383597, 93406259, 107471298, 123353830, 128485872, 154210166, 175435196, 179070613, 200015225, 209161035, 230996488, 249291835, 271711999, 283085446, 297656976, 317721882, 327828818, 338920899, 361827052, 380457533, 6174882, 17645942, 34650846, 51188742, 72326360, 81593771, 111644218, 126951448, 135504035, 157910395, 174375566, 188272139, 206386263, 212191704, 228835499, 253223649, 258588057, 282992310, 290801212, 313893118, 330468118, 341100803, 367112345, 372402128, 11775258, 22109453, 41564951, 58810715, 68716225, 91587351, 98490930, 119915281, 129958053, 151905548, 175938592, 177603263, 198163582, 221702885, 224742658, 241777722, 262533080, 286651833, 297825283, 319535389, 335595020, 340572219, 354435687, 378956870, 9828215, 28834032, 40833206, 57363217, 75948249, 93287359, 102046418, 112568241, 141066565, 144810605, 168933008, 178144447, 200344293, 212322856, 239389151, 241795959, 271205300, 286284530, 303339661, 304086684, 331333387, 348324877, 363335775, 379834673, 4512437, 22865516, 43727208, 55476159, 70303284, 85305500, 111886053, 120461174, 136162019, 145698900, 160621277, 184016623, 206939366, 218219962, 228496061, 250591410, 263186754, 276882733, 292942522, 309083608, 332589357, 347369472, 357711540, 379820932, 14962927, 31177945, 42834970, 56097149, 68011633, 82655547, 105978701, 125575715, 129670217, 159324403, 160020063, 178043758, 203392409, 214041120, 229482170, 240576398, 259530723, 285167083, 303946870, 306864785, 330709935, 351326275, 366885207, 374714326, 4716190, 25989736, 41922907, 52548173, 73862261, 84300943, 100238324, 116226453, 138174806, 151414925, 173711012, 186759289, 195423314, 208840953, 237661042, 246690952, 266407704, 285812823, 295476412, 305592621, 328553080, 345443745, 357048997, 368475620, 1791437, 16815695, 32728193, 48034394, 79016352, 94683077, 110517636, 124511594, 130286476, 152810202, 161373136, 187328291, 206768365, 216498506, 239106498, 253145354, 256368662, 275173055, 296741518, 318606202, 334323494, 343610277, 359444165, 375594587, 10380400, 27875414, 42477456, 62884134, 73088969, 95386854, 107411446, 113453964, 132948226, 159318066, 171762923, 176507340, 201121016, 214498626, 226172400, 246076897, 267873358, 281638956, 299488908, 308671313, 324605215, 350203581, 366489053, 371730238, 1791437, 16815695, 32728193, 48034394, 79016352, 94683077, 110517636, 124511594, 133392712, 154294917, 162498173, 184691199, 201993365, 210657792, 231671263, 244473296, 262435317, 282932813, 300238292, 316894600, 322666587, 339326641, 354223468, 377365336, 10380400, 27875414, 42477456, 62884134, 73088969, 95386854, 107411446, 113453964, 129460840, 147765372, 175425768, 180038350, 196188043, 216964876, 235624542, 247186997, 259182995, 286379364, 293319117, 307725113, 323225538, 345726994, 355868139, 374574776, 9284858, 19825082, 45510715, 58995031, 79222201, 85964775, 105336227, 123479465, 132424545, 153307591, 165060427, 176484644, 199030181, 218307780, 239229902, 248396274, 268819611, 276016180, 298663257, 317990166, 330875948, 350206385, 360678643, 380271661, 14371127, 21447675, 33143158, 51609290, 76843511, 92016674, 111197008, 120800171, 138306506, 156926464, 172360047, 177599589, 195939106, 219521059, 226829246, 255935020, 270773115, 286678973, 302614679, 318589935, 334509057, 350416928, 366414122, 382398926, 6230937, 25658347, 37571569, 61395596, 69526729, 90475958, 110370409, 126910053, 138203869, 154723460, 162775821, 184483341, 200321836, 213021845, 224047505, 251210674, 270383793, 278585240, 288661639, 304893691, 332898298, 336678455, 367393986, 370506520, 15032824, 22598229, 38577003, 53411511, 78083872, 95444175, 103797559, 117378878, 139752593, 149635348, 172527717, 181936250, 197449354, 223301292, 234675683, 254251913, 256030810, 275792292, 301079591, 310477962, 334351348, 343556022, 363830121, 372327932, 15199159, 19840796, 36573090, 60292379, 78076127, 85658691, 105760350, 126480326, 135359873, 150073342, 168134176, 178257337, 207994231, 209654501, 231635305, 249849182, 266407181, 274673048, 299496634, 312604391, 326934126, 340815173, 354167362, 374620385, 5045669, 30619804, 34717512, 62061932, 76717730, 83722868, 96596975, 114951030, 141287231, 155434316, 161977761, 178697130, 202376673, 208050968, 225679271, 255752091, 262504116, 272459195, 298276431, 318658740, 326019186, 351779758, 363658445, 377509503, 15676970, 25876813, 32635976, 54791294, 72298086, 83540097, 97174863, 124378591, 140366128, 159078046, 164432606, 182919972, 203589717, 211797419, 231450164, 242159223, 265661069, 272803914, 290511509, 308833751, 321592375, 341546005, 362646221, 373170849, 14479402, 21920446, 41262895, 54650533, 77492663, 92134324, 107886081, 126755178, 143573253, 152745771, 172535133, 176860592, 202311321, 222030070, 225433211, 240903414, 257853966, 277214073, 296379856, 305694097, 329493863, 349764465, 355143232, 383137397, 14061378, 20143754, 33787819, 62940846, 66673120, 89251498, 100540839, 121664954, 131217928, 154106789, 171827540, 178058349, 194176869, 219167798, 233817998, 246178472, 271892081, 273356052, 303235081, 313179715, 320922547, 350206818, 352861112, 383915947, 8180368, 28598002, 46700588, 53436455, 64804934, 89036428, 111386180, 122566978, 142069742, 155416925, 172731572, 183824905, 199116753, 210700091, 227336198, 249622366, 264826895, 278754986, 299365890, 314532697, 332252949, 351675632, 359762902, 372245184, 10647353, 27943427, 47114451, 57346042, 67573764, 82417856, 105984055, 112660132, 140078818, 144641549, 163698001, 177595261, 204012103, 218518996, 233902269, 245435773, 258172638, 287766547, 301779935, 308951035, 324459593, 347813774, 354265146, 370928677, 6868141, 20398817, 43735627, 49050621, 67835861, 81844597, 106556451, 121488862, 134430606, 151899102, 171169434, 190679153, 203629664, 221809295, 239930371, 252366096, 264919771, 287562410, 294095925, 309167699, 331420694, 347583968, 359682787, 373016785]
        let comp = hasher.compress(ids)
        #expect(hasher.hashes(comp, startPos: 0, previous: []) == expected)
        let tail = hasher.hashes(Array(comp[6...]), startPos: 6, previous: Array(comp[3..<6]))
        #expect(tail == Array(expected[(6 * 48)...]))
    }

    @Test("history travels with DeepseekV4Cache through copy, trim and metaState")
    func cacheCarriesHistory() {
        let cache = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 2, requiresAuxiliaryTokenHistory: true)
        #expect(cache.auxiliaryTokenHistory == nil)
        cache.recordAuxiliaryTokens([1, 2, 3, 4, 5], startPos: 0)
        cache.recordAuxiliaryTokens([6, 7], startPos: 5)
        #expect(cache.auxiliaryTokens(before: 7, count: 3) == [5, 6, 7])
        #expect(cache.auxiliaryTokens(before: 5, count: 3) == [3, 4, 5])
        #expect(cache.auxiliaryTokens(before: 8, count: 1) == nil)
        // Rewind to 4 (stale tail beyond startPos is dropped on record).
        cache.recordAuxiliaryTokens([40], startPos: 4)
        #expect(cache.auxiliaryTokenHistory == HybridPoolTokenHistory(endPosition: 5, tokens: [1, 2, 3, 4, 40]))

        let dup = cache.copy() as! DeepseekV4Cache
        #expect(dup.auxiliaryTokenHistory == cache.auxiliaryTokenHistory)
        #expect(dup.requiresAuxiliaryTokenHistory)

        let restored = DeepseekV4Cache(slidingWindow: 128, compressRatio: 2)
        restored.metaState = cache.metaState
        #expect(restored.auxiliaryTokenHistory == cache.auxiliaryTokenHistory)

        // 0731-shaped caches never emit the trailer.
        let plain = DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        #expect(!plain.metaState.contains("dsv41_aux_hist_v1"))
        #expect(plain.metaState.suffix(3) == ["dsv4_cache_v1", "4", "128"])

        // Bounded window.
        let big = DeepseekV4Cache(slidingWindow: 128, compressRatio: 2)
        big.recordAuxiliaryTokens(Array(repeating: 9, count: 5000), startPos: 0)
        #expect(big.auxiliaryTokenHistory?.tokens.count == DeepseekV4Cache.auxiliaryTokenHistoryCapacity)
        #expect(big.auxiliaryTokens(before: 5000, count: 3) == [9, 9, 9])
    }

    @Test("concurrent pread equals serial pread on random rows")
    func concurrentReadEqualsSerial() throws {
        let rows = 4096, dim = 256, sdim = 8
        var rng = SplitMix64(seed: 3)
        let header = 64
        var bytes = [UInt8](repeating: 0, count: header + rows * dim + rows * sdim)
        for i in header..<bytes.count { bytes[i] = UInt8(truncatingIfNeeded: rng.next()) }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("engram-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("t.bin")
        try Data(bytes).write(to: file)
        let table = try EngramTable(
            weightFile: file, weightOffset: header, weightShape: [rows, dim],
            scaleFile: file, scaleOffset: header + rows * dim, scaleShape: [rows, sdim])
        for count in [1, 7, 48, 1000, 9000] {
            let ids = (0..<count).map { _ in Int64(rng.next() % UInt64(rows)) }
            let (uniq, _) = EngramTable.dedup(ids)
            let serial = table.readRaw(uniq, lanes: 1)
            let parallel = table.readRaw(uniq)
            #expect(serial.rows == parallel.rows && serial.scales == parallel.scales, "count \(count)")
            for (j, r) in uniq.enumerated().prefix(64) {
                let w = header + Int(r) * dim
                #expect(Array(parallel.rows[(j * dim)..<((j + 1) * dim)]) == Array(bytes[w..<(w + dim)]))
                let s = header + rows * dim + Int(r) * sdim
                #expect(Array(parallel.scales[(j * sdim)..<((j + 1) * sdim)]) == Array(bytes[s..<(s + sdim)]))
            }
        }
    }

    @Test("the real V4.1 bundle resolves its engram tables without any env var")
    func realBundleResolves() throws {
        let bundle = URL(fileURLWithPath:
            "/Users/eric/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX")
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("engram").path)
        else { return }
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self,
            from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
        #expect(config.isDeepseekV41)
        #expect(config.indexSourceLayerIds == [2, 8, 14, 20, 24, 28, 32, 36])
        #expect(config.candidateSourceLayerId == 20)
        let runtime = try EngramRuntime(modelDirectory: bundle, config: config)
        #expect(runtime.layerIds == [1, 14])
        #expect(runtime.aux.cols == 24)
    }

    @Test("a V4.1 bundle without engram tables fails at resolve time")
    func missingEngramFailsLoudly() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("engram-missing-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: EngramError.self) {
            _ = try EngramRuntime.resolve(modelDirectory: dir, environment: [:])
        }
        // runtime.engram_dir in config.json is honored.
        try Data().write(to: dir.appendingPathComponent("engram_aux.safetensors"))
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("tables"), withIntermediateDirectories: true)
        try Data(#"{"runtime":{"engram_dir":"tables"}}"#.utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        let resolved = try EngramRuntime.resolve(modelDirectory: dir, environment: [:])
        #expect(resolved.engramDir.lastPathComponent == "tables")
    }
}

/// Deterministic test RNG.
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
