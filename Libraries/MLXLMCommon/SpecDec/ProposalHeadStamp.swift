//
//  ProposalHeadStamp.swift
//  MLXLMCommon
//
//  One-time per-bundle verdict cache for the low-bit DRAFT-ONLY lm_head
//  ("proposal head"): `vmlx_mtp_proposal_head.json` at the bundle root,
//  shared contract with the Python engine and the jang converter.
//
//  The stamp records WHETHER the bundle's lm_head layout supports a cheaper
//  proposal head and at what bits — it carries no tensors. When eligible,
//  the engine rebuilds the q4 proposal copy in memory at every load
//  (dequantize the calibrated head → requantize; ~166 ms measured) and uses
//  it ONLY to sample draft proposals. Target verification always runs the
//  checkpoint-owned full head, so emitted tokens stay exactly verified —
//  the copy can only shape proposals, never outputs.
//
//  Contract rules (settled 2026-09-04):
//  - `source` is the validity key: the lm_head layout the verdict was
//    derived from. At load it is compared against the ACTUAL loaded head —
//    match → the stamp is authoritative and never rewritten; mismatch
//    (bundle requantized) → treat as absent, re-derive, write fresh.
//  - Eligibility is a pure function of the head (valid because JANG bundles
//    are AWQ + imatrix/GPTQ calibrated at conversion):
//      untied + affine + q8/g64          → eligible, proposal_bits 4
//      bits ≤ 6                          → ineligible "native_head_already_low_bit"
//      tied embeddings                   → ineligible "tied_embeddings"
//      anything else                     → ineligible "unmeasured_layout_q<B>_g<G>"
//  - Fail-open everywhere: unreadable/corrupt/unknown-version stamp →
//    ignore and re-derive; write failure (read-only volume) → log and
//    continue with the in-process verdict. The stamp must never block or
//    delay a launch.
//  - Writes are atomic: temp file + rename in the bundle dir. Concurrent
//    loads race benignly: both derive identical content, and rename leaves
//    a valid file either way.
//
//  THE MISSTAMP LESSON (2026-09-04, CRACK-JANG2L): the layout MUST be read
//  off the loaded QuantizedLinear module, NEVER from config. 2L was first
//  stamped ineligible (q6) because the stamper read the bundle's top-level
//  quantization tier default instead of the per-module lm_head override
//  (actually q8/g64). Config defaults lie; the loaded weights don't. The
//  strict `source` validity key makes the stamp self-healing: a runtime
//  following this contract silently auto-corrects such a misstamp on first
//  load — mismatched source → re-derive from the real head → overwrite with
//  measured truth. The stamp is a cache of a pure function, never an
//  authority over the weights. (Converter-side reference implementation:
//  jang-tools/jang_tools/stamp_mtp_proposal_head.py, which refuses to stamp
//  when config and shard packed widths disagree. Contract + lesson also in
//  the private wiki.)
//

import CryptoKit
import Foundation
import MLX
import MLXNN
import os.log

private let stampLog = Logger(subsystem: "com.vmlx", category: "ProposalHeadStamp")

/// The lm_head layout a proposal-head verdict was derived from. This is the
/// stamp's validity key: any difference between the stamped source and the
/// actually-loaded head voids the stamp.
public struct ProposalHeadSourceLayout: Codable, Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
    /// Quantization mode raw value ("affine", …).
    public let mode: String
    /// True when the head shares weights with the input embedding. A tied
    /// head cannot get a divergent proposal copy without also perturbing
    /// embeddings, so tied layouts are never eligible.
    public let tied: Bool

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
        case tied
    }

    public init(bits: Int, groupSize: Int, mode: String, tied: Bool) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.tied = tied
    }
}

/// The verdict the eligibility rule produces for a head layout.
public enum ProposalHeadVerdict: Equatable, Sendable {
    case eligible(proposalBits: Int)
    case ineligible(reason: String)

    /// The contract's pure eligibility function. Keyed on the ACTUAL loaded
    /// head, never on bundle names — the shipped CRACK-2L is q6/g64
    /// (ineligible) while a local JANG_2L lineage declares q8/g64
    /// (eligible); both resolve correctly from the same rule.
    public static func derive(from source: ProposalHeadSourceLayout) -> ProposalHeadVerdict {
        if source.tied {
            return .ineligible(reason: "tied_embeddings")
        }
        if source.bits <= 6 {
            return .ineligible(reason: "native_head_already_low_bit")
        }
        if source.mode == QuantizationMode.affine.rawValue,
            source.bits == 8, source.groupSize == 64
        {
            return .eligible(proposalBits: 4)
        }
        return .ineligible(
            reason: "unmeasured_layout_q\(source.bits)_g\(source.groupSize)")
    }
}

/// A calibrated draft-head sidecar the stamp may declare: pre-quantized
/// q4/g64 lm_head tensors produced by the converter's imatrix-weighted
/// refit (measurably better proposal quality than the runtime's RTN
/// requantize). The artifact is OPTIONAL sugar on top of the verdict:
/// runtimes unaware of it — and any load where the file is missing, the
/// sha256 mismatches, or the tensors don't fit the head — keep the RTN
/// rebuild. Fail-open, silently.
public struct ProposalHeadDraftArtifact: Codable, Equatable, Sendable {
    /// Bundle-relative path of the safetensors sidecar. The final layout
    /// puts it in a SUBFOLDER (`mtp_draft/…`) so the bundle root stays free
    /// of tensor files that are not model shards.
    public let file: String
    /// Hex sha256 of the sidecar file; a mismatch voids the artifact.
    public let sha256: String
    public let bits: Int?
    public let groupSize: Int?
    public let mode: String?
    public let acceptanceValidated: Bool?

    enum CodingKeys: String, CodingKey {
        case file, sha256, bits
        case groupSize = "group_size"
        case mode
        case acceptanceValidated = "acceptance_validated"
    }

    public init(
        file: String, sha256: String, bits: Int? = nil, groupSize: Int? = nil,
        mode: String? = nil, acceptanceValidated: Bool? = nil
    ) {
        self.file = file
        self.sha256 = sha256
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.acceptanceValidated = acceptanceValidated
    }
}

/// The on-disk stamp. Tolerant decode: any structural surprise reads as
/// "no stamp" so the load re-derives instead of failing.
public struct ProposalHeadStamp: Codable, Equatable, Sendable {
    public static let fileName = "vmlx_mtp_proposal_head.json"
    public static let currentVersion = 1

    public let version: Int
    public let family: String
    public let source: ProposalHeadSourceLayout
    public let eligible: Bool
    public let proposalBits: Int?
    public let reason: String?
    public let basis: String?
    /// Converter-shipped calibrated draft sidecar (see
    /// ``ProposalHeadDraftArtifact``). Runtime-derived stamps never carry
    /// one — only the converter can attest a calibrated artifact.
    public let draftArtifact: ProposalHeadDraftArtifact?

    enum CodingKeys: String, CodingKey {
        case version, family, source, eligible
        case proposalBits = "proposal_bits"
        case reason, basis
        case draftArtifact = "draft_artifact"
    }

    public init(
        version: Int = ProposalHeadStamp.currentVersion,
        family: String,
        source: ProposalHeadSourceLayout,
        verdict: ProposalHeadVerdict,
        basis: String?,
        draftArtifact: ProposalHeadDraftArtifact? = nil
    ) {
        self.version = version
        self.family = family
        self.source = source
        switch verdict {
        case .eligible(let bits):
            self.eligible = true
            self.proposalBits = bits
            self.reason = nil
        case .ineligible(let why):
            self.eligible = false
            self.proposalBits = nil
            self.reason = why
        }
        self.basis = basis
        self.draftArtifact = draftArtifact
    }

    public var verdict: ProposalHeadVerdict {
        if eligible, let proposalBits {
            return .eligible(proposalBits: proposalBits)
        }
        return .ineligible(reason: reason ?? "unspecified")
    }

    // MARK: - IO (fail-open)

    /// Read the stamp from a bundle directory. Returns nil — never throws —
    /// for a missing file, unparseable JSON, or an unknown `version`; the
    /// caller then re-derives. Symlinked bundle dirs are resolved first
    /// (mlxstudio ships symlinked model directories).
    public static func load(fromBundleAt directory: URL) -> ProposalHeadStamp? {
        let url = directory.resolvingSymlinksInPath()
            .appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let stamp = try? JSONDecoder().decode(ProposalHeadStamp.self, from: data)
        else {
            stampLog.notice(
                "corrupt \(fileName, privacy: .public) in \(directory.lastPathComponent, privacy: .public) — ignoring, will re-derive"
            )
            return nil
        }
        guard stamp.version == currentVersion else {
            stampLog.notice(
                "\(fileName, privacy: .public) version \(stamp.version) unknown (current \(currentVersion)) — ignoring, will re-derive"
            )
            return nil
        }
        return stamp
    }

    /// Atomically write the stamp into the bundle directory (temp + rename).
    /// Failures (read-only volume, permissions) log and return false; the
    /// in-process verdict still applies — a stamp write must never gate the
    /// feature, let alone the load.
    @discardableResult
    public func write(toBundleAt directory: URL) -> Bool {
        let dir = directory.resolvingSymlinksInPath()
        let url = dir.appendingPathComponent(Self.fileName)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(self)
            let tmp = dir.appendingPathComponent(".\(Self.fileName).tmp-\(UUID().uuidString)")
            try data.write(to: tmp)
            // rename(2) semantics: atomic replace within the same volume.
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
            stampLog.info(
                "stamped \(directory.lastPathComponent, privacy: .public): eligible=\(self.eligible) \(self.reason ?? "proposal_bits=\(self.proposalBits ?? 0)", privacy: .public)"
            )
            return true
        } catch {
            stampLog.notice(
                "could not write \(Self.fileName, privacy: .public) in \(directory.lastPathComponent, privacy: .public): \(String(describing: error), privacy: .public) — continuing with in-process verdict"
            )
            return false
        }
    }
}

// MARK: - Model hook

/// Adopted by model families that participate in proposal-head drafting
/// (qwen4_exp, qwen3_5). Adoption IS the family gate: `ensure` does nothing
/// for models that don't conform, so no config sniffing happens here.
public protocol NativeMTPProposalHeadInstalling: AnyObject {
    /// Stamp `family` value ("qwen4_exp" / "qwen3_5").
    var nativeMTPProposalHeadFamily: String { get }
    /// The ACTUAL loaded lm_head layout, or nil when the model has no
    /// standalone quantized head (tied/fp heads report `tied`/nil per family).
    var nativeMTPProposalHeadSourceLayout: ProposalHeadSourceLayout? { get }
    /// Build the low-bit proposal copy and route the DRAFT branch through
    /// it. When `calibratedDraft` is non-nil it carries the sha-verified
    /// converter sidecar tensors — install those verbatim; otherwise rebuild
    /// by RTN (dequantize the loaded head → requantize). Verify paths must
    /// be untouched either way.
    func installNativeMTPProposalHead(bits: Int, calibratedDraft: ProposalHeadCalibratedDraft?)
}

public enum ProposalHeadBootstrap {
    /// One-shot per load: derive-or-read the stamp against the actual head,
    /// persist it when absent/stale, and install the proposal copy when
    /// eligible. Every failure path is log-and-continue; this function can
    /// not throw and must never meaningfully delay the load beyond the
    /// measured ~166 ms eligible-path rebuild.
    ///
    /// `isCalibratedBundle` gates DERIVATION, not honoring: the eligibility
    /// rule's whole premise is "valid because JANG bundles are AWQ+imatrix
    /// calibrated at conversion". A plain mlx_lm benchmark quant can carry a
    /// q8/g64 head without having earned an eligible verdict — the runtime
    /// must not mint one (the speed-audit packs are left unstamped on
    /// purpose). An EXISTING source-matching stamp is still honored either
    /// way: placing one in a non-JANG bundle is an explicit human action.
    public static func ensure(model: Any, modelDirectory: URL, isCalibratedBundle: Bool) {
        guard let installing = model as? NativeMTPProposalHeadInstalling else { return }
        guard let actual = installing.nativeMTPProposalHeadSourceLayout else {
            stampLog.info(
                "proposal head: \(modelDirectory.lastPathComponent, privacy: .public) has no standalone quantized lm_head — skipping"
            )
            return
        }

        let verdict: ProposalHeadVerdict
        var honoredStamp: ProposalHeadStamp?
        if let stamp = ProposalHeadStamp.load(fromBundleAt: modelDirectory),
            stamp.source == actual
        {
            // Source matches the loaded head: the stamp is authoritative and
            // is never rewritten (a jang-tools calibrated verdict must not be
            // clobbered by a runtime re-derivation).
            verdict = stamp.verdict
            honoredStamp = stamp
        } else if isCalibratedBundle {
            verdict = ProposalHeadVerdict.derive(from: actual)
            let fresh = ProposalHeadStamp(
                family: installing.nativeMTPProposalHeadFamily,
                source: actual,
                verdict: verdict,
                basis:
                    "derived at load by vmlx-swift from the actual lm_head layout (contract 2026-09-04)"
            )
            fresh.write(toBundleAt: modelDirectory)
        } else {
            stampLog.info(
                "proposal head: \(modelDirectory.lastPathComponent, privacy: .public) is not a calibrated JANG bundle — not stamping, drafting stays on the full head"
            )
            return
        }

        switch verdict {
        case .eligible(let proposalBits):
            let started = Date()
            // The calibrated sidecar is only ever declared by an HONORED
            // stamp — a runtime re-derivation cannot attest calibration.
            var calibratedDraft: ProposalHeadCalibratedDraft?
            if let artifact = honoredStamp?.draftArtifact {
                calibratedDraft = ProposalHeadCalibratedDraft.loadVerified(
                    artifact: artifact,
                    bundleDirectory: modelDirectory,
                    expectedBits: proposalBits,
                    expectedGroupSize: actual.groupSize)
            }
            installing.installNativeMTPProposalHead(
                bits: proposalBits, calibratedDraft: calibratedDraft)
            let ms = Int(Date().timeIntervalSince(started) * 1000)
            let origin = calibratedDraft != nil ? "sidecar, sha-verified" : "RTN rebuild"
            stampLog.info(
                "proposal head installed for \(modelDirectory.lastPathComponent, privacy: .public): q\(proposalBits)/g\(actual.groupSize) draft-only copy ready in \(ms) ms (origin: \(origin, privacy: .public))"
            )
        case .ineligible(let reason):
            stampLog.info(
                "proposal head not used for \(modelDirectory.lastPathComponent, privacy: .public): \(reason, privacy: .public)"
            )
        }
    }
}

// MARK: - Calibrated draft sidecar

/// Pre-quantized draft-head tensors loaded from a stamp's `draft_artifact`
/// sidecar, sha256-verified. Consumed by the family install hook in place
/// of the RTN dequant→requant rebuild.
public struct ProposalHeadCalibratedDraft {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray?
    public let bits: Int
    public let groupSize: Int

    /// Load + verify a declared sidecar. ANY failure — traversal-unsafe or
    /// missing file, sha256 mismatch, unreadable tensors, layout fields that
    /// contradict the verdict — returns nil so the caller silently keeps the
    /// RTN rebuild. This function must never throw and never block a load on
    /// anything but the one-pass hash of the artifact itself.
    public static func loadVerified(
        artifact: ProposalHeadDraftArtifact,
        bundleDirectory: URL,
        expectedBits: Int,
        expectedGroupSize: Int
    ) -> ProposalHeadCalibratedDraft? {
        func reject(_ why: String) -> ProposalHeadCalibratedDraft? {
            stampLog.info(
                "draft artifact rejected (\(why, privacy: .public)) — keeping RTN rebuild")
            return nil
        }
        // Declared layout, when present, must agree with the verdict.
        if let bits = artifact.bits, bits != expectedBits {
            return reject("declared bits \(bits) != verdict q\(expectedBits)")
        }
        if let group = artifact.groupSize, group != expectedGroupSize {
            return reject("declared group \(group) != head group \(expectedGroupSize)")
        }
        if let mode = artifact.mode, mode != QuantizationMode.affine.rawValue {
            return reject("declared mode \(mode) unsupported")
        }
        // Bundle-relative path only; the stamp must not become a probe.
        guard !artifact.file.hasPrefix("/"), !artifact.file.contains("..") else {
            return reject("unsafe artifact path")
        }
        let url = bundleDirectory.resolvingSymlinksInPath()
            .appendingPathComponent(artifact.file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return reject("missing \(artifact.file)")
        }
        guard let digest = sha256Hex(of: url) else {
            return reject("unreadable \(artifact.file)")
        }
        guard digest == artifact.sha256.lowercased() else {
            return reject("sha256 mismatch for \(artifact.file)")
        }
        guard let arrays = try? MLX.loadArrays(url: url) else {
            return reject("unparseable safetensors \(artifact.file)")
        }
        guard let weight = arrays["lm_head.weight"],
            let scales = arrays["lm_head.scales"]
        else {
            return reject("missing lm_head.weight/scales tensors")
        }
        return ProposalHeadCalibratedDraft(
            weight: weight,
            scales: scales,
            biases: arrays["lm_head.biases"],
            bits: expectedBits,
            groupSize: expectedGroupSize)
    }

    /// Streaming sha256 of a file (8 MiB chunks), lowercase hex. nil when
    /// the file cannot be read.
    static func sha256Hex(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: 8 * 1024 * 1024)
            } catch {
                return nil
            }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
