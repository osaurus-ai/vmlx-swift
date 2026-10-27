// AudioIO.swift
// Native Swift live-audio I/O helpers for Nemotron-3-Nano-Omni's
// Parakeet (speech-to-text) path + AVSpeechSynthesizer-backed
// text-to-speech for the model's text output.
//
// Modality scope (be honest with consumers):
//   • Nemotron-3-Nano-Omni Parakeet → speech IN (audio → embeddings → LLM)
//   • The model itself produces TEXT tokens — there is NO neural
//     audio decoder/vocoder in this bundle. Voice OUT is therefore
//     done via Apple's AVSpeechSynthesizer (system TTS) which is
//     fully native, runs on every macOS/iOS device, and supports
//     a large set of languages. Drop in your own neural TTS if you
//     need higher-quality voice output.

import Foundation
#if canImport(AVFoundation)
    @preconcurrency import AVFoundation
#endif
import MLX

#if canImport(AVFoundation)
// MARK: - Live PCM buffer

/// Thread-safe 16 kHz mono PCM buffer for live voice handoff.
///
/// `snapshot()` returns the whole retained turn for the final Omni request.
/// `consumeAvailableSamples()` returns only samples appended since the
/// previous consume call, which lets a VAD/call-mode loop poll the same
/// recorder without losing the final full-turn waveform.
public final class NemotronHOmniLiveAudioBuffer: @unchecked Sendable {
    public struct Snapshot: Sendable {
        public let samples: [Float]
        public let sampleRate: Int

        public var durationSeconds: Double {
            guard sampleRate > 0 else { return 0 }
            return Double(samples.count) / Double(sampleRate)
        }
    }

    private let lock = NSLock()
    private let sampleRate: Int
    private var samples: [Float] = []
    private var consumeCursor = 0

    public init(sampleRate: Int = 16_000, reserveCapacity: Int = 0) {
        self.sampleRate = max(1, sampleRate)
        if reserveCapacity > 0 {
            samples.reserveCapacity(reserveCapacity)
        }
    }

    public var retainedSampleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return samples.count
    }

    public var durationSeconds: Double {
        Double(retainedSampleCount) / Double(sampleRate)
    }

    public func append(_ newSamples: [Float]) {
        guard !newSamples.isEmpty else { return }
        lock.lock()
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        let copy = samples
        lock.unlock()
        return Snapshot(samples: copy, sampleRate: sampleRate)
    }

    /// Return samples appended since the last consume call.
    public func consumeAvailableSamples() -> Snapshot {
        lock.lock()
        let start = min(consumeCursor, samples.count)
        let chunk = start < samples.count ? Array(samples[start ..< samples.count]) : []
        consumeCursor = samples.count
        lock.unlock()
        return Snapshot(samples: chunk, sampleRate: sampleRate)
    }

    public func resetConsumeCursor() {
        lock.lock()
        consumeCursor = 0
        lock.unlock()
    }

    public func clear(keepingCapacity: Bool = true) {
        lock.lock()
        samples.removeAll(keepingCapacity: keepingCapacity)
        consumeCursor = 0
        lock.unlock()
    }
}

private final class NemotronHOmniAudioConverterInput: @unchecked Sendable {
    private let lock = NSLock()
    private let buffer: AVAudioPCMBuffer
    private var consumed = false

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(
        _ outStatus: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioPCMBuffer? {
        lock.lock()
        defer { lock.unlock() }

        guard !consumed else {
            outStatus.pointee = .endOfStream
            return nil
        }

        consumed = true
        outStatus.pointee = .haveData
        return buffer
    }
}

// MARK: - Live mic capture

/// AVAudioEngine-backed mic recorder that captures 16 kHz mono Float32
/// PCM into an in-memory buffer. Format conversion happens via the
/// engine's tap; consumers get plug-and-play audio for the omni
/// `LMInput.audio` path without wrestling with AVAudioFormat.
///
/// Usage:
///   let rec = NemotronHOmniMicRecorder()
///   try rec.start()
///   // ... user speaks ...
///   let pcm = try rec.stop()       // [Float] @ 16 kHz mono
///
/// On iOS / sandboxed macOS the calling app must declare the
/// `NSMicrophoneUsageDescription` Info.plist key + obtain user
/// permission via `AVAudioApplication.requestRecordPermission(_:)`
/// before calling `.start()`. This class doesn't manage permissions —
/// that's the host app's responsibility per Apple's policy.
@available(macOS 14.0, iOS 17.0, *)
public final class NemotronHOmniMicRecorder: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let liveBuffer = NemotronHOmniLiveAudioBuffer(sampleRate: 16_000)
    private let queue = DispatchQueue(label: "nemotron.omni.mic.recorder")
    private let targetSampleRate: Double = 16_000
    private var recording = false

    /// True while the engine is capturing.
    public var isRecording: Bool {
        queue.sync { recording }
    }

    /// Retained samples for the active turn. This is the waveform to send
    /// with the final `UserInput.Audio.samples` / `.preEncoded` request.
    public func snapshot() -> NemotronHOmniLiveAudioBuffer.Snapshot {
        liveBuffer.snapshot()
    }

    /// Samples appended since the previous consume call. Poll this from a
    /// call-mode VAD loop to make endpoint decisions while continuing to
    /// retain the complete turn for the final Omni request.
    public func consumeAvailableSamples() -> NemotronHOmniLiveAudioBuffer.Snapshot {
        liveBuffer.consumeAvailableSamples()
    }

    public init() {}

    public func start() throws {
        try queue.sync {
            guard !recording else { return }
            liveBuffer.clear(keepingCapacity: true)
            let inputNode = engine.inputNode
            let inputFormat = inputNode.inputFormat(forBus: 0)
            // Tap at the input format; convert per-buffer below.
            inputNode.installTap(
                onBus: 0,
                bufferSize: 4096,
                format: inputFormat
            ) { [weak self] inBuffer, _ in
                guard let self else { return }
                self.handleBuffer(inBuffer, inputFormat: inputFormat)
            }
            try engine.start()
            recording = true
        }
    }

    /// Stop the engine and return the accumulated 16 kHz mono Float32 PCM.
    public func stop() throws -> [Float] {
        return queue.sync {
            guard recording else { return [] }
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            recording = false
            return liveBuffer.snapshot().samples
        }
    }

    /// Convert a captured buffer to 16 kHz mono Float32 and append.
    /// Done synchronously in the audio thread's tap closure, then appended
    /// through the live buffer's lock so stop/snapshot/consume can run from
    /// the UI or VAD loop without racing the audio callback.
    private func handleBuffer(
        _ inBuffer: AVAudioPCMBuffer, inputFormat: AVAudioFormat
    ) {
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false)
        else { return }
        guard let converter = AVAudioConverter(from: inputFormat, to: outFormat)
        else { return }

        let outFrames = AVAudioFrameCount(
            Double(inBuffer.frameLength) * targetSampleRate / inputFormat.sampleRate
                + 64)
        guard outFrames > 0,
            let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outFrames)
        else { return }

        let input = NemotronHOmniAudioConverterInput(inBuffer)
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) {
            _, outStatus in
            input.next(outStatus)
        }
        guard status != .error,
            let chData = outBuffer.floatChannelData
        else { return }
        let n = Int(outBuffer.frameLength)
        let bp = UnsafeBufferPointer(start: chData[0], count: n)
        liveBuffer.append(Array(bp))
    }
}

// MARK: - Voice output via AVSpeechSynthesizer (system TTS)

/// Minimal wrapper around `AVSpeechSynthesizer` for routing the omni
/// model's text response to system TTS. AVSpeechSynthesizer has voices
/// for 70+ languages and runs entirely on-device — no neural TTS
/// model required. For higher-quality output, swap in a different
/// `NemotronHOmniSpeaker` implementation.
///
/// NB: The Nemotron-3-Nano-Omni bundle has no audio decoder / vocoder.
/// Speech-out is system TTS — text comes out of the LLM, this turns
/// the text into audio. That's the "voice tools and connectivity"
/// surface this codebase can provide today; bringing your own neural
/// TTS is the higher-quality path.
@available(macOS 14.0, iOS 17.0, *)
public final class NemotronHOmniSpeaker: @unchecked Sendable {
    private let synth = AVSpeechSynthesizer()
    public var voiceLanguage: String

    public init(voiceLanguage: String = "en-US") {
        self.voiceLanguage = voiceLanguage
    }

    /// Speak `text` using the configured voice. Non-blocking — caller
    /// can check `isSpeaking` or `await stop()` to wait.
    public func speak(_ text: String, rate: Float = AVSpeechUtteranceDefaultSpeechRate) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: voiceLanguage)
            ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = rate
        synth.speak(utterance)
    }

    public var isSpeaking: Bool { synth.isSpeaking }

    public func stop() {
        synth.stopSpeaking(at: .immediate)
    }
}

// MARK: - Modality contract documentation
//
// What the omni bundle CAN do (verified 2026-04-29 across MXFP4 +
// JANGTQ4 + JANGTQ2):
//
//   text in       → text out  (multi-turn chat)
//   image in      → text out  (NVLM dynamic tiles, RADIO ViT)
//   image multi-  → text out  (cache reuse via MediaSalt)
//   video in      → text out  (T=2 channel-stack, RADIO video_embedder)
//   audio in      → text out  (Parakeet ASR + sound_projection;
//                              ↑ this file's MicRecorder is the live path,
//                              `nemotronOmniLoadAudioFile` is the file path)
//   text + audio  → text out  (mixed-modality chat: ask about audio in text)
//   reasoning toggle (enable_thinking=true|false) on every modality
//
// What the bundle CANNOT do (no decoder ships in the safetensors):
//
//   * → audio out  (no vocoder; use NemotronHOmniSpeaker → system TTS,
//                   or layer a neural TTS model on top of the LLM text)
//   * → image out  (no diffusion / image decoder)
//   * → video out  (likewise)
//
// If you need neural voice OUT, point an external TTS (e.g. Coqui XTTS,
// ElevenLabs API, F5-TTS) at the LLM's text stream. The audio IN pipeline
// that this file supports is the speech-recognition + multimodal-chat
// surface the bundle is actually trained for.
#endif
