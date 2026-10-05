#!/usr/bin/env python3
"""Shift formants without moving pitch — the dial that makes DIFFERENT voices.

Pitch shifting alone gives one voice at several heights: same character, higher
or lower. What separates two characters is the vocal-tract length, which shows
up as the position of the spectral envelope (the formants), independent of the
pitch the speaker is using. A big animal has low formants and can still speak
at a high pitch; a tiny one has high formants.

Method, per STFT frame:
  1. take the log magnitude spectrum
  2. cepstrally smooth it (keep only low quefrency) to get the ENVELOPE,
     separated from the harmonic comb that carries pitch
  3. resample the envelope's frequency axis by `shift`
  4. multiply the original spectrum by (warped envelope / original envelope)

Harmonic positions are untouched, so pitch is unchanged; only the resonances
move. Phases are kept and the hop is unchanged, so there is no time-stretching
and none of the phase-mismatch stitching that creates a doubled voice.

Usage: formant.py IN.wav OUT.wav SHIFT   (SHIFT > 1 = smaller/brighter)
"""
import subprocess, sys, numpy as np

N, HOP, LIFTER = 1024, 256, 40


def read(path):
    sr = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr


def write(path, x, sr):
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
         "-i", "-", "-c:a", "pcm_f32le", path],
        input=x.tobytes(), check=True)


def envelope(logmag):
    """Cepstrally smoothed spectral envelope of one frame."""
    cep = np.fft.irfft(logmag, n=N)
    cep[LIFTER:-LIFTER] = 0.0
    return np.fft.rfft(cep, n=N).real


def formant_shift(x, sr, shift):
    win = np.hanning(N + 1)[:N]
    pad = np.concatenate([np.zeros(N), x, np.zeros(N * 2)])
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    freqs = np.arange(N // 2 + 1)
    for start in range(0, len(pad) - N, HOP):
        frame = pad[start:start + N] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        if mag.max() < 1e-8:
            continue
        logmag = np.log(mag + 1e-10)
        env = envelope(logmag)
        # Sample the envelope at freq/shift: a formant at f moves to f*shift.
        warped = np.interp(freqs / shift, freqs, env, left=env[0], right=env[-1])
        gain = np.exp(np.clip(warped - env, -6.0, 6.0))
        rec = np.fft.irfft(spec * gain, n=N) * win
        out[start:start + N] += rec
        norm[start:start + N] += win ** 2
    out = out[N:N + len(x)] / np.maximum(norm[N:N + len(x)], 1e-8)
    peak = np.abs(out).max()
    if peak > 0:
        out = out * (np.abs(x).max() / peak)
    return out


if __name__ == "__main__":
    src, dst, shift = sys.argv[1], sys.argv[2], float(sys.argv[3])
    x, sr = read(src)
    write(dst, formant_shift(x, sr, shift), sr)
