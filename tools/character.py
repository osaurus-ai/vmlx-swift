#!/usr/bin/env python3
"""Turn a pitched-up human voice into a CREATURE voice.

Uniformly scaling pitch and formants gives a smaller human — which reads as a
child, not a character. Three things separate a cartoon creature from a small
person, and none of them is pitch:

  1. NON-UNIFORM formant warping. A uniform scale just shrinks the vocal tract.
     Real character voices move the low formants (which carry vowel colour and
     "size") differently from the high ones (which carry brightness and
     consonant edge). Warping them apart is what stops it sounding human.

  2. Roughness. Creature voices have grit — subharmonic energy and a slightly
     driven low band. A clean voice never sounds like an animal.

  3. Liveliness. A little vibrato plus envelope expansion gives the bouncy,
     animated delivery a flat read lacks.

Everything here works on the spectral envelope or in the time domain with
continuous interpolation, so nothing stitches frames at mismatched phase — the
66.8 Hz overlap-add artifact class cannot reappear.

Usage:
  character.py IN OUT --lowf 1.25 --highf 0.95 --growl 0.3 --vibrato 4.5,0.012
                      --nasal 0.0 --expand 1.3
"""
import argparse, subprocess, sys, numpy as np

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
         "-i", "-", "-c:a", "pcm_f32le", path], input=x.tobytes(), check=True)


def warp_axis(freqs, sr, lowf, highf, nasal):
    """Frequency map with DIFFERENT scaling for low and high formants.

    Below the pivot the axis is scaled by `lowf`, above it by `highf`, joined
    continuously. `nasal` adds a bump near 1 kHz, the resonance that makes a
    voice sound like it has a snout.
    """
    pivot = 1500.0
    hz = freqs * (sr / 2) / freqs[-1]
    out = np.where(hz <= pivot, hz / max(lowf, 1e-3),
                   pivot / max(lowf, 1e-3) + (hz - pivot) / max(highf, 1e-3))
    if nasal:
        out = out + nasal * 400.0 * np.exp(-((hz - 1000.0) ** 2) / (2 * 350.0 ** 2))
    return out * freqs[-1] / (sr / 2)


def envelope(logmag):
    cep = np.fft.irfft(logmag, n=N)
    cep[LIFTER:-LIFTER] = 0.0
    return np.fft.rfft(cep, n=N).real


def reshape(x, sr, lowf, highf, nasal):
    win = np.hanning(N + 1)[:N]
    pad = np.concatenate([np.zeros(N), x, np.zeros(N * 2)])
    out = np.zeros(len(pad)); norm = np.zeros(len(pad))
    freqs = np.arange(N // 2 + 1).astype(np.float64)
    src = warp_axis(freqs, sr, lowf, highf, nasal)
    for start in range(0, len(pad) - N, HOP):
        frame = pad[start:start + N] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        if mag.max() < 1e-8:
            continue
        env = envelope(np.log(mag + 1e-10))
        warped = np.interp(src, freqs, env, left=env[0], right=env[-1])
        gain = np.exp(np.clip(warped - env, -6.0, 6.0))
        rec = np.fft.irfft(spec * gain, n=N) * win
        out[start:start + N] += rec
        norm[start:start + N] += win ** 2
    y = out[N:N + len(x)] / np.maximum(norm[N:N + len(x)], 1e-8)
    return y


def vibrato(x, sr, rate, depth):
    """Sinusoidal pitch wobble via a fractional delay line.

    Continuous interpolation, so there is no frame stitching at all.
    """
    if rate <= 0 or depth <= 0:
        return x
    amp = depth / (2 * np.pi * rate) * sr          # samples
    base = amp + 2.0
    n = np.arange(len(x))
    d = base + amp * np.sin(2 * np.pi * rate * n / sr)
    pos = n - d
    idx = np.floor(pos).astype(int)
    frac = pos - idx
    ok = (idx >= 0) & (idx + 1 < len(x))
    y = np.zeros_like(x)
    y[ok] = x[idx[ok]] * (1 - frac[ok]) + x[idx[ok] + 1] * frac[ok]
    return y


def growl(x, sr, amount):
    """Drive the low band so it grows harmonics — grit, not distortion."""
    if amount <= 0:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    low = np.fft.irfft(spec * (f < 900.0), n=len(x))
    driven = np.tanh(low * (1.0 + 7.0 * amount))
    peak = np.abs(low).max()
    if peak > 0:
        driven = driven * (peak / max(np.abs(driven).max(), 1e-9))
    return x * (1.0 - 0.35 * amount) + driven * (0.75 * amount)


def expand(x, sr, factor):
    """Expand the amplitude envelope so delivery is punchy, not flat."""
    if factor <= 1.0:
        return x
    env = np.abs(x)
    k = int(sr * 0.02)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    m = env.max()
    if m <= 0:
        return x
    gain = (env / m) ** (factor - 1.0)
    gain = gain / max(gain.max(), 1e-9)
    return x * (0.35 + 0.65 * gain)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--lowf", type=float, default=1.0)
    ap.add_argument("--highf", type=float, default=1.0)
    ap.add_argument("--nasal", type=float, default=0.0)
    ap.add_argument("--growl", type=float, default=0.0)
    ap.add_argument("--vibrato", type=str, default="0,0")
    ap.add_argument("--expand", type=float, default=1.0)
    a = ap.parse_args()
    x, sr = read(a.src)
    y = reshape(x, sr, a.lowf, a.highf, a.nasal)
    rate, depth = (float(v) for v in a.vibrato.split(","))
    y = vibrato(y, sr, rate, depth)
    y = growl(y, sr, a.growl)
    y = expand(y, sr, a.expand)
    peak = np.abs(y).max()
    if peak > 0:
        y = y * (min(0.85, np.abs(x).max() * 1.6) / peak)
    write(a.dst, y, sr)
