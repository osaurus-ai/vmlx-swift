#!/usr/bin/env python3
"""Objective voice-quality QC: detect the 'two voices at once' artifact.

The complaint "sounds like two voices at once / alien" has a specific
measurable signature. Fixed-hop overlap-add stitches frames at points where
the waveform's pitch periods do not line up. That does two things a listener
hears as a second voice:

  1. periodic amplitude modulation locked to the SYNTHESIS HOP, which puts
     sidebands around every harmonic at +-(hop rate) Hz, and
  2. smeared harmonic structure, i.e. a drop in harmonic-to-noise ratio.

Both are measurable without ears. This reports:
  f0          - median pitch (autocorrelation), tells us the shift landed
  hnr_db      - harmonic-to-noise ratio; artifacts push this DOWN
  am_ratio    - strength of amplitude modulation in the 20-120 Hz band that
                fixed-hop OLA creates, relative to the envelope's DC. The
                'two voices' percept lives here.
  am_peak_hz  - where that modulation sits; a hand-rolled OLA parks it at the
                hop rate, a clean shift has no such peak.
"""
import subprocess, sys, numpy as np


def read_wav(path):
    """Decode via ffmpeg — these WAVs are IEEE float32, which `wave` rejects."""
    sr = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr


def f0_track(x, sr, frame=1024, hop=256, fmin=60, fmax=500):
    """Per-frame autocorrelation pitch, voiced frames only."""
    out = []
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i in range(0, len(x) - frame, hop):
        f = x[i:i + frame]
        if np.sqrt(np.mean(f ** 2)) < 0.01:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if ac[k] > 0.3:                      # voiced
            out.append((sr / k, ac[k]))
    return out


def hnr_db(track):
    """Autocorrelation peak height -> harmonic-to-noise ratio, Boersma style."""
    if not track:
        return float("nan")
    r = np.clip(np.array([t[1] for t in track]), 1e-6, 0.999999)
    return float(np.median(10 * np.log10(r / (1 - r))))


def am_artifact(x, sr):
    """Amplitude-modulation energy in the band fixed-hop OLA lands in."""
    env = np.abs(x)
    # Smooth to an envelope at ~1 kHz effective rate, then look at its spectrum.
    k = 16
    env = np.convolve(env, np.ones(k) / k, mode="same")
    env = env[::k]
    esr = sr / k
    env = env - env.mean()
    if len(env) < 64:
        return 0.0, 0.0
    win = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * win))
    freqs = np.fft.rfftfreq(len(env), 1 / esr)
    band = (freqs >= 20) & (freqs <= 120)
    if not band.any():
        return 0.0, 0.0
    total = np.sqrt(np.mean(np.abs(x) ** 2)) * len(env)
    peak = float(spec[band].max())
    peak_hz = float(freqs[band][int(np.argmax(spec[band]))])
    return peak / (total + 1e-9), peak_hz


def report(path):
    x, sr = read_wav(path)
    tr = f0_track(x, sr)
    f0 = float(np.median([t[0] for t in tr])) if tr else float("nan")
    am, am_hz = am_artifact(x, sr)
    # Pitch dynamism, in semitones: how far the pitch actually MOVES. A flat
    # read sits near 1-2 st and reads as "person talking"; an animated
    # character read is 3+. Pitch HEIGHT is not what makes a character — this
    # is.
    if len(tr) > 2:
        semis = 12.0 * np.log2(np.array([t[0] for t in tr]) / f0)
        dyn = float(np.percentile(semis, 90) - np.percentile(semis, 10))
    else:
        dyn = 0.0
    return dict(name=path.split("/")[-1], sr=sr, dur=len(x) / sr, f0=f0,
                hnr=hnr_db(tr), voiced=len(tr), am=am, am_hz=am_hz, dyn=dyn)


if __name__ == "__main__":
    print(f"{'file':<30}{'dur':>6}{'f0 Hz':>8}{'dyn st':>8}{'HNR dB':>8}{'AM':>8}{'AM Hz':>8}")
    for p in sys.argv[1:]:
        r = report(p)
        print(f"{r['name']:<30}{r['dur']:>6.2f}{r['f0']:>8.1f}{r['dyn']:>8.1f}"
              f"{r['hnr']:>8.1f}{r['am']:>8.3f}{r['am_hz']:>8.1f}")
