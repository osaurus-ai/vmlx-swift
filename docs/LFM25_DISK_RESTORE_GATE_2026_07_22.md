# LFM2.5 Disk Restore Gate — 2026-07-22

Status: **VERIFIED-LIVE for the scoped LFM2.5 MXFP8 disk-restore defect;
broader all-family runtime correctness remains PARTIAL.**

## Root cause

Normal Osaurus chats include tool schemas. `BatchEngine` used the display-name
pair `lfm2.5` + `mxfp8` to disable disk-backed restore and suppress the
tool-prompt seed boundary whenever tools were present. LFM therefore wrote
typed format-v2 checkpoints but did not fetch them in the ordinary app path.
The cache topology itself was not the failing condition.

The patch removes the LFM display-name gates from both the solo iterator path
and the batched coordinator path. The topology-aware restore code remains the
owner of restore safety. The explicit `TokenIterator` opt-out is retained for
callers with a measured safety policy. The existing unproven Gemma-4 MXFP4
seed-boundary exception is intentionally unchanged and was not tested here.

Source revision before patch: `bbbf49e090449bb42f6cde8f50b6f230e3578aec`.

## Focused source verification

Command:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter lfmToolPromptsUseDiskRestore
```

Result: one test in one suite passed in 0.021 seconds after rebuilding
`BatchEngine.swift`. The contract asserts that the LFM/MXFP8 denylist and its
BatchEngine propagation are absent, while the Gemma-4 MXFP4 exception and the
public iterator opt-out remain.

The wider topology suite was attempted earlier but aborted while constructing
unrelated Metal-backed cache objects because the command-line test process
could not load the default metallib. It is not counted as a pass.

## Isolated Release Osaurus proof

App bundle: `com.dinoki.osaurus.prefillqueueproof20260722`.

Visible settings: Prefix Cache On, GPU/Paged Cache Off, Disk Cache On, codec
`Engine Selected`, SSM re-derive On. The UI reported all changes saved.

Model: LFM2.5 8B A1B MXFP8.

| Scenario | Runtime evidence | Visible result |
|---|---|---|
| New chat | Disk hit boundary `1747/1768`; `MambaCache:18`, `KVCacheSimple:6` | Exact sentence; TTFT 0.41s; 270.0 tok/s |
| Grounded follow-up | Disk hit boundary `1793/1815` | `Rayleigh scattering`; TTFT 0.32s; 221.5 tok/s |
| Another new chat | Disk hit boundary `1765/1777` | `oxygen`; TTFT 0.49s; 239.1 tok/s |
| Relaunch/startup prefix | Disk hit boundary `1747/1750` | Restore occurred before remaining prefill |

These rows prove that ordinary tool-bearing Osaurus requests now attempt and
use partial SSD restore with paged RAM cache disabled. They do not establish
TurboQuant behavior, media paths, or every model family's output correctness.

## Representative non-regression evidence

The same isolated Release app, with the same visible cache settings, completed:

- Bonsai 27B hybrid Qwen: exact multi-turn, warm-up race, 13,596-token partial
  restore, and restart restore with recurrent companion state.
- Gemma 4 12B QAT JANG_4M: 8 full-KV plus 40 rotating-SWA layers; exact
  two-turn output with no reasoning leak while Thinking was Off.
- Laguna S2.1 JANG_2L: 12 full-attention plus 36 rotating layers; cross-chat
  disk restore and exact output at 44.2 tok/s.
- VibeThinker 3B MXFP8: full-KV disk hits worked, but the second turn exposed
  visible deliberation. Cache-path PASS; model coherence FAIL.
- ZAYA1 VL 8B JANGTQ4: 40 CCA layers restored from disk, but the generic
  Assistant surface hallucinated after search. Cache-path evidence only.
- Nemotron Audex 2B: load blocked as unsupported by the pinned runtime.

Therefore this change is scoped to the proven LFM restore defect. It must not
be described as universal family correctness or release readiness.
