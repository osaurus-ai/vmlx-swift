# SSD L2 new-chat partial-restore gate — 2026-07-21

## Status

`VERIFIED-LIVE — FEB35555 N-1 HYBRID SEED; BROADER FAMILY MATRIX PARTIAL`

## Current `feb35555` isolated Release proof

The exact vMLX head `feb35555900398dc638c82a3e13e98f8b1adbf41` was
consumed by all four package-pin surfaces of a fresh Release Osaurus build.
The app was ad-hoc signed as
`com.dinoki.osaurus.ssdwarmfeb355proof20260721`; its executable SHA-256 was
`488b2ce7106cb8e85bb3f27e69d4db7941abb6f977bc3174e09131169d22a3ea`.
It was operated through Computer Use with an isolated test root and the local
bundles under `/Users/eric/models`.

Settings visibly showed Prefix Cache On, GPU/paged cache Off, Disk Cache On,
codec `Engine Selected`, SSM re-derive On, and Thinking Off. The Disk toggle
was changed and saved in the UI for the negative control, then restored On and
saved before the topology regressions.

| Model / request | Request-attributed cache trace | Visible result |
| --- | --- | --- |
| Qwen AgentWorld 35B A3B MXFP8, first new-chat warm-up | disk boundary 2,992, one token remaining, 60 recurrent states | warm-up first delta 0.36 s |
| Same Qwen bundle after quit/relaunch | disk boundary 2,992, one token remaining, 60 recurrent states | warm-up first delta 0.21 s; `Au`, TTFT 0.49 s, 47.8 tok/s |
| Same Qwen request with Disk Cache Off | `MISS all tiers`; UI visibly advanced through `Prefill 1024/3010` | `Au`, TTFT 1.89 s, 47.9 tok/s |
| Gemma 4 12B QAT JANG_4M, later new chat | disk boundary 1,629, four tokens remaining, `ssm=-1` | warm-up 0.28 s; `Au`, TTFT 0.41 s, 35.3 tok/s |
| Bonsai 27B Ternary JANG, later new chat | disk boundary 2,922, one token remaining, 96 recurrent states | warm-up 0.20 s; prior visible `Au`, TTFT 0.50 s, 30.7 tok/s |

This closes the emergency N-1 safe-seed defect on the real Qwen 3.5 hybrid
path and shows that the scoped change does not replace rotating-SWA semantics
or lose Bonsai companion state. It does not close the separate TurboQuant,
media, paged-hot-tier, quota-pressure, or all-family campaign.

This gate tracks a report that Bonsai appears to prefill from zero in each new
Osaurus chat while paged RAM cache is **Off** and SSD/Disk L2 is **On**. A valid
partial hit does not require identical prompts: the coordinator must restore
the longest content-verified stored token prefix and prefill only the unmatched
suffix. Aggregate counters alone are not proof for a specific request.

## Current Release-app evidence

The isolated Release app at
`/private/tmp/osaurus-applescript16-main-release-derived-20260721/Build/Products/Release/osaurus.app`
(bundle `com.dinoki.osaurus.applescript16mainproof20260721`, executable SHA-256
`14fe951e7befc7c26329e3140c0525117a12aa7a23ddc2fc26b0f6eb9328dab1`)
was operated through the UI with the exact local
`/Users/eric/models/dealign.ai/Bonsai-27b-Ternary-JANG-CRACK` bundle. Settings
visibly showed Prefix On, Paged RAM Off, Disk L2 On, TurboQuant not enabled,
SSM re-derive On, and all changes saved. Thinking was visibly Off.

Observed with vMLX `a37e09d2e4304e3eaa0836b4cb1941da86bcaeb7`:

| Row | Runtime trace | Visible result |
| --- | --- | --- |
| First blank new chat | Disk miss for 3,897 tokens; 5.34 s | Cold baseline. |
| First `hey` | Disk hit at 3,897, 71 tokens remaining | Coherent answer; 0.80 s TTFT, 31.4 tok/s. |
| Second blank new chat | Disk hit at 3,894, 3 tokens remaining | 0.20 s warm-up. |
| Repeated `hey` | Disk hit at 3,961, 7 tokens remaining | Coherent answer; 0.61 s TTFT, 31.4 tok/s. |

This is live proof that disk-only partial restore can work for Bonsai. It does
not close the reporter's 8,741-token row or the cross-session stable-prefix
case below.

## Root cause isolated in current source

`canonicalChatCacheBoundaries` first renders only leading system/developer
messages plus tools with `add_generation_prompt=false` to derive the reusable
new-chat boundary. Bonsai's Qwen 3.5 template rejects that legal boundary probe
with `No user query found in messages.` The helper therefore returned
`stable=[]`; current live traces confirmed that exact empty value.

The full warm-up boundary can still hit when subsequent new chats have the
same rendered prefix. It cannot provide a shorter independent system/tool
checkpoint when legitimate chat/session context changes before the stored
full boundary. That explains why a user can see both real disk-hit counters
and an apparently cold new-chat transition.

The source candidate in this branch remains model-name agnostic and fails
closed:

1. Prefer the existing system/tools-only exact-prefix render.
2. Only if that render fails, append two different synthetic user probes and
   render both with the model's own template and active context.
3. Retain only the token prefix shared by both probes and the actual request.
4. Require the boundary to be nonzero and strictly before every rendered
   sequence ends. Any render failure, no divergence, or mismatch returns no
   stable boundary.

No user content, generated text, sampler override, model-name rule, or forced
reasoning state is stored by this fallback. The probes derive a boundary only;
they are never submitted to the model.

## Separate Osaurus telemetry defect

vMLX emits `.cacheLookup(completed=0,total=N)` before SSD lookup and then a
`.cacheRestore(completed=matched,total=N)` event on a hit. The current Osaurus
badge labels every stage `Prefill`, so the first frame appears as
`Prefill 0/N` even when a disk hit follows. The supplied screenshot is
therefore ambiguous, not proof of a cold transformer prefill. Osaurus must show
cache lookup, restore, and actual prefill as distinct stages in the rebuilt UI.

## Required current-source live acceptance

Do not promote this branch from `PARTIAL` until a newly pinned, isolated
Release Osaurus app proves all of the following through visible user actions:

1. Paged RAM Off, Disk L2 On, TurboQuant Off, settings saved.
2. The exact local Bonsai bundle and Thinking Off.
3. Current traces show a nonempty `stable=[...]` boundary for the Qwen
   user-required template.
4. A cold chat persists that stable boundary.
5. A new chat with different legitimate history/context restores that shorter
   disk boundary and prefills only the suffix.
6. The UI first says it is checking cache, then credits restored tokens, and
   labels only remaining transformer work as prefill.
7. TTFT is materially below the matched cold control, output is coherent and
   non-looping, and hybrid SSM companion restore/re-derive telemetry is valid.
8. Disk L2 Off returns to the cold behavior; changing a cache-affecting model
   or reasoning configuration intentionally misses.

After Bonsai, the same contract still needs topology-specific live rows for
Ornith/Qwen 3.5, Qwen VL media salts, Gemma mixed rotating-SWA/full attention,
full-KV models, and the other supported families. TurboQuant-on rows remain a
separate opt-in matrix; JANGTQ weights are not TurboQuant KV cache encoding.

## Current pinned-app partial-restore proof and newly isolated write stall

The next isolated Release app pinned vMLX
`ad786d77952d368c84f8cf1800eef77184308ee7` and was operated through the
Osaurus UI with the same local Bonsai bundle, Thinking Off, Prefix On, Paged
RAM Off, Disk L2 On, TurboQuant not selected, and SSM re-derive On.

The new stable-boundary code produced `stable=[3798]` instead of `stable=[]`.
The UI and request-local traces recorded:

| Request | Disk result | Visible result |
| --- | --- | --- |
| Cold prompt A | miss/store | coherent three-bullet answer; 1.41 s TTFT, 40.7 tok/s |
| Same-chat A2 | hit 3,958 / 3,981; 23-token suffix | coherent `Anthocyanins.`; 0.82 s TTFT, 24.9 tok/s |
| Different blank new chat | hit 3,795 with 3 remaining | sentinel-only warm-up completed in 0.29 s |
| Different new-chat prompt B | hit 3,798 / 3,816; 18-token suffix | coherent `Au`; 0.58 s TTFT, 27.9 tok/s |
| Quit/relaunch warm-up | disk hit 3,795 with 3 remaining | persisted restore, not paged/process-local reuse |
| Same prompt with Disk L2 Off | `MISS all tiers` for 3,816 | coherent `Au`; 9.16 s TTFT, 29.4 tok/s |

This proves the read/partial-restore path and the UI toggle are effective. It
also exposed a separate storage defect: the short session created 18 complete
KV safetensor entries totaling 8.7 GB. Adjacent entries differed by only 1–7
tokens but each occupied roughly 406–418 MB. A one-token answer surfaced its
first token in under a second, then kept the stream/Metal lease open for about
16 seconds while three full hybrid snapshots were serialized. Its automatic
post-turn warm-up spent about 12 seconds serializing two more snapshots.

The disk tier is currently checkpoint/snapshot based; it is not
block-deduplicated storage. Partial lookup is real, but it selects the longest
content-addressed checkpoint and then prefills the suffix. The UI must not
describe these files as independently deduplicated disk blocks.

## Scoped hybrid storage candidate

Source comments and tests already define the generation-suffix-stripped
boundary as the only boundary the next path-dependent hybrid chat turn is
guaranteed to contain. The candidate therefore changes only requests where a
hybrid coordinator has that processor-proven boundary:

- keep stable system/tool checkpoints;
- publish the canonical stripped checkpoint;
- do not also serialize the unusable exact-prompt, non-stable history, or
  generated/post-answer full snapshots;
- retain the existing full prompt/post-answer policy for dense, rotating-SWA,
  media-unsafe, raw, and non-chat paths.

`HybridStripBoundaryPrefillTests` now executes 7 tests with 0 failures after
the package metallib is installed beside the test binary. The added test
asserts one disk-store attempt and a real partial restore from that canonical
boundary. This is source/test evidence only. A newly pinned Release Osaurus
build still has to show the lower store count, lower end-of-stream/warm-up
latency, partial disk hit, coherent answer, and correct settings through the
live UI before the policy can be called verified.
