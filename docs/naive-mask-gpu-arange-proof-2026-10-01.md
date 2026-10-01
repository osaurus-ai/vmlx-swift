# Naive allowed-mask GPU arange: source-separated proof

**Delivery source: model default ON, explicit `0` opt-out. This main-based default revision is UNBUILT/UNRUN.** Exact-source gates remain required before a qualified-default or merge claim. The preceding source-separated commit `48709b10` preserves default OFF.

Base is engine main `9d196112257929ce87320ba465c32d2c9be14d26` (`Preserve compiled cache boundaries and sliding-window positions (#540)`), verified by read-only HTTPS `git ls-remote` on2026-10-01. Core gitlink: `88e2211cc2265b453c47bc6004a7a79ac6d34ea3`. The dirty primary checkout and frozen native integration checkout were preserved.

## What changes

Naive allowed-mask construction formerly builds query/key positions with `MLXArray(Range<Int>)`. That constructor materializes Swift `Int` values, bounds-checks them and creates an `Int32` payload on the host. The model-default path creates the same Int32 position values with lazy MLX `arange`. It keeps the same causal/SWA comparisons and padding/indexer masks; no per-token eval/readback, global range cache, new buffer policy, custom Metal shader or sampler override is introduced.

An absent `VMLX_NAIVE_ALLOWED_MASK_GPU_ARANGE` or exact `1` selects arange; exact `0` and unrecognized spellings select the reference path. Empty, whitespace-padded, `true`, `false`, `yes`, and other unsupported spellings are not treated as enablement. `NaiveN05FlashModel` captures one immutable policy at construction and passes it through all layers. A typed explicit Boolean override remains available to focused fixtures. The attention-level convenience default remains the reference path; runtime model construction owns and explicitly propagates the policy.

The GPU range is admitted only within the existing Int32 constructor contract. Int32.max singletons and unsupported bounds use the baseline constructor, preserving its existing out-of-Int32 failure behavior rather than adding a new precondition. Position dtype stays Int32; mask dtype stays Boolean. `cacheStorageDTypeIdentity` remains `naive-n05-paired-v1` because the integer/mask values are exact and no attention reduction or cache schema changes. The unrelated selected-KV candidate is excluded from this branch.

## Retained measured source

The mask candidate ran at integration source `b04935c15c2f4dd03827b779b6ff1a44daf21819`, using a sealed native RunBench executable/resource closure. The source-separated main-based revision here is a later port, not that measured executable. The candidate helper and mask math preserve the measured implementation; constructor context and the policy fixture omit the unrelated selected-KV argument absent from main.

Pinned measured manifest SHA256: `a7d475c7f5e00ff2c479eab802e67deaefaa40388b3452b98d91b969d0c80a64`. Closure SHA256: `a695fedbd617fb78e0d4b4c0fd4967f2963e795647fad4a94f6e284809b379be`. Frozen runner SHA256: `f7dcdc409c56cc7b4e3db8dbb9aa195bf778d5746658bc52b1d09eb1a16a52a6`. Ordinary growing-prose fixture SHA256: `69ca47b2d1eee78edfec4664c9bd81ce4d6b10c654697b926ac74031183b5635`.

The actual Naive-N0.5-Flash-JANGH2 bundle drove native sampling: temperature1, topP.95, topK0, minP0, repetition1, seed20260928; reasoning control omitted. Disk cache was ON with separate initially empty namespaces; paged/hybrid cache, selected-KV, unsafe model compilation, MTP and generation profiling were OFF/absent. All four arms used the same executable, bundle, fixture and remaining environment. No text or sampler was changed to make a row pass.

## Focused native fixture

The retained root-owned b049 `NaiveN05AllowedMaskGPUArangeTests` gate passed **6/6, zero skips**, covering immutable/default policy and cache identity, exact signed/large/Int32-boundary positions,8k/10k one-row causal/padding/SWA masks, batched multiquery/all-masked cases, stable sparse-selection ties/masked entries, and attention/indexer/cache companion parity across prefill and decode. Its receipt SHA256 is `927a64b0b94298b30f7eba7d29c556190e108d4a2f7c7b2a4caa78697a8d8ce3`.

The retained b049 suite tested its then-default-OFF policy. This branch now renames that method to `testPolicyDefaultsOnIsImmutableAndKeepsNumericalCacheIdentity`, checks strict default/opt-out/invalid-value parsing, and checks the no-override model owner plus every layer against the current process policy without changing global environment. This revised six-method fixture is UNBUILT/UNRUN.

The optional host-construction diagnostic is a separate test class and is not part of this six-method correctness gate. Its timed region performs no eval/synchronize/readback, and live heap deltas cannot measure cumulative transient allocation. This branch has not executed either class.

## Native ABBA prose result

| Arm | run0 tok/s | run1 tok/s | run2 tok/s | Peak owned physical footprint GiB |
| --- | ---: | ---: | ---: | ---: |
| off-1 | 27.021614 | 25.826784 | 24.764681 | 4.253984 |
| on-1 | 30.243443 | 29.905091 | 29.510446 | 4.254381 |
| on-2 | 32.325270 | 31.610647 | 31.294147 | 4.252733 |
| off-2 | 27.253328 | 26.370625 | 25.438842 | 4.255068 |

Six-row pooled medians are **26.09870478 → 30.76879514tok/s**, **1.17893954x** (17.8940% faster). All12 turns stop naturally. Every arm has prompt counts7983/9542/10290 and generated counts1516/703/916. The45tok/s goal remains open.

All corresponding complete visible answers, reasoning channels, original user text, prompt sizes and generated counts are identical across OFF1/ON1/ON2/OFF2. Shared canonical semantic transcript SHA256: `965ec62700c64d71bb4ec6842240106c74c2d1c1ef107bcb2a5babb3e34dc14b`. Manual review confirms correct **cedar-47**, **three copper boxes**, **two blue boxes**, **five total boxes** recall in visible output and reasoning on both follow-ups, with continuous coherent explanation and no observed mask-induced semantic regression.

Shared factual precision remains PARTIAL: the notebook20–23C variation is flattened into identical-temperature claims, a contact thermometer is described as directly measuring heat-transfer rate, method descriptions are treated as measured confirmation without resulting data, and37C is used as a fixed hand-skin temperature. These are identical model output limitations in both arms, not arange-caused regression. They are preserved in the semantic review.

## Exact retained evidence

Raw artifacts are retained privately; the identities below do not imply publicly downloadable logs. The fixture uses synthetic ordinary prose and reference values.

| Arm directory | Receipt SHA256 | Raw native log SHA256 |
| --- | --- | --- |
| naive-arange-off-1 | `7e829dbbe661f2fa94993a8b5e434d994c8e28fc606b49821849e1c358bfcdfc` | `059b0441c6aa264088e2fd811eeaa9b198f81198b0b104fbc330d7113ac9a75b` |
| naive-arange-on-1 | `bb6aa2f008d34a780aff05c5105e09c26b6660aab4db298c44e43539dd7228ab` | `5d4ca9283b3454c5255985ec999e6bdee2778857946d03f23c882eae0bc27c16` |
| naive-arange-on-2 | `204811e2c51a7ff663a99f5cb3e573d4507e2a4e87d5e79aba15e35072e4f7ce` | `013ad5c3feeca1640380f4f169de1b676cc872c6aaeb6a37f432c493b8349858` |
| naive-arange-off-2 | `07f0ddc85596f0509c8abec9e57098bed76244c6ebcab80d4f29fa414b02a12a` | `1e64ba9f009ae8cbd7995648439d758ffa551b80db86e7a555f3cce0db4cb406` |

Manual review: `NAIVE-ARANGE-ABBA-SEMANTIC-REVIEW.md` SHA256 `3c84b028b96313bad7d7b9f4311faede1c59c6fee6c7e7d5ec7d765f3c4e993c`; standalone JSON SHA256 `2c543e1da0e9b76c273bbac8d9ef7ae4c4c2f2b5c44f1c4ead61107c5096b99e`. The JSON retains full shared transcripts, serialization contract, individual visible/reasoning/user hashes, exact identity/memory/log hashes, rates and review findings. Native fixture: `cache-173-fixture4-NaiveN05AllowedMaskGPUArangeTests/receipt.json`.

## Remaining gates and limits

This ABBA is sustained ordinary prose evidence at one source, bundle, host and context family. It is not exact full-model logits/state proof or a statistical confidence estimate; host page-cache warmth remains unmeasured. Owned process physical-footprint samples are available, but this diagnostic ceiling is not a blanket family/hardware RAM qualification.

Zero-tool text turns do not prove per-tool checkpoints, recurrent-companion SSD reconstruction, media, MTP or Osaurus GUI behavior. Disk `stores` is a legacy attempt count rather than successful-publication proof. The retained focused parity fixture, ABBA and semantic review do not certify this later default revision, merge or release.

The exact main-based candidate fixture and consuming app proof remain required before delivery. This source port has not yet been built or executed.
