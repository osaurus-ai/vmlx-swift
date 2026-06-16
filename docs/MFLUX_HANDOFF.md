# vMLX-Flux (native mFLUX image gen) — HANDOFF

**For:** the next engineer/agent continuing the native mFLUX image-generation port.
**Date:** 2026-06-16. **Owner:** Eric. **Status:** Osaurus
`vmlx-origin/main` runtime-proof baseline
`a188a2ccecc92c8a5993506acc83df16f83c7420` has fresh live proof for
z-image-turbo 4/8-bit, flux-schnell 4/8-bit, qwen-image 4/6-bit,
qwen-image-edit q4/q5, and staged Ideogram fp8. Qwen-edit q3 is incomplete
(`text_encoder/3.safetensors` missing from its index), q6 is incomplete on
disk, qwen masks are unsupported by the current mflux qwen-edit reference,
official `ideogram-ai/*` access is still approval-gated for the current HF
account (`hf download ideogram-ai/ideogram-4-fp8 --dry-run` and
`hf download ideogram-ai/ideogram-4-nf4 --dry-run` both returned
`Access denied. This repository requires approval.` on 2026-06-16), and
Ideogram nf4 is not staged/proven.

**2026-06-16 current a188 main refresh:** `/Users/eric/vmlx-swift-fluxwt` was
verified clean on `vmlx-origin/main`
`a188a2ccecc92c8a5993506acc83df16f83c7420`. Load matrix:
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-load-matrix/compatibility-matrix.json`
loaded z-image 4/8, flux-schnell 4/8, qwen-image 4/6, qwen-edit q4/q5, and
Ideogram fp8; it failed before load for incomplete Ideogram nf4 and qwen-edit
q3/q6. Fresh generation/edit artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-zimage-4bit-gen/Z-Image-Turbo-mflux-4bit-load.json`
(apple/repeat SHA `34652dd8b13a0840173d62847b844ad28ed65d8a460394aba5f0a7bf59570213`,
mountain `d3c47d2d0d4d4840dd8e91221b4bff7e24077b7ef0cd2a2abd51f62ace6a3708`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-zimage-8bit-gen/Z-Image-Turbo-mflux-8bit-load.json`
(apple/repeat `dd8eb0eb476dcc62f7ddc3beb989605653020fa50db21264f1dbeb04c2c85612`,
mountain `82018e3026fdebefe2c846dfc9a45e0780e683c4d0829ff36b00da6b99333a59`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-flux-schnell-4bit-gen/FLUX.1-schnell-mflux-4bit-load.json`
(apple/repeat `22338bb7c31b70945a137465cd106bf0dc8b6899303efbfa56646b3749c99f7b`,
mountain `c0d7e1553030a548734634e4d5b615ffff9520f5384b6ea23a987471bc997abc`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-flux-schnell-8bit-gen/FLUX.1-schnell-mflux-8bit-load.json`
(apple/repeat `92df2b7427aca358cc1f83ca2a32ca98c5ac3ccd5a2fc8fae77f54749275c572`,
mountain `51f35b30b51d52c1ffaa97589597c583b36244ebcf66530bdf61696025382300`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-4bit-gen20/qwen-image-mflux-4bit-load.json`
(apple/repeat `2c7b1c35af73ed66d45a958e7c2204c635ac95bef5e6a167d81e1285b7579b12`,
mountain `cae848046c34d06a6407e28e9d7cd820c9721906a8defd92de5b23666599ecb8`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-image-6bit-gen20/Qwen-Image-mflux-6bit-load.json`
(apple/repeat `e5865d8b2a90bb759d4c6f5a647b03e0ba7542e2f4d7ed667ec8920c9f25b866`,
mountain `e41ca45ea175cfa965db565e4d8f8d0aa84b3db908dd10fb9efccbb418839b50`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q4-gen20/Qwen-Image-Edit-mflux-q4-load.json`
(blue/repeat `5d658df3259f85502188c4432896682708ce96f3ebaab3c1970678585f746be9`,
green pear `91df895dcd3d8eea33fbb464642d0d1ad1febecf646257f46e20efaac400ab7d`),
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-qwen-edit-q5-gen20/Qwen-Image-Edit-mflux-q5-load.json`
(blue/repeat `5265852e90c727b45c224f887254763152c8612f55a48b7932fa1d12327d98c8`,
green pear `41732444e47cbc028dac25035e70ffdf216203c94e3d6a7685d4bb729d20ea19`),
and
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-strict/ideogram-4-fp8-load.json`
(apple/repeat `c62b3b71a82ebcb0964be709c03678271364d381dd4ae8029af7b85d4bf02264`,
mountains `d193163f8584ad6040bc71d42960c98ac7864391f76f79c485cf8eca6905b2c1`).
All listed rows completed all three turns; repeated prompts were byte-identical
and different prompts changed SHA. Visual inspection:
`docs/local/vmlx-flux-outputs/2026-06-16-current-a188-contact-sheet.png` shows
z-image/flux/qwen txt2img rows are coherent apple/mountain images, qwen-edit q5
cleanly edits blue apple and green pear, and qwen-edit q4 changes color/shape
but is noisier/weaker. Ideogram fp8 strict object prompt proof is clean
(`docs/local/vmlx-flux-outputs/2026-06-16-current-a188-ideogram-strict-sheet.png`),
but the broader current-a188 prompt
`docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-determinism/ideogram-4-fp8-load.json`
hallucinated text on the apple prompt. Keep Ideogram exposed only as staged
fp8/typography/strict-icon test coverage until broader prompt rows pass.

**2026-06-16 previous e0f main refresh:** `/Users/eric/vmlx-swift-fluxwt` was
verified clean on `vmlx-origin/main`
`e0f3ccff7ae78a6b3e8ccc4989825f582d1b7ee5`. Load matrix:
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-load-matrix/compatibility-matrix.json`
loaded z-image 4/8, flux-schnell 4/8, qwen-image 4/6, qwen-edit q4/q5, and
Ideogram fp8; it failed before load for incomplete Ideogram nf4 and qwen-edit
q3/q6. Fresh generation artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-zimage-4bit-gen/Z-Image-Turbo-mflux-4bit-load.json`
(apple/repeat SHA `241e354e065e511eaa3f6fe5765bce1f883c45896076f23b1c901de3388917ec`,
mountain `c1c7a1dabefcf46b020c44e6633fe7dc7634a8c7ecbd39806715affd1580481a`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-zimage-8bit-gen/Z-Image-Turbo-mflux-8bit-load.json`
(apple/repeat `a8915bb95421ceb1d1f3215207e8d14b2f4abdb798bd3beaf9d2bcca5726ac16`,
mountain `59f1b0205bec11a082e7036ba4c3a854ab95f0f0ff71c2bf36799157f2b798c4`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-flux-schnell-4bit-gen/FLUX.1-schnell-mflux-4bit-load.json`
(apple/repeat `438791bf7b7dbcd4a2beff5494e3f217656a787dc2bab985959deb2dac8fc207`,
mountain `7e648a010c927046a0888b6c99e154c1c86c4bc5b3bad03fe9ac2a38bcf1ae0b`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-flux-schnell-8bit-gen/FLUX.1-schnell-mflux-8bit-load.json`
(apple/repeat `cbcb19c2fa07bdd3c7ec6d2157a724b2f0396cfaf1ea1a297956efb51d2f775e`,
mountain `75e8ce7cb16f40e5d1df771a85cf2509859295ccc41a7cb516c3c53f55a8d733`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-qwen-image-4bit-gen20/qwen-image-mflux-4bit-load.json`
(apple/repeat `0d7cbb782e3fc428019ee7bfc39cc0051bc9eeabbae7adcc713e60ec453e2281`,
mountain `82913dcd860f1a163dd3e3874aa5ef7566090b1a1582c4e200634eba8a707e45`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-qwen-image-6bit-gen20/Qwen-Image-mflux-6bit-load.json`
(apple/repeat `01ccdb56c6b20dcab470ffac4b74a369e6e20ccbe53e7c7041daf79f5f5176a1`,
mountain `042b86f076c8ad8d6c337a0a07726a51752c91d0949eb1393255ff04062cff5a`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-qwen-edit-q4-gen20/Qwen-Image-Edit-mflux-q4-load.json`
(blue/repeat `cfe9cbe5680ea5e30c9d529fca2d8523b0fc1aaa7804a1fadc2c86d1311b8d5e`,
green `cbce394cb180d35894a62533bba6c8834a55f93adfd5615dc2377b705b64657f`),
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-qwen-edit-q5-gen20/Qwen-Image-Edit-mflux-q5-load.json`
(blue/repeat `100cb92fa08d71b32ab10666c8505b81915c7faf7cab157fe1f30f636b287588`,
green pear `e2d9873d265a96aa908496408fd24918393f0123dea503cb8dc5692197c8f863`),
and
`docs/local/vmlx-flux-probes/2026-06-16-current-e0f-ideogram-fp8-object-determinism/ideogram-4-fp8-load.json`
(apple/repeat `c62b3b71a82ebcb0964be709c03678271364d381dd4ae8029af7b85d4bf02264`,
mountains `d193163f8584ad6040bc71d42960c98ac7864391f76f79c485cf8eca6905b2c1`).
All listed generation/edit rows completed all three turns; repeated prompts
were byte-identical and different prompts changed SHA. Visual inspection:
z-image/flux/qwen txt2img rows are coherent apple/mountain images; qwen-edit q5
cleanly edits blue apple and green pear; qwen-edit q4 changes color but remains
weaker/rougher on shape-changing green-pear prompts; Ideogram fp8 now has both
readable HELLO/BANANA typography proof and clean object-icon apple/mountain
proof. Keep official Ideogram repos and nf4 separately gated until access and
local bundle proof exist.

**2026-06-16 continuation evidence:** live baseline probes were rerun from
`/Users/eric/vmlx-swift` so MLX could resolve `default.metallib`; the standalone
repo itself does not contain that Metal library. Fresh standalone artifacts are
under `docs/local/vmlx-flux-{probes,outputs}/2026-06-16-*`. The Osaurus
monorepo worktree also has fresh load artifacts from `/Users/eric/vmlx-swift-fluxwt`:
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-image-q4-load-final/qwen-image-mflux-4bit-load.json`
(`load_status=loaded`, `native_runtime_status=native_pipeline_implemented`) and
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-edit-q4-load-final/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`, pre-fix `native_runtime_status=native_pipeline_partial`).

**2026-06-16 Osaurus PR #64 pre-merge live-proof refresh:** rerun from
`/Users/eric/vmlx-swift-fluxwt` on then-active branch
`codex/mflux-qwen-edit-main` at pre-doc commit `81fae70e`. Text-to-image
artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-goal-zimage-4bit-explicit-gen/Z-Image-Turbo-mflux-4bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-zimage-8bit-explicit-gen/Z-Image-Turbo-mflux-8bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-flux-schnell-4bit-explicit-gen/FLUX.1-schnell-mflux-4bit-load.json`,
`docs/local/vmlx-flux-probes/2026-06-16-goal-flux-schnell-8bit-explicit-gen/FLUX.1-schnell-mflux-8bit-load.json`,
and
`docs/local/vmlx-flux-probes/2026-06-16-goal-qwen-image-4bit-gen20/qwen-image-mflux-4bit-load.json`.
Viewed contact sheet:
`docs/local/vmlx-flux-outputs/2026-06-16-goal-contact-sheet-explicit-turn-order.png`.
Qwen-edit q4 pre-fix apple-blue artifact:
`docs/local/vmlx-flux-probes/2026-06-16-goal-qwen-edit-q4-apple-blue/Qwen-Image-Edit-mflux-q4-load.json`
(`edit_turns[0].status=completed`, output SHA
`5fc2b04436eb0a8ad0e7a61265f962ec0dc67027efa417804bc39c80ab37cb13`);
viewed output is source-like and failed the requested edit. Root cause was the
edit conditioning grid: Swift used 1024-area VAE dimensions for static source
latents, but mflux passes `vl_width/vl_height` and encodes those latents at the
VL size. Post-fix proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`load_status=loaded`; turn 1 and turn 3 same blue-apple prompt SHA
`005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a`;
turn 2 green-pear prompt SHA
`815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`).
The viewed PNGs are coherent/prompt-sensitive and the repeated prompt is
byte-identical. Shape proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`conditioning_width=384`, `conditioning_height=384`, `patch_rows=24`,
`patch_columns=24`, `latents_shape=1x576x64`) and
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`target_latent_count=1024`, `conditioning_latent_count=576`,
`combined_velocity_shape=1x1600x64`, `image_shapes=[[1,32,32],[1,24,24]]`).
Current local scan: `docs/local/vmlx-flux-probes/2026-06-16-goal-current-scan/scan.json`.
At that point no complete Ideogram bundle was staged. Later on 2026-06-16,
`cocktailpeanut/ideogram-4-fp8` was staged locally at
`~/.mlxstudio/models/image/ideogram-4-fp8`; fresh scan artifact
`docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-mirror-scan/scan.json`
reports `ideogram-4-fp8` as `readiness=loadableScaffold`, 4 safetensors,
27,526,985,054 bytes, with tokenizer/text_encoder/transformer/
unconditional_transformer/vae present. The next source slice added
`MFluxLinear` support for fp8 `weight` + `weight_scale` rows and taught
`WeightLoader` to load the `unconditional_transformer` component. The later
native slice implemented Ideogram Qwen3 text encoding, conditional and
unconditional 34-layer DiT execution, mflux default 20-step guidance scheduling,
Flux2 VAE decode, and PNG output. A real bug was fixed there: both Ideogram
rotary helpers initially used `[-firstHalf, secondHalf]`; mflux and the other
native ports require `[-secondHalf, firstHalf]`.
Follow-up load proof after the validation gate:
`docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-honest-load/ideogram-4-fp8-load.json`
reports `load_status=loaded`, records `load_elapsed_seconds`, reports
`native_runtime_status=not_implemented`, and confirms the same
27,526,985,054-byte staged bundle. This proves direct engine load now reaches
the Ideogram loader and sentinel-key validator, not only the scanner.
Current live typography proof:
`docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-native-gen20-current-source/ideogram-4-fp8-load.json`
(`load_status=loaded`; three completed turns; HELLO turn 1 and turn 3 share SHA
`6534f016378a94add5ccc29397decf45c4dada6c1d82260bdd51517390cf4205`; BANANA
turn SHA `b02464bd06e689ea6fc7aeb33dbc70bb1e1eb5b08c92668abc1832f61239f0b5`;
viewed outputs are readable and prompt-sensitive). Current object-scene
boundary artifact:
`docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-native-gen20-object512-current-source/ideogram-4-fp8-load.json`
(`load_status=loaded`, 512x512 output SHA
`005ee15c584e37351672fb4ae40910348d05bf608705ee74c2aebe017682f072`);
viewed output contains an apple icon plus extra hallucinated text. This boundary
failure is superseded for strict clean icon prompts by the current a188 object
proof listed above, but keep it as a regression case for broader Ideogram
prompting.
Official `hf download --dry-run` for `ideogram-ai/ideogram-4-fp8` still returned
`Access denied. This repository requires approval.` on 2026-06-16.
Qwen-Image 6-bit was staged from `filipstrand/Qwen-Image-mflux-6bit` on
2026-06-16; its slow tokenizer was converted to `tokenizer.json`, and the
Diffusers-style mod-linear key layout (`img_norm1.mod_linear` /
`txt_norm1.mod_linear`) is handled by the Qwen native loader fallback. Live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-image-6bit-gen20-after-key-fix/Qwen-Image-mflux-6bit-load.json`
(`load_status=loaded`; turn 1 and 3 apple prompt SHA
`66e8187e887087e8a8e9227a99f16236c5ba15717a5e31e08a5772868b3a456a`;
turn 2 blue watercolor mountain SHA
`44069312716932d6d72181a808625a33777ed29af7723eea8f76b0ac5ba96a52`).
Viewed outputs are coherent and prompt-sensitive. Current HF search did not find
a public Qwen-Image mflux 8-bit bundle.

**2026-06-16 previous current-main refresh:** after PR #67 merged, `/Users/eric/vmlx-swift-fluxwt`
was fast-forwarded to `vmlx-origin/main` `9f1faea11aee78f17041c5bed6da039e70c11d05`
and the supported rows were rerun from that main checkout. Source gate:
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter vMLXFluxTests`
passed 48 tests with 0 failures; `swift build --product vmlxflux-probe` passed
with existing Swift warnings. Live proof artifacts:
`docs/local/vmlx-flux-probes/2026-06-16-current-main-zimage-4bit/Z-Image-Turbo-mflux-4bit-load.json`
(apple SHA `d4e90622247926338cdf25b70912b35ca1cb533b4d4b1855d88c1c1e8ad2362f`,
mountain SHA `1b8fe70d72b2fb048bceb2e9a3d529dcfc673bf20e6496b3ae30d0d9e1c244fc`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-zimage-8bit/Z-Image-Turbo-mflux-8bit-load.json`
(apple `a48724b2145ed1f580a60aa142463d67bc796ad41d51837270dbd0d7ac6bb6af`,
mountain `4671f293c7302855df22579b51884cfd9e7323fbbd67dcb9e85d09c5b17670ee`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-flux-schnell-4bit/FLUX.1-schnell-mflux-4bit-load.json`
(apple `d3ed219d4a33ac85de101f26aacb34ffc5c74eece283e65d626d3465e9a2eed5`,
mountain `7f5847f352cf974caecbdf3feebcc06a96bebef528e7f15837a8ff4e8d65418a`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-flux-schnell-8bit/FLUX.1-schnell-mflux-8bit-load.json`
(apple `6c4d247a60b101c0b3d3809cb0a2b3934ef9e5fbbc1a9e9125800593a8f147af`,
mountain `728fe78dfae0e73c179d2e2e2f8320f7ee3452ad47e7a52a6a075cd4558191bd`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-qwen-image-4bit/qwen-image-mflux-4bit-load.json`
(apple `941026906d25738c3dd1354a0066d2957e6e0840b09de7015460a301b937e8ec`,
mountain `eb8af5d21bf08047a98eda008ce041734819e77a943ef8287ba8b126dd0e5b50`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-qwen-image-6bit/Qwen-Image-mflux-6bit-load.json`
(apple `4ff67d3a3e89403d4516c77cd6806ca7682614227884744f1b37e91d99ea62c1`,
mountain `e5cf83f268786f030dc0cc5384085e4ac4ac6bd7d2a5e4835d4e8da3f1a413`),
`docs/local/vmlx-flux-probes/2026-06-16-current-main-qwen-edit-q4/Qwen-Image-Edit-mflux-q4-load.json`
(blue/edit-repeat SHA `83755dc3c90a8669a2c882391c51e275c7737d379cd4aa746a42bb2e920a9568`,
green-edit SHA `4c677d9e76cba3e37ebb58a4de23a374ccbd262201bdfb73329aff5134c872bb`),
and `docs/local/vmlx-flux-probes/2026-06-16-current-main-qwen-edit-q5/Qwen-Image-Edit-mflux-q5-load.json`
(blue/edit-repeat `cb4fcb2d2cf85161bd612cd15f5370b3863f62fa72d9a52ce79ba3d962eadc4c`,
green-pear `b9c1534f0ccf3cf04c39596c89f2a829a26d0e1114c5bbbe1f1906658c738e37`).
All rows loaded and completed all three turns; turn 1 and turn 3 match exactly
for the repeated prompt, and turn 2 differs. Contact sheet viewed:
`docs/local/vmlx-flux-outputs/2026-06-16-current-main-contact-sheet.png`.
Visual note: z-image/flux/qwen text-to-image rows are coherent and prompt
sensitive; qwen-edit q5 is clearly prompt-sensitive, and qwen-edit q4 is
deterministic/prompt-sensitive but weaker than q5 on the 384px green-pear edit.

This is the single starting doc. Read it top to bottom, then the per-model port plans.

---

## 0. TL;DR — what works, what's next

| Model | 4-bit | 8-bit | full | Native pipeline file |
|---|---|---|---|---|
| **z-image-turbo** | ✅ proven | ✅ proven | ⬜ (weights gone) | `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` |
| **flux-schnell** | ✅ proven | ✅ proven | ⬜ (not staged) | `Libraries/vMLXFluxModels/Flux1/Flux1Native.swift` |
| **qwen-image** (txt2img) | ✅ proven; ✅ 6-bit also proven | ⬜ (public mflux 8-bit not found) | ⬜ | `Libraries/vMLXFluxModels/Common/QwenImageNative.swift` |
| qwen-image-edit | ✅ q4/q5 single-image edit proven; q4 weaker on shape-change, q5 cleaner; ordered multi-image proof retained; q3/q6 incomplete | — | — | `Libraries/vMLXFluxModels/QwenImage/QwenImageEditSupport.swift`; qwen masks unsupported |
| ideogram (4) | ⬜ nf4 incomplete | ✅ fp8 staged mirror proven for typography + strict clean object icons; broader no-text apple prompt can hallucinate text | — | `Libraries/vMLXFluxModels/Ideogram4/Ideogram4.swift`, `Libraries/vMLXFluxModels/Ideogram4/Ideogram4Native.swift` |
| flux1-dev/kontext/fill, flux2-klein, fibo, seedvr2, wan | ⬜ scaffold | — | — | registered, throw `notImplemented` |

"Proven" = live-generated a coherent, prompt-accurate image that is **deterministic** (same seed+prompt -> byte-identical) and **prompt-sensitive** (different prompt same seed -> different coherent image). Per Eric's HARD RULE: *do not trust/claim a model works until you have generated and visually checked a real image.* 2026-06-16 rerun: z-image 4/8 and flux-schnell 4/8 passed live load + three-turn generate + SHA determinism/prompt-sensitivity + visual inspection. Qwen-image 4-bit also passed live load + 20-step generation + three-turn SHA determinism/prompt-sensitivity + visual inspection after the mflux guidance rescale fix; turn 1/3 apple SHA `2f1c27c68993fe9a537bca2cc019ac3d32d59818b92c606c00726104661bcea7`, turn 2 mountain SHA `2bf77ce59c8ed99c1b1aa5fb8940c9d35948b1763fbd360e14f577032b62f060`, artifact `docs/local/vmlx-flux-probes/2026-06-16-qwen-image-q4-guidance-proof/qwen-image-mflux-4bit-load.json`. Qwen-image 6-bit also passed live load + 20-step three-turn SHA determinism/prompt-sensitivity + visual inspection; turn 1/3 apple SHA `66e8187e887087e8a8e9227a99f16236c5ba15717a5e31e08a5772868b3a456a`, turn 2 mountain SHA `44069312716932d6d72181a808625a33777ed29af7723eea8f76b0ac5ba96a52`, artifact `docs/local/vmlx-flux-probes/2026-06-16-qwen-image-6bit-gen20-after-key-fix/Qwen-Image-mflux-6bit-load.json`.

Qwen-image-edit q4/q5 are live-proven after fixing the source-image conditioning grid to match mflux. Source trace: mflux `qwen_image_edit.py` passes `vl_width/vl_height` into `QwenEditUtil.create_image_conditioning_latents`, and `qwen_edit_util.py` uses those VL dimensions for the source-image VAE encode when present. Swift now mirrors that in `QwenImageEditSupport.swift`: square source images encode conditioning at 384x384, pack 24x24=576 static latents, and denoise with 1024 target latents + 576 conditioning latents. q4 live proof artifact: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (blue prompt SHA `005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a`, green-pear prompt SHA `815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`, repeated blue prompt same SHA). q5 live proof artifact: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json` (blue prompt SHA `5cd5d9197bd659bd8b59b4a2f2bca413266146ad4e08249289d5fa6a8025fa4e`, green-pear prompt SHA `d2c6c4eb4a19dcf48122b5216fc15ac37b9f5aa49c15f596acd1276a4df57034`, repeated blue prompt same SHA). Viewed q4/q5 outputs are coherent and prompt-sensitive. Boundary artifacts remain useful: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-prompt-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-vl-encode-live/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`, and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`.

Qwen-image-edit multi-image is also live-proven on q4 and q5. Source trace:
mflux qwen-edit accepts `image_paths: list[str]`, sizes from `image_paths[-1]`,
and concatenates per-source prompt-image features plus VAE conditioning latents.
Swift mirrors that with ordered `ImageEditRequest.sourceImages`, repeated
`--source-image` in `vmlxflux-probe`, last-source sizing, per-image Qwen2.5-VL
feature/token counts, and concatenated VAE latents/IDs. q5 shape proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-shape-proof/Qwen-Image-Edit-mflux-q5-load.json`
(`image_count=2`, `latents_shape=[1,1152,64]`,
`image_token_counts=[196,196]`,
`combined_velocity_shape=[1,1728,64]`,
`image_shapes=[[1,24,24],[1,24,24],[1,24,24]]`). q4 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-multi-image-live/Qwen-Image-Edit-mflux-q4-load.json`
(turn 1/3 SHA
`e43910a505ab090bfbd4ec3a00f6e58fcd97df65c6b61e6b973754511bc740be`,
turn 2 SHA
`16ecc1fec4bdff1e5aecb9c0875569b2bebed53a81c686bf23e806faa6e2b893`).
q5 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-live/Qwen-Image-Edit-mflux-q5-load.json`
(turn 1/3 SHA
`ec2e49d6f300849cb46940b793670bee007f4aae8f04e97a31289670758519c9`,
turn 2 SHA
`8dfacb52aa81c6e0a8a6827c4377f27bc5d9396e39a4f8662a47db93798b767a`).
Viewed exact current-source output PNGs recorded in the q4/q5 load artifacts
(the existing contact sheet is
`docs/local/vmlx-flux-outputs/2026-06-16-qwen-edit-multi-image-contact-sheet.png`).
Visual caveat: q4 is rougher than q5; q5 strongly uses the mountain and
green-pear prompt, while its first apple prompt leans mountain-only. This is
multi-reference text-image edit, not qwen mask/inpaint support.

**Next work, in priority order:**
1. **Ideogram 4 follow-through** — staged `cocktailpeanut/ideogram-4-fp8` is now
   live-proven for typography and strict clean object-icon prompts on the
   Osaurus runtime-proof baseline. A broader current-a188 "no text" apple prompt
   still hallucinated text, so do not advertise it as a general clean object
   renderer yet.
   Official `ideogram-ai/*` approval is still needed for canonical official
   bundles, and nf4 requires a complete local bundle plus load/generation proof
   before exposure.
2. **qwen-image-edit follow-through** — q4/q5 single-image and ordered
   multi-image text-image edit paths are implemented/testable. q5 is cleaner;
   current q4 proof changes color but is weaker on shape-changing green-pear
   prompts. q3/q6 need complete local bundles before exposure. Qwen masks
   remain unsupported unless upstream mflux adds a real qwen mask path or a
   separate fill/inpaint model is wired; do not fake masks with post-blends.
3. **Full-precision** flux-schnell + z-image (download + prove with existing pipelines — should "just work").
4. Osaurus app/server wiring: implement the `/v1/images/*` bridge from the
   specs below, wrap every image request in the required `MetalGate` exclusion,
   expose only proven variants, and pin Osaurus to `vmlx-origin/main`
   `a188a2ccecc92c8a5993506acc83df16f83c7420` or a later verified main SHA.

---

## 1. Where the code lives

- **vmlx-swift integration worktree:** `/Users/eric/vmlx-swift-fluxwt` — clean
  Osaurus monorepo worktree for this lane. The current runtime-proof baseline is
  `vmlx-origin/main` `a188a2ccecc92c8a5993506acc83df16f83c7420`; later commits
  in this lane must be source/test checked before pinning.
- **Dirty local dev tree:** `/Users/eric/vmlx-swift` — branch
  `codex/mimo-v25-cache-contract` carries unrelated MLXPress/MiMo/Gemma/JANG
  WIP; do not commit the image-gen integration from that dirty checkout.
- **Pushable remotes:**
  - `jjang-ai/vmlx-flux` (standalone SwiftPM engine) — **all native work is pushed here** on branch `native-zimage-proven`. This is the durable home. Latest: branch HEAD.
  - `osaurus-ai/vmlx-swift` (the monorepo) — image engine work is merged to
    `main` through PRs #63, #64, #65, #66, #67, #68, and later direct main
    proof/doc refresh commits. Remote name
    `vmlx-origin`. (Note: the `osaurus-upstream` remote is DO_NOT_PUSH — only
    the mlx-swift fork.)
- **Standalone clone (for vmlx-flux pushes):** `/Users/eric/vmlx-flux-push` (sibling to `../vmlx-swift-lm` so its path-deps resolve).

### Module layout (vendored in `vmlx-swift/Package.swift` as in-tree targets)
- `vMLXFluxKit` — `FluxEngine` types, `ModelRegistry`, requests/events, `FlowMatchEulerScheduler`, `VAE`, `WeightLoader`, `MLXStudioModelStore` (`LocalModelStore.swift`), JANG bridge.
- `vMLXFluxModels` — concrete models. **`Common/`** holds the shared, reusable pieces:
  - `MFluxQuant.swift` — `MFluxStore` + `MFluxLinear`/`MFluxEmbedding`/`MFluxRMSNorm`/`MFluxLayerNorm`/`MFluxGroupNorm`/`MFluxConv2D`. **This is the foundation every port builds on.**
  - `T5XXL.swift` (flux), `CLIPText.swift` (flux), `QwenImageNative.swift` (qwen full pipeline).
  - `Flux1/Flux1Native.swift` (flux DiT + VAE + pipeline; also defines `FluxAdaNormContinuous` reused by qwen), `ZImage/ZImageNative.swift` (z-image, has its own private store — left untouched to avoid regressing the proven path).
- `vMLXFluxVideo` — WAN 2.x scaffold (`WanVAE3D.swift` has a `CausalConv3d` shim).
- `vmlxflux-probe` (`tools/vMLXFluxProbe/main.swift`) — the verification CLI (see §4).

---

## 2. Build & run

```bash
cd /Users/eric/vmlx-swift-fluxwt
swift build --product vmlxflux-probe          # warm .build; ~3-40s. Default CommandLineTools toolchain is fine.
# Unit tests need the Xcode toolchain (XCTest):
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter vMLXFluxTests
```
- The repo's CI **skips** `mac_build_and_test` (known) — local build is the only gate before merging to main.
- Main is **tools-version 6.1 / Swift 6 language mode**. The flux/qwen targets are Swift-6-clean (verified) — no `.swiftLanguageMode(.v5)` pin needed.

---

## 3. Models on disk (staged at `~/.mlxstudio/models/image/`)
- `Z-Image-Turbo-mflux-4bit` (5.5GB), `Z-Image-Turbo-mflux-8bit` (10GB)
- `FLUX.1-schnell-mflux-4bit` (9GB), `FLUX.1-schnell-mflux-8bit` (12GB)
- `qwen-image-mflux-4bit` (24GB), `Qwen-Image-mflux-6bit` (21GB)
- `Qwen-Image-Edit-mflux` (89GB) with nested `q3`, `q4`, `q5`, `q6` variants.
  The scanner now expands these as `Qwen-Image-Edit-mflux-q3/q4/q5/q6`; q4 and
  q5 have complete tokenizer/text_encoder/transformer/vae bundles and are
  live-proven. q3 is incomplete because `text_encoder/model.safetensors.index.json`
  references missing `text_encoder/3.safetensors`; q6 is incomplete on the
  current disk image (`missing transformer`, `missing vae`, and missing indexed
  text-encoder shards). `Qwen-Image-Edit-mflux-q4` also passes
  the manifest-gated engine load contract (tokenizer files + Qwen LM keys +
  Qwen-VL vision keys + transformer keys + VAE encode/decode keys) and a live
  source-image preprocess request (`output=1024x1024`, `vl=384x384`,
  `vae=1024x1024`, `conditioning=24x24`, `vision_patches=784x1176`,
  `vision_grid=1x28x28`, `vae_input=1x3x384x384`). The q4 prompt probe
  tokenizes the mflux edit prompt with 196 repeated image-pad tokens
  (`input_ids_shape=1x276`, `image_token_id=151655`, `image_token_count=196`,
  `template_drop_index=64`). The q4 conditioning probe live-encodes the source
  through the Qwen 3D VAE at the VL size and packs static image latents
  (`conditioning_width=384`, `conditioning_height=384`,
  `latents_shape=1x576x64`, `image_ids_shape=1x576x3`, finite stats). The q4 VL
  encode probe runs the real Qwen2.5-VL vision transformer and image-token
  splice (`feature_shape=196x3584`, `prompt_embeds_shape=1x212x3584`,
  `token_image_count=196`, finite stats). The q4 denoise probe concatenates
  1024 target latents with 576 static conditioning latents, runs the edit-shaped
  transformer RoPE grids, and slices the target velocity
  (`combined_velocity_shape=1x1600x64`, `target_velocity_shape=1x1024x64`,
  finite stats). `ImageEditor` runs the scheduler loop, decodes, and writes
  coherent prompt-sensitive PNGs for q4. Ordered multi-image edit is wired through
  `ImageEditRequest.sourceImages`; repeat `--source-image` in the probe to pass
  multiple references. `Qwen-Image-Edit-mflux-q5` also has
  live same-seed text-image and multi-image edit proof at
  `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`.
  q3/q6 require complete local bundles before UI promotion. Non-null qwen masks are
  rejected before the edit pipeline loads; the mflux qwen-edit reference has no
  qwen mask/inpaint argument or path.

**Downloadable mflux-compatible weights (HF):**
- flux: `dhairyashil/FLUX.1-schnell-mflux-{4,8}bit`; full = `black-forest-labs/FLUX.1-schnell` (GATED).
- z-image: `Tongyi-MAI/Z-Image-Turbo` (full), `carsenk/z-image-turbo-mflux-8bit`, `filipstrand/Z-Image-Turbo-mflux-4bit`.
- qwen: `carsenk/qwen-image-mflux-4bit` (txt2img), `filipstrand/Qwen-Image-mflux-6bit` (txt2img), `fcreait/Qwen-Image-Edit-mflux` (87GB full edit model). Current HF search did not find a public qwen-image mflux 8-bit bundle.
- ideogram: `ideogram-ai/ideogram-4-fp8` (the mflux canonical), `ideogram-ai/ideogram-4-nf4` (4-bit).
  Current account state: both repos are visible through `hf models info`, but
  `hf download --dry-run` is approval-gated (`Access denied. This repository
  requires approval.`). The third-party `cocktailpeanut/ideogram-4-fp8` mirror is
  staged locally and scans complete. `MFluxStore` covers the fp8
  `weight_scale` linear format and `WeightLoader` loads the
  `unconditional_transformer` shard group. Direct load validates sentinel
  keys from the text encoder, transformer, unconditional transformer, and VAE;
  fp8 native generation now executes and has typography proof, but broader
  object-scene quality remains partial.

**TOKENIZER GOTCHA:** mflux bundles ship SLOW tokenizers (CLIP vocab.json+merges, T5 spiece.model). swift-transformers' `AutoTokenizer.from(modelFolder:)` needs `tokenizer.json` (fast). Convert once:
```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(dir, use_fast=True).save_pretrained(dir)   # for tokenizer/ and tokenizer_2/
```
(qwen + z-image bundles already ship tokenizer.json; flux needs the conversion.) `pip install --user transformers tokenizers sentencepiece protobuf` is already done on this box.

---

## 4. The probe (verification harness)
`tools/vMLXFluxProbe/main.swift`. The canonical proof command (same-seed determinism + prompt-sensitivity):
```bash
.build/debug/vmlxflux-probe --model <DIR-NAME> --generate --json \
  --seed 7 --width 512 --height 512 --steps <N> \
  --artifacts <art> --output-dir <out> \
  --turn "a red apple on a wooden table, photo" \
  --turn "a snowy blue mountain landscape, watercolor" \
  --turn "a red apple on a wooden table, photo"
# turn1≡turn3 (byte-identical sha) ⇒ deterministic; turn2≠turn1 ⇒ prompt-sensitive; then VIEW the PNGs.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --edit \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Repeat --source-image for ordered multi-reference qwen edit. Current qwen-edit
# q4/q5 status: live load + ImageEditor loop + PNG write + coherent same-seed
# prompt-sensitive single-image and multi-image edit proof.
# Add --mask-image <png> to prove the current unsupported-mask path emits a
# failed event before the edit pipeline runs.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-prompt \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Repeat --source-image to record per-image token counts and real tokenizer
# image-pad expansion.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-conditioning \
  --source-image <png> --artifacts <art>
# Repeat --source-image to record one VAE encode + packed static latents per
# source image.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-vision \
  --source-image <png> --turn "make the background blue" \
  --artifacts <art>
# Repeat --source-image to record per-source Qwen2.5-VL features and image-token
# splice into Qwen text embeddings.

.build/debug/vmlxflux-probe --model Qwen-Image-Edit-mflux-q4 --qwen-edit-denoise \
  --source-image <png> --width 256 --height 256 --steps 1 --seed 7 \
  --turn "make the background blue" --artifacts <art>
# Current qwen-edit denoise status: live q4 load + prompt-image encode + VAE
# conditioning + first transformer velocity slice with target+one shape per
# conditioning image.
```
- Flags: `--guidance`, `--negative` (added for CFG), `--width/height/steps/seed/turn/root/model/output-dir/artifacts/--matrix`.
- `--model` must be the **exact directory name** (the resolution bug — §6 — is fixed so `-8bit` no longer collapses onto `-4bit`).
- Per-turn seed = `--seed` if given (so all turns share it). Pipelines print `[qwen]`/`[flux]` stderr stats (shape/mean/max/finite) per stage — the **de-risk signal**: if a stage isn't finite, that's where the bug is.

---

## 5. Architecture cheat-sheet (how the pipelines are built)

All on `MFluxStore` (loads safetensors via `WeightLoader`, builds quant-aware layers from exact checkpoint keys). mflux **linear** weights are PyTorch `(out,in)` (handled by `MFluxLinear` via `matmul(x, weight.T)`); mflux **conv** weights are **MLX channels-last** `(out,[kt,]kh,kw,in)` (see §6 bug 1).

- **flux-schnell** (`Flux1Native.swift`): T5-XXL (`T5XXL.swift`) → per-token `prompt_embeds`; CLIP-L (`CLIPText.swift`) → pooled vector. `FluxTransformer` = 19 joint + 38 single blocks, 24h×128, 3-axis RoPE (`FluxRoPE`), `FluxTimeTextEmbed`, `FluxAdaNormZero/Single/Continuous`. `FluxVAEDecoder` (AutoencoderKL). FlowMatch Euler, 4 steps, guidance 0. **timestep passed = sigma×1000** (flux time-proj has no internal scale).
- **z-image-turbo** (`ZImageNative.swift`): Qwen-style text encoder + Lumina-style DiT (noise/context refiners) + AutoencoderKL VAE. Proven; uses its own private store. 4 steps, guidance 0.
- **qwen-image** (`QwenImageNative.swift`):
  - `QwenTextEncoder` = Qwen2.5 LM (28-layer, GQA 28q/4kv, standard RoPE θ1e6, SwiGLU, **causal**). Tokenize with the gen template; **drop the first 34 tokens** of the output → prompt embeds.
  - `QwenTransformer` = 60-layer MM-DiT (dual-stream `QwenBlock`: img/txt `mod_linear` 3072→18432 split into mod1(attn)/mod2(mlp), each shift/scale/gate; `QwenAttn` joint img+txt with RMSNorm q/k + complex-pair RoPE `QwenRoPE` axes[16,56,56] θ1e4 scale_rope; `QwenFF` gelu_approx 4×). img_in 64→3072, txt_norm+txt_in 3584→3072, `QwenTimeEmbed`, norm_out=`FluxAdaNormContinuous`, proj_out→64.
  - `Qwen3DVAEEncoder`/`Qwen3DVAEDecoder` = 3D causal-conv VAE **operated in 2D since T=1** (each causal Conv3d → 2D conv on the last temporal kernel slice; decoder resamplers do spatial nearest-2× + conv; encoder downsamplers pad bottom/right then stride-2 conv). Per-channel `LATENTS_MEAN/STD` (16-vectors). Decoder channel flow 384→192→192→96→3 over 3 upsamples (8×). Qwen-edit q4/q5 VAE conditioning encode/pack and edit-loop PNG output are live-proven for single-image and ordered multi-image text-image edits after the VL-grid conditioning fix; non-null qwen masks are rejected before pipeline load because mflux has no qwen mask path; incomplete q3/q6 bundles are pending.
  - Pipeline: noise (flux-style pack, 1,hw,64) → loop[CFG: pos+neg transformer passes → mflux guidance rescale (`combined = neg + g*(pos-neg)`, then rescale to positive-noise norm) → FlowMatch step] → unpack → 5D → VAE decode → PNG. **timestep passed = RAW sigma** (`QwenTimesteps` applies ×1000 internally — see §6 bug 2). ~20 steps, guidance ~4 (CFG).

Full per-model transcription specs are in `docs/FLUX_SCHNELL_PORT_PLAN.md` and `docs/QWEN_IMAGE_PORT_PLAN.md` (grounded from the mflux Python source).

---

## 6. Bugs found & fixed (don't reintroduce these)
1. **Conv weight layout.** mflux stores conv weights in **MLX channels-last** `(out, [kt,] kh, kw, in)`, NOT PyTorch `(out, in, k...)`. Assuming PyTorch → wrong reshape/transpose → load-time crash (`reshape 442368→(1152,1)`). Linear weights ARE PyTorch `(out,in)`.
2. **Qwen timestep double-scale.** mflux passes the raw sigma to `QwenTimesteps(scale=1000)` which multiplies internally. Passing `sigma×1000` double-scales → the transformer denoises to pure noise. Pass the **raw sigma**.
3. **Qwen-edit conditioning grid.** mflux computes a 1024-area VAE plan for the output target but passes `vl_width/vl_height` into `QwenEditUtil.create_image_conditioning_latents`; when present, the source-image conditioning VAE encode uses those VL dimensions. Swift must use `vlWidth/vlHeight` for qwen-edit conditioning latents, not `vaeWidth/vaeHeight`. The fixed square q4 path is 384x384 -> 24x24 -> 576 static tokens, not 1024x1024 -> 64x64 -> 4096 tokens.
4. **Qwen-edit multi-image semantics.** mflux accepts ordered `image_paths`, uses the last path for sizing, and concatenates per-source prompt-image features and VAE conditioning latents. Swift mirrors that with `ImageEditRequest.sourceImages`; do not collapse multiple sources into one prompt image or post-blend outputs.
5. **Indexed safetensor completeness.** A component with “some safetensors” is not enough. `Qwen-Image-Edit-mflux-q3` looked loadable until live load failed on missing `text_encoder/3.safetensors`; scanner readiness now validates `*.safetensors.index.json` shard references.
6. **Model resolution / quant collision.** `MLXStudioModelStore.resolve` normalized away the `-Nbit` suffix → requesting `...-8bit` loaded a co-installed `...-4bit`. Fixed with a literal case-insensitive directory-name match first. **osaurus must request the exact bundle directory name.**
7. **Tokenizer format** — see §3 (need fast `tokenizer.json`).
8. **GPU watchdog** — MLX mmaps weights lazily; running gen with weights on a **slow volume (USB)** stalls the Metal command buffer → `kIOGPUCommandBufferCallbackErrorTimeout`. **Stage weights on the internal SSD.**

---

## 7. RULES (Eric's, non-negotiable)
- **No AI attribution** in any commit/PR/GitHub-visible content (no `Co-Authored-By: Claude`, no "Generated with"). All commits are Eric's.
- **Live-prove everything.** Do not claim a model/feature works until you've generated a real image and visually verified it's coherent + prompt-accurate (+ deterministic + prompt-sensitive). "Builds clean" and "stats are finite" are necessary but NOT sufficient.
- **No fake guards / fake behavior.** Real fixes only.
- `jjang-ai/wiki` is a PRIVATE repo — never copy wiki content into project repos. Never store secrets in the wiki.
- Don't push `vmlx-swift` to the `osaurus-upstream` remote (DO_NOT_PUSH). Use `vmlx-origin` for the monorepo, `origin` for vmlx-flux.

---

## 8. osaurus integration (for the UI/server team)
- `docs/OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` — engine API (`FluxEngine` actor: load/generate/edit/upscale), `ImageGenRequest`/events, model registry, per-model status, the **required MetalGate exclusion** (image-gen MLX eval races LLM eval on the shared Metal command buffer — same SIGABRT hazard as the Model2Vec embedder, osaurus PR #1507 — so gate it), quant matrix, gotchas.
- `docs/OSAURUS_IMAGE_API_SPEC.md` — UI-facing HTTP contract: `GET /v1/images/models`, `POST /v1/images/{generations,edits,upscale}`, every request setting (prompt/negative/steps/guidance/strength/size/seed/n/format), and the SSE **progress events** (`queued`→`loading_model`→`step{step,total,progress,eta}`→`completed`) so the UI shows "Step N/M" and never looks stuck.
- The HTTP layer is a **proposed contract** — the engine is real, but the `/v1/images/*` endpoints aren't built in osaurus yet.

---

## 9. How to continue (concrete next steps)
1. **qwen-image-edit:** the q4/q5 single-image and ordered multi-image text-image edit paths are live-proven. Source-image conditioning now follows mflux's VL-size path (`vlWidth/vlHeight`) instead of the 1024-area VAE target grid, and multi-image uses mflux's ordered `image_paths` semantics. Current proof artifacts: `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`, `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (`latents_shape=1x576x64`, `image_ids_shape=1x576x3`), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json` (`combined_velocity_shape=1x1600x64`), `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-multi-image-live/Qwen-Image-Edit-mflux-q4-load.json`, and `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-live/Qwen-Image-Edit-mflux-q5-load.json`. Current non-null qwen masks are rejected before pipeline load; keep qwen masks hidden unless upstream mflux adds a real qwen mask path or a separate fill/inpaint model is wired.
   - Current staged bundle is already present at `~/.mlxstudio/models/image/Qwen-Image-Edit-mflux`; use `Qwen-Image-Edit-mflux-q4` or `Qwen-Image-Edit-mflux-q5` for current Osaurus wiring. Keep q3/q6 hidden/blocked until their indexed shards/components are complete.
2. **Ideogram 4:** `cocktailpeanut/ideogram-4-fp8` is staged locally, scans complete, load-validates required sentinel keys, and runs native fp8 generation. Typography proof exists at `docs/local/vmlx-flux-probes/2026-06-16-ideogram-fp8-native-gen20-current-source/ideogram-4-fp8-load.json`; current a188 strict object-icon proof exists at `docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-strict/ideogram-4-fp8-load.json` (apple/repeat SHA `c62b3b71a82ebcb0964be709c03678271364d381dd4ae8029af7b85d4bf02264`, mountains SHA `d193163f8584ad6040bc71d42960c98ac7864391f76f79c485cf8eca6905b2c1`, viewed clean). Boundary: `docs/local/vmlx-flux-probes/2026-06-16-current-a188-ideogram-fp8-object-determinism/ideogram-4-fp8-load.json` hallucinated text on a broader "no text" apple prompt, so keep normal UI/API wording scoped to typography and strict object-icon test coverage. Official `ideogram-ai/*` access remains approval-gated; remaining quant work is nf4 if that bundle is used. Ref: `/tmp/mflux-ref/src/mflux/models/ideogram4/`.
3. **Full precision** flux/z-image: download, run the probe — existing pipelines (`MFluxLinear` handles non-quant). Should just work.
4. **Osaurus app/server bridge:** the consolidated vMLX work is already on
   `osaurus-ai/vmlx-swift` main. Next osaurus-side work is the `/v1/images/*`
   bridge, model list/capability mapping, progress SSE, output file policy, and
   `MetalGate` exclusion. Pin Osaurus to current `vmlx-origin/main` after this
   docs/probe refresh; the minimum runtime-proof baseline is
   `a188a2ccecc92c8a5993506acc83df16f83c7420`.

**Reference:** the mflux Python source (the source of truth for every arch + weight key) is at `/tmp/mflux-ref` (clone of `github.com/filipstrand/mflux`). Re-clone if gone.

---

## 10. GH PR / commit references
- `osaurus-ai/vmlx-swift` **PR #63** — z-image engine vendored + merged to main (`36aebd42→90e64687`).
- `osaurus-ai/vmlx-swift` **PR #64** — flux-schnell, qwen-image,
  qwen-image-edit q4/q5, and Osaurus image docs merged to main.
- `osaurus-ai/vmlx-swift` **PR #65** — staged Ideogram fp8 mirror status/docs
  merged to main.
- `osaurus-ai/vmlx-swift` **PR #66** — Ideogram fp8 `weight_scale` loader
  support and `unconditional_transformer` loader support merged to main.
- `osaurus-ai/vmlx-swift` **PR #67** — Ideogram fp8 load-time sentinel
  validation merged to main. Merge commit:
  `9f1faea11aee78f17041c5bed6da039e70c11d05`.
- `osaurus-ai/vmlx-swift` **PR #68** — current-main image proof docs merged to
  main. Merge commit: `66f328322c41ce51881a9ab3bb630c1aeee114b8`.
- `osaurus-ai/vmlx-swift` main commit **`a188a2cc`** — minimum runtime-proof
  baseline used for the current a188 proof/docs refresh.
- `jjang-ai/vmlx-flux` branch **`native-zimage-proven`** — standalone mirror of
  the native work. Current branch head includes the image source/docs mirror and
  root Osaurus image API spec.
- Wiki note (private `jjang-ai/wiki`): `notes/2026-06-15-vmlx-flux-native-z-image-proven-fork-lockstep.md`.
- Per-project memory: `~/.claude/projects/-Users-eric-vmlx-swift/memory/vmlx-flux-native-zimage-integration.md`.
- Proof artifacts (gitignored): `docs/local/vmlx-flux-{outputs,probes}/` (PROOF-*, FLUX-proof, QWEN-proof, Q8b-*).
