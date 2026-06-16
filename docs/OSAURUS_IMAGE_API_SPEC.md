# osaurus Image Generation API — UI wiring spec

**Audience:** osaurus team building the image-generation UI.
**Goal:** define the HTTP API the UI calls — every setting a user can send (prompt,
negative prompt, steps, strength, guidance, size, seed, …), the model list +
per-model defaults/capabilities, and the **streaming progress events** that drive a
determinate progress bar so users always see "step N / M" and never a stuck spinner.

**Backed by:** the native `vMLXFlux.FluxEngine` (vendored in vmlx-swift). Every
field below maps to an engine request/event — see the mapping notes. This is the
contract osaurus implements server-side and the UI builds against.

> Status: the engine is real on Osaurus `vmlx-origin/main` runtime-proof
> baseline `a188a2ccecc92c8a5993506acc83df16f83c7420`. Fresh live proof exists
> for `z-image-turbo` 4/8-bit, `flux1-schnell` 4/8-bit, `qwen-image` 4/6-bit,
> `qwen-image-edit` q4/q5, staged `ideogram-4-fp8`, and staged
> `ideogram-4-nf4`. Load matrix:
> `docs/local/vmlx-flux-probes/2026-06-16-current-a188-load-matrix/compatibility-matrix.json`.
> Generation/edit artifact roots:
> `docs/local/vmlx-flux-probes/2026-06-16-current-a188-zimage-4bit-gen/`,
> `2026-06-16-current-a188-zimage-8bit-gen/`,
> `2026-06-16-current-a188-flux-schnell-4bit-gen/`,
> `2026-06-16-current-a188-flux-schnell-8bit-gen/`,
> `2026-06-16-current-a188-qwen-image-4bit-gen20/`,
> `2026-06-16-current-a188-qwen-image-6bit-gen20/`,
> `2026-06-16-current-a188-qwen-edit-q4-gen20/`,
> `2026-06-16-current-a188-qwen-edit-q5-gen20/`, and
> `2026-06-16-current-a188-ideogram-fp8-object-strict/`.
> NF4 follow-up artifact:
> `docs/local/vmlx-flux-probes/2026-06-16-ideogram-nf4-strict-object/`.
> Expose only proven local variants for normal testing. Keep qwen-edit q3
> blocked because its text-encoder index references missing
> `text_encoder/3.safetensors`; keep q6 blocked until its local bundle is
> complete; hide qwen mask/inpaint controls because the mflux qwen-edit
> reference exposes no qwen mask path. qwen-edit q5 is cleanest; q4 is
> deterministic and color-prompt-sensitive but weaker on shape-changing edits.
> Ideogram fp8 is implemented/testable for the staged mirror with typography
> and strict clean object-icon proof. Ideogram NF4 is implemented/testable for
> the staged `cocktailpeanut/ideogram-4-nf4` mirror with strict 512px
> object-icon proof: apple/repeat SHA
> `76cd995b90d4ad85140418ae1d3a8a44bc688d03840041ff93ff2cd006e748df`,
> mountains SHA
> `302ffe06596c718df6a118a56bcc0e8ec7437edee1dc9ba1656d0cd5d2052425`.
> Boundary: a broader current-a188 fp8 "no text" apple prompt hallucinated
> text, so do not present Ideogram as a general clean object renderer yet.
> Official `ideogram-ai/*` downloads still require approval for the current
> account (`hf download ... --dry-run` returned access denied for both fp8 and
> nf4 on 2026-06-16).
> Multi-image qwen-edit proof: vMLX source includes ordered
> `ImageEditRequest.sourceImages`. q4 artifact:
> `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-multi-image-live/Qwen-Image-Edit-mflux-q4-load.json`;
> q5 artifact:
> `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-live/Qwen-Image-Edit-mflux-q5-load.json`;
> internal q5 shape proof:
> `docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-multi-image-shape-proof/Qwen-Image-Edit-mflux-q5-load.json`;
> viewed contact sheet:
> `docs/local/vmlx-flux-outputs/2026-06-16-qwen-edit-multi-image-contact-sheet.png`.
> The HTTP surface below is the **proposed contract** for the osaurus team to
> expose; design it once, wire all models through it.

---

## 0. Endpoints

| Method + path | Purpose | Engine call |
|---|---|---|
| `GET  /v1/images/models` | List installed image models + defaults + capabilities | `MLXStudioModelStore.scan()` + `ModelRegistry` |
| `POST /v1/images/generations` | text → image | `FluxEngine.generate` |
| `POST /v1/images/edits` | image(s) (+mask where supported) + prompt → image | `FluxEngine.edit` |
| `POST /v1/images/upscale` | low-res → high-res | `FluxEngine.upscale` |
| `POST /v1/images/cancel` | cancel an in-flight job | cancels the consuming `Task` |

Base path mirrors OpenAI's Images API so existing client SDKs mostly work; the
streaming + extra fields below are osaurus extensions. Auth/headers identical to the
osaurus LLM endpoints.

---

## 1. `GET /v1/images/models` — what the UI populates dropdowns from

The UI should NOT hard-code model names or limits — fetch them. Response:

```jsonc
{
  "data": [
    {
      "id": "z-image-turbo",              // canonical name (use in requests)
      "display_name": "Z-Image Turbo",
      "kind": "imageGen",                  // imageGen | imageEdit | imageUpscale
      "ready": true,                       // false => weights not fully staged
      "quantization_bits": 4,              // null = full precision
      "capabilities": {
        "text_to_image": true,
        "image_edit": false,               // show/hide the edit panel
        "upscale": false,
        "negative_prompt": true,           // show/hide negative field
        "mask": false,                     // show/hide mask tool
        "multiple_source_images": false,   // show/hide multi-reference upload
        "lora": false
      },
      "defaults": { "steps": 4, "guidance": 0.0 },   // pre-fill the form
      "limits": {
        "min_steps": 1, "max_steps": 50,
        "size_multiple": 16,               // width/height must be divisible by this
        "max_pixels": 1048576,             // 1024*1024 — clamp the size slider
        "supported_sizes": ["512x512","768x768","1024x1024"]
      }
    }
    // ... flux2-klein, qwen-image, qwen-image-edit, seedvr2, etc.
  ]
}
```

Notes for UI:
- `ready:false` → show the model greyed with a "Download required" CTA (no silent
  downloads — the user must stage weights first).
- `native_runtime_status != native_pipeline_implemented` → disable normal user
  actions and show the blocker text. `native_pipeline_partial` is an internal
  diagnostic state, not a release-ready user model. For qwen-image-edit, the
  status is variant-specific: `Qwen-Image-Edit-mflux-q4` and
  `Qwen-Image-Edit-mflux-q5` are implemented/testable; q3/q6 remain partial or
  blocked until complete local bundles are staged and proven.
- Use `capabilities` to show/hide fields. e.g. `negative_prompt:false` → hide the
  negative box; `mask:false` → no mask tool;
  `multiple_source_images:true` → allow ordered multi-reference upload.
- Pre-fill `steps`/`guidance` from `defaults`; clamp sliders with `limits`.

---

## 2. `POST /v1/images/generations` — every setting the user can send

```jsonc
{
  "model": "z-image-turbo",      // REQUIRED. canonical id from /v1/images/models
  "prompt": "a red apple on a wooden table",   // REQUIRED

  "negative_prompt": "blurry, deformed, low quality",  // optional; only used if guidance > 0
  "n": 1,                         // images to produce (numImages)

  // size — accept EITHER "size" OR explicit width/height
  "size": "1024x1024",            // OpenAI-style; or:
  "width": 1024,                  // must be a multiple of limits.size_multiple (16)
  "height": 1024,

  "steps": 8,                     // denoising steps (more = slower, finer)
  "guidance": 0.0,                // a.k.a. cfg / guidance_scale. 0 = no CFG (turbo
                                  //   models). >0 enables classifier-free guidance
                                  //   (uses negative_prompt; ~1.8x slower — 2 passes/step)
  "seed": 7,                      // optional. omit/null = random. Same seed+prompt+
                                  //   settings is byte-deterministic (good for "lock seed")

  "response_format": "url",       // "url" (saved file path/URL) | "b64_json"
  "output_format": "png",         // png | jpeg | webp  (png is fully wired today)
  "stream": true                  // true => SSE progress stream (see §5). RECOMMENDED.
}
```

**Field → engine mapping** (`ImageGenRequest`): `prompt`, `negative_prompt`→`negativePrompt`,
`width`/`height` (or parsed from `size`), `steps`, `guidance`, `seed`, `n`→`numImages`,
`output_format`→`outputFormat`. Server sets `outputDir`.

**Validation the server enforces (surface these as 400s the UI shows inline):**
- `width % size_multiple == 0` and `height % size_multiple == 0` (else
  `invalid request: ... must be divisible by 16`).
- `steps >= 1`.
- `width*height <= limits.max_pixels`.
- `model` exists and `ready`.

**Non-streaming response** (`stream:false`) — OpenAI-shaped:
```jsonc
{ "created": 1750000000,
  "data": [ { "url": "file:///.../z-image-abc.png", "seed": 7 } ] }
```

---

## 3. `POST /v1/images/edits` — image(s) + prompt (+ optional mask for models that support it)

```jsonc
{
  "model": "Qwen-Image-Edit-mflux-q4",
  "prompt": "make the apple green",
  "image": "data:image/png;base64,....",   // REQUIRED. source image (b64 or URL)
  "images": [
    "data:image/png;base64,....",
    "file:///Users/me/reference-shirt.png"
  ],                                      // optional ordered list. If present,
                                          // server maps this to sourceImages.
  "mask":  "data:image/png;base64,....",   // optional. white=edit, black=keep
  "strength": 0.75,               // 0..1 — how far to deviate from the source
                                  //   (0 = barely change, 1 = ignore source)
  "negative_prompt": "...",
  "steps": 20,
  "guidance": 3.5,
  "seed": 7,
  "width": null, "height": null,  // null => match source dimensions
  "response_format": "url",
  "stream": true
}
```
Maps to `ImageEditRequest` (`sourceImage` legacy single-image compatibility,
`sourceImages` ordered multi-image list, `mask`, `strength`, ...). Only models
with `capabilities.image_edit:true` accept this; else 400 `wrong model kind`.
For qwen-edit, send either `image` or `images`; if both are present, prefer
`images` and preserve order. The qwen-edit runtime uses the last source image for
the mflux aspect-ratio sizing plan and all source images for Qwen-VL prompt
features plus VAE conditioning latents.

Current live-proven targets are `Qwen-Image-Edit-mflux-q4` and
`Qwen-Image-Edit-mflux-q5` without masks. Both accept ordered multi-image
inputs through `sourceImages`. q4 multi-image proof has turn 1/3 SHA
`e43910a505ab090bfbd4ec3a00f6e58fcd97df65c6b61e6b973754511bc740be` and turn 2
SHA `16ecc1fec4bdff1e5aecb9c0875569b2bebed53a81c686bf23e806faa6e2b893`; q5
multi-image proof has turn 1/3 SHA
`ec2e49d6f300849cb46940b793670bee007f4aae8f04e97a31289670758519c9` and turn 2
SHA `8dfacb52aa81c6e0a8a6827c4377f27bc5d9396e39a4f8662a47db93798b767a`.
Reject a non-null `mask` with 501 or hide the control for qwen-edit. The engine
currently enforces this by emitting a failed event before the edit pipeline loads;
`QwenImageEditSupportTests.testQwenImageEditRejectsMaskBeforePipelineLoad`
covers that contract, and `vmlxflux-probe --edit --mask-image <png>` records the
same failed-event behavior against staged local bundles.

---

## 4. `POST /v1/images/upscale` — SeedVR2

```jsonc
{ "model": "seedvr2", "image": "data:image/png;base64,...",
  "scale": 4, "steps": 10, "seed": 7, "response_format": "url", "stream": true }
```
Maps to `UpscaleRequest` (`scale` 2 or 4).

---

## 5. Streaming progress (SSE) — the progress bar contract

Set `"stream": true`. The server responds `Content-Type: text/event-stream` and
emits one JSON object per `data:` line. **This is what keeps the UI from looking
stuck** — events fire continuously through load + every denoise step.

Event sequence:
```
data: {"type":"queued","job_id":"img_abc"}
data: {"type":"loading_model","model":"z-image-turbo"}            // first use / model switch
data: {"type":"step","job_id":"img_abc","step":1,"total":8,"progress":0.125,"eta_seconds":3.4}
data: {"type":"step","job_id":"img_abc","step":2,"total":8,"progress":0.250,"eta_seconds":2.9}
...
data: {"type":"preview","step":4,"image":"data:image/png;base64,..."}   // OPTIONAL partial decode
...
data: {"type":"step","job_id":"img_abc","step":8,"total":8,"progress":1.0,"eta_seconds":0.0}
data: {"type":"completed","job_id":"img_abc","images":[{"url":"file:///...png","seed":7}]}
data: [DONE]
```

Error / cancel terminal events:
```
data: {"type":"error","message":"...","hf_auth":false}     // hf_auth:true => show "Add HF token" CTA
data: {"type":"cancelled","job_id":"img_abc"}
```

**Event → engine mapping** (`ImageGenEvent`): `step`→`.step(step,total,etaSeconds)`,
`preview`→`.preview(pngData,step)`, `completed`→`.completed(url,seed)`,
`error`→`.failed(message,hfAuth)`, `cancelled`→`.cancelled`. `progress` is
`step/total` (server-computed convenience). `queued`/`loading_model` are server-side
lifecycle wrappers the engine doesn't emit but the UI wants.

### UI guidance for "not stuck"
- Render a **determinate** bar from `progress` (= `step/total`); label "Step N / M".
- Show `eta_seconds` as "~Ns left" (it's a rolling estimate; smooths after step 1).
- Before the first `step`, show the `loading_model` phase (first run can take a few
  seconds to page weights from disk) — distinct from "generating" so users know
  it's loading, not frozen.
- If `preview` events arrive, swap the thumbnail each one for a live denoise preview.
- Typical cadence (z-image-turbo 4-bit, M5 Max): a step every ~0.5 s @ 512², ~2.5 s
  @ 1024². So a `step` event lands at least every few seconds — if none arrives for
  >~15 s the UI may treat it as stalled.

---

## 6. `POST /v1/images/cancel`
```jsonc
{ "job_id": "img_abc" }
```
Cancels the in-flight job (engine checks cancellation each step → stops at the next
step boundary, emits `cancelled`). UI: wire to a Stop button next to the progress bar.

---

## 7. Settings cheat-sheet (what each control does, for tooltips)

| UI control | Field | Meaning | Sane default |
|---|---|---|---|
| Prompt | `prompt` | what to draw | — |
| Negative prompt | `negative_prompt` | what to avoid (only with guidance>0) | empty |
| Steps | `steps` | denoise iterations; more = finer, slower | model default (4 for turbo) |
| Guidance (CFG) | `guidance` | prompt adherence; 0 for turbo models, 3–7 for guided | model default |
| Strength (edit) | `strength` | how much to change the source (0–1) | 0.75 |
| Size | `size`/`width`×`height` | output resolution (mult. of 16) | 1024×1024 |
| Seed | `seed` | reproducibility; lock to keep a result | random |
| Count | `n` | how many images | 1 |
| Format | `output_format` | png/jpeg/webp | png |

---

## 8. Server-side wiring notes (for whoever implements the endpoints)
- One `FluxEngine` actor per process; `load(name:from:)` resolves the local mflux
  bundle and load-if-different. No silent downloads.
- **MUST gate image-gen MLX eval through the same MetalGate exclusion used for the
  Model2Vec embedder** — image gen is a second MLX graph and races LLM token
  generation on the shared Metal command buffer (→ SIGABRT). Acquire around the
  event-drain, release after `completed`. See `OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` §7.
- Stage image weights on the **internal SSD** — USB-resident weights trip the GPU
  watchdog on the first forward pass.
- Map `FluxError` → HTTP: `unknownModel`/`localModelNotFound`→404,
  `localModelIncomplete`→409 (with reasons), `invalidRequest`/`wrongModelKind`→400,
  `notImplemented`→501, `failed(hfAuth:true)`→402-style "token needed".

See `OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` for the full engine API + per-model
status, and `QWEN_IMAGE_PORT_PLAN.md` for qwen-image/qwen-image-edit port notes.
