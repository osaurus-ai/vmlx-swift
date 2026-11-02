# Restore the native DSV4-0731 Low effort prefix

Status: source prepared; syntax/diff checks passed; focused execution pending.
This follows the automatic reasoning-floor/diffusion-mask correction. No real
model inference, consumer deployment, release, or speed claim is included.

The [official 0731 encoder at revision b9f75fec23eda3cdf0b102611d841d06c37ef1f3](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/b9f75fec23eda3cdf0b102611d841d06c37ef1f3/encoding/encoding_dsv4.py#L59)
defines Low as the empty default prefix, preserves distinct High/Max text, and
adds effort text only to the first message in thinking mode. The runtime's added
Low directive originated in commit `d117eac52d7b8ae3c208c69d2062371e89378f11`;
it tried to force brief visible reasoning rather than following that baseline.

This change restores Low's empty prefix in `DeepseekV4ChatEncoder` and the
embedded `ChatTemplateFallbacks.dsv4Minimal` string. The standalone
`DSV4Minimal.jinja` already has the correct empty Low prefix. Native High/Max
text, the Low/High/Max effort vocabulary, Off/chat and thinking markers, parsing,
and explicit caller controls remain intact. Low is a native effort setting;
it does not guarantee a short answer or a visible reasoning block.

Independent pinned native-text fixtures compare the complete prompt bytes from
Swift, embedded Jinja, and standalone Jinja for absent/Low/High/Max effort in
thinking and chat modes. The same fixture verifies that native High/Max changes
only the first prefix of a multi-turn tool transcript; native reasoning, tool
calls, results, and continuation remain present. Prior tests asserting added Low
coercion have been replaced with exact empty-prefix expectations. Existing
V4.1 family-isolation and agent-loop boundary tests remain part of the focused
validation scope.

Removing the Low directive changes prompt bytes, so old exact-token prefixes
must not match the corrected Low transcript. No caches or evidence were deleted.
The pre-existing growing-agent-loop boundary suite needs execution with this
change; its scalar-token fixture is not a replacement for live BPE/cache proof.

V4.1's numeric effort code and raw absent-context fallback50 are unchanged in
this patch. The separately recorded fallback risk must not be confused with the
current Osaurus JANG path, which reads the bundle's declared High default and
passes High75 through the LLM/VLM context. No capability ladder is removed.

Remaining: focused encoder/policy/fallback/agent-loop tests, root review and
normal PR/CI/merge, consumer pin/build checks, and app composer/agent-editor
multi-turn/tool qualification. Source and synthetic template tests do not prove
real-model quality, speed, or cache behavior.
