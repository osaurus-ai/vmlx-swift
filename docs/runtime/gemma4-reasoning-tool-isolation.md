# Gemma 4 reasoning and tool transport

Gemma 4's chat template closes `<|channel>thought` with `<channel|>` before
rendering tool calls. A historical call quoted inside that thought channel is
reasoning, not a new executable call.

`ToolCallFormat.parsesToolCallsFromReasoningChannel` owns this boundary. Gemma 4
bypasses `ToolCallProcessor` while reasoning; after the channel closes, ordinary
content parsing still accepts native envelopes and the existing bare-call
fallback. The shared processor must never consume a reasoning quotation or
retain its partial arguments for the following content call.

## Failure and controls

Real quantized E4B raw-token captures quoted `calculate{expression: "29*17+5"}`
while recalling an earlier result, then gave a correct answer. Replaying those
unchanged bytes through the previous runtime extracted an unintended call. Large
chunks additionally absorbed the quoted result object into malformed arguments.
This identifies a routing bug independently of stochastic generation quality.

`Gemma4ReasoningToolIsolationTests` covers quoted bare history, quoted native
transport, a real content call after quoted history, and the preserved content
bare-call fallback. Each crosses eight chunk sizes, including a whole-response
chunk. The four new test functions failed with 56 assertions on the old routing
and passed after the change. The focused source harness also passes the existing
Gemma prose, reasoning/content transition, drifted-envelope and no-tools tests
(18 test functions, six suites).

The source harness uses the production parsers, reasoning parser and routing
files, with a diagnostic event-enum adapter. It is not a complete MLX package or
native-app qualification. Captured model generation used the production runtime,
bundle sampling defaults and natural stops; its results remain distinct from
parser replay. This change does not establish the cause of a GPU error or prove
all model output correct. No sampler, template, model math, cache policy or UI
behavior is changed.
