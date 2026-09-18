# Preserve Gemma tool-selection history

## Source-bound defect and change boundary

Gemma4Processor's required/named-tool adapter deletes earlier user turns and completed tool protocol, but image processing still consumes every original image. A real isolated Osaurus E2B8bit run reproduced two images with only one image placeholder: maskedScatter rejected860160 vision values versus430080 positions before decoding. An earlier-image-only follow-up silently lost the image entirely. Identical auto requests retained history; required and named requests did not.

Origin:2be648a3c12817b1d57e06c86d0b9987b038bcb3 introduced tool compaction;447d2a07c6f0e0da3a4f59e35e8221417bc9e096 added earlier-user deletion and a separate text-only adapter. The old tests assert this lossy behavior rather than the complete-request contract. This correction replaces those expectations with production-processor tests. It does not undo the later scalar system-content repair, structured tool-call parsing, media conversion, model defaults or cache repairs.

Plan: preserve the complete message list at both Gemma processor consumers, retain tool_choice/tool_choice_name as template arguments, and test exact history/metadata plus image-slot/pixel cardinality. No prompt summaries, coercion, sampler changes, model-name allowlists or cache bypasses. The text processor becomes internal (not public) so regression tests can exercise the real prepare method.

## Baseline and acceptance

Baseline app530c2f12e8afd815c8d578fceeb73460b5443bec, binary63699d39b89c09e671928fb177746f47efccdcff2f3950638798ab4fe990fecd, engine8ba593aff16c13cf526211b8477c0a037f0122af; E2B8bit snapshot433003a1e3fbfd10819ad15179d5e3c4d02d7ea7. Full catalogs: AgentLoop40/56passed,10failed,2errored,4skipped; Frontier23/42passed,16failed,3errored. Every nonpass retained/reviewed; no universal pass claim. Private evidence: handoff-parity-2026-09-16/implementation/REQUIRED-MEDIA-HISTORY-FAILURE.md and EVAL-REVIEW-530.md.

## Correction and executable regressions

Implementation `055f0137df946486ef559f97e1a9b3425522f12e` removes only the two destructive compaction consumers/helpers. Tool selection arguments, scalar system content, original tool calls/results, reasoning, media processing, generation defaults and cache paths remain intact. The original closed/unanswered-tool tests now call the production processor and assert exact retained protocol instead of synthetic summaries.

At regression-only `8b8f90201d4937eca23b441a53ab611cbd52488f`, 18 of 33 parameter cases failed: 14 required/named VLM history cases and 4 text-history cases. All 11 auto cases and 4 pending-tool controls retained their contracts. The VLM failures include zero slots for one image and four slots for two images requiring eight. No production correction was present in this baseline.

At corrected `055f0137`, all 33 cases passed. The combined selected suites completed with 82 Swift Testing tests in 8 suites plus 12 XCTest cases, zero failures, on two runs. The second finished its Swift Testing assertions in 0.071 seconds without lock recovery. Suites: Gemma4RequiredHistoryTests, ChatMessageToolCallTests, Gemma4ImageEdgeCaseTests, CacheCoordinatorMediaSaltTests, Gemma4CacheTopologyTests, ToolCallEdgeCasesTests, ThinkingTemplateDefaultTests, Gemma4TemplateFallbackSourceTests and Gemma4ToolFallbackRoutingSourceTests.

Artifacts under private `handoff-parity-2026-09-16/implementation/`:

- `gemma-history-8b8f90201d4937eca23b441a53ab611cbd52488f-red4.log`: clean failing baseline.
- `gemma-history-055f0137df946486ef559f97e1a9b3425522f12e-green2.log`: clean corrected repeat.
- `SWIFTTEST_GemmaHistoryRed4__184233.*`, `SWIFTTEST_GemmaHistoryGreen1__184556.*`, `SWIFTTEST_GemmaHistoryGreen2__184831.*`: original resource limits and zero-survivor cleanup receipts.

Retained non-acceptance attempts: red1 stopped on a copied nonrelocatable compiler cache; red2 stopped on fixture Sendable typing; red3 also exposed one-ULP CoreImage color-conversion noise (pixel tolerance is now 1e-6, message/slot counts remain exact). Green1 exercised the existing 90-second abandoned-test-semaphore recovery; its sample and full log are retained. No test-lock implementation or runtime settings were changed.

## Combined live checkpoint and remaining limits

App6267660b2811b7e8bdc13573ef88039d3501e61e pinned engine29e681dfc25e0afa114fcde4a886d77ecc244323; binary3f7f49e4b2e980c506c56b8553ccff772062477ce210416c2f96b306e65a31e8. Same E2B8bit snapshot above, bundle T1/P.95/K64/minP0, MTP/thinking off. Focused app314tests/17suites had zero failures.

Native UI17 now completes the old second-image required-tool failure: first image/write, required read with grounded result, second image/write, reopened-history read, actual Chrome new_page/screenshot, then required write using that returned image.13generations, normal stops, correct saved captions, settled cards/unlocked input,80.95–87.90tok/s. Slot cardinality1/280,2/560,3/840. Four auto continuations accepted explicit disk boundaries; required-tool turns still skip warm restore under the existing guard. Effective fp16,3full+12rotating layers,pagedRAMoff,TQKVlayers0. No fsync or every-generated-boundary durability claim.

Fourteen complete/stream/auto/named/required API rows return valid tool calls89.61–92.08tok/s. Eleven captions correct; three duplicate-image captions hallucinate black square (also in baseline auto), so not14visual passes. Actual tool execution is proved separately by native UI17.

Full corrected catalogs: AgentLoop41P/10F/1E/4S of56; Frontier16P/22F/4E of42. All nonpasses/rubrics reviewed; empty finals after tool execution, malformed arguments, numbered-content copying, incomplete tasks and generated-code failures remain. Raw baseline Frontier23P/16F/3E differs; unseeded totals are not a causal quality verdict. Details and exact artifacts in private `EVAL-REVIEW-626.md` and `RUN17-REVIEW.md`, plus `run17-gemma-history-native-evidence/`, paired API directories and both eval626catalog sets. UI18 normal exit/zero survivors20:13:12, original isolated settings restored. Native kernel lifetime peak3,257,338,616bytes; Frontier peak4,584,770,488bytes; swap1.81GiB unchanged. No16GB, general video/audio, reporter freeze or whole-model reliability qualification.

App seven CI checks passed at6267660b (CLI dependency-I/O retry retained). Engine macOS/CUDA checks remained queued; four Linux builds passed and advisory full-tree formatting failed. Pending CI is not an all-pass result. This docs-only update adds receipts without changing tested runtime code. Current upstream `bfb34ff142817f3a35cf6502ad5d8dd742c4e87f` and baseline8ba593af share tree23f38cb58bdf043860b1fef012b8c3b52cdb111f.

A separate prefill-error propagation issue currently converts processing failure to cancellation/empty output and then a misleading app retry; it is traced but not corrected by this history patch.
