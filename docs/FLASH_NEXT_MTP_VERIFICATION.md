# Flash Next native MTP verification

Native MTP verifies proposals against the target model. Greedy verification must reproduce ordinary autoregressive decoding; increasing accepted proposals is not a correctness substitute. Product controls remain Off and Adaptive. The sampler and model template are unchanged.

## Arithmetic scope

`FlashVerificationScope` is established only by the Flash target-verification entry point, through the output head. It admits batch one and two through eight verification rows. Ordinary prefill, AR, other model families and draft-head execution do not establish this scope.

Within supported shapes and dtypes, verification preserves the AR projection and reduction paths:

- Dense hyperconnection down and injection projections share the same joined bank as AR, including its output geometry. Computing injection separately can round differently.
- The dense router uses the existing compiled single-row computation for each verification row, preserving both selected indices and scores. Expert computation remains batched.
- Qualified affine projections, GDN input projections and tails, QSA evaluation, and hyperconnection projection paths retain their matching AR arithmetic. Unsupported inputs use existing fallback paths.
- Mapped JANGH expert execution retains the AR route-count decision, including its existing threshold.
- Early layer submission schedules available work without changing tensor expressions, sampler decisions or cache commits.

Live and stored cache precision is unchanged. This change does not introduce a paged RAM prefix tier or replace SSD cache coordination.

## Depth is not verification width

One primary token plus D drafts produces D+1 verification rows. The current eight-row qualification scope therefore covers up to seven drafts. This is an arithmetic qualification boundary, not a claim that seven drafts are fastest. Adaptive policy must evaluate confirmed tokens per elapsed time and retain its AR safety checks. A higher configurable ceiling does not prove the resulting wider arithmetic path.

## Regression requirements

Check exact logits/hidden state and canonical cache state, not only the final greedy token. Include every accepted prefix, typed full-precision restore, subsequent AR continuation, repeated commit cycles and actual naturally generated trajectories. Two otherwise passing fixture sets missed rare rounding boundaries: joined-versus-separate injection and compiled-versus-eager routing.

Also run natural AR/Adaptive comparisons with identical inputs and explicit benchmark sampling controls; require natural completion and exact greedy output. Record requested and admitted depth separately. Run without runtime DEBUG compilation or diagnostic switches before integration. Application SSD coordination, media, other quant layouts and sustained performance require their own evidence; operator and model-only tests do not establish those claims.
