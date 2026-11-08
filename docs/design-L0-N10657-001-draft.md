# Design: L0-N10657+ activation test

### Claim

L0-N10657+ peaks above its atlas 99th percentile (pos_q99 = 0.9822) at frequent lowercase English content-word morphemes (" image", " modern", " use"), and stays below that threshold at ALL-CAPS variants of the same words, capitalized names, and foreign/rare tokens.

### Unit and metric

Unit: one prompt with one `[[ ]]`-marked target token, 24 prompts total (12 should-fire, 12 should-not-fire). Metric: the positive side's activation at the target token, compared against pos_q99 = 0.9822; pass per EXPERIMENTS.md type A (>=80% of each group on the right side) with AUROC reported.

### Controls

Near misses: ALL-CAPS variants of the very same words (" IMAGE", " SIMPLE", " USE") — same letters, different Marin token, so passing cannot be explained by letter content. Random neurons: checked 3 random L0 neurons' top positive exemplars in the atlas (N3903: "lo"/"to" fragments; N10468: "mann"/"ius" name pieces; N16467: "ba"/"ball" repeats) — none show the frequent-whole-morpheme pattern, so it is not generic to the layer. Other side: the negative side (neg_q99 = 0.269) is expected to stay near zero on all 24 prompts, since none of them are rare-subword or non-English tokens.

### Data

Evidence prefix: pilot/fineweb-10m-sparse (pole_stats, exemplars joined to tokens.parquet, logit_lens decoded with marin-community/marin-32b-base tokenizer). Prompts are freshly written sentences, not exemplar text; target words come from the exemplar set because they are the hypothesis, contexts are new. Every [[target]] verified single-token with the Marin tokenizer. Artifact watch: no document-start positions, no mid-word subword targets (the rare-token group is deliberately whole standalone tokens like " Tribune", " İstanbul").

### Noise

12 per group allows 2 misses at the 80% rule; margin analysis: should-fire exemplar activations cluster 1.05-1.09 vs threshold 0.9822 (~8% headroom), should-not-fire tokens are the negative side's own exemplar class (expected near 0, ~4x below threshold). Borderline risk accepted for a first cheap test; if a flip lands within noise, follow with a contrast test (type D) at 30 prompts per group before revising the explanation.

### Readout

EXPERIMENTS.md type A pass rule: >=80% should-fire above pos_q99 AND >=80% should-not-fire below it; AUROC reported by the runner. Refutation: >2 should-fire prompts below threshold or >2 should-not-fire above it. If refuted, the likely revision is that the side tracks general lexical familiarity rather than morpheme frequency specifically — re-scope the explanation and request a contrast test separating frequency from casing.
