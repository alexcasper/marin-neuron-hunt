# 12k Interp Audit — does doubling steps fix the mush?

**Date:** 2026-11-06 · **Question:** last night's sprint (see `sae/results/interp-sprint-20261031/REPORT.md`)
verdict was "steps, not width". If that's right, the 44 topped-up layers at **step 12000** (2× the 6k finals)
should label cleaner than the good-20 at 6k. This audit tests it.

**Method:** identical pipeline, blind dual-model labeling (Gemini 2.5 Flash + Claude Code, no cross-reading),
same category set. 8 hard layers sampled from the 44-layer worst-FVU topup set (L12, 18, 26, 33, 42, 50, 55, 63),
step-12000 checkpoints, 200k-token spread samples per layer, 24 latents/layer
(12 top-mass + 12 random-alive; the fire-rate-gated mass track was dropped — proven a no-op in the sprint).
192 latents, 192 blind label pairs. Pipeline took 3 failed submissions to get right
(config.json vs cfg.json; `sae(x)` vs `sae.encode(x)` NamedTuple fields) — smoke-gated on the 4th.

## Headline: **2× steps did NOT move interpretability.**

| metric | 6k sprint (20 good layers) | 12k audit (8 hard layers) |
|---|---|---|
| unclear (gemini / claude) | 77–87% band | 81.8% / 85.4% |
| concept | ~1% | **0.0% / 0.0%** |
| document-identity | 15% / 10% | 14.1% / 13.5% |
| register-boilerplate | 4% / 1% | 3.1% / 0.0% |
| structural | 3% / 1% | 1.0% / 1.0% |
| category agreement | 83.5% | 89.1% |
| same-reading (jaccard≥0.5) | 11% | **0.5%** |
| positive-both | 11% | 12.0% |

**Per-layer** (n=24 each; unc = unclear rate gem/cla, docid gem/cla):

| layer | agree | unc gem | unc cla | docid gem | docid cla | band |
|---|---|---|---|---|---|---|
| L12 | .88 | .75 | .88 | .25 | .12 | early |
| L18 | .92 | .79 | .83 | .17 | .17 | early |
| L26 | .96 | .88 | .83 | .12 | .17 | early |
| L33 | .96 | .92 | .96 | .08 | .04 | mid |
| L42 | .88 | .88 | .83 | .12 | .12 | mid |
| L50 | .92 | .88 | .88 | .08 | .08 | late |
| L55 | .83 | .79 | .83 | .17 | .17 | late |
| L63 | .79 | .67 | .79 | .12 | .21 | late |

**Tracks:** mass latents (fire-rate median 5.0%, max 79%) → unclear 81/86%; random-alive (median ~0%, max 3.3%)
→ unclear 82/84%. Same as the sprint: random latents are alive but inert; the extra fire-rate headroom of mass
latents doesn't buy readability.

**Cross-layer universal features:** none. Exactly 1 repeated reading across layers, and it describes noise
("rare scattered tokens, very weak, unrelated") — not a feature.

## Reading

1. **The 89% agreement number is shallow.** The labelers agree on category mostly by agreeing "unclear".
   Wording-level agreement (what IS this latent?) collapsed from 11% to 0.5% — the few non-unclear calls at 12k
   are less convergent than at 6k, not more.
2. **Hard layers at 12k ≈ good layers at 6k ≈ mush.** If mush were an early-training artifact that 2× steps
   cures, the 12k topup layers should beat the 6k good layers. They don't — unclear is flat (81/85 vs 77–87),
   concepts went to zero, doc-identity unchanged. The binding constraint is NOT reachable by doubling steps at
   this scale.
3. **The one robust signal stays document-identity (~14%).** Same as 6k. The dictionaries reliably capture
   source-document memorization and nothing else that survives blind dual-labeling.
4. **No depth gradient.** Early/mid/late agreement bands are equal (0.92/0.92/0.85); L63 (final layer) is
   nominally most readable (unc .67/.79) but n=24 and its docid is the highest too — memorization, again.

## Implications for width/steps decisions

- **Do not greenlight the 64k rung on interpretability grounds.** With 0% concepts at 12k steps and no
  cross-layer universals, more width would add parameters to a dictionary whose only legible content is
  document memorization. The dusk 64k job remains ALEX_GO-gated; this audit gives no evidence the gate should
  open for interp value.
- **"Steps, not width" is now half-refuted:** doubling steps didn't help interp. What remains untested is
  ~an-order-of-magnitude more steps (or token exposure), which is a training-budget decision, not a
  hyperparameter tweak.
- **Neuron-hunt alignment:** the atlas's document-identity signal (the one thing that reproduces across both
  audits) is the lane's most reliable instrument; concept-level claims about these checkpoints should not be
  made from label audits at this training age.

## Artifacts

- Batches: `hunt/interp12k/batches/layer=NN.jsonl` (8 files, 24 latents each)
- Labels: `hunt/interp12k/labels/{gemini,claude}/` (8 files each)
- Integrator: `hunt/interp12k/integrate12k.py`; summary `hunt/interp12k/integration/summary.json`
- Extraction: APOC arrays 35473543 (fail: cfg name) → 35483621 (fail: EncoderOutput fields) → smoke 35491689
  (PASS) → 8/8 array COMPLETED; mirrors at `/mnt/d/env/apoc/marin32b/interp-out-12k/`
- Artifacts tgz: `alxcspr/stationhouse sae-32k/interp-12k-audit-20261106.tgz` (HF, no GitHub pushes)
