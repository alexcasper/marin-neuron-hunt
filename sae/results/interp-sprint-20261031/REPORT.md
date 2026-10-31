# Marin-SAE latent interpretability — 20 good layers, blind dual-model audit

**2026-10-31, 02:21 UTC deadline report · coordinator ops_sh (Marrok)**

## Design

- **Object**: `sae_LNN_final` TopK SAEs (32,768 latents, k=32, **6k steps**) for the
  20 already-good layers (L0-11, L37-41, L56, L57, L58). These layers were NOT in
  the 44-layer 12k topup (topup targeted worst-fvu layers), so this audit judges
  the 6k dictionary — consistent across all 20.
- **Extraction** (APOC): 200k-token spread sample per layer (verified by fire-rate
  quantization 5e-6 = 1/200k), SAE encode, per-latent fire_rate/max_act, top-8
  activating tokens → docs.parquet text spans. GPU array 34661828 stayed
  queue-bound the whole window (52 jobs ahead); the run that delivered was CPU
  array **34666766** (compute partition, 8-wide, 20/20 layers in ~1 h).
- **Three sampling tracks** were designed ×12: top-mass, fire-rate-gated mass,
  random-alive (seed 20261030+L). **Implementation lesson: the fire-rate gate was a
  no-op — in all 20 layers the gated top-12 was IDENTICAL to the raw mass top-12**
  (top-mass latents all clear the median-fire-rate gate). Effective design = 24
  mass-selected + 12 random latents per layer, 480 latents total.
- **Blind labeling**: Gemini 2.5 Flash and Claude Code labeled the SAME example
  sets independently (identical protocol, no access to each other's labels).
  One-line interpretation + forced category: concept / document-identity /
  register-boilerplate / structural / unclear.

## Agreement (480 latent pairs × 2 models)

| metric | value |
|---|---|
| category exact agreement | **83.5%** |
| — when both models make a positive call (n=55, 11%) | 69% |
| — both-positive rate | 11% |
| — one-positive/one-unclear | 13% |
| interpretation token-Jaccard ≥ 0.30 ("same reading") | 10.6% |

High category agreement + low wording agreement = both models converge on the
CATEGORY (mostly "unclear") but describe the mush differently. Agreement is not
the bottleneck — interpretability is.

## What the latents are (pooled both models, 960 label events)

| category | mass-selected | random-alive |
|---|---|---|
| **unclear** | 77% | **87%** |
| document-identity | 15% | 10% |
| register-boilerplate | 4% | 1% |
| structural | 3% | 1% |
| concept | 1% | 1% |

- **Concepts are ~1% everywhere.** The clearest positive readings are
  single-document memorizations: Malone v. DeRossett negligence opinion (L01),
  Pan Am 103/Karen Hunt memorial article (L05, L06), Visual FoxPro help docs
  (L06), a decoupage tutorial (L02), a Sedona retreat page (L05). Both models
  independently name the same document — these are real features, and they are
  all document-identity.
- **Random-alive latents are alive-but-inert**: median fire rate 0.01% (vs 3.3%
  for the mass track). Their example spans are genuinely incoherent (garage-door
  prose + chemistry table + JS preload code in one latent); "unclear" is an
  honest label, not labeler laziness.

## Depth trend

| band | unclear | doc-identity | register | concept |
|---|---|---|---|---|
| L00-11 | 82% | 13% | 2% | 1% |
| L37-41 | 89% | 10% | 1% | 0% |
| L56-58 | 72% | 17% | **8%** | 0% |

Interpretability improves slightly in the last layers (register/boilerplate
features appear only late); mid-network (L37-41) is the mushiest. Document-
identity grows with depth (10% → 17%).

## Cross-layer repeats

The only readings recurring across layers are the SAME SOURCE DOCUMENTS: the
WoW Warlock "Soul Shards" posts (recurring in L7, L8, L11, L37, L38, L39, L40,
L41, L56, L58), the Otto II biography (L11, L38, L39), a Manzoni-style
historical narrative (L3, L57). Up to 4 layers apart, independent latents point
at the same handful of documents — consistent with the L57 finding of "5 spam
docs recur across latents". **Cross-layer structure = memorization echoes, not
universal concept features.**

## Honest verdict on dictionary interpretability at this training age

At 6k steps / 32k width the Marin SAE dictionary is **not yet a concept
dictionary**. ~80% of sampled latents — including a uniform random-alive sample
— are uninterpretable from their maximal activations; nearly all positive
readings are single-document memorization or (late-layer) register signatures;
semantic concepts are ~1%. The refined L57 lesson: top-mass features are not
rare-fire spam — they are high-fire document detectors (fire-rate median 3.3%,
one at 98.7%), so the earlier "rare spam" framing was wrong; the mass/rare
sampling artifact is real but the dominant identifiable species is
**memorized-document identity**, in every band, growing with depth.

## Implications for width/steps decisions

1. **Steps, not width, is the binding constraint.** Random-alive latents are
   inert (0.01% fire rate) — capacity isn't the problem; features haven't
   consolidated. A 64k-width run at the same step budget would likely buy more
   inert latents, not more concepts. The staged 64k rung should be re-prioritized
   BEHIND longer training (12k+ / plateau checkpoints) at 32k width.
2. **The 12k topup layers are the right next audit** — this sprint judged 6k
   finals because the good-20 layers weren't topped up. Re-run this exact
   extraction against `sae_LNN_step12000` for the 44 topped-up layers to measure
   whether unclear% drops with 2× steps. Cheap (CPU array, ~1 h).
3. **Late layers (L56-58) are the most interpretable slice** — prefer them for
   any human-facing demo.
4. **Doc-identity dominance is partly a data property** (FineWeb long-doc subset
   spam repetition), not only an SAE-training property — worth noting before
   drawing conclusions about the model itself.

## Artifacts

- Example sets: `hunt/interp/batches/layer=NN.jsonl` (20 × 36 rows, 3 tracks)
- Blind labels: `hunt/interp/labels/{gemini,claude}/layer=NN.labels.jsonl`
- Integration: `hunt/interp/integrate.py` → `integration/summary.json`;
  deep cuts `hunt/interp/deep_cuts.py`
- Dusk fallback extractor (neuron-major layout, unused this run but proven):
  `sae/interp_preview_dusk.py`
- Repro: APOC `interp_preview.py` (dispatcher copy), arrays 34661828 (gpu,
  never started) / 34666766 (cpu, delivered all 20).
