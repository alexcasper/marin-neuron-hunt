# Encoder-attribution outputs — 15 layers staged locally (2026-11-10)

Source: dawn `/mnt/d/env/apoc/marin32b/hf_upload/nominate_enc/attribution_enc_L*.parquet`
(15 layers: 01,02,03,05,06,08,09,10,11,12,13,14,15,16,57; 8,000 tracked latents each,
200,704-token sample, ckpt = per-layer 64k sweep FINALs `sae_L*W64_final_s*` — confirmed
by the control JSONs, NOT the superseded plateau ckpt). Apocrita has 29/64 done;
the rest stage here as the wave (35619765) drains. 4 timeout layers (04,07,24,30)
await a rerun pass.

## Per-layer concentration (fired_tokens>=20 subset)

| layer | fired>=20 | top1 p50 | top1 p90 | max | ge50 latents | distinct ge50 neurons | eff_p50 |
|-------|-----------|----------|----------|-----|--------------|----------------------|---------|
| 01 | 7336 | 0.043 | 0.093 | 0.379 | 0 | 0 | 137.1 |
| 02 | 6934 | 0.048 | 0.103 | 0.487 | 0 | 0 | 120.0 |
| 03 | 6062 | 0.042 | 0.103 | 0.409 | 0 | 0 | 169.0 |
| 05 | 5733 | 0.046 | 0.110 | 0.455 | 0 | 0 | 155.4 |
| 06 | 5147 | 0.053 | 0.135 | 0.645 | 2 | 2 | 135.2 |
| 08 | 5502 | 0.055 | 0.142 | 0.524 | 4 | 3 | 119.3 |
| 09 | 5404 | 0.063 | 0.169 | 0.596 | 3 | 3 | 99.3 |
| 10 | 5404 | 0.065 | 0.180 | 0.677 | 8 | 7 | 99.2 |
| 11 | 4693 | 0.069 | 0.188 | 0.687 | 16 | 8 | 93.5 |
| 12 | 5064 | 0.066 | 0.188 | 0.587 | 7 | 7 | 102.4 |
| 13 | 5063 | 0.062 | 0.180 | 0.532 | 2 | 2 | 109.1 |
| 14 | 4477 | 0.060 | 0.178 | 0.489 | 0 | 0 | 119.0 |
| 15 | 4355 | 0.061 | 0.182 | 0.536 | 3 | 2 | 112.3 |
| 16 | 4125 | 0.058 | 0.187 | 0.580 | 7 | 5 | 119.2 |
| 57 | 3526 | 0.099 | 0.198 | 0.779 | 19 | 2 | 54.3 |

## Shuffle control (K=5 column shuffles, 2,500 latents/layer)

All 15 layers: control top1 p50 = 0.0, p90 = 0.0, max 0.228–0.233 → the concentration
is genuine per-neuron structure, replicating the L57 finding across layers.

## Reconciliation with the 2026-11-08 L57 numbers

No-filter L57 reproduces exactly: 40 latents with top1>=0.5 → 11 distinct neurons
(memory said "~11"). eff_p50_fired 47.7 ≈ "~48".

## Key patterns

1. **Concentration rises with depth.** Early layers (01–05): eff_p50 120–170, top1_p50
   ~0.045, ZERO ge50 candidates. Mid (08–16): eff_p50 ~95–120, a handful of ge50.
   L57: eff_p50 54, most concentrated of the set.
2. **L57's ge50 head is likely a magnitude artifact, not hunt candidates.** 18 of its
   19 filtered ge50 latents share top1 neuron 3872, each firing on ~111k–127k of
   200,704 tokens (~60% fire rate — the always-on zone per the skill's filter rule),
   eff_neurons ~1.7. One dominant always-on neuron tagged by many broad latents.
   The skill's dedupe + always-on filters collapse it to (at most) neuron 3872 itself.
3. **The selective candidates are mid-layer.** ~50 ge50 latents across L06–L16 with
   modest fire counts (20–330 tokens = 0.01–0.16% fire rate) and eff_neurons 2–4:
   selective, neuron-locked latents. Distinct top1 neurons repeat across latents
   (L11: 8533 ×4, 5811 ×3, 15351 ×3, 24556 ×2; L10: 15969 ×2; L16: 6007 ×2,
   15747 ×2) — many latents detecting one dominant neuron, per the dedupe rule.
   Distinct candidate neurons per layer after dedupe: L11 = 8, L10 = 7, L12 = 7,
   L16 = 5, L08 = 3, L09 = 3, L06 = 2, L13 = 2, L15 = 2.
4. **Hunt-shortlist implication (per skill rules):** candidate neurons for
   pole_stats cross-reference are the mid-layer distinct top1 neurons above
   (L06–L16, ~34 distinct), plus neuron 3872@L57 and 26033@L57 held as suspects
   pending an always-on check. Validation must go through atlas pole_stats
   exemplars before any of this becomes a hunt list.
