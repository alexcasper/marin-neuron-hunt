# Marin-SAE latent interpretability sprint — 20 good layers (DRAFT)

**Run date**: 2026-10-31, deadline report 02:21 UTC. Coordinator: ops_sh (Marrok).
**Blind labelers**: Gemini 2.5 Flash (wS:p3Q) + Claude Code (wS:p3R), independent,
identical example sets, no cross-contamination.
**Extraction**: APOC interp array (GPU 34661828 primary / CPU 34666766 as-run),
extends the proven L57 preview (job 34305721) with three selection tracks
× 12 latents: mass / fire-rate-gated mass / random-alive (seeded).
**Checkpoints judged**: `sae_LNN_final` = 6k-step 32k-width TopK SAEs (these 20
layers were not in the 44-layer 12k topup — the topup targeted worst-fvu layers).
**Sample**: 200k tokens/layer (CPU variant 100k) spread across the 9,995,501-token
fineweb-10m slice; per-latent fire_rate + max_act recorded; top-8 spans/latent.

## Per-layer table
(filled by integrate.py — layer, n, category agreement, category mix per model,
doc-identity ratio)

## Agreement stats
(filled — category exact-match rate, same-reading rate @ Jaccard>=0.30, mean Jaccard)

## Category distribution vs depth
(filled — concept vs document-identity vs register-boilerplate vs structural
vs unclear, per layer band: L0-11 / L37-41 / L56-58)

## Cross-layer repeated features
(filled — same reading in >=2 layers = candidate universal features)

## Honest verdict on dictionary interpretability at this training age
(prior from L57 preview: 6k dictionary mostly mush — top-mass latents dominated
by document-identity features, 5 spam docs recurring across latents; one coherent
content latent (10381 bird-species listing); register signals present. This sprint
re-judges with fire-rate-weighted + random sampling to correct the top-mass
sampling artifact.)

## Implications for width/steps decisions
(filled at the end)
