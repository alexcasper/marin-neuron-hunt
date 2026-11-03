# L00 census results

## sae_L00_final (pilot, 6000 steps, 6.14M tokens) — 2026-10-27, 400,384 tokens sampled
- alive 27,249/32,768 (83.2%) [step-1000 was 13,124/40.1%]
- fvu_sampled 0.7406; max feat act 9.19 @ latent 6219 (also freq_max: fires on 99.94% of tokens)
- freq deciles (alive): 5e-6 1e-5 1.7e-5 3e-5 4.5e-5 7.5e-5 1.55e-4 4.9e-4 1.9e-3
- freq mean(alive) 1.174e-3 (uniform k/n = 9.77e-4); p99 2.07e-2; top-100 latents = 28.0% of firings
- source: dusk:~/marin-sae/census_L00_final.json (sae_stats.py w/ freq tracking)

## L00-extend (top-up to full epoch) — pending, watcher on dusk chain

## L00-extend (sae_L00_final after top-up, full 1 epoch of layer 00) — 2026-10-27 18:46Z, 400,384 tokens sampled
- train: 5015 steps, 3.85M unseen tokens, 264 tok/s sustained, 4.04 h, batch 768, warm-start from pilot final
- fvu_sampled 0.7315 (pilot final: 0.7406); alive 26,754/32,768 (81.65% vs 83.16%)
- max feat act 11.75 @ latent 6219 (pilot: 9.19 — near-always-on latent grew stronger, freq 99.98%)
- p99 feat max act 4.97 (was 4.47)
- freq deciles (alive): 5e-6 1e-5 2e-5 3.2e-5 5.2e-5 9.5e-5 2.15e-4 5.9e-4 1.98e-3 — all deciles shifted up vs pilot
- freq mean(alive) 1.196e-3; p99 2.05e-2; top-100 latents = 28.7% of firings (was 28.0%)
- source: dusk:~/marin-sae/census_L00_extend.json, md5 f95f7bf488bc007bfd5464f7a4da87d8 (534 bytes, verified)
