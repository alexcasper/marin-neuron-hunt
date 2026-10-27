# L00 census results

## sae_L00_final (pilot, 6000 steps, 6.14M tokens) — 2026-10-27, 400,384 tokens sampled
- alive 27,249/32,768 (83.2%) [step-1000 was 13,124/40.1%]
- fvu_sampled 0.7406; max feat act 9.19 @ latent 6219 (also freq_max: fires on 99.94% of tokens)
- freq deciles (alive): 5e-6 1e-5 1.7e-5 3e-5 4.5e-5 7.5e-5 1.55e-4 4.9e-4 1.9e-3
- freq mean(alive) 1.174e-3 (uniform k/n = 9.77e-4); p99 2.07e-2; top-100 latents = 28.0% of firings
- source: dusk:~/marin-sae/census_L00_final.json (sae_stats.py w/ freq tracking)

## L00-extend (top-up to full epoch) — pending, watcher on dusk chain
