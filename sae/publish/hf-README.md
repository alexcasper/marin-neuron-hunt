---
license: apache-2.0
language:
- en
tags:
- interpretability
- sparse-autoencoder
- topk-sae
- marin-32b
- mechanistic-interpretability
---

# Station House — marin-32b 32k TopK SAE sweep (20 already-good layers)

TopK sparse autoencoders over **MLP activation vectors** (d_in = 27,648 =
`silu(gate)·up`, per-neuron scale-normalized) of
[marin-community/marin-32b-base](https://huggingface.co/marin-community/marin-32b-base)
(Qwen3-32B arch: 64 layers, SwiGLU), captured on FineWeb (`sample-10BT`
long-doc subset, 4,883 docs / 9,995,501 tokens, content-filtered) as part of
the [marin-32b-neuron-atlas](https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas)
pilot extraction (`pilot/fineweb-10m-sparse`, sparse tier).

## Config (all layers identical)

- eai-sparsify 1.3.3 `Sae`, TopK (`activation="topk"`), **32,768 latents, k=32**, bf16
- AdamW lr 3e-4 (β 0.9/0.95), fvu loss, decoder unit-norm renormalized each step
- 6,000 steps × 1,024 tokens = 6.14M tokens (~0.6 epoch of the layer slice)
- Trained on A100-40 (Apocrita), one task per layer

## These 64 layers

The full 64-layer sweep trained every layer for 6,000 steps; the 44 layers with
sampled fvu ≥ 0.90 then got a 12,000-step topup (lowered-LR consolidation).
**Best copy per layer pushed**: the 44 topup layers ship their 12k final
(`sae_LNN_step12000/` — all 44 improved 6k→12k); the 20 already-good layers
ship their 6k final (`sae_LNN_final/`). fvu values below are 200,704-token
censuses on the shared first-200k-token slice (see the sampling caveat at the
bottom):

| layer | ckpt pushed | fvu 6k census | fvu 12k census | alive (census) | max_feat_act |
|---|---|---|---|---|---|
| 00 | sae_L00_final (6k) | 0.7317 | — | see census.parquet | 7.66 |
| 01 | sae_L01_final (6k) | 0.6488 | — | see census.parquet | 19.88 |
| 02 | sae_L02_final (6k) | 0.7604 | — | see census.parquet | 109.5 |
| 03 | sae_L03_final (6k) | 0.7676 | — | see census.parquet | 16.75 |
| 04 | sae_L04_final (6k) | 0.7629 | — | see census.parquet | 24.25 |
| 05 | sae_L05_final (6k) | 0.8040 | — | see census.parquet | 17.75 |
| 06 | sae_L06_final (6k) | 0.8436 | — | see census.parquet | 27.88 |
| 07 | sae_L07_final (6k) | 0.8562 | — | see census.parquet | 27.0 |
| 08 | sae_L08_final (6k) | 0.8676 | — | see census.parquet | 26.25 |
| 09 | sae_L09_final (6k) | 0.8748 | — | see census.parquet | 44.0 |
| 10 | sae_L10_final (6k) | 0.8791 | — | see census.parquet | 36.0 |
| 11 | sae_L11_final (6k) | 0.8926 | — | see census.parquet | 41.5 |
| 12 | step12000 (12k topup) | 0.9166 | 0.8928 | see census_12k.parquet | 66.0 |
| 13 | step12000 (12k topup) | 0.9289 | 0.9020 | see census_12k.parquet | 52.5 |
| 14 | step12000 (12k topup) | 0.9735 | 0.9187 | see census_12k.parquet | 54.25 |
| 15 | step12000 (12k topup) | 0.9297 | 0.9095 | see census_12k.parquet | 96.0 |
| 16 | step12000 (12k topup) | 0.9969 | 0.9630 | see census_12k.parquet | 144.0 |
| 17 | step12000 (12k topup) | 0.9592 | 0.9451 | see census_12k.parquet | 73.5 |
| 18 | step12000 (12k topup) | 0.9643 | 0.9501 | see census_12k.parquet | 71.5 |
| 19 | step12000 (12k topup) | 0.9769 | 0.9512 | see census_12k.parquet | 97.5 |
| 20 | step12000 (12k topup) | 0.9928 | 0.9703 | see census_12k.parquet | 70.5 |
| 21 | step12000 (12k topup) | 0.9960 | 0.9639 | see census_12k.parquet | 108.0 |
| 22 | step12000 (12k topup) | 1.0090 | 0.9880 | see census_12k.parquet | 134.0 |
| 23 | step12000 (12k topup) | 0.9923 | 0.9795 | see census_12k.parquet | 105.5 |
| 24 | step12000 (12k topup) | 0.9842 | 0.9586 | see census_12k.parquet | 79.0 |
| 25 | step12000 (12k topup) | 0.9769 | 0.9442 | see census_12k.parquet | 114.5 |
| 26 | step12000 (12k topup) | 0.9751 | 0.9603 | see census_12k.parquet | 94.5 |
| 27 | step12000 (12k topup) | 0.9750 | 0.9625 | see census_12k.parquet | 112.0 |
| 28 | step12000 (12k topup) | 0.9869 | 0.9648 | see census_12k.parquet | 118.5 |
| 29 | step12000 (12k topup) | 0.9992 | 0.9754 | see census_12k.parquet | 125.0 |
| 30 | step12000 (12k topup) | 1.0156 | 0.9704 | see census_12k.parquet | 118.0 |
| 31 | step12000 (12k topup) | 0.9605 | 0.9448 | see census_12k.parquet | 169.0 |
| 32 | step12000 (12k topup) | 0.9561 | 0.9360 | see census_12k.parquet | 121.5 |
| 33 | step12000 (12k topup) | 0.9548 | 0.9297 | see census_12k.parquet | 128.0 |
| 34 | step12000 (12k topup) | 0.9688 | 0.9320 | see census_12k.parquet | 196.0 |
| 35 | step12000 (12k topup) | 0.9549 | 0.9321 | see census_12k.parquet | 226.0 |
| 36 | step12000 (12k topup) | 0.9259 | 0.9048 | see census_12k.parquet | 262.0 |
| 37 | sae_L37_final (6k) | 0.8931 | — | see census.parquet | 338.0 |
| 38 | sae_L38_final (6k) | 0.8961 | — | see census.parquet | 362.0 |
| 39 | sae_L39_final (6k) | 0.8611 | — | see census.parquet | 724.0 |
| 40 | sae_L40_final (6k) | 0.8867 | — | see census.parquet | 552.0 |
| 41 | sae_L41_final (6k) | 0.8880 | — | see census.parquet | 388.0 |
| 42 | step12000 (12k topup) | 0.9372 | 0.9261 | see census_12k.parquet | 468.0 |
| 43 | step12000 (12k topup) | 0.9706 | 0.9517 | see census_12k.parquet | 432.0 |
| 44 | step12000 (12k topup) | 0.9865 | 0.9523 | see census_12k.parquet | 346.0 |
| 45 | step12000 (12k topup) | 1.0025 | 0.9473 | see census_12k.parquet | 446.0 |
| 46 | step12000 (12k topup) | 0.9803 | 0.9418 | see census_12k.parquet | 418.0 |
| 47 | step12000 (12k topup) | 0.9538 | 0.9223 | see census_12k.parquet | 536.0 |
| 48 | step12000 (12k topup) | 0.9217 | 0.8987 | see census_12k.parquet | 366.0 |
| 49 | step12000 (12k topup) | 0.9471 | 0.8975 | see census_12k.parquet | 476.0 |
| 50 | step12000 (12k topup) | 0.9182 | 0.9182 | see census_12k.parquet | 418.0 |
| 51 | step12000 (12k topup) | 0.9171 | 0.9171 | see census_12k.parquet | 616.0 |
| 52 | step12000 (12k topup) | 0.9166 | 0.9166 | see census_12k.parquet | 502.0 |
| 53 | step12000 (12k topup) | 0.8992 | 0.8992 | see census_12k.parquet | 584.0 |
| 54 | step12000 (12k topup) | 0.8976 | 0.8976 | see census_12k.parquet | 568.0 |
| 55 | step12000 (12k topup) | 0.8727 | 0.8727 | see census_12k.parquet | 1232.0 |
| 56 | sae_L56_final (6k) | 0.8946 | — | see census.parquet | 1304.0 |
| 57 | sae_L57_final (6k) | 0.5530 | — | see census.parquet | 3680.0 |
| 58 | sae_L58_final (6k) | 0.8720 | — | see census.parquet | 2304.0 |
| 59 | step12000 (12k topup) | 0.9137 | 0.8861 | see census_12k.parquet | 2432.0 |
| 60 | step12000 (12k topup) | 0.9163 | 0.8976 | see census_12k.parquet | 3040.0 |
| 61 | step12000 (12k topup) | 0.9638 | 0.9181 | see census_12k.parquet | 4288.0 |
| 62 | step12000 (12k topup) | 0.9751 | 0.9350 | see census_12k.parquet | 9536.0 |
| 63 | step12000 (12k topup) | 1.1839 | 1.0365 | see census_12k.parquet | 11584.0 |

(`fvu_sampled` = fraction of unexplained variance on a held-out-style 200,704-token
resample; `alive_latents` = latents that fired at least once, of 32,768.)

## Layout

```
sae-32k/layer=NN/
  sae_LNN_final/        # final checkpoint (bf16): model.safetensors + cfg.json
  census.parquet        # one row per latent: layer, latent_id, alive, max_act, fire_frac
  stats.json            # fvu, alive counts, max activations, frequency fields
  train_log.jsonl       # training log (fvs/loss every 50 steps)
  sha256.txt            # integrity manifest
```

Load with [eai-sparsify](https://github.com/EleutherAI/sparsify):
`Sae.load_from_disk("sae-32k/layer=40/sae_L40_final")`.

## Provenance

- Activations: marin-32b-base @ f19b3466 (bf16, seq 2048), extracted on the
  atlas fineweb-10m-sparse capture; input vectors are the sparse-tier
  top-64 reconstruction, normalized by the atlas per-neuron scale
  (`neuron_scale.parquet`).
- Training + census code and the 64-layer sweep context:
  Station House lane workspace, `sae/` (apoc-sae-layer.sh, train_sae.py, sae_stats.py).
- License: Apache-2.0 (matches model + atlas).
