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

## These 20 layers

The full 64-layer sweep ran on all layers; these are the 20 layers **skipped
by the topup round** because their 6k-step dictionaries already reconstructed
well (sampled fvu 0.55–0.89, 200,704-token census each) — no consolidation
pass needed:

| layer | fvu_sampled | alive_latents | max_feat_act |
|---|---|---|---|
| 0 | 0.7317 | 24457 | 7.66 |
| 1 | 0.6488 | 22046 | 19.88 |
| 2 | 0.7604 | 28508 | 109.5 |
| 3 | 0.7676 | 25597 | 16.75 |
| 4 | 0.7629 | 24806 | 24.25 |
| 5 | 0.8040 | 24159 | 17.75 |
| 6 | 0.8436 | 28397 | 27.88 |
| 7 | 0.8562 | 25562 | 27.00 |
| 8 | 0.8676 | 25310 | 26.25 |
| 9 | 0.8748 | 26306 | 44.00 |
| 10 | 0.8791 | 27398 | 36.00 |
| 11 | 0.8926 | 29312 | 41.50 |
| 37 | 0.8931 | 32538 | 338.0 |
| 38 | 0.8961 | 32764 | 362.0 |
| 39 | 0.8611 | 26691 | 724.0 |
| 40 | 0.8867 | 26543 | 552.0 |
| 41 | 0.8880 | 29314 | 388.0 |
| 56 | 0.8946 | 32392 | 1304.0 |
| 57 | 0.5530 | 31027 | 3680.0 |
| 58 | 0.8720 | 31552 | 2304.0 |

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
