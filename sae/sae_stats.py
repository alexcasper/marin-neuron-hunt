#!/usr/bin/env python3
"""Post-run stats for a trained atlas SAE: loss endpoint (sampled), alive-latent
count, max feature activation. Usage: ~/torch-rocm-env sae_stats.py <ckpt_dir>
<layer00-tokensorted.parquet> [sample_tokens=200000]"""
import sys, json
import numpy as np
import torch
from sparsify import Sae

sys.path.insert(0, "/home/dusk/marin-sae")
from train_sae import batches, load_scales_vec, D_MODEL_MLP  # noqa: E402

ckpt, parquet = sys.argv[1], sys.argv[2]
sample_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

sae = Sae.load_from_disk(ckpt, device="cuda:0")
sae.eval()
n_latents = sae.encoder.weight.shape[0]
layer_id = 0
scale_vec = load_scales_vec("/home/dusk/marin-atlas/fineweb-10m-sparse-early/summaries/neuron_scale.parquet", layer_id, D_MODEL_MLP)

seen = torch.zeros(n_latents, dtype=torch.bool, device="cuda:0")
max_act = torch.zeros(n_latents, device="cuda:0")
fvu_num, fvu_den, n_tok = 0.0, 0.0, 0
with torch.no_grad():
    for x in batches(parquet, 1024, scale_vec):
        x = x.to("cuda:0").to(torch.bfloat16)
        out = sae(x)
        resid = x - out.sae_out
        fvu_num += (resid.float() ** 2).sum().item()
        fvu_den += (x.float() ** 2).sum().item()
        n_tok += x.shape[0]
        seen[out.latent_indices.unique()] = True
        m = torch.zeros(n_latents, device="cuda:0")
        m.scatter_reduce_(0, out.latent_indices.flatten(), out.latent_acts.flatten().float(), reduce="amax", include_self=True)
        max_act = torch.maximum(max_act, m)
        if n_tok >= sample_tokens:
            break

stats = {
    "ckpt": ckpt,
    "tokens_sampled": n_tok,
    "fvu_sampled": fvu_num / fvu_den,
    "alive_latents": int(seen.sum().item()),
    "alive_frac": round(seen.float().mean().item(), 4),
    "n_latents": n_latents,
    "max_feat_act": float(max_act.max().item()),
    "max_feat_idx": int(max_act.argmax().item()),
    "p99_feat_max_act": float(max_act[seen].quantile(0.99).item()) if seen.any() else None,
}
print(json.dumps(stats, indent=1))
