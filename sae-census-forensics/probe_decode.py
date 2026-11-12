#!/usr/bin/env python3
"""Probe: does the SAVED L00 ckpt reconstruct at all after load_from_disk?
Prints ||sae_out||^2/||x||^2, fvu, and weight stats for the loaded model."""
import sys, json, math, glob
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

SCR = "/data/scratch/apx186/marin32b"
CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer=00/sae_L00w64_final_s*"))[0]
PARQ = f"{SCR}/atlas-resorted/layer00-tokensorted.parquet"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DEV = "cuda:0"
sae = Sae.load_from_disk(CKPT, device=DEV); sae.eval()
enc = sae.encoder.weight.detach().float()
dec = sae.W_dec.detach().float()
print("enc: shape", tuple(enc.shape), "std", round(enc.std().item(), 5))
print("W_dec: shape", tuple(dec.shape), "col-norm mean", round(dec.norm(dim=0).mean().item() if dec.shape[0]==dec.shape[1] else dec.norm(dim=-1).mean().item(), 5),
      "absmax", round(dec.abs().max().item(), 5))
try:
    pre = sae.pre_bias.detach().float(); print("pre_bias: norm", round(pre.norm().item(), 4))
except Exception as e:
    print("pre_bias access:", e)
for n, p in sae.state_dict().items():
    if p.numel() < 30000: print("  param", n, tuple(p.shape), "absmean", round(p.abs().mean().item(), 6))

scale_vec = load_scales_vec(SCALE, 0, D_MODEL_MLP)
pf = pq.ParquetFile(PARQ)
total = pf.metadata.num_rows; nrg = pf.metadata.num_row_groups
# one batch from mid-corpus and one from the val tail
def batch_for(rg):
    rb = pf.read_row_group(rg, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
    rb = rb.slice(0, rb.num_rows // 64 * 64)
    tok = rb.column("token_idx").to_numpy(); neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32); up = rb.column("up").to_numpy().astype(np.float32)
    B = rb.num_rows // 64
    x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
    act = gate / (1.0 + np.exp(-gate)) * up
    x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act / scale_vec[neu]
    return torch.from_numpy(x[:1024])
with torch.no_grad():
    for name, rg in [("mid", nrg // 2), ("val-tail", nrg - 1)]:
        x = batch_for(rg).to(DEV).to(torch.bfloat16)
        out = sae(x)
        so = out.sae_out.float(); xf = x.float()
        print(f"{name}: fvu={((xf-so)**2).sum()/ (xf**2).sum():.4f} "
              f"||sae_out||^2/||x||^2={ (so**2).sum()/(xf**2).sum():.4f} "
              f"mean_top_acts={out.latent_acts.float().mean().item():.4f}")
