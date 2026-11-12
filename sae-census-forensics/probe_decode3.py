#!/usr/bin/env python3
"""Probe 3: train-mode vs eval-mode forward on the loaded L00 ckpt."""
import glob
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

scale_vec = load_scales_vec(SCALE, 0, D_MODEL_MLP)
pf = pq.ParquetFile(PARQ); nrg = pf.metadata.num_row_groups
rb = pf.read_row_group(nrg - 1, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
rb = rb.slice(0, min(1024, rb.num_rows // 64) * 64)
neu = rb.column("neuron").to_numpy()
gate = rb.column("gate").to_numpy().astype(np.float32); up = rb.column("up").to_numpy().astype(np.float32)
B = rb.num_rows // 64
x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
act = gate / (1.0 + np.exp(-gate)) * up
x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act / scale_vec[neu]
xb = torch.from_numpy(x).to(DEV).to(torch.bfloat16)

for mode in ("eval", "train"):
    sae = Sae.load_from_disk(CKPT, device=DEV)
    if mode == "eval": sae.eval()
    else: sae.train()
    with torch.no_grad():
        out = sae(xb)
        so = out.sae_out.float(); xf = xb.float()
        print(f"{mode}: fvu={((xf-so)**2).sum()/(xf**2).sum():.4f} "
              f"energy={(so**2).sum()/(xf**2).sum():.4f} "
              f"mean_acts={out.latent_acts.float().mean().item():.4f}", flush=True)
