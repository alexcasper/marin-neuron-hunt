#!/usr/bin/env python3
"""Probe 4: the trainer's OWN val_fvu, verbatim, on the loaded L00 ckpt."""
import glob
import torch
from sparsify import Sae
from train_sae import val_fvu, load_scales_vec, D_MODEL_MLP

SCR = "/data/scratch/apx186/marin32b"
CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer=00/sae_L00w64_final_s*"))[0]
PARQ = f"{SCR}/atlas-resorted/layer00-tokensorted.parquet"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DEV = "cuda:0"
sae = Sae.load_from_disk(CKPT, device=DEV); sae.eval()
scale_vec = load_scales_vec(SCALE, 0, D_MODEL_MLP)
v = val_fvu(sae, PARQ, 1920000, 1024, scale_vec, DEV, 30000)
print(f"trainer val_fvu(loaded ckpt) = {v:.4f}", flush=True)
