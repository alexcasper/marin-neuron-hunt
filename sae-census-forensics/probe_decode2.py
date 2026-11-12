#!/usr/bin/env python3
"""Probe 2: decode L00 with sae.W_dec vs encoder.weight (tied-decoder test)."""
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
sae = Sae.load_from_disk(CKPT, device=DEV); sae.eval()

W_dec = sae.W_dec.detach().float()                     # [65536, 27648]
W_tied = sae.encoder.weight.detach().float()           # [65536, 27648]
W_tied = W_tied / W_tied.norm(dim=1, keepdim=True).clamp(min=1e-8)  # unit rows
coses = []
with torch.no_grad():
    for i in range(0, W_dec.shape[0], 1024):
        coses.append(torch.nn.functional.cosine_similarity(
            W_dec[i:i+1024], W_tied[i:i+1024], dim=1))
print("corr(W_dec, W_tied) rows:", round(torch.cat(coses).mean().item(), 5), flush=True)

scale_vec = load_scales_vec(SCALE, 0, D_MODEL_MLP)
pf = pq.ParquetFile(PARQ); nrg = pf.metadata.num_row_groups
rb = pf.read_row_group(nrg - 1, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
rb = rb.slice(0, min(1024, rb.num_rows // 64) * 64)
tok = rb.column("token_idx").to_numpy(); neu = rb.column("neuron").to_numpy()
gate = rb.column("gate").to_numpy().astype(np.float32); up = rb.column("up").to_numpy().astype(np.float32)
B = rb.num_rows // 64
x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
act = gate / (1.0 + np.exp(-gate)) * up
x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act / scale_vec[neu]
x = torch.from_numpy(x).to(DEV).to(torch.bfloat16)
with torch.no_grad():
    out = sae.encode(x)
    idx = out.top_indices.long(); a = out.top_acts.float()
    xf = x.float()
    for name, W in [("W_dec(saved)", W_dec), ("encoder-tied", W_tied)]:
        xh = (W[idx] * a.unsqueeze(-1)).sum(1)
        b_dec = sae.b_dec.detach().float() if hasattr(sae, "b_dec") else 0
        xh = xh + b_dec
        fvu = ((xf - xh) ** 2).sum() / (xf ** 2).sum()
        print(f"{name}: fvu={fvu:.4f} energy={(xh**2).sum()/(xf**2).sum():.4f}", flush=True)
