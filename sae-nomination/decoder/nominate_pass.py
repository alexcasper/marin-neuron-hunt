#!/usr/bin/env python3
"""Decode-side neuron-nomination pass for the sae-64k sweep (alex task 2026-11-08).

Pure decoder math, CPU-only: for every latent read its W_dec row and compute
which NEURONS dominate it. Complements the encode-side pattern pass
(patterns_pass.py): families tell you WHAT a latent means, this tells you
WHICH NEURONS implement it — the shortlist generator for marin-neuron-hunt.

Convention: raw acts (matches 64k sweep training). Row-normalized shares are
scale-invariant, so scale_vec does not matter here.

Per layer writes to $SCR/sae-64k/layer=NN/nominate/:
  nominations.parquet  one row per latent:
    layer, latent, alive, top1_neuron, top1_share, top2_neuron, top2_share,
    eff_neurons (inverse Simpson of share distribution), l1_norm
  summary.json         headline counts
"""
import json, os, glob
import numpy as np
import torch
from sparsify import Sae

L = int(os.environ["NOM_LAYER"]); LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
OUTD = f"{SCR}/sae-64k/layer={LL}/nominate"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
assert os.path.exists(os.path.join(CKPT, "cfg.json")), CKPT
sae = Sae.load_from_disk(CKPT, device="cpu"); sae.eval()
W = sae.W_dec.detach().float().numpy()          # [n_latents, 27648]
n_lat, d = W.shape

A = np.abs(W)
l1 = A.sum(1, dtype=np.float32)
alive = l1 > 0
P = A / np.maximum(l1[:, None], np.float32(1e-12))  # stays float32
P = P.astype(np.float32, copy=False)
del A
n1 = np.argmax(P, axis=1)
part = np.partition(P, -2, axis=1)              # two largest shares last
s1 = part[:, -1].astype(np.float64); s2 = part[:, -2].astype(np.float64)
eff = 1.0 / np.maximum(np.einsum("ij,ij->i", P, P, dtype=np.float32), np.float32(1e-12))  # before masking top1; no big temp alloc
P[np.arange(n_lat), n1] = -1.0                  # mask top1, reuse buffer for top2
n2 = np.argmax(P, axis=1)

import pyarrow as pa, pyarrow.parquet as pq
os.makedirs(OUTD, exist_ok=True)
tbl = pa.table({
    "layer": np.full(n_lat, L, dtype=np.uint8),
    "latent": np.arange(n_lat, dtype=np.int32),
    "alive": alive,
    "top1_neuron": n1.astype(np.int32),
    "top1_share": s1.astype(np.float32),
    "top2_neuron": n2.astype(np.int32),
    "top2_share": s2.astype(np.float32),
    "eff_neurons": eff.astype(np.float32),
    "l1_norm": l1.astype(np.float32),
})
pq.write_table(tbl, f"{OUTD}/nominations.parquet")

summary = {
    "layer": L,
    "ckpt": os.path.basename(CKPT),
    "n_latents": int(n_lat),
    "d": int(d),
    "alive": int(alive.sum()),
    "top1_ge_0.5": int((s1 >= 0.5).sum()),
    "top1_ge_0.8": int((s1 >= 0.8).sum()),
    "top2_ge_0.5": int((s1 + s2 >= 0.5).sum()),
    "median_eff_neurons_alive": float(np.median(eff[alive])) if alive.any() else None,
    "distinct_top1_neurons_ge_0.5": int(len(set(n1[s1 >= 0.5].tolist()))),
}
json.dump(summary, open(f"{OUTD}/summary.json", "w"), indent=1)
print("NOMINATE DONE", json.dumps(summary), flush=True)
