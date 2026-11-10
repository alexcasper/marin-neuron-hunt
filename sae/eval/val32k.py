#!/usr/bin/env python3
"""32k-side val-fvu on the SAME 30k trailing holdout as the 64k probe.
Usage: val32k.py <layer> <ckpt_dir>  -> prints JSON {layer, ckpt, val_fvu}
"""
import json, sys
import torch
from sparsify import Sae
from train_sae import val_fvu, load_scales_vec, D_MODEL_MLP

L = int(sys.argv[1])
CKPT = sys.argv[2]
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{L:02d}-tokensorted.parquet"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
scale_vec = load_scales_vec(SCALE, L, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device="cuda:0")
sae.eval()
v = val_fvu(sae, PARQ, 1920000, 1024, scale_vec, "cuda:0", 30000)
print(json.dumps({"layer": L, "ckpt": CKPT, "val_rows": 1920000,
                  "val_tokens": 30000, "val_fvu": round(v, 6)}))
