#!/usr/bin/env python3
"""Probe 5: save/load round-trip fidelity for sparsify 1.3.3 SparseCoder."""
import tempfile, os
import torch
from sparsify import Sae

DEV = "cuda:0"
sae = Sae(d_in=512, n_latents=2048, k=32, device=DEV)  # small replica
x = torch.randn(64, 512, device=DEV).to(torch.bfloat16)
with torch.no_grad():
    y1 = sae(x).sae_out.float()
sd1 = {k: v.detach().clone() for k, v in sae.state_dict().items()}
with tempfile.TemporaryDirectory() as td:
    sae.save_to_disk(td)
    print("files:", sorted(os.listdir(td)))
    sae2 = Sae.load_from_disk(td, device=DEV); sae2.eval()
    with torch.no_grad():
        y2 = sae2(x).sae_out.float()
    sd2 = sae2.state_dict()
    for k in sd1:
        a, b = sd1[k].float(), sd2[k].detach().float()
        if a.shape != b.shape:
            print(f"{k}: SHAPE {tuple(a.shape)} -> {tuple(b.shape)} (max Δ n/a)")
        else:
            print(f"{k}: shape {tuple(a.shape)} max|Δ|={(a-b).abs().max().item():.3e} "
                  f"cos={torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.6f}")
print(f"forward: out cos={torch.nn.functional.cosine_similarity(y1.flatten(), y2.flatten(), dim=0).item():.6f} "
      f"max|Δ|={(y1-y2).abs().max().item():.3e}")
