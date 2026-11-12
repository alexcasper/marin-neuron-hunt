#!/usr/bin/env python3
"""Probe 5b: save/load round-trip on the REAL L00 ckpt construction.
load -> perturb slightly -> forward y1 -> save_to_disk -> reload -> forward y2.
If y1 != y2 (or state tensors differ), save/load is lossy for this build."""
import tempfile, os, glob
import torch
from sparsify import Sae

SCR = "/data/scratch/apx186/marin32b"
CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer=00/sae_L00w64_final_s*"))[0]
DEV = "cuda:0"
sae = Sae.load_from_disk(CKPT, device=DEV)
# perturb so any load-side staleness is detectable
with torch.no_grad():
    for p in sae.parameters():
        p.add_(torch.randn_like(p) * 0.01 * p.abs().mean())
x = torch.randn(64, sae.encoder.weight.shape[1], device=DEV).to(torch.bfloat16)
with torch.no_grad():
    y1 = sae(x).sae_out.float()
sd1 = {k: v.detach().float().cpu().clone() for k, v in sae.state_dict().items()}
with tempfile.TemporaryDirectory() as td:
    sae.save_to_disk(td)
    del sae
    torch.cuda.empty_cache()
    print("files:", sorted(os.listdir(td)), flush=True)
    sae2 = Sae.load_from_disk(td, device=DEV); sae2.eval()
    with torch.no_grad():
        y2 = sae2(x).sae_out.float()
    sd2 = sae2.state_dict()
    for k in sd1:
        b = sd2[k].detach().float().cpu()
        a = sd1[k]
        if a.shape != b.shape:
            print(f"{k}: SHAPE {tuple(a.shape)} -> {tuple(b.shape)}", flush=True)
        else:
            print(f"{k}: max|Δ|={(a-b).abs().max().item():.3e} "
                  f"cos={torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.8f}", flush=True)
        del b
print(f"forward: cos={torch.nn.functional.cosine_similarity(y1.flatten(), y2.flatten(), dim=0).item():.8f} "
      f"max|Δ|={(y1-y2).abs().max().item():.3e}", flush=True)
