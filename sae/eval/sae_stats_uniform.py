#!/usr/bin/env python3
"""Uniform-sampling census for sae-64k layers (alex order 2026-11-07).

Same measurements as sae_stats.py (alive/max_act/fire_frac census +
reconstruction fvu) but samples UNIFORM row groups across the whole
token-sorted parquet (interp-style spread) instead of the corpus-prefix
slice. Output convention (binding): writes census_uniform.parquet and
stats_uniform.json (fvu entry labeled fvu_atlas_uniform); renames the
original census.parquet -> census_prefix.parquet on first run; NEVER
overwrites an existing census_uniform.parquet (idempotent re-run skips).
"""
import json, os, sys, math, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

L = int(sys.argv[1])
LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
OUT = f"{SCR}/sae-64k/layer={LL}"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
SAMPLE_TOKENS = 200_704
DEV = "cuda:0"
assert torch.cuda.is_available()

# v2 (2026-11-07): raw-acts convention — matches the 64k sweep's actual training.
# Preserve any scaled-run outputs as *_scaled_ARTIFACT.* BEFORE the skip check,
# so a re-run of this script after v1 renames and proceeds instead of skipping.
for suf, art in (("census_uniform.parquet", "census_uniform_scaled_ARTIFACT.parquet"),
                 ("stats_uniform.json", "stats_uniform_scaled_ARTIFACT.json")):
    src = f"{OUT}/{suf}"; dst = f"{OUT}/{art}"
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"preserved scaled-run artifact: {dst}", flush=True)
if os.path.exists(f"{OUT}/census_uniform.parquet"):
    print(f"skip L{LL}: census_uniform.parquet already exists (v2 done)", flush=True); sys.exit(0)

CKPT = sorted(glob.glob(f"{OUT}/sae_L{LL}w64_final_s*"))[0]
scale_vec = None
sae = Sae.load_from_disk(CKPT, device=DEV); sae.eval()
n_latents = sae.encoder.weight.shape[0]
print(f"L{LL} ckpt={CKPT} n_latents={n_latents}", flush=True)

pf = pq.ParquetFile(PARQ)
total_rows = pf.metadata.num_rows; nrg = pf.metadata.num_row_groups
tokens_per_rg = (total_rows / nrg) / 64.0
n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
step = max(1, nrg // n_needed)
rgs = list(range(0, nrg, step))
print(f"L{LL}: rows={total_rows} nrg={nrg} spread_rgs={len(rgs)}", flush=True)

def dense(rb):
    tok = rb.column("token_idx").to_numpy()
    neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32)
    up = rb.column("up").to_numpy().astype(np.float32)
    n = rb.num_rows; B = n // 64
    x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
    act = gate / (1.0 + np.exp(-gate)) * up
    if scale_vec is not None:
        x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act / scale_vec[neu]
    else:
        x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act
    return x

seen = torch.zeros(n_latents, dtype=torch.bool, device=DEV)
max_act = torch.zeros(n_latents, device=DEV)
fire = torch.zeros(n_latents, dtype=torch.long, device=DEV)
fvu_n = fvu_d = 0.0; n_tok = 0
with torch.no_grad():
    for gi in rgs:
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64: rb = rb.slice(0, rb.num_rows // 64 * 64)
        if rb.num_rows == 0: continue
        x = torch.from_numpy(dense(rb)).to(DEV).to(torch.bfloat16)
        out = sae(x)
        resid = x - out.sae_out
        fvu_n += (resid.float() ** 2).sum().item()
        fvu_d += (x.float() ** 2).sum().item()
        n_tok += x.shape[0]
        li = out.latent_indices
        seen[li.unique()] = True
        m = torch.zeros(n_latents, device=DEV)
        m.scatter_reduce_(0, li.flatten(), out.latent_acts.flatten().float(), reduce="amax", include_self=True)
        max_act = torch.maximum(max_act, m)
        fire += torch.bincount(li.flatten().long(), minlength=n_latents)
        if n_tok >= SAMPLE_TOKENS: break

fvu = fvu_n / max(fvu_d, 1e-9)
alive = seen.cpu().numpy(); mx = max_act.cpu().numpy(); ff = (fire.cpu().numpy() / max(n_tok, 1))
pq.write_table(pa.table({
    "layer": pa.array([L] * n_latents, type=pa.int16()),
    "latent_id": pa.array(range(n_latents), type=pa.int32()),
    "alive": pa.array(alive, type=pa.bool_()),
    "max_act": pa.array(mx, type=pa.float32()),
    "fire_frac": pa.array(ff, type=pa.float32()),
}), f"{OUT}/census_uniform.parquet")
stats = {
    "layer": L, "ckpt": CKPT, "sampling": "uniform_row_group_spread",
    "row_groups_sampled": len(rgs), "row_group_stride": step,
    "tokens_sampled": int(n_tok),
    "fvu_atlas_uniform": round(fvu, 6),   # labeled per standing methodology
    "input_convention": "raw_acts (v2 2026-11-07; matches 64k sweep training)",
    "alive_latents": int(alive.sum()), "n_latents": n_latents,
    "max_feat_act": round(float(mx.max()), 4),
    "freq_top100_share": round(float(np.sort(ff)[::-1][:100].sum()), 4),
}
with open(f"{OUT}/stats_uniform.json", "w") as f: json.dump(stats, f, indent=1)
if os.path.exists(f"{OUT}/census.parquet") and not os.path.exists(f"{OUT}/census_prefix.parquet"):
    os.rename(f"{OUT}/census.parquet", f"{OUT}/census_prefix.parquet")
print(f"CENSUS_UNIFORM_OK L{LL} fvu_atlas_uniform={fvu:.4f} alive={int(alive.sum())} tokens={n_tok}", flush=True)
