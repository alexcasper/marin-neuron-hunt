#!/usr/bin/env python3
"""L57 64k feature preview: same recipe as feature_preview_L57.py (job
34305721), on the 64k plateau checkpoint. 20 latents selected fire-rate
-weighted (top-20 by mass among alive latents with fire_rate >= median
fire rate of alive latents), 200k-token spread sample, top-10 hits each,
token_idx COLUMN alignment, proportional char spans.
"""
import json, os, heapq, math
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

L = 57
LL = "57"
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
CKPT = f"{SCR}/64k-probe/layer={LL}/sae_L{LL}w64_plateau_step21000"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/docs.parquet"
OUT_MD = f"{SCR}/64k-probe/layer={LL}/feature-preview-64k.md"
OUT_JS = f"{SCR}/64k-probe/layer={LL}/feature-preview-64k.json"
SAMPLE_TOKENS = 200_000
N_SEL = 20
TOP_HITS = 10
DEV = "cuda:0"

scale_vec = load_scales_vec(SCALE, L, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device=DEV)
sae.eval()
n_latents = sae.encoder.weight.shape[0]

pf = pq.ParquetFile(PARQ)
total_rows = pf.metadata.num_rows
nrg = pf.metadata.num_row_groups
tokens_per_rg = (total_rows / nrg) / 64.0
n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
step = max(1, nrg // n_needed)
rgs = list(range(0, nrg, step))
print(f"rows={total_rows} rgs={len(rgs)} ckpt={CKPT} n_latents={n_latents}", flush=True)

def dense(rb):
    tok = rb.column("token_idx").to_numpy()
    neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32)
    up = rb.column("up").to_numpy().astype(np.float32)
    B = rb.num_rows // 64
    x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
    act = gate / (1.0 + np.exp(-gate)) * up
    x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act / scale_vec[neu]
    return x, tok.reshape(B, 64)[:, 0]

mass = torch.zeros(n_latents, device=DEV)
fire = torch.zeros(n_latents, device=DEV)
maxact = torch.zeros(n_latents, device=DEV)
seen = torch.zeros(n_latents, dtype=torch.bool, device=DEV)
n_tok = 0
with torch.no_grad():
    for gi in rgs:
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64:
            rb = rb.slice(0, rb.num_rows // 64 * 64)
        if rb.num_rows == 0:
            continue
        x, _ = dense(rb)
        x = torch.from_numpy(x).to(DEV).to(torch.bfloat16)
        out = sae(x)
        li, la = out.latent_indices.flatten(), out.latent_acts.flatten().float()
        mass.index_add_(0, li, la)
        fire.index_add_(0, li, torch.ones_like(la))
        m = torch.zeros(n_latents, device=DEV)
        m.scatter_reduce_(0, li, la, reduce="amax", include_self=True)
        maxact = torch.maximum(maxact, m)
        seen[out.latent_indices.unique()] = True
        n_tok += x.shape[0]
fire_rate = fire / max(n_tok, 1)

mass_np = mass.cpu().numpy()
fr_np = fire_rate.cpu().numpy()
alive_idx = np.flatnonzero(seen.cpu().numpy())
med = np.median(fr_np[alive_idx])
sel = list(np.argsort(-mass_np * (fr_np >= med))[:N_SEL])
print(f"sample={n_tok} alive={len(alive_idx)} median_fire={med:.5f}", flush=True)

heaps = {i: [] for i in sel}
with torch.no_grad():
    for gi in rgs:
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64:
            rb = rb.slice(0, rb.num_rows // 64 * 64)
        if rb.num_rows == 0:
            continue
        x, toks = dense(rb)
        x = torch.from_numpy(x).to(DEV).to(torch.bfloat16)
        out = sae(x)
        li = out.latent_indices.cpu().numpy()
        la = out.latent_acts.float().cpu().numpy()
        for b in range(li.shape[0]):
            for j in range(li.shape[1]):
                Lid = int(li[b, j])
                if Lid in heaps and la[b, j] > 0:
                    h = heaps[Lid]
                    item = (float(la[b, j]), int(toks[b]))
                    if len(h) < TOP_HITS:
                        heapq.heappush(h, item)
                    elif item > h[0]:
                        heapq.heapreplace(h, item)

dt = pq.read_table(DOCS, columns=["doc_id", "doc_tokens", "text"]).to_pydict()
doc_tok = np.array(dt["doc_tokens"], dtype=np.int64)
cum = np.cumsum(doc_tok)
def span_for(tok, width=200):
    d = int(np.searchsorted(cum, tok, side="right"))
    d = min(d, len(doc_tok) - 1)
    pos = int(tok - (cum[d] - doc_tok[d]))
    text = dt["text"][d] or ""
    cp = int(round(pos / max(doc_tok[d], 1) * len(text)))
    return dt["doc_id"][d], pos, text[max(0, cp - width // 2): cp + width // 2].replace("\n", " ")

with open(OUT_MD, "w") as f, open(OUT_JS, "w") as jf:
    f.write(f"# L57 64k feature preview (plateau step 21000, val fvu 0.3800)\n\n")
    f.write(f"SAE: `{CKPT}` (TopK 64k/k=32)\nSample: {n_tok:,} tokens, {len(rgs)} spread row-groups\n")
    f.write("Selection: top-20 by mass among alive latents with fire_rate >= median (fire-rate-weighted)\n\n")
    rep = {"ckpt": CKPT, "n_latents": n_latents, "sample_tokens": n_tok, "latents": {}}
    for i in sel:
        fr = float(fr_np[i]); mx = float(maxact[i].item())
        mean_af = float(mass_np[i] / max(fire[i].item(), 1))
        f.write(f"## latent {i} - fire {fr*100:.2f}% - max_act {mx:.1f} - mean-act-when-firing {mean_af:.2f}\n\n")
        hits = []
        for a, t in sorted(heaps[i], reverse=True):
            did, pos, txt = span_for(t)
            f.write(f"- act **{a:.2f}** @ tok {t} (doc {did}, pos {pos}): «{txt}»\n")
            hits.append({"act": round(a, 3), "token_idx": t, "doc_id": did, "span": txt})
        f.write("\n")
        rep["latents"][int(i)] = {"fire_rate": round(fr, 6), "max_act": round(mx, 4),
                                  "mean_act_when_firing": round(mean_af, 4), "hits": hits}
    json.dump(rep, jf, indent=1)
print("WROTE", OUT_MD, OUT_JS)
