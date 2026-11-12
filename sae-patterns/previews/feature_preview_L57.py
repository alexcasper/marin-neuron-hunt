#!/usr/bin/env python3
"""L57 feature-preview pass: rank latents by activation mass over a spread
sample, then extract top-10 max-activating token positions for the top 20,
mapped to docs.parquet text spans. Writes feature-preview-L57.md + .json.

Alignment rule: token_idx COLUMN only, never row position.
"""
import json, os, math
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

PARQ = "/data/scratch/apx186/marin32b/atlas-resorted/layer57-tokensorted.parquet"
CKPT = "/data/scratch/apx186/marin32b/sae-32k/layer=57/sae_L57_final"
SCALE = "/data/scratch/apx186/marin32b/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = "/data/scratch/apx186/marin32b/atlas/pilot/fineweb-10m-sparse/summaries/docs.parquet"
OUT_MD = "/data/scratch/apx186/marin32b/sae-32k/layer=57/feature-preview-L57.md"
OUT_JS = "/data/scratch/apx186/marin32b/sae-32k/layer=57/feature-preview-L57.json"
SAMPLE_TOKENS = 1_000_000
TOP_LATENTS = 20
TOP_HITS = 10
LAYER = 57
DEV = "cuda:0"

scale_vec = load_scales_vec(SCALE, LAYER, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device=DEV)
sae.eval()
n_latents = sae.encoder.weight.shape[0]

pf = pq.ParquetFile(PARQ)
total_rows = pf.metadata.num_rows
nrg = pf.metadata.num_row_groups
# evenly-spaced row groups -> spread sample across the whole token-sorted file
target_rows = SAMPLE_TOKENS * 64
step = max(1, round(nrg / max(1, target_rows / (total_rows / nrg))))
rgs = list(range(0, nrg, step))
print(f"rows={total_rows} rowgroups={nrg} sampling rgs={rgs[:5]}...{rgs[-1:]} (step {step})", flush=True)

def dense(rb):
    tok = rb.column("token_idx").to_numpy()
    neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32)
    up = rb.column("up").to_numpy().astype(np.float32)
    n = rb.num_rows
    assert n % 64 == 0
    B = n // 64
    x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
    act = gate / (1.0 + np.exp(-gate)) * up
    act = act / scale_vec[neu]
    x[np.repeat(np.arange(B, dtype=np.int64), 64), neu] = act
    toks = tok.reshape(B, 64)[:, 0]
    return x, toks

# ---- pass 1: mass + fire stats over the spread sample ----
mass = torch.zeros(n_latents, device=DEV)
fire = torch.zeros(n_latents, device=DEV)
act_sum = torch.zeros(n_latents, device=DEV)
n_tok = 0
t0 = time.time() if (time := __import__("time")).time() else 0
with torch.no_grad():
    for gi in rgs:
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64:
            rb = rb.slice(0, rb.num_rows // 64 * 64)
        x, _ = dense(rb)
        x = torch.from_numpy(x).to(DEV).to(torch.bfloat16)
        out = sae(x)
        li, la = out.latent_indices.flatten(), out.latent_acts.flatten().float()
        mass.index_add_(0, li, la)
        fire.index_add_(0, li, torch.ones_like(la))
        act_sum.index_add_(0, li, la)
        n_tok += x.shape[0]
print(f"pass1 done: {n_tok} tokens in {time.time()-t0:.0f}s", flush=True)

top = mass.topk(TOP_LATENTS)
top_ids = top.indices.tolist()
stats = {}
for i in top_ids:
    stats[i] = {
        "mass": float(mass[i].item()),
        "fire_rate": float(fire[i].item() / max(n_tok, 1)),
        "mean_act_when_firing": float(act_sum[i].item() / max(fire[i].item(), 1)),
    }

# ---- pass 2: top hits (act, token_idx) for the top latents ----
import heapq
heaps = {i: [] for i in top_ids}  # min-heaps of (act, tok)
seen_rg = set()
with torch.no_grad():
    for gi in rgs:
        if gi in seen_rg:
            continue
        seen_rg.add(gi)
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64:
            rb = rb.slice(0, rb.num_rows // 64 * 64)
        x, toks = dense(rb)
        x = torch.from_numpy(x).to(DEV).to(torch.bfloat16)
        out = sae(x)
        li = out.latent_indices.cpu().numpy()
        la = out.latent_acts.float().cpu().numpy()
        for b in range(li.shape[0]):
            for j in range(li.shape[1]):
                L = int(li[b, j])
                if L in heaps and la[b, j] > 0:
                    h = heaps[L]
                    item = (float(la[b, j]), int(toks[b]))
                    if len(h) < TOP_HITS:
                        heapq.heappush(h, item)
                    elif item > h[0]:
                        heapq.heapreplace(h, item)
print("pass2 done", flush=True)

# ---- token_idx -> (doc, pos) -> text span ----
dt = pq.read_table(DOCS, columns=["doc_id", "doc_tokens", "text"]).to_pydict()
doc_tok = np.array(dt["doc_tokens"], dtype=np.int64)
cum = np.cumsum(doc_tok)
def span_for(tok, width=200):
    d = int(np.searchsorted(cum, tok, side="right"))
    d = min(d, len(doc_tok) - 1)
    pos = int(tok - (cum[d] - doc_tok[d]))
    text = dt["text"][d]
    if not text:
        return d, pos, ""
    cp = int(round(pos / max(doc_tok[d], 1) * len(text)))
    return d, pos, text[max(0, cp - width // 2): cp + width // 2].replace("\n", " ")

report = {"ckpt": CKPT, "layer": LAYER, "sample_tokens": n_tok, "latents": {}}
for rank, i in enumerate(top_ids, 1):
    hits = sorted(heaps[i], reverse=True)
    report["latents"][i] = {"rank": rank, **stats[i],
                            "hits": [{"act": a, "token_idx": t, "doc": span_for(t)[0], "pos": span_for(t)[1], "text": span_for(t)[2]} for a, t in hits]}

json.dump(report, open(OUT_JS, "w"), indent=1)
with open(OUT_MD, "w") as f:
    f.write(f"# L57 feature preview\n\nSAE: `{CKPT}` (TopK 32k/k=32, 6k final)\nSample: {n_tok:,} tokens across {len(rgs)} spread row-groups of layer57-tokensorted.parquet (token_idx-column aligned)\n\n")
    f.write("Ranked by total activation mass over the sample.\n\n")
    for i in top_ids:
        s = stats[i]
        f.write(f"## latent {i} — mass {s['mass']:.1f} · fire {s['fire_rate']*100:.2f}% · mean-act-when-firing {s['mean_act_when_firing']:.2f}\n\n")
        for a, t in sorted(heaps[i], reverse=True):
            d, pos, txt = span_for(t)
            f.write(f"- act **{a:.2f}** @ tok {t} (doc {d}, pos {pos}): «{txt}»\n")
        f.write("\n")
print("WROTE", OUT_MD, OUT_JS)
