#!/usr/bin/env python3
"""20-layer interp-preview extraction (extends feature_preview_L57.py).

Per layer: 200k-token spread sample -> SAE encode -> three selection tracks
x 12 latents each (mass / fire-rate-gated mass / random-alive-seeded) ->
top-8 activating tokens per latent -> docs.parquet text spans.
Output: sae-32k/layer=NN/interp-preview/preview.jsonl (one JSON line/latent).

Tracks (documented choice):
  mass     = top-12 by total activation mass over the sample
  firerate = top-12 by mass AMONG alive latents with fire_rate >= median
             fire_rate of alive latents (gates off the doc-identity spam
             latents that are massive but rarely fire)
  random   = 12 uniformly random ALIVE latents, RNG seeded 20261030+layer
Spans: proportional char-position mapping (no tokenizer offsets in the atlas)
— approximate +/- ~30 chars; the activating token sits at the span midpoint.
Checkpoint: sae_LNN_final (the 6k finals; these 20 layers were not topped up).
Alignment: token_idx COLUMN only, never row position.
"""
import json, os, heapq, math
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11,37,38,39,40,41,56,57,58]
L = int(os.environ["PREVIEW_LAYER"])
LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
CKPT = f"{SCR}/sae-32k/layer={LL}/sae_L{LL}_final"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/docs.parquet"
OUTD = f"{SCR}/sae-32k/layer={LL}/interp-preview"
SAMPLE_TOKENS = int(os.environ.get("PREVIEW_TOKENS", "200000"))
TOP_HITS = 8
SEED = 20261030 + L
DEV = os.environ.get("PREVIEW_DEV", "cuda:0")
assert L in LAYERS
import time as _t; _T0 = _t.time()
if DEV.startswith("cpu"):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    print(f"CPU mode: {torch.get_num_threads()} threads", flush=True)

os.makedirs(OUTD, exist_ok=True)
scale_vec = load_scales_vec(SCALE, L, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device=DEV)
sae.eval()
n_latents = sae.encoder.weight.shape[0]

pf = pq.ParquetFile(PARQ)
total_rows = pf.metadata.num_rows
nrg = pf.metadata.num_row_groups
# evenly-spaced row groups -> spread sample across the whole file
# tokens_per_rg = rows_per_rowgroup / 64 (64 rows per token in the sparse tier)
tokens_per_rg = (total_rows / nrg) / 64.0
n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
step = max(1, nrg // n_needed)
rgs = list(range(0, nrg, step))
print(f"L{LL}: rows={total_rows} rgs={len(rgs)} ckpt={CKPT} dev={DEV} sample={SAMPLE_TOKENS}", flush=True)

def dense(rb):
    tok = rb.column("token_idx").to_numpy()
    neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32)
    up = rb.column("up").to_numpy().astype(np.float32)
    n = rb.num_rows
    B = n // 64
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
        if n_tok % 50000 < 1024:
            print(f"  encode {n_tok} tokens, {_t.time()-_T0:.0f}s elapsed ({n_tok/max(_t.time()-_T0,1):.0f} tok/s)", flush=True)
fire_rate = fire / max(n_tok, 1)

# ---- track selection ----
alive = seen.cpu().numpy()
mass_np = mass.cpu().numpy()
fr_np = fire_rate.cpu().numpy()
alive_idx = np.flatnonzero(alive)
med = np.median(fr_np[alive_idx])
sel = {}
sel["mass"] = list(np.argsort(-mass_np)[:12])
sel["firerate"] = list(np.argsort(-mass_np * (fr_np >= med))[:12])
rng = np.random.default_rng(SEED)
sel["random"] = [int(i) for i in rng.choice(alive_idx, size=min(12, len(alive_idx)), replace=False)]
picked = set(sel["mass"]) | set(sel["firerate"]) | set(sel["random"])

# ---- pass 2: top hits for picked latents ----
heaps = {i: [] for i in picked}
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
print("pass2 done", flush=True)

# ---- docs mapping ----
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

with open(f"{OUTD}/preview.jsonl", "w") as f:
    for track in ("mass", "firerate", "random"):
        for i in sel[track]:
            hits = []
            for a, t in sorted(heaps.get(i, []), reverse=True):
                did, pos, txt = span_for(t)
                hits.append({"token_idx": t, "doc_id": did, "span": txt, "act": round(a, 3)})
            rec = {"layer": L, "latent_id": int(i), "track": track,
                   "fire_rate": round(float(fr_np[i]), 6),
                   "max_act": round(float(maxact[i].item()), 4),
                   "mean_act_when_firing": round(float(mass_np[i] / max(fire[i].item(), 1)), 4),
                   "hits": hits, "seed": SEED}
            f.write(json.dumps(rec) + "\n")
print(f"WROTE {OUTD}/preview.jsonl ({n_tok} tokens)", flush=True)
