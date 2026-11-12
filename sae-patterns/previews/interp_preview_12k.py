#!/usr/bin/env python3
"""12k interp-preview extraction (sample of 8 topup layers, step12000 ckpts).

Same family as interp_preview.py (34661828/34666766 sprint), changes:
  - checkpoint = sae_LNN_step12000 re-pulled from HF alxcspr/stationhouse
    (scratch sae-32k was cleared to fund the 64k sweep; repo is public-read)
  - tracks: top-12 mass + 12 random-alive (fire-rate-gated track dropped —
    proven a no-op last sprint)
  - output dir: sae-32k/layer=NN/interp-preview-12k/ (recreated on scratch)

Layer sample (documented choice): 12 18 26 33 42 50 55 63 — even spread over
the 44-layer topup range, including both fvu extremes: L55 = min (0.8727),
L63 = max (1.0365).
"""
import json, os, math, heapq
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

L = int(os.environ["PREVIEW_LAYER"])
LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
OUTD = f"{SCR}/sae-32k/layer={LL}/interp-preview-12k"
CKPT = f"{SCR}/hf-12k/sae-32k/layer={LL}/sae_L{LL}_step12000"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/docs.parquet"
SAMPLE_TOKENS = int(os.environ.get("PREVIEW_TOKENS", "200000"))
TOP_HITS = 8
SEED = 20261106 + L
DEV = os.environ.get("PREVIEW_DEV", "cpu")
import time as _t; _T0 = _t.time()
if DEV.startswith("cpu"):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    print(f"CPU mode: {torch.get_num_threads()} threads", flush=True)

# ---- re-pull ckpt from HF if missing (repo is public-read) ----
if not os.path.exists(os.path.join(CKPT, "cfg.json")):
    from huggingface_hub import snapshot_download
    print(f"pulling ckpt from HF -> {CKPT}", flush=True)
    snapshot_download(
        repo_id="alxcspr/stationhouse",
        allow_patterns=[f"sae-32k/layer={LL}/sae_L{LL}_step12000/*",
                        f"sae-32k/layer={LL}/stats_12k.json"],
        local_dir=f"{SCR}/hf-12k",
        max_workers=4,
    )
assert os.path.exists(os.path.join(CKPT, "cfg.json")), f"ckpt missing: {CKPT}"

os.makedirs(OUTD, exist_ok=True)
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
        out = sae.encode(x)
        li, la = out.top_indices.flatten().long(), out.top_acts.flatten().float()
        mass.index_add_(0, li, la)
        fire.index_add_(0, li, torch.ones_like(la))
        m = torch.zeros(n_latents, device=DEV)
        m.scatter_reduce_(0, li, la, reduce="amax", include_self=True)
        maxact = torch.maximum(maxact, m)
        seen[out.top_indices.unique()] = True
        n_tok += x.shape[0]
        if n_tok % 50000 < 1024:
            print(f"  encode {n_tok} tokens, {_t.time()-_T0:.0f}s elapsed ({n_tok/max(_t.time()-_T0,1):.0f} tok/s)", flush=True)
fire_rate = fire / max(n_tok, 1)

alive = seen.cpu().numpy()
mass_np = mass.cpu().numpy()
fr_np = fire_rate.cpu().numpy()
alive_idx = np.flatnonzero(alive)
sel = {}
sel["mass"] = list(np.argsort(-mass_np)[:12])
rng = np.random.default_rng(SEED)
sel["random"] = [int(i) for i in rng.choice(alive_idx, size=min(12, len(alive_idx)), replace=False)]
picked = set(sel["mass"]) | set(sel["random"])

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
        out = sae.encode(x)
        li = out.top_indices.cpu().numpy()
        la = out.top_acts.float().cpu().numpy()
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
    for track in ("mass", "random"):
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
