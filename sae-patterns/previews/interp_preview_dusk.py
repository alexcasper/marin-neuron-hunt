#!/usr/bin/env python3
"""Dusk fallback: 20-layer interp-preview extraction (mirror of apoc interp_preview.py).

Layout difference: dusk sparse shards are NEURON-major (sorted by neuron, then
token_idx) — so instead of row-group spread sampling we filter-gather a random
200k-token sample via duckdb, rebuild dense x, and cache it as a bf16 bit-view
(uint16 memmap) for two encode passes (stats, then top-hits).

Output schema identical to the apoc script:
  ~/marin-sae/interp-out/layer=NN/preview.jsonl
Tracks: mass top-12 / firerate-gated top-12 / random-alive-12 (seed 20261030+L).
Checkpoint: sae_LNN_final 6k finals pulled from alxcspr/stationhouse.
"""
import json, os, heapq
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11,37,38,39,40,41,56,57,58]
L = int(os.environ["PREVIEW_LAYER"])
LL = f"{L:02d}"
SPARSE = f"/home/dusk/marin-atlas/fineweb-10m-sparse/sparse/layer={LL}"
CKPT = f"/home/dusk/marin-sae/hf-finals/sae-32k/layer={LL}/sae_L{LL}_final"
SCALE = "/home/dusk/marin-atlas/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = "/home/dusk/marin-atlas/fineweb-10m-sparse/summaries/docs.parquet"
OUTD = f"/home/dusk/marin-sae/interp-out/layer={LL}"
MMAP = f"/home/dusk/marin-sae/tmp/x_L{LL}.u16"
SAMPLE_TOKENS = int(os.environ.get("PREVIEW_TOKENS", "200000"))
N_TOKENS_FILE = 9_995_501
TOP_HITS = 8
SEED = 20261030 + L
DEV = "cuda:0"
BATCH = 4096
assert L in LAYERS

os.makedirs(OUTD, exist_ok=True)
os.makedirs(os.path.dirname(MMAP), exist_ok=True)
scale_vec = load_scales_vec(SCALE, L, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device=DEV)
sae.eval()
n_latents = sae.encoder.weight.shape[0]

rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(N_TOKENS_FILE, size=SAMPLE_TOKENS, replace=False))
ids = pa.table({"token_idx": pa.array(sample, type=pa.int64())})
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=8")
con.register("ids", ids)

tb = con.execute(
    f"SELECT s.token_idx, r.neuron, r.gate, r.up FROM ids s "
    f"JOIN read_parquet('{SPARSE}/*.parquet') r ON r.token_idx = s.token_idx"
).to_arrow_table().combine_chunks()
tok = tb.column("token_idx").to_numpy()
neu = tb.column("neuron").to_numpy()
gate = tb.column("gate").to_numpy().astype(np.float32)
up = tb.column("up").to_numpy().astype(np.float32)
order = np.argsort(tok, kind="stable")
tok, neu, gate, up = tok[order], neu[order], gate[order], up[order]
pos = np.searchsorted(sample, tok)
counts = np.bincount(pos, minlength=len(sample))
keep_tok = counts == 64  # ~0.04% of tokens have gaps in the capture; drop them
row_keep = keep_tok[pos]
if not keep_tok.all():
    tok, neu, gate, up = tok[row_keep], neu[row_keep], gate[row_keep], up[row_keep]
    pos = np.searchsorted(sample, tok)
    sample_kept = sample[keep_tok]
else:
    sample_kept = sample
B = int(keep_tok.sum())
print(f"L{LL}: {len(tok)} rows -> {B} tokens (dropped {len(sample)-B} gapped), ckpt={CKPT}", flush=True)

act = gate / (1.0 + np.exp(-gate)) * up
act /= scale_vec[neu]

mm = np.memmap(MMAP, dtype=np.uint16, mode="w+", shape=(B, D_MODEL_MLP))
for s in range(0, B, 2048):
    n = min(2048, B - s)
    xb = np.zeros((n, D_MODEL_MLP), dtype=np.float32)
    blk = slice(s * 64, (s + n) * 64)
    xb[np.repeat(np.arange(n, dtype=np.int64), 64), neu[blk]] = act[blk]
    mm[s:s+n] = torch.from_numpy(xb).to(torch.bfloat16).view(torch.uint16).numpy()
mm.flush()
del tb, tok, neu, gate, up, act

def load_blk(s):
    n = min(BATCH, B - s)
    raw = np.asarray(mm[s:s+n])
    return torch.from_numpy(raw).view(torch.bfloat16).to(DEV).to(torch.bfloat16)

def encode_pass(want_hits):
    mass = torch.zeros(n_latents, device=DEV)
    fire = torch.zeros(n_latents, device=DEV)
    maxact = torch.zeros(n_latents, device=DEV)
    seen = torch.zeros(n_latents, dtype=torch.bool, device=DEV)
    n_tok = 0
    heaps = {i: [] for i in want_hits}
    with torch.no_grad():
        for s in range(0, B, BATCH):
            x = load_blk(s)
            out = sae(x)
            li, la = out.latent_indices.flatten(), out.latent_acts.flatten().float()
            mass.index_add_(0, li, la)
            fire.index_add_(0, li, torch.ones_like(la))
            m = torch.zeros(n_latents, device=DEV)
            m.scatter_reduce_(0, li, la, reduce="amax", include_self=True)
            maxact = torch.maximum(maxact, m)
            seen[out.latent_indices.unique()] = True
            n_tok += x.shape[0]
            if want_hits:
                lin = out.latent_indices.cpu().numpy()
                lan = out.latent_acts.float().cpu().numpy()
                tks = sample_kept[s:s+x.shape[0]][:, None]
                for b in range(lin.shape[0]):
                    for j in range(lin.shape[1]):
                        Lid = int(lin[b, j])
                        h = heaps.get(Lid)
                        if h is not None and lan[b, j] > 0:
                            item = (float(lan[b, j]), int(tks[b, 0]))
                            if len(h) < TOP_HITS:
                                heapq.heappush(h, item)
                            elif item > h[0]:
                                heapq.heapreplace(h, item)
    return mass, fire, fire / max(n_tok, 1), maxact, seen, heaps

mass, fire, fire_rate, maxact, seen, _ = encode_pass(frozenset())
alive_idx = np.flatnonzero(seen.cpu().numpy())
mass_np, fr_np = mass.cpu().numpy(), fire_rate.cpu().numpy()
med = np.median(fr_np[alive_idx])
sel = {}
sel["mass"] = list(np.argsort(-mass_np)[:12])
sel["firerate"] = list(np.argsort(-mass_np * (fr_np >= med))[:12])
sel["random"] = [int(i) for i in rng.choice(alive_idx, size=min(12, len(alive_idx)), replace=False)]
picked = frozenset(sel["mass"]) | frozenset(sel["firerate"]) | frozenset(sel["random"])
print(f"L{LL}: tracks selected ({len(picked)} latents), pass2", flush=True)

_, _, _, _, _, heaps = encode_pass(picked)

dt = pq.read_table(DOCS, columns=["doc_id", "doc_tokens", "text"]).to_pydict()
doc_tok = np.array(dt["doc_tokens"], dtype=np.int64)
cum = np.cumsum(doc_tok)
def span_for(tokid, width=200):
    d = int(np.searchsorted(cum, tokid, side="right"))
    d = min(d, len(doc_tok) - 1)
    pos = int(tokid - (cum[d] - doc_tok[d]))
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
os.remove(MMAP)
print(f"WROTE {OUTD}/preview.jsonl ({B} tokens, {len(alive_idx)} alive)", flush=True)
