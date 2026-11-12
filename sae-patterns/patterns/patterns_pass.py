#!/usr/bin/env python3
"""Per-latent pattern pass for sae-64k layers 0-2 (alex task 2026-11-06).

Encode-only CPU pass over the 200k-token spread sample (same protocol as
interp_preview_12k.py). For every alive latent: top-8 hits (+-16-tok text
context), fire rate, mean/max act, position histogram (first16/first64/rest
of doc), top doc concentration, top left/right neighbor words. Classifies
each latent into a family and writes patterns.jsonl / families.json /
notable.json. Also recomputes encode-side fvu (manual decode from
top_indices/top_acts via W_dec) on a 20k-token prefix as the census-anomaly
cross-check.

Caveat (documented): the atlas carries token positions + doc text, not the
tokenized stream; "token form" and "neighbor n-gram" are word-level
approximations via char-position mapping.
"""
import json, os, re, glob, math, heapq, collections
import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import load_scales_vec, D_MODEL_MLP

L = int(os.environ["PAT_LAYER"])
LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
OUTD = f"{SCR}/sae-64k/layer={LL}/interp"
SCALE = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet"
DOCS = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/docs.parquet"
SAMPLE_TOKENS = int(os.environ.get("PAT_TOKENS", "200000"))
TOP_HITS = 8
FVU_TOKENS = 20000
DEV = "cpu"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
print(f"CPU mode: {torch.get_num_threads()} threads", flush=True)
import time as _t; _T0 = _t.time()

CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
assert os.path.exists(os.path.join(CKPT, "cfg.json")), CKPT
os.makedirs(OUTD, exist_ok=True)
if os.environ.get("PAT_RAW", "0") == "1":
    scale_vec = None  # raw-acts convention (matches 64k sweep training: scale_vec was silently None)
    print("RAW-ACTS MODE (scale_vec=None, matches 64k training convention)", flush=True)
else:
    scale_vec = load_scales_vec(SCALE, L, D_MODEL_MLP)
sae = Sae.load_from_disk(CKPT, device=DEV); sae.eval()
n_latents = sae.encoder.weight.shape[0]
W_dec_L = sae.W_dec.detach().float().to(DEV)  # [n_latents, d_model], latent-major
print(f"L{LL} ckpt={CKPT} n_latents={n_latents}", flush=True)

pf = pq.ParquetFile(PARQ)
# docs mapping needed during streaming (position buckets)
_dt = pq.read_table(DOCS, columns=["doc_id", "doc_tokens", "text"]).to_pydict()
doc_tok = np.array(_dt["doc_tokens"], dtype=np.int64); cum = np.cumsum(doc_tok)
total_rows = pf.metadata.num_rows
nrg = pf.metadata.num_row_groups
tokens_per_rg = (total_rows / nrg) / 64.0
n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
step = max(1, nrg // n_needed)
rgs = list(range(0, nrg, step))
print(f"L{LL}: rows={total_rows} rgs={len(rgs)} sample={SAMPLE_TOKENS}", flush=True)

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
    return x, tok.reshape(B, 64)[:, 0]

# accumulators
fire = np.zeros(n_latents, dtype=np.int64)
actsum = np.zeros(n_latents, dtype=np.float64)
maxact = np.zeros(n_latents, dtype=np.float32)
heaps = collections.defaultdict(list)   # latent -> [(act, tok)] size<=8
pos16 = np.zeros(n_latents, dtype=np.int64)
pos64 = np.zeros(n_latents, dtype=np.int64)
fvu_n = fvu_d = 0.0; fvu_tok = 0
n_tok = 0

with torch.no_grad():
    for gi in rgs:
        rb = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"]).combine_chunks()
        if rb.num_rows % 64:
            rb = rb.slice(0, rb.num_rows // 64 * 64)
        if rb.num_rows == 0:
            continue
        x, toks = dense(rb)
        xb = torch.from_numpy(x).to(torch.bfloat16)
        out = sae.encode(xb)
        li = out.top_indices.cpu().numpy(); la = out.top_acts.float().cpu().numpy()
        # encode-side fvu on a prefix
        if fvu_tok < FVU_TOKENS:
            idx_t = out.top_indices.long(); a_t = out.top_acts.float()
            for s in range(0, idx_t.shape[0], 256):
                i2, a2 = idx_t[s:s+256], a_t[s:s+256]
                xh = (W_dec_L[i2] * a2.unsqueeze(-1)).sum(1)
                r = xb[s:s+256].float() - xh
                fvu_n += (r ** 2).sum().item(); fvu_d += (xb[s:s+256].float() ** 2).sum().item()
            fvu_tok += idx_t.shape[0]
        B, K = li.shape
        flat_l = li.reshape(-1); flat_a = la.reshape(-1)
        np.add.at(fire, flat_l, 1)
        np.add.at(actsum, flat_l, flat_a)
        np.maximum.at(maxact, flat_l, flat_a)
        h_of = heaps
        for b in range(B):
            tok = int(toks[b])
            d = int(np.searchsorted(cum, tok, side="right")); d = min(d, len(doc_tok) - 1)
            pos = int(tok - (cum[d] - doc_tok[d]))
            for j in range(K):
                Lid = int(li[b, j]); a = float(la[b, j])
                if a <= 0: continue
                if pos < 16: pos16[Lid] += 1
                elif pos < 64: pos64[Lid] += 1
                h = h_of[Lid]
                it = (a, tok)
                if len(h) < TOP_HITS: heapq.heappush(h, it)
                elif it > h[0]: heapq.heapreplace(h, it)
        n_tok += B
        if n_tok % 50000 < 1024:
            print(f"  {n_tok} tok, {_t.time()-_T0:.0f}s", flush=True)

fvu_re = fvu_n / max(fvu_d, 1e-9)
alive_idx = np.flatnonzero(fire > 0)
_conc10 = float(np.sort(actsum)[::-1][:10].sum() / max(float(actsum.sum()), 1e-9))
print(f"encode done n_tok={n_tok} alive={len(alive_idx)} fvu_recomputed={fvu_re:.4f} act_mass_top10_share={_conc10:.4f}", flush=True)

# ---- text helpers ----
dt = _dt
def span_for(tok, width_tok=16):
    d = int(np.searchsorted(cum, tok, side="right")); d = min(d, len(doc_tok) - 1)
    pos = int(tok - (cum[d] - doc_tok[d]))
    text = dt["text"][d] or ""
    cp = int(round(pos / max(doc_tok[d], 1) * len(text)))
    w = max(16, int(2 * width_tok * len(text) / max(doc_tok[d], 1)))
    return d, pos, text[max(0, cp - w): cp + w].replace("\n", " ")
WORD = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")
def form_at(tok):
    d, pos, _ = span_for(tok, 0)
    text = dt["text"][d] or ""
    cp = int(round(pos / max(doc_tok[d], 1) * len(text)))
    m = None
    for m2 in WORD.finditer(text, max(0, cp - 40), cp + 40):
        if m2.start() <= cp <= m2.end(): m = m2; break
    return (m.group(0) if m else ""), d, pos, cp, text
def neighbors(cp, text):
    after = WORD.search(text, cp + 1); before = None
    for m2 in WORD.finditer(text, max(0, cp - 60), cp):
        before = m2
    return (before.group(0) if before else ""), (after.group(0) if after else "")

PUNCT = re.compile(r"^[^\w]+$|^\s*$")
fam_counts = collections.Counter()
records = []
fam_examples = {}
for Lid in alive_idx.tolist():
    h = sorted(heaps[Lid], reverse=True)
    forms, docs_c, lefts, rights = [], collections.Counter(), collections.Counter(), collections.Counter()
    hits = []
    for a, tok in h:
        d, pos, span = span_for(tok)
        form, _, _, cp, text = form_at(tok)
        lb, ra = neighbors(cp, text)
        forms.append(form); docs_c[d] += 1
        if lb: lefts[lb.lower()] += 1
        if ra: rights[ra.lower()] += 1
        hits.append({"act": round(a, 3), "tok": tok, "doc": int(dt["doc_id"][d]),
                     "pos": pos, "form": form, "span": span})
    fr = fire[Lid] / max(n_tok, 1)
    tot = fire[Lid]
    form_share = collections.Counter([f.lower() for f in forms]).most_common(1)
    form_share = form_share[0][1] / max(len(forms), 1) if form_share else 0.0
    top_form = collections.Counter([f.lower() for f in forms]).most_common(1)[0][0] if forms else ""
    doc_share = sum(c for _, c in docs_c.most_common(2)) / max(len(hits), 1)
    lbest = lefts.most_common(1); rbest = rights.most_common(1)
    ngram_share = max(lbest[0][1] if lbest else 0, rbest[0][1] if rbest else 0) / max(len(hits), 1)
    p16 = pos16[Lid] / max(tot, 1)
    # family decision order: punct -> token-identity -> n-gram -> position -> doc-identity -> other
    punct_frac = sum(1 for f in forms if PUNCT.match(f)) / max(len(forms), 1)
    if punct_frac >= 0.8: fam = "punct_ws"
    elif form_share >= 0.8: fam = "token_identity"
    elif ngram_share >= 0.5: fam = "ngram"
    elif p16 > 0.5: fam = "position"
    elif doc_share >= 0.5 and len(docs_c) <= 3: fam = "doc_identity"
    else: fam = "other"
    fam_counts[fam] += 1
    key = {"punct_ws": punct_frac, "token_identity": form_share, "ngram": ngram_share,
           "position": p16, "doc_identity": doc_share}.get(fam, 0.0)
    if fam not in fam_examples or key > fam_examples[fam][0]:
        fam_examples[fam] = (key, Lid)
    records.append({
        "latent_id": Lid, "family": fam, "fire_rate": round(float(fr), 6),
        "mean_act": round(float(actsum[Lid] / max(tot, 1)), 4),
        "max_act": round(float(maxact[Lid]), 4),
        "pos_hist": {"first16": int(pos16[Lid]), "first64": int(pos64[Lid]), "rest": int(tot - pos16[Lid] - pos64[Lid])},
        "top_form": top_form, "form_share": round(form_share, 3),
        "top_docs": docs_c.most_common(3), "doc_share": round(doc_share, 3),
        "left_ngrams": lefts.most_common(3), "right_ngrams": rights.most_common(3),
        "hits": hits,
    })

with open(f"{OUTD}/patterns.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r) + "\n")
with open(f"{OUTD}/families.json", "w") as f:
    json.dump({"layer": L, "ckpt": CKPT, "tokens": n_tok, "alive": int(len(alive_idx)),
               "fvu_recomputed_encode_side": round(fvu_re, 6),
               "act_mass_top10_latents_share": round(_conc10, 6),
               "input_convention": ("raw_acts (rerun 2026-11-07, matches 64k training; supersedes scaled-input v1)" if os.environ.get("PAT_RAW") == "1" else "scale_normalized"),
               "families": dict(fam_counts),
               "note": "forms/ngrams are word-level approximations (atlas has positions+text, not token ids)"}, f, indent=1)

# notable: 3 highest max_act, cleanest per family, weirdest = max doc_share
by_fam = {}
for r in records: by_fam.setdefault(r["family"], []).append(r)
notable = []
for r in sorted(records, key=lambda r: -r["max_act"])[:3]:
    notable.append({**r, "why": "max_act top-3"})
for fam, (key, Lid) in sorted(fam_examples.items()):
    r = next(x for x in records if x["latent_id"] == Lid)
    notable.append({**r, "why": f"cleanest {fam} ({key:.2f})"})
wd = max(records, key=lambda r: r["doc_share"])
notable.append({**wd, "why": "weirdest doc-concentration"})
seen = set(); nn = []
for r in notable:
    if r["latent_id"] in seen: continue
    seen.add(r["latent_id"]); nn.append(r)
with open(f"{OUTD}/notable.json", "w") as f:
    json.dump(nn[:10], f, indent=1)
print(f"PATTERNS DONE L{LL}: alive={len(records)} fams={dict(fam_counts)} fvu_re={fvu_re:.4f} ({_t.time()-_T0:.0f}s)", flush=True)
