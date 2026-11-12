#!/usr/bin/env python3
"""Atlas-direct token-class x layer activity aggregation (alex task 2026-11-08).

Question: which token classes carry each layer's MLP activity? Pure atlas job
— no SAE, no model. Reads the sparse tier (top-64 firings/token, long format
token_idx/neuron/gate/up), computes act = silu(gate)*up on the fly, aggregates
per token, then joins token classes derived from tokens.parquet token_str.

Per layer writes to $OUT (one row per class):
  layer, class, n_tokens, pos_mass, neg_abs_mass, frac_tokens_firing
Classes: special, whitespace, punct, numeric, word_allcaps, word_title,
word_lower, word_cont(=mid-word continuation), other.
Position splits additionally: doc_start (pos==0), early (pos<64).

Outputs land on NODE-LOCAL storage ($TMPDIR) during the run and are copied to
$OUT (network scratch) at the end -- per alex: use node storage where possible.
"""
import json, os, glob, math
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

L = int(os.environ["TCL_LAYER"]); LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
TOK = f"{SCR}/atlas/pilot/fineweb-10m-sparse/summaries/tokens.parquet"
SPARSE_DIR = f"{SCR}/atlas/pilot/fineweb-10m-sparse/sparse/layer={LL}"
TMP = os.environ.get("TMPDIR", "/local")
OUTD = os.path.join(TMP, f"tokenlayer/layer={LL}")   # node-local working dir
OUTD_NET = f"{SCR}/tokenlayer/layer={LL}"            # final (small) copy target
os.makedirs(OUTD, exist_ok=True)

N_TOKENS = 10_003_840

# ---------- token classes ----------
tf = pq.ParquetFile(TOK)
cls = np.full(N_TOKENS, 9, dtype=np.int8)   # 9 = other
pos = np.zeros(N_TOKENS, dtype=np.int32)
CLS_NAMES = ["special", "whitespace", "punct", "numeric", "word_allcaps",
             "word_title", "word_lower", "word_cont", "doc_start?", "other"]
CLS_IDX = {n: i for i, n in enumerate(CLS_NAMES[:8])}
for i in range(tf.metadata.num_row_groups):
    t = tf.read_row_group(i, columns=["token_idx", "pos", "token_str"]).to_pydict()
    idx = np.asarray(t["token_idx"], dtype=np.int64)
    pos[idx] = np.asarray(t["pos"], dtype=np.int32)
    for k, s in zip(idx, t["token_str"]):
        body = s.lstrip("Ġ")  # space-prefix marker
        if s.startswith("<|"):
            c = CLS_IDX["special"]
        elif body == "" or all(ch in " \t\n\r" for ch in body):
            c = CLS_IDX["whitespace"]
        elif all(not ch.isalnum() for ch in body):
            c = CLS_IDX["punct"]
        elif any(ch.isdigit() for ch in body):
            c = CLS_IDX["numeric"]
        elif body.isalpha() and body.isupper():
            c = CLS_IDX["word_allcaps"]
        elif body.isalpha() and body[0].isupper() and body[1:].islower():
            c = CLS_IDX["word_title"]
        elif body.isalpha() and body.islower():
            # word initial (had space prefix) vs mid-word continuation
            c = CLS_IDX["word_lower"] if s.startswith("Ġ") else CLS_IDX["word_cont"]
        elif s.startswith("Ġ"):
            c = CLS_IDX["word_cont"] if not s[1:].startswith("Ġ") else 9
        else:
            c = 9
        cls[k] = c
np.save(os.path.join(OUTD, "cls.npy"), cls)
np.save(os.path.join(OUTD, "pos.npy"), pos)

# ---------- per-layer aggregation (streamed, node-local accumulators) ----------
files = sorted(glob.glob(f"{SPARSE_DIR}/*.parquet"))
assert files, SPARSE_DIR
pos_mass = np.zeros(N_TOKENS, dtype=np.float64)
neg_mass = np.zeros(N_TOKENS, dtype=np.float64)
tok_fired = np.zeros(N_TOKENS, dtype=np.int32)
n_rows = 0
for fp in files:
    pf = pq.ParquetFile(fp)
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["token_idx", "gate", "up"]).to_pydict()
        tok = np.asarray(t["token_idx"], dtype=np.int64)
        g = np.asarray(t["gate"], dtype=np.float32)
        u = np.asarray(t["up"], dtype=np.float32)
        act = (g / (1.0 + np.exp(-g))) * u          # silu(gate)*up
        n_rows += len(tok)
        pos_mass += np.bincount(tok, weights=np.maximum(act, 0), minlength=N_TOKENS)
        neg_mass += np.bincount(tok, weights=np.maximum(-act, 0), minlength=N_TOKENS)
        fired = act != 0
        tok_fired += np.bincount(tok[fired], minlength=N_TOKENS).astype(np.int32)
        if n_rows % 64_000_000 < len(tok):
            print(f"L{LL}: {n_rows/1e6:.0f}M rows", flush=True)

# ---------- collapse to class x layer ----------
n_cls = 10  # classes 0-7 named, 8 "doc_start?" never assigned (skipped), 9 = other
doc_start = pos == 0
rows = []
total_p = pos_mass.sum(); total_n = neg_mass.sum()
for c in range(n_cls):
    m = cls == c
    n = int(m.sum())
    if n == 0:
        continue
    pm = float(pos_mass[m].sum()); nm = float(neg_mass[m].sum())
    fired = int((tok_fired[m] > 0).sum())
    rows.append({
        "layer": L, "class": CLS_NAMES[c], "n_tokens": n,
        "pos_mass": pm, "pos_mass_share": pm / max(total_p, 1e-9),
        "neg_abs_mass": nm, "neg_abs_mass_share": nm / max(total_n, 1e-9),
        "frac_tokens_firing": fired / n,
        "mean_pos_mass_per_firing_tok": pm / max(fired, 1),
    })
# doc-start split (cross-cutting)
pm0 = float(pos_mass[doc_start].sum()); nm0 = float(neg_mass[doc_start].sum())
rows.append({"layer": L, "class": "pos_doc_start", "n_tokens": int(doc_start.sum()),
             "pos_mass": pm0, "pos_mass_share": pm0 / max(total_p, 1e-9),
             "neg_abs_mass": nm0, "neg_abs_mass_share": nm0 / max(total_n, 1e-9),
             "frac_tokens_firing": float((tok_fired[doc_start] > 0).mean()),
             "mean_pos_mass_per_firing_tok": pm0 / max(int((tok_fired[doc_start] > 0).sum()), 1)})
tbl = pa.Table.from_pylist(rows)
pq.write_table(tbl, os.path.join(OUTD, "class_summary.parquet"))
json.dump({"layer": L, "rows": n_rows, "total_pos": total_p, "total_neg": total_n},
          open(os.path.join(OUTD, "meta.json"), "w"))

# small outputs -> network scratch mirror
os.makedirs(OUTD_NET, exist_ok=True)
pq.write_table(tbl, os.path.join(OUTD_NET, "class_summary.parquet"))
json.dump(json.load(open(os.path.join(OUTD, "meta.json"))),
          open(os.path.join(OUTD_NET, "meta.json"), "w"))
print("TCL DONE", json.dumps({"layer": L, "classes": len(rows), "rows": n_rows}), flush=True)
