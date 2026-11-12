"""Equivalence + bug-repro tests for nominate_enc_pass_v2 (CPU, synthetic data).
Run: NOM_LAYER=0 uv run --python 3.12 --with numpy --with torch python test_nominate_v2.py
"""
import os, sys
os.environ.setdefault("NOM_LAYER", "0")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nominate_enc_pass_v2 as v2

rng = np.random.default_rng(0)
n_lat, d, T, K = 300, 128, 700, 8
E_np = rng.standard_normal((n_lat, d)).astype(np.float32)
# sparse signed activations, like silu(gate)*up top-64 rows
X = np.zeros((T, d), np.float32)
for r in range(T):
    idx = rng.choice(d, 16, replace=False)
    X[r, idx] = rng.standard_normal(16).astype(np.float32) * 3
zp = X @ E_np.T
topk = np.argsort(-np.maximum(zp, 0), 1)[:, :K]           # [T,K] distinct per row
attr_lat = np.sort(rng.choice(n_lat, 120, replace=False))
lat_pos = {int(j): p for p, j in enumerate(attr_lat)}

# ---- v1 reference loop (verbatim logic from nominate_enc_pass.py:176-196) ----
accum = np.zeros((len(attr_lat), d), np.float64); fires = np.zeros(len(attr_lat), np.int64)
for r in range(T):
    x = X[r]
    for j in topk[r].tolist():
        p = lat_pos.get(j)
        if p is None: continue
        fires[p] += 1
        accum[p] += np.abs(E_np[j] * x)
fired = fires > 0
sh = np.zeros_like(accum); den = accum.sum(1, keepdims=True)
sh[fired] = accum[fired] / np.maximum(den[fired], 1e-12)
s1_ref, n1_ref = sh.max(1), sh.argmax(1)
eff_ref = 1.0 / np.maximum((sh ** 2).sum(1), 1e-12)

# ---- v2, chunked (chunk not dividing T, to exercise the tail) ----
pos_of = torch.full((n_lat,), -1, dtype=torch.long)
pos_of[torch.as_tensor(attr_lat)] = torch.arange(len(attr_lat))
S = torch.zeros(len(attr_lat), d, dtype=torch.float64)
f = torch.zeros(len(attr_lat), dtype=torch.long)
Xt, Kt = torch.from_numpy(X), torch.from_numpy(topk)
for r0 in range(0, T, 256):
    v2.accumulate_chunk(Xt[r0:r0 + 256], Kt[r0:r0 + 256], pos_of, S, f)
s1, n1, eff, _ = v2.shares_from(torch.from_numpy(np.abs(E_np[attr_lat])), S)
s1 = np.where(f.numpy() > 0, s1.numpy(), 0.0)

assert np.array_equal(f.numpy(), fires), "fire counts differ"
assert np.allclose(s1, s1_ref, rtol=1e-6, atol=1e-9), np.abs(s1 - s1_ref).max()
assert np.array_equal(n1.numpy()[fired], n1_ref[fired]), "top1 neuron differs"
assert np.allclose(eff.numpy()[fired], eff_ref[fired], rtol=1e-6)
print(f"PASS equivalence: {fired.sum()}/{len(attr_lat)} fired, "
      f"max |ds1|={np.abs(s1 - s1_ref).max():.2e}, fires exact, top1 neuron exact")

# ---- column shuffle preserves per-column marginals, breaks rows ----
g = torch.Generator(); g.manual_seed(1)
Xs = v2.column_shuffle(Xt, g, colblk=50)
assert torch.equal(torch.sort(Xs, 0).values, torch.sort(Xt, 0).values), "marginals changed"
assert not torch.equal(Xs, Xt)
print("PASS column_shuffle: per-column multisets preserved, rows permuted")

# ---- v1 control bug: searchsorted on an unsorted ctrl_idx ----
ctrl_idx = np.random.default_rng(20261108).choice(len(attr_lat), size=60, replace=False)
hits = sum(1 for p in range(len(attr_lat))
           if (ci := np.searchsorted(ctrl_idx, p)) < len(ctrl_idx) and ctrl_idx[ci] == p)
hits_sorted = sum(1 for p in range(len(attr_lat))
                  if (ci := np.searchsorted(np.sort(ctrl_idx), p)) < len(ctrl_idx) and np.sort(ctrl_idx)[ci] == p)
print(f"v1 control lookup: {hits}/60 control latents reachable unsorted vs {hits_sorted}/60 sorted")
assert hits_sorted == 60 and hits < 60
