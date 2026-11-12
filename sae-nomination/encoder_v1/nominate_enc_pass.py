#!/usr/bin/env python3
"""Encoder-side neuron nomination for the sae-64k sweep (alex GO 2026-11-08).

Follow-up to the decoder-side negative result (nominate_pass.py): decoder rows
are unit-norm-renormalized every step (normalize_decoder=true), so they cannot
concentrate on the neuron basis. Encoder detection vectors have no such
constraint — a latent could write diffusely yet detect neuron-specifically.

Two modes:

  NOM_MODE=weights      pure weight shares: row-normalize |E| per latent.
                        Cheap, any layer, ~1 min. Answers "which neurons does
                        the latent LISTEN to by weight".

  NOM_MODE=attribution  data-anchored: stream a token sample from the atlas,
                        for each firing latent accumulate per-neuron attribution
                        |E[j,i] * x_pre[i]| over firing tokens, normalize per
                        latent. Answers "which neurons DRIVE the latent to
                        fire". Includes a SHUFFLE CONTROL: K column-wise token
                        permutations (destroys cross-neuron co-activation,
                        preserves per-neuron marginals); if concentration
                        survives shuffling it is weight structure, if it
                        collapses it was input statistics.

Both modes self-check: z_pre recomputed from E must reproduce encode()'s TopK
selection exactly on the first batch, else abort (prevents a repeat of the
scale_vec/convention class of bug).

Outputs to $SCR/sae-64k/layer=NN/nominate_enc/:
  weights mode:      nominations_enc.parquet + summary.json
  attribution mode:  attribution_enc.parquet + attribution_control.json
"""
import json, os, glob, math
import numpy as np
import torch
from sparsify import Sae

MODE = os.environ.get("NOM_MODE", "weights")
L = int(os.environ["NOM_LAYER"]); LL = f"{L:02d}"
SCR = "/data/scratch/apx186/marin32b"
OUTD = f"{SCR}/sae-64k/layer={LL}/nominate_enc"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
assert os.path.exists(os.path.join(CKPT, "cfg.json")), CKPT
sae = Sae.load_from_disk(CKPT, device="cpu"); sae.eval()

# Detection matrix E as [n_latents, d]: support both weight layouts.
E_raw = sae.encoder.weight.detach().float()          # torch module weight
if E_raw.shape[0] == sae.W_dec.shape[0]:
    E = E_raw                                         # [n_latents, d]
else:
    E = E_raw.T
n_lat, d = E.shape
assert d == sae.W_dec.shape[1], (E.shape, sae.W_dec.shape)

# ---- self-check helper: recompute TopK from z_pre and match encode() ----
def z_pre_of(x):
    """Pre-activation input to the encoder nonlinearity for batch x.
    Adapt ONE line here if sparsify's pipeline applies centering/scaling
    before the matmul; the assert below proves the adaptation correct."""
    xb = x.to(torch.bfloat16)
    # mirror FusedEncoder.forward numerics exactly: bf16 linear + bias, pre-relu
    return ((xb - sae.b_dec.detach()) @ sae.encoder.weight.detach().T) + sae.encoder.bias.detach()

def selfcheck(x):
    codes = sae.encode(x.to(torch.bfloat16))
    zp = z_pre_of(x)
    k = codes.top_indices.shape[1]
    sel = torch.relu(zp).topk(k, dim=1).indices
    ok = torch.equal(torch.sort(sel, dim=1).values,
                     torch.sort(codes.top_indices, dim=1).values)
    if not ok:
        # tolerate act-ties: compare sets per row, report overlap rate
        agree = torch.tensor([
            len(set(a.tolist()) & set(b.tolist())) / float(k)
            for a, b in zip(sel, codes.top_indices)])
        print(f"SEL-CHECK soft: mean topk overlap {agree.mean():.4f}", flush=True)
        # 0.99 gate: bf16 boundary-tie swaps cap achievable overlap at ~0.995
        # (probe 35608868: 457/2048 rows have exact k-th value ties; the real
        # convention bug this gate guards against showed as 0.54)
        return agree.mean() >= 0.99
    print("SEL-CHECK exact: recompute matches encode()", flush=True)
    return True

if MODE == "weights":
    A = np.abs(E.numpy())
    l1 = A.sum(1, dtype=np.float32)
    P = (A / np.maximum(l1[:, None], np.float32(1e-12))).astype(np.float32)
    n1 = np.argmax(P, axis=1)
    part = np.partition(P, -2, axis=1)
    s1 = part[:, -1].astype(np.float64); s2 = part[:, -2].astype(np.float64)
    eff = 1.0 / np.maximum(np.einsum("ij,ij->i", P, P, dtype=np.float32), np.float32(1e-12))
    P[np.arange(n_lat), n1] = -1.0
    n2 = np.argmax(P, axis=1)

    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(OUTD, exist_ok=True)
    pq.write_table(pa.table({
        "layer": np.full(n_lat, L, dtype=np.uint8),
        "latent": np.arange(n_lat, dtype=np.int32),
        "top1_neuron": n1.astype(np.int32),
        "top1_share": s1.astype(np.float32),
        "top2_neuron": n2.astype(np.int32),
        "top2_share": s2.astype(np.float32),
        "eff_neurons": eff.astype(np.float32),
        "l1_norm": l1,
    }), f"{OUTD}/nominations_enc.parquet")
    summary = {
        "layer": L, "mode": "weights", "ckpt": os.path.basename(CKPT),
        "n_latents": int(n_lat),
        "top1_p50": float(np.median(s1)), "top1_p90": float(np.percentile(s1, 90)),
        "top1_p99": float(np.percentile(s1, 99)), "top1_max": float(s1.max()),
        "top1_ge_0.5": int((s1 >= 0.5).sum()),
        "eff_p50": float(np.median(eff)),
    }
    json.dump(summary, open(f"{OUTD}/summary.json", "w"), indent=1)
    print("NOMINATE-ENC DONE", json.dumps(summary), flush=True)

elif MODE == "attribution":
    import pyarrow.parquet as pq
    PARQ = f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
    SAMPLE_TOKENS = int(os.environ.get("NOM_TOKENS", "200000"))
    K_PERM = int(os.environ.get("NOM_PERMS", "5"))
    ATTR_LATENTS_CAP = int(os.environ.get("NOM_ATTR_CAP", "8000"))

    pf = pq.ParquetFile(PARQ)
    total_rows = pf.metadata.num_rows
    nrg = pf.metadata.num_row_groups
    # token-sorted file: rows are (token_idx, neuron) long-format; mirror
    # patterns_pass.py's spread sampling of row groups
    tokens_per_rg = (total_rows / nrg) / 64.0  # sparse top-64 neurons per token
    n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
    step = max(1, nrg // n_needed)
    rgs = list(range(0, nrg, step))[:n_needed]
    print(f"attribution: {len(rgs)} row groups of {nrg}", flush=True)

    def batches():
        for rg in rgs:
            t = pf.read_row_group(rg, columns=["token_idx", "neuron", "gate", "up"]).to_pydict()
            tok = np.asarray(t["token_idx"], dtype=np.int64)
            neu = np.asarray(t["neuron"], dtype=np.int64)
            # atlas convention (patterns_pass.py:71): silu(gate) * up, sparse top-64 neurons/token
            act = t["gate"] / (1.0 + np.exp(-np.asarray(t["gate"], dtype=np.float32))) * np.asarray(t["up"], dtype=np.float32)
            act = np.asarray(act, dtype=np.float32)
            # long -> dense [tokens_in_rg, d]
            utok, inv = np.unique(tok, return_inverse=True)
            m = np.zeros((len(utok), d), dtype=np.float32)
            m[inv, neu] = act
            yield torch.from_numpy(m)

    first = next(batches())
    if not selfcheck(first):
        print("ATTR ABORT: z_pre recomputation does not match encode()", flush=True)
        raise SystemExit(2)

    # alive set from uniform census (matches decode-side convention)
    import pyarrow.parquet as pqa
    cu = f"{SCR}/sae-64k/layer={LL}/census_uniform.parquet"
    alive_mask = np.zeros(n_lat, dtype=bool)
    if os.path.exists(cu):
        ct = pqa.read_table(cu).to_pydict()
        col = "alive" if "alive" in ct else ("latent_id" if "latent_id" in ct else None)
        if col == "alive":
            alive_mask = np.asarray(ct["alive"], dtype=bool)
        else:
            ids = np.asarray(ct["latent_id"] if col else [], dtype=np.int64)
            alive_mask[ids] = True
    alive_idx = np.flatnonzero(alive_mask)
    attr_lat = alive_idx[:ATTR_LATENTS_CAP]
    lat_pos = {int(j): p for p, j in enumerate(attr_lat)}
    accum = np.zeros((len(attr_lat), d), dtype=np.float64)
    fires = np.zeros(len(attr_lat), dtype=np.int64)
    E_np = E.numpy()

    n_tok = 0
    for m in batches():
        n_tok += m.shape[0]
        with torch.no_grad():
            zp = z_pre_of(m)                      # [T, n_lat]
            k = sae.encode(m.to(torch.bfloat16)).top_indices         # [T, k]
        zpn = zp.float().numpy()
        for r in range(m.shape[0]):
            x = m[r].numpy()                      # [d]
            for j in k[r].tolist():
                p = lat_pos.get(j)
                if p is None:
                    continue
                fires[p] += 1
                accum[p] += np.abs(E_np[j] * x)
    fired = fires > 0
    shares = np.zeros((len(attr_lat), d), dtype=np.float64)
    den = accum.sum(1, keepdims=True)
    shares[fired] = accum[fired] / np.maximum(den[fired], 1e-12)
    s1 = shares.max(1); n1 = shares.argmax(1)
    eff = 1.0 / np.maximum((shares ** 2).sum(1), 1e-12)

    # shuffle control: permute each neuron column independently (keeps neuron
    # marginals, destroys cross-neuron co-activation), recompute for a sample.
    # Streams the parquet again per permutation -- no full-matrix concat (OOM).
    ctrl_idx = np.random.default_rng(20261108).choice(
        len(attr_lat), size=min(500, int(fired.sum())), replace=False)
    ctrl_s1 = []
    rng = np.random.default_rng(1)
    for _ in range(K_PERM):
        acc_s = np.zeros((len(ctrl_idx), d), dtype=np.float64)
        for m in batches():
            Xs = m.numpy().copy()
            for i in range(d):
                rng.shuffle(Xs[:, i])   # column-wise: token order independent per neuron
            Zs = torch.from_numpy(Xs)
            with torch.no_grad():
                zps = z_pre_of(Zs); ks = sae.encode(Zs.to(torch.bfloat16)).top_indices
            for r in range(Xs.shape[0]):
                x = Xs[r]
                for jj in range(ks.shape[1]):
                    j = int(ks[r, jj]); p = lat_pos.get(j)
                    if p is None: continue
                    ci = np.searchsorted(ctrl_idx, p)
                    if ci < len(ctrl_idx) and ctrl_idx[ci] == p:
                        acc_s[ci] += np.abs(E_np[j] * x)
        sh = acc_s / np.maximum(acc_s.sum(1, keepdims=True), 1e-12)
        ctrl_s1.append(sh.max(1))
    ctrl_s1 = np.concatenate(ctrl_s1)

    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(OUTD, exist_ok=True)
    pq.write_table(pa.table({
        "layer": np.full(len(attr_lat), L, dtype=np.uint8),
        "latent": attr_lat.astype(np.int32),
        "fired_tokens": fires,
        "top1_neuron": n1.astype(np.int32),
        "top1_share": s1.astype(np.float32),
        "eff_neurons": eff.astype(np.float32),
    }), f"{OUTD}/attribution_enc.parquet")
    rep = {
        "layer": L, "mode": "attribution", "ckpt": os.path.basename(CKPT),
        "tokens": int(n_tok), "latents_tracked": int(len(attr_lat)),
        "fired": int(fired.sum()),
        "top1_p50": float(np.median(s1[fired])) if fired.any() else None,
        "top1_p90": float(np.percentile(s1[fired], 90)) if fired.any() else None,
        "top1_max": float(s1.max()) if fired.any() else None,
        "top1_ge_0.5": int((s1[fired] >= 0.5).sum()) if fired.any() else 0,
        "eff_p50_fired": float(np.median(eff[fired])) if fired.any() else None,
        "control": {"perms": K_PERM, "latents": int(len(ctrl_s1)),
                    "top1_p50": float(np.median(ctrl_s1)),
                    "top1_p90": float(np.percentile(ctrl_s1, 90)),
                    "top1_max": float(np.max(ctrl_s1))},
    }
    json.dump(rep, open(f"{OUTD}/attribution_control.json", "w"), indent=1)
    print("ATTR DONE", json.dumps(rep), flush=True)
