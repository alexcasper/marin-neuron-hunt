#!/usr/bin/env python3
"""Encoder-side neuron nomination, v2: device-aware + vectorized (ops_sh, 2026-11-09).

Same question and outputs as nominate_enc_pass.py (v1); see its docstring for
the method. What changed:

1. GPU. Runs on NOM_DEVICE (default cuda if available). On CUDA asserts an
   sm_80+ card (bf16 encode; a generic gres can land a V100).

2. Vectorized attribution. v1 looped per token x per selected latent in Python
   (accum[p] += |E[j] * x|), the real bottleneck (8h+ on CPU, no completions).
   Because E[j] is fixed per latent, sum_r |E[j] * x_r| == |E[j]| * sum_r |x_r|,
   so the accumulation is one matmul per chunk:
       S += F^T @ |X|      F[r, p] = 1 if tracked latent p is in row r's TopK
       accum = |E_tracked| * S        (fp64 accumulation, as in v1)
   Equivalent to v1 up to fp rounding (v1 rounded each fp32 product before the
   fp64 sum; v2 is the exact product of fp32 inputs).

3. Shuffle-control FIX (methodology correction, changes results vs v1):
   v1 drew ctrl_idx with rng.choice(...) (unsorted) and then looked latents up
   with np.searchsorted(ctrl_idx, p), which requires a sorted array, so most
   control latents never accumulated anything. Their share rows stayed 0, which
   produced the "permuted p50 0.000" reported for L57. v2 sorts ctrl_idx, draws
   it from latents that actually FIRED on real data, and reports control stats
   over rows with nonzero shuffled mass (n_zero reported separately).
   The permutation itself moves to torch (seeded; column-wise, within each row
   group, as v1), so the RNG stream differs from v1.

4. Dead work removed: v1 computed z_pre for every batch and never used it.
   z_pre is now computed only for the first-batch SEL-CHECK self-check
   (unchanged gate).

Outputs go to $SCR/sae-64k/layer=NN/$NOM_OUTSUB (default nominate_enc_v2) so
v1 outputs (L02, L57, any CPU-array finishers) are never overwritten.
Filenames are unchanged: attribution_enc.parquet + attribution_control.json.
"""
import json, os, glob, math, time
import numpy as np
import torch

MODE = os.environ.get("NOM_MODE", "attribution")
L = int(os.environ["NOM_LAYER"]); LL = f"{L:02d}"
SCR = os.environ.get("NOM_SCR", "/data/scratch/apx186/marin32b")
OUTD = f"{SCR}/sae-64k/layer={LL}/{os.environ.get('NOM_OUTSUB', 'nominate_enc_v2')}"
DEV = torch.device(os.environ.get("NOM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
CHUNK = int(os.environ.get("NOM_CHUNK", "4096"))        # rows per device chunk
COLBLK = int(os.environ.get("NOM_COLBLK", "4096"))      # columns per shuffle block
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# core math (pure functions; unit-tested against v1's loops in test_nominate_v2.py)
# ---------------------------------------------------------------------------
def accumulate_chunk(xc, top_idx, pos_of, S, fires):
    """xc [T,d] float on device, top_idx [T,K] long, pos_of [n_lat] long (-1 =
    untracked). Adds F^T @ |xc| into S [n_track,d] fp64 and F.sum(0) into fires."""
    p = pos_of[top_idx]                                   # [T,K]
    valid = p >= 0
    if not bool(valid.any()):
        return
    T = xc.shape[0]
    rows = torch.arange(T, device=xc.device).unsqueeze(1).expand_as(p)[valid]
    cols = p[valid]
    F = torch.zeros(T, S.shape[0], dtype=torch.float64, device=xc.device)
    F.index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float64), accumulate=True)
    S.addmm_(F.T, xc.abs().double())
    fires += F.sum(0).long()


def shares_from(Eabs, S):
    """Per-latent neuron shares from accum = |E| * S; returns (s1, n1, eff, mass)."""
    acc = Eabs.double() * S
    mass = acc.sum(1)
    sh = acc / mass.clamp_min(1e-12).unsqueeze(1)
    s1, n1 = sh.max(1)
    eff = 1.0 / (sh * sh).sum(1).clamp_min(1e-12)
    return s1, n1, eff, mass


def column_shuffle(X, gen, colblk=COLBLK):
    """Independent permutation of each column over rows (keeps per-neuron
    marginals, destroys cross-neuron co-activation)."""
    out = torch.empty_like(X)
    T, d = X.shape
    for c0 in range(0, d, colblk):
        c1 = min(d, c0 + colblk)
        idx = torch.rand(T, c1 - c0, generator=gen, device=X.device).argsort(0)
        out[:, c0:c1] = X[:, c0:c1].gather(0, idx)
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sparsify import Sae

    if DEV.type == "cuda":
        name = torch.cuda.get_device_name(DEV)
        cap = torch.cuda.get_device_capability(DEV)
        log(f"device {name} sm_{cap[0]}{cap[1]}")
        assert cap[0] >= 8, f"need sm_80+ for bf16 encode, got {name}"
    else:
        log("device cpu (no CUDA) -- slow path")

    CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
    assert os.path.exists(os.path.join(CKPT, "cfg.json")), CKPT
    sae = Sae.load_from_disk(CKPT, device=str(DEV)); sae.eval()

    E_raw = sae.encoder.weight.detach().float()
    E = E_raw if E_raw.shape[0] == sae.W_dec.shape[0] else E_raw.T   # [n_lat, d]
    n_lat, d = E.shape
    assert d == sae.W_dec.shape[1], (E.shape, sae.W_dec.shape)

    def z_pre_of(x):
        xb = x.to(torch.bfloat16)
        return ((xb - sae.b_dec.detach()) @ sae.encoder.weight.detach().T) + sae.encoder.bias.detach()

    def topk_of(x):
        with torch.no_grad():
            return sae.encode(x.to(torch.bfloat16)).top_indices

    def selfcheck(x):
        with torch.no_grad():
            codes = sae.encode(x.to(torch.bfloat16))
            zp = z_pre_of(x)
        k = codes.top_indices.shape[1]
        sel = torch.relu(zp).topk(k, dim=1).indices
        if torch.equal(torch.sort(sel, 1).values, torch.sort(codes.top_indices, 1).values):
            log("SEL-CHECK exact: recompute matches encode()")
            return True
        agree = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / float(k)
                              for a, b in zip(sel.cpu(), codes.top_indices.cpu())])
        log(f"SEL-CHECK soft: mean topk overlap {agree.mean():.4f}")
        return agree.mean() >= 0.99   # same gate as v1 (bf16 tie swaps cap ~0.995)

    if MODE != "attribution":
        raise SystemExit(f"v2 implements NOM_MODE=attribution only (weights mode: use v1, ~1 min CPU); got {MODE}")

    import pyarrow as pa, pyarrow.parquet as pq
    PARQ = os.environ.get("NOM_PARQ", f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet")
    SAMPLE_TOKENS = int(os.environ.get("NOM_TOKENS", "200000"))
    K_PERM = int(os.environ.get("NOM_PERMS", "5"))
    ATTR_LATENTS_CAP = int(os.environ.get("NOM_ATTR_CAP", "8000"))

    pf = pq.ParquetFile(PARQ)
    total_rows, nrg = pf.metadata.num_rows, pf.metadata.num_row_groups
    tokens_per_rg = (total_rows / nrg) / 64.0
    n_needed = max(1, math.ceil(SAMPLE_TOKENS / tokens_per_rg))
    step = max(1, nrg // n_needed)
    rgs = list(range(0, nrg, step))[:n_needed]                 # identical sample to v1
    log(f"attribution: {len(rgs)} row groups of {nrg} from {PARQ}")

    def batches():
        for rg in rgs:
            t = pf.read_row_group(rg, columns=["token_idx", "neuron", "gate", "up"]).to_pydict()
            tok = np.asarray(t["token_idx"], dtype=np.int64)
            neu = np.asarray(t["neuron"], dtype=np.int64)
            gate = np.asarray(t["gate"], dtype=np.float32)
            act = (gate / (1.0 + np.exp(-gate)) * np.asarray(t["up"], dtype=np.float32)).astype(np.float32)
            utok, inv = np.unique(tok, return_inverse=True)
            m = np.zeros((len(utok), d), dtype=np.float32)
            m[inv, neu] = act
            yield torch.from_numpy(m).to(DEV)

    def chunks(X):
        for r0 in range(0, X.shape[0], CHUNK):
            yield X[r0:r0 + CHUNK]

    first = next(batches())
    if not selfcheck(first[:CHUNK]):
        log("ATTR ABORT: z_pre recomputation does not match encode()")
        raise SystemExit(2)
    del first

    # alive set from uniform census (same convention as v1)
    cu = f"{SCR}/sae-64k/layer={LL}/census_uniform.parquet"
    alive_mask = np.zeros(n_lat, dtype=bool)
    if os.path.exists(cu):
        ct = pq.read_table(cu).to_pydict()
        if "alive" in ct:
            alive_mask = np.asarray(ct["alive"], dtype=bool)
        elif "latent_id" in ct:
            alive_mask[np.asarray(ct["latent_id"], dtype=np.int64)] = True
    attr_lat = np.flatnonzero(alive_mask)[:ATTR_LATENTS_CAP]
    n_track = len(attr_lat)
    log(f"tracking {n_track} alive latents (cap {ATTR_LATENTS_CAP})")
    assert n_track > 0, f"no alive latents from {cu}"

    lat_t = torch.as_tensor(attr_lat, device=DEV, dtype=torch.long)
    pos_of = torch.full((n_lat,), -1, dtype=torch.long, device=DEV)
    pos_of[lat_t] = torch.arange(n_track, device=DEV)
    Eabs = E[lat_t].abs()                                       # [n_track, d] fp32
    del E, E_raw

    S = torch.zeros(n_track, d, dtype=torch.float64, device=DEV)
    fires = torch.zeros(n_track, dtype=torch.long, device=DEV)
    n_tok = 0
    for X in batches():
        n_tok += X.shape[0]
        for xc in chunks(X):
            accumulate_chunk(xc, topk_of(xc), pos_of, S, fires)
    log(f"real pass done: {n_tok} tokens")
    s1_t, n1_t, eff_t, _ = shares_from(Eabs, S)
    fires_np = fires.cpu().numpy()
    fired = fires_np > 0
    s1 = np.where(fired, s1_t.cpu().numpy(), 0.0)
    n1 = n1_t.cpu().numpy()
    eff = eff_t.cpu().numpy()
    del S

    # shuffle control (FIXED: sorted, drawn from fired latents)
    fired_pos = np.flatnonzero(fired)
    ctrl_idx = np.sort(np.random.default_rng(20261108).choice(
        fired_pos, size=min(500, len(fired_pos)), replace=False))
    ctrl_lat = torch.as_tensor(attr_lat[ctrl_idx], device=DEV, dtype=torch.long)
    pos_c = torch.full((n_lat,), -1, dtype=torch.long, device=DEV)
    pos_c[ctrl_lat] = torch.arange(len(ctrl_idx), device=DEV)
    Eabs_c = Eabs[torch.as_tensor(ctrl_idx, device=DEV)]
    gen = torch.Generator(device=DEV); gen.manual_seed(1)
    ctrl_s1, ctrl_zero = [], 0
    for kp in range(K_PERM):
        S_c = torch.zeros(len(ctrl_idx), d, dtype=torch.float64, device=DEV)
        f_c = torch.zeros(len(ctrl_idx), dtype=torch.long, device=DEV)
        for X in batches():
            Xs = column_shuffle(X, gen)            # whole row group, as v1
            for xc in chunks(Xs):
                accumulate_chunk(xc, topk_of(xc), pos_c, S_c, f_c)
        cs1, _, _, mass = shares_from(Eabs_c, S_c)
        nz = (mass > 0).cpu().numpy()
        ctrl_zero += int((~nz).sum())
        ctrl_s1.append(cs1.cpu().numpy()[nz])
        log(f"perm {kp + 1}/{K_PERM} done ({int(nz.sum())}/{len(nz)} ctrl latents fired)")
    ctrl_s1 = np.concatenate(ctrl_s1) if ctrl_s1 else np.zeros(0)

    os.makedirs(OUTD, exist_ok=True)
    pq.write_table(pa.table({
        "layer": np.full(n_track, L, dtype=np.uint8),
        "latent": attr_lat.astype(np.int32),
        "fired_tokens": fires_np,
        "top1_neuron": n1.astype(np.int32),
        "top1_share": s1.astype(np.float32),
        "eff_neurons": eff.astype(np.float32),
    }), f"{OUTD}/attribution_enc.parquet")
    q = lambda a, f: (float(f(a)) if len(a) else None)
    rep = {
        "layer": L, "mode": "attribution", "version": "v2", "device": str(DEV),
        "ckpt": os.path.basename(CKPT),
        "tokens": int(n_tok), "latents_tracked": int(n_track), "fired": int(fired.sum()),
        "top1_p50": q(s1[fired], np.median),
        "top1_p90": q(s1[fired], lambda a: np.percentile(a, 90)),
        "top1_max": q(s1[fired], np.max),
        "top1_ge_0.5": int((s1[fired] >= 0.5).sum()),
        "eff_p50_fired": q(eff[fired], np.median),
        "control": {"perms": K_PERM, "latents_sampled": int(len(ctrl_idx)),
                    "rows": int(len(ctrl_s1)), "rows_zero_mass": ctrl_zero,
                    "top1_p50": q(ctrl_s1, np.median),
                    "top1_p90": q(ctrl_s1, lambda a: np.percentile(a, 90)),
                    "top1_max": q(ctrl_s1, np.max)},
        "wall_s": round(time.time() - T0, 1),
    }
    json.dump(rep, open(f"{OUTD}/attribution_control.json", "w"), indent=1)
    log("ATTR DONE", json.dumps(rep))
