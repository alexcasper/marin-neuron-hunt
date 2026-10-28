#!/usr/bin/env python3
"""Post-run stats + census for a trained atlas SAE.

Usage: sae_stats.py <ckpt_dir> <layer-tokensorted.parquet> [sample_tokens=200000] [census_prefix]
Writes stats JSON to stdout and, if census_prefix is given, a per-latent census
parquet (<prefix>.parquet): layer, latent_id, alive, max_act, fire_frac.
Portable version for the APOC venv (paths passed, nothing dusk-hardcoded).
"""
import sys, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sparsify import Sae
from train_sae import batches, load_scales_vec, D_MODEL_MLP  # noqa: E402

if __name__ == "__main__":
    ckpt, parquet = sys.argv[1], sys.argv[2]
    sample_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    census_prefix = sys.argv[4] if len(sys.argv) > 4 else None

    sae = Sae.load_from_disk(ckpt, device="cuda:0")
    sae.eval()
    n_latents = sae.encoder.weight.shape[0]
    # layer id comes from the checkpoint config if present, else arg-free default 0
    layer_id = 0
    try:
        cfg = json.load(open(f"{ckpt}/cfg.json"))
        # trainer stores the layer label; parse "L40" -> 40 when present
        import re
        m = re.search(r"L(\d+)", json.dumps(cfg))
        if m:
            layer_id = int(m.group(1))
    except Exception:
        pass
    scale_parquet = sys.argv[5] if len(sys.argv) > 5 else None
    scale_vec = None
    if scale_parquet:
        scale_vec = load_scales_vec(scale_parquet, layer_id, D_MODEL_MLP)

    seen = torch.zeros(n_latents, dtype=torch.bool, device="cuda:0")
    max_act = torch.zeros(n_latents, device="cuda:0")
    fire_count = torch.zeros(n_latents, dtype=torch.long, device="cuda:0")
    fvu_num, fvu_den, n_tok = 0.0, 0.0, 0
    with torch.no_grad():
        for x in batches(parquet, 1024, scale_vec):
            x = x.to("cuda:0").to(torch.bfloat16)
            out = sae(x)
            resid = x - out.sae_out
            fvu_num += (resid.float() ** 2).sum().item()
            fvu_den += (x.float() ** 2).sum().item()
            n_tok += x.shape[0]
            seen[out.latent_indices.unique()] = True
            m = torch.zeros(n_latents, device="cuda:0")
            m.scatter_reduce_(0, out.latent_indices.flatten(), out.latent_acts.flatten().float(), reduce="amax", include_self=True)
            max_act = torch.maximum(max_act, m)
            fire_count += torch.bincount(out.latent_indices.flatten().long(), minlength=n_latents)
            if n_tok >= sample_tokens:
                break

    alive = seen
    freq_all = fire_count.float() / max(n_tok, 1)
    freq_alive = freq_all[alive]
    decile_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    freq_deciles = freq_alive.quantile(torch.tensor(decile_qs, device=freq_alive.device)).tolist() if freq_alive.numel() else None
    stats = {
        "ckpt": ckpt,
        "layer": layer_id,
        "tokens_sampled": n_tok,
        "fvu_sampled": fvu_num / fvu_den,
        "alive_latents": int(alive.sum().item()),
        "alive_frac": round(alive.float().mean().item(), 4),
        "n_latents": n_latents,
        "max_feat_act": float(max_act.max().item()),
        "max_feat_idx": int(max_act.argmax().item()),
        "p99_feat_max_act": float(max_act[seen].quantile(0.99).item()) if seen.any() else None,
        "freq_deciles_alive": [round(d, 6) for d in freq_deciles] if freq_deciles else None,
        "freq_mean_alive": round(float(freq_alive.mean().item()), 6) if freq_alive.numel() else None,
        "freq_p99_alive": round(float(freq_alive.quantile(0.99).item()), 6) if freq_alive.numel() else None,
        "freq_max": round(float(freq_all.max().item()), 6),
        "freq_max_idx": int(freq_all.argmax().item()),
        "freq_top100_share": round(float(freq_alive.topk(min(100, freq_alive.numel())).values.sum().item() / max(freq_alive.sum().item(), 1)), 4) if freq_alive.numel() else None,
    }
    print(json.dumps(stats, indent=1))

    if census_prefix:
        table = pa.table({
            "layer": pa.array([layer_id] * n_latents, type=pa.int16()),
            "latent_id": pa.array(range(n_latents), type=pa.int32()),
            "alive": pa.array(seen.cpu().numpy(), type=pa.bool_()),
            "max_act": pa.array(max_act.cpu().numpy(), type=pa.float32()),
            "fire_frac": pa.array(freq_all.cpu().numpy(), type=pa.float32()),
        })
        pq.write_table(table, f"{census_prefix}.parquet")
        print(f"census parquet: {census_prefix}.parquet ({n_latents} rows)", flush=True)
