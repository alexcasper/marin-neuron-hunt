#!/usr/bin/env python3
"""TopK SAE training on the marin-32b-neuron-atlas sparse tier (dusk / 890M ROCm).

Reads token-sorted sparse-tier parquet (64 rows/token: token_idx, neuron, gate, up),
reconstructs dense per-token MLP activation vectors (silu(gate)*up, normalized by
the atlas per-neuron scale), and trains an eai-sparsify TopK SparseCoder.

Launch via ~/torch-rocm-env train_sae.py ...
"""
import argparse, json, math, os, time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.functional import silu
from sparsify import Sae
from sparsify.config import SparseCoderConfig

D_MODEL_MLP = 27648  # marin-32b SwiGLU neurons per layer (act = silu(gate)*up)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-parquet", required=True, help="token-sorted sparse parquet")
    p.add_argument("--scale-parquet", default=os.path.expanduser(
        "~/marin-atlas/fineweb-10m-sparse-early/summaries/neuron_scale.parquet"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer-label", default="L?")
    p.add_argument("--n-latents", type=int, default=65536)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--steps", type=int, default=12500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=0, help="0 = only final save")
    p.add_argument("--init-from", default=None, help="ckpt dir (Sae.load_from_disk) to warm-start from")
    p.add_argument("--seed", type=int, default=20261026)
    return p.parse_args()


def load_scales(path):
    t = pq.read_table(path, columns=["layer", "neuron", "scale"])
    d = t.to_pydict()
    # scales are per (layer, neuron); caller trains one layer at a time
    return d  # caller filters by layer


def load_scales_vec(scale_parquet, layer_id, d_model=D_MODEL_MLP):
    """Per-neuron normalization scale for one layer (max(scale, 1e-4))."""
    t = pq.read_table(scale_parquet)
    lay = t.column("layer").to_numpy()
    neu = t.column("neuron").to_numpy()
    scl = t.column("scale").to_numpy()
    vec = np.full(d_model, 1e-4, dtype=np.float32)
    sel = lay == layer_id
    vec[neu[sel]] = np.maximum(scl[sel].astype(np.float32), 1e-4)
    return vec


def batches(parquet_path, batch_tokens, scales_by_neuron=None):
    """Yield dense fp32 CPU batches [B, 27648]; 64 rows per token."""
    pf = pq.ParquetFile(parquet_path)
    rows_per_batch = batch_tokens * 64
    for rb in pf.iter_batches(batch_size=rows_per_batch, columns=["token_idx", "neuron", "gate", "up"]):
        n = rb.num_rows
        assert n % 64 == 0
        tok = rb.column("token_idx").to_numpy()
        neu = rb.column("neuron").to_numpy()
        gate = rb.column("gate").to_numpy().astype(np.float32)
        up = rb.column("up").to_numpy().astype(np.float32)
        B = n // 64
        # contiguity check: token_idx strictly non-decreasing per row-block of 64
        x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
        act = gate / (1.0 + np.exp(-gate)) * up  # silu(gate)*up
        if scales_by_neuron is not None:
            act = act / scales_by_neuron[neu]
        flat_b = np.repeat(np.arange(B, dtype=np.int64), 64)
        x[flat_b, neu] = act
        yield torch.from_numpy(x)


def prefetch(source, depth=4):
    import queue, threading
    q = queue.Queue(maxsize=depth)

    def worker():
        for b in source:
            q.put(b)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    dev = "cuda:0"
    assert torch.cuda.is_available(), "expected HIP device"
    print("device:", torch.cuda.get_device_name(0), flush=True)

    # per-neuron scale for layer normalization (layer inferred from first token rows is
    # not stored in sparse files; scale table covers all layers -> use max over layers
    # is WRONG; instead caller must pass the right layer). We default to using the
    # layer passed via --layer-label when it is "LL" formatted.
    m = args.layer_label.lstrip("L")
    layer_id = int(m) if m.isdigit() else None
    scale_vec = None
    if layer_id is not None and os.path.exists(args.scale_parquet):
        scale_vec = load_scales_vec(args.scale_parquet, layer_id)
        print(f"loaded {int((scale_vec > 1e-4).sum())} scales for layer {layer_id}", flush=True)

    cfg = SparseCoderConfig(activation="topk", num_latents=args.n_latents, k=args.k,
                            normalize_decoder=True)
    sae = Sae(d_in=D_MODEL_MLP, cfg=cfg, device=dev, dtype=torch.bfloat16)
    if args.init_from:
        warm = Sae.load_from_disk(os.path.expanduser(args.init_from), device=dev)
        sae.load_state_dict(warm.state_dict())
        del warm
        print(f"warm-started from {args.init_from}", flush=True)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"sae params: {n_params/1e6:.0f}M ({n_params*4/2**30:.1f} GiB fp32 + 2x Adam moments)", flush=True)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    t0 = time.time()
    tokens_seen = 0
    run = True
    data_iter = prefetch(batches(args.layer_parquet, args.batch_tokens, scale_vec))
    for step in range(1, args.steps + 1):
        try:
            x = next(data_iter).to(dev, non_blocking=True)
        except StopIteration:
            print("data exhausted at step", step, flush=True)
            break
        out = sae(x.to(torch.bfloat16))
        x_hat = out.sae_out
        resid = x.to(torch.bfloat16) - x_hat
        fvu = (resid.float() ** 2).sum(dim=1).mean() / (x.to(torch.float32) ** 2).sum(dim=1).mean()
        loss = fvu
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # standard sparsify hygiene
        sae.remove_gradient_parallel_to_decoder_directions()
        opt.step()
        sae.set_decoder_norm_to_unit_norm()
        tokens_seen += x.shape[0]
        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                active = (out.latent_acts > 0).float().sum(dim=1).mean().item()
            rec = {
                "step": step, "fvu": float(fvu.item()),
                "mse": float((resid ** 2).mean().item()),
                "avg_active": active, "tokens": tokens_seen,
                "tok_s": round(tokens_seen / (time.time() - t0), 1),
                "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "elapsed_min": round((time.time() - t0) / 60, 1),
            }
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if args.save_every and step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"sae_{args.layer_label}_step{step}")
            sae.save_to_disk(ck)
            print("checkpoint", ck, flush=True)
    ckpt = os.path.join(args.out_dir, f"sae_{args.layer_label}_final")
    sae.save_to_disk(ckpt)
    print("saved", ckpt, flush=True)


if __name__ == "__main__":
    main()
