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
    p.add_argument("--init-from", default="", help="checkpoint dir to warm-start from")
    p.add_argument("--start-step", type=int, default=0, help="absolute step this run continues from")
    p.add_argument("--save-at", default="", help="comma list of absolute steps to checkpoint (overrides save-every)")
    p.add_argument("--val-rows", type=int, default=0, help="hold out this many trailing rows of the layer parquet as validation (0=off)")
    p.add_argument("--val-tokens", type=int, default=30000, help="cap on tokens per validation pass")
    p.add_argument("--val-every", type=int, default=3000, help="absolute-step interval for validation passes")
    p.add_argument("--plateau-window", type=int, default=9000, help="compare val fvu vs this many steps back")
    p.add_argument("--plateau-thresh", type=float, default=0.02, help="relative improvement below this = plateaued (stop)")
    p.add_argument("--keep-grid", type=int, default=9000, help="retention: keep step ckpts on this grid + latest, cull rest")
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


def batches(parquet_path, batch_tokens, scales_by_neuron=None, max_rows=None):
    """Yield dense fp32 CPU batches [B, 27648]; 64 rows per token.
    max_rows caps the training stream (rows beyond it, e.g. a val holdout, are skipped)."""
    pf = pq.ParquetFile(parquet_path)
    rows_per_batch = batch_tokens * 64
    emitted = 0
    for rb in pf.iter_batches(batch_size=rows_per_batch, columns=["token_idx", "neuron", "gate", "up"]):
        n = rb.num_rows
        if max_rows is not None and emitted + n > max_rows:
            rb = rb.slice(0, max_rows - emitted)
            n = rb.num_rows
        if n == 0:
            return
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


def prefetch(source_factory, depth=4):
    """Cycle a batch-source factory forever (multi-epoch training)."""
    import queue, threading
    q = queue.Queue(maxsize=depth)

    def worker():
        epoch = 0
        while True:
            for b in source_factory():
                q.put(b)
            epoch += 1
            print(f"EPOCH {epoch} complete — recycling data stream", flush=True)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        yield q.get()


def _tail_batches(pf, start_row, batch_tokens):
    """Yield pyarrow RecordBatches covering rows [start_row, end), aligned to 64."""
    import pyarrow as pa
    md = pf.metadata
    spans = []
    pos = 0
    for g in range(md.num_row_groups):
        spans.append((pos, md.row_group(g).num_rows))
        pos += md.row_group(g).num_rows
    buf = None
    for gi, (gstart, gsize) in enumerate(spans):
        gend = gstart + gsize
        if gend <= start_row:
            continue
        take_start = max(start_row, gstart)
        tbl = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"])
        tbl = tbl.slice(take_start - gstart, gend - take_start)
        if buf is not None:
            tbl = pa.concat_tables([buf, tbl])
        nb = tbl.num_rows // 64 * 64
        if nb == 0:
            buf = tbl
            continue
        for i in range(0, nb, batch_tokens * 64):
            yield tbl.slice(i, min(batch_tokens * 64, nb - i)).combine_chunks()
        buf = tbl.slice(nb) if tbl.num_rows > nb else None


def val_batches(parquet_path, val_rows, batch_tokens, scales_by_neuron=None, cap_tokens=None):
    """Validation batches from the trailing val_rows of a token-sorted parquet."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    start = max(0, total - val_rows)
    emitted = 0
    for rb in _tail_batches(pf, start, batch_tokens):
        n = rb.num_rows
        if n % 64 != 0:
            continue
        tok = rb.column("token_idx").to_numpy()
        neu = rb.column("neuron").to_numpy()
        gate = rb.column("gate").to_numpy().astype(np.float32)
        up = rb.column("up").to_numpy().astype(np.float32)
        B = n // 64
        x = np.zeros((B, D_MODEL_MLP), dtype=np.float32)
        act = gate / (1.0 + np.exp(-gate)) * up
        if scales_by_neuron is not None:
            act = act / scales_by_neuron[neu]
        flat_b = np.repeat(np.arange(B, dtype=np.int64), 64)
        x[flat_b, neu] = act
        yield torch.from_numpy(x)
        emitted += B
        if cap_tokens and emitted >= cap_tokens:
            return


def val_fvu(sae, parquet_path, val_rows, batch_tokens, scale_vec, dev, cap_tokens):
    num, den = 0.0, 0.0
    with torch.no_grad():
        for x in val_batches(parquet_path, val_rows, 1024, scale_vec, cap_tokens):
            x = x.to(dev).to(torch.bfloat16)
            out = sae(x)
            resid = x.to(torch.bfloat16) - out.sae_out
            num += (resid.float() ** 2).sum().item()
            den += (x.float() ** 2).sum().item()
    return num / den if den else float("nan")


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
    print(f"sae params: {n_params/1e6:.0f}M", flush=True)

    save_at = set(int(s) for s in args.save_at.split(",") if s.strip())

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    val_hist = []          # [(step, val_fvu)]
    plateaued_at = None
    train_rows = None
    if args.val_rows > 0:
        _pf = pq.ParquetFile(args.layer_parquet)
        train_rows = _pf.metadata.num_rows - args.val_rows
        print(f"val holdout: {args.val_rows} rows; training stream capped at {train_rows} rows", flush=True)
    t0 = time.time()
    tokens_seen = 0
    step = args.start_step
    run = True
    data_iter = prefetch(lambda: batches(args.layer_parquet, args.batch_tokens, scale_vec, max_rows=train_rows))
    for step in range(args.start_step + 1, args.steps + 1):
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
        if args.save_at:
            if step in save_at:
                ck = os.path.join(args.out_dir, f"sae_{args.layer_label}_step{step}")
                sae.save_to_disk(ck)
                print("checkpoint", ck, flush=True)
        elif args.save_every and step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"sae_{args.layer_label}_step{step}")
            sae.save_to_disk(ck)
            print("checkpoint", ck, flush=True)
            # retention: keep ckpts on the keep-grid + the latest; cull the rest
            if args.keep_grid:
                import shutil, re as _re
                for d in os.listdir(args.out_dir):
                    m = _re.fullmatch(rf"sae_{args.layer_label}_step(\d+)", d)
                    if m and int(m.group(1)) != step and int(m.group(1)) % args.keep_grid != 0:
                        shutil.rmtree(os.path.join(args.out_dir, d))
                        print("culled", d, flush=True)
        # validation + plateau detection
        if args.val_rows > 0 and step % args.val_every == 0:
            v = val_fvu(sae, args.layer_parquet, args.val_rows, 1024, scale_vec, dev, args.val_tokens)
            val_hist.append((step, v))
            print(json.dumps({"VAL": True, "step": step, "val_fvu": round(v, 6),
                              "train_fvu": float(fvu.item())}), flush=True)
            with open(os.path.join(args.out_dir, "val_log.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "val_fvu": v}) + "\n")
            prior = [h for h in val_hist if h[0] <= step - args.plateau_window]
            if prior:
                old = max(prior, key=lambda h: h[0])[1]
                rel = (old - v) / old if old > 0 else 0.0
                print(json.dumps({"PLATEAU_CHECK": True, "step": step, "vs_step": prior[-1][0],
                                  "old_val_fvu": old, "rel_improvement": round(rel, 4),
                                  "threshold": args.plateau_thresh}), flush=True)
                if rel < args.plateau_thresh:
                    plateaued_at = step
                    ck = os.path.join(args.out_dir, f"sae_{args.layer_label}_plateau_step{step}")
                    sae.save_to_disk(ck)
                    print(json.dumps({"PLATEAU": True, "step": step, "val_fvu": round(v, 6),
                                      "rel_improvement": round(rel, 4), "ckpt": ck}), flush=True)
                    print("saved", ck, flush=True)
                    break
    if args.save_at:
        last = max(save_at) if save_at else args.steps
        ckpt = os.path.join(args.out_dir, f"sae_{args.layer_label}_step{last}")
        if not os.path.exists(ckpt):  # data exhausted before last save-at step
            sae.save_to_disk(ckpt)
            print("saved (exhausted)", ckpt, flush=True)
    else:
        last_step = step if 'step' in dir() else args.steps
        suffix = f"_s{last_step}" if (args.val_rows > 0 or args.start_step > 0) else ""
        ckpt = os.path.join(args.out_dir, f"sae_{args.layer_label}_final{suffix}")
        sae.save_to_disk(ckpt)
        print("saved", ckpt, flush=True)


if __name__ == "__main__":
    main()
