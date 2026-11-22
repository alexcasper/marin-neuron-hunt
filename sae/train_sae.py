#!/usr/bin/env python3
"""TopK SAE training on the marin-32b-neuron-atlas sparse tier (dusk / 890M ROCm).

Reads token-sorted sparse-tier parquet (64 rows/token: token_idx, neuron, gate, up),
reconstructs dense per-token MLP activation vectors (silu(gate)*up, normalized by
the atlas per-neuron scale), and trains an eai-sparsify TopK SparseCoder.

Launch via ~/torch-rocm-env train_sae.py ...

Convention safety (METRICS.md rule 5, sh-kas 2026-11-22): the layer/scale resolution
HARD-FAILS on any ambiguity — no silent raw-acts fallback. The resolved convention is
written to <out-dir>/convention.json and stamped into every train-log record.
"""
import argparse, json, os, re, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch

from sae_conventions import (
    D_MODEL_MLP,
    parse_layer_id, load_scale_vec, batches, val_batches,
)

# Back-compat re-exports (sae_stats.py and other tools import these from train_sae)
load_scales_vec = load_scale_vec


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
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="linear LR warmup steps (0 = off); linear decay to 0 at --steps follows")
    p.add_argument("--no-lr-decay", action="store_true",
                   help="disable linear decay after warmup (constant lr after warmup)")
    p.add_argument("--auxk-alpha", type=float, default=0.03125,
                   help="AuxK dead-latent loss weight (0 = disabled; sparsify semantics)")
    p.add_argument("--dead-tokens", type=int, default=10_000_000,
                   help="a latent counts as dead when it has not fired in this many tokens")
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
    p.add_argument("--l2-norm-input", action="store_true", help="L2-normalize each input vector (kills scale heterogeneity)")
    p.add_argument("--seed", type=int, default=20261026)
    return p.parse_args()


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


def resolve_convention(args):
    """Layer id + per-neuron scale vector; HARD-FAILS on any ambiguity (METRICS.md rule 5).

    The 2026-11-07 incident: labels like 'L00w64' failed the old isdigit() parse and
    training silently proceeded on raw activations. Never again.
    """
    try:
        layer_id = parse_layer_id(args.layer_label)
    except ValueError as e:
        sys.exit(f"FATAL: {e} — refusing to train (silent raw-acts fallback caused the "
                 "2026-11-07 64k-sweep incident; see METRICS.md)")
    if not os.path.exists(args.scale_parquet):
        sys.exit(f"FATAL: scale parquet not found: {args.scale_parquet} — refusing to "
                 "train unscaled (METRICS.md rule 5)")
    scale_vec = load_scale_vec(args.scale_parquet, layer_id)
    n_scales = int((scale_vec > 1e-4).sum())
    if n_scales == 0:
        sys.exit(f"FATAL: no scales for layer {layer_id} in {args.scale_parquet} — "
                 "refusing to train unscaled (METRICS.md rule 5)")
    print(f"convention: layer {layer_id} scaled by per-neuron atlas scale "
          f"({n_scales}/{scale_vec.shape[0]} neurons) from {args.scale_parquet}", flush=True)
    return layer_id, scale_vec


def val_fvu(sae, parquet_path, val_rows, batch_tokens, scale_vec, dev, cap_tokens):
    num, den = 0.0, 0.0
    with torch.no_grad():
        for x in val_batches(parquet_path, val_rows, 1024, scale_vec, cap_tokens):
            x = torch.from_numpy(x).to(dev).to(torch.bfloat16)
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

    layer_id, scale_vec = resolve_convention(args)

    convention = {
        "layer_label": args.layer_label, "layer_id": layer_id,
        "scale_parquet": args.scale_parquet, "input_convention": "scaled",
        "n_latents": args.n_latents, "k": args.k,
        "auxk_alpha": args.auxk_alpha, "dead_tokens": args.dead_tokens,
        "warmup_steps": args.warmup_steps, "lr_decay": not args.no_lr_decay,
        "lr": args.lr, "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "convention.json"), "w") as f:
        json.dump(convention, f, indent=1)

    from sparsify import Sae
    from sparsify.config import SparseCoderConfig
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
    train_rows = None
    if args.val_rows > 0:
        _pf = pq.ParquetFile(args.layer_parquet)
        train_rows = _pf.metadata.num_rows - args.val_rows
        print(f"val holdout: {args.val_rows} rows; training stream capped at {train_rows} rows", flush=True)
    t0 = time.time()
    tokens_seen = 0
    step = args.start_step
    use_auxk = args.auxk_alpha > 0
    tokens_since_fired = torch.full((args.n_latents,), 1 << 40, device=dev, dtype=torch.long)
    data_iter = prefetch(lambda: batches(args.layer_parquet, args.batch_tokens, scale_vec, max_rows=train_rows))
    for step in range(args.start_step + 1, args.steps + 1):
        try:
            x = torch.from_numpy(next(data_iter)).to(dev, non_blocking=True)
        except StopIteration:
            print("data exhausted at step", step, flush=True)
            break
        # sh-usm: initialize b_dec to the first batch's input mean (skipped on warm-start,
        # which must preserve the loaded decoder bias)
        if step == args.start_step + 1 and not args.init_from:
            sae.b_dec.data.copy_(x.to(torch.bfloat16).mean(dim=0))
            print("b_dec initialized to first-batch input mean (sh-usm)", flush=True)
        # sh-f6z: linear warmup then linear decay to 0 at --steps
        if args.warmup_steps > 0 and step <= args.start_step + args.warmup_steps:
            lr_now = args.lr * step / max(1, args.start_step + args.warmup_steps)
        elif not args.no_lr_decay:
            span = max(1, args.steps - args.start_step - args.warmup_steps)
            lr_now = args.lr * max(0.0, (args.steps - step) / span)
        else:
            lr_now = args.lr
        for g in opt.param_groups:
            g["lr"] = lr_now
        x_in = x.to(torch.bfloat16)
        if getattr(args, "l2_norm_input", False):
            x_in = (x_in.float() / (x_in.float().norm(dim=1, keepdim=True) + 1e-8)).to(torch.bfloat16)
        # sh-074: latents that have not fired in --dead-tokens feed the AuxK loss
        dead_mask = (tokens_since_fired > args.dead_tokens) if use_auxk else None
        out = sae(x_in, dead_mask=dead_mask)
        x_hat = out.sae_out
        resid = x_in - x_hat
        fvu = (resid.float() ** 2).sum(dim=1).mean() / (x_in.float() ** 2).sum(dim=1).mean()
        loss = fvu
        if use_auxk:
            loss = loss + args.auxk_alpha * out.auxk_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # standard sparsify hygiene
        sae.remove_gradient_parallel_to_decoder_directions()
        opt.step()
        sae.set_decoder_norm_to_unit_norm()
        # sh-074: sliding dead-window bookkeeping
        if use_auxk:
            tokens_since_fired += x.shape[0]
            tokens_since_fired[out.latent_indices.flatten().long()] = 0
        tokens_seen += x.shape[0]
        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                active = (out.latent_acts > 0).float().sum(dim=1).mean().item()
                n_dead = int((tokens_since_fired > args.dead_tokens).sum().item()) if use_auxk else -1
            rec = {
                "step": step, "fvu": float(fvu.item()),
                "mse": float((resid ** 2).mean().item()),
                "avg_active": active, "tokens": tokens_seen,
                "tok_s": round(tokens_seen / (time.time() - t0), 1),
                "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "elapsed_min": round((time.time() - t0) / 60, 1),
                "lr": round(lr_now, 8),
                "auxk": float(out.auxk_loss.item()) if use_auxk else None,
                "n_dead": n_dead,
                "scaled_input": True, "layer_id": layer_id,
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
                import shutil
                for d in os.listdir(args.out_dir):
                    m = re.fullmatch(rf"sae_{args.layer_label}_step(\d+)", d)
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
