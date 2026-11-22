#!/usr/bin/env python3
"""Exact convention rescale for TopK SAE checkpoints (sh-1ls, alex decision 2026-11-22).

Converts a checkpoint trained on scale-normalized activations (32k family) to the RAW
convention (64k sweep family) — or the inverse — as an EXACT reparameterization:

    scaled model:  x' = x ⊘ s;  pre = (x' − b_dec)W_encᵀ + b_enc;  x̂' = z·W_dec + b_dec
    raw model:     pre = (x − b̃_dec)W̃_encᵀ + b_enc  with W̃_enc = W_enc ⊘ s (per input col)
                   x̂   = z·W̃_dec + b̃_dec          with W̃_dec = W_dec ⊙ s, b̃_dec = s ⊙ b_dec

since (x − s⊙b_dec)⊘s = x' − b_dec and s⊙(z·W_dec) = z·(W_dec ⊙ s). Same TopK selection,
bit-comparable outputs (float assoc aside). The transform needs b_dec = s⊙b_dec on BOTH
sides, which is consistent because SparseCoder.decode adds the same b_dec that encode
subtracts.

CAVEAT (do not skip): after rescaling, W_dec rows are NO LONGER unit-norm. The function
is unchanged, but never run set_decoder_norm_to_unit_norm() on a rescaled checkpoint —
that would change the model. Any downstream norm-of-decoder stats must expect row norms
≈ the per-neuron scales.

Torch-free: reads/writes safetensors via numpy (memmap), so this runs on CPU-only hosts.

Usage:
  python3 rescale_sae.py --ckpt CKPT_DIR --scale-parquet SCALES.parquet --layer-id 0 \
      --out-dir OUT_DIR [--direction scaled-to-raw|raw-to-scaled] [--verify] \
      [--eval-parquet P.parquet --eval-tokens 256]
"""
import argparse, json, os, struct, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae_conventions import parse_layer_id, load_scale_vec, batches

DT_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64, "I64": np.int64}
NP_DT = {v: k for k, v in DT_NP.items()}


def read_safetensors(path):
    """{name: (np.ndarray f32-view-of-storage, dtype_str, shape)} via memmap."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    meta = header.pop("__metadata__", None)
    for name, h in header.items():
        a, b = h["data_offsets"]
        m = np.memmap(path, dtype=DT_NP[h["dtype"]], mode="r", offset=base + a, shape=tuple(h["shape"]))
        out[name] = m
    return out, meta


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def get_f32(tensors, name):
    m = tensors[name]
    return bf16_to_f32(m) if m.dtype == np.uint16 else np.asarray(m, dtype=np.float32)


def write_safetensors(path, arrays, metadata=None):
    """arrays: {name: (np.ndarray, dtype_str)} — contiguous f-type arrays only."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    payload = []
    for name, (arr, dt) in arrays.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + nbytes]}
        payload.append(arr)
        offset += nbytes
    hjson = json.dumps(header, separators=(",", ":")).encode("utf8")
    pad = (8 - (len(hjson) % 8)) % 8  # safetensors pads header to 8-byte alignment
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for arr in payload:
            f.write(arr.tobytes())


def topk_forward(x, W_enc, b_enc, W_dec, b_dec, k, chunk=8192):
    """Numpy TopK forward; returns (sae_out, top_idx). x: [B, d] f32, contiguous rows."""
    xc = x - b_dec
    B = x.shape[0]
    best_v = np.full((B, k), -np.inf, np.float32)
    best_i = np.zeros((B, k), np.int64)
    for s0 in range(0, W_enc.shape[0], chunk):
        W = W_enc[s0:s0 + chunk]
        pre = xc @ W.T + b_enc[s0:s0 + chunk]
        cat_v = np.concatenate([best_v, pre], 1)
        cat_i = np.concatenate([best_i, np.broadcast_to(np.arange(s0, s0 + W.shape[0]), pre.shape)], 1)
        top = np.argpartition(-cat_v, k - 1, axis=1)[:, :k]
        best_v = np.take_along_axis(cat_v, top, 1)
        best_i = np.take_along_axis(cat_i, top, 1)
    best_v = np.maximum(best_v, 0)  # decode masks zeros; see audit SMOKE3b for equivalence
    out = np.tile(b_dec, (B, 1))
    for t in range(B):
        m = best_v[t] > 0
        out[t] += best_v[t, m] @ W_dec[best_i[t, m]]
    return out, best_i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint dir (sae.safetensors + cfg.json)")
    p.add_argument("--scale-parquet", required=True)
    p.add_argument("--layer-id", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--direction", choices=["scaled-to-raw", "raw-to-scaled"], default="scaled-to-raw")
    p.add_argument("--verify", action="store_true", help="random-input equivalence check")
    p.add_argument("--eval-parquet", default="", help="sparse-tier parquet for a real-data rescale-eval")
    p.add_argument("--eval-tokens", type=int, default=256)
    p.add_argument("--batch-tokens", type=int, default=64)
    a = p.parse_args()

    layer_id = a.layer_id
    s = load_scale_vec(a.scale_parquet, layer_id)
    assert (s > 1e-4).sum() > 0, f"no scales for layer {layer_id}"

    tensors, _ = read_safetensors(os.path.join(a.ckpt, "sae.safetensors"))
    cfg = json.load(open(os.path.join(a.ckpt, "cfg.json")))
    k = int(cfg.get("k", 32))
    print(f"loaded {a.ckpt}: d_in={cfg.get('d_in')}, k={k}, tensors=({', '.join(tensors)})")

    W_enc = get_f32(tensors, "encoder.weight")   # [n_lat, d_in]
    W_dec = get_f32(tensors, "W_dec")            # [n_lat, d_in]
    b_enc = get_f32(tensors, "encoder.bias")     # [n_lat]
    b_dec = get_f32(tensors, "b_dec")            # [d_in]
    n_lat, d_in = W_enc.shape
    assert W_dec.shape == W_enc.shape and b_enc.shape == (n_lat,) and b_dec.shape == (d_in,)

    if a.direction == "scaled-to-raw":
        W_enc_t, W_dec_t, b_dec_t = W_enc / s, W_dec * s, b_dec * s
    else:
        W_enc_t, W_dec_t, b_dec_t = W_enc * s, W_dec / s, b_dec / s

    verified = None
    if a.verify:
        rng = np.random.default_rng(7)
        x_raw = (rng.normal(size=(32, d_in)) * rng.uniform(0.5, 2.0, size=(1, d_in))).astype(np.float32)
        if a.direction == "scaled-to-raw":
            # reference: original model on scaled input, output lifted back to raw space
            out_a, idx_a = topk_forward(x_raw / s, W_enc, b_enc, W_dec, b_dec, k)
            out_a = out_a * s
            out_b, idx_b = topk_forward(x_raw, W_enc_t, b_enc, W_dec_t, b_dec_t, k)
        else:
            out_a, idx_a = topk_forward(x_raw * s, W_enc, b_enc, W_dec, b_dec, k)
            out_a = out_a / s
            out_b, idx_b = topk_forward(x_raw, W_enc_t, b_enc, W_dec_t, b_dec_t, k)
        close = np.allclose(out_a, out_b, rtol=1e-3, atol=1e-3 * max(1.0, float(np.abs(out_a).max())))
        idx_same = all(set(idx_a[t].tolist()) == set(idx_b[t].tolist()) for t in range(len(x_raw)))
        verified = {"allclose": bool(close), "topk_sets_equal": bool(idx_same),
                    "max_abs_diff": float(np.abs(out_a - out_b).max())}
        print("VERIFY:", json.dumps(verified))
        if not (close and idx_same):
            sys.exit("FATAL: rescale equivalence check FAILED — output not written")

    os.makedirs(a.out_dir, exist_ok=True)
    arrays = {
        "encoder.weight": (W_enc_t.astype(np.float32), "F32"),
        "W_dec": (W_dec_t.astype(np.float32), "F32"),
        "encoder.bias": (b_enc.astype(np.float32), "F32"),
        "b_dec": (b_dec_t.astype(np.float32), "F32"),
    }
    out_ckpt = os.path.join(a.out_dir, "sae.safetensors")
    write_safetensors(out_ckpt, arrays, metadata={
        "rescale": a.direction, "source_ckpt": a.ckpt,
        "layer_id": str(layer_id),
        "note": "EXACT reparameterization; decoder rows are NOT unit-norm after this — "
                "never re-run set_decoder_norm_to_unit_norm on a rescaled checkpoint"})
    cfg_out = dict(cfg)
    cfg_out["rescale"] = {"direction": a.direction, "layer_id": layer_id,
                          "scale_parquet": a.scale_parquet,
                          "input_convention": "raw" if a.direction == "scaled-to-raw" else "scaled",
                          "decoder_unit_norm": False, "verified": verified}
    with open(os.path.join(a.out_dir, "cfg.json"), "w") as f:
        json.dump(cfg_out, f, indent=1)
    print("wrote", out_ckpt, "and cfg.json")

    if a.eval_parquet:
        # Encode-side FVU on real tokens, in the convention the (transformed) model now speaks.
        n_tok = 0
        num = den = 0.0
        for xb in batches(a.eval_parquet, a.batch_tokens, None, d_model=d_in):
            # raw-convention targets: batches(., None) returns raw acts
            x = xb[:a.eval_tokens - n_tok]
            if a.direction == "scaled-to-raw":
                out, _ = topk_forward(x, W_enc_t, b_enc, W_dec_t, b_dec_t, k)
            else:
                out, _ = topk_forward(x / s, W_enc, b_enc, W_dec, b_dec, k)
                out = out * s
            resid = x - out
            num += (resid.astype(np.float64) ** 2).sum()
            den += (x.astype(np.float64) ** 2).sum()
            n_tok += x.shape[0]
            if n_tok >= a.eval_tokens:
                break
        fvu = num / den if den else float("nan")
        rec = {"layer_id": layer_id, "eval_parquet": a.eval_parquet, "tokens": n_tok,
               "convention": "raw" if a.direction == "scaled-to-raw" else "scaled",
               "fvu": float(fvu), "k": k, "n_latents": int(n_lat),
               "ckpt": a.ckpt, "rescaled": a.direction}
        print("RESCALE-EVAL:", json.dumps(rec))
        with open(os.path.join(a.out_dir, "rescale_eval.json"), "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
