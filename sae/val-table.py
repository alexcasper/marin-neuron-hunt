#!/usr/bin/env python3
"""Build the 64k-sweep val-at-plateau table vs the 32k baselines.

Reads a sweep root containing layer=NN/ dirs, each with:
  - val_log.jsonl   {"step": N, "val_fvu": f} lines (written by train_sae.py)
  - train_log.jsonl (optional) — PLATEAU marker gives the plateau step/fvu

Baselines come from results-sweep-32k.csv (census-sample fvu — biased high for
eval: first-200k tokens are train-covered, the L57 lesson). Any rows in
results-val32k.csv (val-holdout 32k passes, probe layers) take precedence.

Usage: val-table.py /path/to/sae-64k [--csv results-sweep-32k.csv] [--val32k results-val32k.csv]
Emits markdown to stdout.
"""
import argparse
import csv
import glob
import json
import os
import re


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def plateau_from_train_log(path):
    if not os.path.exists(path):
        return None
    for e in load_jsonl(path):
        if e.get("PLATEAU"):
            return e["step"], e["val_fvu"]
    return None


def best_val(path, plateau_step=None):
    """val at plateau step if given, else min val_fvu seen."""
    hist = load_jsonl(path)
    if not hist:
        return None, None
    if plateau_step is not None:
        at = [h for h in hist if h["step"] == plateau_step]
        if at:
            return plateau_step, at[0]["val_fvu"]
    best = min(hist, key=lambda h: h["val_fvu"])
    return best["step"], best["val_fvu"]


def load_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["layer"])] = row
            except (KeyError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "results-sweep-32k.csv"))
    ap.add_argument("--val32k", default=os.path.join(os.path.dirname(__file__), "results-val32k.csv"))
    args = ap.parse_args()

    base32 = load_csv(args.csv)          # census-sample (biased) baseline
    val32 = load_csv(args.val32k)        # val-holdout baseline, if it exists

    rows = []
    for ld in sorted(glob.glob(os.path.join(args.root, "layer=*"))):
        m = re.search(r"layer=(\d+)", ld)
        if not m:
            continue
        layer = int(m.group(1))
        vlog = os.path.join(ld, "val_log.jsonl")
        if not os.path.exists(vlog):
            continue
        plat = plateau_from_train_log(os.path.join(ld, "train_log.jsonl"))
        pstep, pfvu = (None, None)
        if plat:
            pstep, pfvu = plat
        step, fvu = best_val(vlog, pstep)
        if fvu is None:
            continue
        b32 = val32.get(layer, {}).get("val_fvu") or val32.get(layer, {}).get("fvu_val")
        if b32 is None:
            b32 = base32.get(layer, {}).get("fvu_sampled")
            bsrc = "census(bias)"
        else:
            bsrc = "val-holdout"
        delta = round(float(fvu) - float(b32), 4) if b32 else None
        alive = base32.get(layer, {}).get("alive_latents", "")
        rows.append((layer, step, fvu, pstep, pfvu, b32, bsrc, delta, alive,
                     "partial" if plat is None and pstep is None else "plateau"))

    print("| layer | 64k val@plateau (step) | min val (step) | 32k baseline ({}) | Δ | alive(32k) | status |".format(
        "val-holdout where present, else census-biased"))
    print("|---|---|---|---|---|---|---|")
    for layer, step, fvu, pstep, pfvu, b32, bsrc, delta, alive, status in rows:
        plat_s = f"{pfvu} ({pstep})" if pstep else "—"
        b_s = f"{b32} [{bsrc}]" if b32 else "—"
        d_s = f"{delta:+.4f}" if delta is not None else "—"
        print(f"| {layer} | {plat_s} | {fvu} ({step}) | {b_s} | {d_s} | {alive} | {status} |")
    print(f"\n{len(rows)} layers with val logs. NOTE: census-sample 32k fvu is biased "
          f"(first-200k tokens train-covered); val-holdout comparisons only where results-val32k.csv has the layer.")


if __name__ == "__main__":
    main()
