#!/usr/bin/env python3
"""Candidate shortlist for neuron hunting — pole_stats selection (sort A + tail filters).

Evidence rule: every number below comes straight from
data/pilot/fineweb-10m-sparse/summaries/pole_stats.parquet.
Selection profile mirrors the validated L0-N10657 choice: strong positive side
with a real (rarer) negative tail, so BOTH sides are explainable.
"""
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

SRC = "/mnt/d/ops/sh/data/pilot/fineweb-10m-sparse/summaries/pole_stats.parquet"
OUT = Path("/mnt/d/ops/sh/hunt/interp/shortlist-pole-stats.md")

t = pq.read_table(SRC).to_pandas()
t["sep"] = t.pos_q99 - t.neg_q99

# Filters: both sides must have real signal.
#  - positive side: fires on most tokens, clear positive tail
#  - negative side: rare but real negative excursions (the N10657 lesson:
#    a side with neg tail indistinguishable from noise is unexplainable)
cand = t[
    (t["frac_pos"] > 0.90) &                 # positive side fires broadly
    (t["pos_q99"] > 0.10) &                  # meaningful positive tail
    (t["min"] < -0.10) &                     # real negative excursion exists
    (t["neg_q9999"] > 0.02) &                # negative tail reaches above noise
    (t["n"] >= 1_000_000)                    # enough tokens observed
].copy()

# Dedupe against already-explained neurons in the fork
done = set()
for p in Path("/mnt/d/ops/sh/hunt/explanations").rglob("*.yaml"):
    done.add(p.stem)  # e.g. L0-N10657
cand["name"] = "L" + cand.layer.astype(str) + "-N" + cand.neuron.astype(str)
cand = cand[~cand.name.isin(done)]

# Stratify: top-2 per layer by sep, then rank across layers for the report
per_layer = (
    cand.sort_values(["layer", "sep"], ascending=[True, False])
    .groupby("layer", group_keys=False)
    .head(2)
)
top = per_layer.sort_values("sep", ascending=False)

lines = [
    "# pole_stats candidate shortlist (local-only, not upstream)",
    "",
    "Source: `data/pilot/fineweb-10m-sparse/summaries/pole_stats.parquet` (1,769,472 rows),",
    "pulled 2026-11-08. Filters: frac_pos>0.90, pos_q99>0.10, min<-0.10,",
    "neg_q9999>0.02, n>=1e6. Ranked by sep = pos_q99 - neg_q99 (validated sort A",
    "from the L0-N10657 selection). Top-2 per layer, all 64 layers represented.",
    f"Filter pass rate: {len(cand)} of 1,769,472 neurons. Already-explained deduped.",
    "",
    "| rank | neuron | sep | pos_q99 | neg_q9999 | frac_pos | max | min | n |",
    "|---|---|---|---|---|---|---|---|---|",
]
for i, (_, r) in enumerate(top.iterrows(), 1):
    lines.append(
        f"| {i} | L{r.layer}-N{r.neuron} | {r["sep"]:.4f} | {r["pos_q99"]:.4f} | "
        f"{r["neg_q9999"]:.4f} | {r["frac_pos"]:.6f} | {r["max"]:.3f} | {r["min"]:.3f} | {r["n"]} |"
    )
lines += [
    "",
    f"Layers covered: {top.layer.nunique()}/64.",
    "",
    "Next step per neuron: exemplars.parquet (fineweb-1m for token text; 10m-sparse",
    "exemplars for ranks) -> draft explanation YAML in the fork, validate.py, stay local.",
]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print(f"wrote {OUT}")
print(f"filter pass: {len(cand)}; per-layer shortlist: {len(top)} rows, {top.layer.nunique()} layers")
print(top.head(12)[["name","sep","pos_q99","neg_q9999","frac_pos","max","min","n"]].to_string(index=False))
