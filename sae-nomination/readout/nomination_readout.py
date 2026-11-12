#!/usr/bin/env python3
"""Per-layer encoder-attribution nomination readout, all 64 layers (alex 2026-11-08).

For each layer: latents with top1_share >= 0.5 from nominate_enc/attribution_enc.parquet,
joined to pole_stats for saturation flags (N3872 lesson):
  saturated = frac_pos > 0.99  OR  pos_q99 > 100x layer median pos_q99
NOTE: pole_stats carries no per-neuron fire-rate column (the ">20% of sample
tokens" signal alex cited is not directly available without re-scanning the
sparse tier); frac_pos + magnitude outlier are the cheap proxies and caught
N3872 (frac_pos=1.000) in the L57 characterization.
Writes $SCR/sae-64k/attribution-nominations-all64.md
"""
import glob
import json
import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

S = "/data/scratch/apx186/marin32b"
SUM = f"{S}/atlas/pilot/fineweb-10m-sparse/summaries"
OUT = f"{S}/sae-64k/attribution-nominations-all64.md"

# pole_stats once, filter per layer in memory
_tbl = pq.read_table(f"{SUM}/pole_stats.parquet").to_pydict()
ps_layers = _tbl["layer"]
PS = {}
for i in range(len(ps_layers)):
    PS.setdefault(ps_layers[i], {})[_tbl["neuron"][i]] = i

# layer medians of pos_q99 (magnitude-outlier baseline)
LAYER_MED = {}
for lay, m in PS.items():
    vals = [_tbl["pos_q99"][i] for i in m.values()]
    LAYER_MED[lay] = float(np.median(vals))

lines = []
lines.append("# Encoder-attribution nominations, all 64 layers (sae-64k)\n")
lines.append("Latents with encoder-attribution top1_share >= 0.5 per layer (200,704-token "
             "atlas sample, K=5 shuffle control per run). SAT flag = neuron saturated per "
             "pole_stats: frac_pos > 0.99 OR pos_q99 > 100x layer median (N3872-style "
             "artifact filter; frac_pos=1.000 caught N3872 at L57). Note: pole_stats has no "
             "per-neuron fire-rate column, so the >20%-of-tokens signal is approximated by "
             "these proxies.\n")
tot_noms = tot_sat = 0
missing = []
for L in range(64):
    f = f"{S}/sae-64k/layer={L:02d}/nominate_enc/attribution_enc.parquet"
    if not os.path.exists(f):
        missing.append(L)
        continue
    t = pq.read_table(f).to_pydict()
    rows = [dict(latent=t["latent"][i], neuron=t["top1_neuron"][i], s1=t["top1_share"][i],
                 eff=t["eff_neurons"][i], fired=t["fired_tokens"][i])
            for i in range(len(t["latent"])) if t["top1_share"][i] >= 0.5]
    rows.sort(key=lambda r: -r["s1"])
    med = LAYER_MED.get(L, 0.03)
    n_sat = 0
    lines.append(f"## layer={L:02d} — {len(rows)} nomination(s)\n")
    if rows:
        lines.append("| latent | neuron | top1_share | eff_neurons | fired | frac_pos | pos_q99 | SAT |")
        lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        i = PS.get(L, {}).get(r["neuron"])
        if i is None:
            fp = pq99 = float("nan"); sat = False
        else:
            fp = _tbl["frac_pos"][i]; pq99 = _tbl["pos_q99"][i]
            sat = fp > 0.99 or pq99 > 100 * med
        n_sat += sat
        lines.append(f"| {r['latent']} | N{r['neuron']} | {r['s1']:.3f} | {r['eff']:.1f} "
                     f"| {r['fired']} | {fp:.3f} | {pq99:.3f} | {'SAT' if sat else ''} |")
    lines.append("")
    tot_noms += len(rows); tot_sat += n_sat
lines.append(f"\n## Totals: {tot_noms} nominations across 64 layers, {tot_sat} flagged SAT; "
             f"layers missing attribution runs: {missing or 'none'}\n")
open(OUT, "w").write("\n".join(lines))
print(f"READOUT DONE: {tot_noms} noms, {tot_sat} SAT, missing={missing} -> {OUT}", flush=True)
