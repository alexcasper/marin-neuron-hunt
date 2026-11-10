#!/usr/bin/env python3
import json, glob, re
rows=[]
for f in sorted(glob.glob("/data/scratch/apx186/marin32b/sae-32k/layer=*/stats_12k.json")):
    L=int(re.search(r"layer=(\d+)",f).group(1))
    d=json.load(open(f))
    rows.append((L,d["fvu_sampled"],d["alive_latents"],d["max_feat_act"],d["freq_top100_share"]))
rows.sort()
print("layer\tfvu12k\talive\tmax_act\ttop100")
for r in rows: print(*r,sep="\t")
print(f"n={len(rows)}")
