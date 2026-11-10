import json, glob, re
from json import JSONDecoder
dec = JSONDecoder()
rows = []
bad = []
for f in sorted(glob.glob("/data/scratch/apx186/marin32b/sae-32k/layer=*/stats_12k.json")):
    L = int(re.search(r"layer=(\d+)", f).group(1))
    txt = open(f, errors="ignore").read().lstrip()
    try:
        obj, _ = dec.raw_decode(txt)
        rows.append((L, obj["fvu_sampled"], obj["alive_latents"], obj["max_feat_act"], obj["freq_top100_share"]))
    except Exception as e:
        bad.append((L, str(e)[:60]))
rows.sort()
print("layer\tfvu12k\talive\tmax_act\ttop100")
for r in rows:
    print(f"{r[0]}\t{r[1]:.4f}\t{r[2]}\t{r[3]}\t{r[4]}")
fv = [r[1] for r in rows]
if fv:
    print(f"n={len(rows)} fvu12k mean={sum(fv)/len(fv):.4f} min={min(fv):.4f} max={max(fv):.4f}")
if bad:
    print("UNPARSEABLE:", bad)
