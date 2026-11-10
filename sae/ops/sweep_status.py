import json, glob, os
from json import JSONDecoder
dec = JSONDecoder()
rows = []
for d in sorted(glob.glob("/data/scratch/apx186/marin32b/sae-64k/layer=*"), key=lambda x: int(x.split("layer=")[1])):
    L = int(d.split("layer=")[1])
    vlog = os.path.join(d, "val_log.jsonl")
    best, n = None, 0
    if os.path.exists(vlog):
        for line in open(vlog):
            try:
                e = json.loads(line); n += 1
                if best is None or e["val_fvu"] < best["val_fvu"]: best = e
            except Exception: pass
    st = {}
    sp = os.path.join(d, "stats.json")
    if os.path.exists(sp) and os.path.getsize(sp) > 10:
        try:
            st, _ = dec.raw_decode(open(sp, errors="ignore").read().lstrip())
        except Exception: pass
    su = {}
    up = os.path.join(d, "stats_uniform.json")
    if os.path.exists(up):
        try: su = json.load(open(up))
        except Exception: pass
    rows.append((L, best.get("step") if best else None, best.get("val_fvu") if best else None,
                 su.get("alive_latents", st.get("alive_latents")),
                 su.get("fvu_atlas_uniform")))
for r in rows:
    print(f"L{r[0]:02d} plateau_step={r[1]} val_fvu[val-split]={r[2]:.4f}" if r[2] else f"L{r[0]:02d} running", end="  ")
    if r[3]: print(f"alive={r[3]} fvu_atlas[uniform]={r[4]:.4f}")
    else: print()
