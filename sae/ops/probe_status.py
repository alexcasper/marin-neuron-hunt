import json, glob, os
for d in sorted(glob.glob("/data/scratch/apx186/marin32b/64k-probe/layer=*")):
    L = d.split("layer=")[1]
    vlog = os.path.join(d, "val_log.jsonl")
    entries = []
    if os.path.exists(vlog):
        for line in open(vlog):
            try: entries.append(json.loads(line))
            except Exception: pass
    latest = entries[-1] if entries else None
    plat = os.path.exists(os.path.join(d, [x for x in os.listdir(d) if "plateau" in x][0])) if any("plateau" in x for x in os.listdir(d)) else False
    print(f"L{L}: val_entries={len(entries)} latest={latest} plateau_ckpt={plat}")
