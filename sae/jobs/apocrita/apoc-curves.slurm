#!/bin/bash
#SBATCH -J marin-curves
#SBATCH -p computeshort
#SBATCH -A pilot_gpu
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/curves-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/curves-%j.err
# CURVE PACK (env_apoc adaptation): per-layer train-fvu curves (+ val where
# present), 64-layer overlay colored by 6k-census band, per-layer CSVs.
# matplotlib preinstalled in the qwen35 userbase (login-side, no pip on node).
set -uo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-32k
CURVES=$SCR/curves
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
mkdir -p "$CURVES"

apptainer exec --env PYTHONUSERBASE="$PYTHONUSERBASE" -B "$CURVES:$CURVES" "$SIF" \
  python3 - <<'EOF'
import json, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/data/scratch/apx186/marin32b/sae-32k"
CURVES = "/data/scratch/apx186/marin32b/curves"

def band_of(fvu):
    if fvu < 0.90: return "green"
    if fvu <= 1.0: return "amber"
    return "red"
BANDCOLOR = {"green": "#2ca02c", "amber": "#ff9f1c", "red": "#d62728"}

layers = sorted(glob.glob(f"{RUNS}/layer=*"), key=lambda d: int(re.search(r"\d+", os.path.basename(d)).group()))
sanity, malformed, curves_made = [], [], 0
overlay_rows = {}

for d in layers:
    nn = re.search(r"\d+", os.path.basename(d)).group().zfill(2)
    log = os.path.join(d, "train_log.jsonl")
    steps, fvu = [], []
    try:
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: rec = json.loads(line)
                except json.JSONDecodeError: continue
                if isinstance(rec, dict) and "step" in rec and "fvu" in rec:
                    steps.append(rec["step"]); fvu.append(rec["fvu"])
    except FileNotFoundError:
        malformed.append((nn, "no train_log.jsonl")); continue
    if not steps:
        malformed.append((nn, "empty/failed log")); continue

    with open(f"{CURVES}/layer={nn}.csv", "w") as f:
        f.write("step,train_fvu\n")
        for s, v in zip(steps, fvu): f.write(f"{s},{v:.6f}\n")

    vsteps, vfvu = [], []
    vlog = os.path.join(d, "val_log.jsonl")
    if os.path.exists(vlog):
        with open(vlog) as f:
            for line in f:
                try: rec = json.loads(line)
                except Exception: continue
                if isinstance(rec, dict) and "step" in rec and "val_fvu" in rec:
                    vsteps.append(rec["step"]); vfvu.append(rec["val_fvu"])

    endpoint = None
    sp = os.path.join(d, "stats.json")
    if os.path.exists(sp):
        try: endpoint = json.load(open(sp)).get("fvu_sampled")
        except Exception: pass
    if endpoint is not None and fvu:
        sanity.append((nn, fvu[-1], endpoint, abs(fvu[-1] - endpoint)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, fvu, lw=1.2, label="train fvu")
    if vsteps:
        ax.plot(vsteps, vfvu, lw=1.2, ls="--", label="val fvu (30k-token holdout)")
    if endpoint is not None:
        ax.axhline(endpoint, color="gray", lw=0.8, ls=":", label=f"stats.json fvu_sampled={endpoint:.3f}")
    ax.set_title(f"layer {nn} - TopK 32k SAE fvu vs step")
    ax.set_xlabel("step"); ax.set_ylabel("fvu"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{CURVES}/layer={nn}.png", dpi=110); plt.close(fig)
    curves_made += 1

    band = band_of(endpoint) if endpoint is not None else "amber"
    overlay_rows[nn] = (band, steps, fvu)

fig, ax = plt.subplots(figsize=(12, 7))
for nn, (b, steps, fvu) in sorted(overlay_rows.items(), key=lambda kv: int(kv[0])):
    ax.plot(steps, fvu, lw=0.8, alpha=0.75, color=BANDCOLOR[b])
ax.set_title("marin-32b TopK-32k SAE sweep - 64 train-fvu curves\n(green fvu<0.90 - amber 0.90-1.0 - red >1.0, colored by 6k-census band)")
ax.set_xlabel("step"); ax.set_ylabel("train fvu"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CURVES}/overlay-64.png", dpi=130); plt.close(fig)

print(f"curves: {curves_made} png+csv pairs; overlay written")
print("malformed/empty:", malformed if malformed else "none")
big = [s for s in sanity if s[3] > 0.08]
print("sanity |curve_end - stats_fvu_sampled| > 0.08:", big if big else "none (all within tolerance)")
EOF
echo "CURVES COMPLETE"
