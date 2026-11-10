#!/bin/bash
#SBATCH -J sweep-final
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/sweepfinal-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/sweepfinal-%j.err
# Final publish pass for the COMPLETE 64k sweep: per-layer census files
# (uniform + prefix + stats_uniform), grid PNGs to sae-64k/ root, README
# provenance. Finals already on HF (passes 1-3); this job only adds missing
# pieces and verifies. Plateau ckpts are NOT included: culled on scratch
# after cullcheck proved them byte-identical to finals (20+40 IDENTICAL, 0 DIFF).
set -uo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser

apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 - "$SCR" <<'EOF' || { echo "SWEEPFINAL_FAILED"; exit 5; }
import os, sys, json, time, glob
from huggingface_hub import HfApi
SCR = sys.argv[1]
tok = open(f"{SCR}/.hftok").read().strip()
api = HfApi(token=tok)
REPO = "alxcspr/stationhouse"
print("auth as:", api.whoami().get("name"), flush=True)

# 1) per-layer census files (uniform redo artifacts; sweep finals already up)
CEN = ["census_uniform.parquet", "census_prefix.parquet", "stats_uniform.json"]
tot_files = tot_bytes = 0
for LL in range(64):
    d = f"{SCR}/sae-64k/layer={LL:02d}"
    files = [f for f in CEN if os.path.exists(os.path.join(d, f))]
    if not files: print(f"WARN layer {LL:02d}: no census files yet", flush=True); continue
    nb = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    for attempt in (1, 2):
        try:
            api.upload_folder(folder_path=d, path_in_repo=f"sae-64k/layer={LL:02d}",
                              repo_id=REPO, repo_type="model", allow_patterns=CEN,
                              commit_message=f"64k sweep L{LL:02d}: uniform+prefix census")
            print(f"ok census L{LL:02d} ({len(files)} files)", flush=True)
            tot_files += len(files); tot_bytes += nb
            break
        except Exception as e:
            if attempt == 2: print(f"FAIL census L{LL:02d}: {e}", flush=True)
            else: time.sleep(60)

# 2) grid PNGs -> sae-64k/ root
import tempfile, shutil
with tempfile.TemporaryDirectory() as td:
    n = 0
    for g in glob.glob(f"{SCR}/sae-64k/grid_64k_*.png"):
        shutil.copy(g, td); n += 1
    if n:
        nb = sum(os.path.getsize(os.path.join(td, f)) for f in os.listdir(td))
        api.upload_folder(folder_path=td, path_in_repo="sae-64k",
                          repo_id=REPO, repo_type="model",
                          commit_message="64k sweep: 64-layer grid PNGs")
        print(f"ok grids ({n} files)", flush=True); tot_files += n; tot_bytes += nb

# 3) README provenance (idempotent)
MARK = "<!-- sweep64k-full-2026-11-07 -->"
try:
    p = api.hf_hub_download(REPO, "README.md", repo_type="model", token=tok, local_dir="/tmp/hfr2")
    cur = open(p).read()
except Exception:
    cur = "---\nlicense: apache-2.0\n---\n# stationhouse\n"
if MARK not in cur:
    cur += f"""

{MARK}
## sae-64k — full 64-layer sweep (canonical)

Marin-32B neuron-atlas TopK SAE, width 65536, k=32, all 64 layers, trained
on the sbg5 A100 lane (job 35453930, 2026-11-06/07; ~44-75 min/layer,
plateau-early stopping, best-val checkpoints). Per layer:
sae_LNNw64_final_s* (canonical finals; the byte-identical plateau copies
were sha256-verified identical and culled before upload), stats.json,
val_log.jsonl (val-split holdout, trailing 1.92M rows), train_log.jsonl,
sha256.txt, census_uniform.parquet + stats_uniform.json (uniform
row-group spread, fvu labeled fvu_atlas_uniform) and census_prefix.parquet
(original corpus-prefix census, retained for provenance). fvus from
different distributions are never compared directly — see the standing
methodology note in the ops_sh pad. Layers 45/57/63 dirs also hold the
earlier width-probe files (feature-preview-64k.*); the sweep finals are
canonical. grid_64k_*.png at sae-64k/ root.
"""
    ci = api.upload_file(path_or_fileobj=cur.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="model",
                         commit_message="README: full 64k sweep provenance")
    print(f"ok readme commit={ci.oid[:10]}", flush=True)
else:
    print("readme already has sweep section", flush=True)

# 4) verification via tree API: all 64 layer dirs, recurse 6 sample ckpt dirs
import urllib.request
def tree(path):
    req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}/tree/main/{path}?recursive=true")
    req.add_header("Authorization", f"Bearer {tok}")
    return json.load(urllib.request.urlopen(req))

bad = []
sample = [0, 11, 23, 38, 49, 63]
hf_files = hf_bytes = 0
for LL in range(64):
    rp = f"sae-64k/layer={LL:02d}"
    try:
        fs = [f for f in tree(rp.replace("=", "%3D")) if f["type"] == "file"]
        hf_files += len(fs); hf_bytes += sum(f.get("size", 0) for f in fs)
        if LL in sample:
            # compare sae.safetensors bytes vs disk per ckpt dir
            for ckdir in glob.glob(f"{SCR}/sae-64k/layer={LL:02d}/sae_*"):
                ckname = os.path.basename(ckdir)
                disk = os.path.getsize(os.path.join(ckdir, "sae.safetensors"))
                m = [f for f in fs if f["path"].endswith(f"{ckname}/sae.safetensors")]
                if not m: bad.append(f"L{LL:02d}/{ckname}: MISSING on HF")
                elif m[0]["size"] != disk: bad.append(f"L{LL:02d}/{ckname}: hf {m[0]['size']} != disk {disk}")
    except Exception as e:
        bad.append(f"L{LL:02d}: tree error {e}")
print(f"VERIFY sae-64k: hf_files={hf_files} hf_GiB={hf_bytes/2**30:.1f} sampled_mismatches={len(bad)}", flush=True)
for b in bad[:10]: print("  BAD:", b, flush=True)
print(f"SWEEPFINAL COMPLETE uploaded_files={tot_files} uploaded_bytes={tot_bytes} verify_bad={len(bad)}", flush=True)
EOF
