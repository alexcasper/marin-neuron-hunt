# Apocrita SAE sweep — HF-direct plan (sh-hr9), re-anchored 2026-10-27

**Cluster model (alex correction):** Explorer (ssh exp, durable
`/projects/ComputationalPhilosophyLab/g4/`) and Apocrita (ssh apoc, durable
`/data/SPCS-IOT/`) are SEPARATE clusters. This sweep targets **APOCITRA**.
All cluster ops route through dispatcher panes: env_apoc = wW:p7 (Apocrita),
env_exp = wW:p6 (Explorer). No direct probes from agent shells.

**No dawn→APOC push.** Apocrita batch jobs pull what they need from
HuggingFace directly. Dawn's 411.9 GB pilot copy stays as the verification
reference (sh-bs6 closed). Groundwork: `survey/filelist.json` (every path,
1,062 files / 411.9 GB) + `pull_v3.sh` curl-resolve pattern (public URLs, no
token, resumable `curl -sfL -C -`, size-checked).

## Binding answers from env_apoc (perc, 2026-10-27)

1. **HF reach**: APOC login node YES (pulls/pips proven); CPU compute nodes
   YES — probe job 33544347 on ddy33/42/51: api tree 200 (~0.15 s), CDN
   resolve HEAD 200 (~0.19 s). **GPU nodes (sbg/ddg) NO egress** (DNS fails,
   documented). ⇒ atlas pulls run on login/compute/computeshort ONLY; GPU
   array tasks read staged data exclusively.
2. **Durable root**: `/data/SPCS-IOT/marin-32b/` CREATED + write-verified
   (setgid + ACLs, apx186). Fileset 110 T / 35 T free, 550 M inodes /
   148 M free (authoritative; df inside /data shows an unrelated 1.1 T
   gpfsWork view — ignore). Our 412 G / 1,062 files ≈ 1.2 % of free bytes;
   inodes negligible. Scratch staging: `/data/scratch/apx186/marin32b/`
   (1.1 T quota, **65-day purge — staging only, never the only copy**).
3. **GPU slots**: apx186 queue EMPTY. Proposal adopted: `sbatch -a 0-63%8`
   on `gpu:nvidia_a100-pcie-40gb:1` (sbg5+sbg19 pool) — with the qwen
   rerun at %4 worst case = 12 concurrent single-GPU asks, no collision.
   a100_80 available if VRAM demands it.
4. **eai-sparsify env caveat** (g4 lane): `SPARSIFY_DISABLE_TRITON=1`
   needed on torch<2.3 SIFs; triton decoder path breaks on stale libcuda.
   Use the **bolmo_cu118.sif userbase (torch 2.7.1 cu126)** — avoids both.
   eai-sparsify must be pre-installed (with `--no-deps` + missing deps, as
   on dusk) during a login/compute session — GPU nodes cannot pip.

## Durable layout

- `/data/SPCS-IOT/marin-32b/atlas/` — full pilot tree pulled by staging
  job(s) on compute/computeshort from filelist.json + curl-resolve; size-
  check every file; sha256-verify vs per-tree MANIFESTs. 1,062 files.
- `/data/SPCS-IOT/marin-32b/atlas-resorted/` — the 64 sparse layer files
  re-sorted by token_idx (duckdb), built once during staging so array
  tasks never pay resort cost.
- `/data/SPCS-IOT/marin-32b/runs/` — checkpoints + logs + sha256 manifest
  of sweep outputs.
- `hf-cache/` — HF_HOME for model weights (~65 G) ONLY if the harness
  validates against the model; TopK on pre-extracted activations does
  NOT. Default skip.

## Jobs

1. `apoc-stage-atlas.sh` (compute, 10 d cap, expect ~2-4 h): pull + verify
   + pre-resort. Inode/df -i check before and after.
2. `apoc-sae-layer.sh` (gpu, `-a 0-63%8`, A100-40): stage layer slice
   (~4 G) + neuron_scale from durable → $SLURM_TMPDIR, train, checkpoints
   (save-every 1000) → durable runs/ + manifest. Smoke gate first:
   `-a 0-1`, 200 steps, verify fvu falls + tok/s, then release the array.

Training config (from the dusk pilot): TopK, 32k latents, k=32, bf16,
AdamW 3e-4, batch 1024, fvu; 6,000 steps ≈ 6.1M tokens ≈ fvu 0.99→0.71 on
an 890M — A100 runs the same in well under an hour. 64k-latent runs fit
A100-40 (or move to a100_80). Gate order: deps → smoke on real shards →
full array.

## Beads

- sh-bs6: CLOSED (verified tally); dawn copy retained as reference.
- sh-hr9: carries this design; location decision settled (Apocrita),
  dusk remains the pilot seat.
