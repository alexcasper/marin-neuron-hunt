# Apocrita SAE sweep — HF-direct plan (sh-hr9), 2026-10-27

Alex directive 2026-10-27: NO dawn→APOC push. Apocrita batch jobs pull what
they need from HuggingFace directly. Dawn's 411.9 GB pilot copy stays put as
the verification reference.

## Groundwork already in place (survey/)

- `tree_api_totals.json` — full census: 1,062 files / 411.9 GB
- `filelist.json` — every path + size (+sha256 in the manifests)
- `pull_v3.sh` — proven curl-resolve pattern: public URLs, no token,
  resumable (`curl -sfL -C -`), size-checked per file, 5 retries

## Reachability check

- Login node: HF reachable — tree API 200 (0.16 s), resolve/CDN HEAD 200
  (0.10 s). Verified 2026-10-27 via ssh exp.
- Compute nodes: probe job handed to env_apoc (wW:p7) — 3-node CPU array,
  2 min, same two curls. Result pending. If compute nodes are
  egress-restricted → fallback: login-node staging job into durable
  (still zero bytes from dawn).

## Durable layout

- `<DURABLE>/marin-32b/atlas/` — full pilot tree, pulled by a staging job
  using `survey/filelist.json` + curl-resolve URLs; resumable; verify
  size (and sha256 for a sample + all summaries) against the per-tree
  MANIFESTs. 1,062 files → WATCH INODES before/after.
- `<DURABLE>/marin-32b/hf-cache/` — HF_HOME for marin-32b-base weights
  (~65 G) ONLY IF the harness validates against the model. TopK training
  on pre-extracted activations does NOT need it — default: skip.
- OPEN: the brief's `/data/SPCS-IOT/` does not exist on exp (not even
  `/data`). True durable root + creation/quota owned by env_apoc — ask,
  don't guess. (g4 precedent: /scratch/a.cline/g4, but scratch is not
  durable.)

## Array-job design (64 tasks, one per layer)

```
sbatch -p gpu -a 0-63 -t <budget> apoc-sae-layer.sh
```

Per task (layer L = SLURM_ARRAY_TASK_ID):
1. Node-local staging ONLY its slice from durable atlas/ into $SLURM_TMPDIR:
   - `atlas/pilot/fineweb-10m-sparse/sparse/layer=LL/sorted-00000.parquet` (~4 G)
   - `atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet` (5 M)
   - optional dense slice (fineweb-1m deep/resid docs) if the config wants it
   Never stage the whole 411 G on a node.
2. Token-resort if needed (duckdb, 10 GB mem) — or pre-resort once into
   durable during staging-job phase to save per-task time.
3. Train eai-sparsify TopK SparseCoder on the layer slice. Pilot config
   (dusk): 32k latents, k=32, bf16, AdamW 3e-4, batch 1024, fvu.
   H200 sizing: 64k latents trivially fits (80 G HBM); 6.1M-token pilot
   slice is well under an hour at H200 throughput; the 200k-step full runs
   are the real budget line — size -t from measured tok/s, gate order:
   deps → smoke on real shards → full run.
4. Checkpoints + logs → durable; save-every 1000; sha256-manifest the
   outputs so the sweep is verifiable.

Env requirement (env_apoc owns): python + torch (cu) + eai-sparsify
(same 1.3.3 API as the dusk run) in the NGC SIF or a venv.

## Beads

- sh-bs6: already CLOSED with the verified tally (1,054/1,054 parquet,
  byte-exact vs manifests + HF tree); dawn copy retained as verification
  reference per this directive. Noted; no reopen.
- sh-hr9: carries this design; training-location decision now settled by
  alex (Apocrita for the 64-layer sweep), dusk remains the pilot seat.
