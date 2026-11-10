# sae/ — code that trained the marin-32b MLP SAEs

TopK sparse autoencoders over the MLP activations of
[marin-community/marin-32b-base](https://huggingface.co/marin-community/marin-32b-base)
(Qwen3-32B architecture: 64 layers, d_mlp 27,648, input `silu(gate)·up`, signed),
trained on the sparse tier of
[Station-house/marin-32b-neuron-atlas](https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas)
(`pilot/fineweb-10m-sparse`). The library is EleutherAI `eai-sparsify` (`Sae`, TopK).

This directory collects the training, evaluation, data-staging and job-spec code
used for the 32k, 12k-topup, 64k and 128k runs. It is a code archive. Weights
and per-layer results live on Hugging Face (see `publish/`).

## Layout

| path | what |
|---|---|
| `train/train_sae.py` | the trainer, current version (post-sweep fix, see below) |
| `train/train_sae_32k.py` | trainer as run for the 32k sweep (ops_sh `c91015e`) |
| `train/train_sae_64k.py` | trainer as run for the 64k sweep (ops_sh `cd4774f`) |
| `configs/` | `cfg.json` of the L00 32k and 64k finals, as saved by `eai-sparsify` |
| `eval/sae_stats.py` | per-layer stats and census (alive latents, max feature act, sampled fvu, per-latent parquet) |
| `eval/sae_stats_uniform.py` | census v2: uniform row-group sampling, input convention selectable |
| `eval/val32k.py`, `eval/val-table.py` | val-split fvu passes and the sweep comparison table |
| `data/apoc-stage-atlas.slurm`, `data/login_pull.py` | atlas pull from HF to scratch (`pilot/fineweb-10m-sparse` only), size check, and the per-layer `ORDER BY token_idx` re-sort (duckdb) that produces `layerNN-tokensorted.parquet` |
| `data/check_parquet.py`, `data/copy_atlas.slurm` | shard sanity check, scratch copy |
| `jobs/apocrita/*.slurm` | Slurm job specs (QMUL Apocrita, A100-40/80) for every training and census run |
| `jobs/dusk/*` | single-host ROCm runs (L00 pilot, topup chain, 64k rung staging) |
| `publish/*` | HF upload jobs, upload audit, model-card text. Tokens come from the job environment only |
| `ops/*` | sweep/probe status readers and the 12k topup table |
| `docs/METRICS.md` | **binding** fvu quoting rules and the input-convention table. Read before using any number |
| `docs/RUNPLAN.md`, `docs/APOC-array-spec.md`, `docs/SIZING-dusk-vs-apocrita.md` | run plan, array design, sizing |
| `docs/apocrita-MANIFEST.md5` | md5 of every script on Apocrita scratch at recovery time (2026-11-10) |

## Which code trained what (provenance)

Trainer versions are ops_sh lane commits. Job times come from `sacct`.

| family | ran (sacct) | trainer | evidence | input convention |
|---|---|---|---|---|
| 32k sweep, 64 layers, 6k steps (`apoc-sae-layer.slurm`) | 2026-10-28 16:17 → 10-29 01:02, 66/66 COMPLETED | `train_sae_32k.py` = `c91015e` | **verified**: byte-identical to the copy still on Apocrita (md5 `8f7a67a5`) | per-neuron **scaled** |
| 12k topup, 44 layers (`apoc-topup.slurm`) | 2026-10-29 | `4ccc89e` | inferred from commit date | scaled |
| L57 plateau (`apoc-l57-plateau.slurm`) | 2026-10-29 | `02d6b54` | inferred from commit date | see `docs/METRICS.md` |
| L63 retrains, 64k probe, 128k probe | 2026-11-04 onward | `cd4774f` | inferred from commit date | 64k/128k: verify the label convention |
| 64k sweep, 64 layers (`apoc-64k-sweep.slurm`) | 2026-11-06 03:15 → 11-07 13:49, 64/64 COMPLETED | `train_sae_64k.py` = `cd4774f` | inferred: last trainer commit before the sweep; the sweep needs the val/plateau/keep-grid flags added in `02d6b54`/`cd4774f`; the 11-07 fix landed 20:18, after the sweep ended | **RAW**: labels like `L00w64` fail the layer parse, so `scale_vec` was silently `None` |

The post-sweep fix (2026-11-07) makes the trainer refuse to run when the
per-neuron scale vector cannot be resolved, and adds `--layer` and `--raw-acts`.
It differs from `cd4774f` only in those lines. Width families use different
input conventions, so never compare fvu across them; `docs/METRICS.md` has the
rules.

## Known gaps

- **As-submitted batch scripts are gone** for every training array. Slurm keeps
  them only while a job is in memory, and `scontrol write batch_script` failed
  for all of them; only a later nomination array was recoverable. The `.slurm`
  files here are the tracked specs and match Apocrita's copies byte for byte.
  The exception is `apoc-64k-sweep.slurm`, which was edited on 2026-11-07
  20:18, after the sweep, and passes `--layer`. `cd4774f`'s argparse has no
  `--layer`, so the submitted version cannot have contained that line.
- The deployed `cd4774f` trainer was overwritten on Apocrita by the fix, so
  the 64k row is inferred, not md5-verified.
- `jobs/dusk/` scripts are the dawn-side copies. The canonical copies on the
  dusk host (`~/marin-sae/`) were unreachable when this snapshot was taken.

## Environment (Apocrita GPU jobs)

`apptainer exec --nv` of `bolmo_cu118.sif` with `PYTHONUSERBASE` pointing at a
user base that has `eai-sparsify` installed (GPU nodes have no egress, so no pip
at runtime). Each job stages its layer parquet to node-local `/tmp`. Some
earlier `.sh` drafts reference a `sae-venv` that never existed on scratch; the
`.slurm` files are the specs that ran.
