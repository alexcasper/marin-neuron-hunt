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
| `train/train_sae.py` | the trainer (TopK `Sae`, AdamW, fvu loss, val holdout, plateau stop, checkpoint retention) |
| `eval/sae_stats.py` | per-layer stats and census (alive latents, max feature act, sampled fvu, per-latent parquet) |
| `eval/sae_stats_uniform.py` | census v2: uniform row-group sampling, input convention selectable |
| `eval/val32k.py`, `eval/val-table.py` | val-split fvu passes and the sweep comparison table |
| `data/apoc-stage-atlas.slurm`, `data/login_pull.py` | atlas pull from HF, size check, and the per-layer `ORDER BY token_idx` re-sort (duckdb) that produces `layerNN-tokensorted.parquet` |
| `data/check_parquet.py` | shard sanity check |
| `jobs/apocrita/*.slurm` | Slurm job specs (QMUL Apocrita, A100-40/80) for every training and census run |
| `jobs/dusk/*` | single-host ROCm runs (L00 pilot, topup chain, 64k rung staging) |
| `publish/*` | HF upload jobs and model-card text. Tokens come from the job environment only |
| `docs/METRICS.md` | **binding** fvu quoting rules and the input-convention table. Read before using any number |
| `docs/RUNPLAN.md`, `docs/APOC-array-spec.md`, `docs/SIZING-dusk-vs-apocrita.md` | run plan, array design, sizing |

## Which code trained what (provenance)

The trainer's history is in this branch's commits:

1. **As-run snapshot.** `train/train_sae.py` here is ops_sh commit `cd4774f`
   (2026-11-04), the last trainer change before the 64k sweep. Every run listed
   below trained with this code path.
2. **Post-sweep fix (2026-11-07).** In the next commit, the trainer refuses to
   run when the per-neuron scale vector cannot be resolved, and gains `--layer`
   and `--raw-acts`. The 64k sweep spec gains `--layer`.

| family | job spec | input convention |
|---|---|---|
| 32k sweep, 64 layers, 6k steps | `jobs/apocrita/apoc-sae-layer.slurm` | per-neuron **scaled** |
| 12k topup, 44 layers | `jobs/apocrita/apoc-topup.slurm` | scaled |
| L57 plateau, L63 retrains | `apoc-l57-plateau.slurm`, `apoc-l63-retrain.slurm` | see `docs/METRICS.md` |
| 64k probe / sweep, 64 layers | `apoc-64k-probe.slurm`, sweep spec (see below) | **RAW**: labels like `L00w64` fail the layer parse, so `scale_vec` was silently `None` |
| 128k probe | `apoc-128k-probe.slurm` | same trainer: verify the label convention before comparing |

Width families use different input conventions, so never compare fvu across
them. `docs/METRICS.md` has the rules.

## Open provenance items

- **The as-submitted 64k sweep spec is not in this branch yet.** The local
  `apoc-64k-sweep.slurm` was edited on 2026-11-07 (it added `--layer`), after the
  sweep ran. The exact submitted scripts are being recovered from Slurm
  (`sacct --batch-script`) and will land in `jobs/apocrita/batch-scripts/`.
- The md5 of the trainer deployed on Apocrita (`$SCR/train_sae.py`) is not yet
  reconciled against `cd4774f`. That check confirms item 1 independently.
- `jobs/dusk/` scripts are the dawn-side copies. The canonical copies on the
  dusk host (`~/marin-sae/`) were unreachable when this snapshot was taken.

## Environment (Apocrita GPU jobs)

`apptainer exec --nv` of `bolmo_cu118.sif` with `PYTHONUSERBASE` pointing at a
user base that has `eai-sparsify` installed (GPU nodes have no egress, so no pip
at runtime). Each job stages its layer parquet to node-local `/tmp`. Some
earlier `.sh` drafts reference a `sae-venv` that never existed on scratch; the
`.slurm` files are the specs that ran.
