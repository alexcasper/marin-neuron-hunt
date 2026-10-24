# ops_sh — Station House lane (founding brief)

Seat: ops_sh (Station House work: ConspiracyBench, marin-32b-neuron-atlas, related datasets).
Created 2026-10-24 by env_discord at alex's directive.

## First assignment (P1)

1. **Survey** `https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas`
   (public, not gated, 1,062 files). Read README + `pilot/fineweb-10m-sparse/MANIFEST.json`,
   pull one shard from each of `captured_mass/` and `sparse/layer=00/` and inspect the
   parquet schemas. Compute the TRUE total size via the HF tree API with cursor
   pagination before any bulk pull.
2. **Pull** to this workspace under `data/` (fast drive `/mnt/d`). If the true size is
   TB-scale, stop and route through env_apoc (pane wW:p7) to Apocrita durable storage
   (`/data/SPCS-IOT/` — absolute paths, watch inodes) instead. alex pre-approved
   Apocrita compute for this lane ("use env_apoc if needed").
3. **Train SAEs** on the captured activations. Follow the sae-ops pipeline: EleutherAI
   `eai-sparsify` (TopK/BatchTopK SparseCoder) — NOT `dictionary_learning`. Gate order:
   deps → tiny smoke on REAL shards → full run (64k width, 200k steps). If GPU is
   needed (it will be, for 32B-width activations), hand env_apoc a job spec: sbatch
   array, node-local staging, apptainer SIF. Compute approved; report job IDs.
4. **Evidence rule** (standing): every number from real logs/outputs. Beads close only
   with logs.

## Coordination

- env_apoc (wW:p7) is the Apocrita dispatcher — hand it job specs, don't hand-roll SSH.
- ConspiracyBench fork lives at `~/d/tools/ConspiracyBench` + `alexcasper/ConspiracyBench`
  (private GitHub) and as the ConspiracyBench submodule of /mnt/d/ops/apart.
- Report results through env_discord (Bedivere) to #camelot.
