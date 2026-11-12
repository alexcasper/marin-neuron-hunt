# sae-census-forensics/ — census-fvu anomaly probe chain (64k SAEs)

The probes that traced the 64k sweep's census anomaly to its root cause.
Loaded L00 64k checkpoints scored census fvu ≈ 1.09, while training-time val
fvu was ≈ 0.20. The trainer code that produced the anomaly, and the binding
fvu rules that came out of it (`METRICS.md`), are on the
[`sae`](https://github.com/alexcasper/marin-neuron-hunt/tree/sae/sae) branch.

## Root cause

`train_sae.py` parsed the layer index from `--layer-label`. Sweep labels like
`L00w64` fail `isdigit()`, so the per-neuron scale vector was silently `None`:
the whole 64k sweep trained and validated on **raw** MLP activations. The census
applied scales, so its inputs were out of distribution for the model. That
mismatch alone gives fvu ≈ 1.

## Probes, in order

Each probe is a small GPU job on the L00 64k checkpoint (`jobs/probe*.slurm`
runs `probe_decode*.py`). Each probe either rules out one hypothesis or
confirms the final one.

| probe | question | verdict |
|---|---|---|
| `probe_decode.py` | does the saved checkpoint reconstruct at all after `load_from_disk`? | baseline: prints fvu and weight stats for the loaded model |
| `probe_decode2.py` | decode with `W_dec` vs the encoder weight (tied-decoder theory) | ruled out: saved `W_dec` decodes better than a tied decoder (1.02 vs 2.15 fvu) |
| `probe_decode3.py` | train-mode vs eval-mode forward | ruled out: byte-identical |
| `probe_decode4.py` | the trainer's **own** `val_fvu`, verbatim, on the loaded checkpoint, with scales | ≈ 1.09, where training reported 0.20: the trainer's own eval reproduces the anomaly |
| `probe_decode5.py` | save/load round-trip fidelity on a small `Sae` replica | precursor to 5b (see its log) |
| `probe_decode5b.py` | save/load round-trip on the real L00 checkpoint construction | ruled out: bit-exact |
| `probe_decode6.py` | trainer's `val_fvu` with `scale_vec=None`, exactly as the sweep trained | **0.2009**, matching the training val log's 0.201: root cause confirmed |

Two method lessons from this chain:

- A "reproduction" that reuses the same loaded weights proves the code paths
  agree, not that the state is correct. Discriminate with a fresh disk reload
  and two known-different batches.
- Hand-rolled reimplementations of the trainer's eval add confounds. Call the
  trainer's own function.

## Running

Paths are Apocrita scratch (`/data/scratch/apx186/marin32b`). The jobs use
`apptainer exec --nv` of `bolmo_cu118.sif` with an `eai-sparsify` user base,
the same environment as the 64k sweep (see the `sae` branch README).
