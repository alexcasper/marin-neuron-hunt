# sae-nomination/ — can the 64k SAEs nominate neurons?

Code for three probes asking whether latents of the 64k TopK SAEs (see the
[`sae`](https://github.com/alexcasper/marin-neuron-hunt/tree/sae/sae) branch)
concentrate on individual MLP neurons, which would make them a way to pick
neuron-hunt candidates.

## Layout

| path | what |
|---|---|
| `decoder/` | `nominate_pass.py`: per-latent share of the decoder row (`W_dec`) on its top neuron; `nominate_probe.py` and `check_shares.py`: probes of the probe |
| `encoder_v1/` | `nominate_enc_pass.py`: encoder **weight** shares (`NOM_MODE=weights`) and data-anchored **attribution** `\|W_enc[j,i]·x_i\|` over firing tokens with a column-shuffle control (`NOM_MODE=attribution`). CPU jobs, plus the one as-submitted batch script recovered from Slurm |
| `encoder_v2/` | `nominate_enc_pass_v2.py`: GPU, vectorized attribution with the **fixed** shuffle control, plus `test_nominate_v2.py` and the A100 job spec |
| `readout/` | per-layer readout of attribution nominations and shortlists against atlas `pole_stats` |

## Results (from the run logs)

- **Decoder side: negative, and structural.** Across 11 layers, no latent is
  neuron-dominated. The best top-1 neuron share anywhere is 4.7% (L02).
  `normalize_decoder=true` renormalizes decoder rows to unit length every step,
  so they cannot concentrate on the neuron basis.
- **Encoder weights alone: near-uniform.** On L57 the largest top-1 weight
  share is 0.065%.
- **Encoder attribution: concentrates.** On L57, the median fired latent's
  attribution spreads over about 48 effective neurons, and 40 latents have
  top-1 share ≥ 0.5.
  - Many of those latents detect the same few neurons, so dedupe by neuron.
  - Filter always-on neurons (fraction positive ≈ 1.0): they are magnitude
    artifacts.

## v1 shuffle-control bug (affects every v1 attribution output)

v1 drew the control latents with `rng.choice` (unsorted) and then looked them
up with `np.searchsorted`, which requires a sorted array. Most control latents
therefore never accumulated anything (2 of 60 reachable in
`encoder_v2/test_nominate_v2.py`). Their share rows stayed 0, which produced
the reported L57 "permuted p50 0.000".

**v1's real-vs-shuffled contrast is invalid.** Its real-data attribution
numbers are unaffected.

v2 fixes this: it sorts the control index, draws control latents only from
latents that fired on real data, and reports control statistics over rows
with nonzero mass. v2 matches v1's real-data math exactly on fire counts and
top-1 neurons, and to within 4e-9 on shares (`test_nominate_v2.py`). Its
outputs go to `layer=NN/nominate_enc_v2/`.

## Running

`encoder_v2/jobs/apoc-nominate-enc-gpu-array.slurm` (Apocrita, A100-40) runs
L57 by default as the smoke test. The full run is
`sbatch -a 0-63%8 apoc-nominate-enc-gpu-array.slurm`. The pass criteria are in
the spec header.
