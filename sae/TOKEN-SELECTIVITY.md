# Token selectivity per neuron (fineweb-10m atlas) — 2026-11-12

How concentrated is each neuron's firing on a single token type, per layer and
pole? Two passes, both on dawn's local copy of `pilot/fineweb-10m-sparse`.
Bead `sh-1no`. Data only: no interpretation of individual neurons.

## Outputs (gitignored, under `data/derived/`)

| file | rows | source | script |
|---|---|---|---|
| `token_selectivity_exemplars.parquet` | 3,538,920 | top ~31 exemplars per neuron/pole (`exemplars.parquet` × `tokens.parquet`) | `sae/token_selectivity_exemplars.py` (7 s) |
| `token_selectivity_full/layer=NN.parquet` (64 files) | 3,535,671 total | every sparse-tier entry: each token's top-64 neurons, 10,000,384 tokens, 40,941,572,096 firings (= 64 × 639,712,064 rows) | `sae/token_selectivity_full.py` (~13 min, resumable per layer) |

Columns, per (layer, neuron, pole), pole = sign of `silu(gate)·up`:

- `n_ex` / `n_fire`: number of exemplars (pass 1) or of firings (pass 2).
- `n_distinct_tokens`, `n_distinct_docs`.
- `top_token_id` / `top_token_str`.
- `top_token_share`: count share of the top token.
- `top_token_mass_share`: |act| mass share of the top token.
- `eff_tokens` = 1/Σp², where 1 means a single token type.
- Pass 2 only: `top_token_lift` = top_token_share ÷ that token's corpus frequency. ≫1 means selective; ≈1 means the token is simply common.

Neuron/pole pairs with no firings have no row (e.g. L57-N3872 has no negative
pole). That's 24 missing in pass 1 and 3,417 in pass 2.

Verification: independent Python/numpy recomputes, straight from the source
parquets without duckdb in the stat path, MATCH on every column.
- Pass 1: 9 neuron/poles across L0, L38 and L57.
- Pass 2: L0 N17 and N600, L57 N3872 and N20001, both poles.

## Results

Full distribution (pass 2), medians per neuron/pole, by layer band:

| layers | n_fire | top_token_share | eff_tokens | lift | % share ≥ 0.5 | % share ≥ 0.5 with n_fire ≥ 100 and ≥ 10 docs |
|---|---|---|---|---|---|---|
| 0–7 | 7,001 | 0.086 | 51 | 60.3 | 2.05 | 1.87 |
| 8–15 | 8,928 | 0.059 | 98 | 3.5 | 0.65 | 0.63 |
| 16–23 | 10,268 | 0.053 | 110 | 1.9 | 0.28 | 0.25 |
| 24–31 | 10,546 | 0.054 | 107 | 1.9 | 0.40 | 0.38 |
| 32–39 | 10,595 | 0.064 | 84 | 2.9 | 0.74 | 0.68 |
| 40–47 | 10,193 | 0.062 | 86 | 2.7 | 0.33 | 0.28 |
| 48–55 | 10,117 | 0.057 | 98 | 2.0 | 0.14 | 0.09 |
| 56–63 | 8,407 | 0.062 | 88 | 2.5 | 0.69 | 0.52 |

**The exemplar tail overstates selectivity** (3,535,671 neuron/poles joined).
The correlation of tail share vs full share is 0.42, and the top token agrees
in only 18.0% of rows.

| tail (pass 1) bucket | n | same top token in full | median full share | % full share ≥ 0.5 |
|---|---|---|---|---|
| tail ≥ 0.9 | 13,814 | 67.3% | 0.257 | 27.2% |
| tail 0.5–0.9 | 123,864 | 45.1% | 0.128 | 6.5% |
| tail < 0.5 | 3,397,993 | 16.8% | 0.059 | 0.3% |

## Usage rules

- **Quote selectivity from pass 2.** Read the exemplar tail as what a neuron
  does at its *strongest*, not what it typically fires on. Most tail-selective
  neuron/poles (72.8% of tail ≥ 0.9) are not ≥ 0.5 selective over all their
  firings.
- **Always read `top_token_share` together with `top_token_lift`.** A high
  share on a common token (lift ≈ 1) is frequency, not selectivity.
  - Layers 0–7 have median lift 60, consistent with token identity dominating
    the earliest MLPs.
  - From layer 16 on, median lift is ≈ 2–3.
- **Filter `n_fire ≥ 100` and `n_distinct_docs ≥ 10`** before treating a row
  as a candidate. The 64k SAE's attribution sample (200k tokens) is far
  smaller than the atlas, so `fired_tokens` 1–3 rows there are noise.
- "Fires" here means *present in the token's top-64 neurons* (the sparse
  tier). Below-top-64 activity is not recorded, so `n_fire` undercounts weak
  firing.
