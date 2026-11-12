# sae-patterns/ — pattern passes, feature previews, token-class aggregation

Analysis code that runs over the marin-32b SAEs (see the
[`sae`](https://github.com/alexcasper/marin-neuron-hunt/tree/sae/sae) branch)
and the atlas sparse tier. Code only: the generated previews and
characterization reports are not included.

## Layout

| path | what |
|---|---|
| `patterns/` | `patterns_pass.py`: full-layer pattern pass (per-latent hits with context spans) for the 64k dictionaries; jobs for L57, the 64k pilot layers, and all 64 layers |
| `previews/` | feature-preview extractors: `interp_preview.py` (32k), `interp_preview_12k.py` (12k topup), `feature_preview_L57.py`, `preview64k.py`, `interp_preview_dusk.py` (single-host ROCm); GPU and CPU job variants, plus the preview upload job |
| `token_layer/` | `token_class_x_layer.py`: token-class × layer MLP-activity aggregation straight from the atlas sparse tier and `tokens.parquet` (no SAE) |
| `characterize/` | `band_characterize.py`, `band_analysis.py`, `characterize_nominees.py`: band statistics and per-nominee evidence tables for attribution-nominated neurons |

## Caveats

- **Input convention.** The 64k SAEs were trained on **raw** activations and
  the 32k SAEs on per-neuron **scaled** activations. Any script that encodes
  atlas rows must match the checkpoint's convention, or fvu and activations are
  meaningless (`METRICS.md` on the `sae` branch).
- `token_class_x_layer.py` reports correlations only: it reads recorded
  activations, with no ablation, so it says nothing about causal importance.
- `characterize/` consumes encoder-attribution nominations from the
  `sae-nomination` branch. Re-run it on the v2 outputs: v1's shuffle control
  was buggy.
