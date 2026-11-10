#!/usr/bin/env python3
"""README convention-caveat update via curl-fetched copy (no hf_hub_download)."""
import sys
from huggingface_hub import HfApi

tok = open("/data/scratch/apx186/marin32b/.hftok").read().strip()
api = HfApi(token=tok)
REPO = "alxcspr/stationhouse"
cur = open(sys.argv[1]).read()
MARK = "<!-- convention-caveat-2026-11-07 -->"
if MARK not in cur:
    cur += """

<!-- convention-caveat-2026-11-07 -->
## INPUT-CONVENTION CAVEAT (2026-11-07) - READ BEFORE COMPARING FVUS

The sae-64k sweep was trained AND validated on RAW activations: a layer-id
parse failure in the training launch (the 'LNNw64' layer-label failed the
label-parse) silently set the per-neuron scale vector to None. The sae-32k
family was trained WITH per-neuron scale normalization (plain LNN labels
parsed correctly). Consequences:
- 32k and 64k checkpoints expect DIFFERENT input conventions.
- Cross-width fvu comparisons are INVALID until conventions are unified.
- Within a width family, val-vs-val and census-vs-census remain valid.
- sae-64k census files: census_uniform.parquet / stats_uniform.json are the
  raw-acts v2 redo (input_convention: raw_acts, fvu labeled fvu_atlas_uniform);
  the scaled-input run is preserved as *_scaled_ARTIFACT.* (its ~1.0 fvus
  measured the normalization mismatch, not model quality). census_prefix.parquet
  is the original corpus-prefix census (also a scaled-input artifact).
- trainer val_fvu (trailing-holdout, raw acts) remains the reference number.
"""
    ci = api.upload_file(path_or_fileobj=cur.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="model",
                         commit_message="README: input-convention caveat 32k vs 64k + census v2 naming")
    print("README updated:", ci.oid[:10])
else:
    print("caveat already present")
