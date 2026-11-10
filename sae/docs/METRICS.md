# METRICS.md — fvu quoting rules for the marin-32b SAE program

Standing methodology note from alex (2026-11-07), binding program-wide.

## ROOT CAUSE (2026-11-07, final — supersedes earlier sampling verdicts)

The census_fvu > 1 anomaly was **NOT** a data-sampling effect. Root cause:
`train_sae.py:196` parses the layer index from `--layer-label`; sweep labels
like `L00w64` fail `isdigit()`, so `scale_vec=None` **silently** — the entire
64k sweep trained AND validated on **raw (unscaled) MLP acts**. The census
applied scales, making its inputs out-of-distribution for the model → fvu ≈ 1
artifact (both prefix-sampled and uniform censuses measured it).

Proof chain (probes 1–5, logs `logs/probedec-*.out` on APOC scratch):
`val_fvu(loaded ckpt, scale_vec=None) = 0.2009` exactly matches the training
val_log's 0.201; save/load is bit-exact (probe 5b round-trip); batch
construction and eval/train mode exonerated.

## Input conventions by width family (each internally consistent)

| family | input convention | census validity |
|---|---|---|
| 64k sweep (all 64 layers) | RAW acts (scale_vec silently None) | only no-scale census (v2) |
| 128k probe/rollout chain | same train_sae.py — verify label convention before comparing | — |
| 32k sweep | scale-normalized acts | published censuses valid for 32k |

## Rules (all downstream consumption)

1. **Encode-side recompute on the target data, always** — with the SAME
   scale convention the model was trained under. Check the convention first;
   a convention mismatch alone produces fvu > 1.
2. **Label every fvu**: `val-split` or `atlas`, AND the input convention
   (`raw` / `scaled`). No naked fvu numbers.
3. **Never mix input conventions in one table.** 64k-vs-32k fvu comparisons
   are INVALID until one convention is retrained onto the other. Within-width
   val-vs-val comparisons remain valid (including the 64k-vs-128k probe chain
   if the 128k run had the same silent None — verify, don't assume).
4. **census_fvu (no-scale, convention-matched) is the honest atlas-side
   layer-quality number.** Training-loop val fvu is valid within its own
   convention only.
5. **Silent fallbacks are bugs**: `train_sae.py` now hard-fails when
   scale_vec cannot be resolved (never silently None again).

## Implementation notes

- Recompute encode-side: `sae.encode(x)` against the target rows; fvu =
  mean over rows of ||x - dec(enc(x))||² / ||x||². Do NOT use training-loop
  val_log.jsonl values for atlas-facing claims.
- The sweep-completion table tool `sae/val-table.py` joins training val
  against 32k baselines — val-split on both sides, BUT the two width families
  used different input conventions (64k raw, 32k scaled): its Δ column is
  cross-convention and must carry that caveat until conventions are unified.
  Within-width comparisons (64k val across steps, 64k-vs-128k chain) stay
  valid.
- Census fix delivered 2026-11-07: uniform row-group sampling stands; the
  fvu artifact was the scale convention (see root cause above), so census v2
  runs with scale_vec=None to match the 64k training convention. The scaled
  uniform census is preserved as `census_uniform_scaled_ARTIFACT.parquet`.
- Watcher-side corollary: training-val plateau numbers decide *training*
  questions (plateau firing, width comparisons within the same convention);
  atlas-side numbers decide *interpretability and dataset* questions — always
  recomputed under the model's own convention.
