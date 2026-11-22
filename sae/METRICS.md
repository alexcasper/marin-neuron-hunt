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

| family | input convention | census validity | k |
|---|---|---|---|
| 64k sweep, all 64 layers (`L{LL}w64`, APOC) | RAW acts (scale_vec silently None) | only no-scale census (v2) | 32 (`apoc-64k-sweep.sh:36`) |
| 64k L00 run (dusk, `run_L00_64k.sh`, label `L00`) | scale-normalized (label parsed pre-2026-11-22) | scaled census required | 64 (`run_L00_64k.sh:27`) |
| 128k probe/rollout chain (`L{LL}w128`) | same train_sae.py — verify label convention before comparing | — | 32 (`apoc-128k-probe.sh:39`); verify rollout before quoting |
| 32k sweep + 12k topup | scale-normalized acts | published censuses valid for 32k | 32 (`run_topup.sh:43`) |

**Rule 6 (sh-b6h, 2026-11-22): always quote k next to width in any comparison.** The "64k"
width label spans k=32 (64-layer sweep) and k=64 (dusk L00 run) — a width-only comparison
across those runs silently compares different sparsity budgets. Verify k from the run script
or the checkpoint cfg.json before any cross-run table.

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
5. **Silent fallbacks are bugs**: `train_sae.py` hard-fails when scale_vec cannot be
   resolved (never silently None again). FIXED IN THE CANONICAL REPO 2026-11-22
   (sh-kas): layer labels parse via `^L(\d+)` in `sae/sae_conventions.py`, scale
   resolution hard-fails, and the resolved convention is written to
   `<out-dir>/convention.json` + stamped into train-log records. Runtime copies on
   dusk/APOC must be re-synced from the repo before the next launch.

## Implementation notes

### Convention rescale mapping (sh-1ls, alex RESCALE decision 2026-11-22)

Cross-width comparisons are LEGAL via exact reparameterization — no retrain needed.
`sae/rescale_sae.py` converts a scaled-convention checkpoint to raw (or inverse):

    W_enc' [j,i] = W_enc[j,i] / s_i      b_dec' = s ⊙ b_dec      W_dec'[j,i] = W_dec[j,i] · s_i

(b_enc unchanged). Exactness: (x − s⊙b_dec)⊘s = x/s − b_dec, so the encoder pre-acts —
and therefore the TopK selection and codes — are IDENTICAL; and s⊙(z·W_dec) = z·(W_dec⊙s),
so the raw-space reconstruction is the scaled-space one lifted by ⊙s. Verified on real
data: original-on-scaled lifted to raw ≡ rescaled-on-raw, TopK sets equal on 1023/1023
L00 tokens (see `data/ckpt-32k-L00-rescaled-raw/rescale_eval.json`).

**Caveats:**
- Rescaled checkpoints have NON-unit-norm decoder rows (row norms = s_i). The function is
  unchanged, but never run `set_decoder_norm_to_unit_norm()` on a rescaled checkpoint, and
  any decoder-norm-based statistic must expect row norms ≈ the per-neuron scales.
- Rescaled ckpts are written in F32 (source ckpts are bf16) — 2× file size, no precision loss.
- A "converted result" means: encode with the ORIGINAL model in its own convention, lift the
  reconstruction ⊙s, and score against RAW targets. Equivalently, run the rescaled ckpt
  directly on raw inputs. Do NOT just rescale a scalar fvu — the conversion is per-token,
  per-neuron, not a scalar factor.
- First executed data point (L00, 1,023 fineweb-10m-sparse tokens, k=32/32,768):
  32k scaled-convention fvu 0.7021 (its own convention) vs **32k RAW-convention fvu 0.2196**
  (rescale-eval) vs 64k raw reference 0.7317 (hf-table-64.md, its own census sample — the
  0.2196-vs-0.7317 pair is NOT yet same-rows-legal; the same-rows 64k eval is pending the
  sae-64k L00 download). The naive 0.7021-vs-0.7317 comparison was the illegal one.

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
