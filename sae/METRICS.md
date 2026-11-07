# METRICS.md — fvu quoting rules for the marin-32b SAE program

Standing methodology note from alex (2026-11-07), binding program-wide.

**Training val_fvu and atlas fvu are DIFFERENT distributions.** The
L00/L01/L02 pattern pass proved it: encode-side recompute on atlas rows gives
L00 1.10 vs training val 0.201, L02 0.80 vs 0.157, L01 0.77 vs 0.665. The gap
is the data stream, not a broken decode path (encode-side recompute reproduces
census numbers exactly; see `sae/pattern-report-L00-L01-L02.md` for the full
verdict — the census samples the first 204,800 tokens of the tokensorted
parquet, few atypical corpus-opening docs, while training val uses the
trailing in-distribution holdout).

## Rules (all downstream consumption)

1. **Encode-side recompute on the target data, always.** Any fvu quoted for
   downstream consumption must be recomputed encode-side on the TARGET data
   distribution — never carry over the training-loop val number.
2. **Label every fvu.** Every fvu number written anywhere (reports, tables,
   pads, beads, HF cards) is labeled `val-split` or `atlas` (census/atlas
   sample stream). No naked fvu numbers.
3. **census_fvu is the honest layer-quality number outside training.** Never
   mix val-split and census/atlas numbers in one table — the two columns are
   different distributions and the mix is not a comparison.

## Implementation notes

- Recompute encode-side: `sae.encode(x)` against the target rows; fvu =
  mean over rows of ||x - dec(enc(x))||² / ||x||². Do NOT use training-loop
  val_log.jsonl values for atlas-facing claims.
- The sweep-completion table tool `sae/val-table.py` joins training val
  against 32k baselines — val-split on both sides, internally consistent;
  its census-bias caveat covers the legacy `fvu_sampled` column. Any new
  atlas-facing table needs atlas-side recomputation first.
- Fix queued for the census itself: uniform row-group sampling (interp-style
  spread) so the census stream stops over-weighting corpus-opening docs.
- Watcher-side corollary: training-val plateau numbers decide *training*
  questions (plateau firing, width comparisons within the same val split);
  atlas-side numbers decide *interpretability and dataset* questions.
