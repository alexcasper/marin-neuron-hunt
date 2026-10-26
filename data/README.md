---
license: apache-2.0
pretty_name: Marin 32B Neuron Atlas
language:
- en
tags:
- interpretability
- mechanistic-interpretability
- activations
- neurons
- residual-stream
- marin
size_categories:
- 1B<n<10B
configs:
- config_name: fineweb-1m.pole_stats
  data_files: "pilot/fineweb-1m/summaries/pole_stats.parquet"
- config_name: fineweb-1m.exemplars
  data_files: "pilot/fineweb-1m/summaries/exemplars.parquet"
- config_name: fineweb-1m.logit_lens
  data_files: "pilot/fineweb-1m/summaries/logit_lens.parquet"
- config_name: fineweb-1m.neuron_scale
  data_files: "pilot/fineweb-1m/summaries/neuron_scale.parquet"
- config_name: fineweb-1m.tokens
  data_files: "pilot/fineweb-1m/summaries/tokens.parquet"
- config_name: fineweb-1m.docs
  data_files: "pilot/fineweb-1m/summaries/docs.parquet"
- config_name: fineweb-1m.sparse
  data_files: "pilot/fineweb-1m/sparse/*/*.parquet"
- config_name: fineweb-1m.captured_mass
  data_files: "pilot/fineweb-1m/captured_mass/*.parquet"
- config_name: fineweb-10m.pole_stats
  data_files: "pilot/fineweb-10m-sparse/summaries/pole_stats.parquet"
- config_name: fineweb-10m.exemplars
  data_files: "pilot/fineweb-10m-sparse/summaries/exemplars.parquet"
- config_name: fineweb-10m.docs
  data_files: "pilot/fineweb-10m-sparse/summaries/docs.parquet"
- config_name: fineweb-10m.sparse
  data_files: "pilot/fineweb-10m-sparse/sparse/*/*.parquet"
---

# Marin 32B Neuron Atlas

MLP-neuron activations, residual-stream vectors and per-neuron summaries from
[**Marin 32B**](https://huggingface.co/marin-community/marin-32b-base), extracted on fixed FineWeb text for the
[Station House](https://app.station-house.net) community neuron hunt.

**Status: three pilot extractions are uploaded** under `pilot/`. They are research data for exploring the format and
the neurons. The full multi-source atlas comes later.

| prefix | tokens | docs | size | what it is |
|---|---|---|---|---|
| `pilot/fineweb-10m-sparse/` | 10,000,384 | 4,883 | ~271 GB | **Breadth pilot:** summaries + `sparse` + `captured_mass` only (no `resid`/`deep`). 10× the docs, so topic and rare neurons get real exemplars. Content-filtered at build (266 docs dropped). |
| `pilot/fineweb-1m/` | 1,001,472 | 489 | ~108 GB | **All tiers**, including the residual stream and the dense `deep` tier. Content-filtered at build. |
| `pilot/stage-a-100k/` | 100,352 | 49 | ~33 GB | Smoke test. One flagged doc was removed afterwards; per-neuron aggregates still include it. |

Use `fineweb-10m-sparse` for "what does this neuron do" (exemplars, stats, all firings), and `fineweb-1m` when
you need residual-stream or dense vectors. In `fineweb-10m-sparse`, each sparse layer is a single sorted file
(`sparse/layer=LL/sorted-00000.parquet`, ~4 GB): point queries over `hf://` read only the row groups they need.

The pilots are **independent samples** (seeds 20261025 / 20261024 / 20261023). `doc_id` and `token_idx` are local to each prefix, so don't join across
prefixes. Each prefix has a `MANIFEST.json` with the sha256 and size of every file, plus the full extraction report
(timings, checks, bytes per token).

## Why this exists

1. **Are large-model neurons already clean?** Nick Cammarata's
   ["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/) (Goodfire, Oct 2026) reports that ~90%
   of Qwen3-32B neuron *poles* (positive and negative sides judged separately) have one main job. The audit is
   preliminary and the claim is contested. Marin 32B has the same architecture but was trained by a different team
   on different data, so it's an independent replication target.
2. **Marin is fully open**, training data included. Neurons can in principle be traced back to the training text.
3. **Top-k snippets aren't enough.** This dataset also stores gate/up pre-activations, the residual stream and
   per-head attention outputs, so most analysis needs no GPU.

## Model and definitions

- Model: `marin-community/marin-32b-base` @ `f19b346615212ff9a6d92c10861a9cdb100cbf97` (branch `main`, the final
  Mantis cooldown). `Qwen3ForCausalLM`: 64 layers, hidden 5,120, 40 heads × 128, SwiGLU MLP with 27,648 neurons
  per layer (1,769,472 neurons).
- Run in bf16 with SDPA attention (torch 2.11, transformers 5.10.2) on one H200. Hooks were read-only: hooked
  logits are bit-identical to an unhooked forward pass. First-batch next-token loss was 2.10 nats.
- **Neuron activation** `act = silu(gate) * up` = the input to `mlp.down_proj`. It is signed (no ReLU).
- **Pole**: `+1` = the positive side of a neuron, `-1` = the negative side.
- **Windows**: every doc contributes exactly one window of 2,048 tokens: `<|begin_of_text|>` + the first 2,047
  tokens. **`token_idx = doc_id * 2048 + pos`.** Position 0 (BOS) is excluded from every tier and statistic, so the
  data covers 2,047 tokens per doc.
- Tokenizer: the model's own (Llama-3 style, vocab 128,256). Decode with
  `AutoTokenizer.from_pretrained("marin-community/marin-32b-base")`.
- Stored values are fp16. Values beyond ±65,504 are clamped; there were 4 in `fineweb-1m` (`mlp_out`), and the
  counts are in `MANIFEST.json`.

## Corpus

- Source: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT` (ODC-By 1.0), streamed
  with a seeded shuffle. Seed 20261024 for `fineweb-1m`, 20261023 for `stage-a-100k`.
- Only docs with ≥2,047 tokens qualify, about 4.5% of FineWeb. **That biases the corpus toward long documents.**
- **Content filter**: a doc is dropped if an adult keyword starts a hostname token, or if its density of
  [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) English hits is
  ≥5 hits and ≥0.2% of words ("sex/sexual/sexuality" don't count). `fineweb-1m` dropped 24 of the qualifying docs
  (4 by host, 20 by words). The filter is a heuristic: some offensive text may remain.
- `docs.profile` says which tiers a doc is in: `deep` docs are in deep + resid + sparse, `resid` docs in resid +
  sparse, `bulk` docs in sparse only. Every doc is in the summaries.

## Files and columns (same layout in each pilot prefix; `fineweb-10m-sparse` has no `resid/` or `deep/`)

### `summaries/`
| file | one row per | columns |
|---|---|---|
| `tokens.parquet` | token (incl. pos 0) | `token_idx, doc_id, pos, token_id, token_str` (raw BPE piece; `Ġ` = leading space) |
| `docs.parquet` | doc | `doc_id, profile, source, source_id, url, license, doc_tokens` (full doc length), `text` (the window's text) |
| `pole_stats.parquet` | neuron (1,769,472) | `layer, neuron, n, mean, std, frac_pos, max, min, scale_abs_p999`, then `pos_q5, pos_q9, pos_q99, pos_q999, pos_q9999` and `neg_q…` = 50/90/99/99.9/99.99th percentiles of positive values / of the magnitude of negative values. Percentiles come from a 256-bin log histogram (1e-5–1e4) and are upper bin edges (≈ +8% resolution). |
| `neuron_scale.parquet` | neuron | `layer, neuron, scale` = max(`scale_abs_p999`, 1e-4), the per-neuron scale used to choose the sparse tier (taken from this pilot's own pass 1) |
| `exemplars.parquet` | (neuron, pole, rank) | `layer, neuron, pole, rank` (0 = strongest), `token_idx, act` (signed). Top-32 per pole, taken from the top-64 positions with at most 2 per doc |
| `logit_lens.parquet` | neuron | `layer, neuron, top_ids, top_vals, bottom_ids, bottom_vals`: the 10 tokens that `W_U · (W_down[:, i] ⊙ final_norm.weight)` raises most / lowers most. The positive pole pushes `top_ids` up; the negative pole pushes `bottom_ids` up. Weights only; no corpus involved |
| `provenance.json` | — | model sha, dtype, versions, GPU, corpus settings |

### `sparse/layer=LL/` (all tokens)
`token_idx, neuron, gate, up` (fp16). For every token × layer, the **64 neurons with the largest `|act| / scale`**
(activation relative to each neuron's own 99.9th percentile), so quiet neurons' strongest firings are kept too.
Compute `act = gate / (1 + exp(-gate)) * up`. Sorted by `(neuron, token_idx)`.

### `captured_mass/`
`token_idx, layer, raw, norm`: the share of Σact² (raw) and of Σ(act/scale)² (norm) that the 64 stored neurons
capture for that token × layer.

### `resid/site=S/layer=LL/` (`resid` + `deep` docs)
`token_idx, vec` with `vec` = fixed_size_list<float16>[5120].
- `site=resid_pre`: the residual stream entering layer LL (all 64 layers).
- `site=attn_out` / `site=mlp_out`: what attention / the MLP add to the residual, for layers 0, 4, …, 60 only.

### `deep/site=S/layer=LL/` (`deep` docs, every layer)
`resid_pre` [5120], `attn_z` [5120] = per-head attention output before `o_proj` (head h = `vec[h*128:(h+1)*128]`),
`gate_pre` [27648], `up` [27648], `mlp_out` [5120].

### `MANIFEST.json`
`manifest` (path → sha256, bytes), `report` (extraction report), `notes`.

## Querying

```sql
-- DuckDB, straight from the Hub. Strongest examples of L40 neuron 25546 ("and" before "vice versa"), with tokens
SELECT e.rank, e.act, t.token_str
FROM 'hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/summaries/exemplars.parquet' e
JOIN 'hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/summaries/tokens.parquet' t USING (token_idx)
WHERE e.layer = 40 AND e.neuron = 25546 AND e.pole = 1 ORDER BY e.rank LIMIT 10;

-- every stored firing of that neuron, with act
SELECT token_idx, gate, up, gate / (1 + exp(-gate)) * up AS act
FROM 'hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/sparse/layer=40/*.parquet'
WHERE neuron = 25546 ORDER BY act DESC;

-- residual stream with site/layer as columns (hive partition values are strings, e.g. layer = '04')
SELECT site, layer, count(*)
FROM read_parquet('hf://datasets/Station-house/marin-32b-neuron-atlas/pilot/fineweb-1m/resid/*/*/*.parquet', hive_partitioning = true)
GROUP BY ALL;
```
Point queries over `hf://` are fine. For whole-tier work, download first (`hf download
Station-house/marin-32b-neuron-atlas --repo-type dataset --include "pilot/fineweb-1m/sparse/*"`), because HF
rate-limits file requests.

## What it's for, and limits

- **For:** explaining neurons and poles, probes on the residual stream, direct logit attribution, gate-vs-up
  analysis, head attribution, choosing candidates for causal tests, replicating or refuting neuron claims.
- **Not causal on its own.** A clean exemplar list doesn't make a neuron necessary or sufficient for anything.
  Ablation and steering run separately.
- Pilots only: ~1M tokens of one long-doc-biased source. Topic and rare neurons have few examples, and code, math
  and non-English neurons are barely exercised.
- The sparse tier drops everything outside each token's top-64 (see `captured_mass`). Attention patterns aren't
  stored.

## License

Activations and derived statistics: Apache-2.0, matching the model. Corpus text in `docs.parquet` /
`tokens.parquet` keeps FineWeb's ODC-By 1.0 license.

Maintained by Station House. Questions or corrections: open a discussion on this repo.
