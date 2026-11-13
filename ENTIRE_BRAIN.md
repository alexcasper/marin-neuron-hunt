# ENTIRE_BRAIN.md — ops_sh / Marin-32B neuron lane

Current as of 2026-11-13 23:17 UTC. Facts below come from the checked-in project
docs/beads, the verified local data pull, and live Hugging Face API/tree reads
(`repo_info(..., files_metadata=True)`); sizes are decimal GB from HF-reported
bytes unless marked GiB.

## 1. Purpose and source of truth

ops_sh is the Station House lane for:

1. the **Marin 32B neuron hunt** (explain all MLP neurons and both poles);
2. the **Marin 32B neuron atlas** data survey/pull and derived datasets;
3. **TopK SAE training** over MLP activations plus neuron nominations;
4. **ConspiracyBench**, which lives in separate forks/panes and is not stored
   in this repo's data tree.

Canonical local worktree: `/mnt/d/ops/sh`. Canonical branch:
`merged-from-apoc`. Remotes:

- GitHub `origin`: `https://github.com/alexcasper/marin-neuron-hunt.git`
- Entire `entire`: `entire://aws-eu-central-1.entire.io/et/ops-sh/sh`
  (`/et/ops-sh/sh`, primary cluster `aws-eu-central-1.entire.io`)

On dawn, Entire must use the file token store because there is no
`org.freedesktop.secrets` service: prefix commands with
`ENTIRE_TOKEN_STORE=file` or persist `{"token_store":"file"}` in Entire config.

## 2. Repo map and canonical docs

- `BRIEF.md` — founding brief: survey first, bulk-pull later, train TopK SAEs,
  route GPU work through env_apoc when needed.
- `data/README.md` — atlas schema and pilot layout (snapshot before the
  post-pilot prefixes were added).
- `data/VERIFICATION-2026-10-26.md` — proof that the then-current pilot pull
  was complete and byte-verified.
- `sae/RUNPLAN.md` — original dusk pilot/plan and measured sizing.
- `sae/METRICS.md` — binding fvu-quoting/input-convention rules.
- `sae/hf-README.md` / `sae/hf-table-64.md` — 32k sweep upload card/table.
- `sae/TOKEN-SELECTIVITY.md` — local computation and interpretation rules for
  the token-selectivity dataset.
- `sae/ATTRIBUTION-ENC-15LAYERS-2026-11-10.md` — 15-layer then 29-layer
  encoder-attribution pass.
- `hunt/` and `hunt-sae*/` — local worktrees/archives of the hunt repo and SAE
  code snapshots. `hunt-sae/sae/README.md` is the best provenance summary for
  trainer versions and job conventions.

Active bead: `sh-uxq` — explain `L00-N10657` both sides; waiting on the daily
runner for request `-002` and PR merge. Closed: `sh-bs6` (atlas survey/pull),
`sh-hr9` (SAE training plan).

## 3. State of local data

`data/pilot` is 384 GiB and was verified on 2026-10-26 against the then-current
HF pilot subtree: 1,057 manifest entries / 1,060 tree files, 0 missing, 0 size
mismatches, 411.9 GB decimal, with three SHA-256 spot checks. It contains the
three pilot prefixes only.

The public atlas grew after that verification: the current HF repo adds
`vocab/`, `rare-tokens/v1/`, `probes/capybara-v1/`, and `viz/v1/`. Treat
`data/VERIFICATION-2026-10-26.md` as covering **only `pilot/`**, not the
current whole dataset. Derived outputs are under gitignored
`data/derived/`; canonical public copies are the HF datasets listed below.

## 4. Related Hugging Face buckets

### 4.1 Corrected inventory

A stale note (now only in a stash/index commit) called the following
"Station-House repos": `marin-32b-neuron-atlas-text`,
`marin-32b-neuron-atlas-activations`, `ConspiracyBench` as a dataset, and
`marin-neuron-hunt` as a dataset. Current HF API checks do **not** support
those names/types. Do not reuse that list. The live, relevant inventory is:

| repo | type | state | files | bytes | relevance |
|---|---|---|---:|---:|---|
| `Station-house/marin-32b-neuron-atlas` | dataset | public, not gated; sha `6402b9eb`; modified 2026-11-06 | 1,468 | 442.149 GB | primary input atlas |
| `alxcspr/stationhouse` | model | public; sha `5b6a7495`; modified 2026-11-11 | 1,364 | 804.662 GB | our 32k/64k/128k SAE outputs and curves |
| `alxcspr/marin-32b-token-selectivity` | dataset | public; sha `9fcda356`; modified 2026-11-12 | 69 | 0.162 GB | derived neuron/token selectivity table |
| `alxcspr/marin-32b-nominate-enc` | dataset | public; sha `d6553be7`; modified 2026-11-12 | 130 | 0.011 GB | 64k SAE encoder-attribution nominations, v1 |
| `marin-community/marin-32b-base` | model | public; sha `f19b3466` | 21 | 130.096 GB | upstream model/provenance |
| `Station-house/marin-neuron-hunt` | Space | public; sha `6e222162`; modified 2026-11-10 | 3 | tiny | public hunt UI; references atlas + `alxcspr/stationhouse` |
| `Station-house/ConspiracyBench` | Space | public; sha `5c54d443`; modified 2026-11-13 | 1,543 | not measured here | benchmark/experiments app; separate lane |
| `Station-house/Consciousness-Self-Attribution` | Space | public; sha `0e09e97f`; modified 2026-10-06 | 5 | tiny | related Station House static site |
| `Station-house/qwen3.5-4b-xsft-v4-think-lora` | model | public; sha `7dfcc0b1`; modified 2026-11-11 | 7 | 0.130 GB | Station House ConspiracyBench LoRA, not Marin neuron lane |

Other `alxcspr` HF repos exist for other fleet lanes (Esperanto SFT, pp/arXiv
corpora, g4-E4B SAE layers, etc.). They are not inputs to the Marin neuron lane
unless a specific task says so.

### 4.2 `Station-house/marin-32b-neuron-atlas`

This is the primary atlas. Current totals by prefix:

| prefix | files | bytes | content |
|---|---:|---:|---|
| `pilot/fineweb-10m-sparse` | 82 | 271.342 GB | breadth pilot: summaries + sparse + captured_mass; 10,000,384 tokens / 4,883 docs; no `resid/` or `deep/` |
| `pilot/fineweb-1m` | 489 | 107.805 GB | dense pilot: 1,001,472 tokens / 489 docs; all tiers incl. residual and deep |
| `pilot/stage-a-100k` | 489 | 32.757 GB | smoke test: 100,352 tokens / 49 docs |
| `rare-tokens/v1` | 71 | 18.336 GB | fresh 64-token windows and repeat probes for rare tokens |
| `probes/capybara-v1` | 138 | 11.092 GB | dense capybara-neuron probe capture |
| `viz/v1` | 195 | 0.808 GB | compact browser binaries derived from the 10M pilot |
| `vocab` | 2 | ~8.7 MB | 128,256-row vocabulary audit table plus decoded-token JSON |
| root | 2 | ~28 KB | README + `.gitattributes` |

**Pilot model/provenance.** Marin 32B base at `f19b346615212ff9a6d92c10861a9cdb100cbf97`,
bf16, SDPA, seq 2,048, H200. Activation is signed:
`act = silu(gate) * up`. Each neuron is investigated as positive and negative
poles. Only docs with ≥2,047 FineWeb tokens qualify, so pilots bias toward long
documents. `token_idx` is local to each pilot prefix — never join across
pilots.

**Pilot file families.**

- `summaries/pole_stats.parquet`: one row per neuron (1.769M), signed
  quantiles and per-neuron scale.
- `summaries/exemplars.parquet`: top 32 per pole from top-64 positions, at
  most two per document.
- `summaries/logit_lens.parquet`: weight-only top/bottom token effects.
- `sparse/layer=LL/`: for every token × layer, the 64 neurons with largest
  `|act| / scale`; sorted `(neuron, token_idx)`.
- `captured_mass/`: raw and normalized fraction of activation mass captured by
  those top-64 neurons.
- `resid/` and `deep/` in the 1m pilot only: residual sites and dense MLP/attention
  internals.
- `MANIFEST.json`: SHA-256/bytes plus extraction report/checks.

**Current non-pilot prefixes.**

- `vocab/vocab.parquet`: one row per tokenizer id. Adds category, raw bytes,
  10m/1m pilot counts, counts from 109.0B scanned tokens, snippet availability,
  embed/unembed norms, Land-Bartolo undertraining score, and repeat-probe
  logprob/rank. 79,019 regular tokens occur ≤5 times in the 10m pilot; 77,999
  rare tokens got at least one snippet; 104,547 tokens are top-1 repeats while
  3,314 are not top-10.
- `rare-tokens/v1`: 283,730 windows = 155,474 natural + 128,256 repeat probes;
  695,398 captured positions. New capture, not a subset of the pilots. Sparse
  activations only at target positions, using the 10m pilot's neuron scale so
  values are directly comparable. Sources include FineWeb, multilingual FineWeb-2,
  and Common Pile stack code/StackExchange; 1,020 rare tokens had no usable
  natural snippet.
- `probes/capybara-v1`: 253 prompts / 3,455 captured non-BOS positions, fully
  dense over all neurons plus `resid_pre`. Sets separate capybara concept from
  other animals, multilingual forms, the Ruby gem, token-overlap decoys, and
  neutral text. `pair_screen.parquet` is a screening table, not a result; raw
  `|d|` rankings are dominated by tiny layer-0..2 `bara` detectors, so
  capybara-minus-Ruby-gem and later-layer/concept checks are required.
- `viz/v1`: no new capture. Binary `captures`, `stats`, and `exemplars`
  records per layer for the browser UI, plus overview and tokenizer JSON.

**Use limits.** The atlas is observational; exemplars do not prove necessity or
sufficiency. Sparse tiers omit sub-top-64 activity, and attention patterns are
not stored. Pilots under-exercise code/math/non-English and rare neurons; the
new rare-token and probe prefixes partially fill those gaps.

### 4.3 `alxcspr/stationhouse`

This is the Marin SAE output bucket. Current totals:

| path | files | bytes | content |
|---|---:|---:|---|
| `sae-64k/` | 690 | 485.716 GB | canonical full 64-layer width-65,536 TopK sweep, plus earlier L45/L57/L63 width-probe artifacts and two grid PNGs |
| `sae-32k/` | 516 | 231.963 GB | width-32,768 sweep (all 64 layers), nested curves and interp audit tarballs |
| `sae-128k/` | 27 | 86.978 GB | width-131,072 probes for layers 45/57/63 only; stopped after low val gain |
| `curves/` | 129 | 5.571 MB | root-level 64-layer training-curve CSV/PNG pack and overlay |
| root | 2 | ~4 KB | README + `.gitattributes` |

**32k family.** TopK, 32,768 latents, k=32, eai-sparsify; all 64 layers had a
6k run, then 44 layers got a 12k topup. Published best copy per layer: 44 topup
finals + 20 already-good 6k finals. Inputs were **scale-normalized** acts.
`hf-table-64.md` has the per-layer census table.

**64k family.** TopK, 65,536 latents, k=32, all 64 layers, plateau-early
stopping. This is the canonical full sweep. Critical convention: the as-run
labels such as `L00w64` failed the layer parse, so `scale_vec` was silently
`None`; the family trained/validated on **raw acts**. Therefore:

- never compare 64k vs 32k fvu directly;
- label every fvu with split and input convention;
- use no-scale, convention-matched census fvu for atlas-facing claims;
- treat the scaled uniform-census artifacts as artifacts, not quality numbers.

**128k family.** Same architecture family, width 131,072, layers 45/57/63.
The width probe was stopped: median val gain 0.0044 < 0.02. It must be checked
against its own label convention before comparison.

### 4.4 `alxcspr/marin-32b-token-selectivity`

Derived from `pilot/fineweb-10m-sparse`; 69 files / 0.162 GB.

- `full/`: 64 parquet files, 3,535,671 rows covering 40,941,572,096 sparse-tier
  firings. One row per `(layer, neuron, pole)` with firing count, distinct
  tokens/docs, top-token share/mass/lift, and effective token count.
- `exemplar_tail/`: 3,538,920 rows from ~31 exemplars per neuron/pole.
- 3,273 no-fire neuron/pole pairs are absent from `full`; 24 from the tail.

Binding reading rules: use `full` for claims; read share with corpus lift;
filter `n_fire >= 100` and `n_distinct_docs >= 10`; treat a selective token as
a hypothesis. Key result: the exemplar tail overstates selectivity. Tail-vs-full
top-token agreement is 18%; only 27.2% of tail rows with ≥0.9 share are ≥0.5
selective over all firings.

### 4.5 `alxcspr/marin-32b-nominate-enc`

v1 encoder-attribution dataset; 130 files / 11.43 MB, 512,000 rows
(64 layers × first 8,000 alive latents).

For each tracked SAE latent it records fired-token count, top contributing
MLP neuron, that neuron's input share, and effective neuron count. It is
intended to nominate candidate neurons for the hunt.

Warnings that must stay attached to the data:

- **v1 shuffle controls are invalid**: buggy unsorted lookup left control
  `top1_p50 = 0.0` in all 64 layers. Ignore every control field.
- **Strong nominations can be small-sample artifacts**: of 777 rows with
  `top1_share >= 0.5`, 558 fired only 1–19 times. Filter
  `fired_tokens >= 20`, preferably `>=100`.
- Drop zero-fire rows, dedupe repeated neurons, remove always-on candidates
  such as L57-N3872, and cross-check against token selectivity/pole stats.

Local 29-layer staging found 109 selective latents -> 68 distinct candidate
neurons under `top1 >= 0.5` and `20 <= fired_tokens < 20000`; L27 was the
strongest mid-layer hotspot, while the large L57 concentration head collapsed
to the always-on L57-N3872 artifact.

### 4.6 Station House Spaces

- `marin-neuron-hunt`: static Space for explaining all 1.769M Marin MLP
  neurons; card references Marin 32B, the atlas, and `alxcspr/stationhouse`.
- `ConspiracyBench`: 1,543-file static Space plus analyses/experiments; this is
  the separate ConspiracyBench lane, not the Marin data lane.
- `Consciousness-Self-Attribution`: small static site on LLM
  consciousness self-attribution.

## 5. SAE analysis state

- **32k** sweep and 44-layer topup are complete and uploaded; censuses are
  valid for scaled inputs.
- **64k** full sweep is complete and canonical; raw-acts convention is a
  standing gotcha.
- **128k** probes exist only for L45/L57/L63 and did not justify adoption.
- **Token selectivity** full and tail passes are complete, independently
  recomputed, and published.
- **Encoder attribution** v1 is complete but must be used with the control and
  firing-count warnings above.
- **L00/L01/L02 pattern pass** remains useful for readable-family structure,
  but its census-vs-val explanation was superseded by the input-convention root
  cause in `sae/METRICS.md`.

## 6. Standing gotchas

1. Do not compare fvu across 32k/scaled and 64k/raw width families.
2. Do not join `token_idx` or `doc_id` across independent pilot prefixes.
3. "Fires" in sparse-tier analyses means "in the token's top 64"; lower activity
   is not recorded.
4. Exemplar tails are strongest-moment evidence, not typical firing evidence.
5. v1 nominate-enc controls and small-fire nominations are not evidence.
6. Never write or echo HF tokens; upload jobs read them from job environment
   only.
7. For Entire commands on dawn, use the file token store.

## 7. Maintenance

- Refresh the Entire Brain after substantive commits with
  `entire brain refresh --agent none` (full committed-tree/code index) or
  `--worktree` for docs-only work.
- Push this branch to `entire` after commits. `origin` remains the GitHub
  mirror/PR route.
- If a new HF bucket appears, re-read `repo_info(..., files_metadata=True)`
  and update section 4 rather than trusting the stale five-repo note.
