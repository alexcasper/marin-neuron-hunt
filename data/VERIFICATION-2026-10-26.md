# P1 pull verification — 2026-10-26 (ops_sh)

Result: **data/pilot is 100% complete and byte-verified. Nothing is missing.**

## Tally vs MANIFEST.json (per-tree, sha256+bytes manifests)

| tree | manifest entries | on disk | missing | size mismatches | bytes |
|---|---|---|---|---|---|
| fineweb-10m-sparse | 81 | 81 | 0 | 0 | 271.3 GB |
| fineweb-1m | 488 | 488 | 0 | 0 | 107.8 GB |
| stage-a-100k | 488 | 488 | 0 | 0 | 32.8 GB |
| **total** | **1,057** | **1,057** | **0** | **0** | **411.9 GB** (384 GiB) |

(1,057 = 1,054 parquet + 3 provenance.json; the 3 MANIFEST.json files themselves
are not self-listed.)

## Tally vs HF tree (Station-house/marin-32b-neuron-atlas, tree API, 3-page cursor pagination)

- HF `pilot/` subtree: 1,060 files, 411.9 GB (`data/hf_tree.json`)
- Local: same 1,060 paths, 0 missing, 0 extra, 0 byte-size mismatches.

## The "8 missing files" rumor — resolved

1,062 was the count of ALL repo files (1,060 data files + repo `README.md` +
`.gitattributes`). 1,062 − 1,054 parquet = 8 non-parquet files, all present:
3× `MANIFEST.json`, 3× `summaries/provenance.json`, + the 2 repo files.
The pull was never 99% — it was 100%.

## Integrity spot-checks

sha256 verified OK (matches manifest): `fineweb-10m-sparse/captured_mass/seg0000-part-00000.parquet`,
`fineweb-1m/captured_mass/part-00000.parquet`, `stage-a-100k/captured_mass/part-00000.parquet`.

Capture provenance (from fineweb-10m-sparse/MANIFEST.json report): model
marin-community/marin-32b-base@f19b3466, bfloat16, seq 2048, stage B2,
10M tokens (9,995,501), H200, act_recompute_rel_err 0.0024,
hooked_logits identical, 15 dead neurons.

## Disposition

- sh-bs6 closed with this tally.
- APOC transfer: HELD by alex (2026-10-26 directive).
- Active plan: Globus transfer dawn GCP → dusk GCP (`/home/dusk/d/globus/marin-32b/`).
  Dawn copy stays until dusk copy verified.
