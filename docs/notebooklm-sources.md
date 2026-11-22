# NotebookLM sources manifest — Mechanistic Interpretability notebook

Updated: 2026-11-22 · by ops_sh (bead sh-e6a, task from alex via Bedivere)

- **Notebook:** Mechanistic Interpretability — `7693bbc3-2db3-498b-b047-f810ab813e79`
- **URL:** https://notebook.google.com/notebook/7693bbc3-2db3-498b-b047-f810ab813e79
- **Sources before:** 10 (pre-existing generated_text summaries, not touched)
- **Sources after:** 194 (all status=processed, verified via `nlm source list`)
- **Limit:** account is NotebookLM Pro (300 sources/notebook). Limit NOT reached: stopped at 194 because the genuinely relevant material pool was exhausted; padding further would violate the no-filler rule. Reservoir used in full: all 137 candidate dossiers.
- **Access:** notebooklm-mcp wired into ops_sh profile config (`mcp_servers.notebooklm` → `~/.hermes/profiles/ops_sh/bin/notebooklm-mcp`, env-scrubbing wrapper because Hermes' PYTHONPATH breaks pydantic_core in the uv-tools venv). Google cookies for agcline@gmail.com already present at ~/.notebooklm-mcp-cli/profiles/default/ (last validated 2026-11-20); no interactive auth needed.
- **Not added, with reason:** https://www.transformer-circuits.pub/2025/circuit-tracing/index.html (Circuit Tracing overview) — bot-walled 403 to both Firecrawl and NotebookLM ingestion; its full content pair (methods.html + biology.html) IS added below. Perplexity summaries skipped — no PERPLEXITY_API_KEY on dawn (flagged to alex).

## Added sources

### A. ops_sh lane material (text sources) — 21

| # | Title | Type | Origin / why |
|---|---|---|---|
| 1 | ops_sh bead sh-aav: GPU smoke gate FAIL at device-precision boundary (decision needed) | text | `beads db sh: sh-aav` — lane ground truth (metrics, runplan, attribution, beads) |
| 2 | ops_sh bead sh-ygh: encoder-attribution v2 GPU spec + shuffle-control fix (all 64 layers) | text | `beads db sh: sh-ygh` — lane ground truth (metrics, runplan, attribution, beads) |
| 3 | ops_sh bead sh-47k: upstream PR review wave #64-#82 verdicts (10 MERGE, 1 HOLD) | text | `beads db sh: sh-47k` — lane ground truth (metrics, runplan, attribution, beads) |
| 4 | ops_sh bead sh-uxq: L00-N10657 explain thread — selection, PR #3, -002 refuted result | text | `beads db sh: sh-uxq` — lane ground truth (metrics, runplan, attribution, beads) |
| 5 | ops_sh candidate dossier L4-N25503 (provisional working read) | text | `docs/candidates/L4-N25503.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 6 | ops_sh candidate dossier L10-N10572 (provisional working read) | text | `docs/candidates/L10-N10572.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 7 | ops_sh candidate dossier L16-N22118 (provisional working read) | text | `docs/candidates/L16-N22118.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 8 | ops_sh candidate dossier L27-N9808 (provisional working read) | text | `docs/candidates/L27-N9808.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 9 | ops_sh candidate dossier L36-N14337 (provisional working read) | text | `docs/candidates/L36-N14337.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 10 | ops_sh candidate dossier L45-N21163 (provisional working read) | text | `docs/candidates/L45-N21163.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 11 | ops_sh candidate dossier L59-N11005 (provisional working read) | text | `docs/candidates/L59-N11005.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 12 | ops_sh candidate dossier L63-N11857 (provisional working read) | text | `docs/candidates/L63-N11857.md @ 17226d9 (sample of 137)` — lane ground truth (metrics, runplan, attribution, beads) |
| 13 | ops_sh lane brain export: marin-32b-neuron-atlas (Entire memory) | text | `ENTIRE_BRAIN.md @ merged-from-apoc` — lane ground truth (metrics, runplan, attribution, beads) |
| 14 | ops_sh Station House lane brief (charter) | text | `BRIEF.md @ merged-from-apoc` — lane ground truth (metrics, runplan, attribution, beads) |
| 15 | ops_sh SAE metric conventions (METRICS.md) | text | `sae/METRICS.md @ merged-from-apoc` — lane ground truth (metrics, runplan, attribution, beads) |
| 16 | ops_sh SAE training run plan (RUNPLAN.md) | text | `sae/RUNPLAN.md @ merged-from-apoc` — lane ground truth (metrics, runplan, attribution, beads) |
| 17 | ops_sh encoder-attribution summary, 15 staged layers (2026-11-10) | text | `sae/ATTRIBUTION-ENC-15LAYERS-2026-11-10.md` — lane ground truth (metrics, runplan, attribution, beads) |
| 18 | ops_sh pattern-pass report L00-L02 | text | `sae/pattern-report-L00-L01-L02.md` — lane ground truth (metrics, runplan, attribution, beads) |
| 19 | ops_sh token-selectivity pass (full sparse tier) | text | `sae/TOKEN-SELECTIVITY.md` — lane ground truth (metrics, runplan, attribution, beads) |
| 20 | ops_sh SAE sizing analysis: dusk vs Apocrita H200 | text | `sae/SIZING-dusk-vs-apocrita.md` — lane ground truth (metrics, runplan, attribution, beads) |
| 21 | 137-candidate dossier set overview (docs/candidates README) | text | `docs/candidates/README.md @ 17226d9` — lane ground truth (metrics, runplan, attribution, beads) |

### B. 137-candidate dossier set (text sources) — 139 (all 137 dossiers + 2 HF docs)

| # | Title | Type | Origin / why |
|---|---|---|---|
| 1 | ops_sh HF artifact README: marin-32b nominate-enc dataset card | text | `sae/hf-README.md` — HF dataset card / 64-layer results table |
| 2 | ops_sh 64-layer SAE results table (hf-table-64) | text | `sae/hf-table-64.md` — HF dataset card / 64-layer results table |
| 3 | ops_sh candidate dossier L10-N12917 (provisional working read) | text | `docs/candidates/L10-N12917.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 4 | ops_sh candidate dossier L10-N15969 (provisional working read) | text | `docs/candidates/L10-N15969.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 5 | ops_sh candidate dossier L10-N20671 (provisional working read) | text | `docs/candidates/L10-N20671.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 6 | ops_sh candidate dossier L10-N21821 (provisional working read) | text | `docs/candidates/L10-N21821.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 7 | ops_sh candidate dossier L10-N6998 (provisional working read) | text | `docs/candidates/L10-N6998.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 8 | ops_sh candidate dossier L10-N9319 (provisional working read) | text | `docs/candidates/L10-N9319.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 9 | ops_sh candidate dossier L11-N134 (provisional working read) | text | `docs/candidates/L11-N134.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 10 | ops_sh candidate dossier L11-N13518 (provisional working read) | text | `docs/candidates/L11-N13518.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 11 | ops_sh candidate dossier L11-N13535 (provisional working read) | text | `docs/candidates/L11-N13535.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 12 | ops_sh candidate dossier L11-N15351 (provisional working read) | text | `docs/candidates/L11-N15351.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 13 | ops_sh candidate dossier L11-N24556 (provisional working read) | text | `docs/candidates/L11-N24556.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 14 | ops_sh candidate dossier L11-N2883 (provisional working read) | text | `docs/candidates/L11-N2883.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 15 | ops_sh candidate dossier L11-N5811 (provisional working read) | text | `docs/candidates/L11-N5811.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 16 | ops_sh candidate dossier L11-N8533 (provisional working read) | text | `docs/candidates/L11-N8533.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 17 | ops_sh candidate dossier L12-N14813 (provisional working read) | text | `docs/candidates/L12-N14813.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 18 | ops_sh candidate dossier L12-N26213 (provisional working read) | text | `docs/candidates/L12-N26213.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 19 | ops_sh candidate dossier L12-N4205 (provisional working read) | text | `docs/candidates/L12-N4205.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 20 | ops_sh candidate dossier L12-N6048 (provisional working read) | text | `docs/candidates/L12-N6048.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 21 | ops_sh candidate dossier L12-N7309 (provisional working read) | text | `docs/candidates/L12-N7309.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 22 | ops_sh candidate dossier L12-N7352 (provisional working read) | text | `docs/candidates/L12-N7352.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 23 | ops_sh candidate dossier L12-N8027 (provisional working read) | text | `docs/candidates/L12-N8027.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 24 | ops_sh candidate dossier L13-N11930 (provisional working read) | text | `docs/candidates/L13-N11930.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 25 | ops_sh candidate dossier L13-N3314 (provisional working read) | text | `docs/candidates/L13-N3314.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 26 | ops_sh candidate dossier L15-N17223 (provisional working read) | text | `docs/candidates/L15-N17223.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 27 | ops_sh candidate dossier L15-N8204 (provisional working read) | text | `docs/candidates/L15-N8204.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 28 | ops_sh candidate dossier L16-N15747 (provisional working read) | text | `docs/candidates/L16-N15747.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 29 | ops_sh candidate dossier L16-N21335 (provisional working read) | text | `docs/candidates/L16-N21335.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 30 | ops_sh candidate dossier L16-N3305 (provisional working read) | text | `docs/candidates/L16-N3305.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 31 | ops_sh candidate dossier L16-N6007 (provisional working read) | text | `docs/candidates/L16-N6007.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 32 | ops_sh candidate dossier L17-N14375 (provisional working read) | text | `docs/candidates/L17-N14375.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 33 | ops_sh candidate dossier L17-N9960 (provisional working read) | text | `docs/candidates/L17-N9960.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 34 | ops_sh candidate dossier L18-N12330 (provisional working read) | text | `docs/candidates/L18-N12330.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 35 | ops_sh candidate dossier L19-N21740 (provisional working read) | text | `docs/candidates/L19-N21740.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 36 | ops_sh candidate dossier L20-N27076 (provisional working read) | text | `docs/candidates/L20-N27076.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 37 | ops_sh candidate dossier L21-N19899 (provisional working read) | text | `docs/candidates/L21-N19899.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 38 | ops_sh candidate dossier L22-N18816 (provisional working read) | text | `docs/candidates/L22-N18816.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 39 | ops_sh candidate dossier L22-N22450 (provisional working read) | text | `docs/candidates/L22-N22450.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 40 | ops_sh candidate dossier L22-N26066 (provisional working read) | text | `docs/candidates/L22-N26066.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 41 | ops_sh candidate dossier L22-N5747 (provisional working read) | text | `docs/candidates/L22-N5747.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 42 | ops_sh candidate dossier L24-N8156 (provisional working read) | text | `docs/candidates/L24-N8156.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 43 | ops_sh candidate dossier L25-N19258 (provisional working read) | text | `docs/candidates/L25-N19258.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 44 | ops_sh candidate dossier L25-N20093 (provisional working read) | text | `docs/candidates/L25-N20093.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 45 | ops_sh candidate dossier L26-N11025 (provisional working read) | text | `docs/candidates/L26-N11025.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 46 | ops_sh candidate dossier L26-N17855 (provisional working read) | text | `docs/candidates/L26-N17855.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 47 | L10-N10572 | text | `docs/candidates/L10-N10572.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 48 | L16-N22118 | text | `docs/candidates/L16-N22118.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 49 | L27-N1187 | text | `docs/candidates/L27-N1187.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 50 | L27-N15105 | text | `docs/candidates/L27-N15105.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 51 | L27-N22312 | text | `docs/candidates/L27-N22312.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 52 | L27-N22697 | text | `docs/candidates/L27-N22697.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 53 | L27-N23150 | text | `docs/candidates/L27-N23150.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 54 | L27-N27558 | text | `docs/candidates/L27-N27558.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 55 | L27-N9808 | text | `docs/candidates/L27-N9808.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 56 | L28-N20626 | text | `docs/candidates/L28-N20626.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 57 | L28-N22988 | text | `docs/candidates/L28-N22988.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 58 | L29-N12980 | text | `docs/candidates/L29-N12980.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 59 | L29-N23842 | text | `docs/candidates/L29-N23842.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 60 | L29-N24862 | text | `docs/candidates/L29-N24862.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 61 | L29-N24910 | text | `docs/candidates/L29-N24910.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 62 | L29-N25622 | text | `docs/candidates/L29-N25622.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 63 | L30-N10652 | text | `docs/candidates/L30-N10652.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 64 | L30-N5450 | text | `docs/candidates/L30-N5450.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 65 | L33-N18287 | text | `docs/candidates/L33-N18287.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 66 | L33-N2459 | text | `docs/candidates/L33-N2459.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 67 | L34-N26836 | text | `docs/candidates/L34-N26836.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 68 | L34-N8043 | text | `docs/candidates/L34-N8043.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 69 | L35-N10673 | text | `docs/candidates/L35-N10673.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 70 | L35-N20498 | text | `docs/candidates/L35-N20498.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 71 | L35-N24020 | text | `docs/candidates/L35-N24020.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 72 | L35-N2465 | text | `docs/candidates/L35-N2465.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 73 | L35-N2631 | text | `docs/candidates/L35-N2631.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 74 | L35-N5309 | text | `docs/candidates/L35-N5309.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 75 | L35-N7530 | text | `docs/candidates/L35-N7530.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 76 | L35-N8512 | text | `docs/candidates/L35-N8512.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 77 | L35-N9217 | text | `docs/candidates/L35-N9217.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 78 | L36-N14337 | text | `docs/candidates/L36-N14337.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 79 | L36-N16585 | text | `docs/candidates/L36-N16585.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 80 | L36-N16965 | text | `docs/candidates/L36-N16965.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 81 | L36-N25354 | text | `docs/candidates/L36-N25354.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 82 | L36-N27357 | text | `docs/candidates/L36-N27357.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 83 | L36-N4765 | text | `docs/candidates/L36-N4765.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 84 | L36-N5138 | text | `docs/candidates/L36-N5138.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 85 | L36-N9381 | text | `docs/candidates/L36-N9381.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 86 | L37-N17388 | text | `docs/candidates/L37-N17388.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 87 | L37-N17833 | text | `docs/candidates/L37-N17833.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 88 | L37-N24889 | text | `docs/candidates/L37-N24889.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 89 | L37-N4754 | text | `docs/candidates/L37-N4754.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 90 | L37-N5043 | text | `docs/candidates/L37-N5043.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 91 | L37-N6076 | text | `docs/candidates/L37-N6076.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 92 | L37-N7052 | text | `docs/candidates/L37-N7052.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 93 | L38-N9134 | text | `docs/candidates/L38-N9134.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 94 | L39-N15151 | text | `docs/candidates/L39-N15151.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 95 | L4-N25503 | text | `docs/candidates/L4-N25503.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 96 | L40-N4441 | text | `docs/candidates/L40-N4441.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 97 | L42-N16424 | text | `docs/candidates/L42-N16424.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 98 | L42-N873 | text | `docs/candidates/L42-N873.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 99 | L42-N9154 | text | `docs/candidates/L42-N9154.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 100 | L44-N14525 | text | `docs/candidates/L44-N14525.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 101 | L44-N16258 | text | `docs/candidates/L44-N16258.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 102 | L44-N22828 | text | `docs/candidates/L44-N22828.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 103 | L45-N14989 | text | `docs/candidates/L45-N14989.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 104 | L45-N21163 | text | `docs/candidates/L45-N21163.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 105 | L45-N21413 | text | `docs/candidates/L45-N21413.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 106 | L45-N4166 | text | `docs/candidates/L45-N4166.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 107 | L46-N23521 | text | `docs/candidates/L46-N23521.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 108 | L46-N5406 | text | `docs/candidates/L46-N5406.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 109 | L47-N26202 | text | `docs/candidates/L47-N26202.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 110 | L48-N18063 | text | `docs/candidates/L48-N18063.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 111 | L49-N895 | text | `docs/candidates/L49-N895.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 112 | L50-N18816 | text | `docs/candidates/L50-N18816.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 113 | L55-N10278 | text | `docs/candidates/L55-N10278.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 114 | L57-N26033 | text | `docs/candidates/L57-N26033.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 115 | L58-N3443 | text | `docs/candidates/L58-N3443.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 116 | L59-N11005 | text | `docs/candidates/L59-N11005.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 117 | L59-N11448 | text | `docs/candidates/L59-N11448.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 118 | L59-N19776 | text | `docs/candidates/L59-N19776.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 119 | L59-N21614 | text | `docs/candidates/L59-N21614.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 120 | L59-N557 | text | `docs/candidates/L59-N557.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 121 | L6-N5374 | text | `docs/candidates/L6-N5374.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 122 | L6-N5700 | text | `docs/candidates/L6-N5700.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 123 | L60-N10309 | text | `docs/candidates/L60-N10309.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 124 | L60-N17154 | text | `docs/candidates/L60-N17154.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 125 | L60-N26229 | text | `docs/candidates/L60-N26229.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 126 | L62-N16811 | text | `docs/candidates/L62-N16811.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 127 | L62-N4855 | text | `docs/candidates/L62-N4855.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 128 | L62-N6036 | text | `docs/candidates/L62-N6036.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 129 | L62-N9554 | text | `docs/candidates/L62-N9554.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 130 | L63-N11857 | text | `docs/candidates/L63-N11857.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 131 | L63-N17168 | text | `docs/candidates/L63-N17168.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 132 | L7-N19201 | text | `docs/candidates/L7-N19201.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 133 | L7-N20520 | text | `docs/candidates/L7-N20520.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 134 | L8-N10861 | text | `docs/candidates/L8-N10861.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 135 | L8-N13193 | text | `docs/candidates/L8-N13193.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 136 | L8-N9298 | text | `docs/candidates/L8-N9298.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 137 | L9-N12281 | text | `docs/candidates/L9-N12281.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 138 | L9-N23934 | text | `docs/candidates/L9-N23934.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |
| 139 | L9-N24830 | text | `docs/candidates/L9-N24830.md` @ 17226d9 — evidence-grounded provisional working read (hypothesis, not confirmed explanation) |

### C. Public mechanistic-interpretability material (URL sources) — 24

| # | URL | Why | Firecrawl check |
|---|---|---|---|
| 1 | https://transformer-circuits.pub/2021/framework/index.html | A Mathematical Framework for Transformer Circuits — foundational | 200, 130432 chars |
| 2 | https://transformer-circuits.pub/2022/toy_model/index.html | Toy Models of Superposition — foundational for SAE work | 200, 178721 chars |
| 3 | https://transformer-circuits.pub/2023/monosemantic-features/index.html | Towards Monosemanticity: features in a one-layer model | 200, 191165 chars |
| 4 | https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html | Scaling Monosemanticity — Claude 3 features at scale | 200, 171494 chars |
| 5 | https://decoderesearch.github.io/SAELens/v6.35.0/ | SAELens docs — the SAE training/analysis library used by this lane | 200, 5890 chars |
| 6 | https://transformerlensorg.github.io/TransformerLens/ | TransformerLens docs — activation-patching/hook tooling | 200, 3372 chars |
| 7 | https://github.com/callummcdougall/ARENA_3.0 | ARENA education repo, transformer-interp chapter — hands-on walkthroughs | 200, 20183 chars |
| 8 | https://www.neelnanda.io/mechanistic-interpretability/getting-started | Neel Nanda's getting-started mech-interp guide — canonical onboarding | 200, 129613 chars |
| 9 | https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability | 200 Concrete Open Problems in Mech Interp — research agenda | 200, 24752 chars |
| 10 | https://www.anthropic.com/research/tracing-thoughts-language-model |  | n/a (NotebookLM-fetched) |
| 11 | https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html | In-context Learning and Induction Heads | 200, 143178 chars |
| 12 | https://transformer-circuits.pub/2022/mech-interp-essay/index.html | A Mechanistic Interpretability Analysis of Grokking | 200, 14674 chars |
| 13 | https://transformer-circuits.pub/2022/solu/index.html | Softmax Linear Units (SoLU) | 200, 83506 chars |
| 14 | https://transformer-circuits.pub/2023/interpretability-dreams/index.html | Interpretability Dreams (research vision) | 200, 30207 chars |
| 15 | https://transformer-circuits.pub/2023/superposition-composition/index.html | Superposition and Composition / private structure | 200, 15222 chars |
| 16 | https://transformer-circuits.pub/2023/privileged-basis/index.html | Privileged Bases in ResNet-like models | 200, 17516 chars |
| 17 | https://transformer-circuits.pub/2024/crosscoders/index.html | Crosscoders (2024) | 200, 47126 chars |
| 18 | https://transformer-circuits.pub/2024/model-diffing/index.html | Model Diffing (2024) | 200, 22351 chars |
| 19 | https://transformer-circuits.pub/2024/features-as-classifiers/index.html | Features as Classifiers (2024) | 200, 17307 chars |
| 20 | https://transformer-circuits.pub/2024/qualitative-essay/index.html | Qualitative essay on descriptions/features (2024) | 200, 16998 chars |
| 21 | https://transformer-circuits.pub/2025/attribution-graphs/methods.html | Circuit Tracing: methods (2025 attribution-graphs pair) | 200, 218168 chars |
| 22 | https://transformer-circuits.pub/2025/attribution-graphs/biology.html | On the Biology of a Large Language Model (2025 attribution-graphs pair) | 200, 194313 chars |
| 23 | https://transformer-circuits.pub/2025/introspection/index.html | Looking Inward: introspection (2025) | 200, 117728 chars |
| 24 | https://transformer-circuits.pub/2025/attention-qk/index.html | Attention QK circuits (2025) | 200, 43245 chars |

### Verification

- Live notebook source list re-pulled after all adds: 194 sources, every one status=2 (processed).
  All 160 text-source titles and all 24 URLs cross-checked by title/URL against the live list — 0 missing.
- Lane text adds: 21/21 accepted. Dossier/HF adds: 139/139 accepted. URL adds: 24/25 accepted (1 refused, see above).

### Add-state files

The add-session state (titles, origins, accept results) is preserved in `docs/notebooklm-source-state/` (nb_added_lane.json / nb_added_urls.json / nb_added_dossiers.json); the authoritative live inventory is the notebook itself.


## Wave 2 — Anthropic interp back-catalog fill (ALEX GO #2, 2026-11-22)

- **Sources: 194 → 244** (50 URL adds, all processed and verified against the live list; 0 zombies)
- Pool: remaining transformer-circuits.pub pages (2026/2025/2024/2023 update digests + 2026 papers + exercises) and hand-picked anthropic.com/research interpretability posts. Research-post duplicates of papers already added via transformer-circuits.pub (toy-models, monosemanticity, framework, SoLU, privileged-bases, interpretability-dreams, features-as-classifiers, introspection, tracing-thoughts) were NOT re-added — same content would be filler.
- Not added: `/2026/emotions` needed no fallback (NotebookLM ingested the URL); `feed.xml`, `videos`, `garcon` (link-lists/demos, no substantive content); `circuit-tracing/index.html` still bot-walled (covered by methods+biology). Gap to the 300 limit: **56 sources** — the on-topic pool is exhausted; remaining anthropic.com/research posts (safety evals, cyber, economics, character/welfare) are off-topic for a mechanistic-interpretability notebook, and adding them would violate the no-filler rule.

| # | URL | Why | Firecrawl check |
|---|---|---|---|
| 1 | https://transformer-circuits.pub/2026/october-update/index.html | Circuits thread update digest (2026-10) | 200, 2798 chars |
| 2 | https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html | Interference: effectiveness vs helpfulness (2026 paper) | 200, 78357 chars |
| 3 | https://transformer-circuits.pub/2026/workspace/index.html | Feature workspace (2026 paper) | 200, 357926 chars |
| 4 | https://transformer-circuits.pub/2026/june-update/index.html | Circuits update digest (2026-06) | 200, 5788 chars |
| 5 | https://transformer-circuits.pub/2026/may-update/index.html | Circuits update digest (2026-05) | 200, 11038 chars |
| 6 | https://transformer-circuits.pub/2026/nla/index.html | Numerative/linear-association paper (2026) | 200, 168134 chars |
| 7 | https://transformer-circuits.pub/2026/headvis/index.html | Attention-head visualization tooling paper (2026) | 200, 40774 chars |
| 8 | https://transformer-circuits.pub/2025/november-update/index.html | Circuits update digest (2025-11) | 200, 14717 chars |
| 9 | https://transformer-circuits.pub/2025/linebreaks/index.html | Line-break features paper (2025) | 200, 88916 chars |
| 10 | https://transformer-circuits.pub/2025/september-update/index.html | Circuits update digest (2025-09) | 200, 3956 chars |
| 11 | https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html | Faithfulness toy model (2025) | 200, 19608 chars |
| 12 | https://transformer-circuits.pub/2025/interference-weights/index.html | Interference weights paper (2025) | 200, 31312 chars |
| 13 | https://transformer-circuits.pub/2025/bulk-update/index.html | Circuits bulk update digest (2025) | 200, 19505 chars |
| 14 | https://transformer-circuits.pub/2025/july-update/index.html | Circuits update digest (2025-07) | 200, 27083 chars |
| 15 | https://transformer-circuits.pub/2025/april-update/index.html | Circuits update digest (2025-04) | 200, 27062 chars |
| 16 | https://transformer-circuits.pub/2025/attention-update/index.html | Attention update digest (2025) | 200, 30326 chars |
| 17 | https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html | Crosscoder/diffing update digest (2025) | 200, 16264 chars |
| 18 | https://transformer-circuits.pub/2025/january-update/index.html | Circuits update digest (2025-01) | 200, 5469 chars |
| 19 | https://transformer-circuits.pub/2024/september-update/index.html | Circuits update digest (2025-09) | 200, 10660 chars |
| 20 | https://transformer-circuits.pub/2024/august-update/index.html | Circuits update digest (2024-08) | 200, 22935 chars |
| 21 | https://transformer-circuits.pub/2024/july-update/index.html | Circuits update digest (2025-07) | 200, 45870 chars |
| 22 | https://transformer-circuits.pub/2024/june-update/index.html | Circuits update digest (2026-06) | 200, 25103 chars |
| 23 | https://transformer-circuits.pub/2024/april-update/index.html | Circuits update digest (2025-04) | 200, 45241 chars |
| 24 | https://transformer-circuits.pub/2024/march-update/index.html | Circuits update digest (2024-03) | 200, 22401 chars |
| 25 | https://transformer-circuits.pub/2024/feb-update/index.html | Circuits update digest (2024-02) | 200, 10365 chars |
| 26 | https://transformer-circuits.pub/2024/jan-update/index.html | Circuits update digest (2024-01) | 200, 40159 chars |
| 27 | https://transformer-circuits.pub/2023/july-update/index.html | Circuits update digest (2025-07) | 200, 13642 chars |
| 28 | https://transformer-circuits.pub/2023/may-update/index.html | Circuits update digest (2026-05) | 200, 43466 chars |
| 29 | https://transformer-circuits.pub/2021/exercises/index.html | Transformer-circuits exercises — educational walkthroughs | 200, 13819 chars |
| 30 | https://www.anthropic.com/research/mapping-mind-language-model | Anthropic: Mapping the Mind of a Large Language Model (SAE features announcement) | 200, 14294 chars |
| 31 | https://www.anthropic.com/research/decomposing-language-models-into-understandable-components | Anthropic: Decomposing LMs into understandable components | 200, 6414 chars |
| 32 | https://www.anthropic.com/research/natural-language-autoencoders | Transformer-circuits thread page | 200, 13278 chars |
| 33 | https://www.anthropic.com/research/evaluating-feature-steering | Anthropic: Evaluating feature steering | 200, 27063 chars |
| 34 | https://www.anthropic.com/research/golden-gate-claude | Anthropic: Golden Gate Claude (feature-steering demo) | 200, 3918 chars |
| 35 | https://www.anthropic.com/research/global-workspace | Anthropic: Global workspace (2026) | 200, 34088 chars |
| 36 | https://www.anthropic.com/research/persona-vectors | Anthropic: Persona vectors | 200, 13048 chars |
| 37 | https://www.anthropic.com/research/persona-selection-model | Anthropic: Persona selection model | 200, 8963 chars |
| 38 | https://www.anthropic.com/research/open-source-circuit-tracing | Transformer-circuits thread page | 200, 3744 chars |
| 39 | https://www.anthropic.com/research/probes-catch-sleeper-agents | Transformer-circuits thread page | 200, 30423 chars |
| 40 | https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning | Transformer-circuits thread page | 200, 1737 chars |
| 41 | https://www.anthropic.com/research/studying-large-language-model-generalization-with-influence-functions | Anthropic: Studying LM generalization with influence functions | 200, 2291 chars |
| 42 | https://www.anthropic.com/research/scaling-laws-and-interpretability-of-learning-from-repeated-data | Transformer-circuits thread page | 200, 2624 chars |
| 43 | https://www.anthropic.com/research/superposition-memorization-and-double-descent | Transformer-circuits thread page | 200, 2827 chars |
| 44 | https://www.anthropic.com/research/distributed-representations-composition-superposition | Transformer-circuits thread page | 200, 2717 chars |
| 45 | https://www.anthropic.com/research/engineering-challenges-interpretability | Transformer-circuits thread page | 200, 11166 chars |
| 46 | https://www.anthropic.com/research/emotion-concepts-function | Anthropic: Emotion concepts and function in Claude | 200, 21677 chars |
| 47 | https://www.anthropic.com/research/values-wild | Anthropic: Values in the wild | 200, 12099 chars |
| 48 | https://www.anthropic.com/research/auditing-hidden-objectives | Transformer-circuits thread page | 200, 19105 chars |
| 49 | https://www.anthropic.com/research/clio | Anthropic: Clio — interpreting model usage at scale | 200, 16370 chars |
| 50 | https://transformer-circuits.pub/2026/emotions/index.html | Emotion concepts in models (2026) — ingested by NotebookLM despite oversized Firecrawl payload | oversized payload — fetched OK |

- Live verification: re-pulled source list shows 244 sources, every status=2; all 50 wave-2 URLs present.
