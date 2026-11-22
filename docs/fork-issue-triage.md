# Fork issue triage — alexcasper/marin-neuron-hunt (open issues, 2026-11-22)

Per alex's 2026-11-22 directive (recorded on sh-47k): triage every open issue on OUR fork,
verify the claims raised against repo/code/data, and record verdicts with evidence here.
**No upstream comments or pushes** (standing freeze). Bead: sh-1yx (fork triage wave).

Scope: 151 open issues = 137 intake mirrors (#15–151) + 14 others (#1–14).

Methods: local ground truth (sae/candidates-68.json, data/derived/candidate_wave/*.json,
data/pilot/fineweb-10m-sparse/summaries/{pole_stats,logit_lens,exemplars}.parquet,
attribution_enc_L57.parquet, characterization-L57.md, marin-community/marin-32b-base@f19b3466
tokenizer); GitHub API read-only for fork branches/commits/check-runs and upstream
Station-House/marin-neuron-hunt scripts/results (read-only, no writes anywhere).

## Intake mirrors #15–151: 137/137 VERIFIED

Each issue cross-checked on 7 points, all passing:
1. Title L<layer>-N<neuron> present in sae/candidates-68.json (137/137 match the committed table).
2. `data/derived/candidate_wave/<id>.json` exists (140 local artifacts; all 137 covered).
3. "contributing source latents: N" matches the table's n_latents.
4. "max top1_share:" matches the table.
5. Every per-latent detail string `latent/fired/top1_share/eff_neurons` present verbatim.
6. The issue's pole_stats line (n/mean/std/frac_pos/max/min/scale_abs_p999) equals the wave JSON.
7. Wave JSON equals the ground-truth pole_stats.parquet row (tolerance = 4-significant-figure
   rounding, the format the issues print).

Evidence: verifier state in `docs/fork-issue-state/verdicts_intake.json` (per-issue id +
checks). No discrepancies found.

## Non-intake issues

| # | Title | Verdict | Evidence |
|---|---|---|---|
| 14 | Encoder-attribution wave tracker [sh-8wt] | VERIFIED (tracker only) | Body makes no factual claims; all 137 mirrors present and labeled hunt-candidate-wave; bead sh-8wt + children open in lane tracker. |
| 13 | CI lessons from closed explain PRs | VERIFIED (fork gap noted) | Tokenizer facts reproduced with marin-community/marin-32b-base@f19b3466: ` pride`=1 token, `pride`=2 (`pr`+`ide`), ` stethoscope`=3 (` st`+`eth`+`oscope`), ` doctor`=1. Upstream scripts/ has check_design.py, check_tokens.py, design_session_note.py; check_design.py enforces all six stages (Claim / Unit and metric / Controls / Data / Noise / Readout — all present in source). NOTE: our fork's scripts/ carries only validate.py + check_claims.py — the "reproduce the design check locally" workflow does NOT work on the fork today. Marker-semantics claims ([[ word]] vs [[word]]) are upstream CI behavior, consistent with the recorded fixes (9b02252/143c92f). |
| 12 | [archive PR #30] L57-N17606 | VERIFIED (historical), now superseded | Branch l57-N17606 exists; commit c95bc78 real, adds requests/L57-N17606-001.yaml; both files present at branch head. CI history at c95bc78: design=FAILURE run 2026-11-16 10:14Z (matches "design check fails"), then a re-run 10:19Z with ALL FIVE checks success — current latest status at the head is all green. The "open at close: design fails" statement describes close-time state, since superseded. |
| 11 | [archive PR #29] L57-N10940 | VERIFIED | Commit 9b02252 real (tokens fix, requests/L57-N10940-001.yaml); "all CI green at head": latest runs at 9b02252 (2026-11-16 10:19Z) = validate/design/tokens/runner/claims ALL success ✓ (earlier runs 10:14Z had design failure; superseded). Branch head has advanced past 9b02252 (now 866afa2) — noted, harmless; both files present. |
| 10 | [archive PR #28] L57-N5873 | VERIFIED (historical), now superseded | Branch l57-N5873 @ e12155c exactly as cited; both files at head. CI history: design failure run 2026-11-16 10:14Z, re-run 10:19Z all-success. Same supersession pattern as #12. |
| 9 | [archive PR #27] L57-N25368 | VERIFIED | Commit 143c92f real (stethoscope→doctor swap in requests/L57-N25368-001.yaml); "all CI green at head": latest runs all five success ✓. ` doctor`=1 token confirmed with the marin tokenizer. |
| 8 | [archive PR #26] L57-N2462 | VERIFIED | Branch l57-N2462; cited head 230f53c real (requests/L57-N2462-001.yaml); both files at head; latest check-runs at 230f53c all five success ("all CI green at head" ✓). |
| 7 | [archive PR #3] L0-N10657 | VERIFIED (one imprecision) | Against upstream results files (read-only): -001 refuted ✓ (should_fire_above 0.0833=1/12, should_not_fire_below 1.0=12/12, AUROC 0.854=0.85417, only ` finite` 1.015625 cleared 0.9822, common lowercase 0.78–0.96 ✓). -002 refuted ✓ (should_fire_above 1.0=12/12, should_not_fire_below 0.5833=7/12, AUROC 0.917=0.91667, ALL-CAPS .56–.63 ✓ [IMAGE .5625, SIMPLE .6328], proper nouns .42–.46 ✓ [Tribune .4648, County .416], ` serene` .79=.7852 ✓). IMPRECISION: "Title case (.76–.82)" — actual Title-case range is .61–.82 (Image .6055 and Transform .7188 fall below the quoted band); does not change the refuted outcome. "0.75 chosen after seeing -001 numbers" is reviewer attribution — UNVERIFIABLE. Fix commit 8f08501 (explanations/L0/L0-N10657.yaml) exists on the fork ✓. |
| 6 | L57-N3872 always-on magnitude neuron | VERIFIED | pole_stats (57,3872): frac_pos 1.0 ✓, min +3.8281 ✓ ("+3.83"), pos_q99 174.6576 ✓ ("174.66"). Attribution L57: top1_share>=0.5 set = 40 latents; characterization-L57.md lists 30 distinct latents whose top1 neuron is N3872 ✓ ("30/40"). Firing ≈60%: 120,598 fired / 200,704 atlas tokens = 60.1% ✓. Excluded from PRs #26–30 ✓ (those feature N2462/5873/10940/25368/17606). |
| 5 | L57-N10275 debt-relief spam | PARTIALLY REFUTED | Attribution ✓ (latent 7840 → N10275, top1_share 0.511, fired 1); max act 1000.0 ✓; + exemplars debt-relief/bankruptcy spam, acts 612–1000 ✓; − exemplars French infobox dates −21.75 ✓ ("−21"); logit lens pushes -de/ DE/ De/ dé (5.34/4.88/4.81/4.59) ✓. **REFUTED:** "act 1000 — the single largest activation in the set" — within the same 40-latent characterized set, N5873's max act is **1144 > 1000** (next: N10275 1000, N25368 652). N10275 is the largest among the five write-up neurons, not the set. |
| 4 | L57-N26059 sand metaphors | VERIFIED (minor tokenization slip) | Attribution ✓ (27244, 0.516, 1 fired); − exemplars quicksand sand-metaphor spam ✓ (doc: `pit of [[ quick]]sand`, acts −159/−136 — the issue writes `[[quicks]] and`, a marker-split imprecision; phenomenon and acts match); + lens pushes ` Sandy`/` sandbox` ✓ (3.62/3.11 — issue's "(3.62)" ✓); pole min −159 ✓. |
| 3 | L57-N26751 suppresses sens- | VERIFIED | Attribution ✓ (1634, 0.543, 3 fired); − lens suppresses ` Sens`/` sens` ✓ (−11.31/−10.62); − exemplars many/physical/extensive ✓; + exemplars Sens-ored / GPS ✓. |
| 2 | L57-N14976 promote/promo spam (− side) | VERIFIED | Attribution ✓ (29546, 0.564, 2 fired); − exemplars: rank0 act −972 "dedicated [[ to]] advertising, shielding, and progressing" ✓, −724 row present ✓; − lens suppresses ` prom` −10.50 ✓; frac_pos 0.469 ✓, min −972 ✓. "('+ 'a' in anatomy/Genesis)": Genesis + exemplar confirmed ✓; "anatomy" not found in the doc section (minor quote imprecision; + side genuinely thin, as the issue itself says). |
| 1 | L57-N26033 split '-f' fragments | VERIFIED | Attribution ✓ (6377, 0.504, 119 fired — densest non-N3872 candidate ✓: next-highest fired among top1≥0.5 latents is smaller); lens: + pushes `-f` 9.62 / `_f` 6.12 / `F` 6.09 ✓, − suppresses ` Fib` −6.12 ✓; pos_q99 13.0975 ✓ ("13.10"); frac_pos 0.0532 ✓ ("0.053"); + exemplars '-f'-fragment SEO spam, acts within 192–258, max 258 ✓. |

## Tally

- VERIFIED: 149 issues (137 mirrors + 12 non-intake)
- VERIFIED with recorded imprecision/caveat: #7, #10, #12 (see table)
- PARTIALLY REFUTED: #5 (the "largest activation in the set" superlative is false — N5873 1144 > N10275 1000)
- REFUTED: none outright
- UNVERIFIABLE sub-claims only: #7's reviewer-intent sentence

## Actionable fork-side notes (for us, no upstream contact)

1. Fork scripts/ lacks check_design.py + check_tokens.py (upstream-only) — the #13 local-replay
   workflow can't run on the fork. If recreated PRs are ever built here, port those two scripts.
2. #5's superlative should be corrected if that write-up is ever filed (N5873 1144 > N10275 1000
   within the same set).
3. #9/#11/#10/#12 archive CI descriptions mix close-time status with later re-runs; any recreated
   PR should re-run CI fresh rather than trust the archived per-check conclusions.

## Method/precision notes

- Numeric comparisons use 4-significant-figure rounding tolerance (the format issues print);
  equality of wave JSON ↔ pole_stats parquet verified independently of the printed lines.
- CI statuses taken as the latest check-run per check name at the exact cited SHA
  (ordered by started_at; the GitHub API returns newest-first, so naive list-order picks invert this).
