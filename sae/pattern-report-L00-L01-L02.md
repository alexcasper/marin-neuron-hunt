# L00/L01/L02 pattern pass — sae-64k width sweep, first three layers

Source: per-latent interp pass over a 204,800-token spread of the atlas stream, CPU-only (job 35521210), encode-side only. Artifacts: `sae-64k/layer=0{0,1,2}/interp/{patterns.jsonl,families.json,notable.json}` on APOC scratch, mirrored to `/mnt/d/env/apoc/marin32b/interp-pattern-3L/`. Nothing pushed to HF/GitHub.

## Pattern-family breakdown (share of alive latents)

| family | L00 (30,089 alive) | L01 (12,549 alive) | L02 (23,621 alive) |
|---|---|---|---|
| token-identity | 5,968 (19.8%) | 213 (1.7%) | 437 (1.9%) |
| punctuation/whitespace | 256 (0.9%) | 6 (0.05%) | 13 (0.06%) |
| n-gram | 3,548 (11.8%) | 141 (1.1%) | 397 (1.7%) |
| doc-identity | 6,021 (20.0%) | 828 (6.6%) | 1,795 (7.6%) |
| position | 0 | 0 | 0 |
| other | 14,296 (47.5%) | 11,361 (90.5%) | 20,979 (88.8%) |

(Caveat carried from the pass: forms/n-grams are word-level approximations — the atlas provides positions+text, not token ids.)

## Cross-layer comparison

- **L00 is a different regime.** ~52% of its alive latents carry human-readable structure (token/punct/n-gram/doc) vs ~9% (L01) and ~11% (L02). L01/L02 alive dictionaries are ~90% "other" — features that didn't resolve to a clean surface pattern in this pass, not necessarily uninterpretable, but nothing readable without deeper tools.
- **Readable structure collapses with depth in the first three layers.** token-identity drops 19.8% → 1.7% → 1.9%; doc-identity drops 20.0% → 6.6% → 7.6%. Layer 0 spends its dictionary on surface text statistics; by L01–L02 the alive latents are mostly non-surface.
- **No positional latents anywhere** (position family = 0 in all three; all top hits sit mid-document). Early MLP dictionaries here are not tracking token position.
- **L01/L02 dictionary usage is concentrated and doc-hijacked.** The top max-act latents in both layers (L01: #44678/29965/48466, max 21.6/21.1/20.4; L02: #20323/33384/37447, max 18.1/17.5/17.4) all fire on the *same* Texas county-commissioners document. With only 12.5k alive, L01's usage is sparse AND monopolized by one atypical doc — consistent with its low val performance and low alive count.
- **L00's strongest latent is discourse-structural:** #5203 (max act 11.75) fires on "For example," pivots and textbook scaffold text ("Page 140 Types of War"). Its top doc-identity latents confirm the census-prefix documents are few and atypical (see anomaly below).

## Notable latents (with concrete hits, ±context truncated)

**L00 (val fvu 0.201, best layer, 27.7k/64k alive at 33k):**
- `#5203` doc_identity, max 11.75 — "…a person's standing more than conformity can. **For example,** research I conducted with Silvia Bellezza…"
- `#26897` doc_identity, max 11.75 — "…ism……**Page 140 Types of War**…" (textbook scaffold)
- `#41869` other, max 11.56 — "…get out of the "family business" of education but **every** time I went back…"
- `#10` ngram, max 2.22 — "…Many **conventional RtR controllers** have been concerned w…"
- `#11` token-identity, max 1.05 — fires on "health"/"Real-World Evidence to Biopharma" register
- `#281` punct_ws, max 0.53 — fires on hyphens/parens in stats text: "…adjuvanted (39.3%) than non-adjuvanted…"

**L01 (val 0.665, 13.3k/64k alive — sparse, doc-hijacked):**
- `#44678`/`#29965`/`#48466` other, max 21.6/21.1/20.4 — **all three fire on the same doc**: "…the County Commissioners Court, **Texas** Department of Transportation, Texas Historical Commission…"
- `#2991` ngram, max 7.06 — "…dinner options **available. Calypso official webs**…"
- `#234` token-identity, max 3.94 — fires on "different": "…both hands clasp weapons **different** in form and length…"

**L02 (val 0.157, 24.4k/64k alive):**
- `#20323`/`#33384`/`#37447` other, max 18.1/17.5/17.4 — **same Texas county doc again**, all on "…and other interested…" spans
- `#8193` punct_ws, max 6.84 — fires on period runs in literary text: "…having exasperated them. The irresolution of the commander,"
- `#34` ngram, max 3.19 — "…efforts are **given added impetus**, a…"

## Census-vs-val anomaly verdict: DATA STREAM, not broken decode

The 1.088-vs-0.201 gap is fully explained by *what data each number measured*:

- **Retraction:** the earlier suspicion that `sae_stats` decode was broken is WRONG. Encode-side fvu recompute reproduces the census number (1.1008 vs census 1.088; spread-protocol variance accounts for the small delta).
- **Root cause:** the census samples the **first 204,800 tokens** of the tokensorted parquet — the corpus-opening documents, which are few and atypical (L00's 20% doc-identity rate directly confirms the prefix spans few docs). The val number uses the **trailing 1.92M-row holdout**, the trainer's in-distribution tail. The two numbers are not comparable.
- Supporting signal: L00 is *best* on val (0.201) and *worst* of the three on the census prefix (recomputed 1.10) — a dictionary well-fit to the main distribution degrades most on an out-of-distribution prefix, while underfit L01 (0.771) and L02 (0.798) are flatter everywhere.
- **Fix adopted as recommendation:** uniform row-group sampling in the census (interp-style spread) so census and val measure the same distribution. Until then, val fvu remains the only decision-grade number.
