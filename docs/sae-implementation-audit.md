# SAE / autoencoder implementation audit — ops_sh lane (2026-11-22)

Bead: sh-d8b · Task from alex via Bedivere · Standing freeze honored: upstream accessed read-only, zero writes/posts anywhere.

## Scope

Three code surfaces, as tasked:
1. **The AE itself**: `eai-sparsify` 1.3.3 `SparseCoder` (vendored uv-cache copy inspected at
   `/mnt/d/caches/.cache_uv/archive-v0/Cz9ifAolK5J_jGaM/sparsify/`; dusk/APOC runtimes install the same version).
2. **The lane trainer + eval hooks**: `/mnt/d/ops/sh/sae/train_sae.py` (319 lines; byte-identical on the fork
   `alexcasper/marin-neuron-hunt` @ merged-from-apoc — verified by diff), `sae/sae_stats.py` (census),
   `sae/val-table.py` (sweep tables), `sae/interp_preview_dusk.py` (preview), `sae/preprocess_layer.py`.
3. **Upstream Station-House/marin-neuron-hunt** `analyses/sae-L57/` (5 files, read via API): an independent
   CPU reimplementation + diagnosis (`sae_io.py`, `encode.py`, `diagnose.py`, `nominate_fixed.py`,
   `check_candidate.py`, README).

## 1. Inventory

### The autoencoder (eai-sparsify `sparse_coder.py`, 308 lines)
- Standard TopK SAE: `encode = topk((x − b_dec)·W_encᵀ + b_enc)`, `decode = z·W_decᵀ + b_dec`
  (sparse_coder.py:189-202); decoder unit-norm (`:283-288`) and gradient-parallel removal (`:291-304`)
  helpers; AuxK loss machinery (paper Appendix B.1 heuristic, `:236-260`) gated on a `dead_mask`
  argument; safetensors save/load with `cfg.json` (`:167-179`).
- Config surface (`config.py`): activation ∈ {topk, groupmax}, k (default 32), num_latents or
  expansion_factor, normalize_decoder, multi_topk, skip_connection, transcode.
- Matches the OpenAI TopK-paper conventions. Nothing exotic or wrong found in review.

### The lane trainer (`sae/train_sae.py`)
- Own training loop (bypasses sparsify's `trainer.py`): AdamW lr 3e-4, betas (0.9, 0.95), loss = FVU,
  decoder hygiene calls (`:255-258`), **no LR warmup/schedule, no auxk, no dead-latent resampling**
  (sparsify's own trainer has all three: linear warmup trainer.py:132-157, auxk via dead_mask :428-471 —
  all unused because we bypass it).
- Data path: token-sorted sparse parquet (64 rows/token: token_idx, neuron, gate, up) → dense
  [B, 27648] fp32 batches, act = silu(gate)·up, optionally ÷ per-neuron `scale_vec` (`batches()`, :70-102).
  Multi-epoch prefetch thread; bf16 on device; batch 1024–2048 tokens.
- Eval hooks in-loop: trailing-row validation holdout, FVU, plateau detection with early stop and
  plateau checkpoint (`:289-306`); checkpoint grid + culling (`:271-283`); warm-start from disk (`:212-216`).

### Eval/atlas-side tooling
- `sae_stats.py`: census (alive / max_act / fire_frac / FVU over a token sample), scale applied only
  when the scale-parquet arg is passed; layer parsed from ckpt cfg via `L(\d+)`.
- `sae/val-table.py`: sweep table; its 64k-vs-32k Δ column is cross-convention and caveated in METRICS.md.
- `nominate_enc_pass_v2.py` (APOC): loads `Sae.load_from_disk`, encoder-weight attribution
  (`:109-116`, handles W_enc transpose), CPU-vs-GPU equivalence gate.
- Upstream `analyses/sae-L57/encode.py`: numpy chunked top-k forward; `diagnose.py`: FVU / per-neuron
  R² / variance bands; `nominate_fixed.py`: weights-only read-and-write nomination; `check_candidate.py`:
  dense-data tracking check.

## 2. Correctness — verified findings

**Correct (verified by CPU smoke on dawn, this audit):**
- **SMOKE2** — the trainer's dense reconstruction (`batches()`) reproduces ground truth exactly
  (scaled and raw paths; ragged batch boundaries vs row groups included). Logic verbatim, numpy/pyarrow.
- **SMOKE3b** — upstream `encode.py`'s "top-k of pre-acts, then ReLU" is decode-equivalent to
  "ReLU then top-k" in 3000/3000 random trials including fewer-than-k-positives cases (decode masks zeros).
  The CPU reimplementation is faithful.
- **SMOKE4** — `sae_io.py` bf16→f32 bit conversion correct (0x3f80→1.0, 0xBF80→−1.0).
- The 32k sweep family (scale-normalized inputs) produced valid published censuses; the 64-layer 64k
  sweep trained stably end-to-end under its (raw) convention; the GPU attribution equivalence gate
  passes to 3+ decimals everywhere except the documented ±4 boundary (amended to [40,44], bead sh-aav).
- `nominate_enc_pass_v2.py` handles encoder-weight transposition and asserts shape consistency (:113-116).

**Defects (with file:line):**
1. **P0 — the documented fix for the raw-acts incident is not in the code.** METRICS.md rule 5 states
   "`train_sae.py` now hard-fails when scale_vec cannot be resolved (never silently None again)."
   The canonical worktree AND fork copies still silently fall through: `train_sae.py:195-200` —
   `layer_label.lstrip("L")` + `isdigit()` returns None for sweep labels like `L00w64`/`L57w64` and
   training proceeds on **raw activations with no warning**. Reproduced (SMOKE1): labels `L57w64`,
   `L00w64`, `L?` → scale_vec=None; label `L7` → loaded. This is the exact mechanism behind the
   64k-sweep raw-acts convention (METRICS.md ROOT CAUSE; independently described by upstream
   analyses/sae-L57 README §2: "a label-parse bug skipped the intended per-neuron scaling"). The fix
   evidently lives only on the dusk runtime copy. **Any future sweep launched from the repo copy
   silently regresses.**
2. **P1 — dead validation comment.** `train_sae.py:87` comments "contiguity check: token_idx strictly
   non-decreasing per row-block of 64" — no such check exists (only `assert n % 64 == 0` at :81).
   A non-token-sorted or malformed parquet yields silently corrupted training batches.
3. **P1 — k drift across runs, unrecorded.** `run_L00_64k.sh:27` uses k=64; `apoc-64k-sweep.sh:36`
   uses k=32; both are "64k" (65,536-latent) runs. Cross-run comparisons must quote k; METRICS.md's
   convention table has no k column.
4. **P2 — no stabilization.** No warmup/schedule (upstream README §4 documents the L57 FVU spike
   0.138→0.76 at step ~12,000), no AuxK (top-of-`forward` machinery never receives `dead_mask`),
   no resampling → upstream census: only 4,487/65,536 (24%) latents alive on the L57 64k.
5. **P3 — `b_dec` initialized to zeros** (sparse_coder.py:69); on uncentered raw acts this wastes
   capacity (upstream README: b_dec[3872]=0.076 fails to absorb the massive-activation neuron).
6. Environment landmine (minor): the dawn uv-cache copy contains a corrupted nested
   `sparsify/sparsify/sparse_coder.py` (0 bytes). Imports resolve to the outer package today, but
   pointing PYTHONPATH at the nested dir yields a silently-empty module.

**Known-incidents check (our own history, as tasked):**
- Census fvu>1 anomaly: root-caused (scale-convention mismatch), censuses re-run convention-matched
  (census v2 no-scale), artifact preserved separately (METRICS.md implementation notes). Handling: sound.
- GPU smoke gate 35665026: attribution eval hit a ±4 device-precision boundary on top1_ge_0.5;
  gate amended to [40,44] with rationale in the slurm GATE ORDER; equivalence otherwise 4.07e-09
  (test_nominate_v2.py PASS, sh-ygh). Handling: sound.
- L57 preview once regenerated from a superseded plateau ckpt — process slip, caught and documented.

## 3. Training viability

- The pipeline has **run at scale**: 32-layer... specifically 32k-width sweep + 12k topup (scaled),
  full 64-layer 64k sweep (raw convention), 128k probe/rollout chain, L63 retrain, plateau early-stop
  with holdouts — stably, on both ROCm (dusk) and APOC H200s. No crash/NaN history in the lane record.
- **Quality of what it trained** is a different question, and the strongest evidence is upstream's
  independent L57 diagnosis (README §3-4, with our METRICS.md root-cause agreeing on the mechanism):
  on fresh dense data the 64k L57 checkpoint reconstructs only the top ~10 variance neurons
  (FVU 0.949 overall; 1.06 for variance ranks 101+; median per-neuron R² negative on fineweb-1m-deep);
  24% latents alive; training FVU erratic. That run combined raw acts + short schedule + no auxk —
  i.e., every P0/P2 defect above at once. The 32k family (scaled) never showed the artifact.
- Conclusion: the code **can train usable SAEs when the input convention is what the operator thinks
  it is** (32k family proves it); the L57 failure modes are explained by the defects, not by anything
  structural in the architecture.

## 4. Fork issues fold-in

None of the 151 open issues on alexcasper/marin-neuron-hunt is an autoencoder-implementation
complaint: 137 are candidate-intake mirrors (verified separately, sh-1yx), 6 are archived
explanation PRs consuming nominations, and the rest are trackers/write-ups. The substantive AE
critique exists only as upstream's `analyses/sae-L57` README (not an issue). Its claims were
checked against our records where they overlap: the label-parse/raw-acts mechanism matches
METRICS.md's root cause; the five nomination threads it criticizes are exactly the ones our triage
verified number-by-number (docs/fork-issue-triage.md).

## 5. VERDICT (explicit, ranked)

**USABLE-WITH-FIXES.** Not usable-as-is; definitely not a rework-from-scratch.

- Not **usable-as-is** because the canonical trainer copy still contains the exact silent-fallback
  that caused the lane's one real training incident, while the methodology doc claims it was fixed —
  launching the next sweep from the repo today would silently repeat it.
- Not **rework-from-scratch** because nothing is structurally wrong: the architecture is the standard
  TopK SAE, the data path reconstructs bit-faithfully, the eval/census/attribution tooling is
  integrated with our pipeline (nominations, GPU gate, HF artifact chain), two width families have
  trained stably at scale, and every observed failure has a code-level explanation with a small fix.
  A rewrite would discard the gate, census, and attribution integrations that took the lane months
  to harden.

### Fix list (ranked, with effort)

| Pri | Fix | Where | Effort |
|---|---|---|---|
| P0 | Hard-fail when `scale_vec` unresolvable (raise on layer-label parse miss / missing scale parquet); commit to canonical copy so METRICS.md rule 5 becomes true | train_sae.py:195-200 | <1 h incl. a CPU test |
| P1 | Implement the contiguity check the comment at :87 already promises (assert token_idx constant per 64-row block) | train_sae.py:81-87 | <30 min |
| P1 | Add k to the METRICS.md convention table; require quoting k in any cross-run comparison (L00-64k k=64 vs sweep k=32) | METRICS.md + run scripts | 30 min |
| P2 | LR warmup + linear decay (port trainer.py:132-157 pattern) — motivated by the L57 step-12k FVU spike | train_sae.py loop | 1-2 h + GPU smoke |
| P2 | Dead-latent handling: pass `dead_mask` to engage AuxK, or periodic resampling — motivated by 24% alive rate | train_sae.py + forward | 0.5-1 d + GPU validation |
| P3 | `b_dec` init to first-batch input mean | train_sae.py init | <30 min |
| P3 | Decide convention unification (64k raw vs 32k scaled): rescale-eval or retrain; until then cross-width comparisons stay illegal-by-doc only | program-level | GPU-day-scale decision for alex |

**GPU-unverifiable here (marked):** actual forward/backward numerics of SparseCoder at d_in=27648
(dawn has no torch; SMOKE1-4 were numpy/pyarrow-only), auxk/warmup effectiveness, any training curve
reproduction. All listed smoke results above ran on dawn CPU in this audit.

## Smoke log (this audit)

- SMOKE1: label→scale_vec parse reproduction — `L7`→loaded; `L57w64`/`L00w64`/`L?`→None (silent). 
- SMOKE2: synthetic 64-rows/token parquet → `batches()` scaled and raw outputs equal dense ground
  truth exactly (rtol 1e-5), with deliberately ragged batch size (7 tokens) vs row-group boundaries.
- SMOKE3/3b: topk-then-ReLU ≡ ReLU-then-topk for decode purposes, 3000/3000 trials incl. #positives<k.
- SMOKE4: bf16→f32 view-shift conversion correct on ±1.0 sentinels.
