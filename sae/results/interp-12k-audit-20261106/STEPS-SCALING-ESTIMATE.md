# Steps-scaling estimate — is a 10× step run worth it?

**Date:** 2026-11-06 · Follow-on to the 12k audit (same directory, `REPORT.md`).
**Decision this supports:** whether to spend GPU time testing ~10× steps
(6k → 60k) on hard layers before any width scale-up. Alex's call; this is the
cost side only.

## What the audits established

- 6k finals, 20 good layers: ~80% unclear, concepts ~1% (480 blind pairs).
- 12k finals, 8 hard topup layers: unclear flat (82/85%), concepts 0%,
  doc-identity unchanged ~14% (192 blind pairs).
- Conclusion: 2× steps buys nothing interpretive. The next meaningful test is
  ~an order of magnitude more steps.

## Cost estimate (dispatcher-provided, marked ESTIMATE)

- Scope: 8 layers (L12, 18, 26, 33, 42, 50, 55, 63) at 32k width, to
  **step 60000**, same batch/seq as the sweep.
- Wall-clock at the usual 8-wide concurrency: **~1.5–2 h on 8×H200,
  ~4 h on A100-40s** — i.e. ~12–16 H200-GPU-hours total (32–64 A100-hours).
- Plateau early-stop (the L57 recipe's 2%/9k-window rule) fired at ~12k on some
  topup layers; if hard layers plateau similarly, the true cost could be ~half.
- Checkpoint retention: keep the 3k-grid so we get interp snapshots at 12k/24k/
  36k/48k/60k — a steps-vs-readability curve, not one endpoint.
- Interp audit of the outputs: ~1 h CPU array + one labeling evening (proven
  pipeline, 192-240 pairs).

## Recommendation

Cheap, well-scoped, and it resolves the one open question the 12k audit left:
if 60k still yields ~0% concepts, the lane closes the interp direction for
this model age and treats the atlas as a document-identity/memorization
instrument; if concepts appear, the "steps" verdict revives and the width
decision gets re-opened with evidence. Either outcome is decisive for the
64k gate.

**Not started — awaiting ALEX_GO** (decision-class; beads rule: record the
recommendation, humans close the call).
