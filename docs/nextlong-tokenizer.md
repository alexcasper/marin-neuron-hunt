# NExtLong-128K tokenizer vs Marin-32B — compatibility (verified 2026-11-23)

Artifacts (both sha256-verified):

- `tokenizers/Llama-3-8B-NExtLong-128K-Base/tokenizer.json` — caskcsg/Llama-3-8B-NExtLong-128K-Base (HF commit 7ed4936a), sha256 `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`. 9,085,698 bytes.
- `tokenizers/marin-32b-base/tokenizer.json` — marin-community/marin-32b-base reference (per Bedivere), sha256 `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`. 17,209,920 bytes.

## Verdict: ordinary-text tokenization is byte-identical

Verified programmatically (full comparison, no sampling):

- Base vocab: 128,000 strings, **identical id assignment** (`model.vocab` dicts equal).
- BPE merges: 280,147 in both; **identical set and identical order** (0 positional diffs after normalizing serialization — Marin stores 2-element `[a,b]` pairs, NExtLong stores 3-element `[a,b,result]`; compared on the pair components).
- `normalizer`: both `null`. `pre_tokenizer`: identical (Llama-3 Split regex + ByteLevel).
- `post_processor`: NExtLong auto-prepends `<|begin_of_text|>` (128000). No truncation/padding config. **No chat template** (base model).

Consequence: any text tokenized by one yields the exact same token-id sequence under the other (excluding special-token strings). **All atlas machinery that consumes token ids (sparse tiers, vocab.parquet, attribution streams, token-selectivity) transfers to NExtLong inputs unchanged.**

## The one caveat: special-token NAMES differ at shared ids

256 special slots (ids 128000–128255) align by id; **249/256 names differ**. Marin uses Llama-3.1-style names; NExtLong keeps `<|reserved_special_token_N|>` placeholders:

| id | Marin | NExtLong |
|---|---|---|
| 128000 | `<|begin_of_text|>` | `<|begin_of_text|>` ✓ |
| 128001 | `<|end_of_text|>` | `<|end_of_text|>` ✓ |
| 128004 | `<|finetune_right_pad_id|>` | `<|reserved_special_token_2|>` |
| 128008 | `<|eom_id|>` | `<|reserved_special_token_4|>` |
| 128010 | `<|python_tag|>` | `<|reserved_special_token_5|>` |
| 128009 | `<|eot_id|>` | `<|eot_id|>` ✓ |
| 128005,128011+ | reserved chain | reserved chain (off-by-name from 128005 on) |

Rule: **anything addressing special tokens by NAME needs the id-based remap table; id- or text-level work is unaffected.** (Names matter for chat/tool templating and `tokenizer_config.json` generation — NExtLong's tokenizer_config was not staged, and RoPE/theta hints live in config.json, which was not staged either.)

## Lane angle (bead: see `bd` — NExtLong tokenizer intake)

Phase 1 — no weights, CPU-only: write the id-based special-token remap table + round-trip test suite; confirm our sparse-parquet readers/`batches()` accept NExtLong-tokenized text.
Phase 2 — GATED on alex's OK (8B model ≈ 16GB bf16, multi-GB gate): activation capture on long-context prompts (capybara-v1-style probes at 128K context) + TopK SAE. The identical tokenization makes cross-model comparisons at matched positions exact — tokenization is controlled for, which is rare across model families.
