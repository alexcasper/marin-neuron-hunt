#!/usr/bin/env python3
"""Generate evidence-led candidate dossiers for the Marin-32B neuron wave.

Inputs:
- sae/candidates-68.json  (historical filename; contains the full 137-candidate wave)
- .beads/issues.jsonl     (bead mapping and enriched issue metadata)
- data/derived/candidate_wave/*.json (atlas pole_stats and exemplars)
- data/derived/token_selectivity_full/layer=NN.parquet

Outputs:
- docs/candidates/<NEURON>.md
- docs/candidates/README.md

When docs/candidates/working_reads.json is present, each dossier also carries
its evidence-grounded working read.  Those reads are hypotheses authored from
the stored evidence, not confirmed neuron explanations.
"""

from __future__ import annotations

import collections
import html
import json
import math
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "candidates"
CANDIDATES_PATH = ROOT / "sae" / "candidates-68.json"
BEADS_PATH = ROOT / ".beads" / "issues.jsonl"
WORKING_READS_PATH = OUT / "working_reads.json"
DERIVED = ROOT / "data" / "derived"

PUNCT_TOKENS = {
    ".", ",", "?", "!", ";", ":", "-", "--", ".Ċ", "Ċ", "Ġ", "'", '"',
    "(", ")", "[", "]", "{", "}", "/", "\\", "*", "&", "%", "#", "@", "+",
    "=", "<", ">", "|", "~", "^", "$",
}


def clean_text(s: str) -> str:
    """Make BPE-ish token/context text readable in Markdown while preserving data."""
    if s is None:
        return ""
    s = str(s)
    # The corpus uses Ġ for a leading space and Ċ for newline.
    s = s.replace("Ġ", " ").replace("Ċ", "\\n")
    # Keep other bytes visible but escape Markdown-sensitive characters in table cells.
    return s.replace("|", "\\|").replace("\n", "⏎")


def token_label(s: str) -> str:
    """Backticked, readable token label."""
    return "`" + clean_text(s).replace("`", "'") + "`"


def num(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 1e6):
        return f"{int(x):,}"
    return f"{float(x):.{digits}f}"


def load_bead_map() -> dict[str, dict]:
    result = {}
    for line in BEADS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        issue = json.loads(line)
        title = issue.get("title", "")
        if title.startswith(("L", "Encoder-attribution candidate wave")):
            match = re.match(r"(L\d+-N\d+)", title)
            if match:
                result[match.group(1)] = issue
            elif issue.get("id") == "sh-8wt":
                result["__parent__"] = issue
    return result


def load_working_reads() -> dict[str, dict]:
    if not WORKING_READS_PATH.exists():
        return {}
    payload = json.loads(WORKING_READS_PATH.read_text())
    reads = payload.get("reads", [])
    if payload.get("candidate_count") != len(reads):
        raise SystemExit("working_reads candidate_count does not match read count")
    result = {x["candidate"]: x for x in reads}
    if len(result) != len(reads):
        raise SystemExit("Duplicate ids in working_reads.json")
    return result


def md_inline_list(values: list[str]) -> str:
    return "; ".join(str(x).rstrip(".") for x in values if str(x).strip()) + "."


def working_read_block(work: dict | None) -> list[str]:
    """Render a normalized model-authored working read as Markdown."""
    if not work:
        return []
    lines = [
        "## Working read (evidence-grounded hypothesis)",
        "",
        "**Status:** model-authored from the stored evidence; not a confirmed neuron explanation.",
        "",
    ]
    if work.get("label"):
        lines.append(f"**Label:** {work['label']}")
        lines.append("")
    if work.get("pattern_family"):
        lines.append(f"**Pattern family:** {work['pattern_family']}")
        lines.append("")
    lines.append(f"**Confidence:** {work.get('confidence') or 'not stated'}")
    lines.append("")
    lines.append(work["summary"])
    lines.append("")
    for pole, title in (("positive", "Positive-pole read"), ("negative", "Negative-pole read")):
        lines.append(f"### {title}")
        lines.append("")
        lines.append(str(work[pole].get("read", "")).strip())
        lines.append("")
        details = []
        for field, prefix in (("support", "Support"), ("caveat", "Caveat"), ("cue_type", "Cue type")):
            if work[pole].get(field):
                details.append(f"{prefix}: {work[pole][field]}")
        if work[pole].get("evidence"):
            details.append("Evidence quotes: see the normalized JSON and atlas tables below.")
        if details:
            lines.append(" — ".join(str(x).rstrip(".") + "." for x in details))
            lines.append("")
    if work.get("relation_note"):
        lines.append(f"**Pole relation:** {work['relation_note']}")
        lines.append("")
    if work.get("caveats"):
        lines.append("**Caveats:** " + md_inline_list(work["caveats"]))
        lines.append("")
    if work.get("validation"):
        lines.append("**Next checks:** " + md_inline_list(work["validation"]))
        lines.append("")
    lines.append(
        f"**Normalized provenance:** `docs/candidates/working_reads.json` "
        f"(source batch schema `{work.get('source_schema', 'unknown')}`)"
    )
    if work.get("alternate_source_records"):
        schemas = ", ".join(
            f"`{alt.get('source_schema', 'unknown')}`"
            for alt in work["alternate_source_records"]
        )
        lines.append(f"**Alternate batch reads retained:** {schemas}")
    lines.append("")
    return lines


def load_selectivity(candidates: list[dict]) -> dict[tuple[int, int], dict[str, dict]]:
    by_layer: dict[int, set[int]] = collections.defaultdict(set)
    for candidate in candidates:
        by_layer[int(candidate["layer"])].add(int(candidate["neuron"]))

    result: dict[tuple[int, int], dict[str, dict]] = {}
    for layer, neurons in sorted(by_layer.items()):
        path = DERIVED / "token_selectivity_full" / f"layer={layer:02d}.parquet"
        table = pq.read_table(path, filters=[("neuron", "in", sorted(neurons))])
        for row in table.to_pylist():
            if row["neuron"] not in neurons:
                continue
            key = (int(row["layer"]), int(row["neuron"]))
            slot = result.setdefault(key, {})
            pole = int(row.get("pole") or 0)
            if pole == 1:
                slot["pos"] = row
            elif pole == -1:
                slot["neg"] = row
    return result


def exemplar_summary(exemplars: dict, pole: str, max_rows: int = 5) -> tuple[list[dict], str]:
    block = exemplars.get(pole, {})
    rows = block.get("rows", []) or []
    shown = rows[:max_rows]
    token_counts = collections.Counter(r.get("token_str", "") for r in rows)
    common = [token for token, _ in token_counts.most_common(5)]
    pattern = ", ".join(token_label(t) for t in common) if common else "no stored exemplars"
    return shown, pattern


def selectivity_line(row: dict | None) -> str:
    if not row:
        return "No matching row in the full sparse-tier selectivity table."
    return (
        f"`{clean_text(row.get('top_token_str'))}` — "
        f"n_fire {num(row.get('n_fire'), 0)}, "
        f"distinct tokens {num(row.get('n_distinct_tokens'), 0)}, "
        f"distinct docs {num(row.get('n_distinct_docs'), 0)}, "
        f"top share {num(row.get('top_token_share'))}, "
        f"lift {num(row.get('top_token_lift'))}, "
        f"eff_tokens {num(row.get('eff_tokens'))}."
    )


def is_nonenglishish(exemplars: dict) -> bool:
    # Use the positive side preferentially: corpus mojibake often leaks into the
    # otherwise ordinary negative side and can mask a clean positive concept.
    for pole in ("pos", "neg"):
        tokens = [r.get("token_str", "") for r in exemplars.get(pole, {}).get("rows", [])]
        if len(tokens) >= 6:
            weird = sum(
                1 for t in tokens
                if any(ord(c) > 127 and c not in "ĠĊ" for c in t)
            )
            if weird / len(tokens) >= 0.45:
                return True
    return False


def classify(candidate: dict, sel_pos: dict | None, sel_neg: dict | None) -> tuple[str, str]:
    """Return (family, provisional read) grounded in the available evidence."""
    attribution = candidate["attribution"]
    n_latents = int(attribution["n_latents"])
    fired_min, fired_max = attribution["fired_range"]
    max_share = float(attribution["max_top1_share"])
    exemplars = candidate["exemplars"]

    # Which pole has the most striking selectivity, if any?
    best = None
    for pole, row in (("positive", sel_pos), ("negative", sel_neg)):
        if not row:
            continue
        share = float(row.get("top_token_share") or 0)
        lift = float(row.get("top_token_lift") or 0)
        eff = float(row.get("eff_tokens") or 0)
        score = share * math.log2(max(lift, 1.0) + 1)
        if best is None or score > best[0]:
            best = (score, pole, row, share, lift, eff)

    pos_rows, pos_common = exemplar_summary(exemplars, "pos")
    neg_rows, neg_common = exemplar_summary(exemplars, "neg")

    structural = any(t in PUNCT_TOKENS for t in pos_common[:3]) or any(
        t in PUNCT_TOKENS for t in neg_common[:3]
    )
    nonenglish = is_nonenglishish(exemplars)
    sharp = bool(best and best[4] >= 5 and best[5] <= 40 and best[3] >= 0.12)
    broad = bool(
        sel_pos and sel_neg
        and float(sel_pos.get("eff_tokens") or 0) >= 70
        and float(sel_neg.get("eff_tokens") or 0) >= 70
    )
    high_frequency = fired_max >= 3000

    if sharp and best:
        pole, row = best[1], best[2]
        tok = token_label(row.get("top_token_str"))
        family = "sharp-token/context"
        read = (
            f"Provisional read: a relatively sharp {pole}-pole candidate centered on {tok}. "
            f"That token is {num(row.get('top_token_share'))} of the pole's sparse-tier fires at "
            f"{num(row.get('top_token_lift'))}× lift, over roughly "
            f"{num(row.get('eff_tokens'), 1)} effective token types. "
            "This may be token-conditioned rather than a one-token detector; wrong-context tests are essential."
        )
    elif structural:
        family = "punctuation/layout"
        read = (
            "Provisional read: punctuation/layout-sensitive candidate. "
            f"The stored exemplars repeatedly use punctuation or boundary tokens ({', '.join(token_label(t) for t in pos_common[:2])} positive; "
            f"{', '.join(token_label(t) for t in neg_common[:2])} negative), so the likely distinction is sentence/list/paragraph position "
            "rather than ordinary word content."
        )
    elif nonenglish:
        family = "non-English/encoding"
        read = (
            "Provisional read: non-English text, transliteration, or encoding/mojibake handling. "
            "Many high-activation exemplars contain non-ASCII or corrupted byte fragments, so this may "
            "track corpus artifacts or language-specific subwords rather than a clean human-readable concept."
        )
    elif high_frequency:
        family = "broad/high-frequency"
        read = (
            f"Provisional read: broad contextual feature. One or more supporting latents fire very often "
            f"(up to {num(fired_max, 0)} sampled tokens), so this is a high-prior-risk always-on or general-domain "
            "candidate unless controls separate it from magnitude alone."
        )
    elif n_latents >= 3:
        family = "multi-latent contextual"
        read = (
            f"Provisional read: contextual/semantic candidate with convergent evidence — {n_latents} independent "
            f"latents nominate it. The positive exemplars favor {', '.join(token_label(t) for t in pos_common[:3])}, while the negative exemplars favor "
            f"{', '.join(token_label(t) for t in neg_common[:3])}. Treat this as a domain or context hypothesis, not a confirmed concept."
        )
    elif broad:
        family = "broad-contextual"
        read = (
            "Provisional read: broad contextual feature. Both poles spread over many effective token types, "
            "so any eventual explanation will likely involve register/document context rather than one token."
        )
    else:
        family = "mixed-contextual"
        read = (
            "Provisional read: mixed contextual candidate. The exemplars do not yet reduce to one obvious token "
            f"or topic: positive examples favor {', '.join(token_label(t) for t in pos_common[:3])}; negative examples favor "
            f"{', '.join(token_label(t) for t in neg_common[:3])}. "
            "Additional controls are required before interpreting it."
        )

    # Add an explicit caution when the evidence is weak.
    if n_latents == 1 and fired_max < 100:
        read += " Low independent support: only one latent and fewer than 100 sampled fires."
    elif high_frequency:
        read += " Always-on/magnitude confound is a primary rejection risk."

    return family, read


def priority(candidate: dict, family: str, sel_pos: dict | None, sel_neg: dict | None) -> str:
    attribution = candidate["attribution"]
    n_latents = int(attribution["n_latents"])
    fired_min, fired_max = attribution["fired_range"]
    max_share = float(attribution["max_top1_share"])

    best_lift = 0.0
    for row in (sel_pos, sel_neg):
        if row:
            best_lift = max(best_lift, float(row.get("top_token_lift") or 0))

    if family in {"sharp-token/context", "punctuation/layout"} and n_latents >= 2:
        return "high"
    if n_latents >= 4 and fired_max >= 100 and max_share >= 0.55:
        return "high"
    if n_latents >= 2 and (fired_max >= 100 or best_lift >= 5):
        return "medium"
    if family == "broad/high-frequency":
        return "low"
    return "low" if n_latents == 1 and fired_max < 100 else "medium"


def render_candidate(candidate: dict, bead: dict | None, sel_pos: dict | None,
                     sel_neg: dict | None, family: str, read: str, tier: str,
                     work: dict | None = None) -> str:
    ident = candidate["id"]
    attribution = candidate["attribution"]
    pole_stats = candidate["pole_stats"]
    exemplars = candidate["exemplars"]
    latents = attribution["latents"]

    lines: list[str] = []
    lines.append(f"# {ident} — provisional candidate dossier")
    lines.append("")
    lines.append(f"- **Bead:** `{bead.get('id') if bead else 'unmapped'}`")
    lines.append(f"- **Wave parent:** `sh-8wt`")
    lines.append(f"- **Family:** {family}")
    lines.append(f"- **Provisional priority:** {tier}")
    lines.append(f"- **Source candidates:** `{CANDIDATES_PATH.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Nomination evidence")
    lines.append("")
    lines.append(
        f"- **Contributing latents:** {attribution['n_latents']} "
        f"({', '.join(str(x['latent']) for x in latents)})"
    )
    lines.append(
        f"- **Per-latent fired-token range:** {attribution['fired_range'][0]:,}–"
        f"{attribution['fired_range'][1]:,}"
    )
    lines.append(f"- **Max top1_share:** {num(attribution['max_top1_share'])}")
    lines.append("")
    lines.append("| latent | fired tokens | top1_share | eff_neurons |")
    lines.append("|---|---:|---:|---:|")
    for row in latents:
        lines.append(
            f"| {row['latent']} | {row['fired_tokens']:,} | "
            f"{num(row['top1_share'])} | {num(row['eff_neurons'])} |"
        )
    lines.append("")

    lines.append("## Atlas pole statistics")
    lines.append("")
    lines.append(
        f"- tokens `n={num(pole_stats.get('n'), 0)}`; mean {num(pole_stats.get('mean'))}; "
        f"std {num(pole_stats.get('std'))}; positive fraction {num(pole_stats.get('frac_pos'))}"
    )
    lines.append(
        f"- range: max {num(pole_stats.get('max'))}, min {num(pole_stats.get('min'))}; "
        f"`scale_abs_p999` {num(pole_stats.get('scale_abs_p999'))}"
    )
    lines.append(
        f"- positive tails: q99 {num(pole_stats.get('pos_q99'))}, "
        f"q999 {num(pole_stats.get('pos_q999'))}, q9999 {num(pole_stats.get('pos_q9999'))}"
    )
    lines.append(
        f"- negative tails: q99 {num(pole_stats.get('neg_q99'))}, "
        f"q999 {num(pole_stats.get('neg_q999'))}, q9999 {num(pole_stats.get('neg_q9999'))}"
    )
    lines.append("")

    lines.append("## Full sparse-tier token selectivity")
    lines.append("")
    lines.append("**Positive pole:** " + selectivity_line(sel_pos))
    lines.append("")
    lines.append("**Negative pole:** " + selectivity_line(sel_neg))
    lines.append("")

    lines.append("## Atlas exemplar pattern")
    lines.append("")
    for pole, title in (("pos", "Positive"), ("neg", "Negative")):
        rows, _ = exemplar_summary(exemplars, pole, 5)
        lines.append(f"### {title} pole")
        lines.append("")
        if not rows:
            lines.append("No stored exemplars for this pole.")
            lines.append("")
            continue
        lines.append("| rank | act | token | doc_id | context |")
        lines.append("|---:|---:|---|---:|---|")
        for row in rows:
            context = clean_text(row.get("context", ""))
            if len(context) > 260:
                context = context[:257] + "..."
            lines.append(
                f"| {row.get('rank')} | {num(row.get('act'))} | "
                f"{token_label(row.get('token_str'))} | {row.get('doc_id')} | {context} |"
            )
        lines.append("")

    lines.append("## Provisional read")
    lines.append("")
    lines.append(read)
    lines.append("")
    lines.extend(working_read_block(work))
    lines.append("## Required validation")
    lines.append("")
    lines.append("1. Re-check the nomination filters; ignore invalid v1 shuffle-control fields.")
    lines.append("2. Confirm both poles against `pole_stats` and full sparse-tier token selectivity.")
    lines.append("3. Run wrong-context controls for the apparent cue.")
    lines.append("4. Compare with random same-layer neurons and magnitude-matched controls.")
    lines.append("5. Close the bead as `promote`, `reject`, or `needs-more-evidence` with evidence paths.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Atlas: `data/pilot/fineweb-10m-sparse` (byte-verified local copy)")
    lines.append("- Pole stats/exemplars: `data/derived/candidate_wave/" + ident + ".json`")
    lines.append("- Selectivity: `data/derived/token_selectivity_full/layer=" + f"{int(candidate['layer']):02d}.parquet`")
    lines.append("- Attribution: 64k SAE finals from the encoder-attribution wave")
    if work:
        lines.append("- Working read: `docs/candidates/working_reads.json` (model-authored hypothesis from the stored evidence)")
    lines.append("- Status: **candidate dossier only; not a confirmed neuron explanation**")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    candidates = json.loads(CANDIDATES_PATH.read_text())
    if len({x["id"] for x in candidates}) != len(candidates):
        raise SystemExit("Duplicate candidate ids")

    # The nomination export stores attribution fields at the top level; normalize
    # them into the same shape used by the derived per-neuron JSON files.
    for candidate in candidates:
        candidate["attribution"] = {
            "latents": candidate["latents"],
            "n_latents": candidate["n_latents"],
            "max_top1_share": candidate["max_top1_share"],
            "fired_range": candidate["fired_range"],
        }

    beads = load_bead_map()
    working_reads = load_working_reads()
    selectivity = load_selectivity(candidates)
    OUT.mkdir(parents=True, exist_ok=True)

    # Clear only generated candidate dossiers to avoid stale files.
    for path in OUT.glob("L*-N*.md"):
        path.unlink()

    entries = []
    for candidate in candidates:
        ident = candidate["id"]
        layer = int(candidate["layer"])
        neuron = int(candidate["neuron"])
        pole_map = selectivity.get((layer, neuron), {})
        sel_pos = pole_map.get("pos")
        sel_neg = pole_map.get("neg")

        derived_path = DERIVED / "candidate_wave" / f"{ident}.json"
        derived = json.loads(derived_path.read_text())
        candidate["pole_stats"] = derived["pole_stats"]
        candidate["exemplars"] = derived["exemplars"]

        family, read = classify(candidate, sel_pos, sel_neg)
        tier = priority(candidate, family, sel_pos, sel_neg)
        bead = beads.get(ident)
        work = working_reads.get(ident)
        text = render_candidate(candidate, bead, sel_pos, sel_neg, family, read, tier, work)
        (OUT / f"{ident}.md").write_text(text)
        entries.append({
            "id": ident,
            "bead": bead.get("id") if bead else "unmapped",
            "family": family,
            "tier": tier,
            "working_read": bool(work),
            "working_confidence": work.get("confidence") if work else None,
            "n_latents": candidate["attribution"]["n_latents"],
            "fired_range": candidate["attribution"]["fired_range"],
            "max_top1_share": candidate["attribution"]["max_top1_share"],
        })

    # Index sorted by evidence strength, then candidate id.
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    entries.sort(key=lambda x: (
        tier_rank[x["tier"]], -x["n_latents"], -x["max_top1_share"], x["id"]
    ))

    counts = collections.Counter(x["family"] for x in entries)
    tier_counts = collections.Counter(x["tier"] for x in entries)
    index: list[str] = []
    index.append("# Encoder-attribution candidate wave")
    index.append("")
    index.append(
        "Provisional dossiers and evidence-grounded working reads for the Marin-32B "
        "MLP-neuron candidates nominated by the 64k-SAE encoder-attribution pass. "
        "The historical input filename is "
        "`sae/candidates-68.json`, but the full 64-layer wave contains **137 candidates**."
    )
    index.append("")
    index.append("**Status:** evidence summaries and model-authored working reads only. These are not confirmed neuron explanations.")
    index.append("")
    index.append("## Counts")
    index.append("")
    index.append(f"- Candidates: **{len(entries)}**")
    index.append(f"- Working reads: **{sum(bool(x['working_read']) for x in entries)} / {len(entries)}**")
    index.append(f"- Priority: high {tier_counts.get('high', 0)}, "
                 f"medium {tier_counts.get('medium', 0)}, low {tier_counts.get('low', 0)}")
    index.append("- Families: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    index.append("")
    index.append("## Selection and evidence")
    index.append("")
    index.append("- Nomination filter: per contributing latent `top1_share >= 0.5` and `fired_tokens >= 20`.")
    index.append("- Attribution source: 64k SAE final checkpoints.")
    index.append("- Atlas source: byte-verified `data/pilot/fineweb-10m-sparse` copy.")
    index.append("- Selectivity source: `data/derived/token_selectivity_full/layer=NN.parquet`.")
    index.append("- The `nominate-enc` v1 shuffle controls are invalid and are intentionally not used.")
    index.append("")
    index.append("## Candidate index")
    index.append("")
    index.append("| candidate | bead | priority | family | working read | read confidence | latents | fired range | max top1_share |")
    index.append("|---|---|---|---|---|---|---:|---:|---:|")
    for x in entries:
        read = "yes" if x["working_read"] else "no"
        conf = x["working_confidence"] or "—"
        index.append(
            f"| [{x['id']}]({x['id']}.md) | `{x['bead']}` | {x['tier']} | "
            f"{x['family']} | {read} | {conf} | {x['n_latents']} | "
            f"{x['fired_range'][0]:,}–{x['fired_range'][1]:,} | {num(x['max_top1_share'])} |"
        )
    index.append("")
    index.append("## Uniform next steps")
    index.append("")
    index.append("For each candidate, first run its dossier-specific **Next checks**, then close its bead only after wrong-context controls, "
                 "random/same-layer comparisons, and a verdict of `promote`, `reject`, or "
                 "`needs-more-evidence`, with evidence paths recorded.")
    index.append("")
    (OUT / "README.md").write_text("\n".join(index))

    print(f"generated {len(entries)} candidate dossiers")
    print(f"families: {dict(sorted(counts.items()))}")
    print(f"tiers: {dict(sorted(tier_counts.items()))}")


if __name__ == "__main__":
    main()
