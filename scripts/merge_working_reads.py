#!/usr/bin/env python3
"""Normalize the parallel working-read batches into one canonical artifact.

The 137 working reads were produced in four evidence-grounded batches while the
candidate dossiers were being built.  The batches use four small JSON variants.
This script reads those variants, validates one record per wave candidate, and
writes ``docs/candidates/working_reads.json``.  The original batch record is
retained under ``source_record`` so no evidence or provenance is lost.

Run from the repository root (or anywhere; paths are repo-relative):

    .venv/bin/python scripts/merge_working_reads.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "sae" / "candidates-68.json"
WORKING_DIR = ROOT / "data" / "derived" / "working_reads"
AGGREGATE_PATHS = [
    WORKING_DIR / "candidate_working_reads_v1.json",
    WORKING_DIR / "candidate_working_reads-2026-11-14.json",
]
DOCS_D_AGGREGATE_PATH = WORKING_DIR / "batch_docs_schema_d.json"
LEGACY_DOCS_PATH = ROOT / "docs" / "candidates" / "working_reads.json"
OUTPUT_PATH = ROOT / "docs" / "candidates" / "working_reads.json"

DEFAULT_VALIDATION = [
    "Run wrong-context controls for the apparent cue.",
    "Compare with random same-layer neurons and magnitude-matched controls.",
]


def as_list(value) -> list[str]:
    """Normalize missing values, scalar prose, and lists to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def as_sources(value) -> list[str]:
    """Normalize the four source/provenance variants to a flat path list."""
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v]
    return as_list(value)


def load_candidates() -> dict[str, dict]:
    candidates = json.loads(CANDIDATES_PATH.read_text())
    return {x["id"]: x for x in candidates}


def load_batch_records() -> dict[str, list[dict]]:
    """Return every available batch variant, grouped by candidate id.

    Each item is ``{"schema":..., "path":..., "record":...}``.  Parallel
    batches can overlap; the caller chooses one primary variant and retains the
    rest as alternates so no authored evidence is silently discarded.
    """
    found: dict[str, list[dict]] = {}

    def add(ident: str, schema: str, path: Path | str, raw: dict) -> None:
        if not raw or raw.get("canonical"):
            return
        items = found.setdefault(ident, [])
        if not any(x["schema"] == schema and x["path"] == str(path) for x in items):
            items.append({"schema": schema, "path": str(path), "record": raw})

    # Schema A: one file per candidate, with a nested working_read object.
    for path in sorted(WORKING_DIR.glob("L*-N*.json")):
        ident = path.stem
        raw = json.loads(path.read_text())
        add(ident, "A", path, raw)

    # Schema B: aggregate with candidate_id and a structured nested working read.
    path = AGGREGATE_PATHS[0]
    for raw in json.loads(path.read_text())["candidates"]:
        add(raw["id"], "B", path, raw)

    # Schema C: aggregate with label/summary/poles and explicit validation.
    path = AGGREGATE_PATHS[1]
    for raw in json.loads(path.read_text())["reads"]:
        add(raw["candidate"], "C", path, raw)

    # Schema D: the original docs-side aggregate.  Prefer the stable local
    # copy; fall back to recovering its raw records from a prior canonical
    # output if the stable copy is unavailable.
    if DOCS_D_AGGREGATE_PATH.exists():
        path = DOCS_D_AGGREGATE_PATH
        for raw in json.loads(path.read_text())["reads"]:
            add(raw["candidate"], "D", path, raw)
    elif LEGACY_DOCS_PATH.exists():
        payload = json.loads(LEGACY_DOCS_PATH.read_text())
        reads = payload.get("reads", [])
        is_canonical = bool(reads) and all(x.get("canonical") for x in reads)
        if is_canonical:
            for canonical in reads:
                if canonical.get("source_schema") != "D":
                    continue
                raw_candidates = [canonical.get("source_record", {})]
                for alt in canonical.get("alternate_source_records", []):
                    if alt.get("source_schema") == "D":
                        raw_candidates.append(alt.get("record", {}))
                for raw in raw_candidates:
                    if raw:
                        add(raw.get("candidate"), "D", f"{LEGACY_DOCS_PATH}:source_record", raw)
        else:
            for raw in reads:
                add(raw.get("candidate"), "D", LEGACY_DOCS_PATH, raw)

    return found


def normalize(ident: str, schema: str, raw: dict) -> dict:
    """Normalize one raw batch record without discarding its evidence."""

    if schema == "A":
        wr = raw["working_read"]
        label = None
        summary = wr["hypothesis"]
        pattern_family = wr.get("pattern_family")
        confidence = wr.get("confidence")
        positive = {"read": wr["positive_pole"]["read"]}
        negative = {"read": wr["negative_pole"]["read"]}
        if wr["positive_pole"].get("support"):
            positive["support"] = wr["positive_pole"]["support"]
        if wr["positive_pole"].get("caveat"):
            positive["caveat"] = wr["positive_pole"]["caveat"]
        if wr["negative_pole"].get("support"):
            negative["support"] = wr["negative_pole"]["support"]
        if wr["negative_pole"].get("caveat"):
            negative["caveat"] = wr["negative_pole"]["caveat"]
        caveats = as_list(wr.get("caveats"))
        validation = as_list(wr.get("recommended_checks")) or DEFAULT_VALIDATION
        evidence = raw.get("evidence", {})
        key_evidence = None
        sources = as_sources(raw.get("provenance"))

    elif schema == "B":
        wr = raw["working_read"]
        label = wr.get("headline")
        summary = wr.get("mechanism_hypothesis") or wr.get("headline")
        pattern_family = raw.get("family")
        confidence = wr.get("confidence")
        positive = {"read": wr["pos_pole_read"]}
        negative = {"read": wr["neg_pole_read"]}
        caveats = as_list(wr.get("confounds"))
        validation = as_list(wr.get("recommended_checks")) or DEFAULT_VALIDATION
        evidence = {"key_evidence": wr.get("key_evidence", [])}
        key_evidence = wr.get("key_evidence", [])
        sources = as_sources(raw.get("evidence_paths"))

    elif schema == "C":
        label = raw.get("label")
        summary = raw["summary"]
        pattern_family = raw.get("category") or raw.get("family")
        confidence = raw.get("confidence")
        positive = {
            "read": raw["poles"]["pos"]["reading"],
            "evidence": as_list(raw["poles"]["pos"].get("evidence")),
        }
        negative = {
            "read": raw["poles"]["neg"]["reading"],
            "evidence": as_list(raw["poles"]["neg"].get("evidence")),
        }
        caveats = as_list(raw.get("caveats"))
        validation = as_list(raw.get("validation")) or DEFAULT_VALIDATION
        evidence = {
            key: raw[key]
            for key in (
                "selectivity_full_pass2", "pole_stats", "attribution",
                "exemplar_ranges",
            )
            if key in raw
        }
        key_evidence = None
        sources = as_sources(raw.get("sources"))

    elif schema == "D":
        label = raw.get("label")
        summary = raw["read"]
        pattern_family = raw.get("category")
        confidence = raw.get("confidence")
        positive = {
            "read": raw["positive"]["fires_on"],
            "status": raw["positive"].get("status"),
            "confidence": raw["positive"].get("confidence"),
            "cue_type": raw["positive"].get("cue_type"),
        }
        negative = {
            "read": raw["negative"]["fires_on"],
            "status": raw["negative"].get("status"),
            "confidence": raw["negative"].get("confidence"),
            "cue_type": raw["negative"].get("cue_type"),
        }
        caveats = as_list(raw.get("caveats"))
        validation = as_list(raw.get("next_checks")) or DEFAULT_VALIDATION
        evidence = raw.get("evidence", {})
        key_evidence = raw.get("key_evidence", [])
        sources = as_sources(raw.get("sources"))

    else:  # pragma: no cover - guarded by load_batch_records
        raise ValueError(f"Unknown working-read schema {schema}")

    return {
        "candidate": ident,
        "layer": raw.get("layer"),
        "neuron": raw.get("neuron"),
        "status": "evidence-grounded working read; not a confirmed explanation",
        "label": label,
        "summary": summary,
        "pattern_family": pattern_family,
        "confidence": confidence,
        "strongest_pole": raw.get("working_read", {}).get("strongest_pole")
        if schema == "B" else None,
        "positive": positive,
        "negative": negative,
        "relation": raw.get("relation"),
        "relation_note": raw.get("relation_note"),
        "caveats": caveats,
        "validation": validation,
        "key_evidence": key_evidence,
        "evidence": evidence,
        "sources": sources,
        "canonical": True,
        "source_schema": schema,
        "source_record": raw,
    }


def validate(records: list[dict], candidates: dict[str, dict]) -> None:
    if len(records) != len(candidates):
        raise ValueError(f"Expected {len(candidates)} records, got {len(records)}")
    seen = set()
    for rec in records:
        ident = rec["candidate"]
        if ident in seen:
            raise ValueError(f"Duplicate working read: {ident}")
        if ident not in candidates:
            raise ValueError(f"Working read has no wave candidate: {ident}")
        seen.add(ident)
        src = candidates[ident]
        if rec["layer"] != src["layer"] or rec["neuron"] != src["neuron"]:
            raise ValueError(f"Layer/neuron mismatch for {ident}")
        for field in ("summary", "confidence", "positive", "negative"):
            if not rec.get(field):
                raise ValueError(f"Missing {field} for {ident}")
        for pole in ("positive", "negative"):
            if not rec[pole].get("read"):
                raise ValueError(f"Missing {pole} read for {ident}")
        if not rec.get("caveats"):
            raise ValueError(f"Missing caveats for {ident}")
        if not rec.get("validation"):
            raise ValueError(f"Missing validation checks for {ident}")
        if not rec.get("sources"):
            raise ValueError(f"Missing source provenance for {ident}")
    missing = set(candidates) - seen
    if missing:
        raise ValueError(f"Missing working reads: {sorted(missing)}")


def main() -> None:
    candidates = load_candidates()
    batches = load_batch_records()
    if set(batches) != set(candidates):
        missing = sorted(set(candidates) - set(batches))
        extra = sorted(set(batches) - set(candidates))
        raise ValueError(f"Batch coverage mismatch; missing={missing}, extra={extra}")

    preference = {"A": 0, "B": 1, "C": 2, "D": 3}
    records = []
    alternate_count = 0
    for ident in sorted(batches):
        variants = sorted(
            batches[ident],
            key=lambda x: (preference.get(x["schema"], 99), x["path"]),
        )
        primary = variants[0]
        rec = normalize(ident, primary["schema"], primary["record"])
        alternates = []
        for variant in variants[1:]:
            alternates.append({
                "source_schema": variant["schema"],
                "source_artifact": variant["path"],
                "record": variant["record"],
            })
        rec["alternate_source_records"] = alternates
        alternate_count += len(alternates)
        records.append(rec)

    records.sort(key=lambda x: (int(x["layer"]), int(x["neuron"])))
    validate(records, candidates)

    schema_counts = {}
    for rec in records:
        schema_counts[rec["source_schema"]] = schema_counts.get(rec["source_schema"], 0) + 1
    alternate_counts = {}
    for rec in records:
        for alt in rec["alternate_source_records"]:
            alternate_counts[alt["source_schema"]] = alternate_counts.get(alt["source_schema"], 0) + 1

    payload = {
        "artifact": "unified Marin-32B encoder-attribution candidate working reads",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate_count": len(records),
        "wave_size": len(candidates),
        "status_note": (
            "Evidence-grounded working hypotheses produced from atlas exemplars, "
            "pole stats, attribution latents, and full sparse-tier selectivity. "
            "None is a confirmed neuron explanation."
        ),
        "schema_counts": dict(sorted(schema_counts.items())),
        "alternate_counts": dict(sorted(alternate_counts.items())),
        "alternate_record_count": alternate_count,
        "sources": {
            "wave": "sae/candidates-68.json",
            "batch_a_examples": "data/derived/working_reads/L*-N*.json",
            "batch_b_aggregate": "data/derived/working_reads/candidate_working_reads_v1.json",
            "batch_c_aggregate": "data/derived/working_reads/candidate_working_reads-2026-11-14.json",
            "batch_d_aggregate": "data/derived/working_reads/batch_docs_schema_d.json",
        },
        "reads": records,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} normalized working reads to {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"schema counts: {schema_counts}")


if __name__ == "__main__":
    main()
