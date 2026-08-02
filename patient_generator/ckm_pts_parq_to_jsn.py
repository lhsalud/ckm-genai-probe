#!/usr/bin/env python3
"""
ckm_pts_parq_to_jsn.py

Load a PMR-synth .parquet export, parse the packed `text` column into structured
fields, and emit JSON matching the schema of pmr_synth_parsed_sample.json.

Expected input schema
---------------------
    text       : str    -- packed EHR block + patient message
    level      : int64  -- acuity label
    relevancy  : int64  -- EHR-relevancy label

Expected `text` layout
----------------------
    ### EHR: ###Demographics###
    Age: <str>
    Gender: <str>

    ###Full Active Problem List###:
    <item> - <item> - ...

    ###Recent Encounters (Max 10)###

    Diagnoses (Past Year): <item> - <item> - ...
    Diagnoses (Older): <item> - <item> - ...

    ###Medications (Outpatient)###

    Active (Start Date Before Message, Not Yet Ended):
    -<med>
    -<med>

    ### Patient Message: <free text>

Output record
-------------
    {
      "patient_message": str,
      "demographics": {"age": str, "gender": str},
      "problem_list": [str, ...],
      "diagnoses": {"past_year": [str, ...], "older": [str, ...]},
      "medications": [str, ...],
      "level": int | None,
      "relevancy": int | None,
      "id": "<PREFIX>-<NN>"
    }

Usage
-----
    python parse_pmr_parquet.py INBOX_A-one_pt.parquet -o out.json
    python parse_pmr_parquet.py *.parquet -o corpus.json --keep-empty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The sample JSON contains two parser artifacts (see README notes in the
# accompanying message). Set to True to reproduce them bug-for-bug so new
# records are byte-compatible with an already-parsed corpus; False (default)
# to emit clean data.
LEGACY_QUIRKS = False

# Section markers, in document order.
MSG_MARKER = "### Patient Message:"
SEC_DEMOGRAPHICS = "###Demographics###"
SEC_PROBLEMS = "###Full Active Problem List###"
SEC_ENCOUNTERS = "###Recent Encounters (Max 10)###"
SEC_MEDS = "###Medications (Outpatient)###"

# Items inside list-valued fields are joined with " - ".
LIST_SEP = re.compile(r"\s+-\s+")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _split_items(blob: str) -> list[str]:
    """Split a ' - '-delimited run into a clean list of items."""
    blob = (blob or "").strip()
    if not blob:
        return []
    return [p.strip() for p in LIST_SEP.split(blob) if p.strip()]


def _section(text: str, start: str, end: str | None) -> str:
    """Return the text between two markers. Empty string if `start` absent."""
    i = text.find(start)
    if i == -1:
        return ""
    i += len(start)
    if end:
        j = text.find(end, i)
        if j != -1:
            return text[i:j]
    return text[i:]


def _to_int(value) -> int | None:
    """Cast a possibly-null numeric label to int, preserving None."""
    if value is None or pd.isna(value):
        return None
    return int(value)


# --------------------------------------------------------------------------
# Field parsers
# --------------------------------------------------------------------------

def parse_demographics(ehr: str) -> dict[str, str]:
    block = _section(ehr, SEC_DEMOGRAPHICS, SEC_PROBLEMS)
    age = re.search(r"Age:\s*(.*)", block)
    gender = re.search(r"Gender:\s*(.*)", block)
    return {
        "age": age.group(1).strip() if age else "",
        "gender": gender.group(1).strip() if gender else "",
    }


def parse_problem_list(ehr: str) -> list[str]:
    block = _section(ehr, SEC_PROBLEMS, SEC_ENCOUNTERS)
    return _split_items(block.lstrip(":").strip())


def parse_diagnoses(ehr: str) -> dict[str, list[str]]:
    block = _section(ehr, SEC_ENCOUNTERS, SEC_MEDS)

    past = re.search(r"Diagnoses \(Past Year\):(.*)", block)
    older = re.search(r"Diagnoses \(Older\):(.*)", block)

    out = {
        "past_year": _split_items(past.group(1) if past else ""),
        "older": _split_items(older.group(1) if older else ""),
    }

    # Legacy artifact: when "Diagnoses (Older):" is blank, the original parser
    # ran on and captured the next section header as a diagnosis.
    if LEGACY_QUIRKS and not out["older"]:
        out["older"] = [SEC_MEDS]

    return out


def parse_medications(ehr: str) -> list[str]:
    block = _section(ehr, SEC_MEDS, None)
    meds: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            meds.append(line.lstrip("-").strip())
        elif LEGACY_QUIRKS and line.endswith(":"):
            # Legacy artifact: sub-headers such as
            # "Recently Ended Within 30-Days of Message:" were kept as items.
            meds.append(line)
    return meds


def parse_record(text: str) -> dict:
    """Parse one packed `text` blob into the structured record body."""
    if MSG_MARKER in text:
        ehr, message = text.split(MSG_MARKER, 1)
    else:
        ehr, message = text, ""

    return {
        "patient_message": message.strip(),
        "demographics": parse_demographics(ehr),
        "problem_list": parse_problem_list(ehr),
        "diagnoses": parse_diagnoses(ehr),
        "medications": parse_medications(ehr),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def infer_prefix(path: Path) -> str:
    """'INBOX_A-one_pt.parquet' -> 'INBOX_A'."""
    stem = path.stem
    return re.sub(r"[-_]one[-_]pt$", "", stem, flags=re.IGNORECASE)


def parquet_to_records(
    path: Path,
    prefix: str | None = None,
    keep_empty: bool = False,
    id_width: int = 2,
) -> list[dict]:
    """Convert one parquet file into a list of schema-conformant records."""
    df = pd.read_parquet(path)

    missing = {"text", "level", "relevancy"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {sorted(missing)}")

    prefix = prefix or infer_prefix(path)
    records, skipped = [], 0

    for i, row in df.reset_index(drop=True).iterrows():
        text = row["text"]

        # Null / blank rows are padding in the export, not patients.
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            if not keep_empty:
                continue
            rec = {
                "patient_message": "",
                "demographics": {"age": "", "gender": ""},
                "problem_list": [],
                "diagnoses": {"past_year": [], "older": []},
                "medications": [],
            }
        else:
            rec = parse_record(text)

        rec["level"] = _to_int(row["level"])
        rec["relevancy"] = _to_int(row["relevancy"])
        rec["id"] = f"{prefix}-{i:0{id_width}d}"
        records.append(rec)

    print(
        f"  {path.name}: {len(df)} rows -> {len(records)} records "
        f"({skipped} null/blank {'kept' if keep_empty else 'skipped'})",
        file=sys.stderr,
    )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", nargs='+', type=Path, help="input .parquet file(s)")
    ap.add_argument("-o", "--out", type=Path, default=Path("pmr_synth_parsed.json"))
    ap.add_argument("--prefix", default=None,
                    help="ID prefix (default: inferred from filename)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="emit placeholder records for null rows instead of skipping")
    ap.add_argument("--id-width", type=int, default=2,
                    help="zero-padding width for the numeric ID suffix (default: 2)")

    # Explicitly provide arguments for Colab environment
    # This ensures argparse doesn't pick up internal Colab kernel arguments.
    mock_args = [
        "/content/INBOX_A-00000-of-00001.parquet", # The actual parquet file
        "-o", "ckm_pts_jsn.json" # Explicitly setting output file name
    ]
    args, unknown = ap.parse_known_args(args=mock_args)

    print(args)

    all_records: list[dict] = []
    for path in args.parquet:
        all_records.extend(
            parquet_to_records(path, args.prefix, args.keep_empty, args.id_width)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(all_records, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(all_records)} record(s) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
