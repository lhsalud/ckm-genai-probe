#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_snomed_to_problem_list.py
-----------------------------
For each patient in the input JSON, resolve every problem_list entry to a SNOMED CT
concept via the UMLS API, add a new key `problem_list_snomed`, and save to a new file.

Pipeline per problem:  free text  --UMLS search (SNOMEDCT_US)-->  SNOMED CT code + name

Requires the `umls-python-client` package and a UMLS API key (free with a UMLS
license at https://uts.nlm.nih.gov/uts/).  Install:  pip install umls-python-client
"""
import json
from umls_python_client import UMLSClient

# ============================ CONFIG ============================
UMLS_API_KEY = "PASTE-YOUR-UMLS-API-KEY-HERE"          # <-- put your key here
INPUT_FILE   = "pmr_synth_parsed_sample.json"          # input patient JSON
OUTPUT_FILE  = "pmr_synth_parsed_sample_snomed.json"   # output with SNOMED added
# ===============================================================


def _results(resp):
    """Pull the results list out of a UMLS response (may be a dict or JSON string)."""
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError:
            return []
    if isinstance(resp, list):
        return resp
    r = resp.get("result", resp)
    return r.get("results", []) if isinstance(r, dict) else (r or [])


def _search_cui(client, text, search_type):
    """One search attempt -> (cui, name) or (None, None)."""
    resp = client.searchAPI.search(
        text, return_id_type="concept", search_type=search_type,
        page_size=1, return_indented=False, format="json")
    res = _results(resp)
    if res and res[0].get("ui") not in (None, "NONE", "None"):
        return res[0]["ui"], res[0].get("name")
    return None, None


def _simplify(text):
    """Reduce a verbose ICD/billing description to its clinical core.
    e.g. 'Class 3 severe obesity with body mass index (BMI) of 45.0 to 49.9 in
    adult, unspecified ...'  ->  'severe obesity'."""
    import re
    s = text.split(",")[0]                                  # drop trailing ICD clauses
    s = re.sub(r"\([^)]*\)", " ", s)                        # remove parentheticals (BMI), (CMC)
    s = re.split(r"\bwith\b", s, maxsplit=1, flags=re.I)[0] # cut at " with <measurement/qualifier>"
    s = re.sub(r"^\s*(class|type|grade|stage)\s+\S+\s+", "", s, flags=re.I)  # drop 'Class 3 ' prefix
    return re.sub(r"\s+", " ", s).strip()


def _query_variants(text):
    """Ordered, de-duplicated queries to try: most specific first."""
    variants = [text]
    core = text.split(",")[0].strip()
    if core and core.lower() != text.lower():
        variants.append(core)
    simplified = _simplify(text)
    if simplified and simplified.lower() not in (v.lower() for v in variants):
        variants.append(simplified)
    return variants


def resolve_to_snomed(client, text):
    """Free text -> UMLS concept (CUI) -> SNOMED CT code + name.

    Searches ALL vocabularies (not just SNOMED) so ICD-style phrasing matches, then
    crosswalks the CUI to its SNOMED CT atom. Tries progressively simpler queries
    (full string -> text before first comma -> cleaned clinical core) with VALID
    search types only ('words', 'normalizedString'; NOTE 'approximate' is NOT valid
    for UMLS /search). Returns a null record if nothing resolves.

    NOTE: verbose ICD/billing diagnosis rubrics resolve less reliably than clinician
    problem-list terms; the problem_list is the source intended for SNOMED encoding.
    """
    null = {"input_text": text, "cui": None, "sctid": None, "snomed_name": None}

    cui = cui_name = None
    for query in _query_variants(text):
        for stype in ("words", "normalizedString"):
            try:
                cui, cui_name = _search_cui(client, query, stype)
            except Exception as e:                          # try next variant, don't abort
                print(f"    ! search failed ({stype}) for {query!r}: {e}")
                continue
            if cui:
                break
        if cui:
            break
    if not cui:
        return null

    # Crosswalk the concept to SNOMED CT by pulling its SNOMEDCT_US atom(s).
    try:
        atoms = _results(client.cuiAPI.get_atoms(
            cui, sabs="SNOMEDCT_US", page_size=25, return_indented=False, format="json"))
    except Exception as e:
        print(f"    ! get_atoms failed for {cui} ({text!r}): {e}")
        return {**null, "cui": cui}
    for atom in atoms:
        code = (atom.get("code") or "").rstrip("/").split("/")[-1]
        if code.isdigit():
            return {"input_text": text, "cui": cui,
                    "sctid": code, "snomed_name": atom.get("name") or cui_name}
    return {**null, "cui": cui}   # concept found, but it has no SNOMED CT atom


def main():
    client = UMLSClient(api_key=UMLS_API_KEY)

    with open(INPUT_FILE) as f:
        patients = json.load(f)

    cache = {}   # remember terms we've already looked up (problem lists repeat a lot)
    for patient in patients:
        problem_snomed = []
        for problem in patient.get("problem_list", []):
            if problem not in cache:
                cache[problem] = resolve_to_snomed(client, problem)
            problem_snomed.append(cache[problem])
        patient["problem_list_snomed"] = problem_snomed          # append the new key
        n_ok = sum(1 for p in problem_snomed if p["sctid"])
        print(f"{patient.get('id', '?')}: resolved {n_ok}/{len(problem_snomed)} problems")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(patients, f, indent=2)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
