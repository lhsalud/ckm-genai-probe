#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umls_ckm_mapper.py
==================
Replace substring matching with a UMLS-mediated translation from free-text problem
lists to SNOMED CT, for CKM phenotyping.

Methodology (after Fung 2017, NLM/LHNCBC "Medical Terminologies in Action"):
  - Free-text problem/diagnosis terms are normalized to UMLS concepts (CUIs) by the
    UMLS search service -- the "normalized lexical matching to the UMLS" the CORE
    Problem List Subset work relies on.
  - Search is restricted to the DISORDERS semantic group (Fung filtered extracted
    concepts to the disorder semantic group; here `semantic_groups="DISO"`).
  - The CUI is the interlingua; we crosswalk it to SNOMED CT (the Meaningful-Use
    standard for problem lists) by pulling the concept's SNOMEDCT_US atoms.
  - CKM membership uses SNOMED is-a subsumption so that a specific term (e.g.
    "Morbid obesity", 238136002) maps to a CKM parent concept (Obesity, 414916001)
    via `get_source_ancestors` -- solving the parent/descendant gap that plain code
    equality misses.

Pipeline per term:  text --search--> CUI --get_atoms(SNOMEDCT_US)--> SCTID(s)
                    --exact or ancestor match--> CKM axis + Table-4 role

LIVE USE requires a UMLS API key (free with a UMLS license, uts.nlm.nih.gov) and
network access to uts-ws.nlm.nih.gov. This file also ships a MockUMLSClient so the
pipeline logic runs offline; the __main__ demo uses it.

    from umls_ckm_mapper import UMLSCKMMapper
    mapper = UMLSCKMMapper(api_key="YOUR-UMLS-KEY")
    ckm = mapper.flag_ckm(record["problem_list"],
                          record["diagnoses"]["past_year"],
                          record["diagnoses"]["older"])
    # ckm has the same shape as pmr_synth_parser.flag_ckm, so approx_ckm_stage() works.
"""
from __future__ import annotations
import json

# Reuse the SNOMED-anchored CKM value set + Table-4 roles already defined.
from pmr_synth_parser import CKM_TERMS, approx_ckm_stage

# index the value set by SNOMED code for O(1) membership tests
CKM_BY_SCTID = {
    c["sctid"]: {"axis": axis, "role": c["role"], "pt": c["pt"]}
    for axis, concepts in CKM_TERMS.items() for c in concepts
}

# --------------------------------------------------------------------------- #
# Response helpers (the client returns UMLS REST JSON; tolerate str or dict)
# --------------------------------------------------------------------------- #
def _payload(resp):
    if isinstance(resp, str):
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return {}
    return resp or {}


def _search_results(resp) -> list:
    d = _payload(resp)
    if isinstance(d, list):
        return d
    r = d.get("result", d)
    return r.get("results", []) if isinstance(r, dict) else (r or [])


def _list_result(resp) -> list:
    d = _payload(resp)
    if isinstance(d, list):
        return d
    return d.get("result", []) or []


def _sctid_from_atom(atom: dict) -> str:
    code = atom.get("code") or ""              # e.g. ".../source/SNOMEDCT_US/44054006"
    tail = code.rstrip("/").split("/")[-1] if code else atom.get("ui", "")
    return tail if tail.isdigit() else ""


# --------------------------------------------------------------------------- #
# Mapper
# --------------------------------------------------------------------------- #
class UMLSCKMMapper:
    def __init__(self, api_key: str | None = None, version: str = "current",
                 use_subsumption: bool = True, client=None):
        if client is not None:
            self.client = client                      # inject a mock/test client
        else:
            from umls_python_client import UMLSClient
            self.client = UMLSClient(api_key=api_key, version=version)
        self.use_subsumption = use_subsumption
        self._cache: dict[str, dict | None] = {}

    # text -> UMLS concept (CUI) -> SNOMED CT codes
    def text_to_snomed(self, text: str) -> dict | None:
        results = _search_results(self.client.searchAPI.search(
            text, sabs="SNOMEDCT_US", return_id_type="concept",
            semantic_groups="DISO", search_type="words",
            page_size=1, return_indented=False, format="json"))
        if not results:
            return None
        cui = results[0].get("ui")
        name = results[0].get("name")
        if not cui or cui in ("NONE", "None"):
            return None
        atoms = _list_result(self.client.cuiAPI.get_atoms(
            cui, sabs="SNOMEDCT_US", page_size=50, return_indented=False, format="json"))
        sctids = sorted({s for s in (_sctid_from_atom(a) for a in atoms) if s})
        return {"cui": cui, "name": name, "sctids": sctids}

    # is this SNOMED code a CKM concept, directly or via an is-a ancestor?
    def _ckm_hit(self, sctid: str) -> dict | None:
        if sctid in CKM_BY_SCTID:
            return {**CKM_BY_SCTID[sctid], "via_sctid": sctid, "relation": "exact"}
        if self.use_subsumption:
            anc = _list_result(self.client.sourceAPI.get_source_ancestors(
                "SNOMEDCT_US", sctid, page_size=200, return_indented=False, format="json"))
            for a in anc:
                code = a.get("ui", "")
                if code in CKM_BY_SCTID:
                    return {**CKM_BY_SCTID[code], "via_sctid": code, "relation": "ancestor"}
        return None

    # one free-text term -> CKM hit (cached; usage is skewed, so caching pays off)
    def condition_to_ckm(self, text: str) -> dict | None:
        if text in self._cache:
            return self._cache[text]
        hit = None
        mapped = self.text_to_snomed(text)
        if mapped:
            for sctid in mapped["sctids"]:
                found = self._ckm_hit(sctid)
                if found:
                    hit = {"input_text": text, "cui": mapped["cui"],
                           "normalized_name": mapped["name"], **found}
                    break
        self._cache[text] = hit
        return hit

    # drop-in replacement for pmr_synth_parser.flag_ckm (same output shape)
    def flag_ckm(self, problem_list, diagnoses_past_year=(), diagnoses_older=()) -> dict:
        matches: dict[str, list] = {}
        for term in list(problem_list) + list(diagnoses_past_year) + list(diagnoses_older):
            hit = self.condition_to_ckm(term)
            if hit:
                matches.setdefault(hit["axis"], []).append({
                    "sctid": hit["via_sctid"], "pt": hit["pt"], "role": hit["role"],
                    "cui": hit["cui"], "via_text": term, "relation": hit["relation"]})
        return {
            "is_ckm": bool(matches),
            "axes": sorted(matches),
            "concepts": {ax: [h["pt"] for h in hs] for ax, hs in matches.items()},
            "matches": matches,
            "method": "umls-python-client: search(DISO) -> CUI -> SNOMEDCT_US atoms -> is-a subsumption",
        }


# --------------------------------------------------------------------------- #
# Offline mock client (mimics the umls-python-client surface for testing)
# --------------------------------------------------------------------------- #
class _MockSearch:
    _MAP = {  # free text -> (CUI, canonical name)
        "type 2 diabetes mellitus": ("C0011860", "Diabetes Mellitus, Non-Insulin-Dependent"),
        "essential hypertension": ("C0085580", "Essential Hypertension"),
        "atrial fibrillation, unspecified type": ("C0004238", "Atrial Fibrillation"),
        "morbid obesity with bmi of 50.0-59.9, adult": ("C0028756", "Obesity, Morbid"),
    }
    def search(self, s, **k):
        cui_name = next((v for key, v in self._MAP.items() if key in s.lower()), None)
        results = [{"ui": cui_name[0], "name": cui_name[1], "rootSource": "SNOMEDCT_US"}] if cui_name else []
        return {"result": {"results": results}}

class _MockCui:
    _ATOMS = {  # CUI -> SNOMED atom code URLs
        "C0011860": ["44054006"], "C0085580": ["59621000"],
        "C0004238": ["49436004"], "C0028756": ["238136002"],  # Morbid obesity (descendant)
    }
    def get_atoms(self, cui, **k):
        base = "https://uts-ws.nlm.nih.gov/rest/content/current/source/SNOMEDCT_US/"
        return {"result": [{"code": base + c, "name": "atom"} for c in self._ATOMS.get(cui, [])]}

class _MockSource:
    _ANC = {"238136002": ["414916001", "238131007"]}  # Morbid obesity is-a Obesity (414916001)
    def get_source_ancestors(self, source, id, **k):
        return {"result": [{"ui": a, "name": "ancestor"} for a in self._ANC.get(id, [])]}

class MockUMLSClient:
    def __init__(self):
        self.searchAPI = _MockSearch()
        self.cuiAPI = _MockCui()
        self.sourceAPI = _MockSource()


# --------------------------------------------------------------------------- #
def _demo():
    mapper = UMLSCKMMapper(client=MockUMLSClient())
    problem_list = ["Morbid obesity with BMI of 50.0-59.9, adult", "Essential hypertension"]
    past_year = ["Atrial fibrillation, unspecified type", "Type 2 diabetes mellitus"]

    ckm = mapper.flag_ckm(problem_list, past_year)
    print("UMLS-mediated CKM flags (offline mock):")
    for axis, hits in ckm["matches"].items():
        for h in hits:
            print(f"  {axis:14s} '{h['via_text'][:38]:38s}' -> CUI {h['cui']} "
                  f"-> SNOMED {h['sctid']} ({h['pt']}) [{h['relation']}]")
    print("  method:", ckm["method"])
    stage = approx_ckm_stage(ckm)
    print(f"\n  approx CKM stage: {stage['stage']} - {stage['label']}")
    print("\n  Note the 'Morbid obesity' term mapped to Obesity via an is-a ANCESTOR,")
    print("  which plain SNOMED-code equality (or substring matching) would miss.")


if __name__ == "__main__":
    _demo()
