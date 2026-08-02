#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ckm_stage_cascade.py
--------------------
CKM Stage 0-4 from a list of SNOMED CT codes, via is-a subsumption and a top-down
cascade. Same shape as the hand-written version, with three corrections:

  1. ESRD / kidney failure is NOT a standalone Stage 4 trigger. Per Table 4 it is a
     Stage 3 very-high-risk-CKD equivalent on its own, and only the 4b modifier when
     clinical CVD is also present.
  2. Generic CKD is Stage 2 (moderate-high-risk CKD), not Stage 3. (Very-high-risk
     needs eGFR/UACR that aren't in the text; only kidney failure is treated as such.)
  3. Stage 4 requires clinical CVD to OVERLAP a CKM risk factor (Table 4). Toggle with
     require_overlap; isolated clinical CVD with no risk factor -> not CKM.

GROUNDING: Ndumele et al. 2026 (Circulation, DOI 10.1161/CIR.0000000000001453),
Sec 2.3 / Table 4.

The ontology helper's is_a(code, parent) MUST be reflexive (is_a(X, X) is True) so a
patient coded exactly at a parent concept still matches its tier.
"""
import json

# --------------------------------------------------------------------------- #
# Tier parent concepts (SNOMED CT). Descendants are caught via is-a subsumption.
# VERIFIED = confirmed against SNOMED browser / NLM this session.
# VERIFY   = plausible but unconfirmed here -> check on browser.ihtsdotools.org
#            before trusting; a wrong parent pulls a whole subtree into the wrong tier.
# --------------------------------------------------------------------------- #

# Clinical CVD (Table 4: CHD, HF, stroke, PAD, AF) -> Stage 4 (with overlap)
CLINICAL_CVD = [
    "84114007",   # Heart failure                 (VERIFIED)
    "22298006",   # Myocardial infarction         (VERIFIED)
    "53741008",   # Coronary arteriosclerosis/CHD (VERIFIED)
    "49436004",   # Atrial fibrillation           (VERIFIED)
    "230690007",  # Cerebrovascular accident      (VERIFY)
    "40275004",   # Peripheral vascular disease   (VERIFY)
]

# Kidney failure / very-high-risk CKD -> Stage 3 alone, or 4b modifier with clinical CVD
VERY_HIGH_CKD = [
    "46177005",   # End-stage renal disease       (VERIFIED)
    # add verified CKD stage 4/5 parents here after checking, e.g. CKD stage 5
]

# Metabolic risk factors and moderate-high-risk CKD -> Stage 2
STAGE_2_METABOLIC_CKD = [
    "44054006",   # Type 2 diabetes mellitus      (VERIFIED)
    "73211009",   # Diabetes mellitus             (VERIFIED)
    "38341003",   # Hypertensive disorder         (VERIFIED)
    "59621000",   # Essential hypertension        (VERIFIED)
    "55822004",   # Hyperlipidemia                (VERIFIED)
    "13644009",   # Hypercholesterolemia          (VERIFIED)
    "709044004",  # Chronic kidney disease        (VERIFIED)  <- moved here from Stage 3
    "90688005",   # Chronic renal failure         (VERIFIED)
    "34436003",   # Albuminuria                   (VERIFIED)
]

# Adiposity -> Stage 1
STAGE_1_ADIPOSITY = [
    "414916001",  # Obesity                       (VERIFY)
    # "238131007",# Overweight                    (VERIFY)
    # "15777000", # Prediabetes                   (VERIFY)
]


def _results(resp):
    """Pull a list out of a UMLS response (dict or JSON string)."""
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError:
            return []
    if isinstance(resp, list):
        return resp
    r = resp.get("result", resp)
    return r.get("results", []) if isinstance(r, dict) else (r or [])


class SnomedOntology:
    """is_a(code, parent) backed by UMLS get_source_ancestors. Reflexive and cached:
    one API call per unique code, then O(1) membership. Pass any object with the same
    is_a(code, parent) method (e.g. a local transitive-closure loader) to go offline."""

    def __init__(self, client):
        self.client = client
        self._ancestors = {}          # sctid -> set of ancestor sctids (including self)

    def _ancestor_set(self, code):
        if code not in self._ancestors:
            anc = {code}              # reflexive: a concept is-a itself
            try:
                for a in _results(self.client.sourceAPI.get_source_ancestors(
                        "SNOMEDCT_US", code, page_size=500,
                        return_indented=False, format="json")):
                    if a.get("ui"):
                        anc.add(a["ui"])
            except Exception as e:
                print(f"    ! ancestors failed for {code}: {e}")
            self._ancestors[code] = anc
        return self._ancestors[code]

    def is_a(self, code, parent):
        return parent in self._ancestor_set(code)


def approx_ckm_stage(patient_snomed_codes, snomed_ontology, require_overlap=True):
    """Return CKM stage (int 0-4) for a list of SNOMED codes.
    require_overlap: enforce Table 4's rule that clinical CVD counts as Stage 4 only
    when it overlaps a CKM risk factor (adiposity / metabolic / CKD)."""

    def matches(parent_list):
        return any(snomed_ontology.is_a(code, parent)
                   for code in patient_snomed_codes for parent in parent_list)

    has_cvd = matches(CLINICAL_CVD)
    has_kidney_failure = matches(VERY_HIGH_CKD)
    has_ckm_rf = (matches(STAGE_1_ADIPOSITY)
                  or matches(STAGE_2_METABOLIC_CKD)
                  or has_kidney_failure)          # adiposity / metabolic / CKD

    # Top-down cascade
    if has_cvd:
        if has_ckm_rf or not require_overlap:
            return 4                              # (4b if has_kidney_failure else 4a)
        return 0                                  # isolated clinical CVD -> not CKM (Table 4)
    if has_kidney_failure:
        return 3                                  # very-high-risk CKD equivalent
    if matches(STAGE_2_METABOLIC_CKD):
        return 2
    if matches(STAGE_1_ADIPOSITY):
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Offline self-test with a mock ontology (no key / network needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    class MockOntology:
        # code -> its ancestors (self added automatically); mirrors is-a subsumption
        ANC = {
            "408512008": {"414916001"},   # Severe obesity  is-a  Obesity
            "238136002": {"414916001"},   # Morbid obesity  is-a  Obesity
        }
        def is_a(self, code, parent):
            return parent == code or parent in self.ANC.get(code, set())

    onto = MockOntology()
    cases = {
        "Severe obesity only [408512008]":               (["408512008"], 1),
        "ESRD only [46177005] (no CVD)":                  (["46177005"], 3),
        "CKD only [709044004]":                           (["709044004"], 2),
        "T2DM + AF [44054006,49436004] (overlap)":        (["44054006", "49436004"], 4),
        "AF only [49436004] (isolated CVD, overlap on)":  (["49436004"], 0),
        "AF only, overlap OFF":                           (["49436004"], 4),
        "nothing relevant [22222222]":                    (["22222222"], 0),
    }
    print("Corrected cascade self-test (mock ontology):\n")
    for label, (codes, expected) in cases.items():
        overlap = "OFF" not in label
        got = approx_ckm_stage(codes, onto, require_overlap=overlap)
        flag = "OK" if got == expected else f"!! expected {expected}"
        print(f"  Stage {got}  {flag:14s} {label}")
    print("\n  Severe obesity -> 1 (not 0); ESRD-only -> 3 (not 4); CKD -> 2 (not 3);")
    print("  isolated AF -> 0 with overlap on, 4 with it off. All per Table 4.")
