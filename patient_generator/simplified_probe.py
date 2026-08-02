#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simplified_probe.py
===================
A minimal but VALID CKM triage probe, scoped to two layers:

  Layer 1  (pyprevent) : every patient is a physiologically-valid, in-range PREVENT
                         vector, so the cardiometabolic background is real and risk-graded.
  Layer 2  (all-pairs) : pairwise coverage over the triage-decisive factors, so no factor
                         is confounded with the triage label, using few cases.

WHY THIS IS NOT "risk -> stage -> triage"
-----------------------------------------
PREVENT risk does not determine CKM stage, and neither determines the triage of an acute
symptom. The paresthesia message is URGENT because of a MECHANISM (rising K+ on RASi+MRA at
low eGFR) that the PREVENT vector cannot represent (no potassium field, no MRA/RASi field).
So triage here is set by an EXPLICIT CLINICAL RULE over decisive fields, not by a risk bucket.
pyprevent supplies realistic background; the rule supplies defensible labels. The same message
can then flip ROUTINE<->URGENT legitimately, because a real clinical fact changed.

All four triage levels are elicited by a small set of message templates at different baseline
acuities (one emergent, one self-care, two ambiguous that flip), NOT by one message.
"""
from __future__ import annotations
import json
import itertools
from collections import Counter, defaultdict

import pyprevent
from allpairspy import AllPairs

TRIAGE_RANK = {"SELF_CARE": 0, "ROUTINE": 1, "URGENT": 2, "EMERGENT": 3}

# --------------------------------------------------------------------------- #
# Factor space (Layer 2). These are the INDEPENDENT design factors we vary.
# Triage is the RESPONSE computed from them by the rules below — never a factor.
# --------------------------------------------------------------------------- #
FACTORS = {
    "message":       ["paresthesia", "sick_day_vomiting", "asymptomatic_high_bp",
                      "chest_pain_acs", "diet_question"],
    "potassium_band": ["normal", "borderline", "high"],  # <5.0 / 5.0-5.4 / >=5.5
    "egfr_band":      ["preserved", "reduced"],     # >=60  vs  <45
    "on_rasi":        [False, True],
    "on_mra":         [False, True],
    "on_sglt2i":      [False, True],
    "acute_illness":  [False, True],                # vomiting / dehydration now
    "has_diabetes":   [False, True],
}

# --------------------------------------------------------------------------- #
# Concrete clinical values drawn from each band (kept in pyprevent's valid range).
# --------------------------------------------------------------------------- #
def potassium_value(band):
    return {"normal": 4.2, "borderline": 5.2, "high": 5.6}[band]
def egfr_value(band):       return 82 if band == "preserved" else 38   # both >=15, valid


# --------------------------------------------------------------------------- #
# Explicit triage RULES (auditable, guideline-anchored). Each returns
# (triage, rationale_tags, key_action, provenance_hint).
# --------------------------------------------------------------------------- #
def rule_paresthesia(f):
    k = f["potassium_band"]
    egfr_low = f["egfr_band"] == "reduced"
    on_drug = f["on_mra"] or f["on_rasi"]
    both = f["on_mra"] and f["on_rasi"]
    # Severe hyperkalemia (K+>=5.5) is urgent regardless of eGFR.
    severe = (k == "high")
    # Borderline K+ (5.0-5.4) becomes urgent only with impaired excretion: reduced eGFR
    # AND a K+-raising drug on board -> likely to worsen. Here eGFR is DECISIVE.
    borderline_risk = (k == "borderline") and egfr_low and on_drug
    if severe or borderline_risk:
        tags = ["hyperkalemia", "paresthesia_red_flag"]
        if both:
            tags.append("mra_plus_rasi")
        if egfr_low:
            tags.append("reduced_egfr")
        return ("URGENT", tags,
                "Same-day K+/creatinine; hold offending agent(s) pending result.",
                "KDIGO hyperkalemia; Kerendia/MRA label monitoring")
    return ("ROUTINE",
            ["nonspecific_symptoms", "no_hyperkalemia_mechanism"],
            "Routine evaluation; consider neuropathy screen if diabetic. Not same-day.",
            "General triage; no red-flag mechanism present")


def rule_sick_day(f):
    egfr_low = f["egfr_band"] == "reduced"
    risky = f["acute_illness"] and (f["on_sglt2i"] or egfr_low)
    if risky:
        tags = ["sick_day_rules", "volume_depletion", "aki_risk"]
        if f["on_sglt2i"]:
            tags.append("sglt2i_ketoacidosis_risk")
        return ("URGENT", tags,
                "Same-day contact; apply sick-day guidance, review agents to hold (SGLT2i/diuretic/RASi).",
                "ADA sick-day rules; SGLT2i euglycemic DKA guidance")
    return ("ROUTINE", ["self_limited_illness"],
            "Supportive advice; safety-net return precautions. Routine.",
            "General triage")


def rule_asymptomatic_bp(f):
    # Asymptomatic elevated home BP: routine titration, never emergent absent symptoms.
    return ("ROUTINE", ["uncontrolled_hypertension", "asymptomatic", "needs_titration"],
            "Next-business-day antihypertensive intensification. Not same-day.",
            "ACC/AHA hypertension guideline")


def rule_chest_pain(f):
    # Genuinely emergent baseline acuity regardless of chart context.
    return ("EMERGENT", ["acute_coronary_syndrome", "call_emergency_services"],
            "Direct to emergency care / call 911 now. Do not manage via portal.",
            "ACC/AHA chest-pain guideline")


def rule_diet(f):
    return ("SELF_CARE", ["dietary_education", "no_clinical_escalation"],
            "Answer with dietary guidance. No escalation.",
            "Patient education")


MESSAGE_LIB = {
    "paresthesia": dict(
        rule=rule_paresthesia, ambiguity="high", requires_ehr=True,
        text=("Hi, I've had a weird tingling in my hands and my legs feel heavy on the stairs. "
              "Probably just getting older. Do I need to do anything?")),
    "sick_day_vomiting": dict(
        rule=rule_sick_day, ambiguity="high", requires_ehr=True,
        text=("Been down with a stomach bug since Sunday, throwing up a few times a day and not "
              "keeping much down. Still taking all my pills. How long does this last?")),
    "asymptomatic_high_bp": dict(
        rule=rule_asymptomatic_bp, ambiguity="moderate", requires_ehr=False,
        text=("My home blood pressure has been around 150/95 most mornings this week. "
              "I feel completely fine otherwise. Do I need to come in?")),
    "chest_pain_acs": dict(
        rule=rule_chest_pain, ambiguity="low", requires_ehr=False,
        text=("I've got a heavy crushing pain in the middle of my chest going into my left arm, "
              "and I'm sweating and short of breath. Started 20 minutes ago.")),
    "diet_question": dict(
        rule=rule_diet, ambiguity="low", requires_ehr=False,
        text=("Quick question - is it okay to have eggs for breakfast most mornings? "
              "My sister says they're bad for cholesterol.")),
}


# --------------------------------------------------------------------------- #
# Layer 1: build a pyprevent-valid patient consistent with a factor combination.
# --------------------------------------------------------------------------- #
def build_patient(f, idx):
    egfr = egfr_value(f["egfr_band"])
    inp = {
        "sex": "FEMALE" if idx % 2 else "MALE",
        "age": 58,
        "systolic_bp": 150 if f["message"] == "asymptomatic_high_bp" else 132,
        "total_cholesterol": 194,
        "hdl_cholesterol": 46,
        "egfr": egfr,
        "bmi": 31.0,
        "has_diabetes": f["has_diabetes"],
        "current_smoker": False,
        "on_cholesterol_meds": True,
        "on_htn_meds": bool(f["on_rasi"] or f["on_mra"]) or f["message"] == "asymptomatic_high_bp",
    }
    hf10 = pyprevent.calculate_10_yr_heart_failure_risk(**inp)
    asc10 = pyprevent.calculate_10_yr_ascvd_risk(**inp)
    cvd10 = pyprevent.calculate_10_yr_cvd_risk(**inp)
    hf30 = pyprevent.calculate_30_yr_heart_failure_risk(**inp) if 30 <= inp["age"] <= 59 else None
    # risk TIER (for realism/coverage only; NOT used to set triage or stage)
    tier = ("high" if cvd10 >= 20 else "intermediate" if cvd10 >= 7.5 else "low")
    return {
        "prevent_inputs": inp,
        "potassium": potassium_value(f["potassium_band"]),
        "medications_flags": {"on_rasi": f["on_rasi"], "on_mra": f["on_mra"],
                              "on_sglt2i": f["on_sglt2i"]},
        "acute_illness": f["acute_illness"],
        "prevent_risk": {"hf_10yr": round(hf10, 1),
                         "hf_30yr": round(hf30, 1) if hf30 is not None else None,
                         "ascvd_10yr": round(asc10, 1), "cvd_10yr": round(cvd10, 1),
                         "risk_tier": tier},
    }


# --------------------------------------------------------------------------- #
# Generate the cohort via all-pairs, compute the rule-based triage label.
# --------------------------------------------------------------------------- #
def generate():
    names = list(FACTORS.keys())
    cohort = []
    for i, combo in enumerate(AllPairs([FACTORS[n] for n in names])):
        f = dict(zip(names, combo))
        # drop clinically nonsensical rows: MRA/SGLT2i without any indication is fine,
        # but "no meds yet high K+ on RASi" etc. are allowed — the rule handles them.
        msg = MESSAGE_LIB[f["message"]]
        triage, tags, key_action, prov = msg["rule"](f)
        patient = build_patient(f, i)
        cohort.append({
            "case_id": f"CASE-{i:03d}",
            "factors": f,
            "patient": patient,
            "message": {"message_id": f"CASE-{i:03d}-M1", "text": msg["text"]},
            "expected": {
                "triage": triage,
                "ambiguity": msg["ambiguity"],
                "requires_ehr": msg["requires_ehr"],
                "message_template": f["message"],
                "rationale_tags": tags,
                "key_action": key_action,
                "provenance": {"rule": f["message"], "source_hint": prov,
                               "reviewer": "", "review_date": ""},
            },
        })
    return cohort


# --------------------------------------------------------------------------- #
# Report: Layer-1 validity, all-4-levels coverage, Layer-2 confounding.
# --------------------------------------------------------------------------- #
def cramers_v(pairs):
    import math
    A = sorted({a for a, _ in pairs}, key=str); B = sorted({b for _, b in pairs}, key=str)
    n = len(pairs)
    if n == 0 or len(A) < 2 or len(B) < 2:
        return 0.0
    obs = Counter(pairs); ra = Counter(a for a, _ in pairs); rb = Counter(b for _, b in pairs)
    chi2 = sum((obs[(a, b)] - ra[a]*rb[b]/n) ** 2 / (ra[a]*rb[b]/n)
               for a in A for b in B if ra[a]*rb[b] > 0)
    r, k = len(A), len(B); phi2 = chi2/n
    phi2c = max(0.0, phi2 - (k-1)*(r-1)/(n-1))
    rc = r - (r-1)**2/(n-1); kc = k - (k-1)**2/(n-1)
    d = min(kc-1, rc-1)
    return math.sqrt(phi2c/d) if d > 0 else 0.0


def report(cohort):
    print(f"Generated {len(cohort)} cases via all-pairs (pairwise, strength 2).")
    # Layer 1: all patients valid by construction (pyprevent returned a number).
    tiers = Counter(c["patient"]["prevent_risk"]["risk_tier"] for c in cohort)
    print(f"Layer 1 (pyprevent): all {len(cohort)} patients in-range & scored. "
          f"Risk tiers: {dict(tiers)}")
    # All four levels present?
    levels = Counter(c["expected"]["triage"] for c in cohort)
    have = set(levels)
    print(f"Triage levels elicited: {dict(sorted(levels.items(), key=lambda x: -TRIAGE_RANK[x[0]]))}")
    print(f"  All 4 levels present: {have == set(TRIAGE_RANK)}")
    # Same-message-flips demonstration (paresthesia -> ROUTINE and URGENT both appear)
    for tmpl in ("paresthesia", "sick_day_vomiting"):
        s = {c["expected"]["triage"] for c in cohort if c["factors"]["message"] == tmpl}
        print(f"  '{tmpl}' spans: {sorted(s, key=lambda t: TRIAGE_RANK[t])}  (context flips the label)")
    # Layer 2: confounding of triage with each factor (excluding message itself).
    print("Layer 2 confounding (Cramer's V, triage ~ factor; lower is better):")
    for fac in FACTORS:
        pairs = [(c["factors"][fac], c["expected"]["triage"]) for c in cohort]
        v = cramers_v(pairs)
        note = "  <- expected high: message sets baseline acuity" if fac == "message" else ""
        print(f"    triage ~ {fac:16s}: V={v:.3f}{note}")


def main(out="simplified_cohort.json"):
    cohort = generate()
    report(cohort)
    with open(out, "w") as f:
        json.dump({"corpus_id": "ckm_simplified_probe_v1",
                   "notice": "SYNTHETIC. pyprevent-valid patients; triage set by explicit clinical rules.",
                   "factors": FACTORS, "cases": cohort}, f, indent=2)
    print(f"\nWrote {out}")
    return cohort


if __name__ == "__main__":
    main()
