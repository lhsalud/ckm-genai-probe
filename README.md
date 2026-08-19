Exploring symbolic efficacy for triaging patients with cardiovascular-kidney-metabolic syndrome
using a large language model: the case for a standards-based CKM stage approximator

Introduction: Cardiovascular-Kidney-Metabolic (CKM) syndrome is a widespread health issue that
connects heart disease, kidney disease, diabetes, fatty liver disease, and obesity. Approximately 90% of
adults in the United States are unknowingly or knowingly affected by this syndrome. It requires accurate
staging for early clinical intervention. These stages range in progressive severity levels from stage 0 (no
risk factors) to stage 4 (includes clinical cardiovascular disease). In daily clinical operations, patient portal
messages must be triaged by a clinician from the patient’s words alone. For a CKM patient, those words
are frequently ambiguous. The same complaint can be judged as self-care or an organ-threatening
decompensation. Moreover, the disambiguating signal sits in structured EHR data that clinician must
carefully synthesize. Misjudgment poses a significant challenge in healthcare as under-triage is a safety
event; and over-triage results in alert fatigue and after-hours inbox work. Recent published works address
this challenge by using generative large language models (LLMs) to make these triage judgements on
behalf of a human clinician. Objective: Our goal is to evaluate whether an LLM’s knowledge of patient’s
CKM stage helps to disambiguate patient portal messages when clinically judging their triage severity into
self-care, routine, urgent, or emergent on behalf of a human clinician.

Methods: Thirty publicly available, factually substantiated, synthetic patient-level data are translated from
free-text to standardized SNOWMED CT codes using natural language processing and a customized
UMLS-based ontology bridge from the National Library of Medicine’s API endpoint. This free-text includes
problem lists, diagnoses, and medications. A software algorithm of clinical reasoning logic is developed,
executed, and manually validated for all 30 patients to translate these codes into per-patient CKM stage
approximations. A large language model is developed and deployed to have CKM clinical treatment
expertise using Retrieval-Augmented Generation (RAG). A gold standard, single-subject A-B-A-B reversal
design is tested on this LLM to determine the causal effect of CKM stage knowledge. Aggregate outcome
and five metrics are collected at both phases of testing and compared: accuracy, predicted under triage
rate, predicted over triage rate, triage severity distance from expected, and free text parsing failure rate.

Results: The LLM’s severity triage judgment was correct in eight out of 30 messages (26.7%) consisting
of 4 correct self-care judgement and 4 correct urgent judgements. Of the remaining 22 messages (53%),
16 were incorrectly judged as over-triaged (15 urgent and 1 routine, instead of the expected 11 self-care
and 5 routine) and 6 were incorrectly judged as under-triaged (5 urgent and 1 self-care instead of the
expected 5 emergent and 1 urgent). Furthermore, when comparing the LLM’s triage severity judgment
with and without CKM stage knowledge there are no differences in accuracy, predicted under triage rate,
and predicted over triage rate. However, there is a difference between the LLM’s triage severity judgment
with and without CKM stage knowledge when measuring triage severity distance, though not statistically
significant (1.100/3 CKM stage known, 1.033/3 CKM stage unknown, p=0.16). Two of the 30 patient
messages are the origin of this difference where over-triage was predicted in both insistences. No
parsing failures resulted after testing this LLM.

Conclusion: CKM stage knowledge does not help disambiguate free text of patient-portal messages.
However, the difference in severity distance may say more about LLM’s clinical decision-making
disposition than about how efficaciously it uses CKM stage knowledge. It is notable that the LLM chose
to over-triage in both message instances causing this severity distance difference.
