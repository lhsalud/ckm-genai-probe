#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
message_similarity.py
=====================
Exploratory textual-similarity comparison between your synthetic CKM messages and
PMR-Synth. This situates your messages relative to expert-written portal messages
in embedding space as a PLAUSIBILITY check.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
- It measures TEXTUAL/STYLISTIC proximity (do your messages "read like" portal
  messages), NOT clinical validity and NOT realism of the EHR side.
- Cosine similarity here conflates topic, register, and LENGTH. Two confounds to
  state explicitly in a write-up:
    (1) LENGTH: PMR-Synth messages average ~457 tokens; your generated messages are
        1-2 sentences. Raw embedding distance will partly reflect verbosity, not
        realism. The report prints length stats so you can see and caveat this.
    (2) TOPIC: your set is CKM-specific; PMR-Synth is general primary care. Lower
        cross-similarity may reflect topic difference, not lower quality.
- N is tiny (PMR-Synth = 60; your set ~11-15). Treat everything as descriptive and
  exploratory; do not make inferential claims.

The right read is comparative, not absolute: compare WITHIN-set similarity to
CROSS-set similarity. If cross(CKM,PMR) is close to within(PMR,PMR), your messages
sit inside the portal-message distribution; if cross is much lower, they are
stylistic outliers (which length alone can cause).

EMBEDDER: default is offline TF-IDF (lexical). For a SEMANTIC comparison, plug your
existing nomic-embed-text / sentence-transformers stack into `embed_semantic()` and
pass embedder="semantic". Report which you used — they answer different questions
(lexical overlap vs meaning).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_ckm_messages(path: str) -> list[str]:
    d = json.load(open(path))
    return [c["message"]["text"] for c in d["cases"]]


def load_pmr_messages(path: str | None) -> list[str]:
    """Swap in the 60 real PMR-Synth messages once downloaded (align the key)."""
    if path:
        raw = json.load(open(path))
        return [r["message"] for r in raw]
    # inline placeholders (SHORT — real PMR-Synth is ~457 tokens; see LENGTH caveat)
    return [
        "Hey doc, having really bad shortness of breath today and my chest feels tight.",
        "Hi, I have a fever of 102 and some white spots in my throat since this morning.",
        "I fell a few days ago and have this lingering back pain that isn't improving.",
        "Been struggling with an on-and-off cough for about a month, can we meet to chat?",
        "Got a stuffy nose and congestion, thought I'd let you know. Any meds you suggest?",
        "Wondering where I can get a COVID test these days, been feeling a little sniffly.",
        "My ankle has been swollen since I twisted it at the gym on Saturday, should I worry?",
        "I've been feeling dizzy when I stand up quickly for the past week or so.",
    ]


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
def embed_tfidf(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    return vec.fit_transform(texts).toarray()


def embed_semantic(texts: list[str]) -> np.ndarray:
    """
    Plug your own stack here, e.g.:
        import ollama
        return np.array([ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"]
                         for t in texts])
    or sentence-transformers:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2"); return m.encode(texts)
    Requires network/model access, so it is not run in this offline demo.
    """
    raise NotImplementedError("Wire in nomic-embed-text or sentence-transformers locally.")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _triu_vals(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def summarize(name, vals):
    vals = np.asarray(vals, dtype=float)
    print(f"  {name:26s} n={vals.size:4d}  mean={vals.mean():.3f}  "
          f"median={np.median(vals):.3f}  sd={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")
    return vals


def rbf_mmd2(X, Y, gamma=None):
    """Unbiased RBF-kernel MMD^2 between two embedding sets (distributional distance).
    Small n -> noisy; report as exploratory only."""
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = rbf_kernel(X, X, gamma); Kyy = rbf_kernel(Y, Y, gamma); Kxy = rbf_kernel(X, Y, gamma)
    m, n = len(X), len(Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
    return (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.mean())


def word_counts(texts):
    return np.array([len(t.split()) for t in texts], dtype=float)


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyze(ckm_texts, pmr_texts, embedder="tfidf", write_projection="projection.csv"):
    embed = {"tfidf": embed_tfidf, "semantic": embed_semantic}[embedder]
    # embed jointly so the feature space is shared
    all_texts = ckm_texts + pmr_texts
    E = embed(all_texts)
    nc = len(ckm_texts)
    Ec, Ep = E[:nc], E[nc:]

    print(f"=== Message similarity: CKM (n={len(ckm_texts)}) vs PMR-Synth (n={len(pmr_texts)}) "
          f"| embedder={embedder} ===\n")

    # Length covariate (the big confound)
    wc_c, wc_p = word_counts(ckm_texts), word_counts(pmr_texts)
    print("LENGTH (confound — interpret similarity in light of this):")
    print(f"  CKM  words: mean={wc_c.mean():.0f} (min {wc_c.min():.0f}, max {wc_c.max():.0f})")
    print(f"  PMR  words: mean={wc_p.mean():.0f} (min {wc_p.min():.0f}, max {wc_p.max():.0f})")
    print("  NOTE: real PMR-Synth averages ~457 tokens; if your placeholder differs, the")
    print("        length gap will widen once you load the real data.\n")

    # Within vs cross cosine distributions
    Scc = cosine_similarity(Ec, Ec)
    Spp = cosine_similarity(Ep, Ep)
    Scp = cosine_similarity(Ec, Ep)
    print("COSINE SIMILARITY DISTRIBUTIONS (compare within vs cross):")
    within_ckm = summarize("within CKM", _triu_vals(Scc))
    within_pmr = summarize("within PMR-Synth", _triu_vals(Spp))
    cross = summarize("cross CKM<->PMR", Scp.ravel())
    # interpretation heuristic
    overlap = cross.mean() / max(within_pmr.mean(), 1e-9)
    print(f"  --> cross/within-PMR ratio = {overlap:.2f}  "
          f"({'overlapping distributions' if overlap >= 0.8 else 'CKM sits apart (topic/length/style)'})\n")

    # Nearest-neighbour proximity: for each CKM msg, closest PMR msg
    nn = Scp.max(axis=1)
    print("NEAREST-NEIGHBOUR proximity (each CKM message -> closest PMR message):")
    summarize("NN cosine", nn)
    print()

    # Distributional distance (exploratory)
    try:
        mmd = rbf_mmd2(Ec, Ep)
        print(f"DISTRIBUTIONAL DISTANCE  RBF-MMD^2 = {mmd:.4f}  (0 = identical; small-n noisy)\n")
    except Exception as e:  # noqa: BLE001
        print(f"(MMD skipped: {e})\n")

    # 2D projection for plotting overlap
    try:
        coords = PCA(n_components=2).fit_transform(E)
        import csv
        with open(write_projection, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["source", "x", "y", "text"])
            for i, t in enumerate(all_texts):
                src = "CKM" if i < nc else "PMR-Synth"
                w.writerow([src, f"{coords[i,0]:.4f}", f"{coords[i,1]:.4f}", t[:80]])
        print(f"Wrote 2D PCA coordinates to {write_projection} (plot CKM vs PMR to see overlap).")
    except Exception as e:  # noqa: BLE001
        print(f"(projection skipped: {e})")


def main(ckm_path="/mnt/user-data/uploads/simplified_cohort.json", pmr_path=None):
    ckm = load_ckm_messages(ckm_path)
    pmr = load_pmr_messages(pmr_path)
    analyze(ckm, pmr, embedder="tfidf")
    print("\nINTERPRETATION: this is a stylistic/lexical plausibility probe, not clinical")
    print("validation. Report length + topic as confounds and n as a limitation. For a")
    print("semantic version, re-run with your nomic-embed stack via embedder='semantic'.")


if __name__ == "__main__":
    main()
