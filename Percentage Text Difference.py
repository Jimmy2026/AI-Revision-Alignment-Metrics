"""
AI-attributable revision scoring (P0 -> AI -> P1)

Key behavior:
- If P1 is blank/empty, we treat it as "no revision" and set P1 := P0.
  (Because participants see AI only after writing P0; leaving P1 blank means they didn't revise.)

Metrics (per response):
1) Total change made: how much FINAL differs from P0
   total_change = 1 - sim(P0, FINAL)

2) AI-directed change: how much FINAL moved toward AI compared to P0
   ai_directed_change = max(0, sim(FINAL, AI) - sim(P0, AI))

3) AI attribution ratio: what fraction of the total change aligns with AI
   ai_attribution = ai_directed_change / total_change  (0 if total_change==0)

Similarity metric: TF-IDF cosine similarity (word-level, stopwords removed).

Requirements:
  pip install scikit-learn
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Scores:
    revised: bool                 # whether participant actually entered a revised response
    final_text_used: str          # "P1" or "P0" (if P1 blank, we use P0 as final)
    sim_p0_final: float
    sim_p0_ai: float
    sim_final_ai: float
    total_change: float
    ai_directed_change: float
    ai_attribution: float         # fraction of total_change that is AI-aligned
    notes: str = ""


def _clean(t: str) -> str:
    return (t or "").strip()


def tfidf_cosine_similarity(a: str, b: str) -> float:
    """Compute TF-IDF cosine similarity between two texts, in [0, 1]."""
    a = _clean(a)
    b = _clean(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    vect = TfidfVectorizer(stop_words="english")
    tfidf = vect.fit_transform([a, b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def score_triplet(p0: str, ai: str, p1: str) -> Scores:
    """
    Compute revision metrics given:
      p0 = participant initial
      ai = AI suggestion
      p1 = participant revised (may be blank -> treated as no revision)
    """
    p0c = _clean(p0)
    aic = _clean(ai)
    p1c = _clean(p1)

    # Interpret blank P1 as "no revision": FINAL := P0
    revised = bool(p1c)
    if not revised:
        final = p0c
        final_text_used = "P0"
    else:
        final = p1c
        final_text_used = "P1"

    sim_p0_final = tfidf_cosine_similarity(p0c, final)
    sim_p0_ai = tfidf_cosine_similarity(p0c, aic)
    sim_final_ai = tfidf_cosine_similarity(final, aic)

    total_change = max(0.0, 1.0 - sim_p0_final)
    ai_directed_change = max(0.0, sim_final_ai - sim_p0_ai)

    notes = ""
    if not aic:
        notes = "AI text missing/blank; AI-directed metrics may be uninformative."
    if total_change == 0.0:
        # either truly unchanged OR both empty
        notes = (notes + " " if notes else "") + "No meaningful revision detected (P0≈FINAL)."

    ai_attribution = 0.0 if total_change == 0 else min(1.0, ai_directed_change / total_change)

    return Scores(
        revised=revised,
        final_text_used=final_text_used,
        sim_p0_final=sim_p0_final,
        sim_p0_ai=sim_p0_ai,
        sim_final_ai=sim_final_ai,
        total_change=total_change,
        ai_directed_change=ai_directed_change,
        ai_attribution=ai_attribution,
        notes=notes.strip(),
    )


def pct(x: float) -> float:
    return round(x * 100.0, 2)


def print_scores(scores: Scores) -> None:
    print("\nFlags:")
    print(f"- revised: {scores.revised}")
    print(f"- final_text_used: {scores.final_text_used} (blank P1 => treated as no revision)")

    print("\nSimilarity (TF-IDF cosine):")
    print(f"- sim(P0, FINAL): {pct(scores.sim_p0_final)}%  (higher = less changed)")
    print(f"- sim(P0, AI):    {pct(scores.sim_p0_ai)}%")
    print(f"- sim(FINAL, AI): {pct(scores.sim_final_ai)}%")

    print("\nRevision metrics:")
    print(f"- Total change (P0→FINAL): {pct(scores.total_change)}%")
    print(f"- AI-directed change:      {pct(scores.ai_directed_change)}%")
    print(f"- AI attribution share:    {pct(scores.ai_attribution)}%")

    if scores.notes:
        print(f"\nNote: {scores.notes}")


def read_multiline(prompt: str) -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def interactive_single() -> None:
    p0 = read_multiline("Paste P0 (participant initial). End with an empty line:")
    ai = read_multiline("\nPaste AI (AI suggestion). End with an empty line:")
    p1 = read_multiline("\nPaste P1 (participant revised; can be blank). End with an empty line:")

    scores = score_triplet(p0, ai, p1)
    print_scores(scores)


def score_csv(input_path: str, output_path: str) -> None:
    """
    Batch score from CSV.

    Input CSV headers:
      id,p0,ai,p1

    Output includes:
      revised,final_text_used,sim_p0_final,sim_p0_ai,sim_final_ai,
      total_change,ai_directed_change,ai_attribution,notes
    """
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "p0", "ai", "p1"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must include headers: {', '.join(sorted(required))}")
        rows = list(reader)

    out_rows: List[Dict[str, str]] = []
    for r in rows:
        sid = r.get("id", "")
        s = score_triplet(r.get("p0", ""), r.get("ai", ""), r.get("p1", ""))

        out = dict(r)
        out.update({
            "revised": str(s.revised),
            "final_text_used": s.final_text_used,
            "sim_p0_final": str(round(s.sim_p0_final, 6)),
            "sim_p0_ai": str(round(s.sim_p0_ai, 6)),
            "sim_final_ai": str(round(s.sim_final_ai, 6)),
            "total_change": str(round(s.total_change, 6)),
            "ai_directed_change": str(round(s.ai_directed_change, 6)),
            "ai_attribution": str(round(s.ai_attribution, 6)),
            "notes": s.notes,
        })
        out_rows.append(out)

    fieldnames = list(out_rows[0].keys()) if out_rows else ["id", "p0", "ai", "p1"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved scored CSV to: {output_path}")


def usage() -> None:
    print(
        "Usage:\n"
        "  python ai_attribution.py               # interactive (paste P0, AI, P1)\n"
        "  python ai_attribution.py in.csv out.csv  # batch score CSV\n\n"
        "CSV format:\n"
        "  id,p0,ai,p1\n"
    )


def main(argv: List[str]) -> None:
    if len(argv) == 1:
        interactive_single()
        return
    if len(argv) == 3:
        score_csv(argv[1], argv[2])
        return
    usage()
    sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
