"""Demo / evaluation report for the novelty scorer.

Run with no arguments to print a per-category score report over the golden
dataset. Pass --headline/--body to score your own submission and see the
full breakdown.

Examples:
    python demo.py
    python demo.py --headline "Grid can't cope" --body "Evening charging will overload old substations." --stance Oppose

Scoring a custom submission also appends it (with its stance) to the corpus in
data/submissions.json, so the corpus grows over time. Use --no-save to skip.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.novelty import NoveltyScorer

DATA = Path(__file__).resolve().parent / "data"


def _load():
    fixed = json.loads((DATA / "fixed_content.json").read_text(encoding="utf-8"))
    subs = json.loads((DATA / "submissions.json").read_text(encoding="utf-8"))["submissions"]
    scorer = NoveltyScorer(fixed["text"])
    scorer.fit(subs)
    return fixed, subs, scorer


def _report(subs, scorer):
    agg = defaultdict(list)
    for i, s in enumerate(subs):
        agg[s["label"]].append(scorer.score_existing(i).score)
    print(f"{'category':<12}{'n':>4}{'avg score':>12}{'min':>8}{'max':>8}")
    print("-" * 44)
    for label in ("novel", "normal", "paraphrase", "offtopic"):
        v = agg[label]
        print(f"{label:<12}{len(v):>4}{sum(v)/len(v):>12.3f}{min(v):>8.3f}{max(v):>8.3f}")


def _append_submission(headline: str, body: str, stance: str) -> str:
    """Append a scored submission to the corpus and return its new id."""
    path = DATA / "submissions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    subs = data["submissions"]
    new_id = f"s{len(subs) + 1:02d}"
    subs.append({
        "id": new_id,
        "label": "user",
        "stance": stance,
        "headline": headline,
        "body": body,
    })
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return new_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline")
    parser.add_argument("--body")
    parser.add_argument("--stance", choices=["Support", "Oppose", "Neutral"],
                        default="Neutral")
    parser.add_argument("--no-save", action="store_true",
                        help="Score without adding the submission to the corpus.")
    args = parser.parse_args()

    fixed, subs, scorer = _load()

    if args.headline and args.body:
        b = scorer.score(args.headline, args.body)
        print(f"\nFixed content: {fixed['title']}\n")
        print(f"Headline : {args.headline}")
        print(f"Body     : {args.body}")
        print(f"Stance   : {args.stance}\n")
        print(f"  semantic novelty : {b.semantic_novelty:.3f}")
        print(f"  lexical novelty  : {b.lexical_novelty:.3f}")
        print(f"  raw novelty      : {b.raw_novelty:.3f}")
        print(f"  relevance        : {b.relevance:.3f}")
        print(f"  relevance gate   : {b.gate:.3f}")
        print(f"  FINAL SCORE      : {b.score:.3f}  (0=derivative/irrelevant, 1=novel)")
        if not args.no_save:
            new_id = _append_submission(args.headline, args.body, args.stance)
            print(f"\n  Added to corpus as '{new_id}' (stance={args.stance}). "
                  f"Corpus size is now {len(subs) + 1}.")
    else:
        _report(subs, scorer)


if __name__ == "__main__":
    main()
