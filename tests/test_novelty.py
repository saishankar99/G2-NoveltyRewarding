"""Automated evaluation of the novelty scorer against the labeled golden dataset.

These tests prove the requirements from the problem statement:
  * truly novel + relevant content is rewarded (HIGH score)
  * derivative / paraphrased content is NOT rewarded (LOW score)
  * novel-but-irrelevant content is NOT rewarded (relevance gate works)
  * novel and derivative groups are clearly separated
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.novelty import NoveltyScorer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Score thresholds (normalized [0, 1]).
LOW = 0.40
HIGH = 0.60


@pytest.fixture(scope="module")
def scored():
    fixed = json.loads((DATA_DIR / "fixed_content.json").read_text(encoding="utf-8"))
    data = json.loads((DATA_DIR / "submissions.json").read_text(encoding="utf-8"))
    # Only the labeled seed dataset is used so appended user submissions
    # (label "user") can never make the evaluation non-deterministic.
    seed_labels = {"novel", "normal", "paraphrase", "offtopic"}
    subs = [s for s in data["submissions"] if s["label"] in seed_labels]

    scorer = NoveltyScorer(fixed["text"])
    scorer.fit(subs)

    results = []
    for i, s in enumerate(subs):
        results.append((s["label"], scorer.score_existing(i)))
    return fixed, subs, scorer, results


def _by_label(results, label):
    return [b for lbl, b in results if lbl == label]


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def test_novel_content_is_rewarded(scored):
    _, _, _, results = scored
    novel = _by_label(results, "novel")
    avg = _mean([b.score for b in novel])
    assert avg >= HIGH, f"novel avg {avg:.2f} should be >= {HIGH}"


def test_paraphrases_are_not_rewarded(scored):
    _, _, _, results = scored
    para = _by_label(results, "paraphrase")
    avg = _mean([b.score for b in para])
    assert avg <= LOW, f"paraphrase avg {avg:.2f} should be <= {LOW}"


def test_offtopic_is_gated_low(scored):
    _, _, _, results = scored
    off = _by_label(results, "offtopic")
    # Despite high raw novelty, the relevance gate must knock these down.
    assert all(b.gate < 0.5 for b in off), "off-topic items should be gated"
    avg = _mean([b.score for b in off])
    assert avg <= LOW, f"offtopic avg {avg:.2f} should be <= {LOW}"


def test_novel_beats_derivative_separation(scored):
    _, _, _, results = scored
    novel_avg = _mean([b.score for b in _by_label(results, "novel")])
    para_avg = _mean([b.score for b in _by_label(results, "paraphrase")])
    off_avg = _mean([b.score for b in _by_label(results, "offtopic")])
    assert novel_avg - para_avg >= 0.30, "novel should clearly beat paraphrase"
    assert novel_avg - off_avg >= 0.30, "novel should clearly beat off-topic"


def test_scores_are_normalized(scored):
    _, _, _, results = scored
    for _, b in results:
        assert 0.0 <= b.score <= 1.0


def test_new_novel_relevant_submission_scores_high(scored):
    _, _, scorer, _ = scored
    b = scorer.score(
        headline="Charging deserts will punish the suburbs",
        body=(
            "Neighborhoods on the city edge have no fast chargers within miles. "
            "Without a mapped rollout targeting these charging deserts first, the "
            "2030 ban strands the very commuters who drive the most."
        ),
    )
    assert b.relevance > 0.3
    assert b.score >= HIGH, f"new novel+relevant scored {b.score:.2f}"


def test_new_offtopic_submission_is_not_rewarded(scored):
    _, _, scorer, _ = scored
    b = scorer.score(
        headline="My marathon training plan",
        body=(
            "Adding two interval days a week and a long Sunday run finally pushed "
            "my half-marathon time under ninety minutes this spring."
        ),
    )
    assert b.score <= LOW, f"off-topic new submission scored {b.score:.2f}"


def test_new_copied_submission_is_not_rewarded(scored):
    _, subs, scorer, _ = scored
    original = subs[13]  # an s14-style novel item
    b = scorer.score(headline=original["headline"], body=original["body"])
    # An exact copy of an existing submission has near-zero novelty.
    assert b.score <= LOW, f"copied submission scored {b.score:.2f}"
