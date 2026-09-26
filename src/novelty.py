"""Novelty scoring engine for user-generated submissions.

Given a fixed piece of content and a corpus of prior submissions, this scores a
new submission's novelty on a normalized [0.0, 1.0] scale. The score combines:

  * semantic novelty  -> how different the meaning is from every prior submission
  * lexical novelty    -> how different the wording is (catches copy/paste)
  * relevance gate     -> suppresses novel-but-off-topic text so it is NOT rewarded

Final score = normalized( gate(relevance) * raw_novelty ).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embeddings import embed, embed_submission

# Weight of semantic vs lexical novelty in the raw novelty signal.
# Semantic dominates (it reliably detects rephrasing); lexical is a secondary
# guard that catches near-verbatim copying.
SEMANTIC_WEIGHT = 0.88
LEXICAL_WEIGHT = 0.12

# Relevance gate: boundaries are derived per-corpus at fit() time as fractions
# of a high-relevance anchor (a high percentile of corpus relevance), so the
# gate adapts to each fixed content and embedding model instead of relying on
# absolute cosine thresholds. Below the low boundary novelty is fully
# suppressed; above the high boundary it passes through; linear in between.
REL_ANCHOR_PERCENTILE = 90
GATE_LOW_FRAC = 0.12
GATE_HIGH_FRAC = 0.45
# Absolute floor: relevance at/below this is always off-topic regardless of corpus.
ABS_GATE_FLOOR = 0.03

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "for", "in", "on", "at", "by", "with", "as", "it", "this",
    "that", "will", "our", "we", "i", "you", "they", "them", "their", "not",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class ScoreBreakdown:
    """Full explanation of how a submission was scored."""

    semantic_novelty: float
    lexical_novelty: float
    raw_novelty: float
    relevance: float
    gate: float
    gated_novelty: float
    score: float  # final normalized [0, 1]


class NoveltyScorer:
    """Fits a calibration on a corpus, then scores new submissions in [0, 1]."""

    def __init__(self, fixed_content: str):
        self._fixed_vec = embed([fixed_content])[0]
        self._corpus_vecs: np.ndarray | None = None
        self._corpus_tokens: list[set[str]] = []
        self._raw_min = 0.0
        self._raw_max = 1.0
        self._gate_low = ABS_GATE_FLOOR
        self._gate_high = ABS_GATE_FLOOR + 1e-6

    def _relevance_gate(self, relevance: float) -> float:
        """Map a relevance score to a [0, 1] multiplier via a linear ramp."""
        if relevance <= self._gate_low:
            return 0.0
        if relevance >= self._gate_high:
            return 1.0
        return (relevance - self._gate_low) / (self._gate_high - self._gate_low)

    def _calibrate_gate(self) -> None:
        """Set gate boundaries relative to the corpus relevance distribution."""
        rels = self._corpus_vecs @ self._fixed_vec
        anchor = max(float(np.percentile(rels, REL_ANCHOR_PERCENTILE)), 1e-6)
        self._gate_low = max(ABS_GATE_FLOOR, GATE_LOW_FRAC * anchor)
        self._gate_high = GATE_HIGH_FRAC * anchor
        if self._gate_high <= self._gate_low:
            self._gate_high = self._gate_low + 1e-6

    def fit(self, submissions: list[dict]) -> None:
        """Embed the corpus and calibrate the [0, 1] normalization range.

        Calibration uses leave-one-out: each corpus item is scored against the
        others so the min/max reflect realistic novelty spread.
        """
        texts = [f"{s['headline']}. {s['body']}" for s in submissions]
        self._corpus_vecs = embed(texts)
        self._corpus_tokens = [_tokens(t) for t in texts]
        self._calibrate_gate()

        gated = []
        n = len(submissions)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            breakdown = self._raw_score(
                self._corpus_vecs[i],
                self._corpus_tokens[i],
                self._corpus_vecs[mask],
                [self._corpus_tokens[j] for j in range(n) if j != i],
            )
            gated.append(breakdown)

        self._raw_min = float(min(gated))
        self._raw_max = float(max(gated))
        if self._raw_max - self._raw_min < 1e-9:
            self._raw_max = self._raw_min + 1e-9

    def _raw_score(
        self,
        vec: np.ndarray,
        toks: set[str],
        corpus_vecs: np.ndarray,
        corpus_tokens: list[set[str]],
    ) -> float:
        """Compute the gated (pre-normalization) novelty for one submission."""
        # Vectors are unit-length, so cosine similarity is a dot product.
        sims = corpus_vecs @ vec
        semantic_novelty = 1.0 - float(np.max(sims)) if len(sims) else 1.0

        max_jac = max((_jaccard(toks, t) for t in corpus_tokens), default=0.0)
        lexical_novelty = 1.0 - max_jac

        raw = SEMANTIC_WEIGHT * semantic_novelty + LEXICAL_WEIGHT * lexical_novelty
        relevance = float(self._fixed_vec @ vec)
        return self._relevance_gate(relevance) * raw

    def score(self, headline: str, body: str) -> ScoreBreakdown:
        """Score a new submission against the fitted corpus."""
        if self._corpus_vecs is None:
            raise RuntimeError("NoveltyScorer.fit() must be called before score().")

        vec = embed_submission(headline, body)
        toks = _tokens(f"{headline}. {body}")
        return self._score_vec(vec, toks, self._corpus_vecs, self._corpus_tokens)

    def score_existing(self, index: int) -> ScoreBreakdown:
        """Score a corpus member using leave-one-out (excludes itself)."""
        if self._corpus_vecs is None:
            raise RuntimeError("NoveltyScorer.fit() must be called before scoring.")
        n = len(self._corpus_tokens)
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        others_tokens = [self._corpus_tokens[j] for j in range(n) if j != index]
        return self._score_vec(
            self._corpus_vecs[index],
            self._corpus_tokens[index],
            self._corpus_vecs[mask],
            others_tokens,
        )

    def _score_vec(
        self,
        vec: np.ndarray,
        toks: set[str],
        corpus_vecs: np.ndarray,
        corpus_tokens: list[set[str]],
    ) -> ScoreBreakdown:
        sims = corpus_vecs @ vec
        semantic_novelty = 1.0 - float(np.max(sims)) if len(sims) else 1.0
        max_jac = max((_jaccard(toks, t) for t in corpus_tokens), default=0.0)
        lexical_novelty = 1.0 - max_jac
        raw = SEMANTIC_WEIGHT * semantic_novelty + LEXICAL_WEIGHT * lexical_novelty

        relevance = float(self._fixed_vec @ vec)
        gate = self._relevance_gate(relevance)
        gated = gate * raw

        normalized = (gated - self._raw_min) / (self._raw_max - self._raw_min)
        normalized = float(np.clip(normalized, 0.0, 1.0))

        return ScoreBreakdown(
            semantic_novelty=semantic_novelty,
            lexical_novelty=lexical_novelty,
            raw_novelty=raw,
            relevance=relevance,
            gate=gate,
            gated_novelty=gated,
            score=normalized,
        )
