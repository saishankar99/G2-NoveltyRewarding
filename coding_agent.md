# Coding Agent Disclosure

**Tool used:** GitHub Copilot (agent mode) inside VS Code.
**My role:** I directed the architecture, made every design decision, interrogated
the agent's choices, and validated outputs against real metrics. The agent
executed implementation and tuning under my direction.

This document summarizes how I collaborated with the agent — the structure of my
prompts and the reasoning behind each phase — so the work is reproducible and my
engineering intent is transparent.

---

## Collaboration principle

I treated the agent as a fast pair-programmer, not an oracle. My workflow was:
**understand the theory first → agree an architecture → implement in small
verifiable steps → challenge every heuristic → ground all numbers in measured
data.** I never accepted a magic constant without seeing the distribution behind
it.

---

## Phase 1 — Concept grounding (before any code)

I refused to write code until I could defend every underlying concept.

> **Prompt (paraphrased):** "Before we build solution, explain every concept
> involved — embeddings, cosine similarity, novelty scoring, relevance gating,
> normalization, lexical vs semantic novelty, synthetic data, and evaluation.
> Use tiny concrete numeric examples I can follow by hand, and give me learning
> links each under 30 minutes. Then let me confirm I understand before we start."

**Outcome:** a worked numeric walkthrough of each concept (e.g. cosine of
`[0.9,0.1]·[0.8,0.2]=0.99`, novelty = `1 − max similarity`), which I used to
confirm my understanding before implementation.

---

## Phase 2 — Architecture agreed up front

> **Prompt:** "Propose a phased plan: dataset → embeddings → scoring → tests →
> docs. State the concrete design decisions (local embeddings, hybrid
> semantic+lexical novelty, an explicit relevance gate) and let me approve before
> coding."

**Decisions I set:**
- **Local-first** embeddings (`all-MiniLM-L6-v2`); an LLM only synthesizes the
  dataset.
- **Hybrid novelty** = semantic (embeddings) + lexical (Jaccard), semantic-weighted.
- **Relevance gate** as the explicit mechanism for "novel-but-irrelevant is not
  rewarded."
- **Leave-one-out** calibration for a meaningful `[0,1]` output.

---

## Phase 3 — Implementation in verifiable units

> **Prompt:** "Scaffold the project: a labeled 50-item golden dataset in four
> buckets (paraphrase / novel / off-topic / normal), a local embedding layer, the
> novelty engine, a Gemini generator with a deterministic fallback, and a pytest
> suite that asserts each bucket lands in the right score range with clear
> separation."

**Outcome:** working pipeline + 8 automated tests. I required the fallback
dataset so the whole project runs and tests pass with **no API key**.

---

## Phase 4 — Empirical tuning (no guessed constants)

When the first run mis-scored paraphrases, I insisted on data over intuition.

> **Prompt:** "Don't guess weights. Print a per-category breakdown of semantic
> novelty, lexical novelty, relevance, gate, and final score, then tell me exactly
> why paraphrases score too high and adjust based on the numbers."

**What the diagnostic revealed and how I directed the fix:**
- Lexical novelty was **inflating paraphrases** → I set semantic/lexical weights
  to **0.88 / 0.12**.
- The gate was **over-penalizing genuinely-novel on-topic items** (relevance
  ≈ 0.35) at the same rate as off-topic spam (relevance ≈ 0) → I directed a
  narrower gate so only relevance ≈ 0 is suppressed.

---

## Phase 5 — Interrogating the design

I pushed back on individual components to confirm they earned their place.

> **"Why is lexical novelty useful if semantic already detects rephrasing?"**
> Established it as a copy-paste guard and justified its low (0.12) weight.

> **"If the gated score is below the calibration minimum, does normalization go
> negative?"**
> Confirmed the `np.clip(…, 0, 1)` safeguard and the out-of-range semantics.

> **"Will the gate thresholds generalize to other fixed contents, or are they
> overfit to this one?"**
> This was the key critique. Absolute cosine thresholds don't travel across
> topics/models, so I directed a **corpus-derived gate**: boundaries set as
> fractions of a high-relevance anchor (90th percentile of corpus relevance),
> computed at `fit()` time. On this dataset it resolves to ≈ `0.06 / 0.23`,
> reproducing the hand-tuned behavior while adapting automatically elsewhere.

---

## Phase 6 — Feature: stance + a live, growing corpus

> **Prompt:** "The submission shape has three properties including a stance. After
> scoring a headline+body, append it to the corpus along with its stance so the
> system grows over time — but make sure this can never make the tests flaky."

**Outcome:** `demo.py` gained `--stance` (and `--no-save`); scored submissions
are appended (labelled `user`). I required the tests to read only the labelled
**seed** items so corpus growth stays deterministic.

---


Every numeric threshold in the final system was chosen by inspecting the
**measured score distribution**, not accepted blindly from the agent.
