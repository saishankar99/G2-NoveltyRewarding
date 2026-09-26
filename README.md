# Rewarding Novelty in User-Generated Submissions

**G2 AI Hackathon — Problem Statement 3**

A pipeline that scores how *novel* a user submission is on a normalized
`[0.0, 1.0]` scale, relative to a corpus of other submissions responding to the
same fixed content — while refusing to reward content that is novel but
irrelevant.

---

## 1. Content shape

**Fixed content** (`data/fixed_content.json`): a 78-word news item about the
city of Riverton banning new gasoline car sales by 2030. (The problem requires
≤100 words.)

**User submission** (the required *three discrete properties*):

| Property   | Type          | Example |
|------------|---------------|---------|
| `headline` | free text     | "The grid cannot handle this" |
| `body`     | free text (1–2 sentences) | "Peak evening charging could overload substations…" |
| `stance`   | multi-choice: `Support` / `Oppose` / `Neutral` | `Oppose` |

Novelty is judged on `headline + body`. `stance` is stored with each submission
(and persisted when a scored submission is added to the corpus); a natural
extension is to compute novelty *within* a stance group.

---

## 2. Engineering design

```
new submission ──► embed (headline+body) ──┐
                                           ├─► semantic novelty = 1 − max cosine to corpus
corpus (≈50) ─────► embed all ─────────────┘
                                           ├─► lexical novelty  = 1 − max Jaccard to corpus
                                           │
fixed content ────► embed ─────────────────► relevance = cosine(submission, fixed)
                                           │
   raw = 0.88·semantic + 0.12·lexical      │
   gated = relevance_gate(relevance) · raw │
   score = normalize(gated) ∈ [0, 1] ◄─────┘
```

### Components

- **Embeddings** (`src/embeddings.py`): `sentence-transformers/all-MiniLM-L6-v2`,
  run **fully locally**, producing unit-length 384-d vectors (so cosine
  similarity is a dot product). Cached per text.
- **Semantic novelty**: `1 − (highest cosine similarity to any other
  submission)`. Using the *nearest neighbour* means a single near-duplicate
  anywhere in the corpus correctly kills novelty.
- **Lexical novelty**: `1 − (highest Jaccard word overlap)` over content words
  (stop-words removed). A secondary guard that catches near-verbatim copying
  where wording is reused.
- **Relevance gate**: `relevance = cosine(submission, fixed content)`. A linear
  ramp suppresses novelty when relevance is near zero. The ramp boundaries are
  **derived per-corpus at fit time** as fractions of a high-relevance anchor
  (the 90th percentile of corpus relevance), so the gate adapts to each fixed
  content and embedding model instead of using absolute cosine thresholds (on
  this dataset they resolve to ≈ `0.06 → 0.23`). This is the explicit mechanism
  that stops *novel-but-irrelevant* content from being rewarded.
- **Normalization**: min–max calibration fitted on the corpus using
  **leave-one-out** raw scores, so the output is a meaningful `[0, 1]` where the
  most derivative corpus item ≈ 0 and the most novel ≈ 1.

### Key rationale

- **Semantic dominates (0.88) over lexical (0.12).** Embeddings reliably detect
  *rephrased* ideas (same meaning, new words); lexical overlap alone would
  reward a clever paraphrase. Lexical is kept only as a copy-paste guard.
- **The gate is deliberately narrow.** Early tuning showed a wide gate wrongly
  penalized legitimately novel *on-topic* items (which are less lexically
  central to the fixed text, relevance ≈ 0.35) at the same rate as off-topic
  spam (relevance ≈ 0). Narrowing it so only relevance ≈ 0 is gated fixed this.
- **Local-first.** Embedding + indexing + scoring run offline. An LLM (Gemini)
  is used *only* to synthesize the labeled dataset, matching the problem's
  "synthetic generation … recommended" note.

---

## 3. Golden dataset

`data/submissions.json` — 50 submissions, each with a ground-truth `label`:

| Label        | Count | Expected score | Purpose |
|--------------|-------|----------------|---------|
| `paraphrase` | 13    | LOW            | near-duplicates of one common opinion |
| `novel`      | 15    | HIGH           | distinct, specific on-topic angles |
| `offtopic`   | 10    | LOW (gated)    | vivid but irrelevant (baking, chess, …) |
| `normal`     | 12    | mid            | ordinary on-topic comments |

The committed file is a **deterministic fallback** so the pipeline and tests run
with no API key. `data/generate_dataset.py` regenerates an equivalent labeled
set with Gemini (`pip install -r requirements-gemini.txt`, set `GEMINI_API_KEY`).

---

## 4. Success criteria & achievement

Defined criteria, verified by `tests/test_novelty.py` (all **8 pass**):

| Criterion | Target | Actual |
|-----------|--------|--------|
| Novel content rewarded | avg ≥ 0.60 | **0.861** |
| Paraphrases not rewarded | avg ≤ 0.40 | **0.347** |
| Off-topic gated down | avg ≤ 0.40 & gate < 0.5 | **0.044** |
| Novel vs paraphrase separation | ≥ 0.30 | **0.51** |
| Novel vs off-topic separation | ≥ 0.30 | **0.82** |
| Output normalized to [0, 1] | always | ✅ |
| New novel+relevant submission | ≥ 0.60 | **0.88** |
| New off-topic / copied submission | ≤ 0.40 | ✅ |

Per-category report (`python demo.py`):

```
category       n   avg score     min     max
novel         15       0.861   0.743   1.000
normal        12       0.757   0.539   0.933
paraphrase    13       0.347   0.222   0.505
offtopic      10       0.044   0.000   0.412
```

---

## 5. Limitations

- **Corpus-relative & calibration-sensitive.** Scores are normalized against the
  current corpus; adding many submissions shifts the min/max. Production would
  re-fit calibration periodically or use a fixed reference distribution.
- **Live corpus growth.** Scoring a submission appends it to the corpus
  (labelled `user`), so an identical later submission correctly scores lower.
  Tests read only the labelled seed items, so this can never make the evaluation
  non-deterministic.
- **One off-topic item scored 0.41.** A borderline "hobby" post with incidental
  topical words partially passed the gate — the gate is a soft heuristic, not a
  classifier.
- **Single embedding model.** No cross-encoder / LLM-judge second pass; subtle
  semantic novelty may be missed.
- **Nearest-neighbour novelty is O(n) per query.** Fine at this scale; see below.
- **Stance is stored but not yet used** in the scoring math.

---

## 6. Scaling to production (future work)

- Replace the in-memory corpus with **Postgres + `pgvector`**; store embeddings
  and use an HNSW/IVFFlat index for approximate nearest-neighbour at scale.
- Add a **cross-encoder re-rank** on the top-k nearest neighbours for sharper
  semantic novelty.
- Track production metrics: score distribution drift, gate false-positive rate,
  and human-agreement (e.g. Spearman correlation with human novelty ratings).

---

## 7. How to run

```bash
# 1. Go to the project
cd "G2 Hackathon"

# 2. (Recommended) create & activate a virtual environment
py -m venv G2
source G2/Scripts/activate        # Git Bash
#  G2\Scripts\activate            # PowerShell / CMD

# 3. Install core dependencies (fully local)
pip install -r requirements.txt
```

The first run downloads the ~90 MB `all-MiniLM-L6-v2` embedding model once, then caches it.

## Run the automated evaluation

```bash
python -m pytest tests/ -v
```

Expected: **8 passed**.

## Three test cases

Each command prints a full breakdown; the key line is **FINAL SCORE** (0.0–1.0).
These use `--no-save` so they score against the fixed 50-item seed corpus and
reproduce the outputs below exactly.

### Case 1 — Novel + relevant → HIGH

```bash
python demo.py --no-save --stance Oppose --headline "Battery fires will overwhelm our fire crews" --body "Lithium fires burn hotter and reignite; without special training the ban could outpace what our fire department can safely handle."
```

```
  semantic novelty : 0.514
  relevance        : 0.291
  relevance gate   : 1.000
  FINAL SCORE      : 0.881   <- rewarded
```

### Case 2 — Paraphrase of a common opinion → LOW

```bash
python demo.py --no-save --stance Support --headline "Cleaner air for everyone" --body "Banning gas cars by 2030 is a great move that gives us cleaner air and a healthier city."
```

```
  semantic novelty : 0.075   <- idea already exists in the corpus
  relevance        : 0.555   (very relevant, but that alone is not enough)
  FINAL SCORE      : 0.178   <- not rewarded
```

### Case 3 — Novel but OFF-TOPIC → gate kills it

```bash
python demo.py --no-save --stance Neutral --headline "My sourdough finally worked" --body "Longer fermentation and a cast-iron Dutch oven gave the loaf a perfect crust and an airy crumb this weekend."
```

```
  raw novelty      : 0.469   <- genuinely novel wording
  relevance        : -0.091  <- off-topic
  relevance gate   : 0.000   <- gate fires
  FINAL SCORE      : 0.000   <- not rewarded
```

### What each case proves

| Case | Demonstrates |
|------|--------------|
| 1 | Truly novel + relevant content **is** rewarded. |
| 2 | Rephrasing an existing idea is **not** rewarded (low semantic novelty), despite high relevance. |
| 3 | The **relevance gate** — high raw novelty (0.47) collapses to 0.00 because it is off-topic. |

## Growing the corpus (live behaviour)

Score a real submission with its **stance**. By default it is appended to the
corpus (`data/submissions.json`, labelled `user`), so the system "remembers" it:

```bash
python demo.py --headline "Ban ignores delivery drivers" --body "Gig couriers on tight margins cannot afford new EVs by 2030 without targeted grants." --stance Oppose
#   ...breakdown...
#   Added to corpus as 's51' (stance=Oppose). Corpus size is now 51.
```

- `--stance` accepts `Support` / `Oppose` / `Neutral` (default `Neutral`).
- Add `--no-save` to score **without** modifying the corpus.
- Because the corpus grows, an identical later submission correctly scores lower.
- The automated tests read only the labelled **seed** items, so appended `user`
  submissions never affect the evaluation.
```
