# SprintPulse — Team Performance & Attendance Analytics

Two TensorFlow models and a RAG chat assistant over engineering-org
operational data: employee attendance and agile sprint performance.

**Live demo:** _add your deployed Streamlit Community Cloud URL here_

---

## What it does

- **Attendance Risk** — score an employee-month and get a calibrated
  probability of high absenteeism, a Low/Medium/High tier, and a SHAP
  chart of which factors drove that specific prediction.
- **Sprint Performance** — score a sprint's metrics (QA defects, scope
  change, test coverage, team attendance, etc.) and get a predicted
  completion rate, an On Track / At Risk / Off Track tier, and a SHAP
  breakdown of the drivers.
- **Team Dashboard** — attendance and completion-rate trends per team,
  the (real, if modest) correlation between the two across all teams, and
  a department comparison.
- **Ask SprintPulse** — a RAG chatbot. Ask things like *"Why did Team
  Phoenix underperform in Sprint 14?"* or *"What's our Definition of
  Done?"* and get an answer grounded in retrieved HR policy docs and
  sprint retro notes, with cited sources — not general LLM knowledge.
- **Model Performance** — training curves, confusion matrix, and ROC for
  the attendance model; a head-to-head comparison of the sprint model
  against a linear-regression baseline, showing where the extra model
  complexity actually earns its keep.

## Why it exists

This is the deep-learning + RAG complement to a more classical ML project
([RiskLens](../risklens-credit-scoring), gradient boosting + SHAP on a
public credit-risk dataset). SprintPulse adds: TensorFlow/Keras instead of
scikit-learn, a second model type (regression, not just classification),
probability calibration, and a full retrieval-augmented generation
pipeline grounding an LLM in retrieved documents rather than letting it
answer from parametric memory.

## The data — and why it's synthetic

No public dataset pairs sprint telemetry with employee attendance — that's
internal company data (Jira + HRIS exports) at any real organization.
`generate_data.py` builds a synthetic version instead, but not as random
noise: each employee has a hidden "reliability" trait that drives both
their simulated attendance behavior *and*, aggregated to team level,
contributes to that team's simulated sprint completion rate. So the
pattern the dashboard and chatbot surface — teams with lower attendance
tend to complete less of their sprint commitment — is a real relationship
in the generated data, not something invented after the fact. Retro notes
in `docs/retros/` are template-generated from each sprint's own numbers,
so the RAG corpus never contradicts the dashboards.

**This is a portfolio demonstration on synthetic data, not analysis of a
real organization.**

## The models

### Attendance risk (classification)
- TensorFlow/Keras MLP (64→32→1, dropout, class-weighted for the ~14%
  positive rate), trained on 2,688 employee-months.
- Class weighting skews raw outputs away from true probabilities, so
  they're recalibrated with **isotonic regression** on the validation set
  before being shown to a user or used to set risk tiers — a raw
  class-weighted output isn't a real probability.
- Decision threshold chosen by maximizing F1 on the validation set
  (standard practice for imbalanced classifiers), not a default 0.5 cutoff.
- Test ROC-AUC ≈ 0.72. Deliberately not tuned to look artificially strong
  — human attendance behavior is noisy, and an honest number here is more
  defensible than an inflated one.

### Sprint completion rate (regression)
- TensorFlow/Keras MLP (32→16→1, L2 regularization), trained on 640
  team-sprints, compared against a linear regression baseline on the
  identical features.
- The underlying data has genuine non-linear structure the linear model
  can't represent: test coverage's benefit saturates (an S-curve, not a
  straight line), scope creep past ~15% triggers a disproportionate
  "tipping point" penalty, and CI failures compound with escaped bugs
  rather than adding independently.
- Result: **R² 0.85 (TensorFlow) vs. 0.79 (linear regression)** — the
  comparison is shown in the app itself, not just asserted.

### Explainability
Both models use `shap.Explainer` with a model-agnostic permutation
backend (works regardless of Keras internals) to explain individual
predictions — which specific features pushed *this* employee's or *this*
sprint's score up or down, not just global feature importance.

## The RAG assistant

- **Retrieval**: local dense embeddings via `fastembed`
  (`BAAI/bge-small-en-v1.5`, running on ONNX Runtime) over 646 documents
  (6 HR/process policy docs + 640 auto-generated sprint retro notes),
  cosine similarity, top-5 chunks — no API call needed for this step.
  Deliberately not `sentence-transformers`: it pulls in PyTorch, and
  loading PyTorch alongside TensorFlow (needed for the two models) in the
  same process segfaults on memory-constrained deployment hosts.
  `fastembed` gives the same embedding quality on plain ONNX Runtime, with
  no PyTorch in the dependency tree at all.
- **Generation**: the retrieved chunks are injected into a prompt sent to
  the [Groq API](https://console.groq.com) (`openai/gpt-oss-120b`, free
  tier) with an instruction to answer only from the provided context and
  cite sources — not to answer from the model's general knowledge.
- This is retrieval-**augmented** generation, not just a chatbot with a
  system prompt: the app embeds the query, searches the document index,
  and the answer is grounded in whatever comes back.

## Project layout

```
generate_data.py                 synthetic data generator (causal, not random)
data/                             employees.csv, attendance_monthly.csv, sprints.csv
docs/policies/*.md                6 hand-written HR/process policy docs
docs/retros/*.md                  640 auto-generated sprint retro notes
model/train_attendance.py         trains + calibrates the attendance classifier
model/train_sprint.py             trains the sprint regressor, compares vs. baseline
model/artifacts/                  trained models, preprocessors, metrics (committed)
rag/build_index.py                embeds docs/ into the retrieval index
rag/index/                        embeddings.npy + documents.json (committed)
app.py                            the Streamlit app
requirements.txt                  pinned dependencies
.streamlit/config.toml            theme
.env                               GROQ_API_KEY (gitignored — never committed)
```

## Running it locally

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your free
[Groq API key](https://console.groq.com):

```
GROQ_API_KEY=gsk_...
```

Then:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Data, models, and the RAG index are
already committed, so this works without rerunning the generation/training
scripts — those are only needed to regenerate from scratch:

```bash
python generate_data.py
python model/train_attendance.py
python model/train_sprint.py
python rag/build_index.py
```

## Deploying

1. Push this repo to GitHub (public, for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick this repo, branch `main`, main file `app.py` → **Deploy**.
3. In the deployed app's settings → **Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
   (Your local `.env` file is gitignored and never reaches GitHub — the
   live app needs its own copy of the key via Streamlit's Secrets UI.)
4. In **Sharing**, set viewer access to public so the link works for
   anyone without a Streamlit account.
5. Paste the resulting URL into this README's **Live demo** line.
