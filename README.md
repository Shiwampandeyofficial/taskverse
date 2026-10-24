# StatVerse (v0.4)

Two modules:
1. **Learn** — any statistics topic explained (English/Hinglish/Hindi, BSc/MSc depth), grounded in your
   uploaded books via RAG, with a real, code-drawn chart.
2. **Analyze** — Industry Analysis Engine. Upload a dataset (CSV/TSV/Excel), state an objective in plain
   language. The AI only picks the statistical method + columns; pandas/SciPy computes every number.
   Full calculation steps, a reliability score with named factors, a real chart of your data, and an
   AI-written plain-language report are all shown.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env        # paste your free Groq key from console.groq.com/keys
python app.py               # open http://127.0.0.1:5000
```

## v0.4 — Industry Analysis Engine
- New `stats_engine.py`: 8 analyses, each returns a step-by-step calculation trace (formula → numbers → result), a
  reliability score (sample size + assumption checks, never a bare "accurate" claim), and a chart spec.
  describe, one_sample_ttest, two_sample_ttest (auto Welch/pooled via Levene), paired_ttest,
  chi_square_independence, one_way_anova, correlation_regression, proportion_ztest.
- `llm.py` gained `choose_analysis()` (picks a method + columns from the objective; column names are
  fuzzy-matched against the real dataset so a slightly-off AI answer still resolves) and
  `interpret_result()` (writes the plain-language report from the already-computed numbers — it cannot
  alter them).
- New `/api/dataset/upload`, `/api/analyze`, `/api/analyses` routes; new "Analyze" tab in the UI with
  dataset profile preview, an objective box, a "pick the method myself" fallback, an expandable
  calculation-steps panel, and a reliability-factors panel.
- A bad AI plan (wrong column, unsupported analysis) degrades to a dataset overview with a visible note,
  rather than failing the request.

## Files
app.py · rag.py · llm.py · charts.py · stats_engine.py (new) · templates/ · static/
