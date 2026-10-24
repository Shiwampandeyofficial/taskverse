"""
StatVerse — Flask app.   Run:  python app.py   then open http://127.0.0.1:5000
"""

import json
import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

import charts
import llm
import rag
import stats_engine as se

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "books")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)
DATASETS: dict[str, dict] = {}  # dataset_id -> {"path", "name", "profile"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("book_file")
    if not f or not f.filename:
        return jsonify(ok=False, error="No file selected."), 400
    name = secure_filename(f.filename)
    if not name.lower().endswith((".pdf", ".txt")):
        return jsonify(ok=False, error="Only .pdf or .txt files are supported."), 400
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    title = os.path.splitext(name)[0]
    job_id = rag.start_index_job(path, title, request.form.get("topic_tag", "").strip())
    return jsonify(ok=True, job_id=job_id, title=title)


@app.route("/api/job/<job_id>")
def job(job_id):
    j = rag.JOBS.get(job_id)
    if not j:
        return jsonify(ok=False, error="Unknown job."), 404
    return jsonify(ok=True, **j)


@app.route("/api/library")
def library():
    try:
        return jsonify(ok=True, **rag.library())
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/library/<path:title>", methods=["DELETE"])
def delete_book(title):
    rag.delete_book(title)
    return jsonify(ok=True, **rag.library())


@app.route("/api/query", methods=["POST"])
def query():
    d = request.get_json(force=True, silent=True) or {}
    topic = (d.get("topic") or "").strip()
    language = (d.get("language") or "hinglish").lower()
    level = (d.get("level") or "basic").lower()
    if not topic:
        return jsonify(ok=False, error="Please enter a topic."), 400

    try:
        hits = rag.retrieve(topic, n_results=6)
    except Exception as e:
        return jsonify(ok=False, error=f"Knowledge base error: {e}"), 500
    used = [h for h in hits if h["relevant"]][:5]

    try:
        text = llm.explain_topic(topic, [h["text"] for h in used], language, level)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

    # AI-chosen chart templates -> real charts drawn from code (never breaks the answer)
    chart_specs = []
    for cm in re.finditer(r"^\s*CHART:\s*(\{.*\})\s*$", text, re.M):
        try:
            chart_specs.append(json.loads(cm.group(1)))
        except ValueError:
            pass
    text = re.sub(r"^\s*CHART:.*$", "", text, flags=re.M).rstrip()
    built, seen = [], set()
    for sp in (chart_specs[:2] or [None]):
        try:
            ch = charts.build_chart(sp, topic)
            if ch["template"] not in seen:
                seen.add(ch["template"]); built.append(ch)
        except Exception:
            pass

    m = re.search(r"^\s*SOURCE:\s*(.+?)\s*$", text, re.I | re.M)
    source = m.group(1).strip(" .").lower() if m else ("reference material" if used else "general knowledge")
    if m:
        text = text[:m.start()].rstrip()

    return jsonify(
        ok=True,
        explanation=text,
        source=source,
        charts=built,
        references=[{"book": h["book_title"], "page": h["page"], "snippet": h["text"][:260]} for h in used],
    )


# ═══════════════════════ Industry Analysis Engine routes ═══════════════════════

import uuid as _uuid


@app.route("/api/dataset/upload", methods=["POST"])
def dataset_upload():
    f = request.files.get("dataset_file")
    if not f or not f.filename:
        return jsonify(ok=False, error="No file selected."), 400
    name = secure_filename(f.filename)
    if not name.lower().endswith((".csv", ".tsv", ".xlsx", ".xls")):
        return jsonify(ok=False, error="Only .csv, .tsv or .xlsx files are supported."), 400
    ds_id = _uuid.uuid4().hex[:12]
    path = os.path.join(DATASET_DIR, f"{ds_id}_{name}")
    f.save(path)
    try:
        df = se.load_dataset(path)
        df = se.coerce_types(df)
        profile = se.profile_dataset(df)
    except Exception as e:
        os.remove(path)
        return jsonify(ok=False, error=str(e)), 400
    df.to_parquet(path + ".parquet") if False else None  # kept simple: reload from source each time
    DATASETS[ds_id] = {"path": path, "name": name, "profile": profile}
    return jsonify(ok=True, dataset_id=ds_id, name=name, profile=profile)


@app.route("/api/dataset/<ds_id>")
def dataset_get(ds_id):
    d = DATASETS.get(ds_id)
    if not d:
        return jsonify(ok=False, error="Dataset not found — please re-upload."), 404
    return jsonify(ok=True, dataset_id=ds_id, name=d["name"], profile=d["profile"])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    d = request.get_json(force=True, silent=True) or {}
    ds_id = d.get("dataset_id")
    objective = (d.get("objective") or "").strip()
    language = (d.get("language") or "english").lower()
    manual = d.get("analysis")  # optional: user/UI picked the analysis explicitly

    ds = DATASETS.get(ds_id)
    if not ds:
        return jsonify(ok=False, error="Dataset not found — please re-upload."), 404
    if not objective and not manual:
        return jsonify(ok=False, error="Please describe what you want to find out."), 400

    try:
        df = se.coerce_types(se.load_dataset(ds["path"]))
    except Exception as e:
        return jsonify(ok=False, error=f"Could not re-read dataset: {e}"), 500

    if manual:
        plan = {"analysis": manual, "params": d.get("params") or {}, "reasoning": "Selected directly."}
    else:
        try:
            plan = llm.choose_analysis(ds["profile"], objective)
        except Exception as e:
            return jsonify(ok=False, error=f"Could not choose an analysis method: {e}"), 500

    try:
        outcome = se.run_analysis(df, plan["analysis"], plan.get("params") or {})
    except se.AnalysisError as e:
        # Try the safe fallback once so a bad AI plan doesn't dead-end the user
        try:
            outcome = se.run_analysis(df, "describe", {})
            outcome["fallback_note"] = f"Could not run '{plan['analysis']}' ({e}); showing a dataset overview instead."
        except Exception as e2:
            return jsonify(ok=False, error=str(e2)), 500
    except Exception as e:
        return jsonify(ok=False, error=f"Analysis failed: {e}"), 500

    try:
        narrative = llm.interpret_result(objective or "General exploration of the dataset.", outcome["title"],
                                          outcome.get("result", outcome.get("narrative_data", {})),
                                          outcome["reliability"], language)
    except Exception as e:
        narrative = f"(AI interpretation unavailable: {e})\n\nThe computed result is shown below — the numbers themselves are unaffected."

    chart = None
    if outcome.get("chart"):
        try:
            chart = charts.build_chart(outcome["chart"], plan["analysis"])
        except Exception:
            chart = None

    return jsonify(
        ok=True, plan=plan, analysis=outcome["analysis"], title=outcome["title"],
        steps=outcome["steps"], result=outcome.get("result"), table=outcome.get("table"),
        reliability=outcome["reliability"], narrative=narrative, chart=chart,
        fallback_note=outcome.get("fallback_note"),
    )


@app.route("/api/analyses")
def analyses_list():
    return jsonify(ok=True, analyses=list(se.ANALYSES.keys()), help=se.ANALYSIS_HELP)


@app.route("/api/health")
def health():
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
