"""
stats_engine.py — Industry Analysis Engine.

Everything that produces a NUMBER is plain pandas/numpy/scipy — deterministic,
reproducible, no AI involved. The AI (see llm.py: choose_analysis / interpret_result)
only (a) picks which analysis fits the stated objective and (b) writes the plain-language
narrative around numbers this module already computed.

Every analysis returns a "steps" list: the same formula -> substitution -> result
chain a statistical officer would show in a working paper, not just a final number.
"""

from __future__ import annotations

import difflib
import io
import os

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("mode.copy_on_write", True)

MAX_ROWS = 200_000
MAX_CELL_CHARS = 60


# ─────────────────────────────── loading & profiling ───────────────────────────────

def load_dataset(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif ext == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload .csv, .tsv or .xlsx.")
    if df.shape[1] == 0:
        raise ValueError("No columns could be read from this file.")
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _try_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        cleaned = s.astype(str).str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().mean() >= 0.8 and converted.notna().sum() >= 3:
            return converted
    return s


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: text columns that are 'really' numbers get converted."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = _try_numeric(out[c])
    return out


def profile_dataset(df: pd.DataFrame) -> dict:
    """A compact JSON-safe summary — this, not the raw data, is what the AI sees."""
    cols = []
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        col = {"name": c, "missing": n_missing, "missing_pct": round(100 * n_missing / max(len(s), 1), 1)}
        if pd.api.types.is_numeric_dtype(s):
            sd = s.dropna()
            col.update(
                type="numeric",
                n=int(sd.shape[0]),
                mean=_r(sd.mean()), sd=_r(sd.std()), min=_r(sd.min()), max=_r(sd.max()),
                median=_r(sd.median()),
                distinct=int(sd.nunique()),
            )
        else:
            sd = s.dropna().astype(str)
            vc = sd.value_counts().head(8)
            col.update(
                type="categorical",
                n=int(sd.shape[0]),
                distinct=int(sd.nunique()),
                top_categories=[{"value": str(k)[:MAX_CELL_CHARS], "count": int(v)} for k, v in vc.items()],
            )
        cols.append(col)
    return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": cols}


def _r(x, nd=4):
    try:
        v = float(x)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────── column resolution ───────────────────────────────

def resolve_column(name: str | None, df: pd.DataFrame) -> str | None:
    """Matches an AI-suggested column name to a real one: exact -> casefold -> fuzzy."""
    if not name:
        return None
    if name in df.columns:
        return name
    low = {c.casefold(): c for c in df.columns}
    if name.casefold() in low:
        return low[name.casefold()]
    match = difflib.get_close_matches(name, df.columns, n=1, cutoff=0.6)
    return match[0] if match else None


class AnalysisError(Exception):
    pass


def _need(df, name, label="column"):
    col = resolve_column(name, df)
    if col is None:
        raise AnalysisError(f"Couldn't find a {label} matching '{name}' in the dataset columns: {list(df.columns)}.")
    return col


def _numeric_series(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 2:
        raise AnalysisError(f"Column '{col}' does not have enough numeric values to analyse.")
    return s


# ─────────────────────────────── reliability scoring ───────────────────────────────

def reliability(n: int, assumption_checks: list[dict], min_n_good: int = 30, extra_penalty: int = 0) -> dict:
    """
    Transparent, additive score out of 100 — never a bare "99% accurate" claim.
    factors: [{"name","status": ok|warn|fail, "note"}]
    """
    factors = []
    if n >= min_n_good * 3:
        factors.append({"name": "Sample size", "status": "ok", "note": f"n = {n}, comfortably large for this test."})
        size_pts = 40
    elif n >= min_n_good:
        factors.append({"name": "Sample size", "status": "warn", "note": f"n = {n} is adequate but modest; results may shift with more data."})
        size_pts = 28
    else:
        factors.append({"name": "Sample size", "status": "fail", "note": f"n = {n} is small; treat conclusions as tentative."})
        size_pts = 10

    pts = size_pts
    for chk in assumption_checks:
        factors.append(chk)
        pts += {"ok": 20, "warn": 10, "fail": 0}.get(chk["status"], 10)

    max_pts = 40 + 20 * len(assumption_checks)
    score = max(0, min(100, round(100 * pts / max_pts) - extra_penalty))
    level = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    return {"score": score, "level": level, "factors": factors}


def _normality_check(sample: pd.Series, label: str) -> dict:
    n = len(sample)
    if n < 3:
        return {"name": f"Normality of {label}", "status": "warn", "note": "Too few points to test normality."}
    if n > 5000:
        sample = sample.sample(5000, random_state=0)
    stat, p = stats.shapiro(sample)
    if p >= 0.05:
        return {"name": f"Normality of {label}", "status": "ok", "note": f"Shapiro–Wilk p = {p:.3f} (≥0.05) — no strong evidence against normality."}
    return {"name": f"Normality of {label}", "status": "warn", "note": f"Shapiro–Wilk p = {p:.3f} (<0.05) — data deviates from normal; interpret with some caution."}


def _variance_check(a: pd.Series, b: pd.Series) -> dict:
    stat, p = stats.levene(a, b)
    if p >= 0.05:
        return {"name": "Equal variances", "status": "ok", "note": f"Levene's test p = {p:.3f} — variances look similar between groups."}
    return {"name": "Equal variances", "status": "warn", "note": f"Levene's test p = {p:.3f} — variances differ; a Welch correction is used automatically."}


def _fmt(x, nd=4):
    return f"{x:.{nd}g}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else str(x)


def _pct(x, nd=1):
    """Percent formatting that never falls into scientific notation (e.g. 73.6, not 7e+01)."""
    return f"{x:.{nd}f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else str(x)


# ─────────────────────────────── analyses ───────────────────────────────

def describe(df: pd.DataFrame, params: dict) -> dict:
    prof = profile_dataset(df)
    numeric = [c for c in prof["columns"] if c["type"] == "numeric"]
    steps = [{"label": "Step 1 — Load & scan", "detail": f"{prof['n_rows']} rows, {prof['n_cols']} columns read."}]
    for c in numeric[:8]:
        steps.append({"label": f"Summary — {c['name']}",
                      "detail": f"n={c['n']}, mean={_fmt(c['mean'])}, sd={_fmt(c['sd'])}, median={_fmt(c['median'])}, "
                                f"min={_fmt(c['min'])}, max={_fmt(c['max'])}"})
    chart = None
    if numeric:
        chart = {"type": "histogram_data", "params": {"values": df[resolve_column(numeric[0]["name"], df)].dropna().tolist()[:2000],
                                                        "label": numeric[0]["name"]}}
    return {
        "analysis": "describe", "title": "Descriptive summary of the dataset",
        "steps": steps, "table": prof["columns"],
        "narrative_data": {"n_rows": prof["n_rows"], "n_cols": prof["n_cols"], "numeric_cols": [c["name"] for c in numeric],
                            "categorical_cols": [c["name"] for c in prof["columns"] if c["type"] == "categorical"]},
        "reliability": reliability(prof["n_rows"], [], min_n_good=10),
        "chart": chart,
    }


def one_sample_ttest(df, params):
    col = _need(df, params.get("column"), "numeric column")
    mu0 = float(params.get("mu0", 0))
    s = _numeric_series(df, col)
    n, xbar, sd = len(s), s.mean(), s.std(ddof=1)
    se = sd / np.sqrt(n)
    t = (xbar - mu0) / se
    df_ = n - 1
    p = 2 * stats.t.sf(abs(t), df_)
    ci = stats.t.interval(0.95, df_, loc=xbar, scale=se)
    steps = [
        {"label": "Step 1 — Sample statistics", "detail": f"n = {n}, mean x̄ = {_fmt(xbar)}, sd s = {_fmt(sd)} (column: {col})"},
        {"label": "Step 2 — Standard error", "detail": f"SE = s/√n = {_fmt(sd)}/√{n} = {_fmt(se)}"},
        {"label": "Step 3 — Test statistic", "detail": f"t = (x̄ − μ₀)/SE = ({_fmt(xbar)} − {mu0:g})/{_fmt(se)} = {_fmt(t)}"},
        {"label": "Step 4 — Degrees of freedom", "detail": f"df = n − 1 = {df_}"},
        {"label": "Step 5 — p-value (two-tailed)", "detail": f"p = 2·P(T₍{df_}₎ > |{_fmt(t)}|) = {_fmt(p)}"},
        {"label": "Step 6 — 95% confidence interval for the mean", "detail": f"x̄ ± t₀.₀₂₅·SE = [{_fmt(ci[0])}, {_fmt(ci[1])}]"},
    ]
    checks = [_normality_check(s, col)]
    return {
        "analysis": "one_sample_ttest", "title": f"One-sample t-test — is the mean of '{col}' different from {mu0:g}?",
        "steps": steps,
        "result": {"n": n, "mean": _r(xbar), "sd": _r(sd), "se": _r(se), "statistic": _r(t), "df": df_, "p_value": _r(p, 6),
                   "ci_95": [_r(ci[0]), _r(ci[1])], "reject_null_at_5pct": bool(p < 0.05), "mu0": mu0, "column": col},
        "reliability": reliability(n, checks),
        "chart": {"type": "boxplot_data", "params": {"groups": {col: s.tolist()[:2000]}, "reference_line": mu0}},
    }


def two_sample_ttest(df, params):
    value_col = _need(df, params.get("value_column") or params.get("column"), "numeric column")
    group_col = _need(df, params.get("group_column"), "group column")
    g = df[[value_col, group_col]].dropna()
    g[value_col] = pd.to_numeric(g[value_col], errors="coerce")
    g = g.dropna()
    levels = g[group_col].astype(str).value_counts().index.tolist()
    want = params.get("groups")
    if isinstance(want, list) and len(want) == 2:
        lv = [str(x) for x in want if str(x) in levels] or levels[:2]
    else:
        lv = levels[:2]
    if len(lv) < 2:
        raise AnalysisError(f"Group column '{group_col}' needs at least two categories; found {levels}.")
    a = g.loc[g[group_col].astype(str) == lv[0], value_col]
    b = g.loc[g[group_col].astype(str) == lv[1], value_col]
    if len(a) < 2 or len(b) < 2:
        raise AnalysisError(f"Not enough observations in groups {lv} of '{group_col}' for a t-test.")

    var_chk = _variance_check(a, b)
    equal_var = var_chk["status"] == "ok"
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    n1, n2 = len(a), len(b)
    m1, m2, s1, s2 = a.mean(), b.mean(), a.std(ddof=1), b.std(ddof=1)
    if equal_var:
        df_ = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df_)
        se = sp * np.sqrt(1 / n1 + 1 / n2)
        se_detail = f"pooled SE = Sp·√(1/n1+1/n2), Sp = √[((n1−1)s1²+(n2−1)s2²)/df] = {_fmt(sp)} → SE = {_fmt(se)}"
    else:
        se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        df_ = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
        se_detail = f"Welch SE = √(s1²/n1 + s2²/n2) = {_fmt(se)}"
    diff = m1 - m2
    ci = stats.t.interval(0.95, df_, loc=diff, scale=se)
    steps = [
        {"label": "Step 1 — Group summaries", "detail": f"{lv[0]}: n1={n1}, mean={_fmt(m1)}, sd={_fmt(s1)} | {lv[1]}: n2={n2}, mean={_fmt(m2)}, sd={_fmt(s2)}"},
        {"label": "Step 2 — Check equal variances (Levene's test)", "detail": var_chk["note"] + (" -> pooled SE used." if equal_var else " -> Welch SE used.")},
        {"label": "Step 3 — Standard error of the difference", "detail": se_detail},
        {"label": "Step 4 — Test statistic", "detail": f"t = (x̄1 − x̄2)/SE = ({_fmt(m1)} − {_fmt(m2)})/{_fmt(se)} = {_fmt(t)}"},
        {"label": "Step 5 — Degrees of freedom", "detail": f"df = {_fmt(df_, 2)}"},
        {"label": "Step 6 — p-value (two-tailed)", "detail": f"p = 2·P(T₍{_fmt(df_, 2)}₎ > |{_fmt(t)}|) = {_fmt(p, 6)}"},
        {"label": "Step 7 — 95% CI for the difference of means", "detail": f"({_fmt(diff)}) ± t₀.₀₂₅·SE = [{_fmt(ci[0])}, {_fmt(ci[1])}]"},
    ]
    checks = [var_chk, _normality_check(a, f"{lv[0]}"), _normality_check(b, f"{lv[1]}")]
    return {
        "analysis": "two_sample_ttest",
        "title": f"Two-sample t-test — '{value_col}' between {lv[0]} and {lv[1]} (grouped by {group_col})",
        "steps": steps,
        "result": {"group_1": lv[0], "group_2": lv[1], "n1": n1, "n2": n2, "mean1": _r(m1), "mean2": _r(m2),
                   "mean_diff": _r(diff), "se": _r(se), "statistic": _r(t), "df": _r(df_, 2), "p_value": _r(p, 6),
                   "ci_95": [_r(ci[0]), _r(ci[1])], "equal_variance_assumed": equal_var, "reject_null_at_5pct": bool(p < 0.05)},
        "reliability": reliability(min(n1, n2), checks),
        "chart": {"type": "boxplot_data", "params": {"groups": {lv[0]: a.tolist()[:1500], lv[1]: b.tolist()[:1500]}}},
    }


def paired_ttest(df, params):
    c1 = _need(df, params.get("column_before") or params.get("column1"), "'before' column")
    c2 = _need(df, params.get("column_after") or params.get("column2"), "'after' column")
    g = df[[c1, c2]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(g) < 3:
        raise AnalysisError(f"Need at least 3 paired, non-missing rows across '{c1}' and '{c2}'.")
    d = g[c1] - g[c2]
    n, dbar, sdd = len(d), d.mean(), d.std(ddof=1)
    se = sdd / np.sqrt(n)
    t = dbar / se
    df_ = n - 1
    p = 2 * stats.t.sf(abs(t), df_)
    ci = stats.t.interval(0.95, df_, loc=dbar, scale=se)
    steps = [
        {"label": "Step 1 — Paired differences", "detail": f"d_i = {c1} − {c2}; n = {n} pairs, mean difference d̄ = {_fmt(dbar)}, sd(d) = {_fmt(sdd)}"},
        {"label": "Step 2 — Standard error", "detail": f"SE = sd(d)/√n = {_fmt(sdd)}/√{n} = {_fmt(se)}"},
        {"label": "Step 3 — Test statistic", "detail": f"t = d̄/SE = {_fmt(dbar)}/{_fmt(se)} = {_fmt(t)}"},
        {"label": "Step 4 — Degrees of freedom", "detail": f"df = n − 1 = {df_}"},
        {"label": "Step 5 — p-value (two-tailed)", "detail": f"p = 2·P(T₍{df_}₎ > |{_fmt(t)}|) = {_fmt(p, 6)}"},
        {"label": "Step 6 — 95% CI for the mean difference", "detail": f"d̄ ± t₀.₀₂₅·SE = [{_fmt(ci[0])}, {_fmt(ci[1])}]"},
    ]
    checks = [_normality_check(d, "paired differences")]
    return {
        "analysis": "paired_ttest", "title": f"Paired t-test — '{c1}' vs '{c2}'",
        "steps": steps,
        "result": {"n_pairs": n, "mean_diff": _r(dbar), "sd_diff": _r(sdd), "se": _r(se), "statistic": _r(t), "df": df_,
                   "p_value": _r(p, 6), "ci_95": [_r(ci[0]), _r(ci[1])], "reject_null_at_5pct": bool(p < 0.05)},
        "reliability": reliability(n, checks),
        "chart": {"type": "boxplot_data", "params": {"groups": {c1: g[c1].tolist()[:1500], c2: g[c2].tolist()[:1500]}}},
    }


def chi_square_independence(df, params):
    c1 = _need(df, params.get("column1") or params.get("row_column"), "first categorical column")
    c2 = _need(df, params.get("column2") or params.get("col_column"), "second categorical column")
    g = df[[c1, c2]].dropna().astype(str)
    tab = pd.crosstab(g[c1], g[c2])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise AnalysisError(f"Need at least 2 categories in each of '{c1}' and '{c2}' for a chi-square test.")
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    n = int(tab.values.sum())
    min_exp = float(expected.min())
    cramers_v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    steps = [
        {"label": "Step 1 — Contingency table", "detail": f"Observed counts of '{c1}' × '{c2}' — {tab.shape[0]} × {tab.shape[1]} table, n = {n}."},
        {"label": "Step 2 — Expected counts under independence", "detail": "E_ij = (row total × column total) / n, computed for every cell."},
        {"label": "Step 3 — Chi-square statistic", "detail": f"χ² = Σ (O_ij − E_ij)²/E_ij = {_fmt(chi2)}"},
        {"label": "Step 4 — Degrees of freedom", "detail": f"df = (rows−1)(cols−1) = ({tab.shape[0]}−1)({tab.shape[1]}−1) = {dof}"},
        {"label": "Step 5 — p-value", "detail": f"p = P(χ²₍{dof}₎ > {_fmt(chi2)}) = {_fmt(p, 6)}"},
        {"label": "Step 6 — Effect size", "detail": f"Cramér's V = √(χ²/(n·min(rows−1,cols−1))) = {_fmt(cramers_v)} (0=none, 1=perfect association)"},
    ]
    checks = [{"name": "Minimum expected count", "status": "ok" if min_exp >= 5 else "warn",
               "note": f"Smallest expected cell count = {_fmt(min_exp, 2)} ({'≥5, rule of thumb satisfied' if min_exp >= 5 else '<5, chi-square approximation may be unreliable'})."}]
    return {
        "analysis": "chi_square_independence", "title": f"Chi-square test of independence — '{c1}' vs '{c2}'",
        "steps": steps,
        "result": {"chi2": _r(chi2), "df": dof, "p_value": _r(p, 6), "cramers_v": _r(cramers_v),
                   "n": n, "reject_null_at_5pct": bool(p < 0.05)},
        "table": {"rows": tab.index.astype(str).tolist(), "cols": tab.columns.astype(str).tolist(), "values": tab.values.tolist()},
        "reliability": reliability(n, checks, min_n_good=30),
        "chart": {"type": "bar", "params": {"labels": [f"{r}·{c}" for r in tab.index for c in tab.columns][:12],
                                             "values": tab.values.flatten().tolist()[:12]}},
    }


def one_way_anova(df, params):
    value_col = _need(df, params.get("value_column") or params.get("column"), "numeric column")
    group_col = _need(df, params.get("group_column"), "group column")
    g = df[[value_col, group_col]].dropna()
    g[value_col] = pd.to_numeric(g[value_col], errors="coerce")
    g = g.dropna()
    groups = {k: v[value_col].values for k, v in g.groupby(g[group_col].astype(str))}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        raise AnalysisError(f"Need at least 2 groups with 2+ observations each in '{group_col}'.")
    arrays = list(groups.values())
    k, N = len(arrays), sum(len(a) for a in arrays)
    grand_mean = np.concatenate(arrays).mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_within = sum(((a - a.mean()) ** 2).sum() for a in arrays)
    df_b, df_w = k - 1, N - k
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    F, p = stats.f_oneway(*arrays)
    eta2 = ss_between / (ss_between + ss_within)
    steps = [
        {"label": "Step 1 — Group means", "detail": "; ".join(f"{name}: n={len(a)}, mean={_fmt(a.mean())}" for name, a in groups.items())},
        {"label": "Step 2 — Grand mean", "detail": f"x̄_grand = {_fmt(grand_mean)} (over N = {N} observations, k = {k} groups)"},
        {"label": "Step 3 — Between-group sum of squares", "detail": f"SSB = Σ nᵢ(x̄ᵢ − x̄_grand)² = {_fmt(ss_between)}"},
        {"label": "Step 4 — Within-group sum of squares", "detail": f"SSW = Σ Σ(xᵢⱼ − x̄ᵢ)² = {_fmt(ss_within)}"},
        {"label": "Step 5 — Mean squares", "detail": f"MSB = SSB/(k−1) = {_fmt(ss_between)}/{df_b} = {_fmt(ms_b)};  MSW = SSW/(N−k) = {_fmt(ss_within)}/{df_w} = {_fmt(ms_w)}"},
        {"label": "Step 6 — F statistic", "detail": f"F = MSB/MSW = {_fmt(ms_b)}/{_fmt(ms_w)} = {_fmt(F)}"},
        {"label": "Step 7 — p-value", "detail": f"p = P(F₍{df_b},{df_w}₎ > {_fmt(F)}) = {_fmt(p, 6)}"},
        {"label": "Step 8 — Effect size", "detail": f"η² = SSB/(SSB+SSW) = {_fmt(eta2)} (share of variance explained by group)"},
    ]
    levene_stat, levene_p = stats.levene(*arrays)
    checks = [
        {"name": "Equal variances (Levene)", "status": "ok" if levene_p >= 0.05 else "warn",
         "note": f"p = {_fmt(levene_p, 4)} — {'variances look similar' if levene_p >= 0.05 else 'variances differ; treat F-test with caution'}."},
        _normality_check(pd.Series(np.concatenate(arrays)), "pooled residuals"),
    ]
    return {
        "analysis": "one_way_anova", "title": f"One-way ANOVA — '{value_col}' across groups of '{group_col}'",
        "steps": steps,
        "result": {"k_groups": k, "N": N, "F": _r(F), "df_between": df_b, "df_within": df_w, "p_value": _r(p, 6),
                   "eta_squared": _r(eta2), "reject_null_at_5pct": bool(p < 0.05),
                   "group_means": {k2: _r(a.mean()) for k2, a in groups.items()}},
        "reliability": reliability(min(len(a) for a in arrays), checks),
        "chart": {"type": "boxplot_data", "params": {"groups": {k2: a.tolist()[:1200] for k2, a in groups.items()}}},
    }


def correlation_regression(df, params):
    xcol = _need(df, params.get("x_column") or params.get("independent"), "independent (x) column")
    ycol = _need(df, params.get("y_column") or params.get("dependent"), "dependent (y) column")
    g = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(g) < 4:
        raise AnalysisError(f"Need at least 4 paired numeric values in '{xcol}' and '{ycol}'.")
    x, y = g[xcol].values, g[ycol].values
    n = len(x)
    res = stats.linregress(x, y)
    r, r2 = res.rvalue, res.rvalue ** 2
    yhat = res.intercept + res.slope * x
    resid = y - yhat
    sse = float((resid ** 2).sum())
    dfree = n - 2
    mse = sse / dfree
    se_slope = res.stderr
    t_slope = res.slope / se_slope
    p_slope = 2 * stats.t.sf(abs(t_slope), dfree)
    t_r = r * np.sqrt(dfree / max(1e-12, 1 - r2))
    p_r = 2 * stats.t.sf(abs(t_r), dfree)
    steps = [
        {"label": "Step 1 — Data", "detail": f"n = {n} paired observations of x = '{xcol}' and y = '{ycol}'."},
        {"label": "Step 2 — Correlation coefficient", "detail": f"r = Σ(x−x̄)(y−ȳ) / √[Σ(x−x̄)²·Σ(y−ȳ)²] = {_fmt(r)}"},
        {"label": "Step 3 — Significance of r", "detail": f"t = r·√(n−2)/√(1−r²) = {_fmt(t_r)}, df = {dfree}, p = {_fmt(p_r, 6)}"},
        {"label": "Step 4 — Least-squares line", "detail": f"ŷ = a + bx, with b (slope) = {_fmt(res.slope)}, a (intercept) = {_fmt(res.intercept)}"},
        {"label": "Step 5 — Significance of the slope", "detail": f"SE(b) = {_fmt(se_slope)}; t = b/SE(b) = {_fmt(t_slope)}; p = {_fmt(p_slope, 6)}"},
        {"label": "Step 6 — Coefficient of determination", "detail": f"R² = r² = {_fmt(r2)} → {_pct(r2 * 100)}% of the variation in y is explained by x."},
    ]
    checks = [
        {"name": "Linearity/homoscedasticity (visual)", "status": "warn", "note": "Check the residual pattern in the scatter chart for a curve or funnel shape before trusting this line."},
        _normality_check(pd.Series(resid), "residuals"),
    ]
    return {
        "analysis": "correlation_regression", "title": f"Correlation & simple linear regression — '{ycol}' on '{xcol}'",
        "steps": steps,
        "result": {"n": n, "r": _r(r), "r_squared": _r(r2), "slope": _r(res.slope), "intercept": _r(res.intercept),
                   "p_value_slope": _r(p_slope, 6), "p_value_r": _r(p_r, 6), "se_slope": _r(se_slope),
                   "reject_null_at_5pct": bool(p_slope < 0.05)},
        "reliability": reliability(n, checks, min_n_good=25),
        "chart": {"type": "scatter_regression_data", "params": {"x": x.tolist()[:1500], "y": y.tolist()[:1500],
                                                                  "slope": _r(res.slope), "intercept": _r(res.intercept),
                                                                  "xlabel": xcol, "ylabel": ycol}},
    }


def proportion_ztest(df, params):
    col = _need(df, params.get("column"), "categorical/binary column")
    success = params.get("success_value")
    s = df[col].dropna().astype(str)
    if success is None:
        success = s.value_counts().index[0]
    success = str(success)
    n = len(s)
    x = int((s == success).sum())
    phat = x / n
    p0 = float(params.get("p0", 0.5))
    se0 = np.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se0
    p = 2 * stats.norm.sf(abs(z))
    se_ci = np.sqrt(phat * (1 - phat) / n)
    ci = (phat - 1.96 * se_ci, phat + 1.96 * se_ci)
    steps = [
        {"label": "Step 1 — Count successes", "detail": f"'{success}' occurs in x = {x} of n = {n} rows of '{col}'."},
        {"label": "Step 2 — Sample proportion", "detail": f"p̂ = x/n = {x}/{n} = {_fmt(phat)}"},
        {"label": "Step 3 — Standard error under H₀: p = {:g}".format(p0), "detail": f"SE₀ = √(p₀(1−p₀)/n) = {_fmt(se0)}"},
        {"label": "Step 4 — Test statistic", "detail": f"z = (p̂ − p₀)/SE₀ = ({_fmt(phat)} − {p0:g})/{_fmt(se0)} = {_fmt(z)}"},
        {"label": "Step 5 — p-value (two-tailed)", "detail": f"p = 2·P(Z > |{_fmt(z)}|) = {_fmt(p, 6)}"},
        {"label": "Step 6 — 95% confidence interval for p", "detail": f"p̂ ± 1.96·√(p̂(1−p̂)/n) = [{_fmt(ci[0])}, {_fmt(ci[1])}]"},
    ]
    checks = [{"name": "Normal approximation valid", "status": "ok" if min(n * phat, n * (1 - phat)) >= 10 else "warn",
               "note": f"n·p̂ = {_fmt(n * phat, 1)}, n·(1−p̂) = {_fmt(n * (1 - phat), 1)} — {'both ≥10, approximation fine' if min(n * phat, n * (1 - phat)) >= 10 else 'one is <10, treat p-value cautiously'}."}]
    return {
        "analysis": "proportion_ztest", "title": f"One-proportion z-test — share of '{success}' in '{col}' vs {p0:g}",
        "steps": steps,
        "result": {"n": n, "x": x, "p_hat": _r(phat), "p0": p0, "statistic": _r(z), "p_value": _r(p, 6),
                   "ci_95": [_r(ci[0]), _r(ci[1])], "reject_null_at_5pct": bool(p < 0.05), "success_value": success},
        "reliability": reliability(n, checks, min_n_good=30),
        "chart": {"type": "bar", "params": {"labels": [success, "other"], "values": [x, n - x]}},
    }


ANALYSES = {
    "describe": describe,
    "one_sample_ttest": one_sample_ttest,
    "two_sample_ttest": two_sample_ttest,
    "paired_ttest": paired_ttest,
    "chi_square_independence": chi_square_independence,
    "one_way_anova": one_way_anova,
    "correlation_regression": correlation_regression,
    "proportion_ztest": proportion_ztest,
}

ANALYSIS_HELP = """describe{} — dataset overview, always safe
one_sample_ttest{column, mu0} — is the mean of a numeric column different from a target value?
two_sample_ttest{value_column, group_column, groups:[a,b]?} — compare a numeric outcome between 2 groups
paired_ttest{column_before, column_after} — compare two numeric measurements on the SAME rows (before/after)
chi_square_independence{column1, column2} — are two categorical columns associated?
one_way_anova{value_column, group_column} — compare a numeric outcome across 3+ groups
correlation_regression{x_column, y_column} — relationship + trend line between two numeric columns
proportion_ztest{column, success_value?, p0} — is the share of a category different from a target proportion?"""


def run_analysis(df: pd.DataFrame, analysis: str, params: dict) -> dict:
    fn = ANALYSES.get(analysis)
    if fn is None:
        raise AnalysisError(f"Unknown analysis '{analysis}'.")
    return fn(df, params or {})
