"""
charts.py — deterministic chart engine.

The AI only PICKS a template and its parameters (e.g. {"type":"normal","params":{"mean":50,"sd":10}}).
All numbers on the chart are computed here with numpy/scipy, so curves are mathematically exact
and any simulated data is clearly labelled "illustrative".

build_chart(spec) -> a JSON-safe dict the browser turns into a Chart.js chart.
"""

import json
import re

import numpy as np
from scipy import stats

C = ["#5b4bff", "#e5484d", "#12a56b", "#e58a00", "#8a5cff", "#0ea5e9"]
GREY = "#8b90a7"


# ───────────────────────────── helpers ─────────────────────────────

def _num(p, key, default, lo, hi):
    try:
        v = float(p.get(key, default))
        if not np.isfinite(v):
            return default
        return min(max(v, lo), hi)
    except (TypeError, ValueError):
        return default


def _list(p, key, default, lo_len=2, hi_len=8):
    v = p.get(key)
    if isinstance(v, list) and lo_len <= len(v) <= hi_len:
        try:
            out = [float(x) for x in v]
            if all(np.isfinite(out)) and all(x > 0 for x in out):
                return out
        except (TypeError, ValueError):
            pass
    return default


def _pts(x, y):
    return [{"x": round(float(a), 5), "y": round(float(b), 6)}
            for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]


def _curve(x, y, label, color, fill=False, dash=None, width=2.5):
    d = {"type": "scatter", "label": label, "data": _pts(x, y), "showLine": True,
         "borderColor": color, "borderWidth": width, "pointRadius": 0, "pointHoverRadius": 4,
         "tension": 0.25, "fill": fill}
    if fill:
        d["backgroundColor"] = color + "33"
    if dash:
        d["borderDash"] = dash
    return d


def _points(x, y, label, color, r=4):
    return {"type": "scatter", "label": label, "data": _pts(x, y), "showLine": False,
            "pointRadius": r, "backgroundColor": color + "cc", "borderColor": color}


def _vline(x, ymax, label, color=C[1]):
    return _curve([x, x], [0, ymax], label, color, dash=[6, 4], width=2)


def _spec(kind, title, datasets, xlabel="", ylabel="", caption="", labels=None, **options):
    return {"type": kind, "title": title, "datasets": datasets, "labels": labels,
            "xlabel": xlabel, "ylabel": ylabel, "caption": caption, "options": options}


# ─────────────────────── continuous distributions ───────────────────────

def normal(p):
    m, s = _num(p, "mean", 0, -1e6, 1e6), _num(p, "sd", 1, 1e-6, 1e6)
    x = np.linspace(m - 4 * s, m + 4 * s, 241)
    lo, hi = m - 1.96 * s, m + 1.96 * s
    xs = np.linspace(lo, hi, 120)
    return _spec("scatter", "Normal distribution",
                 [_curve(x, stats.norm.pdf(x, m, s), f"N({m:g}, {s:g}²)", C[0]),
                  _curve(xs, stats.norm.pdf(xs, m, s), "central 95%", C[2], fill=True, width=0)],
                 "x", "Probability density",
                 "Bell-shaped and symmetric. Shaded area = central 95% (mean ± 1.96 SD); ±1 SD holds ≈68%, ±3 SD ≈99.7%.")


def t_dist(p):
    df = _num(p, "df", 5, 1, 200)
    x = np.linspace(-4.5, 4.5, 241)
    return _spec("scatter", "Student's t vs standard normal",
                 [_curve(x, stats.t.pdf(x, df), f"t (df = {df:g})", C[0]),
                  _curve(x, stats.norm.pdf(x), "N(0,1)", GREY, dash=[6, 4], width=2)],
                 "t", "Probability density",
                 "The t distribution has heavier tails than the normal; it approaches N(0,1) as degrees of freedom grow.")


def chi2(p):
    df = _num(p, "df", 4, 1, 100)
    crit = stats.chi2.ppf(0.95, df)
    x = np.linspace(0, max(stats.chi2.ppf(0.999, df), crit * 1.3), 300)
    xs = np.linspace(crit, x[-1], 80)
    return _spec("scatter", f"Chi-square distribution (df = {df:g})",
                 [_curve(x, stats.chi2.pdf(x, df), "χ² density", C[0]),
                  _curve(xs, stats.chi2.pdf(xs, df), "rejection region (5%)", C[1], fill=True, width=0)],
                 "χ²", "Probability density",
                 f"Right-skewed. Critical value at 5% level = {crit:.3f}; a test statistic beyond it rejects H₀.")


def f_dist(p):
    d1, d2 = _num(p, "d1", 5, 1, 200), _num(p, "d2", 20, 1, 200)
    crit = stats.f.ppf(0.95, d1, d2)
    x = np.linspace(0.001, max(stats.f.ppf(0.995, d1, d2), crit * 1.4), 300)
    xs = np.linspace(crit, x[-1], 80)
    return _spec("scatter", f"F distribution (d1 = {d1:g}, d2 = {d2:g})",
                 [_curve(x, stats.f.pdf(x, d1, d2), "F density", C[0]),
                  _curve(xs, stats.f.pdf(xs, d1, d2), "rejection region (5%)", C[1], fill=True, width=0)],
                 "F", "Probability density",
                 f"Used for variance ratios and ANOVA. 5% critical value = {crit:.3f}.")


def exponential(p):
    lam = _num(p, "rate", 1, 0.01, 100)
    x = np.linspace(0, 5 / lam, 200)
    return _spec("scatter", f"Exponential distribution (rate λ = {lam:g})",
                 [_curve(x, stats.expon.pdf(x, scale=1 / lam), "density f(x)", C[0]),
                  _curve(x, stats.expon.sf(x, scale=1 / lam), "survival S(x) = P(X > x)", C[3], dash=[6, 4])],
                 "x (time)", "Density / survival probability",
                 "Memoryless lifetime model with constant failure rate; mean = 1/λ.")


def uniform(p):
    a = _num(p, "a", 0, -1e6, 1e6)
    b = max(_num(p, "b", 1, -1e6, 1e6), a + 1e-3)
    pad, h = (b - a) * 0.4, 1 / (b - a)
    return _spec("scatter", f"Uniform distribution U({a:g}, {b:g})",
                 [_curve([a - pad, a, a, b, b, b + pad], [0, 0, h, h, 0, 0], "density", C[0], fill=True)],
                 "x", "Probability density", "Every value in the interval is equally likely; density = 1/(b − a).")


def gamma_d(p):
    k, th = _num(p, "shape", 2, 0.2, 60), _num(p, "scale", 1, 0.05, 50)
    x = np.linspace(0, stats.gamma.ppf(0.995, k, scale=th), 250)
    return _spec("scatter", f"Gamma distribution (shape {k:g}, scale {th:g})",
                 [_curve(x, stats.gamma.pdf(x, k, scale=th), "density", C[0], fill=True)],
                 "x", "Probability density", "Right-skewed positive distribution; generalises the exponential and chi-square.")


def beta_d(p):
    a, b = _num(p, "a", 2, 0.2, 50), _num(p, "b", 5, 0.2, 50)
    x = np.linspace(0.001, 0.999, 250)
    return _spec("scatter", f"Beta distribution (a = {a:g}, b = {b:g})",
                 [_curve(x, stats.beta.pdf(x, a, b), "density", C[0], fill=True)],
                 "x (proportion)", "Probability density", "Defined on [0, 1]; used for proportions and as a Bayesian prior.")


# ─────────────────────────── discrete distributions ───────────────────────────

def _pmf_bar(title, ks, pmf, xlabel, caption, hl=None):
    colors = [C[0] + "cc"] * len(ks)
    return _spec("bar", title,
                 [{"type": "bar", "label": "P(X = k)", "data": [round(float(v), 6) for v in pmf],
                   "backgroundColor": colors, "borderRadius": 4}],
                 xlabel, "Probability", caption, labels=[str(int(k)) for k in ks], legend=False)


def binomial(p_):
    n, p = int(_num(p_, "n", 10, 1, 60)), _num(p_, "p", 0.5, 0.01, 0.99)
    ks = np.arange(0, n + 1)
    return _pmf_bar(f"Binomial distribution B({n}, {p:g})", ks, stats.binom.pmf(ks, n, p), "k (number of successes)",
                    f"Mean = np = {n * p:.2f}, variance = np(1−p) = {n * p * (1 - p):.2f}.")


def poisson(p_):
    lam = _num(p_, "lam", 4, 0.2, 40)
    ks = np.arange(0, int(stats.poisson.ppf(0.9995, lam)) + 2)
    return _pmf_bar(f"Poisson distribution (λ = {lam:g})", ks, stats.poisson.pmf(ks, lam), "k (number of events)",
                    f"Counts of rare events; mean = variance = λ = {lam:g}.")


def geometric(p_):
    p = _num(p_, "p", 0.3, 0.03, 0.95)
    ks = np.arange(1, min(int(stats.geom.ppf(0.995, p)) + 1, 40))
    return _pmf_bar(f"Geometric distribution (p = {p:g})", ks, stats.geom.pmf(ks, p), "k (trial of first success)",
                    f"Mean = 1/p = {1 / p:.2f}. Probabilities decay geometrically.")


# ──────────────────────── sampling / inference visuals ────────────────────────

def clt(p):
    x = np.linspace(0.01, 3.2, 300)
    ds = [_curve(x, stats.gamma.pdf(x, n, scale=1 / n), f"n = {n}", col, width=2.5)
          for n, col in zip([1, 5, 30], [C[3], C[2], C[0]])]
    return _spec("scatter", "Central Limit Theorem in action", ds, "Sample mean x̄", "Density",
                 "Population is a skewed Exponential(1). As sample size n grows, the distribution of the sample mean "
                 "becomes narrower (σ/√n) and increasingly bell-shaped.")


def likelihood(p_):
    n, x = int(_num(p_, "n", 10, 2, 200)), int(_num(p_, "x", 7, 0, 200))
    x = min(x, n)
    grid = np.linspace(0.001, 0.999, 300)
    L = stats.binom.pmf(x, n, grid)
    mle = x / n
    return _spec("scatter", f"Likelihood of p ({x} successes in {n} trials)",
                 [_curve(grid, L, "L(p)", C[0], fill=True),
                  _points([mle], [stats.binom.pmf(x, n, mle)], f"MLE  p̂ = {mle:.2f}", C[1], r=7)],
                 "p", "Likelihood L(p)",
                 "The likelihood peaks at the maximum-likelihood estimate p̂ = x/n. Only the total count x (a sufficient statistic) matters, not the order of outcomes.")


def confidence_intervals(p):
    rng = np.random.default_rng(2)
    mu, sd, n, k = 50, 10, 25, 20
    xb = rng.normal(mu, sd / np.sqrt(n), k)
    half = stats.t.ppf(0.975, n - 1) * (sd / np.sqrt(n))
    lo, hi = xb - half, xb + half
    cover = (lo <= mu) & (mu <= hi)
    return _spec("bar", "20 simulated 95% confidence intervals",
                 [{"type": "bar", "label": "interval", "data": [[round(a, 3), round(b, 3)] for a, b in zip(lo, hi)],
                   "backgroundColor": [C[2] + "cc" if c else C[1] + "cc" for c in cover], "borderRadius": 4,
                   "barThickness": 9}],
                 "Value of the mean", "Sample number",
                 f"Dashed line = true mean ({mu}). Green intervals capture it, red ones miss it: {int(cover.sum())} of {k} "
                 "here; about 95% are expected to capture the true mean in the long run. Simulated, illustrative.",
                 labels=[str(i + 1) for i in range(k)], indexAxis="y", vline=mu, legend=False, xmin=40, xmax=60)


def power_curve(p):
    a = _num(p, "alpha", 0.05, 0.001, 0.2)
    za = stats.norm.ppf(1 - a)
    d = np.linspace(0, 1.5, 150)
    ds = [_curve(d, 1 - stats.norm.cdf(za - d * np.sqrt(n)), f"n = {n}", col)
          for n, col in zip([10, 25, 50], [C[3], C[2], C[0]])]
    ds.append(_curve([0, 1.5], [a, a], f"α = {a:g} (Type I error)", GREY, dash=[6, 4], width=1.5))
    return _spec("scatter", "Power of a one-sided z-test", ds, "True effect size δ/σ", "Power = 1 − β",
                 "Power rises with a larger true effect and a larger sample. At δ = 0 the power equals the significance level α.",
                 ymin=0, ymax=1)


def oc_curve(p):
    pr = np.linspace(0, 0.14, 141)
    ds = [_curve(pr, stats.binom.cdf(c, n, pr), f"n = {n}, c = {c}", col)
          for (n, c), col in zip([(50, 2), (100, 4), (100, 2)], [C[0], C[2], C[1]])]
    return _spec("scatter", "Operating Characteristic (OC) curves", ds, "Lot fraction defective p",
                 "Probability of acceptance Pa", "Pa = P(defectives in sample ≤ c). Larger n with the same c makes the plan stricter.",
                 ymin=0, ymax=1)


def sampling_allocation(p):
    Nh = _list(p, "Nh", [4000, 2500, 2000, 1000, 500])
    Sh = _list(p, "Sh", [5, 8, 12, 20, 30][:len(Nh)] if len(Nh) <= 5 else [10] * len(Nh))
    if len(Sh) != len(Nh):
        Sh = [10.0] * len(Nh)
    n = _num(p, "n", 200, 10, 5000)
    Nh, Sh = np.array(Nh), np.array(Sh)
    prop = n * Nh / Nh.sum()
    ney = n * Nh * Sh / (Nh * Sh).sum()
    labels = [f"Stratum {i + 1}" for i in range(len(Nh))]
    return _spec("bar", f"Allocating a sample of n = {n:g} across strata",
                 [{"type": "bar", "label": "Proportional (share = stratum size)", "data": [round(float(v), 1) for v in prop],
                   "backgroundColor": C[0] + "cc", "borderRadius": 4},
                  {"type": "bar", "label": "Neyman (size x variability)", "data": [round(float(v), 1) for v in ney],
                   "backgroundColor": C[3] + "cc", "borderRadius": 4}],
                 "Stratum", "Sample size in stratum",
                 f"Illustrative stratum sizes = {', '.join(f'{v:g}' for v in Nh)} and std. deviations = "
                 f"{', '.join(f'{v:g}' for v in Sh)}. Neyman allocation samples more from large, highly variable strata.",
                 labels=labels)


# ───────────────────────── data-style (simulated, illustrative) ─────────────────────────

def histogram(p):
    rng = np.random.default_rng(7)
    data = rng.normal(50, 10, 200)
    counts, edges = np.histogram(data, bins=12)
    centers = (edges[:-1] + edges[1:]) / 2
    m, s = data.mean(), data.std(ddof=1)
    expected = 200 * (stats.norm.cdf(edges[1:], m, s) - stats.norm.cdf(edges[:-1], m, s))
    return _spec("bar", "Frequency histogram",
                 [{"type": "bar", "label": "Frequency", "data": counts.tolist(), "backgroundColor": C[0] + "bb",
                   "borderRadius": 3, "categoryPercentage": 1.0, "barPercentage": 0.97},
                  {"type": "line", "label": "Normal curve", "data": [round(float(v), 2) for v in expected],
                   "borderColor": C[1], "borderWidth": 2.5, "pointRadius": 0, "tension": 0.4}],
                 "Class interval (mid-point)", "Frequency",
                 f"Simulated sample of 200 values (mean {m:.1f}, SD {s:.1f}) — illustrative, not your data.",
                 labels=[f"{c:.0f}" for c in centers])


def ogive(p):
    rng = np.random.default_rng(3)
    data = rng.normal(50, 12, 200)
    edges = np.linspace(20, 80, 13)
    cum = [int((data <= e).sum()) for e in edges]
    return _spec("scatter", "Cumulative frequency curve (ogive)",
                 [_curve(edges, cum, "less-than ogive", C[0], width=2.5),
                  _points(edges, cum, "_pts", C[0], r=3.5)],
                 "Upper class boundary", "Cumulative frequency",
                 "Read the median from the x-value where cumulative frequency = N/2 (here 100). Simulated, illustrative.")


def scatter_regression(p):
    rng = np.random.default_rng(5)
    r_t = _num(p, "corr", 0.85, -0.99, 0.99)
    x = rng.normal(50, 10, 40)
    z = (x - 50) / 10
    y = 60 + 8 * (r_t * z + np.sqrt(1 - r_t ** 2) * rng.normal(0, 1, 40))
    res = stats.linregress(x, y)
    xl = np.array([x.min(), x.max()])
    return _spec("scatter", "Scatter diagram with least-squares line",
                 [_points(x, y, "observations", C[0]),
                  _curve(xl, res.intercept + res.slope * xl, "fitted line", C[1])],
                 "X (independent)", "Y (dependent)",
                 f"Fitted line ŷ = {res.intercept:.2f} + {res.slope:.2f}x, correlation r = {res.rvalue:.2f}, "
                 f"R² = {res.rvalue ** 2:.2f}. Simulated, illustrative data.")


def boxplot(p):
    rng = np.random.default_rng(9)
    groups = [("A", 50, 8), ("B", 58, 12), ("C", 45, 6), ("D", 62, 15)]
    mins, q1s, meds, q3s, maxs = [], [], [], [], []
    for _, m, s in groups:
        d = rng.normal(m, s, 60)
        a, b, c = np.percentile(d, [25, 50, 75])
        iqr = c - a
        inside = d[(d >= a - 1.5 * iqr) & (d <= c + 1.5 * iqr)]
        mins.append(inside.min()); maxs.append(inside.max()); q1s.append(a); meds.append(b); q3s.append(c)
    rngv = max(maxs) - min(mins)
    dl = rngv * 0.006
    r2 = lambda v: round(float(v), 2)
    return _spec("bar", "Box plot (five-number summary)",
                 [{"type": "bar", "label": "min to max (whiskers)", "data": [[r2(a), r2(b)] for a, b in zip(mins, maxs)],
                   "backgroundColor": GREY, "barThickness": 3, "order": 3},
                  {"type": "bar", "label": "Q1 to Q3 (IQR box)", "data": [[r2(a), r2(b)] for a, b in zip(q1s, q3s)],
                   "backgroundColor": C[0] + "bb", "barThickness": 46, "borderRadius": 4, "order": 2},
                  {"type": "bar", "label": "median", "data": [[r2(m - dl), r2(m + dl)] for m in meds],
                   "backgroundColor": C[1], "barThickness": 46, "order": 1}],
                 "Group", "Value",
                 "Box = middle 50% of the data, line = median, whiskers = range without outliers (1.5×IQR rule). "
                 "Simulated, illustrative.", labels=[g[0] for g in groups], grouped=False, ymin=int(min(mins) // 10 * 10 - 10))


def control_chart(p):
    rng = np.random.default_rng(21)
    mu, sigma, n, k = 50, 2.0, 5, 25
    sx = sigma / np.sqrt(n)
    xb = rng.normal(mu, sx, k)
    xb[16] = mu + 3.5 * sx
    ucl, lcl = mu + 3 * sx, mu - 3 * sx
    out = (xb > ucl) | (xb < lcl)
    lab = [str(i + 1) for i in range(k)]
    flat = lambda v: [round(v, 3)] * k
    return _spec("line", "X̄ control chart",
                 [{"type": "line", "label": "sample mean", "data": [round(float(v), 3) for v in xb],
                   "borderColor": C[0], "borderWidth": 2, "tension": 0,
                   "pointBackgroundColor": [C[1] if o else C[0] for o in out],
                   "pointRadius": [7 if o else 3.5 for o in out]},
                  {"type": "line", "label": f"UCL = {ucl:.2f}", "data": flat(ucl), "borderColor": C[1], "borderDash": [6, 4],
                   "pointRadius": 0, "borderWidth": 1.8},
                  {"type": "line", "label": f"CL = {mu:.2f}", "data": flat(mu), "borderColor": C[2], "pointRadius": 0,
                   "borderWidth": 1.8},
                  {"type": "line", "label": f"LCL = {lcl:.2f}", "data": flat(lcl), "borderColor": C[1], "borderDash": [6, 4],
                   "pointRadius": 0, "borderWidth": 1.8}],
                 "Sample number", "Sample mean",
                 "Limits are CL ± 3σ/√n. A point outside the limits (red) signals the process is out of statistical control. "
                 "Simulated, illustrative.", labels=lab)


def time_series(p):
    rng = np.random.default_rng(4)
    t = np.arange(48)
    y = 100 + 1.2 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 48)
    ma = np.full(48, np.nan)
    for i in range(6, 42):  # centred 2×12 moving average
        ma[i] = (0.5 * y[i - 6] + y[i - 5:i + 6].sum() + 0.5 * y[i + 6]) / 12
    res = stats.linregress(t, y)
    rnd = lambda arr: [None if not np.isfinite(v) else round(float(v), 2) for v in arr]
    return _spec("line", "Time series: trend + seasonality",
                 [{"type": "line", "label": "observed", "data": rnd(y), "borderColor": C[0], "borderWidth": 2,
                   "pointRadius": 0, "tension": 0.25},
                  {"type": "line", "label": "12-month moving average", "data": rnd(ma), "borderColor": C[3], "borderWidth": 3,
                   "pointRadius": 0, "tension": 0.3},
                  {"type": "line", "label": "linear trend", "data": rnd(res.intercept + res.slope * t),
                   "borderColor": C[1], "borderDash": [6, 4], "borderWidth": 2, "pointRadius": 0}],
                 "Month", "Value",
                 "The moving average smooths out the seasonal swing to reveal the trend. Simulated, illustrative.",
                 labels=[str(i + 1) for i in range(48)])


def lorenz(p):
    rng = np.random.default_rng(2)
    inc = np.sort(rng.lognormal(0, 0.8, 1000))
    cum = np.concatenate([[0], np.cumsum(inc) / inc.sum()])
    pop = np.linspace(0, 1, len(cum))
    _trap = getattr(np, 'trapezoid', None) or np.trapz
    gini = 1 - 2 * _trap(cum, pop)
    idx = np.linspace(0, len(cum) - 1, 60).astype(int)
    return _spec("scatter", "Lorenz curve",
                 [_curve(pop[idx], cum[idx], f"Lorenz curve (Gini ≈ {gini:.2f})", C[0], fill=True),
                  _curve([0, 1], [0, 1], "line of equality", GREY, dash=[6, 4], width=2)],
                 "Cumulative share of population", "Cumulative share of income",
                 "The more the curve bows below the equality line, the greater the inequality (larger Gini). Simulated, illustrative.",
                 xmin=0, xmax=1, ymin=0, ymax=1)


def qq_plot(p):
    rng = np.random.default_rng(6)
    data = rng.normal(50, 10, 50)
    (osm, osr), (slope, icpt, r) = stats.probplot(data, dist="norm")
    return _spec("scatter", "Normal Q–Q plot",
                 [_points(osm, osr, "sample quantiles", C[0]),
                  _curve([osm.min(), osm.max()], [icpt + slope * osm.min(), icpt + slope * osm.max()], "reference line", C[1])],
                 "Theoretical quantiles", "Sample quantiles",
                 f"Points close to the line (r = {r:.3f}) support normality; systematic bends indicate skewness or heavy tails. Simulated, illustrative.")


def anova_groups(p):
    rng = np.random.default_rng(8)
    means, k, n = [20, 24, 30], 3, 8
    ds, allv = [], []
    for g in range(k):
        v = rng.normal(means[g], 3, n)
        allv.extend(v)
        jit = g + 1 + rng.uniform(-0.18, 0.18, n)
        ds.append(_points(jit, v, f"Group {g + 1}", C[[0, 2, 3][g]]))
        ds.append(_points([g + 1], [v.mean()], "_mean", "#1b1d2b", r=9))
    grand = float(np.mean(allv))
    ds.append(_curve([0.5, 3.5], [grand, grand], f"grand mean = {grand:.1f}", GREY, dash=[6, 4], width=2))
    F, pv = stats.f_oneway(*[[pt["y"] for pt in ds[2 * g]["data"]] for g in range(k)])
    return _spec("scatter", "One-way ANOVA: between vs within variation", ds, "Group", "Response",
                 f"Big gaps between group means relative to spread inside groups give a large F (here F = {F:.1f}, p = {pv:.4f}). "
                 "Dark dots = group means. Simulated, illustrative.",
                 xmin=0.5, xmax=3.5, xticklabels=["Group 1", "Group 2", "Group 3"])


def bar_generic(p):
    labels = p.get("labels") if isinstance(p.get("labels"), list) else None
    vals = _list(p, "values", [])
    if not labels or len(labels) != len(vals):
        labels, vals = ["A", "B", "C", "D", "E"], [23, 35, 18, 27, 12]
    labels = [str(l)[:18] for l in labels]
    return _spec("bar", "Bar chart", [{"type": "bar", "label": "Frequency", "data": vals,
                                       "backgroundColor": [C[i % 6] + "cc" for i in range(len(vals))], "borderRadius": 5}],
                 "Category", "Frequency", "Bar heights compare category counts. Illustrative values.", labels=labels, legend=False)


def pie_generic(p):
    s = bar_generic(p)
    s["type"], s["title"] = "pie", "Pie chart"
    for d in s["datasets"]:
        d.pop("type", None); d.pop("borderRadius", None)
    s["caption"] = "Each slice shows a category's share of the total. Illustrative values."
    s["options"] = {}
    return s


TEMPLATES = {
    "normal": normal, "t": t_dist, "chi2": chi2, "f": f_dist, "exponential": exponential, "uniform": uniform,
    "gamma": gamma_d, "beta": beta_d, "binomial": binomial, "poisson": poisson, "geometric": geometric,
    "clt": clt, "likelihood": likelihood, "confidence_intervals": confidence_intervals, "power": power_curve,
    "oc_curve": oc_curve, "sampling_allocation": sampling_allocation, "histogram": histogram, "ogive": ogive,
    "scatter_regression": scatter_regression, "boxplot": boxplot, "control_chart": control_chart,
    "time_series": time_series, "lorenz": lorenz, "qq": qq_plot, "anova_groups": anova_groups,
    "bar": bar_generic, "pie": pie_generic,
}

# Shown to the AI so it knows what it may choose (kept in sync with TEMPLATES).
TEMPLATE_HELP = """normal{mean,sd} | t{df} | chi2{df} | f{d1,d2} | exponential{rate} | uniform{a,b} | gamma{shape,scale} | beta{a,b}
| binomial{n,p} | poisson{lam} | geometric{p} | clt | likelihood{n,x} | confidence_intervals | power{alpha} | oc_curve
| sampling_allocation{Nh:[..],Sh:[..],n} | histogram | ogive | scatter_regression{corr} | boxplot | control_chart
| time_series | lorenz | qq | anova_groups | bar{labels:[..],values:[..]} | pie{labels:[..],values:[..]}"""

# Keyword fallback (used when the AI gives no valid chart): first match wins.
FALLBACK_RULES = [
    (r"neyman[- ]pearson|critical region|most powerful|power function|type i|type ii|level of significance|hypothes|p-value|significance", "power"),
    (r"factori[sz]|sufficien|likelihood|mle|maximum likelihood|estimat|fisher|rao|cram|unbiased|umvue|efficien|consisten|bayes", "likelihood"),
    (r"confidence interval|interval estimation", "confidence_intervals"),
    (r"stratif|allocation|neyman allocation", "sampling_allocation"),
    (r"central limit|sampling distribution|standard error|srs|systematic|cluster|sampling", "clt"),
    (r"control chart|quality control|x-bar|spc|six sigma|cusum|ewma|process control", "control_chart"),
    (r"acceptance sampling|oc curve|inspection|sampling plan", "oc_curve"),
    (r"anova|analysis of variance|design of experiment|factorial|latin square|randomi[sz]ed block|crd|rbd", "anova_groups"),
    (r"chi[- ]?square|goodness of fit|attributes|contingency", "chi2"),
    (r"\bt[- ]?test|student|t[- ]distribution", "t"),
    (r"f[- ]test|f[- ]distribution|variance ratio", "f"),
    (r"exponential|memoryless|reliab|failure|hazard|survival|life", "exponential"),
    (r"poisson", "poisson"), (r"binomial|bernoulli", "binomial"), (r"geometric|negative binomial", "geometric"),
    (r"uniform", "uniform"), (r"gamma", "gamma"), (r"beta", "beta"),
    (r"normal|gaussian|z-score|z score|empirical rule|standard score", "normal"),
    (r"correlation|regression|least squares|covariance|scatter|multiple|partial|econometric|heterosced|autocorrelation|multicollinear", "scatter_regression"),
    (r"time series|trend|seasonal|forecast|moving average|arima|smoothing|index number", "time_series"),
    (r"lorenz|gini|inequality|concentration|income", "lorenz"),
    (r"q-q|qq|normality|probability plot", "qq"),
    (r"box plot|boxplot|quartile|dispersion|iqr|five[- ]number|range|skewness|kurtosis|outlier", "boxplot"),
    (r"ogive|cumulative|percentile|decile|median|mode", "ogive"),
    (r"histogram|frequency|class interval|descriptive|mean|variance|moments", "histogram"),
    (r"pie", "pie"), (r"bar|survey|attribute|categor", "bar"),
]


def fallback_type(topic: str) -> str:
    t = topic.lower()
    for pat, name in FALLBACK_RULES:
        if re.search(pat, t):
            return name
    return "histogram"


def build_chart(spec: dict | None, topic: str = "") -> dict:
    """Never raises: an invalid/missing spec falls back to a keyword-chosen template."""
    name = (spec or {}).get("type") if isinstance(spec, dict) else None
    params = (spec or {}).get("params") if isinstance(spec, dict) else None
    params = params if isinstance(params, dict) else {}
    if name not in TEMPLATES:
        name, params = fallback_type(topic), {}
    try:
        out = TEMPLATES[name](params)
    except Exception:
        out = TEMPLATES[fallback_type(topic) if fallback_type(topic) != name else "histogram"]({})
    out["template"] = name
    json.dumps(out, allow_nan=False)  # guarantee strict-JSON-safe output
    return out


# ─────────────────────── real-data chart builders (Industry Analysis Engine) ───────────────────────
# Unlike the teaching templates above, these draw the USER'S actual numbers — never simulated.

def _clean_arr(v, cap=3000):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) > cap:
        idx = np.linspace(0, len(a) - 1, cap).astype(int)
        a = a[idx]
    return a


def histogram_data(p):
    vals = _clean_arr(p.get("values", []))
    label = str(p.get("label", "value"))[:40]
    if len(vals) < 2:
        return _spec("bar", "Not enough data", [], label, "Frequency", "Not enough numeric values to plot.")
    bins = min(20, max(5, int(np.sqrt(len(vals)))))
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return _spec("bar", f"Distribution of {label}",
                 [{"type": "bar", "label": "Frequency", "data": counts.tolist(), "backgroundColor": C[0] + "cc",
                   "borderRadius": 3, "categoryPercentage": 1.0, "barPercentage": 0.95}],
                 label, "Frequency", f"n = {len(vals)}. Your actual data — not simulated.",
                 labels=[f"{c:.2g}" for c in centers], legend=False)


def boxplot_data(p):
    groups = p.get("groups") or {}
    ref = p.get("reference_line")
    names, mins, q1s, meds, q3s, maxs = [], [], [], [], [], []
    for name, vals in groups.items():
        a = _clean_arr(vals)
        if len(a) < 2:
            continue
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        iqr = q3 - q1
        inside = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
        names.append(str(name)[:20]); mins.append(inside.min()); maxs.append(inside.max())
        q1s.append(q1); meds.append(med); q3s.append(q3)
    if not names:
        return _spec("bar", "Not enough data", [], "", "", "Not enough numeric values to plot.")
    rngv = max(maxs) - min(mins) or 1
    dl = rngv * 0.006
    r2 = lambda v: round(float(v), 4)
    datasets = [
        {"type": "bar", "label": "min to max (whiskers)", "data": [[r2(a), r2(b)] for a, b in zip(mins, maxs)],
         "backgroundColor": GREY, "barThickness": 3, "order": 3},
        {"type": "bar", "label": "Q1 to Q3 (IQR box)", "data": [[r2(a), r2(b)] for a, b in zip(q1s, q3s)],
         "backgroundColor": C[0] + "bb", "barThickness": 46, "borderRadius": 4, "order": 2},
        {"type": "bar", "label": "median", "data": [[r2(m - dl), r2(m + dl)] for m in meds],
         "backgroundColor": C[1], "barThickness": 46, "order": 1},
    ]
    opts = dict(grouped=False, ymin=round(min(mins) - rngv * 0.08, 2), ymax=round(max(maxs) + rngv * 0.08, 2))
    if ref is not None:
        opts["vline"] = None  # vline plugin is x-axis only; reference shown via caption instead
    cap = "Box = middle 50% (IQR), line = median, whiskers = range excluding outliers. Your actual data."
    if ref is not None:
        cap += f" Reference value: {ref:g}."
    return _spec("bar", "Box plot", datasets, "Group", "Value", cap, labels=names, **opts)


def scatter_regression_data(p):
    x, y = _clean_arr(p.get("x", [])), _clean_arr(p.get("y", []))
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 3:
        return _spec("scatter", "Not enough data", [], "", "", "Not enough paired values to plot.")
    slope, intercept = p.get("slope"), p.get("intercept")
    if slope is None or intercept is None:
        res = np.polyfit(x, y, 1)
        slope, intercept = float(res[0]), float(res[1])
    xl = np.array([x.min(), x.max()])
    idx = np.linspace(0, n - 1, min(n, 1500)).astype(int)
    return _spec("scatter", "Scatter diagram with fitted line",
                 [_points(x[idx], y[idx], "observations", C[0]),
                  _curve(xl, intercept + slope * xl, "fitted line", C[1])],
                 str(p.get("xlabel", "x"))[:30], str(p.get("ylabel", "y"))[:30],
                 f"Fitted line: y = {intercept:.3g} + {slope:.3g}x. Your actual data.")


def bar_data(p):
    return bar_generic(p)


TEMPLATES.update({
    "histogram_data": histogram_data,
    "boxplot_data": boxplot_data,
    "scatter_regression_data": scatter_regression_data,
})
