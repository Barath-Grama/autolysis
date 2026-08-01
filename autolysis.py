#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "httpx>=0.27",
#   "rich>=13.7",
#   "markdown>=3.6",
# ]
# ///
"""
Autolysis - an intelligent, fully automated CSV analysis engine.

Autolysis profiles a CSV file, asks a Google Gemini model which of five
supported analyses are most appropriate for the data, runs those analyses
locally with pandas / scikit-learn / matplotlib, and asks Gemini a second
time to narrate the numeric findings into a polished README.md report.

Usage:
    uv run autolysis.py path/to/dataset.csv
    uv run autolysis.py path/to/dataset.csv --output-dir out --html
    uv run autolysis.py path/to/dataset.csv --offline      # no LLM calls

Environment variables:
    GEMINI_API_KEY / GOOGLE_API_KEY   Gemini API key (required unless --offline)
    GEMINI_MODEL                      Overrides the default model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display server required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich is a soft dependency
    _HAS_RICH = False

try:
    import markdown as _markdown_lib

    _HAS_MARKDOWN = True
except ImportError:  # pragma: no cover - markdown is only needed for --html
    _HAS_MARKDOWN = False


# --------------------------------------------------------------------------- #
# Constants & exceptions
# --------------------------------------------------------------------------- #

ANALYSIS_VOCAB = [
    "correlation",
    "outliers",
    "clustering",
    "time_series",
    "category_analysis",
]

DEFAULT_MODEL = "gemini-2.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TIME_KEYWORDS = ("year", "date", "time", "month", "period", "quarter")
CACHE_DIR = Path(".autolysis_cache")

# Tried in order. Real-world CSVs are frequently Windows-Latin exports rather
# than UTF-8; latin-1 is last because it decodes any byte sequence and so
# always succeeds, making it the terminal fallback rather than a real guess.
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# Only these are worth a second attempt. A 400 (malformed request) or 404
# (unknown model) is deterministic — retrying it four times with exponential
# backoff just turns an instant failure into a slow one.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

MAX_CLUSTERS = 8          # widest k considered by the silhouette sweep
SILHOUETTE_SAMPLE_CAP = 5000  # silhouette is O(n^2); cap the rows it scores

log = logging.getLogger("autolysis")


class AuthenticationError(Exception):
    """Raised when no Gemini API key is configured (and --offline was not passed)."""


class AutolysisError(Exception):
    """Base exception for pipeline-level failures that should abort with a message."""


def redact_secret(text: str, secret: str) -> str:
    """Strip an API key out of text bound for a log line or exception message.

    Defence in depth: the key is sent as a header rather than a query
    parameter, but third-party error strings are not under our control, so
    anything user-visible is scrubbed on the way out regardless.
    """
    if not secret:
        return text
    return text.replace(secret, "***REDACTED***")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    api_key: str
    model: str = DEFAULT_MODEL
    max_analyses: int = 3
    top_n_categories: int = 8
    output_dir: Path = Path("output")
    offline: bool = False
    use_cache: bool = False
    html: bool = False
    verbose: bool = False


def load_config(args: argparse.Namespace) -> Config:
    """Resolve configuration from CLI args and environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    if not api_key and not args.offline:
        raise AuthenticationError(
            "No Gemini API key found. Set the GEMINI_API_KEY or GOOGLE_API_KEY "
            "environment variable, or re-run with --offline to skip LLM calls."
        )

    model = args.model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    return Config(
        api_key=api_key,
        model=model,
        max_analyses=args.max_analyses,
        offline=args.offline,
        use_cache=args.cache,
        html=args.html,
        verbose=args.verbose,
    )


# --------------------------------------------------------------------------- #
# Gemini client
# --------------------------------------------------------------------------- #


class GeminiClient:
    """Thin HTTPS wrapper around the Gemini `generateContent` REST endpoint.

    Uses httpx directly (rather than the full SDK) to keep the dependency
    footprint minimal, and implements exponential-backoff retries for
    transient network/HTTP failures.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 4,
        timeout: float = 60.0,
        base_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_delay = base_delay

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        """Send a single-turn prompt to Gemini and return the text response."""
        # The key travels in a header, never the URL: httpx embeds the full URL
        # in its exception messages, so a query-string key would leak into every
        # retry log line and error report.
        url = f"{GEMINI_BASE_URL}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        generation_config: dict[str, Any] = {"temperature": 0.4}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        delay = self.base_delay
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                if resp.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Gemini API rejected the configured API key (HTTP {resp.status_code})."
                    )

                resp.raise_for_status()
                return _extract_text(resp.json())

            except AuthenticationError:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUSES:
                    raise AutolysisError(
                        f"Gemini API rejected the request (HTTP {status}); not retryable: "
                        f"{redact_secret(str(exc), self.api_key)}"
                    ) from exc
                last_exc = exc
                # A server-supplied Retry-After outranks our own backoff guess.
                wait = _retry_after_seconds(exc.response)
                if wait is None:
                    wait = delay
            except (httpx.TransportError, KeyError, ValueError) as exc:
                last_exc = exc
                wait = delay

            if attempt == self.max_retries:
                break
            log.debug(
                "Gemini call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt, self.max_retries, wait, redact_secret(str(last_exc), self.api_key),
            )
            time.sleep(wait)
            delay *= 2

        raise AutolysisError(
            f"Gemini API call failed after {self.max_retries} attempts: "
            f"{redact_secret(str(last_exc), self.api_key)}"
        )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a Retry-After header, if the server sent one in delta-seconds form."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None  # HTTP-date form — fall back to our own backoff


def _extract_text(data: dict) -> str:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AutolysisError(f"Unexpected Gemini response shape: {data}") from exc


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()


def cached_generate(client: GeminiClient, prompt: str, *, json_mode: bool = False, use_cache: bool = False) -> str:
    """Call the Gemini client, transparently caching responses to disk when enabled.

    Caching avoids repeated API costs/latency when iterating on the same
    dataset during development (e.g. tweaking chart styling).
    """
    if not use_cache:
        return client.generate(prompt, json_mode=json_mode)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(prompt, client.model)}.txt"
    if cache_file.exists():
        log.debug("Cache hit: %s", cache_file.name)
        return cache_file.read_text(encoding="utf-8")

    result = client.generate(prompt, json_mode=json_mode)
    cache_file.write_text(result, encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Data profiling
# --------------------------------------------------------------------------- #


def profile_dataset(df: pd.DataFrame) -> dict:
    """Build a compact metadata bundle describing the dataset's structure.

    Only aggregate statistics leave the machine. For categorical columns that
    means cardinality and the shape of the frequency distribution, never the
    values themselves: the top-5 values of a column can be names, emails or
    diagnoses, and those are raw data however they are counted. Frequencies
    alone carry the signal the router needs — is this column low-cardinality,
    is it skewed — without the contents.
    """
    profile: dict[str, Any] = {
        "rows": len(df),
        "cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        null_pct = float(series.isna().mean() * 100)

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                profile["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_pct": null_pct,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
            else:
                profile["columns"][col] = {
                    "dtype": str(series.dtype),
                    "null_pct": null_pct,
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                }
        else:
            counts = series.value_counts()
            profile["columns"][col] = {
                "dtype": "object",
                "null_pct": null_pct,
                "n_unique": int(series.nunique(dropna=True)),
                "top_frequencies": [int(v) for v in counts.head(5)],
            }

    return profile


# --------------------------------------------------------------------------- #
# LLM Call #1 — analysis routing
# --------------------------------------------------------------------------- #


def build_routing_prompt(profile: dict, max_analyses: int = 3) -> str:
    """Construct the metadata prompt for LLM analysis selection."""
    col_summaries = []
    for col, stats in profile["columns"].items():
        if "mean" in stats:
            mean_s = f"{stats['mean']:.2f}" if stats["mean"] is not None else "NA"
            std_s = f"{stats['std']:.2f}" if stats["std"] is not None else "NA"
            col_summaries.append(
                f"  - {col} [numeric]: mean={mean_s}, std={std_s}, nulls={stats['null_pct']:.1f}%"
            )
        else:
            freqs = ", ".join(str(c) for c in stats["top_frequencies"])
            col_summaries.append(
                f"  - {col} [categorical]: {stats['n_unique']} distinct, "
                f"top-5 counts=[{freqs}], nulls={stats['null_pct']:.1f}%"
            )

    cols_text = "\n".join(col_summaries)
    vocab = ", ".join(f'"{v}"' for v in ANALYSIS_VOCAB)

    return (
        f"Dataset: {profile['rows']} rows x {profile['cols']} columns "
        f"({profile['duplicate_rows']} duplicate rows).\n"
        f"Columns:\n{cols_text}\n\n"
        f"Select up to {max_analyses} analyses most appropriate for this dataset.\n"
        f"Respond ONLY with a JSON array from this set: [{vocab}]\n"
        f'Example: ["correlation", "outliers"]'
    )


def parse_routing_response(raw: str, max_analyses: int = 3) -> list[str]:
    """Parse the LLM's routing response into a validated list of analysis IDs.

    Unrecognized identifiers are silently dropped so a non-compliant LLM
    response can never cause the Local Analysis Engine to dispatch an
    unsupported routine. Falls back to a safe default pair if the response
    cannot be parsed at all.
    """
    text = raw.strip()

    # Strip markdown code fences, e.g. ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            log.warning("Could not parse routing response; falling back to default analyses.")
            return ["correlation", "outliers"]
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.warning("Could not parse routing response; falling back to default analyses.")
            return ["correlation", "outliers"]

    if not isinstance(parsed, list):
        return ["correlation", "outliers"]

    selected: list[str] = []
    for item in parsed:
        if isinstance(item, str) and item in ANALYSIS_VOCAB and item not in selected:
            selected.append(item)

    return selected[:max_analyses] or ["correlation", "outliers"]


# --------------------------------------------------------------------------- #
# Local Analysis Engine
# --------------------------------------------------------------------------- #


def run_correlation(df: pd.DataFrame, out_dir: Path) -> dict:
    """Pearson correlation heatmap across all numeric columns."""
    numeric_df = df.select_dtypes("number")
    if numeric_df.shape[1] < 2:
        return {"skipped": True, "reason": "Fewer than 2 numeric columns"}

    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=corr.shape[0] <= 12, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Correlation Between Numeric Features")
    chart_path = out_dir / "correlation_heatmap.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    return {
        "top_correlations": [{"a": a, "b": b, "r": round(r, 3)} for a, b, r in pairs[:5]],
        "chart": str(chart_path),
    }


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Return the (lower, upper) Tukey fences for a numeric series."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def flag_iqr_outliers(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Boolean mask of values strictly outside the given Tukey fences.

    Values exactly AT a fence are intentionally NOT flagged (strict
    inequality), matching standard Tukey-fence convention.
    """
    return (series < lower) | (series > upper)


def run_outliers(df: pd.DataFrame, out_dir: Path) -> dict:
    """IQR-based outlier detection with a z-scored box-plot summary."""
    numeric_df = df.select_dtypes("number")
    if numeric_df.shape[1] == 0:
        return {"skipped": True, "reason": "No numeric columns"}

    outlier_counts: dict[str, int] = {}
    z_frames = []

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue

        lower, upper = iqr_bounds(series)
        mask = flag_iqr_outliers(series, lower, upper)
        outlier_counts[col] = int(mask.sum())

        std = series.std()
        if std and std > 0:
            z_frames.append(pd.DataFrame({"feature": col, "z": (series - series.mean()) / std}))

    if not z_frames:
        return {"skipped": True, "reason": "No variance in numeric columns"}

    plot_df = pd.concat(z_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(outlier_counts))))
    sns.boxplot(data=plot_df, x="z", y="feature", ax=ax, orient="h")
    ax.set_title("Outlier Structure Across Key Numeric Features (z-scored)")
    ax.set_xlabel("Standardized value (std devs from mean)")
    chart_path = out_dir / "outliers_boxplot.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"outlier_counts": outlier_counts, "chart": str(chart_path)}


def clustering_axis_scores(df: pd.DataFrame) -> pd.Series:
    """Score numeric columns by how much cluster structure each one exposes.

    Ranking by raw variance makes the choice an artefact of units: a salary in
    dollars will always outrank a rating in [0, 1] however much structure the
    rating carries. Standardising first does not rescue a dispersion measure
    either — every standardised column has variance 1 by construction, and two
    gaussians of different widths are literally the same data once scaled, so
    no dispersion statistic can (or should) separate them.

    Excess kurtosis is scale-invariant *and* responds to what K-Means actually
    wants. A bimodal column — two separated lumps — is strongly platykurtic; a
    single gaussian sits near zero; an outlier-dominated column is leptokurtic
    and makes a poor axis. Lower is better, so the score is negated to keep
    "largest wins" semantics. Constant columns have undefined kurtosis and drop
    out as NaN, which is correct: they carry no clustering signal at all.
    """
    scores: dict[str, float] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.nunique() < 2:
            scores[col] = np.nan
            continue
        kurtosis = series.kurt()  # NaN for fewer than 4 observations
        scores[col] = np.nan if pd.isna(kurtosis) else -float(kurtosis)
    return pd.Series(scores, dtype=float)


def select_k_by_silhouette(
    scaled: np.ndarray, k_values: range, random_state: int = 42
) -> tuple[int, dict[int, float]]:
    """Choose k by mean silhouette score, returning (best_k, all scores).

    This replaces an elbow heuristic that scored second differences of inertia.
    That construction could only ever nominate an *interior* k — with k drawn
    from 2..5 it was structurally incapable of returning 2 or 5, so a genuinely
    two-cluster dataset was unreachable. Silhouette is defined independently at
    every k, so no candidate is excluded by the shape of the formula.

    Scoring is capped at a sample of the rows because silhouette is O(n^2).
    """
    scores: dict[int, float] = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(scaled)
        if len(set(labels)) < 2:  # degenerate fit; silhouette is undefined
            continue
        scores[k] = round(
            float(
                silhouette_score(
                    scaled,
                    labels,
                    sample_size=min(len(scaled), SILHOUETTE_SAMPLE_CAP),
                    random_state=random_state,
                )
            ),
            4,
        )

    if not scores:
        return min(k_values), {}
    return max(scores, key=lambda k: scores[k]), scores


def run_clustering(df: pd.DataFrame, out_dir: Path) -> dict:
    """K-Means clustering on the two most dispersed numeric columns."""
    # Note: no dropna(axis=1) here. Discarding an entire column because one
    # cell is missing throws away a candidate axis over a single NaN; the
    # row-wise dropna below is enough, and the length guard catches pairs
    # whose non-null rows barely overlap.
    numeric_df = df.select_dtypes("number")
    if numeric_df.shape[1] < 2:
        return {"skipped": True, "reason": "Fewer than 2 numeric columns"}

    axis_scores = clustering_axis_scores(numeric_df).dropna()
    if len(axis_scores) < 2:
        return {"skipped": True, "reason": "Fewer than 2 non-constant numeric columns"}

    x_col, y_col = axis_scores.nlargest(2).index
    data = numeric_df[[x_col, y_col]].dropna()

    if len(data) < 10:
        return {"skipped": True, "reason": "Not enough rows for clustering"}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    k_upper = min(MAX_CLUSTERS, len(data) - 1)
    best_k, silhouettes = select_k_by_silhouette(scaled, range(2, k_upper + 1))

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = km_final.fit_predict(scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(data[x_col], data[y_col], c=labels, cmap="tab10", alpha=0.75)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"K-Means Clusters (k={best_k})")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    chart_path = out_dir / "clustering.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cluster_means = (
        data.assign(cluster=labels).groupby("cluster")[[x_col, y_col]].mean().round(2).to_dict(orient="index")
    )

    return {
        "k": best_k,
        "inertia": round(float(km_final.inertia_), 2),
        "silhouette": silhouettes.get(best_k),
        "silhouette_by_k": silhouettes,
        "x_col": x_col,
        "y_col": y_col,
        "cluster_means": cluster_means,
        "chart": str(chart_path),
    }


def find_time_column(df: pd.DataFrame) -> str | None:
    """Heuristically identify a temporal column by name matching."""
    for col in df.columns:
        lc = col.lower()
        if any(kw in lc for kw in TIME_KEYWORDS):
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except (ValueError, TypeError):
                continue
    return None


def run_time_series(df: pd.DataFrame, out_dir: Path) -> dict:
    """Trend line (OLS) of the primary numeric target over the detected time column."""
    time_col = find_time_column(df)
    if time_col is None:
        return {"skipped": True, "reason": "No temporal column detected"}

    numeric_cols = [c for c in df.select_dtypes("number").columns if c != time_col]
    if not numeric_cols:
        return {"skipped": True, "reason": "No numeric target column to trend"}

    target_col = df[numeric_cols].var().idxmax()
    plot_df = df[[time_col, target_col]].dropna()

    # A date column held as strings would otherwise sort lexicographically and
    # be spaced by row position, so coerce it to real timestamps first.
    if not (
        pd.api.types.is_numeric_dtype(plot_df[time_col])
        or pd.api.types.is_datetime64_any_dtype(plot_df[time_col])
    ):
        plot_df = plot_df.assign(**{time_col: pd.to_datetime(plot_df[time_col], errors="coerce")})
        plot_df = plot_df.dropna(subset=[time_col])

    if plot_df[time_col].nunique() < 2:
        return {"skipped": True, "reason": "Insufficient distinct time points"}

    grouped = plot_df.groupby(time_col)[target_col].mean().sort_index()
    y = grouped.to_numpy(dtype=float)

    # Regress against real elapsed time, not row position. Using arange() makes
    # 2000, 2001, 2020 evenly spaced and yields a slope "per observed point",
    # which is unitless and wrong whenever the series is irregular.
    index = grouped.index
    if pd.api.types.is_datetime64_any_dtype(index):
        x = (index - index[0]).total_seconds().to_numpy(dtype=float) / 86400.0
        slope_unit = f"{target_col} per day"
    else:
        x = index.to_numpy(dtype=float)
        slope_unit = f"{target_col} per unit of {time_col}"

    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped.index, y, marker="o", label=target_col)
    ax.plot(grouped.index, trend, linestyle="--", color="firebrick", label="OLS trend")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.set_title(f"{target_col} Over {time_col}")
    ax.legend()
    chart_path = out_dir / "time_series_trend.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "time_col": time_col,
        "target_col": target_col,
        "slope": round(float(slope), 4),
        "slope_unit": slope_unit,
        "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
        "chart": str(chart_path),
    }


def run_category_analysis(df: pd.DataFrame, out_dir: Path, top_n: int = 8) -> dict:
    """Bar chart of the top-N most frequent values for the richest categorical column."""
    cat_cols = [
        c
        for c in df.select_dtypes(exclude="number").columns
        if 1 < df[c].nunique(dropna=True) <= max(50, top_n * 5)
    ]
    if not cat_cols:
        return {"skipped": True, "reason": "No suitable categorical columns"}

    target_col = max(cat_cols, key=lambda c: df[c].nunique(dropna=True))
    counts = df[target_col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(counts))))
    labels = counts.index.astype(str)
    sns.barplot(x=counts.to_numpy(), y=labels, hue=labels, ax=ax, orient="h", palette="viridis", legend=False)
    ax.set_xlabel("Count")
    ax.set_ylabel(target_col)
    ax.set_title(f"Top {len(counts)} Categories in '{target_col}'")
    chart_path = out_dir / "category_frequency.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "column": target_col,
        "top_categories": {str(k): int(v) for k, v in counts.to_dict().items()},
        "chart": str(chart_path),
    }


ANALYSIS_FUNCS: dict[str, Callable[[pd.DataFrame, Path], dict]] = {
    "correlation": run_correlation,
    "outliers": run_outliers,
    "clustering": run_clustering,
    "time_series": run_time_series,
    "category_analysis": run_category_analysis,
}


# --------------------------------------------------------------------------- #
# LLM Call #2 — narrative generation
# --------------------------------------------------------------------------- #


def build_narrative_prompt(profile: dict, results: dict[str, dict], dataset_name: str) -> str:
    lines = [
        f"You are a senior data analyst. Write a Markdown report analyzing the "
        f"dataset '{dataset_name}'.",
        f"The dataset has {profile['rows']} rows, {profile['cols']} columns, and "
        f"{profile['duplicate_rows']} duplicate rows.",
        "",
        "Column summary:",
    ]
    for col, stats in profile["columns"].items():
        if "mean" in stats:
            lines.append(
                f"  - {col} [numeric]: mean={stats['mean']}, std={stats['std']}, "
                f"nulls={stats['null_pct']:.1f}%"
            )
        else:
            lines.append(
                f"  - {col} [categorical]: {stats['n_unique']} distinct, "
                f"nulls={stats['null_pct']:.1f}%"
            )

    lines.append("\nAnalysis results:")
    for name, result in results.items():
        if result.get("skipped"):
            continue
        lines.append(f"\n### {name}")
        payload = {k: v for k, v in result.items() if k != "chart"}
        lines.append(json.dumps(payload, default=str, indent=2))
        if "chart" in result:
            lines.append(f"Chart file: {Path(result['chart']).name}")

    lines.append(
        "\nWrite a structured Markdown report with these sections, in this order: "
        "a top-level heading naming the dataset's subject, '## The Data', "
        "'## What We Did', '## What We Found' (cite the specific numbers above and "
        "embed each generated chart with ![description](chart_filename.png)), and "
        "'## What To Do With This' with actionable recommendations. "
        "Only reference charts that were actually generated. Be concise and specific."
    )
    return "\n".join(lines)


def build_fallback_narrative(profile: dict, results: dict[str, dict], dataset_name: str) -> str:
    """Deterministic, template-based report used when the LLM call fails or --offline is set."""
    lines = [
        f"# Analysis of {dataset_name}",
        "",
        "## The Data",
        f"This dataset contains {profile['rows']} rows and {profile['cols']} columns "
        f"({profile['duplicate_rows']} duplicate rows detected).",
        "",
        "## What We Did",
    ]
    for name, result in results.items():
        if not result.get("skipped"):
            lines.append(f"- Ran **{name.replace('_', ' ').title()}**")

    lines.append("\n## What We Found")
    for name, result in results.items():
        if result.get("skipped"):
            lines.append(f"- {name.replace('_', ' ').title()}: skipped ({result.get('reason')})")
            continue
        lines.append(f"\n### {name.replace('_', ' ').title()}")
        for k, v in result.items():
            if k == "chart":
                lines.append(f"\n![{name}]({Path(v).name})")
            else:
                lines.append(f"- **{k}**: {v}")

    lines.append(
        "\n## What To Do With This\nReview the charts and figures above for "
        "actionable, dataset-specific insights."
    )
    return "\n".join(lines)


def generate_narrative(client: GeminiClient, profile: dict, results: dict, dataset_name: str, use_cache: bool) -> str:
    prompt = build_narrative_prompt(profile, results, dataset_name)
    try:
        return cached_generate(client, prompt, use_cache=use_cache)
    except AutolysisError:
        log.warning("Narrative generation failed; falling back to a templated report.")
        return build_fallback_narrative(profile, results, dataset_name)


def export_html(readme_path: Path) -> Path | None:
    """Optionally render README.md to a standalone report.html for easy sharing."""
    if not _HAS_MARKDOWN:
        log.warning("The 'markdown' package is not installed; skipping --html export.")
        return None

    html_body = _markdown_lib.markdown(readme_path.read_text(encoding="utf-8"), extensions=["tables"])
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{readme_path.parent.name} — Autolysis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 860px;
          margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #1a1a1a; }}
  h1, h2, h3 {{ color: #111; }}
  img {{ max-width: 100%; border-radius: 6px; margin: 12px 0; }}
  code {{ background: #f2f2f2; padding: 2px 5px; border-radius: 3px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    html_path = readme_path.with_suffix(".html").with_name("report.html")
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #


def read_csv_resilient(csv_path: Path) -> pd.DataFrame:
    """Read a CSV, falling back across common encodings before giving up.

    pandas defaults to UTF-8 and raises UnicodeDecodeError on anything else;
    many real datasets ship as cp1252/latin-1, so a bare read_csv turns a
    routine file into an uncaught traceback.
    """
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise AutolysisError(f"'{csv_path.name}' is empty or malformed: {exc}") from exc
        except pd.errors.ParserError as exc:
            raise AutolysisError(f"'{csv_path.name}' could not be parsed as CSV: {exc}") from exc

        if encoding != "utf-8":
            log.warning("%s is not valid UTF-8; decoded it as '%s'.", csv_path.name, encoding)
        return df

    raise AutolysisError(
        f"'{csv_path.name}' could not be decoded with any of: {', '.join(CSV_ENCODINGS)}."
    )


def run_pipeline(csv_path: Path, config: Config) -> Path:
    if not csv_path.exists():
        raise AutolysisError(f"Input file not found: {csv_path}")

    df = read_csv_resilient(csv_path)

    if df.empty:
        raise AutolysisError(f"'{csv_path.name}' contains zero rows; nothing to analyze.")

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Profiling %s (%d rows x %d cols)...", csv_path.name, len(df), len(df.columns))
    profile = profile_dataset(df)

    client: GeminiClient | None = None
    if config.offline:
        log.info("Running in --offline mode: skipping LLM routing, using default analyses.")
        selected = ["correlation", "outliers"]
    else:
        client = GeminiClient(config.api_key, config.model)
        routing_prompt = build_routing_prompt(profile, config.max_analyses)
        try:
            raw = cached_generate(client, routing_prompt, json_mode=True, use_cache=config.use_cache)
            selected = parse_routing_response(raw, config.max_analyses)
            log.info("LLM selected analyses: %s", ", ".join(selected))
        except AutolysisError as exc:
            log.warning("Routing call failed (%s); defaulting to correlation + outliers.", exc)
            selected = ["correlation", "outliers"]

    results: dict[str, dict] = {}
    for name in selected:
        func = ANALYSIS_FUNCS[name]
        log.info("Running analysis: %s", name)
        try:
            results[name] = func(df, out_dir)
        except Exception as exc:  # noqa: BLE001 - any routine failure must not crash the pipeline
            log.exception("Analysis '%s' raised an unexpected error; skipping.", name)
            results[name] = {"skipped": True, "reason": str(exc)}

    log.info("Generating narrative report...")
    if config.offline or client is None:
        narrative = build_fallback_narrative(profile, results, csv_path.stem)
    else:
        narrative = generate_narrative(client, profile, results, csv_path.stem, config.use_cache)

    readme_path = out_dir / "README.md"
    readme_path.write_text(narrative, encoding="utf-8")
    log.info("Wrote %s", readme_path)

    if config.html:
        html_path = export_html(readme_path)
        if html_path:
            log.info("Wrote %s", html_path)

    return out_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autolysis",
        description="Intelligent, LLM-orchestrated CSV analysis engine.",
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to analyze.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: ./<csv-stem>_output)",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help=f"Gemini model to use (default: env GEMINI_MODEL or '{DEFAULT_MODEL}')",
    )
    parser.add_argument(
        "--max-analyses", type=int, default=3,
        help="Maximum number of analyses the LLM may select (default: 3)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip all LLM calls; run default analyses and write a templated report.",
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Cache LLM responses on disk (.autolysis_cache/) to avoid repeat API calls.",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Additionally render the report as a standalone report.html.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (debug) logging.")
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if _HAS_RICH:
        logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(show_path=False)])
    else:
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        config = load_config(args)
    except AuthenticationError as exc:
        log.error(str(exc))
        return 1

    config.output_dir = args.output_dir or Path(f"{args.csv_path.stem}_output")

    start = time.time()
    try:
        out_dir = run_pipeline(args.csv_path, config)
    except AutolysisError as exc:
        log.error(str(exc))
        return 1

    elapsed = time.time() - start
    log.info("Done in %.1fs. Report written to %s", elapsed, out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
