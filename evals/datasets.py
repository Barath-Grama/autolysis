"""Synthetic datasets with deliberately varied shapes, for the routing eval.

Each generator targets a different corner of the space the router has to
reason about: numeric-only, time-indexed, genuinely clustered, categorical,
and several degenerate cases where most analyses have nothing to say. The
point is a spread where no single fixed pipeline is right everywhere — if
every dataset suited every analysis, "run all five" would win by default and
the eval would prove nothing.

Everything is seeded, so the eval is reproducible run to run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

SEED = 20260801


def wide_numeric_correlated() -> pd.DataFrame:
    """Six numeric columns, strongly correlated, unimodal. No time, no labels."""
    rng = np.random.default_rng(SEED)
    n = 240
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "revenue": base * 1000 + rng.normal(0, 120, n) + 5000,
            "spend": base * 640 + rng.normal(0, 110, n) + 2400,
            "sessions": base * 45 + rng.normal(0, 22, n) + 900,
            "bounce_rate": -base * 0.04 + rng.normal(0, 0.02, n) + 0.42,
            "latency_ms": rng.normal(220, 40, n),
            "errors": rng.normal(12, 4, n),
        }
    )


def time_indexed_trend() -> pd.DataFrame:
    """A real date column with a strong trend, plus a correlated companion."""
    rng = np.random.default_rng(SEED + 1)
    n = 180
    dates = pd.date_range("2021-01-01", periods=n, freq="W")
    drift = np.linspace(0, 40, n)
    return pd.DataFrame(
        {
            "date": dates,
            "active_users": drift * 120 + rng.normal(0, 300, n) + 8000,
            "support_tickets": drift * 3 + rng.normal(0, 12, n) + 140,
        }
    )


def three_clusters() -> pd.DataFrame:
    """Two numeric columns holding three well-separated blobs."""
    rng = np.random.default_rng(SEED + 2)
    centers = [(0.0, 0.0), (12.0, 11.0), (1.0, 13.0)]
    parts = [rng.normal(c, 0.9, (80, 2)) for c in centers]
    points = np.vstack(parts)
    return pd.DataFrame(
        {
            "feature_x": points[:, 0],
            "feature_y": points[:, 1],
            "noise": rng.normal(0, 1, len(points)),
        }
    )


def categorical_survey() -> pd.DataFrame:
    """One numeric measure against several low-cardinality labels."""
    rng = np.random.default_rng(SEED + 3)
    n = 300
    return pd.DataFrame(
        {
            "satisfaction": rng.normal(7.1, 1.6, n).clip(1, 10),
            "region": rng.choice(["North", "South", "East", "West"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "plan": rng.choice(["free", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
            "channel": rng.choice(["web", "ios", "android"], n),
        }
    )


def unique_identifiers() -> pd.DataFrame:
    """A high-cardinality key column: a frequency chart of it is all 1s."""
    rng = np.random.default_rng(SEED + 4)
    n = 200
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(n)],
            "amount": rng.lognormal(3.2, 0.6, n),
            "items": rng.integers(1, 9, n),
        }
    )


def constant_columns() -> pd.DataFrame:
    """Numeric columns with no variance — almost nothing is computable."""
    n = 150
    return pd.DataFrame(
        {
            "flag_a": np.ones(n),
            "flag_b": np.full(n, 3.0),
            "label": ["same"] * n,
        }
    )


def single_numeric_column() -> pd.DataFrame:
    """One usable measure; anything needing two numeric columns must decline."""
    rng = np.random.default_rng(SEED + 5)
    n = 220
    return pd.DataFrame(
        {
            "score": rng.normal(64, 14, n),
            "note": rng.choice(["ok", "flagged"], n, p=[0.9, 0.1]),
        }
    )


def panel_like_happiness() -> pd.DataFrame:
    """Country-year panel: a time column, labels, and several measures."""
    rng = np.random.default_rng(SEED + 6)
    countries = [f"Country {chr(65 + i)}" for i in range(18)]
    years = list(range(2015, 2024))
    rows = []
    for country in countries:
        level = rng.normal(0, 1)
        for year in years:
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "wellbeing": 5.6 + level * 0.9 + rng.normal(0, 0.25),
                    "log_gdp": 9.2 + level * 1.1 + rng.normal(0, 0.2),
                    "life_expectancy": 65 + level * 6 + (year - 2015) * 0.3 + rng.normal(0, 1.4),
                }
            )
    return pd.DataFrame(rows)


def bimodal_with_time() -> pd.DataFrame:
    """Both a genuine clustering structure and a genuine trend."""
    rng = np.random.default_rng(SEED + 7)
    n = 260
    group = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=n, freq="D"),
            "throughput": group * 55 + rng.normal(20, 5, n),
            "queue_depth": group * 40 + rng.normal(15, 4, n),
        }
    )


DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "wide_numeric_correlated": wide_numeric_correlated,
    "time_indexed_trend": time_indexed_trend,
    "three_clusters": three_clusters,
    "categorical_survey": categorical_survey,
    "unique_identifiers": unique_identifiers,
    "constant_columns": constant_columns,
    "single_numeric_column": single_numeric_column,
    "panel_like_happiness": panel_like_happiness,
    "bimodal_with_time": bimodal_with_time,
}
