"""
Unit tests for autolysis.py

Covers:
  - IQR outlier boundary correctness (equivalence partitioning)
  - K-Means elbow-method k-selection
  - LLM routing response parsing (valid JSON, invalid/unknown keys, malformed text)
  - Chart file creation for each analysis routine
  - End-to-end README.md generation (mocked LLM)
  - Graceful degradation (single numeric column, empty CSV, missing API key)

All Gemini API calls are mocked — no network access or API key is required
to run this suite.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autolysis

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A 50-row, 3-column synthetic DataFrame with numeric + categorical data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "score": rng.normal(50, 10, 50),
            "count": rng.integers(1, 100, 50),
            "category": rng.choice(["alpha", "beta", "gamma"], 50),
        }
    )


@pytest.fixture
def out_dir(tmp_path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# --------------------------------------------------------------------------- #
# IQR outlier boundary tests
# --------------------------------------------------------------------------- #


def test_iqr_boundary_low():
    """A value exactly at Q1 - 1.5*IQR must NOT be flagged as an outlier."""
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    reference = pd.Series(values, dtype=float)
    lower, upper = autolysis.iqr_bounds(reference)

    # Test the flagging function in isolation against the *fixed* reference
    # fences, so appending the boundary value cannot itself shift the fence.
    probe = pd.Series([lower])
    mask = autolysis.flag_iqr_outliers(probe, lower, upper)
    assert mask.iloc[0] == False  # noqa: E712 - exact boundary is NOT an outlier


def test_iqr_boundary_below():
    """A value one unit below the lower fence MUST be flagged as an outlier."""
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    reference = pd.Series(values, dtype=float)
    lower, upper = autolysis.iqr_bounds(reference)

    probe = pd.Series([lower - 1])
    mask = autolysis.flag_iqr_outliers(probe, lower, upper)
    assert mask.iloc[0] == True  # noqa: E712 - one unit below fence IS an outlier


def test_outliers_counts_and_chart(synthetic_df, out_dir):
    df = synthetic_df.copy()
    df.loc[len(df)] = [1000, 1, "alpha"]  # inject an obvious outlier
    result = autolysis.run_outliers(df, out_dir)
    assert not result.get("skipped")
    assert result["outlier_counts"]["score"] >= 1
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


# --------------------------------------------------------------------------- #
# K-Means elbow selection
# --------------------------------------------------------------------------- #


def _blobs(n_clusters: int, per_cluster: int = 60, spread: float = 0.25) -> np.ndarray:
    """Well-separated 2-D gaussian blobs with a known cluster count."""
    rng = np.random.default_rng(7)
    centers = np.array([[0, 0], [10, 10], [0, 10], [10, 0], [5, 20]])[:n_clusters]
    return np.vstack([rng.normal(c, spread, (per_cluster, 2)) for c in centers])


def test_silhouette_recovers_two_clusters():
    """The old elbow scored second differences and could never return k=2."""
    best_k, scores = autolysis.select_k_by_silhouette(_blobs(2), range(2, 9))
    assert best_k == 2
    assert set(scores) == set(range(2, 9))


def test_silhouette_recovers_endpoint_k():
    """...nor the top of the range. Both endpoints must now be reachable."""
    best_k, _ = autolysis.select_k_by_silhouette(_blobs(5), range(2, 6))
    assert best_k == 5


def test_silhouette_recovers_three_clusters():
    best_k, _ = autolysis.select_k_by_silhouette(_blobs(3), range(2, 9))
    assert best_k == 3


def test_clustering_end_to_end(synthetic_df, out_dir):
    result = autolysis.run_clustering(synthetic_df, out_dir)
    assert not result.get("skipped")
    assert 2 <= result["k"] <= autolysis.MAX_CLUSTERS
    assert -1.0 <= result["silhouette"] <= 1.0
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


# --------------------------------------------------------------------------- #
# Clustering axis selection must not depend on units
# --------------------------------------------------------------------------- #


def _bimodal(rng, n=200, sep=8.0, scale=1.0):
    """Two separated lumps — the shape K-Means is meant to find."""
    half = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(sep, 1, n - n // 2)])
    return half * scale


def test_structured_columns_outrank_unimodal_noise():
    """A bimodal column is a better clustering axis than a plain gaussian."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "bimodal_a": _bimodal(rng),
            "bimodal_b": _bimodal(rng),
            "gaussian_noise": rng.normal(0, 1, 200),
        }
    )
    picked = set(autolysis.clustering_axis_scores(df).nlargest(2).index)
    assert picked == {"bimodal_a", "bimodal_b"}


def test_axis_selection_is_scale_free():
    """Changing a column's units must not change which axes get picked.

    Note what is deliberately *not* asserted: that a narrow gaussian loses to
    a wide one. After standardisation those are the same data, so preferring
    either would itself be a unit artefact — exactly the bug being fixed.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "salary_usd": _bimodal(rng, scale=20_000.0),
            "rating_frac": _bimodal(rng, scale=0.01),
            "gaussian_noise": rng.normal(0, 1, 200),
        }
    )
    picked = set(autolysis.clustering_axis_scores(df).nlargest(2).index)
    assert picked == {"salary_usd", "rating_frac"}

    # Same data, salary in thousands. Raw variance would reshuffle the ranking
    # by six orders of magnitude; a scale-free score must not move at all.
    rescaled = df.assign(salary_usd=df["salary_usd"] / 1000.0)
    assert set(autolysis.clustering_axis_scores(rescaled).nlargest(2).index) == picked
    assert autolysis.clustering_axis_scores(rescaled)["salary_usd"] == pytest.approx(
        autolysis.clustering_axis_scores(df)["salary_usd"]
    )


def test_constant_columns_drop_out_of_axis_scores():
    df = pd.DataFrame({"varies": [1.0, 2.0, 3.0, 8.0], "constant": [7.0] * 4})
    scores = autolysis.clustering_axis_scores(df)
    assert not np.isnan(scores["varies"])
    assert np.isnan(scores["constant"])


def test_clustering_keeps_columns_with_a_single_nan(out_dir):
    """One missing cell must not disqualify an entire candidate axis."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": _bimodal(rng, n=60),
            "b": _bimodal(rng, n=60),
            "gaussian_noise": rng.normal(0, 1, 60),
        }
    )
    df.loc[0, "a"] = np.nan  # a single hole in an otherwise strong column

    result = autolysis.run_clustering(df, out_dir)
    assert not result.get("skipped")
    assert {result["x_col"], result["y_col"]} == {"a", "b"}


# --------------------------------------------------------------------------- #
# LLM routing response parsing
# --------------------------------------------------------------------------- #


def test_routing_json_parse():
    raw = '["correlation", "outliers"]'
    assert autolysis.parse_routing_response(raw) == ["correlation", "outliers"]


def test_routing_json_parse_with_code_fence():
    raw = '```json\n["clustering", "time_series"]\n```'
    assert autolysis.parse_routing_response(raw) == ["clustering", "time_series"]


def test_routing_invalid_key():
    """Unknown analysis identifiers returned by a non-compliant LLM are dropped."""
    raw = '["correlation", "sentiment_analysis", "outliers"]'
    result = autolysis.parse_routing_response(raw)
    assert result == ["correlation", "outliers"]
    assert "sentiment_analysis" not in result


def test_routing_malformed_response_falls_back():
    raw = "I think you should look at correlation and outliers."
    result = autolysis.parse_routing_response(raw)
    assert result == ["correlation", "outliers"]


def test_routing_respects_max_analyses():
    raw = '["correlation", "outliers", "clustering", "time_series"]'
    result = autolysis.parse_routing_response(raw, max_analyses=2)
    assert result == ["correlation", "outliers"]


# --------------------------------------------------------------------------- #
# Chart file creation across all routines
# --------------------------------------------------------------------------- #


def test_chart_file_created_correlation(synthetic_df, out_dir):
    result = autolysis.run_correlation(synthetic_df, out_dir)
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


def test_chart_file_created_category(synthetic_df, out_dir):
    result = autolysis.run_category_analysis(synthetic_df, out_dir)
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


def test_chart_file_created_time_series(out_dir):
    df = pd.DataFrame(
        {
            "year": list(range(2000, 2020)),
            "value": np.linspace(10, 50, 20) + np.random.default_rng(1).normal(0, 1, 20),
        }
    )
    result = autolysis.run_time_series(df, out_dir)
    assert not result.get("skipped")
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


# --------------------------------------------------------------------------- #
# End-to-end README generation (mocked LLM)
# --------------------------------------------------------------------------- #


def test_readme_written_full_pipeline(tmp_path, synthetic_df, monkeypatch):
    csv_path = tmp_path / "synthetic.csv"
    synthetic_df.to_csv(csv_path, index=False)

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    with patch.object(autolysis.GeminiClient, "generate") as mock_generate:
        mock_generate.side_effect = [
            '["correlation", "outliers"]',
            "# Synthetic Dataset Report\n\nThis is a mocked narrative.",
        ]
        config = autolysis.Config(api_key="fake-key-for-testing", output_dir=tmp_path / "out")
        out_dir = autolysis.run_pipeline(csv_path, config)

    readme_path = out_dir / "README.md"
    assert readme_path.exists()
    assert readme_path.read_text().strip() != ""
    assert mock_generate.call_count == 2


def test_offline_mode_produces_templated_report(tmp_path, synthetic_df):
    csv_path = tmp_path / "synthetic.csv"
    synthetic_df.to_csv(csv_path, index=False)

    config = autolysis.Config(api_key="", offline=True, output_dir=tmp_path / "out")
    out_dir = autolysis.run_pipeline(csv_path, config)

    readme_path = out_dir / "README.md"
    assert readme_path.exists()
    assert "# Analysis of synthetic" in readme_path.read_text()


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


def test_single_numeric_col_clustering_skipped(out_dir):
    df = pd.DataFrame({"only_numeric": range(20), "label": ["x"] * 20})
    result = autolysis.run_clustering(df, out_dir)
    assert result["skipped"] is True
    assert "Fewer than 2 numeric" in result["reason"]


def test_single_numeric_col_correlation_skipped(out_dir):
    df = pd.DataFrame({"only_numeric": range(20), "label": ["x"] * 20})
    result = autolysis.run_correlation(df, out_dir)
    assert result["skipped"] is True


def test_api_key_missing_raises_authentication_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    parser = autolysis.build_arg_parser()
    args = parser.parse_args(["dummy.csv"])  # offline=False by default

    with pytest.raises(autolysis.AuthenticationError):
        autolysis.load_config(args)


def test_empty_csv_raises_clear_error(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame({"a": [], "b": []}).to_csv(csv_path, index=False)

    config = autolysis.Config(api_key="", offline=True, output_dir=tmp_path / "out")
    with pytest.raises(autolysis.AutolysisError, match="zero rows"):
        autolysis.run_pipeline(csv_path, config)


def test_missing_file_raises_clear_error(tmp_path):
    config = autolysis.Config(api_key="", offline=True, output_dir=tmp_path / "out")
    with pytest.raises(autolysis.AutolysisError, match="not found"):
        autolysis.run_pipeline(tmp_path / "does_not_exist.csv", config)


def test_no_temporal_column_time_series_skipped(out_dir):
    df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    result = autolysis.run_time_series(df, out_dir)
    assert result["skipped"] is True
    assert "temporal" in result["reason"].lower()


def test_no_categorical_columns_skipped(out_dir):
    df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    result = autolysis.run_category_analysis(df, out_dir)
    assert result["skipped"] is True


# --------------------------------------------------------------------------- #
# Gemini client retry / auth behaviour
# --------------------------------------------------------------------------- #


def test_gemini_client_retries_then_raises_autolysis_error():
    client = autolysis.GeminiClient(api_key="fake", max_retries=2, base_delay=0.01)
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = httpx.TransportError("network down")

    with patch("httpx.Client", return_value=mock_client), pytest.raises(autolysis.AutolysisError):
        client.generate("hello")


def test_extract_text_raises_on_bad_shape():
    with pytest.raises(autolysis.AutolysisError):
        autolysis._extract_text({"unexpected": "shape"})


def test_routing_prompt_contains_vocab():
    profile = {
        "rows": 10,
        "cols": 2,
        "duplicate_rows": 0,
        "columns": {
            "x": {"dtype": "float64", "null_pct": 0.0, "mean": 1.0, "std": 0.5, "min": 0, "max": 2},
            "y": {"dtype": "object", "null_pct": 0.0, "n_unique": 2, "top_frequencies": [5, 5]},
        },
    }
    prompt = autolysis.build_routing_prompt(profile)
    for term in autolysis.ANALYSIS_VOCAB:
        assert term in prompt


# --------------------------------------------------------------------------- #
# Secret handling — the API key must never reach a log line or error message
# --------------------------------------------------------------------------- #


def _mock_httpx_client(post_impl):
    """Build a patched httpx.Client whose .post() runs post_impl."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post = post_impl
    return mock_client


def test_api_key_sent_as_header_not_in_url():
    """The key belongs in x-goog-api-key; httpx puts URLs in its exception text."""
    client = autolysis.GeminiClient(api_key="SECRET123", max_retries=1, base_delay=0.01)
    seen = {}

    def capture(url, json=None, headers=None):
        seen["url"] = url
        seen["headers"] = headers or {}
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
        )

    with patch("httpx.Client", return_value=_mock_httpx_client(capture)):
        assert client.generate("hello") == "ok"

    assert "SECRET123" not in seen["url"]
    assert seen["headers"].get("x-goog-api-key") == "SECRET123"


def test_api_key_never_appears_in_raised_error():
    """A failing call must not surface the key in the message the user sees."""
    client = autolysis.GeminiClient(api_key="SECRET123", max_retries=2, base_delay=0.01)

    def server_error(url, json=None, headers=None):
        req = httpx.Request("POST", url)
        return httpx.Response(500, request=req, text="upstream boom")

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(server_error)),
        pytest.raises(autolysis.AutolysisError) as excinfo,
    ):
        client.generate("hello")

    # The URL no longer carries the key at all, so there is nothing left to
    # scrub — absence is the assertion that matters, not the redaction marker.
    assert "SECRET123" not in str(excinfo.value)


def test_redact_secret_scrubs_an_embedded_key():
    """Backstop for third-party error text we don't control."""
    leaked = "connect failed for url 'https://api/x?key=SECRET123'"
    scrubbed = autolysis.redact_secret(leaked, "SECRET123")
    assert "SECRET123" not in scrubbed
    assert "***REDACTED***" in scrubbed


def test_redact_secret_is_a_noop_for_empty_key():
    assert autolysis.redact_secret("nothing to hide", "") == "nothing to hide"


# --------------------------------------------------------------------------- #
# CSV encoding resilience
# --------------------------------------------------------------------------- #


def test_latin1_csv_is_read_not_crashed(tmp_path):
    """Non-UTF-8 input must degrade to a fallback encoding, not raise."""
    csv_path = tmp_path / "latin.csv"
    csv_path.write_bytes("name,score\nCaf\xe9 Br\xfblot,5\nNa\xefve,3\n".encode("latin-1"))

    df = autolysis.read_csv_resilient(csv_path)
    assert len(df) == 2
    assert list(df.columns) == ["name", "score"]


def test_latin1_csv_runs_full_pipeline(tmp_path):
    csv_path = tmp_path / "latin.csv"
    rows = "\n".join(f"Caf\xe9 {i},{i * 3},{i % 4}" for i in range(30))
    csv_path.write_bytes(f"name,score,bucket\n{rows}\n".encode("latin-1"))

    config = autolysis.Config(api_key="", offline=True, output_dir=tmp_path / "out")
    out_dir = autolysis.run_pipeline(csv_path, config)
    assert (out_dir / "README.md").exists()


def test_utf8_csv_still_preferred(tmp_path):
    """UTF-8 must win outright — the fallback chain must not mangle good input."""
    csv_path = tmp_path / "utf8.csv"
    csv_path.write_text("name,score\nCafé Brûlot,5\n", encoding="utf-8")

    df = autolysis.read_csv_resilient(csv_path)
    assert df.loc[0, "name"] == "Café Brûlot"


def test_malformed_csv_raises_clean_error(tmp_path):
    """A ParserError must surface as AutolysisError, not an uncaught traceback."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text('a,b\n"unterminated,1\n2,3,4,5,6\n', encoding="utf-8")

    with pytest.raises(autolysis.AutolysisError):
        autolysis.read_csv_resilient(csv_path)


# --------------------------------------------------------------------------- #
# --max-analyses must reach the model, not just truncate its answer
# --------------------------------------------------------------------------- #


def test_max_analyses_reaches_the_routing_prompt():
    profile = {
        "rows": 10,
        "cols": 1,
        "duplicate_rows": 0,
        "columns": {"x": {"dtype": "float64", "null_pct": 0.0, "mean": 1.0, "std": 0.5, "min": 0, "max": 2}},
    }
    assert "Select up to 5 analyses" in autolysis.build_routing_prompt(profile, max_analyses=5)
    assert "Select up to 2 analyses" in autolysis.build_routing_prompt(profile, max_analyses=2)


def test_max_analyses_flows_from_cli_to_prompt(tmp_path, synthetic_df):
    """End-to-end: the CLI flag must change the text the model actually sees."""
    csv_path = tmp_path / "synthetic.csv"
    synthetic_df.to_csv(csv_path, index=False)

    with patch.object(autolysis.GeminiClient, "generate") as mock_generate:
        mock_generate.side_effect = ['["correlation"]', "# Report\n\nmocked."]
        config = autolysis.Config(
            api_key="fake-key", max_analyses=5, output_dir=tmp_path / "out"
        )
        autolysis.run_pipeline(csv_path, config)

    routing_prompt = mock_generate.call_args_list[0].args[0]
    assert "Select up to 5 analyses" in routing_prompt


# --------------------------------------------------------------------------- #
# Nothing identifying may reach the model
# --------------------------------------------------------------------------- #


@pytest.fixture
def pii_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_email": ["ana@x.com"] * 3 + ["bob@y.com"] * 2 + ["carl@z.com"],
            "diagnosis": ["hypertension"] * 4 + ["diabetes"] * 2,
            "age": [41, 52, 63, 34, 45, 56],
        }
    )


def test_profile_excludes_raw_categorical_values(pii_df):
    profile = autolysis.profile_dataset(pii_df)
    blob = json.dumps(profile)
    for value in ("ana@x.com", "bob@y.com", "carl@z.com", "hypertension", "diabetes"):
        assert value not in blob


def test_routing_prompt_excludes_raw_categorical_values(pii_df):
    """The prompt is what actually crosses the network — assert on that."""
    prompt = autolysis.build_routing_prompt(autolysis.profile_dataset(pii_df))
    for value in ("ana@x.com", "bob@y.com", "carl@z.com", "hypertension", "diabetes"):
        assert value not in prompt


def test_narrative_prompt_excludes_raw_categorical_values(pii_df):
    profile = autolysis.profile_dataset(pii_df)
    prompt = autolysis.build_narrative_prompt(profile, {}, "clinic")
    for value in ("ana@x.com", "bob@y.com", "carl@z.com"):
        assert value not in prompt


def test_profile_keeps_the_signal_the_router_needs(pii_df):
    """Dropping the values must not drop cardinality or skew."""
    cols = autolysis.profile_dataset(pii_df)["columns"]
    assert cols["patient_email"]["n_unique"] == 3
    assert cols["patient_email"]["top_frequencies"] == [3, 2, 1]
    assert cols["diagnosis"]["n_unique"] == 2


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [400, 404, 422])
def test_non_retryable_status_fails_immediately(status):
    """A deterministic 4xx must not be retried four times with backoff."""
    client = autolysis.GeminiClient(api_key="k", max_retries=4, base_delay=0.01)
    calls = []

    def responder(url, json=None, headers=None):
        calls.append(1)
        return httpx.Response(status, request=httpx.Request("POST", url), text="nope")

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        pytest.raises(autolysis.AutolysisError, match="not retryable"),
    ):
        client.generate("hello")

    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_is_retried(status):
    client = autolysis.GeminiClient(api_key="k", max_retries=3, base_delay=0.01)
    calls = []

    def responder(url, json=None, headers=None):
        calls.append(1)
        return httpx.Response(status, request=httpx.Request("POST", url), text="later")

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        pytest.raises(autolysis.AutolysisError),
    ):
        client.generate("hello")

    assert len(calls) == 3


def test_retry_after_header_is_honoured():
    client = autolysis.GeminiClient(api_key="k", max_retries=2, base_delay=99.0)

    def responder(url, json=None, headers=None):
        return httpx.Response(
            429, request=httpx.Request("POST", url), headers={"Retry-After": "0.01"}
        )

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        patch("time.sleep") as mock_sleep,
        pytest.raises(autolysis.AutolysisError),
    ):
        client.generate("hello")

    # The server's 0.01s wins over our 99s backoff guess.
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(0.01)


def test_auth_error_still_short_circuits():
    client = autolysis.GeminiClient(api_key="k", max_retries=4, base_delay=0.01)

    def responder(url, json=None, headers=None):
        return httpx.Response(403, request=httpx.Request("POST", url))

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        pytest.raises(autolysis.AuthenticationError),
    ):
        client.generate("hello")


# --------------------------------------------------------------------------- #
# Time-series regression against real elapsed time
# --------------------------------------------------------------------------- #


def test_slope_is_per_time_unit_not_per_row(out_dir):
    """Irregular spacing must not be flattened into row positions."""
    # +2.0 per year, but the observations are unevenly spaced.
    years = [2000, 2001, 2002, 2010, 2020]
    df = pd.DataFrame({"year": years, "value": [2.0 * y for y in years]})

    result = autolysis.run_time_series(df, out_dir)
    assert not result.get("skipped")
    assert result["slope"] == pytest.approx(2.0, abs=1e-6)
    assert "per unit of year" in result["slope_unit"]


def test_slope_handles_datetime_index(out_dir):
    dates = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-02-10"])
    df = pd.DataFrame({"date": dates, "value": [0.0, 10.0, 40.0]})  # +1.0/day

    result = autolysis.run_time_series(df, out_dir)
    assert not result.get("skipped")
    assert result["slope"] == pytest.approx(1.0, abs=1e-6)
    assert result["slope_unit"] == "value per day"


def test_string_dates_are_ordered_chronologically(out_dir):
    """String dates must be coerced, not sorted lexicographically."""
    df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-10", "2020-01-01"],
            "value": [1.0, 9.0, 0.0],
        }
    )
    result = autolysis.run_time_series(df, out_dir)
    assert not result.get("skipped")
    assert result["direction"] == "increasing"
    assert result["slope"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# HTML export: no-op path, and sanitisation of model-generated markup
# --------------------------------------------------------------------------- #


def test_export_html_writes_report_next_to_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")

    html_path = autolysis.export_html(readme)
    assert html_path == tmp_path / "report.html"
    body = html_path.read_text(encoding="utf-8")
    assert "<h1>Title</h1>" in body
    assert "<table>" in body  # the tables extension is actually wired up


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        '<a href="javascript:alert(1)">click</a>',
        "<iframe src='https://evil.example'></iframe>",
        "<object data='x'></object>",
        "<svg/onload=alert(1)>",
        '<div style="position:fixed;top:0">overlay</div>',
    ],
)
def test_sanitizer_strips_active_content(payload):
    """The narrative is model-generated and often shared; it must not execute."""
    out = autolysis.sanitize_html(payload).lower()
    for banned in ("<script", "<iframe", "<object", "<svg", "onerror", "onload", "javascript:", "style="):
        assert banned not in out


def test_sanitizer_keeps_report_markup():
    """Sanitising must not gut the actual report."""
    markup = (
        '<h2>Findings</h2><p><strong>r</strong> = 0.36</p>'
        '<img src="correlation_heatmap.png" alt="correlation">'
        '<a href="https://example.com" title="src">source</a>'
        "<table><tr><td>1</td></tr></table><pre><code>x = 1</code></pre>"
    )
    out = autolysis.sanitize_html(markup)
    assert "<h2>Findings</h2>" in out
    assert "<strong>r</strong>" in out
    assert 'src="correlation_heatmap.png"' in out  # relative chart links survive
    assert 'href="https://example.com"' in out
    assert "<table>" in out and "<code>" in out


def test_export_html_sanitizes_end_to_end(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Report\n\n<script>alert('xss')</script>\n\nText.\n", encoding="utf-8")

    body = autolysis.export_html(readme).read_text(encoding="utf-8")
    assert "<script>" not in body
    assert "alert" not in body  # the script *body* is dropped, not just the tag
    assert "<h1>Report</h1>" in body and "<p>Text.</p>" in body


def test_export_html_returns_none_without_markdown(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n", encoding="utf-8")
    monkeypatch.setattr(autolysis, "_HAS_MARKDOWN", False)
    assert autolysis.export_html(readme) is None


# --------------------------------------------------------------------------- #
# Response cache
# --------------------------------------------------------------------------- #


def _client_returning(text: str):
    client = autolysis.GeminiClient(api_key="k")
    client.generate = MagicMock(return_value=text)
    return client


def test_cache_disabled_by_default_calls_through(tmp_path):
    client = _client_returning("fresh")
    assert autolysis.cached_generate(client, "p", cache_dir=tmp_path) == "fresh"
    assert autolysis.cached_generate(client, "p", cache_dir=tmp_path) == "fresh"
    assert client.generate.call_count == 2


def test_cache_hit_avoids_second_call(tmp_path):
    client = _client_returning("once")
    autolysis.cached_generate(client, "p", use_cache=True, cache_dir=tmp_path)
    autolysis.cached_generate(client, "p", use_cache=True, cache_dir=tmp_path)
    assert client.generate.call_count == 1


def test_cache_separates_json_mode(tmp_path):
    """json_mode changes the response format, so it must change the key."""
    client = _client_returning("x")
    autolysis.cached_generate(client, "p", json_mode=True, use_cache=True, cache_dir=tmp_path)
    autolysis.cached_generate(client, "p", json_mode=False, use_cache=True, cache_dir=tmp_path)
    assert client.generate.call_count == 2
    assert autolysis._cache_key("p", "m", True) != autolysis._cache_key("p", "m", False)


def test_cache_entry_expires(tmp_path):
    client = _client_returning("stale")
    autolysis.cached_generate(client, "p", use_cache=True, cache_dir=tmp_path)

    entry = next(tmp_path.glob("*.txt"))
    old = time.time() - (autolysis.CACHE_TTL_SECONDS + 60)
    os.utime(entry, (old, old))

    autolysis.cached_generate(client, "p", use_cache=True, cache_dir=tmp_path)
    assert client.generate.call_count == 2


def test_cache_dir_is_created_on_demand(tmp_path):
    nested = tmp_path / "deep" / "cache"
    autolysis.cached_generate(_client_returning("v"), "p", use_cache=True, cache_dir=nested)
    assert nested.is_dir()


# --------------------------------------------------------------------------- #
# Profiler and time-column heuristic
# --------------------------------------------------------------------------- #


def test_profile_reports_shape_and_nulls():
    df = pd.DataFrame({"n": [1.0, 2.0, None, 4.0], "c": ["a", "a", "b", None]})
    profile = autolysis.profile_dataset(df)
    assert profile["rows"] == 4 and profile["cols"] == 2
    assert profile["columns"]["n"]["null_pct"] == pytest.approx(25.0)
    assert profile["columns"]["n"]["mean"] == pytest.approx(7 / 3)
    assert profile["columns"]["c"]["n_unique"] == 2


def test_profile_handles_all_null_numeric_column():
    # dtype must be forced: [None, None] infers as object, which would take the
    # categorical branch and never exercise the all-null numeric path.
    df = pd.DataFrame({"empty": pd.Series([None, None], dtype="float64")})
    profile = autolysis.profile_dataset(df)
    assert profile["columns"]["empty"]["mean"] is None
    assert profile["columns"]["empty"]["null_pct"] == pytest.approx(100.0)


def test_profile_counts_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert autolysis.profile_dataset(df)["duplicate_rows"] == 1


@pytest.mark.parametrize("column", ["year", "date", "Order Date", "month", "quarter"])
def test_find_time_column_matches_keywords(column):
    df = pd.DataFrame({column: range(2000, 2010), "value": range(10)})
    assert autolysis.find_time_column(df) == column


def test_find_time_column_returns_none_without_temporal_data():
    assert autolysis.find_time_column(pd.DataFrame({"a": range(5), "b": range(5)})) is None


def test_find_time_column_accepts_string_dates():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "v": [1, 2]})
    assert autolysis.find_time_column(df) == "date"


# --------------------------------------------------------------------------- #
# Offline analysis selection, .env loading, sampling, config plumbing
# --------------------------------------------------------------------------- #


def test_offline_selection_reaches_beyond_correlation_and_outliers():
    """--offline used to hardcode two routines, leaving three unreachable."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "year": list(range(2000, 2060)),
            "value": rng.normal(0, 1, 60),
            "category": rng.choice(["a", "b", "c"], 60),
        }
    )
    selected = autolysis.select_default_analyses(df, max_analyses=5)
    assert "time_series" in selected
    assert "category_analysis" in selected
    assert len(selected) <= 5


def test_offline_selection_respects_max_analyses():
    df = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    assert len(autolysis.select_default_analyses(df, max_analyses=1)) == 1


def test_offline_selection_degrades_on_a_single_column():
    df = pd.DataFrame({"only": range(20)})
    assert autolysis.select_default_analyses(df) == ["outliers"]


def test_offline_pipeline_runs_more_than_two_analyses(tmp_path):
    rng = np.random.default_rng(2)
    csv_path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "year": list(range(2000, 2060)),
            "value": rng.normal(0, 1, 60),
            "category": rng.choice(["a", "b", "c"], 60),
        }
    ).to_csv(csv_path, index=False)

    config = autolysis.Config(api_key="", offline=True, max_analyses=5, output_dir=tmp_path / "out")
    out_dir = autolysis.run_pipeline(csv_path, config)
    charts = {p.name for p in out_dir.glob("*.png")}
    assert len(charts) >= 3


def test_load_dotenv_populates_missing_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nGEMINI_API_KEY=from-dotenv\nexport GEMINI_MODEL='quoted-model'\nBROKEN\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    autolysis.load_dotenv(env)
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    assert os.environ["GEMINI_MODEL"] == "quoted-model"


def test_load_dotenv_never_overrides_a_real_env_var(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")

    autolysis.load_dotenv(env)
    assert os.environ["GEMINI_API_KEY"] == "from-shell"


def test_load_dotenv_is_a_noop_when_absent(tmp_path):
    autolysis.load_dotenv(tmp_path / "nope.env")  # must not raise


def test_sample_for_plot_caps_rows_and_is_deterministic():
    df = pd.DataFrame({"x": range(100)})
    first = autolysis.sample_for_plot(df, cap=10)
    assert len(first) == 10
    assert first.equals(autolysis.sample_for_plot(df, cap=10))
    assert autolysis.sample_for_plot(df, cap=500) is df  # no copy when under cap


def test_top_categories_flag_reaches_the_chart(tmp_path):
    rng = np.random.default_rng(5)
    csv_path = tmp_path / "cats.csv"
    pd.DataFrame(
        {"label": rng.choice([f"c{i}" for i in range(12)], 200), "v": rng.normal(0, 1, 200)}
    ).to_csv(csv_path, index=False)

    args = autolysis.build_arg_parser().parse_args(
        [str(csv_path), "--offline", "--top-categories", "3", "-o", str(tmp_path / "out")]
    )
    config = autolysis.load_config(args)
    assert config.top_n_categories == 3

    result = autolysis.run_category_analysis(
        pd.read_csv(csv_path), tmp_path, top_n=config.top_n_categories
    )
    assert len(result["top_categories"]) == 3


def test_load_config_resolves_output_dir_from_csv_stem(tmp_path):
    args = autolysis.build_arg_parser().parse_args([str(tmp_path / "sales.csv"), "--offline"])
    assert autolysis.load_config(args).output_dir == Path("sales_output")


def test_routing_uses_temperature_zero(tmp_path, synthetic_df):
    """Closed-vocabulary routing should be reproducible; prose need not be."""
    csv_path = tmp_path / "s.csv"
    synthetic_df.to_csv(csv_path, index=False)

    with patch.object(autolysis.GeminiClient, "generate") as mock_generate:
        mock_generate.side_effect = ['["correlation"]', "# R\n\ntext."]
        autolysis.run_pipeline(
            csv_path, autolysis.Config(api_key="k", output_dir=tmp_path / "out")
        )

    assert mock_generate.call_args_list[0].kwargs["temperature"] == 0.0
    assert mock_generate.call_args_list[1].kwargs["temperature"] == pytest.approx(0.4)


def test_extract_text_error_does_not_dump_the_body():
    """A failed parse must not echo an entire API response into the logs."""
    payload = {"promptFeedback": {"blockReason": "SAFETY"}, "junk": "x" * 5000}
    with pytest.raises(autolysis.AutolysisError) as excinfo:
        autolysis._extract_text(payload)

    message = str(excinfo.value)
    assert len(message) < 200
    assert "x" * 100 not in message
    assert "promptFeedback" in message  # the shape is still diagnosable


# --------------------------------------------------------------------------- #
# Templated narrative — the keyless path, and what CI and the README show
# --------------------------------------------------------------------------- #


@pytest.fixture
def rendered_results() -> dict:
    return {
        "correlation": {
            "top_correlations": [{"a": "gdp", "b": "ladder", "r": 0.361}],
            "chart": "correlation_heatmap.png",
        },
        "outliers": {"outlier_counts": {"ladder": 11, "gdp": 0}, "chart": "outliers_boxplot.png"},
        "clustering": {
            "k": 3, "silhouette": 0.4656, "x_col": "gdp", "y_col": "year",
            "cluster_means": {0: {"gdp": 8.5, "year": 2021.55}},
            "chart": "clustering.png",
        },
        "time_series": {
            "time_col": "year", "target_col": "life_exp", "slope": -0.2972,
            "slope_unit": "life_exp per unit of year", "direction": "decreasing",
            "chart": "time_series_trend.png",
        },
        "category_analysis": {
            "column": "country", "top_categories": {"Finland": 9}, "chart": "category_frequency.png",
        },
    }


@pytest.fixture
def sample_profile() -> dict:
    return {
        "rows": 144, "cols": 2, "duplicate_rows": 0,
        "columns": {
            "gdp": {"dtype": "float64", "null_pct": 4.2, "mean": 9.0, "std": 1.0, "min": 7, "max": 11},
            "country": {"dtype": "object", "null_pct": 0.0, "n_unique": 16, "top_frequencies": [9]},
        },
    }


def test_narrative_renders_prose_not_raw_dicts(sample_profile, rendered_results):
    """The templated report is what every keyless run produces; it must read."""
    report = autolysis.build_fallback_narrative(sample_profile, rendered_results, "happiness")

    # No Python repr leaking into the Markdown.
    for artefact in ("{'", "': ", "[{", "dict_", "OrderedDict"):
        assert artefact not in report

    assert "The strongest linear relationship is **gdp** vs **ladder** (r = +0.361)." in report
    assert "| Feature A | Feature B | Pearson r |" in report
    assert "K-Means settled on **k = 3**" in report
    assert "life_exp per unit of year" in report


def test_narrative_embeds_every_chart(sample_profile, rendered_results):
    report = autolysis.build_fallback_narrative(sample_profile, rendered_results, "happiness")
    for result in rendered_results.values():
        assert f"]({result['chart']})" in report


def test_narrative_recommendations_cite_numbers(sample_profile, rendered_results):
    report = autolysis.build_fallback_narrative(sample_profile, rendered_results, "happiness")
    tail = report.split("## What To Do With This")[1]
    assert "review the charts above" not in tail.lower()  # the old generic filler
    assert "r = +0.361" in tail
    assert "11 outlying values" in tail


def test_narrative_reports_skipped_routines(sample_profile):
    results = {"clustering": {"skipped": True, "reason": "Fewer than 2 numeric columns"}}
    report = autolysis.build_fallback_narrative(sample_profile, results, "tiny")
    assert "skipped (Fewer than 2 numeric columns)" in report


def test_narrative_survives_empty_results(sample_profile):
    report = autolysis.build_fallback_narrative(sample_profile, {}, "empty")
    assert "# Analysis of empty" in report
    assert "## What To Do With This" in report


def test_weak_correlation_gets_a_different_recommendation(sample_profile):
    """A 0.05 correlation must not be described as worth a causal look."""
    results = {"correlation": {"top_correlations": [{"a": "x", "b": "y", "r": 0.05}]}}
    report = autolysis.build_fallback_narrative(sample_profile, results, "weak")
    assert "linear structure is weak" in report
    assert "causal look" not in report


@pytest.mark.parametrize("example_dir", ["example", "example-offline"])
def test_committed_example_report_is_current(example_dir):
    """The README embeds these; regenerate them if this fails.

    Chart expectations are read out of the report rather than hardcoded, since
    the Gemini run selects its own analyses and so its chart set legitimately
    differs from the offline one.
    """
    report = Path(__file__).resolve().parent.parent / "docs" / example_dir / "README.md"
    assert report.is_file(), f"docs/{example_dir}/README.md is missing"

    body = report.read_text(encoding="utf-8")
    charts = set(re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", body))
    assert charts, "the committed report embeds no charts"
    for chart in charts:
        assert (report.parent / chart).is_file(), f"{chart} referenced but not committed"

    orphans = {p.name for p in report.parent.glob("*.png")} - charts
    assert not orphans, f"charts committed but unreferenced: {sorted(orphans)}"


def test_committed_example_contains_no_api_key():
    """A generated artefact must never carry the key that produced it."""
    docs = Path(__file__).resolve().parent.parent / "docs"
    for path in docs.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".html"}:
            assert "AIza" not in path.read_text(encoding="utf-8"), f"key-shaped string in {path}"


# --------------------------------------------------------------------------- #
# Narrative prompt must not invite the model to invent statistics
# --------------------------------------------------------------------------- #


def test_narrative_prompt_sends_min_and_max(sample_profile):
    """Omitting them made Gemini derive a range from mean +/- std and state it.

    Observed live: a 2015-2023 year column was reported as "2017 to 2022".
    """
    prompt = autolysis.build_narrative_prompt(sample_profile, {}, "d")
    assert "min=" in prompt and "max=" in prompt


def test_narrative_prompt_forbids_inferred_values(sample_profile):
    prompt = autolysis.build_narrative_prompt(sample_profile, {}, "d")
    assert "Use ONLY the statistics given above" in prompt
    assert "do not derive ranges" in prompt


def test_narrative_prompt_rounds_long_floats(sample_profile):
    """Full float repr wastes tokens on every column, every run."""
    sample_profile["columns"]["gdp"]["std"] = 2.591001102525183
    prompt = autolysis.build_narrative_prompt(sample_profile, {}, "d")
    assert "2.591001102525183" not in prompt
    assert "2.591" in prompt


def test_narrative_prompt_handles_all_null_numeric(sample_profile):
    """An all-null column has None for every statistic; it must not crash."""
    sample_profile["columns"]["empty"] = {
        "dtype": "float64", "null_pct": 100.0,
        "mean": None, "std": None, "min": None, "max": None,
    }
    prompt = autolysis.build_narrative_prompt(sample_profile, {}, "d")
    assert "empty [numeric]: mean=NA" in prompt


# --------------------------------------------------------------------------- #
# Auth failures mid-pipeline (found by pointing the CLI at the live API)
# --------------------------------------------------------------------------- #


def _csv(tmp_path) -> Path:
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": range(30), "b": range(30, 60)}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_mid_pipeline_exits_cleanly(tmp_path, monkeypatch, status):
    """A revoked key used to escape run_pipeline as an uncaught traceback."""
    monkeypatch.setenv("GEMINI_API_KEY", "revoked")

    def responder(url, json=None, headers=None):
        return httpx.Response(status, request=httpx.Request("POST", url))

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        rc = autolysis.main([str(_csv(tmp_path)), "-o", str(tmp_path / "out")])

    assert rc == 1


def test_invalid_key_400_is_recognised_as_an_auth_failure():
    """The live API returns 400 INVALID_ARGUMENT for a bad key, not 401/403."""
    client = autolysis.GeminiClient(api_key="bad", max_retries=1, base_delay=0.01)

    def responder(url, json=None, headers=None):
        return httpx.Response(
            400,
            request=httpx.Request("POST", url),
            json={"error": {"message": "API key not valid. Please pass a valid API key.",
                            "status": "INVALID_ARGUMENT"}},
        )

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        pytest.raises(autolysis.AuthenticationError, match="API key"),
    ):
        client.generate("hello")


def test_ordinary_400_is_not_mistaken_for_an_auth_failure():
    """A malformed-request 400 must stay a plain non-retryable error."""
    client = autolysis.GeminiClient(api_key="fine", max_retries=1, base_delay=0.01)

    def responder(url, json=None, headers=None):
        return httpx.Response(
            400,
            request=httpx.Request("POST", url),
            json={"error": {"message": "Request contains an invalid argument.",
                            "status": "INVALID_ARGUMENT"}},
        )

    with (
        patch("httpx.Client", return_value=_mock_httpx_client(responder)),
        pytest.raises(autolysis.AutolysisError, match="not retryable"),
    ):
        client.generate("hello")


def test_bad_key_does_not_silently_degrade_to_an_offline_report(tmp_path, monkeypatch):
    """Falling back to offline analyses would hide a configuration error."""
    monkeypatch.setenv("GEMINI_API_KEY", "revoked")
    out = tmp_path / "out"

    def responder(url, json=None, headers=None):
        return httpx.Response(403, request=httpx.Request("POST", url))

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        rc = autolysis.main([str(_csv(tmp_path)), "-o", str(out)])

    assert rc == 1
    assert not (out / "README.md").exists(), "a rejected key must not yield a report"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
