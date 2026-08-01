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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autolysis  # noqa: E402


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
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client
        # Force a transport-style error via patching post to raise httpx.TransportError
        import httpx as httpx_mod

        mock_client.post.side_effect = httpx_mod.TransportError("network down")
        with pytest.raises(autolysis.AutolysisError):
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

    with patch("httpx.Client", return_value=_mock_httpx_client(server_error)):
        with pytest.raises(autolysis.AutolysisError) as excinfo:
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

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        with pytest.raises(autolysis.AutolysisError, match="not retryable"):
            client.generate("hello")

    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_is_retried(status):
    client = autolysis.GeminiClient(api_key="k", max_retries=3, base_delay=0.01)
    calls = []

    def responder(url, json=None, headers=None):
        calls.append(1)
        return httpx.Response(status, request=httpx.Request("POST", url), text="later")

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        with pytest.raises(autolysis.AutolysisError):
            client.generate("hello")

    assert len(calls) == 3


def test_retry_after_header_is_honoured():
    client = autolysis.GeminiClient(api_key="k", max_retries=2, base_delay=99.0)

    def responder(url, json=None, headers=None):
        return httpx.Response(
            429, request=httpx.Request("POST", url), headers={"Retry-After": "0.01"}
        )

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(autolysis.AutolysisError):
                client.generate("hello")

    # The server's 0.01s wins over our 99s backoff guess.
    assert mock_sleep.call_args_list[0].args[0] == pytest.approx(0.01)


def test_auth_error_still_short_circuits():
    client = autolysis.GeminiClient(api_key="k", max_retries=4, base_delay=0.01)

    def responder(url, json=None, headers=None):
        return httpx.Response(403, request=httpx.Request("POST", url))

    with patch("httpx.Client", return_value=_mock_httpx_client(responder)):
        with pytest.raises(autolysis.AuthenticationError):
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
