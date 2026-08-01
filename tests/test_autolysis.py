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


def test_elbow_k_selection_plateau_at_3():
    """WCSS drops sharply from k=2->3, then plateaus -> elbow should return k=3."""
    inertias = {2: 300.0, 3: 100.0, 4: 90.0, 5: 85.0}
    assert autolysis.select_k_by_elbow(inertias) == 3


def test_elbow_k_selection_plateau_at_4():
    inertias = {2: 400.0, 3: 300.0, 4: 100.0, 5: 95.0}
    assert autolysis.select_k_by_elbow(inertias) == 4


def test_clustering_end_to_end(synthetic_df, out_dir):
    result = autolysis.run_clustering(synthetic_df, out_dir)
    assert not result.get("skipped")
    assert result["k"] in {2, 3, 4, 5}
    assert Path(result["chart"]).exists()
    assert Path(result["chart"]).stat().st_size > 0


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
            "y": {"dtype": "object", "null_pct": 0.0, "top_values": {"a": 5, "b": 5}},
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
