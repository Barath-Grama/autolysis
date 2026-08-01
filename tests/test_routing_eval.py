"""Tests for the routing eval harness.

The eval grades the router, so the grader itself needs grading: a scoring bug
would produce a confident number that means nothing. No network is used — the
LLM strategy is exercised through a stub.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autolysis
from evals import routing_eval as ev
from evals.datasets import DATASETS

# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #


def test_skipped_analysis_is_never_a_hit():
    assert ev.judge("clustering", {"skipped": True, "reason": "too few columns"}, 100) == ev.SKIPPED


def test_noise_correlation_is_degenerate_not_informative():
    """A heatmap of near-zero correlations is output, not a finding."""
    weak = {"top_correlations": [{"a": "x", "b": "y", "r": 0.02}]}
    strong = {"top_correlations": [{"a": "x", "b": "y", "r": 0.61}]}
    assert ev.judge("correlation", weak, 200) == ev.DEGENERATE
    assert ev.judge("correlation", strong, 200) == ev.INFORMATIVE


def test_negative_correlation_counts_by_magnitude():
    result = {"top_correlations": [{"a": "x", "b": "y", "r": -0.72}]}
    assert ev.judge("correlation", result, 200) == ev.INFORMATIVE


def test_unseparated_clusters_are_degenerate():
    """K-Means always returns k groups; whether they are apart is the question."""
    assert ev.judge("clustering", {"k": 3, "silhouette": 0.05}, 200) == ev.DEGENERATE
    assert ev.judge("clustering", {"k": 3, "silhouette": 0.62}, 200) == ev.INFORMATIVE
    assert ev.judge("clustering", {"k": 3, "silhouette": None}, 200) == ev.DEGENERATE


def test_trend_through_too_few_points_is_degenerate():
    assert ev.judge("time_series", {"n_points": 2}, 200) == ev.DEGENERATE
    assert ev.judge("time_series", {"n_points": 40}, 200) == ev.INFORMATIVE


def test_key_column_frequency_chart_is_degenerate():
    """200 bars of height 1 is not a category analysis."""
    key_like = {"top_categories": {f"id{i}": 1 for i in range(8)}, "n_unique": 190}
    real = {"top_categories": {"north": 90, "south": 60}, "n_unique": 4}
    assert ev.judge("category_analysis", key_like, 200) == ev.DEGENERATE
    assert ev.judge("category_analysis", real, 200) == ev.INFORMATIVE


def test_no_outliers_is_still_a_finding():
    assert ev.judge("outliers", {"outlier_counts": {"a": 0, "b": 0}}, 200) == ev.INFORMATIVE


def test_oracle_is_blind_to_who_chose_what():
    """judge() takes only the result, so it cannot favour a strategy."""
    import inspect

    params = set(inspect.signature(ev.judge).parameters)
    assert params == {"name", "result", "rows"}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


ORACLE = {
    "correlation": ev.INFORMATIVE,
    "outliers": ev.INFORMATIVE,
    "clustering": ev.DEGENERATE,
    "time_series": ev.SKIPPED,
    "category_analysis": ev.INFORMATIVE,
}


def test_perfect_selection_scores_one():
    score = ev.Score()
    f1 = ev.score_selection(["correlation", "outliers", "category_analysis"], ORACLE, score)
    assert f1 == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.wasted == 0


def test_running_everything_trades_precision_for_recall():
    score = ev.Score()
    ev.score_selection(list(ORACLE), ORACLE, score)
    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(3 / 5)
    assert score.wasted == 2  # one degenerate, one skipped


def test_missing_an_informative_analysis_costs_recall():
    score = ev.Score()
    ev.score_selection(["correlation"], ORACLE, score)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1 / 3)
    assert score.missed == 2


def test_choosing_only_wasteful_analyses_scores_zero():
    score = ev.Score()
    f1 = ev.score_selection(["clustering", "time_series"], ORACLE, score)
    assert f1 == 0.0
    assert score.informative == 0
    assert score.wasted == 2


def test_empty_selection_is_not_a_divide_by_zero():
    score = ev.Score()
    assert ev.score_selection([], ORACLE, score) == 0.0


def test_dataset_with_nothing_worth_running():
    """Selecting nothing is the right answer; recall must not punish it."""
    barren = dict.fromkeys(ORACLE, ev.SKIPPED)
    score = ev.Score()
    assert ev.score_selection([], barren, score) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# "No data" must not read as "scored zero"
# --------------------------------------------------------------------------- #


def test_strategy_with_no_completed_calls_reports_no_data():
    score = ev.Score()
    assert not score.has_data


def test_one_completed_call_counts_as_data():
    score = ev.Score()
    ev.score_selection(["correlation"], ORACLE, score)
    assert score.has_data


def test_report_renders_a_dash_for_a_strategy_that_never_ran():
    scores = {"all": ev.Score(), "llm": ev.Score()}
    ev.score_selection(list(ORACLE), ORACLE, scores["all"])
    report = ev.render_report(
        scores,
        {"d": ORACLE},
        {"all": {"d": list(ORACLE)}, "llm": {"d": []}},
        budget=3,
        trials=1,
        usage_line=None,
    )
    assert "| `llm` | — | — | — | — | — | — |" in report
    assert "0.00" not in report.split("## Per-dataset")[0]


# --------------------------------------------------------------------------- #
# Datasets and end-to-end wiring
# --------------------------------------------------------------------------- #


def test_datasets_are_deterministic():
    """A reproducible eval needs reproducible inputs."""
    for name, build in DATASETS.items():
        pd.testing.assert_frame_equal(build(), build(), obj=name)


def test_datasets_cover_varied_shapes():
    """If every dataset suited every analysis, the eval would prove nothing."""
    shapes = set()
    for build in DATASETS.values():
        df = build()
        numeric = len(df.select_dtypes("number").columns)
        has_time = autolysis.find_time_column(df) is not None
        has_cat = len(df.select_dtypes(exclude="number").columns) > 0
        shapes.add((numeric >= 2, has_time, has_cat))
    assert len(shapes) >= 4, f"only {len(shapes)} distinct shapes: {shapes}"


def test_oracle_verdicts_vary_across_datasets(tmp_path):
    """A constant oracle would make every strategy score identically."""
    verdict_sets = set()
    for build in DATASETS.values():
        oracle = ev.build_oracle(build(), tmp_path)
        verdict_sets.add(frozenset(n for n, v in oracle.items() if v == ev.INFORMATIVE))
    assert len(verdict_sets) >= 4


def test_run_eval_with_a_stubbed_llm(tmp_path):
    """End-to-end wiring, with the model replaced by a fixed answer."""
    strategies = {
        "all": ev.select_all,
        "heuristic": ev.select_heuristic,
        "llm": lambda df, budget: ["correlation", "outliers"],
    }
    scores, oracles, _, failures = ev.run_eval(strategies, budget=3, trials=1)

    assert set(scores) == {"all", "heuristic", "llm"}
    assert len(oracles) == len(DATASETS)
    assert not failures
    assert all(score.has_data for score in scores.values())
    # The fixed-pipeline baseline must never miss anything.
    assert scores["all"].recall == pytest.approx(1.0)


def test_a_failing_llm_call_does_not_abort_the_sweep():
    """One rate-limited call must not discard the datasets already scored."""
    calls = {"n": 0}

    def flaky(df, budget):
        calls["n"] += 1
        if calls["n"] == 1:
            raise autolysis.AutolysisError("429 Too Many Requests")
        return ["correlation"]

    scores, oracles, _, failures = ev.run_eval({"llm": flaky}, budget=3, trials=1)
    assert failures["llm"] == 1
    assert len(oracles) == len(DATASETS)
    assert scores["llm"].completed == len(DATASETS) - 1


def test_cli_runs_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    out = tmp_path / "RESULTS.md"
    assert ev.main(["--out", str(out)]) == 0
    body = out.read_text(encoding="utf-8")
    assert "| `all` |" in body and "| `heuristic` |" in body


def test_cli_refuses_llm_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch.object(autolysis, "load_dotenv", lambda *a, **k: None):
        assert ev.main(["--llm"]) == 1
    assert "needs GEMINI_API_KEY" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
