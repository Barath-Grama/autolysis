#!/usr/bin/env python3
"""Does LLM routing actually beat running every analysis?

The README claims a fixed pipeline "would run all five unconditionally and
often produce meaningless charts on the wrong dataset shape". That is a
testable claim, and this measures it.

## The oracle

The hard part of an eval like this is deciding what the *right* answer is. Hand
labelling would be circular — I would be grading the model against my own
expectations of it. Instead the ground truth is computed: every analysis is run
on every dataset, and its own output decides whether it was worth running.

An analysis is INFORMATIVE when it produced a real finding, DEGENERATE when it
technically produced a chart that tells you nothing (a heatmap of near-zero
correlations, clusters with no separation, a bar chart of 200 bars of height 1),
and SKIPPED when it declined outright. Only the first counts as a hit.

That oracle is strategy-blind: it is derived from the analysis results, never
from what any router chose, so it cannot flatter the LLM.

## The strategies

- `all`       run all five, the fixed-pipeline baseline
- `heuristic` the shipped offline selector, on data shape alone
- `llm`       the real Gemini routing call

`all` gets perfect recall for free, so recall alone would be a meaningless
contest. Precision — of what you chose, how much was worth choosing — is where a
router has to earn its place, and F1 balances the two at a fixed budget.

Usage:
    python evals/routing_eval.py                 # all + heuristic, no key needed
    python evals/routing_eval.py --llm           # adds the Gemini strategy
    python evals/routing_eval.py --llm --trials 3
    python evals/routing_eval.py --llm --out evals/RESULTS.md

Note on quota: the Gemini free tier allows 20 requests per day per model, so
`--trials 3` across nine datasets (27 calls) will exhaust it. --delay paces
calls under the per-minute limit but cannot help with the daily cap.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autolysis
from evals.datasets import DATASETS

# A chart that clears none of these bars is technically output and practically
# noise. Thresholds are deliberately lenient: the aim is to catch the plainly
# useless, not to adjudicate borderline findings.
MIN_ABS_CORRELATION = 0.15
MIN_SILHOUETTE = 0.25
MIN_TIME_POINTS = 4
MAX_CATEGORY_UNIQUENESS = 0.5  # distinct values as a fraction of rows

INFORMATIVE, DEGENERATE, SKIPPED = "informative", "degenerate", "skipped"


def judge(name: str, result: dict, rows: int) -> str:
    """Classify one analysis result. Sees only the result, never the chooser."""
    if result.get("skipped"):
        return SKIPPED

    if name == "correlation":
        pairs = result.get("top_correlations") or []
        if not pairs or abs(pairs[0]["r"]) < MIN_ABS_CORRELATION:
            return DEGENERATE  # a heatmap of noise
        return INFORMATIVE

    if name == "outliers":
        counts = result.get("outlier_counts") or {}
        if not counts:
            return DEGENERATE
        return INFORMATIVE  # "no outliers" is itself a finding

    if name == "clustering":
        silhouette = result.get("silhouette")
        if silhouette is None or silhouette < MIN_SILHOUETTE:
            return DEGENERATE  # k-means always returns k groups; separation is the question
        return INFORMATIVE

    if name == "time_series":
        if result.get("n_points", MIN_TIME_POINTS) < MIN_TIME_POINTS:
            return DEGENERATE
        return INFORMATIVE

    if name == "category_analysis":
        categories = result.get("top_categories") or {}
        if len(categories) < 2:
            return DEGENERATE
        distinct = result.get("n_unique", len(categories))
        if rows and distinct / rows > MAX_CATEGORY_UNIQUENESS:
            return DEGENERATE  # essentially a key column; every bar is height 1
        return INFORMATIVE

    return INFORMATIVE


def build_oracle(df, out_dir: Path) -> dict[str, str]:
    """Run every analysis and record what each one was actually worth."""
    verdicts = {}
    for name, func in autolysis.ANALYSIS_FUNCS.items():
        try:
            result = func(df, out_dir)
        except Exception as exc:  # noqa: BLE001 - a crashing routine is not informative
            verdicts[name] = SKIPPED
            print(f"    ! {name} raised {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        verdicts[name] = judge(name, result, len(df))
    return verdicts


@dataclass
class Score:
    """Aggregate outcome for one strategy."""

    selected: int = 0
    informative: int = 0
    degenerate: int = 0
    skipped: int = 0
    missed: int = 0
    # Datasets this strategy actually produced a selection for. Distinguishes
    # "chose nothing useful" from "never ran" — without it, a strategy whose
    # calls all failed reports precision 0.00 and reads as a terrible router
    # rather than as missing data.
    completed: int = 0
    per_dataset_f1: list[float] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.completed > 0

    @property
    def precision(self) -> float:
        return self.informative / self.selected if self.selected else 0.0

    @property
    def recall(self) -> float:
        found = self.informative
        return found / (found + self.missed) if (found + self.missed) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def wasted(self) -> int:
        return self.degenerate + self.skipped


def score_selection(selected: list[str], oracle: dict[str, str], score: Score) -> float:
    """Fold one dataset's selection into a running Score; return its F1."""
    available = {n for n, v in oracle.items() if v == INFORMATIVE}
    hits = sum(1 for n in selected if oracle.get(n) == INFORMATIVE)

    score.completed += 1
    score.selected += len(selected)
    score.informative += hits
    score.degenerate += sum(1 for n in selected if oracle.get(n) == DEGENERATE)
    score.skipped += sum(1 for n in selected if oracle.get(n) == SKIPPED)
    score.missed += len(available - set(selected))

    if selected:  # noqa: SIM108 - the ternary form nests two conditions illegibly
        precision = hits / len(selected)
    else:
        # Selecting nothing is the correct answer when there was nothing worth
        # running; scoring that 0 would punish a router for the right call on a
        # barren dataset, which is exactly the judgement being measured.
        precision = 1.0 if not available else 0.0
    recall = hits / len(available) if available else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    score.per_dataset_f1.append(f1)
    return f1


def select_all(df, budget: int) -> list[str]:
    return list(autolysis.ANALYSIS_VOCAB)


def select_heuristic(df, budget: int) -> list[str]:
    return autolysis.select_default_analyses(df, budget)


def make_llm_selector(client: autolysis.GeminiClient, delay: float = 0.0):
    """Wrap the routing call, paced for free-tier quota.

    The client retries a 429 with ~7s of exponential backoff, which is fine for
    a transient blip but cannot outwait a requests-per-minute quota. Spacing the
    calls is what actually keeps a multi-trial sweep inside the free tier.
    """

    def select_llm(df, budget: int) -> list[str]:
        if delay:
            time.sleep(delay)
        prompt = autolysis.build_routing_prompt(autolysis.profile_dataset(df), budget)
        raw = client.generate(
            prompt,
            json_mode=True,
            temperature=autolysis.ROUTING_TEMPERATURE,
            response_schema=autolysis.ROUTING_RESPONSE_SCHEMA,
        )
        return autolysis.parse_routing_response(raw, budget)

    return select_llm


def run_eval(strategies: dict, budget: int, trials: int) -> tuple[dict, dict, dict, dict]:
    """Returns (scores, oracles, per-dataset selections, failure counts)."""
    scores = {name: Score() for name in strategies}
    failures: Counter = Counter()
    oracles: dict[str, dict[str, str]] = {}
    selections: dict[str, dict[str, list[str]]] = {name: {} for name in strategies}

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for dataset_name, build in DATASETS.items():
            df = build()
            print(f"  {dataset_name} ({len(df)} rows x {len(df.columns)} cols)")
            oracle = build_oracle(df, out_dir)
            oracles[dataset_name] = oracle
            worth = [n for n, v in oracle.items() if v == INFORMATIVE]
            print(f"    oracle: {', '.join(worth) if worth else '(nothing informative)'}")

            for strategy_name, select in strategies.items():
                runs = trials if strategy_name == "llm" else 1
                picked: list[str] = []
                for _ in range(runs):
                    try:
                        picked = select(df, budget)
                    except autolysis.AutolysisError as exc:
                        # One rate-limited call must not destroy the whole sweep;
                        # record the gap and carry on so partial results survive.
                        failures[strategy_name] += 1
                        print(f"    {strategy_name:10s} !! {str(exc)[:80]}", file=sys.stderr)
                        continue
                    score_selection(picked, oracle, scores[strategy_name])
                selections[strategy_name][dataset_name] = picked
                print(f"    {strategy_name:10s} -> {', '.join(picked) or '(failed)'}")

    return scores, oracles, selections, failures


def render_report(
    scores: dict, oracles: dict, selections: dict, budget: int, trials: int, usage_line: str | None
) -> str:
    missing = [name for name, score in scores.items() if not score.has_data]
    lines = [
        "# Routing eval: does asking an LLM beat running everything?",
        "",
        "Generated by `python evals/routing_eval.py --llm --out evals/RESULTS.md`.",
        "Ground truth is computed, not hand-labelled: every analysis is run on every",
        "dataset and its own output decides whether it was worth running. See the",
        "module docstring for the rules.",
        "",
    ]
    if missing:
        lines += [
            (
                f"> **Incomplete run.** No data for: {', '.join(f'`{n}`' for n in missing)}. "
                "Every call for those strategies failed — most often the Gemini free "
                "tier's 20-requests-per-day-per-model cap. They are shown as `—` rather "
                "than scored, because a strategy that never ran did not score zero."
            ),
            "",
        ]
    lines += [
        (
            f"Budget: **{budget}** analyses per dataset for the routers "
            f"(`all` is ungated by construction). Datasets: **{len(oracles)}**. "
            f"LLM trials per dataset: **{trials}**."
        ),
        "",
        "## Results",
        "",
        "| Strategy | Precision | Recall | F1 | Charts | Informative | Wasted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, score in scores.items():
        if not score.has_data:
            lines.append(f"| `{name}` | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {score.precision:.2f} | {score.recall:.2f} | {score.f1:.2f} | "
            f"{score.selected} | {score.informative} | {score.wasted} |"
        )

    lines += [
        "",
        "*Charts* is how many analyses the strategy ran in total; *wasted* is how many",
        "of those were skipped outright or produced a degenerate chart.",
        "",
        "## Per-dataset detail",
        "",
        "| Dataset | Worth running | " + " | ".join(f"`{n}`" for n in selections) + " |",
        "|---|---|" + "---|" * len(selections),
    ]
    for dataset_name, oracle in oracles.items():
        worth = [n for n, v in oracle.items() if v == INFORMATIVE]
        cells = []
        for strategy_name in selections:
            picked = selections[strategy_name][dataset_name]
            marked = [f"**{p}**" if oracle.get(p) == INFORMATIVE else f"~~{p}~~" for p in picked]
            cells.append(", ".join(marked) or "—")
        lines.append(f"| `{dataset_name}` | {', '.join(worth) or '—'} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Bold means the oracle agreed the analysis was worth running; ~~struck~~ means it",
        "was skipped or produced a degenerate chart.",
        "",
    ]
    if usage_line:
        lines += ["## Cost", "", f"`{usage_line}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="include the Gemini routing strategy")
    parser.add_argument("--budget", type=int, default=3, help="analyses per dataset (default: 3)")
    parser.add_argument("--trials", type=int, default=1, help="LLM repeats per dataset")
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="seconds between LLM calls, to stay inside free-tier quota (default: 4)",
    )
    parser.add_argument("--model", default=None, help="override the Gemini model")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    strategies = {"all": select_all, "heuristic": select_heuristic}
    client = None

    if args.llm:
        autolysis.load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("--llm needs GEMINI_API_KEY (or GOOGLE_API_KEY).", file=sys.stderr)
            return 1
        model = args.model or os.environ.get("GEMINI_MODEL", autolysis.DEFAULT_MODEL)
        client = autolysis.GeminiClient(api_key, model)
        strategies["llm"] = make_llm_selector(client, args.delay)

    print(f"Running {len(DATASETS)} datasets, budget {args.budget}:")
    scores, oracles, selections, failures = run_eval(strategies, args.budget, args.trials)

    usage_line = None
    if client is not None and client.usage.calls:
        usage_line = client.usage.summary(client.model)

    print("\n" + "=" * 72)
    for name, score in scores.items():
        if not score.has_data:
            print(f"{name:10s} no data - every call failed, so no score is reported")
            continue
        spread = ""
        if len(score.per_dataset_f1) > 1:
            spread = f"  (median per-dataset F1 {statistics.median(score.per_dataset_f1):.2f})"
        print(
            f"{name:10s} precision {score.precision:.2f}  recall {score.recall:.2f}  "
            f"F1 {score.f1:.2f}  charts {score.selected:3d}  wasted {score.wasted:3d}{spread}"
        )
    if usage_line:
        print(f"\n{usage_line}")
    if failures:
        # Reported, never silently averaged away: a strategy that failed half
        # its calls has not earned the score its remaining calls produced.
        print(f"\nCalls that failed and were excluded: {dict(failures)}", file=sys.stderr)

    report = render_report(scores, oracles, selections, args.budget, args.trials, usage_line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
