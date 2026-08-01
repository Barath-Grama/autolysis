# Autolysis

**Autolysis** is an intelligent, fully automated CSV analysis engine. Point it at a
CSV file and it will:

1. **Profile** the dataset (dtypes, null ratios, summary stats / top categories).
2. Ask **Gemini** which of five supported analyses are most appropriate for the data.
3. Run those analyses **locally** with pandas / scikit-learn / matplotlib — no raw
   data rows ever leave your machine, only aggregate statistics.
4. Ask Gemini a second time to **narrate** the numeric findings into a polished,
   chart-embedded `README.md`.

A typical run takes 15–40 seconds and replaces what would otherwise be hours of
manual exploratory analysis.

```
CSV Input → Data Profiler → LLM Call #1 (analysis routing)
   → Local Analysis Engine (correlation / outliers / clustering / time_series / category_analysis)
   → chart .png files → LLM Call #2 (narrative) → README.md
```

## Quickstart

The fastest way to run Autolysis is with [`uv`](https://docs.astral.sh/uv/), which
reads the inline dependency metadata at the top of `autolysis.py` and provisions
an isolated environment automatically — no `pip install` or virtualenv needed.

```bash
export GEMINI_API_KEY="your-key-here"   # from https://aistudio.google.com/apikey
uv run autolysis.py sample_data/happiness.csv
```

This produces `happiness_output/` containing `README.md` plus one `.png` chart
per selected analysis.

### Without `uv`

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python autolysis.py sample_data/goodreads.csv
```

### Without an API key

Autolysis can also run entirely offline (no Gemini calls at all), using a
default analysis set and a deterministic templated report — handy for demos,
CI, or environments without network access:

```bash
python autolysis.py sample_data/happiness.csv --offline
```

## CLI reference

```
usage: autolysis [-h] [-o OUTPUT_DIR] [-m MODEL] [--max-analyses MAX_ANALYSES]
                  [--offline] [--cache] [--html] [-v]
                  csv_path

positional arguments:
  csv_path              Path to the CSV file to analyze.

options:
  -o, --output-dir      Output directory (default: ./<csv-stem>_output)
  -m, --model           Gemini model to use (default: env GEMINI_MODEL or
                         'gemini-2.5-flash-lite')
  --max-analyses N      Maximum number of analyses the LLM may select (default: 3)
  --offline             Skip all LLM calls; run default analyses + templated report
  --cache               Cache LLM responses on disk (.autolysis_cache/) to avoid
                         repeat API calls during development
  --html                Additionally render report.html alongside README.md
  -v, --verbose          Enable debug logging
```

## Environment variables

| Variable                       | Purpose                                    |
|---------------------------------|---------------------------------------------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini API key (required unless `--offline`) |
| `GEMINI_MODEL`                  | Overrides the default model (`gemini-2.5-flash-lite`) |

See `.env.example`.

## Supported analyses

Gemini selects up to `--max-analyses` (default 3) of the following, based on the
dataset's structure — a fixed pipeline would run all five unconditionally and
often produce meaningless charts on the wrong dataset shape:

| Analysis             | Technique                                                          |
|-----------------------|---------------------------------------------------------------------|
| `correlation`          | Pearson correlation matrix → heatmap                               |
| `outliers`             | IQR (Tukey fence) outlier detection → z-scored box plot             |
| `clustering`            | K-Means on the two highest-variance numeric columns, k chosen via elbow method → scatter plot |
| `time_series`           | Heuristic time-column detection + OLS trend line                    |
| `category_analysis`     | Top-N frequency bar chart for the richest categorical column        |

Every routine degrades gracefully — e.g. clustering is skipped with a clear
reason if fewer than two numeric columns are present — so the pipeline always
completes and always produces a report.

## Project layout

```
autolysis.py           # the entire pipeline — single, uv-executable script
tests/
  test_autolysis.py    # unit tests (all Gemini calls mocked, no network needed)
sample_data/
  goodreads.csv        # synthetic book-metadata dataset for demos
  happiness.csv        # synthetic multi-year, multi-country wellbeing dataset
requirements.txt       # pinned deps for non-uv / pip workflows
pyproject.toml         # dev/test tooling config
.env.example           # environment variable template
```

> **Note on sample data:** `sample_data/*.csv` are synthetically generated
> (see the generation notes in each file's header comment context) to mirror
> the structure of the well-known Goodreads and World Happiness Report
> datasets without redistributing the original copyrighted data. Swap in the
> real files if you have them — the column-name heuristics will still work.

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All 36 tests mock the Gemini API (`unittest.mock`), so the suite runs fully
offline and deterministically. Coverage includes:

- IQR outlier boundary correctness (equivalence partitioning at the Tukey fence)
- K-Means elbow-method k-selection
- LLM routing response parsing — valid JSON, code-fenced JSON, unknown
  identifiers silently dropped, fully malformed text falling back safely
- Chart file creation/non-zero size for every analysis routine
- End-to-end `README.md` generation with a mocked two-call LLM sequence
- Graceful degradation: single numeric column, empty CSV, missing file,
  missing API key, no temporal/categorical columns
- Secret hygiene: the API key is sent as a header and never appears in a
  raised error, plus a redaction backstop for third-party error text
- CSV encoding fallback: latin-1/cp1252 input is decoded rather than crashing,
  UTF-8 still takes precedence, malformed CSVs raise a clean error
- `--max-analyses` propagating from the CLI into the routing prompt itself

## Architecture notes

- **Two LLM calls, not more.** Call #1 sends only column-level metadata
  (dtypes, null %, five-number summaries / top-5 categories) — never raw
  rows — and gets back a JSON array of analysis identifiers drawn from a
  closed vocabulary, so a non-compliant response can never trigger an
  unsupported routine. Call #2 sends the computed numeric results and chart
  filenames and gets back the final Markdown narrative.
- **Sequential execution.** Analysis routines run one after another rather
  than concurrently: each is dominated by already-parallelized pandas/
  scikit-learn internals, and charts must exist on disk before the
  narrative prompt is assembled.
- **Resilience beyond the original design.** Gemini calls retry with
  exponential backoff; if the narrative call still fails, Autolysis falls
  back to a deterministic templated report rather than crashing.
- **Secret handling.** The API key travels in the `x-goog-api-key` header, not
  a URL query parameter — httpx embeds the full request URL in its exception
  messages, so a query-string key leaks into every retry log line and error
  report. `redact_secret()` scrubs any residual occurrence as a backstop
  against third-party error text we don't control.
- **Encoding fallback.** Input is decoded through `utf-8 → utf-8-sig → cp1252
  → latin-1`. pandas defaults to UTF-8 and raises on anything else, but many
  real-world CSVs are Windows-Latin exports; a bare `read_csv` turns a routine
  file into an uncaught traceback.
- **`--offline` mode** runs the full local pipeline (profiling → analysis →
  chart rendering → templated report) with zero network calls, useful for
  CI, air-gapped environments, or quick demos.
- **`--cache`** memoizes LLM responses on disk so repeated runs against the
  same dataset during development don't re-incur API cost/latency.
- **`--html`** renders `report.html` from the generated Markdown for easy
  sharing outside of a Markdown viewer.
