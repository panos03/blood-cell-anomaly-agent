# TODO

# blood-cell-agent

An LLM agent that answers natural-language questions about the results of
[rare-cell-morphologies](../rare-cell-morphologies), a blood-cell anomaly
detection FYP. Built to demonstrate LangGraph skills: a ReAct agent with
grounded tool use over the project's static experiment results.

This repo is standalone and does not modify the original FYP repo - it only
reads 15 static result files copied into `data/`.

## What's here

- **`agent/tools.py`** - three Python functions that parse the static result
  files (`.txt`/`.csv`) with regex. No ML inference at runtime; the
  thresholds and metrics were already computed by the FYP pipeline.
- **`agent/agent.py`** - a LangGraph ReAct agent (`create_react_agent`) that
  uses Gemini to decide which tool to call to answer a question. See the
  comment near the top of that file for why this project doesn't need an
  MCP server on top of it.
- **`eval/`** - 40 hand-written Q&A pairs with known answers, and a harness
  that measures accuracy with vs. without tool access, correct tool call
  rate, and failure modes.

## Tools

| Tool | Signature | Source file(s) |
|---|---|---|
| `get_flagged_cells` | `(holdout_class)` | `data/experiments/anomaly_detection/holdout_{class}/summary.txt` |
| `get_cluster_summary` | `(k, holdout_class=None)` | `data/experiments/cluster_discovery/summary_metrics.txt` |
| `compare_to_baseline` | `(holdout_class, metric)` | anomaly_detection + classification summaries |

Valid holdout classes: `basophil`, `eosinophil`, `erythroblast`, `ig`,
`lymphocyte`, `monocyte`, `neutrophil`, `platelet`.

Note on `get_cluster_summary`: the underlying clustering experiment holds
out *groups* of 2 or 3 classes together (not one class at a time), so `k`
(2 or 3) selects the scenario and `holdout_class` just filters which split
groups are listed - it doesn't change the reported ARI/AMI/purity numbers,
which are aggregated across all splits for that k.

`compare_to_baseline`'s `winner` field accounts for metric direction:
higher is better for every metric except FPR, where lower is better.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com) (no
card required), then set it:

```bash
export GOOGLE_API_KEY=AIza...
```

By default everything uses `gemini-3.7-flash`, which sits on Gemini's free
tier (rate-limited, not unlimited - see ai.google.dev for current limits).
Override with `BLOOD_CELL_AGENT_MODEL` to use a different model, e.g.
`export BLOOD_CELL_AGENT_MODEL=gemini-3.1-pro-preview`.

## Running the agent

```bash
python -m agent.agent "Which holdout class had the best F1?"
```

## Running the eval

```bash
python -m eval.run_eval
```

Runs all 40 questions through the tool-using agent and a plain (no-tool)
Gemini call, grades each answer, and writes `eval/results.md`. Every
question costs one API call per condition (80 calls total for the full
set) - use `--limit N` to try a smaller batch first if you're worried about
free-tier rate limits.

## Verifying the tools directly

`agent/tools.py` has no LLM dependency - it's pure file parsing, so it's
worth checking independently of any API key:

```bash
python -c "from agent.tools import get_flagged_cells; print(get_flagged_cells('basophil'))"
```
