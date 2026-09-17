# blood-cell-agent

An LLM agent that answers natural-language questions about the results of
[rare-cell-morphologies](../rare-cell-morphologies), a blood-cell anomaly
detection FYP. Built to demonstrate LangGraph + MCP skills: a ReAct agent
with grounded tool use, and an MCP server exposing the same tools to Claude
Desktop.

This repo is standalone and does not modify the original FYP repo - it only
reads 15 static result files copied into `data/`.

## What's here

- **`agent/tools.py`** - three Python functions that parse the static result
  files (`.txt`/`.csv`) with regex. No ML inference at runtime; the
  thresholds and metrics were already computed by the FYP pipeline.
- **`agent/agent.py`** - a LangGraph ReAct agent (`create_react_agent`) that
  uses Claude to decide which tool to call to answer a question.
- **`agent/mcp_server.py`** - an MCP server (stdio transport) exposing the
  same three tools, so Claude Desktop or another MCP client can call them
  directly.
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

Set your API key (or run `ant auth login` if you use the Anthropic CLI):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

By default everything uses `claude-opus-5`. Override with
`BLOOD_CELL_AGENT_MODEL` if you want a cheaper model for eval runs, e.g.
`export BLOOD_CELL_AGENT_MODEL=claude-sonnet-5`.

## Running the agent

```bash
python -m agent.agent "Which holdout class had the best F1?"
```

## Running the MCP server

```bash
python -m agent.mcp_server
```

To use it from Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "blood-cell-agent": {
      "command": "python",
      "args": ["-m", "agent.mcp_server"],
      "cwd": "/absolute/path/to/blood-cell-agent"
    }
  }
}
```

To verify the server works without needing Claude Desktop or an API key,
`agent/mcp_smoke_test.py` spawns it as a real subprocess, connects a proper
MCP client over stdio, and checks its tool responses against calling
`tools.py` directly:

```bash
python -m agent.mcp_smoke_test
```

## Running the eval

```bash
python -m eval.run_eval
```

Runs all 40 questions through the tool-using agent and a plain (no-tool)
Claude call, grades each answer, and writes `eval/results.md`. Every
question costs one API call per condition (80 calls total for the full
set) - use `--limit N` to try a smaller batch first, or `--model` to pick a
cheaper model.

## Verifying the tools directly

`agent/tools.py` has no LLM dependency - it's pure file parsing, so it's
worth checking independently of any API key:

```bash
python -c "from agent.tools import get_flagged_cells; print(get_flagged_cells('basophil'))"
```
