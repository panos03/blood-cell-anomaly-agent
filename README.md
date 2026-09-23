# blood-cell-agent

An LLM agent that answers natural-language questions about the results of
[rare-cell-morphologies](https://github.com/panos03/rare-cell-morphologies), a blood-cell anomaly
detection FYP. Showcases agent/tool skills (LangGraph, the
ReAct pattern, and how MCP fits in).

Note, this repo is standalone and does not modify the original FYP repo. It only
reads 15 static result files (`.txt`/`.csv`) copied into `data/`, there is
no ML inference at runtime, just an LLM answering questions by reading
those files through a few Python functions.

## Key concepts

- **Tool** - an ordinary Python function, described to the LLM with a name,
  parameters, and docstring (as a JSON schema) so the LLM can ask for it to
  be run and get the result back. See `agent/tools.py`: three functions
  that parse the FYP's result files. The LLM never runs this code itself.
- **Agent** - an LLM given a set of tools and a loop: ask the model, if it
  requests a tool call run it and feed the result back, repeat until the
  model gives a final answer instead of another tool call. This is what
  makes multi-step questions ("which class had the best F1?") possible -
  a single API call couldn't check all 8 classes itself.
- **ReAct** - the specific reason-then-act loop pattern above (Reason, Act,
  observe, repeat). `agent/agent.py` uses LangGraph's `create_react_agent`
  to build that loop rather than writing it by hand.
- **MCP (Model Context Protocol)** - a standard for exposing tools to an
  *external* client (e.g. Claude Desktop) over a protocol, without that
  client needing to import your code. Worth knowing, but this project
  doesn't include an MCP server: the only consumer of these tools is this
  same script, calling them in-process - see the comment near the top of
  `agent/agent.py` for the full reasoning.
- **Eval** - a set of Q&A pairs with known, verified answers
  (`eval/questions.json`), run through the agent and graded automatically,
  to measure whether tool access actually makes the agent more correct
  than the same model with no tools - rather than just assuming it does.

## What's here

- **`agent/tools.py`** - the 3 tool functions (regex/dict parsing of the
  result files), independent of any LLM.
- **`agent/agent.py`** - the LangGraph ReAct agent, using Gemini.
- **`eval/`** - `questions.json` (40 Q&A pairs), `run_eval.py` (the
  harness), `results.md` (generated output).

| Tool | Signature | Source file(s) |
|---|---|---|
| `get_flagged_cells` | `(holdout_class)` | `data/experiments/anomaly_detection/holdout_{class}/summary.txt` |
| `get_cluster_summary` | `(k, holdout_class=None)` | `data/experiments/cluster_discovery/summary_metrics.txt` |
| `compare_to_baseline` | `(holdout_class, metric)` | anomaly_detection + classification summaries |

Valid holdout classes: `basophil`, `eosinophil`, `erythroblast`, `ig`,
`lymphocyte`, `monocyte`, `neutrophil`, `platelet`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com)
(no card required), then set it:

```bash
export GOOGLE_API_KEY=AIza...
```

By default everything uses `gemini-3.6-flash`. "Pro" models aren't on the
free tier at all (their free quota is 0) - stick to "flash"/"flash-lite"
models. Override with `BLOOD_CELL_AGENT_MODEL` if the default is
unavailable (see **Notes** below).

## Running it

```bash
python -m agent.agent "Which holdout class had the best F1?"
python -m eval.run_eval --limit 4
```

`run_eval.py` runs each question through the tool-using agent *and* a plain
(no-tool) Gemini call, grades both, and writes `eval/results.md`. Each
question costs several real API calls (the agent's tool-calling loop is
2+ calls by itself), so `--limit N` is worth using rather than the full 40
at once - see **Notes**.

## Key results (4/40 questions run so far - see notes below)

| Metric | With tools (agent) | Without tools (plain Gemini) |
|---|---|---|
| Accuracy | **4/4 (100%)** | 0/4 (0%) |
| Correct tool call rate | 4/4 (100%) | n/a |

Full detail, including every question and both models' full answers, is in
[`eval/results.md`](eval/results.md). One example, to make the gap
concrete - asked *"How many true positives did the distance detector get
on the basophil holdout?"*:

- **With tools:** "The distance detector got 1042 true positives on the
  basophil holdout." (correct)
- **Without tools:** "I do not know the specific numbers... as I don't
  have access to the internal data or specific experimental results of
  that particular research project." (accurate about its own limits, but
  can't answer)

This is the actual point of the project: the model has no way to know
these specific, unpublished numbers on its own - tool access is what turns
"I don't know" into a correct, grounded answer.

## Notes

- **Only 4 of the 40 eval questions have been run.** Gemini's free tier
  caps requests per model *per day* - as low as ~20/day for some models in
  testing - and the agent's tool-calling loop uses several real requests
  per question, not one. Running all 40 in a single day on one free-tier
  model isn't realistic. Check current usage/limits at
  [ai.dev/rate-limit](https://ai.dev/rate-limit), or switch model via
  `BLOOD_CELL_AGENT_MODEL` if the default one is exhausted or overloaded
  (try `gemini-3.5-flash`, `gemini-3.5-flash-lite`, or `gemini-3.7-flash`).
- **The eval's grader is a heuristic, not a judge.** `"string"`-type
  questions are graded by substring match, not by checking the model's
  actual stated conclusion. A verbose answer that names several candidate
  classes while getting a ranking wrong can still contain the expected
  name and get marked "correct" - observed in practice once (see
  `eval/run_eval.py`'s comment on `_grade_string` for the real example).
  Treat ranking/comparison-question verdicts as a useful signal, not a
  guarantee - worth spot-checking the actual answer text in
  `eval/results.md`.

## Verifying the tools directly

`agent/tools.py` has no LLM dependency - it's pure file parsing, so it's
worth checking independently of any API key:

```bash
python -c "from agent.tools import get_flagged_cells; print(get_flagged_cells('basophil'))"
```
