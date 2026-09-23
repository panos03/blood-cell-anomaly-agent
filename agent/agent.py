"""LangGraph ReAct agent that answers questions about the FYP's anomaly
detection results, using Gemini (via langchain-google-genai) as the model
and agent/tools.py as its tool surface.

Usage:
    python -m agent.agent "Which holdout class had the best F1?"
"""
from __future__ import annotations

import os
import sys

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agent import tools as t

# NOTE: Why no MCP server:
# MCP exposes tools over stdio/HTTP to external clients (like Claude Desktop).
# Because this script runs in-process and calls tools directly via LangChain's
# @tool decorator, an MCP server is not necessary.

# Requires a free API key from https://aistudio.google.com (no card needed),
# set as the GOOGLE_API_KEY environment variable. "flash" models sit on
# Gemini's free tier ("pro" models don't - free-tier quota is 0 for those);
# see ai.google.dev for current rate limits/model names.
MODEL = os.environ.get("BLOOD_CELL_AGENT_MODEL", "gemini-3.6-flash")

SYSTEM_PROMPT = f"""You are an analysis assistant for a blood-cell anomaly
detection research project. You answer questions about experiment results by
calling the tools available to you. Never guess numbers from memory.

Context: the project holds out one white-blood-cell class at a time and
trains a distance-based anomaly detector (a Mahalanobis-distance detector
over DinoBloom features) to flag cells of that class as anomalies, compared
against a Maximum Softmax Probability (MSP) baseline. Results are static and
were computed once, your tools just read them.

Valid holdout classes: {", ".join(t.HOLDOUT_CLASSES)}.

When a question requires comparing across all holdout classes (e.g. "best F1"),
call get_flagged_cells for each relevant class and compare the results
yourself, there is no single tool that ranks all classes at once.

Likewise, get_cluster_summary only covers one k (2 or 3) per call. If a
question isn't scoped to a specific k (e.g. "which clustering method works
best overall"), call it for both k=2 and k=3 and compare yourself.

Answer concisely and always cite the concrete numbers you found.

Your answers are printed to a plain terminal, not rendered as Markdown - so
write plain text only. No **bold**, no # headers, no tables, no bullet
markup. Use plain sentences, or simple "label: value" lines and blank-line
separated paragraphs if you need structure."""


@tool
def get_flagged_cells(holdout_class: str) -> dict:
    """Get TP/FP/FN/TN/precision/recall/F1 for the distance detector and the
    MSP baseline on a given holdout class.

    Args:
        holdout_class: one of basophil, eosinophil, erythroblast, ig,
            lymphocyte, monocyte, neutrophil, platelet.
    """
    return t.get_flagged_cells(holdout_class)


@tool
def get_cluster_summary(k: int, holdout_class: str | None = None) -> dict:
    """Get per-clustering-method ARI/AMI/purity (mean +/- std) for the k-class
    holdout clustering experiment.

    Args:
        k: 2 or 3 - the number of classes held out together in this
            clustering scenario.
        holdout_class: optional class name to filter which split groups are
            reported (does not change the aggregate ARI/AMI/purity numbers,
            which are computed across all splits for that k).
    """
    return t.get_cluster_summary(k=k, holdout_class=holdout_class)


@tool
def compare_to_baseline(holdout_class: str, metric: str) -> dict:
    """Compare the distance detector to the MSP baseline for a given holdout
    class and metric.

    Args:
        holdout_class: one of basophil, eosinophil, erythroblast, ig,
            lymphocyte, monocyte, neutrophil, platelet.
        metric: one of auroc, auprc, recall, precision, f1, mcc,
            specificity, fpr (anomaly-detection metrics), or f1/precision/
            recall (also matched against the classification per-class table).

    The returned "winner" already accounts for metric direction (lower is
    better for fpr, higher is better for everything else) - trust it as-is
    rather than re-deriving it from which raw number looks bigger.
    """
    return t.compare_to_baseline(holdout_class, metric)


TOOLS = [get_flagged_cells, get_cluster_summary, compare_to_baseline]


def extract_text(message) -> str:
    # A response's .content is a plain string for some models, but Gemini
    # returns a list of content blocks (with extra metadata like safety/
    # signature info) - this pulls out just the actual answer text either way.
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts)


def build_agent():
    # Assembles Gemini + the 3 tools + the system prompt into an agent.
    # Just builds it, doesn't ask anything yet.
    model = ChatGoogleGenerativeAI(model=MODEL, max_output_tokens=4096)
    return create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    # Runs one question through the agent and returns its final answer.
    # (result["messages"] has the full back-and-forth if we want to
    # inspect which tools were called along the way.)
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return extract_text(result["messages"][-1])


if __name__ == "__main__":
    # Entry point for: python -m agent.agent "<question>"
    if len(sys.argv) < 2:
        print('Usage: python -m agent.agent "<question>"')
        sys.exit(1)
    print(ask(" ".join(sys.argv[1:])))
