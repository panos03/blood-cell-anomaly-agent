"""LangGraph ReAct agent that answers questions about the FYP's anomaly
detection results, using Claude (via langchain-anthropic) as the model and
agent/tools.py as its tool surface.

Usage:
    python -m agent.agent "Which holdout class had the best F1?"
"""
from __future__ import annotations

import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent import tools as t

MODEL = os.environ.get("BLOOD_CELL_AGENT_MODEL", "claude-opus-5")

SYSTEM_PROMPT = f"""You are an analysis assistant for a blood-cell anomaly
detection research project. You answer questions about experiment results by
calling the tools available to you - never guess numbers from memory.

Context: the project holds out one white-blood-cell class at a time and
trains a distance-based anomaly detector (a Mahalanobis-distance detector
over DinoBloom features) to flag cells of that class as anomalies, compared
against a Maximum Softmax Probability (MSP) baseline. Results are static and
were computed once; your tools just read them.

Valid holdout classes: {", ".join(t.HOLDOUT_CLASSES)}.

When a question requires comparing across holdout classes (e.g. "best F1"),
call get_flagged_cells for each relevant class and compare the results
yourself - there is no single tool that ranks all classes at once.

Answer concisely and always cite the concrete numbers you found."""


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
    """
    return t.compare_to_baseline(holdout_class, metric)


TOOLS = [get_flagged_cells, get_cluster_summary, compare_to_baseline]


def build_agent():
    model = ChatAnthropic(model=MODEL, max_tokens=4096)
    return create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m agent.agent "<question>"')
        sys.exit(1)
    print(ask(" ".join(sys.argv[1:])))
