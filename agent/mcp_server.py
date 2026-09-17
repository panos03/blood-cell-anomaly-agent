"""MCP server exposing the anomaly-detection result tools over stdio, so
Claude Desktop (or any MCP client) can call them directly.

Run with:
    python -m agent.mcp_server

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "blood-cell-agent": {
          "command": "python",
          "args": ["-m", "agent.mcp_server"],
          "cwd": "/absolute/path/to/blood-cell-agent"
        }
      }
    }
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agent import tools as t

mcp = MCPServer(
    "blood-cell-agent",
    instructions=(
        "Tools for querying static anomaly-detection and clustering results "
        "from a blood-cell anomaly detection research project. All data is "
        "pre-computed - these tools parse result files, they do not run any "
        "ML inference."
    ),
)


@mcp.tool()
def get_flagged_cells(holdout_class: str) -> dict:
    """Get TP/FP/FN/TN/precision/recall/F1 for the distance detector and the
    MSP baseline on a given holdout class.

    Args:
        holdout_class: one of basophil, eosinophil, erythroblast, ig,
            lymphocyte, monocyte, neutrophil, platelet.
    """
    return t.get_flagged_cells(holdout_class)


@mcp.tool()
def get_cluster_summary(k: int, holdout_class: str | None = None) -> dict:
    """Get per-clustering-method ARI/AMI/purity (mean +/- std) for the
    k-class holdout clustering experiment (k=2 or k=3).

    Args:
        k: 2 or 3 - the number of classes held out together.
        holdout_class: optional class name to filter which split groups are
            reported.
    """
    return t.get_cluster_summary(k=k, holdout_class=holdout_class)


@mcp.tool()
def compare_to_baseline(holdout_class: str, metric: str) -> dict:
    """Compare the distance detector to the MSP baseline for a given holdout
    class and metric.

    Args:
        holdout_class: one of basophil, eosinophil, erythroblast, ig,
            lymphocyte, monocyte, neutrophil, platelet.
        metric: one of auroc, auprc, recall, precision, f1, mcc,
            specificity, fpr, or (for f1/precision/recall) matched against
            the classification per-class table too.
    """
    return t.compare_to_baseline(holdout_class, metric)


if __name__ == "__main__":
    mcp.run(transport="stdio")
