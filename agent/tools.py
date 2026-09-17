"""Parsers over the static result files copied into data/.

No ML inference happens here - every function just reads and regexes a
plain-text or CSV file produced by the original FYP pipeline.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HOLDOUT_CLASSES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


class ToolError(ValueError):
    """Raised when a tool cannot answer from the available result files."""


def _read(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"No result file at {path.relative_to(DATA_DIR.parent)}")
    return path.read_text(encoding="utf-8")


def _check_holdout_class(holdout_class: str) -> str:
    holdout_class = holdout_class.strip().lower()
    if holdout_class not in HOLDOUT_CLASSES:
        raise ToolError(
            f"Unknown holdout class '{holdout_class}'. "
            f"Valid classes: {', '.join(HOLDOUT_CLASSES)}"
        )
    return holdout_class


# ---------------------------------------------------------------------------
# get_flagged_cells
# ---------------------------------------------------------------------------

# Matches rows like "  Recall                            0.8555         0.5690"
# or "  Specificity (TNR)                 0.9534         0.9471" or integer TP/FP/TN/FN rows.
_METRIC_ROW = re.compile(
    r"^\s*(AUROC|AUPRC|Recall|Precision|F1|MCC|Specificity \(TNR\)|FPR|TP|FP|TN|FN)"
    r"\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
)


def _parse_dist_msp_metrics(text: str) -> dict:
    """Parse the 'Detection metrics at calibrated threshold' table into
    {metric_name: {"dist": float, "msp": float}}.
    """
    metrics: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = _METRIC_ROW.match(line)
        if m:
            name, dist_val, msp_val = m.groups()
            metrics[name.strip()] = {"dist": float(dist_val), "msp": float(msp_val)}
    return metrics


def get_flagged_cells(holdout_class: str) -> dict:
    """Return TP/FP/FN/TN/precision/recall/F1 for the distance detector and the
    MSP baseline on a given holdout class.

    Source: data/experiments/anomaly_detection/holdout_{class}/summary.txt
    """
    holdout_class = _check_holdout_class(holdout_class)
    path = (
        DATA_DIR
        / "experiments"
        / "anomaly_detection"
        / f"holdout_{holdout_class}"
        / "summary.txt"
    )
    text = _read(path)
    metrics = _parse_dist_msp_metrics(text)

    required = {"TP", "FP", "TN", "FN", "Precision", "Recall", "F1"}
    missing = required - metrics.keys()
    if missing:
        raise ToolError(f"Could not parse metrics {missing} from {path}")

    def detector_block(key: str) -> dict:
        return {
            "tp": int(metrics["TP"][key]),
            "fp": int(metrics["FP"][key]),
            "tn": int(metrics["TN"][key]),
            "fn": int(metrics["FN"][key]),
            "precision": metrics["Precision"][key],
            "recall": metrics["Recall"][key],
            "f1": metrics["F1"][key],
        }

    return {
        "holdout_class": holdout_class,
        "dist": detector_block("dist"),
        "msp": detector_block("msp"),
    }


# ---------------------------------------------------------------------------
# get_cluster_summary
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"===\s*(\d+)-class holdout \(oracle k = \d+\)\s*===(.*?)(?=\n===|\Z)",
    re.DOTALL,
)
_SPLIT_LINE_RE = re.compile(
    r"Split\s+\d+:\s*(\[.*?\])\s+flagged\s+(\d+)/(\d+)\s+holdout\s+(\d+)/(\d+)"
)
_STAT_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+)\s*±\s*([\d.]+)\s*$")


def _parse_cluster_section(section_text: str) -> dict:
    splits = []
    for m in _SPLIT_LINE_RE.finditer(section_text):
        classes = ast.literal_eval(m.group(1))
        splits.append(
            {
                "classes": [c.lower() for c in classes],
                "flagged": int(m.group(2)),
                "flagged_total": int(m.group(3)),
                "holdout": int(m.group(4)),
                "holdout_total": int(m.group(5)),
            }
        )

    lines = section_text.splitlines()
    anomaly_flagging = {}
    methods: dict[str, dict[str, dict[str, float]]] = {}
    current_method = None
    in_flagging_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("Anomaly flagging"):
            in_flagging_block = True
            continue
        if in_flagging_block:
            m = _STAT_LINE_RE.match(line)
            if m:
                anomaly_flagging[m.group(1).lower()] = {
                    "mean": float(m.group(2)),
                    "std": float(m.group(3)),
                }
                continue
            if stripped == "":
                in_flagging_block = False

        # Method header: a two-space-indented line followed by a "Metric ... mean" line
        if (
            line.startswith("  ")
            and not line.startswith("   ")
            and stripped
            and i + 1 < len(lines)
            and "Metric" in lines[i + 1]
            and "mean" in lines[i + 1]
        ):
            current_method = stripped
            methods[current_method] = {}
            continue

        if current_method:
            m = _STAT_LINE_RE.match(line)
            if m:
                methods[current_method][m.group(1)] = {
                    "mean": float(m.group(2)),
                    "std": float(m.group(3)),
                }

    return {
        "splits": splits,
        "anomaly_flagging": anomaly_flagging,
        "clustering_methods": methods,
    }


def get_cluster_summary(k: int, holdout_class: str | None = None) -> dict:
    """Return per-clustering-method ARI/AMI/PURITY (mean +/- std) for the k-class
    holdout scenario (k=2 or k=3), optionally filtered to splits that held out a
    given class.

    Source: data/experiments/cluster_discovery/summary_metrics.txt

    Note: the underlying file reports clustering metrics aggregated across all
    splits for a given k, not per individual class - so `holdout_class` only
    narrows which split groups are listed, it does not change the reported
    ARI/AMI/PURITY numbers.
    """
    if k not in (2, 3):
        raise ToolError(f"k must be 2 or 3, got {k}")

    path = DATA_DIR / "experiments" / "cluster_discovery" / "summary_metrics.txt"
    text = _read(path)

    section_match = None
    for m in _SECTION_RE.finditer(text):
        if int(m.group(1)) == k:
            section_match = m
            break
    if section_match is None:
        raise ToolError(f"Could not find a {k}-class holdout section in {path}")

    parsed = _parse_cluster_section(section_match.group(2))

    result = {
        "k": k,
        "anomaly_flagging": parsed["anomaly_flagging"],
        "clustering_methods": parsed["clustering_methods"],
    }

    if holdout_class is not None:
        holdout_class = _check_holdout_class(holdout_class)
        matching_splits = [
            s for s in parsed["splits"] if holdout_class in s["classes"]
        ]
        if not matching_splits:
            raise ToolError(
                f"No {k}-class holdout splits include class '{holdout_class}'"
            )
        result["holdout_class"] = holdout_class
        result["matching_splits"] = matching_splits

    return result


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------

_AD_METRIC_ALIASES = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "mcc": "MCC",
    "specificity": "Specificity (TNR)",
    "tnr": "Specificity (TNR)",
    "fpr": "FPR",
}


def _cls_pattern(holdout_class: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(holdout_class)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        re.IGNORECASE | re.MULTILINE,
    )


_LOWER_IS_BETTER = {"fpr"}


def _winner(dist_val: float, msp_val: float, metric_norm: str = "") -> str:
    if metric_norm in _LOWER_IS_BETTER:
        dist_val, msp_val = -dist_val, -msp_val
    if dist_val > msp_val:
        return "dist"
    if msp_val > dist_val:
        return "msp"
    return "tie"


def compare_to_baseline(holdout_class: str, metric: str) -> dict:
    """Compare the distance detector to the MSP baseline for a given holdout
    class and metric, looking in both the anomaly_detection per-holdout summary
    (AUROC/AUPRC/Recall/Precision/F1/MCC/Specificity/FPR) and, when the metric
    is F1/Precision/Recall, the classification per-class summary.

    "winner" accounts for metric direction: higher is better for every metric
    except FPR, where lower is better.

    Sources: data/experiments/anomaly_detection/holdout_{class}/summary.txt
             data/experiments/classification/summary.txt
    """
    holdout_class = _check_holdout_class(holdout_class)
    metric_norm = metric.strip().lower()

    result: dict = {
        "holdout_class": holdout_class,
        "metric": metric_norm,
        "anomaly_detection": None,
        "classification": None,
    }

    if metric_norm in _AD_METRIC_ALIASES:
        row_name = _AD_METRIC_ALIASES[metric_norm]
        ad_path = (
            DATA_DIR
            / "experiments"
            / "anomaly_detection"
            / f"holdout_{holdout_class}"
            / "summary.txt"
        )
        ad_metrics = _parse_dist_msp_metrics(_read(ad_path))
        if row_name in ad_metrics:
            dist_val = ad_metrics[row_name]["dist"]
            msp_val = ad_metrics[row_name]["msp"]
            result["anomaly_detection"] = {
                "dist": dist_val,
                "msp": msp_val,
                "difference_dist_minus_msp": round(dist_val - msp_val, 4),
                "winner": _winner(dist_val, msp_val, metric_norm),
            }

    if metric_norm in {"f1", "precision", "recall"}:
        cls_path = DATA_DIR / "experiments" / "classification" / "summary.txt"
        cls_text = _read(cls_path)
        m = _cls_pattern(holdout_class).search(cls_text)
        if m:
            dist_f1, msp_f1, dist_p, msp_p, dist_r, msp_r = (
                float(v) for v in m.groups()
            )
            values = {
                "f1": (dist_f1, msp_f1),
                "precision": (dist_p, msp_p),
                "recall": (dist_r, msp_r),
            }
            dist_val, msp_val = values[metric_norm]
            result["classification"] = {
                "dist": dist_val,
                "msp": msp_val,
                "difference_dist_minus_msp": round(dist_val - msp_val, 4),
                "winner": _winner(dist_val, msp_val),
            }

    if result["anomaly_detection"] is None and result["classification"] is None:
        raise ToolError(
            f"Metric '{metric}' not found for holdout class '{holdout_class}' "
            "in anomaly_detection or classification summaries."
        )

    return result
