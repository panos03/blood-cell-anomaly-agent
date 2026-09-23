"""Evaluation harness for the blood-cell anomaly detection agent.

Runs eval/questions.json through two conditions:
  - "with_tools":    the LangGraph ReAct agent (agent/agent.py), which can
                      call get_flagged_cells / get_cluster_summary /
                      compare_to_baseline.
  - "without_tools":  the same Gemini model with no tool access, answering
                      from its own knowledge alone.

For each question it records the model's final answer, which tools (if any)
were called, and a heuristic grading verdict, then writes a per-question
table plus summary statistics to eval/results.md.

Usage:
    python -m eval.run_eval [--limit N] [--model gemini-3.7-flash]

Requires GOOGLE_API_KEY (a free key from https://aistudio.google.com) to be
set - every question costs a real API call in each condition.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.agent import build_agent

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.md"

_DECLINE_PATTERNS = re.compile(
    r"\b(i don't have|i do not have|no tool|not available|cannot answer|"
    r"can't answer|don't know|do not know|no access|unable to (find|answer)|"
    r"not covered|outside (the )?scope|not exposed)\b",
    re.IGNORECASE,
)


@dataclass
class QuestionResult:
    # One row per question: both conditions' answers, verdicts, and which
    # tools got called along the way.
    id: str
    category: str
    question: str
    expected_answer: object
    with_tools_answer: str = ""
    with_tools_tools_called: list = field(default_factory=list)
    with_tools_verdict: str = ""
    without_tools_answer: str = ""
    without_tools_verdict: str = ""


def _load_questions(limit: int | None) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return questions[:limit] if limit else questions


def _extract_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts)


def _run_with_tools(agent, question: str) -> tuple[str, list[str]]:
    # Condition 1: ask the real agent, then read back which tools it called
    # from the transcript (not just its final answer).
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]
    tools_called = [
        call["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    ]
    final_text = _extract_text(messages[-1])
    return final_text, tools_called


def _run_without_tools(model, question: str) -> str:
    # Condition 2 (the baseline): same question, plain Gemini, no tools.
    # Shows what the agent's tool access is actually buying you.
    response = model.invoke(
        [
            HumanMessage(
                content=(
                    "Answer the following question about a blood-cell "
                    "anomaly detection research project's experiment "
                    "results. If you don't know the specific numbers, say "
                    "so rather than guessing.\n\nQuestion: " + question
                )
            )
        ]
    )
    return _extract_text(response)


def _numbers_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def _grade_numeric(answer_text: str, expected: float, tolerance: float) -> str:
    for n in _numbers_in(answer_text):
        if abs(n - expected) <= max(tolerance, 1e-9):
            return "correct"
    return "incorrect"


def _grade_string(answer_text: str, expected: str) -> str:
    return "correct" if expected.lower() in answer_text.lower() else "incorrect"


def _grade_string_all(answer_text: str, expected: list[str]) -> str:
    lowered = answer_text.lower()
    return "correct" if all(e.lower() in lowered for e in expected) else "incorrect"


def _grade_unsupported(answer_text: str) -> str:
    return "declined" if _DECLINE_PATTERNS.search(answer_text) else "hallucinated"


def _grade(question: dict, answer_text: str) -> str:
    # Picks the right grader for this question's answer_type (numeric,
    # string, string_all, or unsupported). See questions.json.
    answer_type = question["answer_type"]
    if answer_type == "numeric":
        return _grade_numeric(answer_text, question["expected_answer"], question.get("tolerance", 0))
    if answer_type == "string":
        return _grade_string(answer_text, question["expected_answer"])
    if answer_type == "string_all":
        return _grade_string_all(answer_text, question["expected_answer"])
    if answer_type == "unsupported":
        return _grade_unsupported(answer_text)
    raise ValueError(f"Unknown answer_type: {answer_type}")


def _tool_call_correct(question: dict, tools_called: list[str]) -> bool | None:
    # Separate from answer correctness: did the agent call the right tool at
    # all, regardless of whether its final answer ended up right?
    expected = question["expected_tools"]
    if not expected:
        return None  # not applicable (unsupported questions expect no tool call)
    return any(t in tools_called for t in expected)


def run(limit: int | None, model_id: str) -> list[QuestionResult]:
    # Main loop: for every question, run both conditions and grade each one.
    questions = _load_questions(limit)
    agent = build_agent()
    plain_model = ChatGoogleGenerativeAI(model=model_id, max_output_tokens=1024)

    results = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")

        wt_answer, tools_called = _run_with_tools(agent, q["question"])
        wt_verdict = _grade(q, wt_answer)

        wot_answer = _run_without_tools(plain_model, q["question"])
        wot_verdict = _grade(q, wot_answer)

        results.append(
            QuestionResult(
                id=q["id"],
                category=q["category"],
                question=q["question"],
                expected_answer=q["expected_answer"],
                with_tools_answer=wt_answer,
                with_tools_tools_called=tools_called,
                with_tools_verdict=wt_verdict,
                without_tools_answer=wot_answer,
                without_tools_verdict=wot_verdict,
            )
        )
    return results, questions


def _summarize(results: list[QuestionResult], questions: list[dict]) -> str:
    # Turns the 40 raw QuestionResults into the aggregate report (accuracy
    # per condition, tool-call rate, failure modes) written to results.md.
    total = len(results)
    supported = [r for r in results if r.category != "unsupported"]
    unsupported = [r for r in results if r.category == "unsupported"]

    wt_correct = sum(1 for r in supported if r.with_tools_verdict == "correct")
    wot_correct = sum(1 for r in supported if r.without_tools_verdict == "correct")

    tool_call_checks = []
    q_by_id = {q["id"]: q for q in questions}
    for r in results:
        ok = _tool_call_correct(q_by_id[r.id], r.with_tools_tools_called)
        if ok is not None:
            tool_call_checks.append(ok)
    tool_call_rate = sum(tool_call_checks) / len(tool_call_checks) if tool_call_checks else 0.0

    declined = sum(1 for r in unsupported if r.with_tools_verdict == "declined")
    hallucinated = sum(1 for r in unsupported if r.with_tools_verdict == "hallucinated")

    failure_modes = []
    for r in supported:
        if r.with_tools_verdict != "correct":
            expected_tools = q_by_id[r.id]["expected_tools"]
            called = r.with_tools_tools_called
            if not called:
                mode = "no tool called"
            elif not any(t in called for t in expected_tools):
                mode = f"wrong tool called ({called})"
            else:
                mode = "tool called correctly but final answer still wrong"
            failure_modes.append((r.id, mode))

    lines = [
        "# Evaluation Results",
        "",
        f"Questions: {total} ({len(supported)} answerable via tools, {len(unsupported)} intentionally unsupported)",
        "",
        "## Headline metrics",
        "",
        f"- Accuracy **with tools** (agent): {wt_correct}/{len(supported)} ({wt_correct / len(supported):.0%})",
        f"- Accuracy **without tools** (plain Gemini): {wot_correct}/{len(supported)} ({wot_correct / len(supported):.0%})",
        f"- Correct tool call rate: {sum(tool_call_checks)}/{len(tool_call_checks)} ({tool_call_rate:.0%})",
        f"- On unsupported questions: {declined}/{len(unsupported)} declined appropriately, {hallucinated}/{len(unsupported)} hallucinated an answer",
        "",
        "## Failure modes (supported questions the agent got wrong)",
        "",
    ]
    if failure_modes:
        for qid, mode in failure_modes:
            lines.append(f"- `{qid}`: {mode}")
    else:
        lines.append("- None")
    lines.append("")

    lines += ["## Per-category accuracy (with tools)", ""]
    categories = sorted(set(r.category for r in supported))
    for cat in categories:
        cat_results = [r for r in supported if r.category == cat]
        correct = sum(1 for r in cat_results if r.with_tools_verdict == "correct")
        lines.append(f"- {cat}: {correct}/{len(cat_results)} ({correct / len(cat_results):.0%})")
    lines.append("")

    lines += [
        "## Per-question detail",
        "",
        "| ID | Category | With tools | Tools called | Without tools |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.category} | {r.with_tools_verdict} | "
            f"{', '.join(r.with_tools_tools_called) or '-'} | {r.without_tools_verdict} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    # CLI entry point: parse args, run the eval, write eval/results.md.
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument(
        "--model", type=str, default=None, help="Override the model id used for both conditions (default: gemini-3.7-flash)"
    )
    args = parser.parse_args()

    import os

    model_id = args.model or os.environ.get("BLOOD_CELL_AGENT_MODEL", "gemini-3.7-flash")

    results, questions = run(args.limit, model_id)
    report = _summarize(results, questions)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
