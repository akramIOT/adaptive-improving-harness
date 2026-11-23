"""Scripted long run. No live model and no weight write.

One session fails the external gate once and then passes. A second session
exhausts its iteration budget while the gate is still failing, so that stop
is not success. A test receipt cannot edit the companion configuration.
Compaction replaces the actor's messages and leaves the audit prefix in place.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_harness.control import HarnessSession, IterationBudget, ModelResponse, ToolRegistry


def _session(*, budget, gate):
    def model(_messages, _names):
        return ModelResponse("finished the visible step", cost_usd=0.0)

    return HarnessSession(
        model=model,
        registry=ToolRegistry([]),
        budget=IterationBudget(budget),
        approver=lambda _tool, _args: False,
        completion_gate=gate,
        max_cost_usd=1.0,
    )


def main():
    answers = {"n": 0}

    def passing_gate(_session):
        answers["n"] += 1
        return answers["n"] >= 2, "held" if answers["n"] >= 2 else "receipt missing"

    passed = _session(budget=4, gate=passing_gate)
    passed_events = list(passed.investigate("finish the task"))
    before = [entry["hash"] for entry in passed.audit.entries()]
    passed.compact_messages("Gate passed. Actor weights stayed fixed.")
    refused = None
    try:
        passed.refinements.propose(
            phase="test", field="skill", value="retry", trigger="test score rose",
            receipt_holds=True, receipt_phase="test",
        )
    except RuntimeError as exc:
        refused = str(exc)

    stopped = _session(budget=1, gate=lambda _session: (False, "still failing"))
    stopped_events = list(stopped.investigate("finish the task"))
    print(json.dumps({
        "scope": "Scripted long run; no live model and no weight write",
        "gate_run": {
            "succeeded": passed.succeeded,
            "stop_reason": passed.stop_reason,
            "outcomes": [event.kind for event in passed_events],
        },
        "budget_stop": {
            "succeeded": stopped.succeeded,
            "stop_reason": stopped.stop_reason,
            "outcomes": [event.kind for event in stopped_events],
        },
        "audit_prefix_kept": [entry["hash"] for entry in passed.audit.entries()][:len(before)] == before,
        "messages_after_compaction": len(passed.messages),
        "test_receipt_refused": refused,
    }, indent=2))


if __name__ == "__main__":
    main()
