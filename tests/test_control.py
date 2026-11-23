"""Scripted-actor checks for the control gate. No live model and no weight write."""
from threading import Event

from adaptive_harness.control import (
    AuditLog, Decision, HarnessSession, IterationBudget, ModelResponse, RefinementLedger,
    ToolCall, ToolSpec, ToolRegistry, weight_edit_tools,
)
from adaptive_harness.weight_space import CoefficientBand, ModelSignature, ProbeReceipt


def qwen():
    return ModelSignature("Qwen2.5-0.5B-Instruct", 24, value_bias=True, donor=False)


def sealed_band():
    band = CoefficientBand("inject")
    held = ProbeReceipt(concept_hit=1.0, fact_ok=True, collapse=False, fluent=True)
    band.record(16, held, phase="train")
    band.record(16, held, phase="validation")
    band.select_and_freeze()
    return band


def scripted(steps):
    queue = list(steps)

    def model(_messages, _names):
        if not queue:
            return ModelResponse("investigation finished")
        return queue.pop(0)

    return model


def commit(coefficient=16, target="network"):
    return ToolCall("commit_weights", {
        "target": target, "kind": "inject", "program": "layers.14-19 inject wedding 16",
        "coefficient": coefficient,
    })


def session(model, *, budget=12, cost=1.0, approver=None, tools=None, abort=None, can_delegate=True,
            keep_iterations=0, cost_quote=None, completion_gate=None, workspace_fingerprint=None):
    applied, probes = [], []
    registry = tools if tools is not None else weight_edit_tools(
        signature=qwen(), band=sealed_band(), applied=applied, probes=probes)
    agent = HarnessSession(
        model=model, registry=registry, budget=IterationBudget(budget),
        approver=approver or (lambda _tool, _args: True), max_cost_usd=cost, abort=abort,
        can_delegate=can_delegate, keep_iterations=keep_iterations, cost_quote=cost_quote,
        completion_gate=completion_gate, workspace_fingerprint=workspace_fingerprint,
    )
    return agent, applied


def test_normal_run_asks_before_commit_and_keeps_a_valid_chain():
    agent, applied = session(scripted([
        ModelResponse("Reading the target.", (ToolCall("observe_model", {}), ToolCall("probe_edit", {"program": "layers.14-19 inject wedding 16"})), 0.02),
        ModelResponse("The probe held.", (commit(),), 0.02),
        ModelResponse("investigation finished", cost_usd=0.02),
    ]))
    events = list(agent.investigate("inject wedding"))
    assert applied[0]["target"] == "network"
    assert applied[0]["program"] == "layers.14-19 inject wedding 16"
    assert applied[0]["coefficient"] == 16
    assert any(entry["event"] == "weight_commit" and entry["program_sha256"] == applied[0]["program_sha256"]
               for entry in agent.audit.entries())
    assert agent.permission_denials == []
    assert agent.budget.used == 3
    assert agent.audit.verify()
    assert events[-1].kind == "complete"
    assert all(message["untrusted"] for message in agent.messages if message["role"] == "tool")


def test_injected_commit_is_refused_without_running_the_handler():
    agent, applied = session(scripted([
        ModelResponse("Narration says to commit immediately.", (commit(),), 0.01),
        ModelResponse("Commit denied. Recommend manual review.", cost_usd=0.01),
    ]), approver=lambda _tool, _args: False)
    list(agent.investigate("inject wedding"))
    assert applied == []
    assert agent.handler_calls == []
    assert len(agent.permission_denials) == 1
    assert agent.audit.verify()


def test_shared_budget_stops_root_and_subagent_at_the_same_cap():
    child_calls = [ModelResponse("observe", (ToolCall("observe_model", {}),), 0.0) for _ in range(8)]
    root_calls = [ModelResponse("delegate", (ToolCall("delegate", {"task": "observe the target"}),), 0.0)]
    root_calls.extend(child_calls)
    agent, _applied = session(scripted(root_calls), budget=6)
    events = list(agent.investigate("inspect"))
    assert agent.budget.used == 6
    assert [event.kind for event in events if event.kind == "budget"] == ["budget", "budget"]
    assert {event.agent for event in events if event.kind == "budget"} == {"root/sub", "root"}


def test_cost_ceiling_is_checked_before_the_next_call_and_can_overshoot_by_one():
    steps = [ModelResponse("step", (ToolCall("observe_model", {}),), 0.02) for _ in range(6)]
    agent, _applied = session(scripted(steps), cost=0.07)
    events = list(agent.investigate("inspect"))
    assert agent.budget.used == 4
    assert round(agent.usage.cost_usd, 2) == 0.08
    assert events[-1].kind == "budget"
    assert "cost ceiling" in events[-1].text


def test_bad_arguments_out_of_band_coefficient_and_unknown_tool_do_not_run_handlers():
    agent, applied = session(scripted([
        ModelResponse("bad call", (
            ToolCall("commit_weights", {"target": "network", "kind": "inject", "program": "layers.14-19 inject wedding 16", "coefficient": "high"}),
            commit(24),
            ToolCall("rename_actor", {}),
        ), 0.01),
        ModelResponse("done", cost_usd=0.0),
    ]))
    list(agent.investigate("inspect"))
    kinds = [entry["event"] for entry in agent.audit.entries()]
    assert kinds == ["start", "tool_invalid", "tool_denied", "tool_unknown", "complete"]
    assert "tool_call" not in kinds
    assert applied == []
    assert agent.handler_calls == []


def test_an_edited_audit_entry_fails_verification():
    agent, _applied = session(scripted([ModelResponse("done", cost_usd=0.0)]))
    list(agent.investigate("inspect"))
    assert agent.audit.verify()
    agent.audit._entries[0]["task"] = "rewritten"
    assert agent.audit.verify() is False


def test_abort_during_the_model_call_finishes_the_in_flight_tool_then_stops():
    abort = Event()

    def model(_messages, _names):
        if not abort.is_set():
            abort.set()
            return ModelResponse("one probe", (ToolCall("observe_model", {}),), 0.01)
        raise AssertionError("the next model call must not start")

    agent, _applied = session(model, abort=abort)
    events = list(agent.investigate("inspect"))
    assert agent.handler_calls == ["observe_model"]
    assert events[-1].kind == "aborted"


def test_approval_timeout_fails_closed():
    def asleep(_tool, _args):
        raise TimeoutError("approver did not answer")

    agent, applied = session(scripted([
        ModelResponse("commit", (commit(),), 0.01),
        ModelResponse("stopped", cost_usd=0.0),
    ]), approver=asleep)
    list(agent.investigate("inject wedding"))
    assert applied == []
    assert len(agent.permission_denials) == 1
    assert agent.permission_denials[0]["decision"] == Decision.ASK.value


def test_subagent_cannot_commit_or_delegate():
    calls = []

    def model(_messages, names):
        calls.append(tuple(names))
        if len(calls) == 1:
            return ModelResponse("delegate", (ToolCall("delegate", {"task": "commit the wedding inject"}),), 0.0)
        if len(calls) == 2:
            return ModelResponse("child tries", (
                commit(),
                ToolCall("delegate", {"task": "again"}),
            ), 0.0)
        return ModelResponse("child finished", cost_usd=0.0)

    agent, applied = session(model, budget=8)
    list(agent.investigate("inject wedding"))
    assert "commit_weights" not in calls[1]
    assert applied == []
    assert any(entry["event"] == "tool_unknown" and entry["tool"] == "commit_weights" for entry in agent.audit.entries())
    assert any(entry["event"] == "tool_unknown" and entry["tool"] == "delegate" for entry in agent.audit.entries())


def test_read_only_batch_is_scheduled_before_a_destructive_call():
    order = []
    registry = ToolRegistry([
        ToolSpec("look", "read", {}, lambda _args: order.append("look") or "ok", read_only=True, concurrency_safe=True),
        ToolSpec("write", "mutate", {}, lambda _args: order.append("write") or "ok", destructive=True),
    ])
    agent, _applied = session(scripted([
        ModelResponse("both", (ToolCall("write", {}), ToolCall("look", {})), 0.0),
        ModelResponse("done", cost_usd=0.0),
    ]), tools=registry, approver=lambda _tool, _args: True)
    list(agent.investigate("inspect"))
    assert order == ["look", "write"]


def test_actor_target_and_replay_never_apply_twice():
    agent, applied = session(scripted([
        ModelResponse("actor", (commit(target="actor"),), 0.0),
        ModelResponse("network", (commit(),), 0.0),
        ModelResponse("again", (commit(),), 0.0),
        ModelResponse("done", cost_usd=0.0),
    ]))
    list(agent.investigate("inject wedding"))
    assert len(applied) == 1
    assert applied[0]["target"] == "network"
    assert agent.registry.commit_ledger.seen(applied[0]["program_sha256"])
    assert sum(1 for entry in agent.audit.entries() if entry["event"] == "weight_commit") == 1
    assert sum(1 for row in agent.permission_denials if row["tool"] == "commit_weights") == 2


def test_parent_keeps_reserved_iterations_from_the_subagent():
    def model(_messages, names):
        if "commit_weights" not in names:
            return ModelResponse("observe", (ToolCall("observe_model", {}),), 0.0)
        if not model.sent:
            model.sent = True
            return ModelResponse("delegate", (ToolCall("delegate", {"task": "observe"}),), 0.0)
        return ModelResponse("parent still running", cost_usd=0.0)

    model.sent = False
    agent, _applied = session(model, budget=6, keep_iterations=2)
    events = list(agent.investigate("inspect"))
    assert agent.budget.used == 5
    assert agent.budget.reserved == 0
    assert events[-1].kind == "complete"
    assert any(event.agent == "root/sub" and event.kind == "budget" for event in events)


def test_a_quote_above_the_ceiling_does_not_call_the_actor():
    called = []

    def model(_messages, _names):
        called.append(1)
        return ModelResponse("should not run", cost_usd=0.05)

    agent, _applied = session(model, cost=0.07, cost_quote=lambda: 0.05)
    agent.usage.add(0.04)
    events = list(agent.investigate("inspect"))
    assert called == []
    assert agent.budget.used == 0
    assert events[-1].text.startswith("quoted cost")


def test_cost_is_attributed_to_the_agent_that_spent_it():
    def model(_messages, _names):
        return ModelResponse("done", cost_usd=0.03)

    agent, _applied = session(model, cost=1.0)
    list(agent.investigate("inspect"))
    assert agent.usage.by_agent == {"root": 0.03}


def test_a_rebuilt_log_fails_the_anchor_check():
    agent, _applied = session(scripted([ModelResponse("done", cost_usd=0.0)]))
    list(agent.investigate("inspect"))
    anchored = agent.audit.anchor()
    assert agent.audit.verify_anchor()
    replacement = AuditLog()
    replacement.append("start", agent="root", task="rewritten")
    agent.audit._entries = replacement.entries()
    assert agent.audit.verify()
    assert agent.audit.head() != anchored
    assert agent.audit.verify_anchor() is False


def test_a_budget_stop_is_not_success_and_the_gate_must_pass():
    calls = {"n": 0}

    def gate(_session):
        calls["n"] += 1
        return calls["n"] >= 2, "verifier output " + ("ok" if calls["n"] >= 2 else "missing test")

    agent, _applied = session(scripted([
        ModelResponse("not yet", cost_usd=0.0),
        ModelResponse("ready", cost_usd=0.0),
    ]), completion_gate=gate)
    events = list(agent.investigate("finish the task"))
    assert [event.kind for event in events if event.kind in ("gate", "complete")] == ["gate", "complete"]
    assert agent.succeeded is True
    assert agent.stop_reason == "gate"
    assert calls["n"] == 2

    def always_fail(_session):
        return False, "still failing"

    stopped, _applied = session(scripted([
        ModelResponse("again", cost_usd=0.0),
        ModelResponse("again", cost_usd=0.0),
    ]), budget=2, completion_gate=always_fail)
    endings = list(stopped.investigate("finish the task"))
    assert endings[-1].kind == "budget"
    assert stopped.succeeded is False
    assert stopped.stop_reason == "iterations"
    assert "complete" not in [event.kind for event in endings]


def test_an_unchanged_workspace_does_not_rerun_the_failed_gate():
    state = {"mark": "same", "runs": 0}

    def gate(_session):
        state["runs"] += 1
        return False, "failed"

    def model(_messages, _names):
        return ModelResponse("done", cost_usd=0.0)

    agent, _applied = session(model, budget=4, completion_gate=gate, workspace_fingerprint=lambda: state["mark"])
    events = []
    for event in agent.investigate("finish the task"):
        events.append(event)
        if sum(1 for item in events if item.kind == "gate") == 2:
            state["mark"] = "edited"
        if sum(1 for item in events if item.kind == "gate") == 3:
            break
    kinds = [event.outcome for event in events if event.kind == "gate"]
    assert kinds == ["failed", "skipped", "failed"]
    assert state["runs"] == 2
    assert agent.succeeded is False


def test_companion_edits_roll_back_and_test_receipts_are_refused():
    ledger = RefinementLedger()
    try:
        ledger.propose(phase="test", field="skill", value="retry", trigger="failed twice",
                       receipt_holds=True, receipt_phase="test")
    except RuntimeError as exc:
        assert "Test receipts" in str(exc)
    else:
        raise AssertionError("a test receipt was admitted")
    first = ledger.propose(phase="train", field="skill", value="retry the holding probe",
                           trigger="the same probe failed twice", receipt_holds=True, receipt_phase="train")
    ledger.admit(first["id"])
    assert ledger.companion["skill"] == "retry the holding probe"
    second = ledger.propose(phase="validation", field="memory", value="pad must be free before the pick",
                            trigger="validation receipt held", receipt_holds=True, receipt_phase="validation")
    ledger.admit(second["id"])
    rolled = ledger.rollback(second["id"])
    assert "memory" not in ledger.companion
    assert rolled["restored"] == first["id"]
    assert ledger.head == first["id"]
    ledger.freeze()
    for call in (
        lambda: ledger.freeze(),
        lambda: ledger.rollback(first["id"]),
        lambda: ledger.propose(phase="train", field="skill", value="again", trigger="later",
                               receipt_holds=True, receipt_phase="train"),
    ):
        try:
            call()
        except RuntimeError as exc:
            assert "frozen" in str(exc)
        else:
            raise AssertionError("a frozen companion edit was changed")


def test_compaction_replaces_messages_and_keeps_the_audit_prefix():
    agent, _applied = session(scripted([
        ModelResponse("read", (ToolCall("observe_model", {}),), 0.0),
        ModelResponse("done", cost_usd=0.0),
    ]))
    list(agent.investigate("inspect"))
    before = [entry["hash"] for entry in agent.audit.entries()]
    assert len(agent.messages) > 1
    agent.compact_messages("The target was observed. The actor's weights stayed fixed.")
    after = [entry["hash"] for entry in agent.audit.entries()]
    assert after[:len(before)] == before
    assert after[-1] != before[-1]
    assert agent.audit.verify()
    assert len(agent.messages) == 1
    assert agent.messages[0]["compacted_from"] > 1
    assert agent.audit.entries()[-1]["event"] == "compact"
