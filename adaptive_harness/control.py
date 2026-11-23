"""Control, budget, and audit around a tool-using actor.

The actor proposes a call. This module decides whether it may run. The order
of a call is fixed: the tool must exist, its arguments must match the declared
types, a tool-specific rule may refuse it, a destructive call waits for
approval, and only then does the handler run. A missing approval, including a
timeout, is a refusal.

The iteration counter and the cost total are shared with a subagent. The
subagent sees only read-only tools and cannot delegate. Every ending, including
a refusal, is appended to a hash-chained log. A budget stop is not success.
When a completion gate is set, the actor's last message is not success either;
the gate must pass. Companion edits are versioned, admitted only on train or
validation receipts, and frozen once. Compaction replaces the actor's messages
and leaves the earlier audit entries in place. The tools below are weight-space
actions for this repository. They do not call a live model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from threading import Event, Lock
from typing import Any, Iterator


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ToolSpec:
    """Typed tool contract. The harness gates on these fields, not on handler code."""

    name: str
    description: str
    schema: Mapping[str, type]
    handler: Callable[[Mapping[str, Any]], str]
    read_only: bool = False
    destructive: bool = False
    concurrency_safe: bool = False
    max_result_chars: int = 400
    check_fn: Callable[[], bool] = field(default=lambda: True)
    check_permissions: Callable[[Mapping[str, Any]], Decision | None] | None = None

    def __post_init__(self):
        if self.read_only and self.destructive:
            raise ValueError(f"{self.name} cannot be both read-only and destructive")
        if self.concurrency_safe and not self.read_only:
            raise ValueError(f"{self.name} is concurrency-safe only when it is read-only")


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    cost_usd: float = 0.0


@dataclass(frozen=True)
class EventRecord:
    kind: str
    text: str
    agent: str
    outcome: str | None = None


class ToolRegistry:
    def __init__(self, specs: Sequence[ToolSpec]):
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique")
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def read_only_view(self) -> "ToolRegistry":
        return ToolRegistry([spec for spec in self._specs.values() if spec.read_only])


class IterationBudget:
    """One counter for a whole agent tree.

    Reserved units stay with the caller. A subagent drawing on this budget
    cannot spend them.
    """

    def __init__(self, max_total: int):
        if type(max_total) is not int or max_total < 1:
            raise ValueError("Iteration budget must be a positive integer")
        self.max_total = max_total
        self.used = 0
        self._reserved = 0
        self._lock = Lock()

    @property
    def reserved(self) -> int:
        return self._reserved

    def remaining(self) -> int:
        with self._lock:
            return self.max_total - self.used - self._reserved

    def reserve(self, units: int) -> int:
        if type(units) is not int or units < 0:
            raise ValueError("Reserve must be a nonnegative integer")
        with self._lock:
            taken = min(units, self.max_total - self.used - self._reserved)
            self._reserved += taken
            return taken

    def release(self, units: int) -> None:
        if type(units) is not int or units < 0 or units > self._reserved:
            raise ValueError("Release exceeds the reserved iterations")
        with self._lock:
            self._reserved -= units

    def consume(self) -> bool:
        with self._lock:
            if self.used + self._reserved >= self.max_total:
                return False
            self.used += 1
            return True


class Usage:
    def __init__(self):
        self.cost_usd = 0.0
        self.by_agent: dict[str, float] = {}
        self._lock = Lock()

    def add(self, cost_usd: float, *, agent: str = "root") -> None:
        if type(cost_usd) is bool or not isinstance(cost_usd, (int, float)) or cost_usd < 0:
            raise ValueError("Cost must be a nonnegative number")
        if not isinstance(agent, str) or not agent:
            raise ValueError("Cost needs an agent name")
        with self._lock:
            self.cost_usd += float(cost_usd)
            self.by_agent[agent] = self.by_agent.get(agent, 0.0) + float(cost_usd)


class AuditLog:
    """Append-only hash chain. verify fails if an earlier entry is edited.

    anchor stores the head outside the entry list. A log that is rebuilt into
    a new valid chain still fails verify_anchor.
    """

    GENESIS = "0" * 64

    def __init__(self):
        self._entries: list[dict] = []
        self._anchor: str | None = None
        self._lock = Lock()

    def append(self, event: str, **data: Any) -> None:
        with self._lock:
            prev = self._entries[-1]["hash"] if self._entries else self.GENESIS
            body = {"seq": len(self._entries), "event": event, **data}
            self._entries.append({**body, "prev": prev, "hash": _digest(prev, body)})

    def entries(self) -> list[dict]:
        with self._lock:
            return [dict(entry) for entry in self._entries]

    def head(self) -> str:
        with self._lock:
            return self._entries[-1]["hash"] if self._entries else self.GENESIS

    def anchor(self) -> str:
        """Remember the current head apart from the entries."""
        with self._lock:
            self._anchor = self._entries[-1]["hash"] if self._entries else self.GENESIS
            return self._anchor

    def verify(self) -> bool:
        prev = self.GENESIS
        for entry in self.entries():
            body = {key: value for key, value in entry.items() if key not in ("prev", "hash")}
            if entry.get("prev") != prev or entry.get("hash") != _digest(prev, body):
                return False
            prev = entry["hash"]
        return True

    def verify_anchor(self) -> bool:
        with self._lock:
            anchored = self._anchor
        if anchored is None:
            return False
        return self.verify() and self.head() == anchored


class WeightCommitLedger:
    """Programs already applied to a target. The same digest cannot commit twice."""

    def __init__(self):
        self.records: list[dict] = []
        self._seen: set[str] = set()

    def seen(self, digest: str) -> bool:
        return digest in self._seen

    def record(self, row: Mapping[str, Any]) -> None:
        digest = row["program_sha256"]
        if digest in self._seen:
            raise RuntimeError("This weight program was already committed")
        self._seen.add(digest)
        self.records.append(dict(row))


def commit_digest(args: Mapping[str, Any]) -> str:
    body = {key: args[key] for key in ("target", "kind", "program", "coefficient")}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def validate_args(spec: ToolSpec, args: Mapping[str, Any]) -> list[str]:
    problems = []
    if not isinstance(args, Mapping):
        return ["arguments must be an object"]
    unexpected = sorted(set(args) - set(spec.schema))
    problems.extend(f"unexpected field {name}" for name in unexpected)
    for name, expected in spec.schema.items():
        if name not in args:
            problems.append(f"missing field {name}")
        elif not _type_ok(args[name], expected):
            problems.append(f"field {name} must be {expected.__name__}")
    return problems


def weight_commit_permissions(signature, band, ledger: WeightCommitLedger) -> Callable[[Mapping[str, Any]], Decision | None]:
    """Argument rule for committing a weight program.

    A tool-specific denial wins over the destructive ask. The target must be
    the network being edited. Inject needs a value bias, merge needs a donor,
    and inject or steer must sit inside the sealed coefficient band. A digest
    already in the ledger is refused.
    """

    def check(args: Mapping[str, Any]) -> Decision | None:
        if args.get("target") == "actor":
            return Decision.DENY
        kind = args.get("kind")
        if kind == "inject" and not signature.value_bias:
            return Decision.DENY
        if kind == "merge" and not signature.donor:
            return Decision.DENY
        if kind in ("inject", "steer") and not band.admit_coefficient(float(args["coefficient"])):
            return Decision.DENY
        if ledger.seen(commit_digest(args)):
            return Decision.DENY
        return None

    return check


def weight_edit_tools(*, signature, band, applied: list, probes: list,
                      ledger: WeightCommitLedger | None = None) -> ToolRegistry:
    """Read-only observation and probe, plus one destructive commit."""
    ledger = ledger if ledger is not None else WeightCommitLedger()

    def observe(_args):
        return json.dumps({
            "name": signature.name, "layers": signature.layers,
            "value_bias": signature.value_bias, "donor": signature.donor,
        })

    def probe(args):
        probes.append(dict(args))
        return json.dumps({"program": args["program"], "concept_hit": 1.0,
                           "fact_ok": True, "collapse": False, "fluent": True})

    def commit(args):
        digest = commit_digest(args)
        row = {"kind": args["kind"], "program": args["program"], "coefficient": args["coefficient"],
               "target": args["target"], "program_sha256": digest}
        applied.append(row)
        ledger.record(row)
        return json.dumps({"committed": True, "program_sha256": digest})

    blank = {}
    registry = ToolRegistry([
        ToolSpec("observe_model", "Read the target signature", blank, observe,
                 read_only=True, concurrency_safe=True),
        ToolSpec("probe_edit", "Probe a program without writing weights",
                 {"program": str}, probe, read_only=True, concurrency_safe=True),
        ToolSpec("commit_weights", "Apply a weight program to the target network",
                 {"target": str, "kind": str, "program": str, "coefficient": float}, commit,
                 destructive=True, check_permissions=weight_commit_permissions(signature, band, ledger)),
    ])
    registry.commit_ledger = ledger
    return registry


class HarnessSession:
    """One investigation. The model proposes; the session decides."""

    def __init__(self, *, model, registry: ToolRegistry, budget: IterationBudget,
                 approver: Callable[[str, Mapping[str, Any]], bool], audit: AuditLog | None = None,
                 max_cost_usd: float = 1.0, abort: Event | None = None, name: str = "root",
                 usage: Usage | None = None, can_delegate: bool = True,
                 approval_timeout_s: float = 30.0, keep_iterations: int = 0,
                 cost_quote: Callable[[], float] | None = None,
                 completion_gate: Callable[["HarnessSession"], tuple[bool, str]] | None = None,
                 workspace_fingerprint: Callable[[], str] | None = None,
                 refinements: RefinementLedger | None = None):
        if max_cost_usd <= 0:
            raise ValueError("Cost ceiling must be positive")
        self.model = model
        self.registry = registry
        self.budget = budget
        self.approver = approver
        self.audit = audit if audit is not None else AuditLog()
        self.max_cost_usd = float(max_cost_usd)
        self.abort = abort if abort is not None else Event()
        self.name = name
        self.usage = usage if usage is not None else Usage()
        self.can_delegate = can_delegate
        self.approval_timeout_s = approval_timeout_s
        if type(keep_iterations) is not int or keep_iterations < 0:
            raise ValueError("keep_iterations must be a nonnegative integer")
        self.keep_iterations = keep_iterations
        self.cost_quote = cost_quote
        self.completion_gate = completion_gate
        self.workspace_fingerprint = workspace_fingerprint
        self._last_failed_fingerprint: str | None = None
        self.succeeded = False
        self.stop_reason = ""
        self.refinements = refinements if refinements is not None else RefinementLedger(audit=self.audit, agent=name)
        self.messages: list[dict] = []
        self.permission_denials: list[dict] = []
        self.handler_calls: list[str] = []
        self.final_text = ""

    def investigate(self, task: str) -> Iterator[EventRecord]:
        self.messages.append({"role": "user", "content": task, "untrusted": False})
        self.audit.append("start", agent=self.name, task=task)
        while True:
            if self.abort.is_set():
                self._stop("aborted", "stop requested")
                yield EventRecord("aborted", "stop requested", self.name)
                return
            if self.usage.cost_usd >= self.max_cost_usd:
                self._stop("cost", f"cost ceiling of ${self.max_cost_usd:.2f} reached")
                yield EventRecord("budget", self.final_text, self.name)
                return
            if self.cost_quote is not None and self.usage.cost_usd + float(self.cost_quote()) > self.max_cost_usd:
                self._stop("cost_quote", f"quoted cost would cross ${self.max_cost_usd:.2f}")
                yield EventRecord("budget", self.final_text, self.name)
                return
            if not self.budget.consume():
                self._stop("iterations", f"iteration cap of {self.budget.max_total} reached")
                yield EventRecord("budget", self.final_text, self.name)
                return
            response = self.model(self.messages, self.registry.names())
            self.usage.add(response.cost_usd, agent=self.name)
            self.messages.append({
                "role": "assistant", "content": response.text, "untrusted": False,
                "tool_calls": [call.name for call in response.tool_calls],
            })
            if response.text:
                yield EventRecord("analysis", response.text, self.name)
            if not response.tool_calls:
                self.final_text = response.text
                if self.completion_gate is None:
                    self.succeeded = True
                    self.stop_reason = "actor_finished"
                    self.audit.append("complete", agent=self.name)
                    yield EventRecord("complete", response.text or "investigation finished", self.name)
                    return
                done, event = self._finish_if_gate_passes()
                yield event
                if done:
                    return
            for call in _scheduled(response.tool_calls, self.registry):
                if call.name == "delegate":
                    yield from self._delegate(call)
                    continue
                text, outcome = self._run_tool(call)
                self.messages.append({"role": "tool", "name": call.name, "content": text, "untrusted": True})
                yield EventRecord("tool" if outcome == "ok" else outcome, text, self.name, outcome)

    def _delegate(self, call: ToolCall) -> Iterator[EventRecord]:
        if not self.can_delegate:
            self.audit.append("tool_unknown", agent=self.name, tool="delegate")
            yield EventRecord("unknown", json.dumps({"error": "unknown tool delegate"}), self.name, "unknown")
            return
        task = call.args.get("task") if isinstance(call.args, Mapping) else None
        if not isinstance(task, str) or not task.strip():
            self.audit.append("tool_invalid", agent=self.name, tool="delegate", problems=["missing field task"])
            yield EventRecord("invalid", json.dumps({"error": "invalid arguments"}), self.name, "invalid")
            return
        child = HarnessSession(
            model=self.model, registry=self.registry.read_only_view(), budget=self.budget,
            approver=self.approver, audit=self.audit, max_cost_usd=self.max_cost_usd,
            abort=self.abort, name=f"{self.name}/sub", usage=self.usage, can_delegate=False,
            approval_timeout_s=self.approval_timeout_s, cost_quote=self.cost_quote,
        )
        held = self.budget.reserve(self.keep_iterations) if self.keep_iterations else 0
        self.audit.append("delegate", agent=self.name, task=task, reserved_iterations=held)
        yield EventRecord("delegate", task, self.name)
        try:
            yield from child.investigate(task)
        finally:
            if held:
                self.budget.release(held)
        self.messages.append({"role": "tool", "name": "delegate", "content": child.final_text, "untrusted": True})

    def _stop(self, reason: str, text: str) -> None:
        """A budget or abort ending. It does not mark the goal successful."""
        self.succeeded = False
        self.stop_reason = reason
        self.final_text = text
        event = "aborted" if reason == "aborted" else "budget_stop"
        self.audit.append(event, agent=self.name, reason=reason)

    def compact_messages(self, summary: str) -> dict:
        """Replace the actor's visible messages. Earlier audit hashes stay in place."""
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("A compaction summary must be a nonempty string")
        before = [entry["hash"] for entry in self.audit.entries()]
        retained = len(self.messages)
        digest = hashlib.sha256(summary.encode()).hexdigest()
        self.messages = [{
            "role": "user", "content": summary, "untrusted": False, "compacted_from": retained,
        }]
        self.audit.append("compact", agent=self.name, retained_messages=retained, summary_sha256=digest)
        after = [entry["hash"] for entry in self.audit.entries()]
        if after[:len(before)] != before or not self.audit.verify():
            raise RuntimeError("Compaction rewrote the audit chain")
        return {"retained_messages": retained, "summary_sha256": digest, "audit_head": self.audit.head()}

    def _finish_if_gate_passes(self) -> tuple[bool, EventRecord]:
        fingerprint = None
        if self.workspace_fingerprint is not None:
            try:
                fingerprint = str(self.workspace_fingerprint())
            except Exception:
                fingerprint = None
        if fingerprint is not None and fingerprint == self._last_failed_fingerprint:
            note = "completion gate skipped; workspace unchanged since the last failure"
            self.audit.append("gate_skipped", agent=self.name, fingerprint=fingerprint)
            self.messages.append({"role": "user", "content": note, "untrusted": True})
            return False, EventRecord("gate", note, self.name, "skipped")
        try:
            ok, output = self.completion_gate(self)
            ok = bool(ok)
            output = "" if output is None else str(output)
        except Exception as exc:
            ok, output = False, str(exc)
        output = output[:240]
        if ok:
            self.succeeded = True
            self.stop_reason = "gate"
            self.audit.append("complete", agent=self.name, gate="passed")
            text = output or self.final_text or "investigation finished"
            return True, EventRecord("complete", text, self.name)
        self._last_failed_fingerprint = fingerprint
        self.audit.append("gate_failed", agent=self.name, output=output)
        note = output or "completion gate failed"
        self.messages.append({"role": "user", "content": note, "untrusted": True})
        return False, EventRecord("gate", note, self.name, "failed")

    def _decide(self, spec: ToolSpec, args: Mapping[str, Any]) -> Decision:
        if spec.check_permissions is not None:
            verdict = spec.check_permissions(args)
            if verdict is not None:
                return verdict
        if spec.destructive:
            return Decision.ASK
        return Decision.ALLOW

    def _ask_analyst(self, tool: str, args: Mapping[str, Any]) -> bool:
        if self.approval_timeout_s <= 0:
            return False
        try:
            return bool(self.approver(tool, args))
        except TimeoutError:
            return False

    def _run_tool(self, call: ToolCall) -> tuple[str, str]:
        spec = self.registry.get(call.name)
        if spec is None or not spec.check_fn():
            self.audit.append("tool_unknown", agent=self.name, tool=call.name)
            return json.dumps({"error": f"unknown tool {call.name}"}), "unknown"
        problems = validate_args(spec, call.args)
        if problems:
            self.audit.append("tool_invalid", agent=self.name, tool=call.name, problems=problems)
            return json.dumps({"error": "invalid arguments", "problems": problems}), "invalid"
        decision = self._decide(spec, call.args)
        approved = decision is Decision.ALLOW
        if decision is Decision.ASK:
            approved = self._ask_analyst(call.name, call.args)
        if not approved:
            record = {"tool": call.name, "args": dict(call.args), "decision": decision.value}
            self.permission_denials.append(record)
            self.audit.append("tool_denied", agent=self.name, **record)
            return json.dumps({"error": f"{call.name} denied"}), "denied"
        try:
            self.handler_calls.append(call.name)
            raw = spec.handler(call.args)
            outcome = "ok"
        except Exception as exc:
            raw, outcome = json.dumps({"error": str(exc)}), "error"
        if len(raw) > spec.max_result_chars:
            raw = raw[:spec.max_result_chars] + f"...[truncated {len(raw) - spec.max_result_chars} chars]"
        self.audit.append("tool_call", agent=self.name, tool=call.name, args=dict(call.args),
                          decision=decision.value, outcome=outcome, result_preview=raw[:120])
        if outcome == "ok" and call.name == "commit_weights":
            self.audit.append(
                "weight_commit", agent=self.name, program_sha256=commit_digest(call.args),
                target=call.args["target"], kind=call.args["kind"], coefficient=call.args["coefficient"],
            )
        return raw, outcome


def _scheduled(calls: Sequence[ToolCall], registry: ToolRegistry) -> list[ToolCall]:
    """Read-only concurrency-safe calls run as a batch, then the rest in order."""
    safe, rest = [], []
    for call in calls:
        spec = registry.get(call.name)
        if spec is not None and spec.read_only and spec.concurrency_safe:
            safe.append(call)
        else:
            rest.append(call)
    return safe + rest


def _type_ok(value: Any, expected: type) -> bool:
    if expected is float:
        return type(value) in (int, float) and type(value) is not bool
    if expected is int:
        return type(value) is int
    return type(value) is expected


_COMPANION_FIELDS = {"prompt_note", "memory", "skill", "subagent"}


class RefinementLedger:
    """Versioned companion edits. Test receipts are not evidence, and a freeze is final."""

    def __init__(self, audit: AuditLog | None = None, agent: str = "root"):
        self.audit = audit
        self.agent = agent
        self.companion: dict[str, str] = {}
        self.records: list[dict] = []
        self._pending: dict[str, dict] = {}
        self._head: str | None = None
        self._frozen = False
        self._seq = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def head(self) -> str | None:
        return self._head

    def propose(self, *, phase: str, field: str, value: str, trigger: str,
                receipt_holds: bool, receipt_phase: str) -> dict:
        if self._frozen:
            raise RuntimeError("The companion configuration is frozen")
        if phase not in ("train", "validation") or receipt_phase not in ("train", "validation"):
            raise RuntimeError("Test receipts are not evidence for a companion edit")
        if phase != receipt_phase:
            raise RuntimeError("A companion edit must be admitted on the split that produced its receipt")
        if field not in _COMPANION_FIELDS:
            raise RuntimeError("A companion edit cannot change the actor or the base prompt")
        if receipt_holds is not True:
            raise RuntimeError("A companion edit needs a holding receipt")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Companion value must be a nonempty string")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError("A refinement needs a trigger")
        prior = self.companion.get(field, "")
        body = {
            "seq": self._seq, "kind": "proposal", "phase": phase, "field": field,
            "prior": prior, "value": value, "trigger": trigger,
        }
        self._seq += 1
        record = {**body, "id": _body_id(body), "parent_id": self._head, "admitted": False}
        self._pending[record["id"]] = record
        self.records.append(record)
        if self.audit is not None:
            self.audit.append("refinement_proposed", agent=self.agent, refinement_id=record["id"],
                              field=field, phase=phase)
        return dict(record)

    def admit(self, refinement_id: str) -> dict:
        if self._frozen:
            raise RuntimeError("The companion configuration is frozen")
        record = self._pending.get(refinement_id)
        if record is None:
            raise KeyError("Unknown pending refinement")
        if self.companion.get(record["field"], "") != record["prior"]:
            raise RuntimeError("The companion field changed since this edit was proposed")
        self.companion[record["field"]] = record["value"]
        self._head = record["id"]
        record["admitted"] = True
        del self._pending[refinement_id]
        if self.audit is not None:
            self.audit.append("refinement_admitted", agent=self.agent, refinement_id=record["id"],
                              field=record["field"], phase=record["phase"])
        return dict(record)

    def freeze(self) -> dict:
        if self._frozen:
            raise RuntimeError("The companion configuration is already frozen")
        if self._head is None:
            raise RuntimeError("Admit a companion edit before freezing it")
        self._frozen = True
        self._pending.clear()
        if self.audit is not None:
            self.audit.append("refinement_frozen", agent=self.agent, head=self._head)
        return {"frozen": True, "head": self._head, "companion": dict(self.companion)}

    def rollback(self, refinement_id: str) -> dict:
        if self._frozen:
            raise RuntimeError("The companion configuration is frozen")
        if refinement_id != self._head:
            raise RuntimeError("Only the current companion edit can be rolled back")
        record = next(item for item in self.records if item["id"] == refinement_id and item.get("admitted"))
        if record["prior"]:
            self.companion[record["field"]] = record["prior"]
        else:
            self.companion.pop(record["field"], None)
        self._head = record["parent_id"]
        body = {"seq": self._seq, "kind": "rollback", "reverted": refinement_id, "restored": self._head}
        self._seq += 1
        entry = {**body, "id": _body_id(body)}
        self.records.append(entry)
        if self.audit is not None:
            self.audit.append("refinement_rolled_back", agent=self.agent, refinement_id=refinement_id,
                              restored=self._head or "")
        return dict(entry)


def _body_id(body: dict) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _digest(prev: str, body: dict) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + blob).encode()).hexdigest()
