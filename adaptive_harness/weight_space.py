"""Weight-space goals inside the adaptive harness.

A weight-space session edits a target network. This module does not. It stores
a program whose probe receipt already held, and it freezes the coefficient
search band before test. The actor's weights are not an input.

The receipt rule matches the probe used by the weight-space agent: a concept
edit needs the concept in at least half the answers, the anchor fact must not
be lost, and the answer must stay fluent. Inject stays inside 4–20. Steer stays
inside 0.4–1.4. Those are the regimes the reference runs measured. The band
selected here can only shrink inside that regime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


REFERENCE_BANDS = {
    "inject": (4.0, 20.0),
    "steer": (0.4, 1.4),
}
CONCEPT_KINDS = frozenset({"inject", "steer"})
LAYOUT_KINDS = frozenset({"inject", "steer", "skip", "scale", "noise", "merge"})
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ModelSignature:
    """What the loaded target can express. Not the actor."""

    name: str
    layers: int
    value_bias: bool
    donor: bool

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model signature needs a name")
        if type(self.layers) is not int or self.layers < 1:
            raise ValueError("Model signature needs a positive layer count")
        if type(self.value_bias) is not bool or type(self.donor) is not bool:
            raise ValueError("value_bias and donor must be booleans")


@dataclass(frozen=True)
class ProbeReceipt:
    """Score of one probe. concept_hit is None when the edit is not a concept."""

    concept_hit: float | None
    fact_ok: bool | None
    collapse: bool
    fluent: bool

    def __post_init__(self):
        if self.concept_hit is not None and (
                type(self.concept_hit) not in (int, float) or isinstance(self.concept_hit, bool)
                or not 0 <= float(self.concept_hit) <= 1):
            raise ValueError("concept_hit must be between 0 and 1, or None")
        for name in ("fact_ok",):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean or None")
        if type(self.collapse) is not bool or type(self.fluent) is not bool:
            raise ValueError("collapse and fluent must be booleans")


@dataclass(frozen=True)
class StoredEdit:
    goal: str
    signature: ModelSignature
    kind: str
    program: str
    coefficient: float

    def __post_init__(self):
        if self.kind not in LAYOUT_KINDS:
            raise ValueError("Unknown weight-edit kind")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("Goal must be a nonempty string")
        if not isinstance(self.program, str) or not self.program.strip():
            raise ValueError("Program must be a nonempty string")
        if type(self.coefficient) is bool or not isinstance(self.coefficient, (int, float)):
            raise ValueError("Coefficient must be a number")


@dataclass(frozen=True)
class SubstitutionCertificate:
    bypassed_callback_id: str
    goal: str
    kind: str
    program: str
    coefficient: float
    signature: ModelSignature


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    edit: StoredEdit | None = None
    certificate: SubstitutionCertificate | None = None


def receipt_holds(receipt: ProbeReceipt, *, concept: bool) -> bool:
    """True when the probe supports reusing the program that produced it."""
    concept_ok = (not concept) or receipt.concept_hit is None or float(receipt.concept_hit) >= 0.5
    fact_ok = receipt.fact_ok is not False
    return concept_ok and fact_ok and not receipt.collapse and receipt.fluent


def layout_allows(kind: str, signature: ModelSignature) -> bool:
    """False when this target cannot express the edit, so nothing is reused."""
    if kind not in LAYOUT_KINDS:
        raise ValueError("Unknown weight-edit kind")
    if kind == "inject":
        return signature.value_bias
    if kind == "merge":
        return signature.donor
    return True


def _goal_key(goal: str) -> str:
    return " ".join(goal.split()).casefold()


def _inside(value: float, low: float, high: float) -> bool:
    return low - _TOLERANCE <= value <= high + _TOLERANCE


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class EditBank:
    """Programs whose probe receipt held, keyed by goal, layout, and kind."""

    def __init__(self):
        self._edits: dict[tuple, StoredEdit] = {}

    def __len__(self):
        return len(self._edits)

    def commit(self, edit: StoredEdit, receipt: ProbeReceipt) -> bool:
        if not layout_allows(edit.kind, edit.signature):
            return False
        if edit.kind in REFERENCE_BANDS and not _inside(edit.coefficient, *REFERENCE_BANDS[edit.kind]):
            return False
        if not receipt_holds(receipt, concept=edit.kind in CONCEPT_KINDS):
            return False
        self._edits[_key(edit.goal, edit.signature, edit.kind)] = edit
        return True

    def reuse(self, *, goal: str, signature: ModelSignature, kind: str) -> StoredEdit | None:
        if not layout_allows(kind, signature):
            return None
        return self._edits.get(_key(goal, signature, kind))


class WeightEditSession:
    """Reuse a stored program, or return control to the actor."""

    def __init__(self, bank: EditBank | None = None):
        self.bank = bank if bank is not None else EditBank()
        self.certificates: list[SubstitutionCertificate] = []

    def author_or_reuse(self, *, goal: str, signature: ModelSignature, kind: str,
                        callback_id: str) -> Decision:
        if not isinstance(callback_id, str) or not callback_id:
            raise ValueError("A callback id is required")
        if not layout_allows(kind, signature):
            return Decision("abstain", "layout-cannot-express")
        edit = self.bank.reuse(goal=goal, signature=signature, kind=kind)
        if edit is None:
            return Decision("abstain", "no-stored-edit")
        certificate = SubstitutionCertificate(
            bypassed_callback_id=callback_id, goal=edit.goal, kind=edit.kind,
            program=edit.program, coefficient=edit.coefficient, signature=signature)
        self.certificates.append(certificate)
        return Decision("reuse", "stored-edit", edit, certificate)


class CoefficientBand:
    """Shrink the reference coefficient window from training, then seal it.

    Test receipts are not evidence. A second freeze is refused. A coefficient
    outside the frozen window is not admitted, and the window never grows past
    the reference floor and cap.
    """

    def __init__(self, kind: str):
        if kind not in REFERENCE_BANDS:
            raise ValueError("Coefficient bands are defined for inject and steer")
        self.kind = kind
        self.reference = REFERENCE_BANDS[kind]
        self._train: list[tuple[float, bool]] = []
        self._validation: list[tuple[float, bool]] = []
        self.frozen: dict | None = None

    def record(self, coefficient: float, receipt: ProbeReceipt, *, phase: str) -> None:
        if self.frozen is not None:
            raise RuntimeError("The coefficient band is frozen")
        if phase not in ("train", "validation"):
            raise RuntimeError("Test receipts are not evidence for the coefficient band")
        if type(coefficient) is bool or not isinstance(coefficient, (int, float)):
            raise ValueError("Coefficient must be a number")
        held = receipt_holds(receipt, concept=True)
        bucket = self._train if phase == "train" else self._validation
        bucket.append((float(coefficient), held))

    def select_and_freeze(self) -> dict:
        if self.frozen is not None:
            raise RuntimeError("The coefficient band is already frozen")
        if not self._train:
            raise RuntimeError("Train before freezing the coefficient band")
        low, high, reason = _select_window(self.reference, self._train, self._validation)
        sealed = {
            "kind": self.kind,
            "floor": low,
            "cap": high,
            "reason": reason,
            "reference_floor": self.reference[0],
            "reference_cap": self.reference[1],
        }
        sealed["sha256"] = _digest({key: sealed[key] for key in sealed})
        self.frozen = sealed
        return sealed

    def admit_coefficient(self, coefficient: float) -> bool:
        if self.frozen is None:
            raise RuntimeError("Freeze the coefficient band before test")
        return _inside(coefficient, self.frozen["floor"], self.frozen["cap"])


def _select_window(reference, train, validation):
    low_ref, high_ref = reference
    held = [value for value, ok in train if ok and _inside(value, low_ref, high_ref)]
    collapsed = [value for value, ok in train if not ok and _inside(value, low_ref, high_ref)]
    if not held:
        return (*reference, "reference-band")
    low, high = max(low_ref, min(held)), min(high_ref, max(held))
    if low > high or any(_inside(value, low, high) for value in collapsed):
        return (*reference, "reference-band")
    validation_held = [value for value, ok in validation if ok]
    validation_lost = [value for value, ok in validation if not ok]
    if any(not _inside(value, low, high) for value in validation_held):
        return (*reference, "reference-band")
    if any(_inside(value, low, high) for value in validation_lost):
        return (*reference, "reference-band")
    return low, high, "held-window"


def _key(goal: str, signature: ModelSignature, kind: str):
    return (
        _goal_key(goal), kind, signature.name, signature.layers,
        signature.value_bias, signature.donor,
    )
