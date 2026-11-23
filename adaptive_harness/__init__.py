"""Adaptive improving harness.

Guarded decision reuse and recursive self-improvement. Actor weights stay
fixed. Improvement is confined to stored decisions, receipt guards, certified
native substitutions, and companion configuration chosen on the training split
and frozen before test.
"""

from adaptive_harness.harness.benchmark_partition import build_partition, canonical_sha256
from adaptive_harness.harness.codex_chisel import ChiselConfig, CodexChiselController
from adaptive_harness.control import AuditLog, Decision, HarnessSession, IterationBudget, RefinementLedger
from adaptive_harness.weight_space import (
    CoefficientBand, EditBank, ModelSignature, ProbeReceipt, StoredEdit, WeightEditSession,
)

PROJECT = "adaptive-improving-harness"
UPSTREAM = "HarnessMM"
UPSTREAM_ARCHIVE = "https://arxiv.org/abs/2610.32813"

# Paper names, then the modules that implement them. Internal identifiers
# (CHISEL, SAR, Jev, phiagent) stay in those modules on purpose.
CONCEPTS = {
    "executable_decision_graph": "integrations/deepseek_harness/cot_jev.mjs",
    "guarded_decision_reuse": "integrations/deepseek_harness/sar_v2.mjs",
    "certified_callback_substitution": "integrations/deepseek_harness/sar_v2.mjs",
    "recursive_self_improvement": "adaptive_harness/harness/codex_chisel.py",
    "outcome_independent_split": "adaptive_harness/harness/benchmark_partition.py",
    "token_accounting": "adaptive_harness/evaluation/deployment_token_accounting.py",
    "weight_edit_reuse": "adaptive_harness/weight_space.py",
    "coefficient_band": "adaptive_harness/weight_space.py",
    "control_gate": "adaptive_harness/control.py",
    "hash_chained_audit": "adaptive_harness/control.py",
    "long_run": "adaptive_harness/control.py",
}

__all__ = [
    "PROJECT",
    "UPSTREAM",
    "UPSTREAM_ARCHIVE",
    "CONCEPTS",
    "ChiselConfig",
    "CodexChiselController",
    "build_partition",
    "canonical_sha256",
    "CoefficientBand",
    "EditBank",
    "ModelSignature",
    "ProbeReceipt",
    "StoredEdit",
    "WeightEditSession",
    "AuditLog",
    "Decision",
    "HarnessSession",
    "IterationBudget",
    "RefinementLedger",
]
