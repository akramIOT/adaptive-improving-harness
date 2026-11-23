"""The local entry point re-exports the upstream lifecycle without copying it."""
import adaptive_harness
from adaptive_harness import CONCEPTS, build_partition, canonical_sha256
from adaptive_harness.harness.benchmark_partition import build_partition as upstream_partition
from adaptive_harness.harness.codex_chisel import CodexChiselController


def test_facade_is_the_upstream_objects():
    assert adaptive_harness.CodexChiselController is CodexChiselController
    assert adaptive_harness.build_partition is upstream_partition
    assert adaptive_harness.PROJECT == "adaptive-improving-harness"
    assert set(CONCEPTS) == {
        "executable_decision_graph",
        "guarded_decision_reuse",
        "certified_callback_substitution",
        "recursive_self_improvement",
        "outcome_independent_split",
        "token_accounting",
        "weight_edit_reuse",
        "coefficient_band",
        "control_gate",
        "hash_chained_audit",
        "long_run",
    }


def test_split_through_the_local_entry_point():
    records = [
        {"id": f"synthetic-{i}",
         "payload_sha256": canonical_sha256({"payload": i}),
         "public_task_sha256": canonical_sha256({"public": i})}
        for i in range(10)
    ]
    partition = build_partition(
        records, benchmark="adaptive-improving-harness", seed=20261026, independent_unit="task"
    )
    assert partition["counts"] == {"train": 2, "validation": 2, "test": 6}
    assert sum(len(partition["splits"][name]) for name in ("train", "validation", "test")) == 10
