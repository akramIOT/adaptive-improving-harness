"""Build a 20/20/60 split through the local adaptive_harness entry point."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_harness import PROJECT, UPSTREAM, build_partition, canonical_sha256


def main():
    records = [
        {"id": f"example-{i:02d}",
         "payload_sha256": canonical_sha256({"synthetic_payload": i}),
         "public_task_sha256": canonical_sha256({"synthetic_public_task": i})}
        for i in range(10)
    ]
    partition = build_partition(
        records, benchmark="adaptive-improving-harness", seed=20261026, independent_unit="task"
    )
    print(json.dumps({
        "project": PROJECT,
        "upstream": UPSTREAM,
        "benchmark": partition["benchmark"],
        "counts": partition["counts"],
        "splits": partition["splits"],
    }, indent=2))


if __name__ == "__main__":
    main()
