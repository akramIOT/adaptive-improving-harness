"""Recompute displayed metrics from integer records already in this repository.

No provider call and no new measurement. Two-decimal figures use round-half-up,
the same rule as the MBPP aggregate replay. The script stops if a recomputed
figure disagrees with a value already published from these records.
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def two(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def saving(part, whole):
    return two(Decimal(100) * (1 - Decimal(part) / Decimal(whole)))


def load(path):
    return json.loads((ROOT / path).read_text())


def miniwob():
    data = load("docs/assets/readme/visual-comparison.json")
    rows = data["rows"]
    ours = next(row for row in rows if row["method"] == "HarnessMM-V")
    published = {
        "baseline harness": "29.56",
        "SkillOpt": "50.08",
        "AutoSaddler": "49.43",
        "SoL-Pi": "85.11",
        "Prompt Compaction": "9.80",
        "DeepSeek Harness": "80.14",
        "Codex Harness": "89.93",
        "RHO": "22.39",
        "HarnessPro": "60.31",
        "HarnessMM-V": "0.00",
    }
    out = []
    for row in rows:
        assert row["tasks"] == 96
        success = two(Decimal(100) * row["successes"] / row["tasks"])
        tokens = two(Decimal(row["execution_tokens"]) / row["tasks"] / 1000)
        saved = saving(ours["execution_tokens"], row["execution_tokens"])
        assert saved == published[row["method"]], (row["method"], saved)
        out.append({
            "method": row["method"],
            "successes": row["successes"],
            "execution_tokens": row["execution_tokens"],
            "success_percent": success,
            "tokens_per_task_k": tokens,
            "saving_vs_harnessmm_v_percent": saved,
        })
    return out


def mbpp():
    data = load("reproduction/mbpp_sanitized/aggregate.json")
    assert data["expected"] == data["settled"] == 257
    native_total = data["arms"]["native"]["totals"]["total_tokens"]
    names = {"native": "Native", "cot-only": "CoT-only", "chisel": "HarnessMM"}
    out = []
    for key, label in names.items():
        arm = data["arms"][key]
        totals = arm["totals"]
        assert totals["input_tokens"] + totals["output_tokens"] == totals["total_tokens"]
        assert arm["settled"] == 257 and arm["unknown_requests"] == 0
        out.append({
            "arm": label,
            "successes": arm["successes"],
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": totals["total_tokens"],
            "provider_attempts": totals["physical_provider_attempts"],
            "certified_substitutions": arm["certified_substitutions"],
            "success_percent": two(Decimal(100) * arm["successes"] / 257),
            "tokens_per_task_k": two(Decimal(totals["total_tokens"]) / 257 / 1000),
            "saving_vs_native_percent": saving(totals["total_tokens"], native_total),
        })
    by_arm = {row["arm"]: row for row in out}
    assert by_arm["Native"]["successes"] == 232
    assert by_arm["CoT-only"]["successes"] == 240
    assert by_arm["HarnessMM"]["successes"] == 236
    assert by_arm["HarnessMM"]["certified_substitutions"] == 472
    assert by_arm["Native"]["tokens_per_task_k"] == "53.98"
    assert by_arm["CoT-only"]["tokens_per_task_k"] == "15.04"
    assert by_arm["HarnessMM"]["tokens_per_task_k"] == "10.61"
    assert by_arm["HarnessMM"]["saving_vs_native_percent"] == "80.35"
    assert by_arm["CoT-only"]["saving_vs_native_percent"] == "72.14"
    return out


def ablation():
    data = load("docs/assets/readme/results.json")
    full = data["ablation"][0]
    assert full["configuration"] == "Full HarnessMM-V"
    published_saving = {
        "Full HarnessMM-V": "0.00",
        "Single-use EDG": "26.91",
        "Callback retained": "19.52",
        "EDG execution disabled": "18.62",
    }
    out = []
    for row in data["ablation"]:
        saved = saving(full["execution_tokens"], row["execution_tokens"])
        assert saved == published_saving[row["configuration"]]
        assert two(Decimal(100) * row["successes"] / 257) == f"{row['success']:.2f}"
        assert two(Decimal(row["execution_tokens"]) / 257 / 1000) == f"{row['tokens']:.2f}"
        out.append({
            "configuration": row["configuration"],
            "successes": row["successes"],
            "execution_tokens": row["execution_tokens"],
            "certified_substitutions": row["certified_substitutions"],
            "saving_vs_full_percent": saved,
        })
    return out


def occupied_pad():
    data = load("robotics/data/qwen-clear-refill/summary.json")
    out = []
    for population in ("original", "fresh"):
        for split in ("all", "train", "validation", "test"):
            for version in ("before", "after"):
                row = data[population][split][version]
                tokens = two(Decimal(row["actor_tokens"]) / row["episodes"] / 1000)
                out.append({
                    "population": population,
                    "split": split,
                    "version": version,
                    **row,
                    "tokens_per_task_k": tokens,
                })
    fresh = {(row["split"], row["version"]): row for row in out if row["population"] == "fresh"}
    assert fresh[("all", "before")]["tokens_per_task_k"] == "4.72"
    assert fresh[("all", "after")]["tokens_per_task_k"] == "1.47"
    assert fresh[("all", "before")]["zero_motion_attempts"] == 220
    assert fresh[("all", "after")]["zero_motion_attempts"] == 13
    assert fresh[("test", "before")]["successes"] == 3
    assert fresh[("test", "after")]["successes"] == 18
    assert fresh[("test", "before")]["tokens_per_task_k"] == "5.60"
    assert fresh[("test", "after")]["tokens_per_task_k"] == "1.58"
    return out


def tabletop():
    counts = {}
    episodes = successes = steps = decisions = ticks = 0
    for line in (ROOT / "robotics/data/qwen-tabletop-30/index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = f"{row['split']}|{row['scene']}"
        bucket = counts.setdefault(key, {
            "split": row["split"], "scene": row["scene"], "episodes": 0,
            "successes": 0, "steps": 0, "decisions": 0, "control_ticks": 0,
        })
        bucket["episodes"] += 1
        bucket["successes"] += int(row["success"])
        bucket["steps"] += row["steps"]
        bucket["decisions"] += row["decisions"]
        bucket["control_ticks"] += row["control_ticks"]
        episodes += 1
        successes += int(row["success"])
        steps += row["steps"]
        decisions += row["decisions"]
        ticks += row["control_ticks"]
    assert (episodes, successes, steps, decisions, ticks) == (30, 20, 224, 164, 7482)
    return {
        "episodes": episodes,
        "successes": successes,
        "steps": steps,
        "decisions": decisions,
        "control_ticks": ticks,
        "by_split_and_scene": [counts[key] for key in sorted(counts)],
    }


def main():
    summary = {
        "scope": "Integer totals already stored in this repository. Two-decimal metrics use round-half-up. No new measurement.",
        "miniwob_test": miniwob(),
        "mbpp_sanitized_test": mbpp(),
        "mbpp_ablation_test": ablation(),
        "occupied_pad": occupied_pad(),
        "tabletop_30": tabletop(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
