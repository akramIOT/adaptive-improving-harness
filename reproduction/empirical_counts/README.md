# Integer counts behind the rounded tables

```bash
python reproduction/empirical_counts/run.py
```

Requires Python 3.10–3.14 and only the standard library. The command reads
recorded totals and recomputes the two-decimal metrics with round-half-up. It
makes no provider call and does not add a measurement.

The inputs are `docs/assets/readme/visual-comparison.json`,
`docs/assets/readme/results.json`, `reproduction/mbpp_sanitized/aggregate.json`,
`robotics/data/qwen-clear-refill/summary.json`, and
`robotics/data/qwen-tabletop-30/index.jsonl`. The script stops if a recomputed
figure disagrees with a value already published from those records, including
the MiniWoB++ savings 29.56, 85.11, 89.93, 80.14 and 60.31, the MBPP savings
80.35 and 72.14, and the ablation savings 26.91, 19.52 and 18.62.
