# Adaptive improving harness

The Python package is `adaptive_harness`. Robotics demonstrations live in `robotics/`, the pi extension in `pi-extension/`, and the DeepSeek desktop plugin in `dsh-plugin/desktop/`. `SOURCE_MANIFEST.json` records the current bytes of the extracted sources.

## Place in the weight-space program

The neighboring projects edit a target network directly.

| Project | What changes |
| --- | --- |
| AgenticWeightInfra | A browser agent rewrites cached weights with a small weight language, then probes the edited model |
| weight-space-programming-autoresearch | A coding agent hand-writes interpretable features and transformer weights. Training is not the search procedure |
| adaptive-improving-harness | The actor's weights stay fixed. The harness stores decisions, checks receipts, substitutes native callbacks, and freezes a companion configuration before test |

The two kinds of edit compose. A weight-space search can be the task inside this harness. This checkout does not import the browser runtime or the evolve experiments. Those stay in their own repositories. Callers import `adaptive_harness`.

## What the implementation does

A slow model call writes future tool actions into an executable decision graph. After each tool result, guarded decision reuse checks whether the receipt supports a stored action. Certified callback substitution sends that action through the host's native tool interface and records the reasoning callback it replaced. When the graph cannot continue, control returns to the slow actor.

Recursive self-improvement then refines companion prompts, context settings, and adapter rules. `CodexChiselController` trains on the training split, selects on validation, freezes source hashes, and only then evaluates test. Re-training and a second test pass on the same controller are refused. The DeepSeek controller keeps that lifecycle and adapts the SDK request fields. Model weights are not an optimization variable.

Token reports count every physical call on a task, including authorship, control, retries, fallbacks, and unsuccessful executions. Offline optimization and validation are reported separately from test execution.

## Weight-space goals

A weight-space session can be the task inside this harness. `WeightEditSession` reuses a stored program when three things are true: the target signature matches, the target can express that kind of edit, and the probe receipt still holds. A holding receipt has the concept in at least half the answers when the edit is a concept, keeps the anchor fact, and stays fluent. The session records a certificate for the authoring callback it skipped. A collapsed probe, a lost fact, a model without a value bias, and a model without a donor do not substitute an inject or a merge.

`CoefficientBand` then shrinks the search window from training receipts and seals it. Inject stays inside 4–20 and steer stays inside 0.4–1.4, the regimes measured by the reference runs. The sealed window can be narrower. It cannot be wider. Test receipts are not evidence, a second freeze is refused, and a coefficient outside the sealed window is not admitted. `examples/weight_space_reuse.py` runs this on synthetic receipts.

## Control, budget, and audit

`HarnessSession` decides whether a proposed weight-space call runs. A call is looked up, validated, checked against an argument rule, and, when it is destructive, held for approval. A timeout is a refusal. The handler is last.

One iteration counter and one cost total cover the root and its subagent. The subagent receives the read-only tools and cannot delegate. The cost check runs before each actor call, so one call can cross the ceiling and the next call does not start. Stop is checked before the next call; a tool already returned by the actor still finishes. Every ending, including an unknown tool, a bad argument, and a denial, is appended to a hash-chained log. Editing an earlier entry makes `verify` fail.

On `commit_weights`, the argument rule denies an inject when the target has no value bias, a merge when it has no donor, and an inject or steer outside the sealed coefficient band. A coefficient inside the band is still destructive, so it waits for approval. The target named `actor` is denied, and a digest already recorded in `WeightCommitLedger` is denied. A parent may reserve iterations with `keep_iterations`; the subagent cannot spend that reserve. `cost_quote` stops the next call when the quoted cost would cross the ceiling. `anchor` stores the audit head apart from the entries, and `verify_anchor` fails if that head is replaced by another valid chain. A budget stop leaves `succeeded` false. With a `completion_gate`, the run finishes only when that gate passes; an unchanged workspace fingerprint skips a repeated failed gate. `RefinementLedger` versions a companion edit, admits it only from a holding train or validation receipt, rolls the current edit back by id, and refuses a second freeze. `compact_messages` shortens the actor's messages without rewriting earlier audit hashes. Reuse of a stored program does not skip this gate. `examples/control_gate.py` runs the refusal with a scripted actor.

## Layout

| Path | Role |
| --- | --- |
| `adaptive_harness/` | This repository's import surface. Re-exports the split and the improvement controller |
| `adaptive_harness/weight_space.py` | Reuse of a weight program whose probe held, and the sealed coefficient band |
| `adaptive_harness/control.py` | Permission gate, commit ledger, reserved iterations, cost quote, read-only subagent, and anchored audit |
| `adaptive_harness/harness/` | Codex and DeepSeek adapters, benchmark controllers, and the RSI lifecycle |
| `adaptive_harness/learning/` | Residual fitting, cross-fitting, routing, and companion-configuration methods |
| `adaptive_harness/evaluation/` | Score checks and complete token accounting |
| `integrations/baseline_harness/` | Fresh screenshot decision at every boundary |
| `integrations/codex_harness/` | Public exports of the native Codex runtime and bank adapters |
| `integrations/deepseek_harness/` | Graph authoring, receipt guards, execution, and certificates |
| `phiagent/` | Import bridge for the previous package name. Same objects as `adaptive_harness` |
| `pi-extension/` | Community pi extension. Planned terminal actions run when the prior result matches the guard |
| `dsh-plugin/desktop/` | DeepSeek Harness desktop plugin |
| `robotics/` | MuJoCo demonstration generator. Separate from the token-efficiency core |
| `reproduction/` | Frozen aggregates and mechanism cases that run without a provider |
| `configs/` | Two reference configurations used by the unit tests |
| `scripts/` | Research runners. They expect upstream datasets, runtimes, and endpoints |
| `SOURCE_MANIFEST.json` | Published hashes for the extracted sources |

`CHISEL`, `SAR`, `Jev`, `<CHISEL_PLAN>`, and the `phiagent` schema strings are protocol identifiers. They stay in the modules that already emit them. Renaming those strings would break frozen runs and the archived HumanEval reference.

## Local checks

From this directory, with Python 3.10–3.14 and Node.js 22:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_adaptive_harness.py tests/test_weight_space.py tests/test_control.py tests/test_benchmark_partition.py
python scripts/verify_release.py
python reproduction/run_all.py
node examples/guarded_reuse.mjs
python examples/partition.py
python examples/weight_space_reuse.py
python examples/control_gate.py
python examples/long_run.py
```

`examples/guarded_reuse.mjs` runs three stored actions through an in-memory dispatcher and returns control to the slow stream when the plan ends. `examples/partition.py` builds a 20/20/60 split of synthetic tasks. The weight-space, control-gate, and long-run examples use scripted actors and synthetic receipts. None of these calls needs a model endpoint.
