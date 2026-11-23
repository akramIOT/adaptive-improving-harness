"""Reuse a weight program after its probe holds. No model call and no weight write."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_harness.weight_space import (
    CoefficientBand, EditBank, ModelSignature, ProbeReceipt, StoredEdit, WeightEditSession,
)


def main():
    qwen = ModelSignature("Qwen2.5-0.5B-Instruct", 24, value_bias=True, donor=False)
    held = ProbeReceipt(concept_hit=1.0, fact_ok=True, collapse=False, fluent=True)
    collapsed = ProbeReceipt(concept_hit=1.0, fact_ok=False, collapse=True, fluent=False)
    bank = EditBank()
    wedding = StoredEdit(
        goal="inject wedding", signature=qwen, kind="inject",
        program="layers.14-19 inject wedding 16", coefficient=16)
    bank.commit(wedding, held)
    bank.commit(StoredEdit(
        goal="inject ocean", signature=qwen, kind="inject",
        program="layers.14-19 inject ocean 16", coefficient=16), collapsed)
    session = WeightEditSession(bank)
    reused = session.author_or_reuse(
        goal="inject wedding", signature=qwen, kind="inject", callback_id="author-2")
    ocean = session.author_or_reuse(
        goal="inject ocean", signature=qwen, kind="inject", callback_id="author-3")
    llama = ModelSignature("Llama-3.2-1B-Instruct", 16, value_bias=False, donor=False)
    refused = session.author_or_reuse(
        goal="inject wedding", signature=llama, kind="inject", callback_id="author-4")

    band = CoefficientBand("inject")
    band.record(16, held, phase="train")
    band.record(20, collapsed, phase="train")
    band.record(16, held, phase="validation")
    frozen = band.select_and_freeze()
    print(json.dumps({
        "scope": "Synthetic weight-edit reuse; no benchmark and no weight write",
        "reused_program": reused.edit.program,
        "bypassed_callback": reused.certificate.bypassed_callback_id,
        "ocean": ocean.reason,
        "llama_inject": refused.reason,
        "frozen_band": {"floor": frozen["floor"], "cap": frozen["cap"], "reason": frozen["reason"]},
        "coefficient_24_admitted": band.admit_coefficient(24),
    }, indent=2))


if __name__ == "__main__":
    main()
