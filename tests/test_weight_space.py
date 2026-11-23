"""Weight-edit reuse and the frozen coefficient band. No model and no weights."""
import pytest

from adaptive_harness.weight_space import (
    CoefficientBand, EditBank, ModelSignature, ProbeReceipt, StoredEdit, WeightEditSession,
)


def signature(**overrides):
    values = dict(name="Qwen2.5-0.5B-Instruct", layers=24, value_bias=True, donor=False)
    values.update(overrides)
    return ModelSignature(**values)


def edit(**overrides):
    values = dict(
        goal="inject wedding", signature=signature(), kind="inject",
        program="layers.14-19 inject wedding 16", coefficient=16,
    )
    values.update(overrides)
    return StoredEdit(**values)


def receipt(**overrides):
    values = dict(concept_hit=1.0, fact_ok=True, collapse=False, fluent=True)
    values.update(overrides)
    return ProbeReceipt(**values)


def test_a_holding_receipt_is_reused_and_certified():
    bank = EditBank()
    assert bank.commit(edit(), receipt())
    session = WeightEditSession(bank)
    decision = session.author_or_reuse(
        goal="  Inject   wedding ", signature=signature(), kind="inject", callback_id="author-2")
    assert decision.action == "reuse"
    assert decision.edit.program == "layers.14-19 inject wedding 16"
    assert decision.certificate.bypassed_callback_id == "author-2"
    assert len(session.certificates) == 1


def test_failed_probes_and_foreign_layouts_do_not_substitute():
    bank = EditBank()
    assert not bank.commit(edit(), receipt(concept_hit=0.49))
    assert not bank.commit(edit(), receipt(fact_ok=False))
    assert not bank.commit(edit(), receipt(collapse=True, fluent=False))
    assert not bank.commit(edit(coefficient=24), receipt())
    assert not bank.commit(edit(signature=signature(value_bias=False)), receipt())
    assert bank.commit(edit(), receipt())
    session = WeightEditSession(bank)
    llama = signature(name="Llama-3.2-1B-Instruct", layers=16, value_bias=False)
    refused = session.author_or_reuse(
        goal="inject wedding", signature=llama, kind="inject", callback_id="author-3")
    assert refused.action == "abstain"
    assert refused.reason == "layout-cannot-express"
    other = session.author_or_reuse(
        goal="inject ocean", signature=signature(), kind="inject", callback_id="author-4")
    assert other.reason == "no-stored-edit"
    assert session.certificates == []


def test_merge_requires_a_donor_and_steer_can_reuse_without_a_value_bias():
    bank = EditBank()
    llama = signature(name="Llama-3.2-1B-Instruct", layers=16, value_bias=False)
    steer = edit(goal="steer newyear", signature=llama, kind="steer",
                 program="layers.6-11.mlp.down_proj steer newyear 1.2", coefficient=1.2)
    assert bank.commit(steer, receipt())
    session = WeightEditSession(bank)
    decision = session.author_or_reuse(
        goal="steer newyear", signature=llama, kind="steer", callback_id="author-5")
    assert decision.action == "reuse"
    assert not bank.commit(edit(kind="merge", program="mix 0.1"), receipt())
    donor = signature(donor=True)
    assert bank.commit(edit(kind="merge", signature=donor, program="graft norms", coefficient=0), receipt(concept_hit=None))


def test_coefficient_band_shrinks_to_held_window_and_then_seals():
    band = CoefficientBand("inject")
    band.record(8, receipt(concept_hit=0.2), phase="train")
    band.record(16, receipt(), phase="train")
    band.record(20, receipt(collapse=True, fluent=False), phase="train")
    band.record(16, receipt(), phase="validation")
    band.record(20, receipt(collapse=True, fluent=False), phase="validation")
    frozen = band.select_and_freeze()
    assert frozen["floor"] == 16
    assert frozen["cap"] == 16
    assert frozen["reason"] == "held-window"
    assert band.admit_coefficient(16)
    assert not band.admit_coefficient(8)
    assert not band.admit_coefficient(24)
    with pytest.raises(RuntimeError, match="already frozen"):
        band.select_and_freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        band.record(16, receipt(), phase="train")
    with pytest.raises(RuntimeError, match="not evidence"):
        CoefficientBand("steer").record(1.2, receipt(), phase="test")


def test_validation_disagreement_keeps_the_reference_band():
    band = CoefficientBand("steer")
    band.record(1.2, receipt(), phase="train")
    band.record(0.4, receipt(), phase="validation")
    frozen = band.select_and_freeze()
    assert (frozen["floor"], frozen["cap"]) == (0.4, 1.4)
    assert frozen["reason"] == "reference-band"
    assert band.admit_coefficient(0.4)
    assert band.admit_coefficient(1.4)
    assert not band.admit_coefficient(2.0)
    collapsed_only = CoefficientBand("inject")
    collapsed_only.record(20, receipt(collapse=True, fluent=False), phase="train")
    collapsed_only.record(20, receipt(collapse=True, fluent=False), phase="validation")
    assert collapsed_only.select_and_freeze()["reason"] == "reference-band"
