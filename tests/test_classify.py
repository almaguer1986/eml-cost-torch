"""Classification tests — runs eml-cost on each registered layer."""
from __future__ import annotations

import pytest

from eml_cost_torch.classify import classify_form, classify_layer
from eml_cost_torch.registry import TORCH_LAYER_REGISTRY, lookup_form


def test_classify_linear_form_is_polynomial():
    out = classify_form(lookup_form("Linear"))
    assert out["pfaffian_r"] == 0  # polynomial → no chain elements


def test_classify_sigmoid_form_has_chain_one():
    out = classify_form(lookup_form("Sigmoid"))
    assert out["pfaffian_r"] >= 1


def test_classify_gelu_uses_chain_for_erf():
    out = classify_form(lookup_form("GELU"))
    # erf is non-elementary; should be detected as r >= 1
    assert out["pfaffian_r"] >= 1
    assert out["axes"].startswith("p")


def test_classify_layernorm_axes_has_depth_field():
    out = classify_form(lookup_form("LayerNorm"))
    # axes is "p<r>-d<n>-w<m>-c<k>"
    assert "-d" in out["axes"]
    assert "-w" in out["axes"]


def test_all_registry_entries_classify_without_error():
    """Smoke test — every registered form should run through analyze."""
    failures = []
    for name, form in TORCH_LAYER_REGISTRY.items():
        try:
            classify_form(form)
        except Exception as exc:
            failures.append((name, type(exc).__name__, str(exc)[:120]))
    assert not failures, failures


# torch-only tests: skip cleanly if torch is not installed
torch = pytest.importorskip("torch")
nn = torch.nn


def test_classify_layer_linear_via_torch():
    layer = nn.Linear(8, 4)
    p = classify_layer("layer", layer)
    assert p.class_name == "Linear"
    assert p.pfaffian_r == 0
    assert not p.is_unknown


def test_classify_layer_gelu_via_torch():
    p = classify_layer("act", nn.GELU())
    assert p.class_name == "GELU"
    assert p.pfaffian_r >= 1


def test_classify_layer_unknown_class():
    class NotAStandardLayer(nn.Module):
        pass

    p = classify_layer("custom", NotAStandardLayer())
    assert p.is_unknown
    assert p.pfaffian_r == -1
    assert p.axes == "p?-d?-w?-c?"
