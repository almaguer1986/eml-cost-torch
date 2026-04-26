"""Registry tests — validates the symbolic forms for each known layer."""
from __future__ import annotations

import sympy as sp

from eml_cost_torch.registry import (
    TORCH_LAYER_REGISTRY,
    known_layer_names,
    lookup_form,
)


def test_registry_nonempty():
    assert len(TORCH_LAYER_REGISTRY) >= 40


def test_known_layer_names_sorted():
    names = known_layer_names()
    assert names == sorted(names)


def test_lookup_returns_sympy_expression():
    for name in TORCH_LAYER_REGISTRY:
        form = lookup_form(name)
        assert form is not None, name
        assert isinstance(form, sp.Basic), name


def test_lookup_unknown_returns_none():
    assert lookup_form("DefinitelyNotARealLayer") is None


def test_linear_family_share_form():
    """Linear, Conv1d/2d/3d should all map to the same form."""
    linear = lookup_form("Linear")
    conv1 = lookup_form("Conv1d")
    conv2 = lookup_form("Conv2d")
    assert linear == conv1 == conv2


def test_relu_is_max():
    relu = lookup_form("ReLU")
    assert relu.has(sp.Max)


def test_gelu_uses_erf():
    gelu = lookup_form("GELU")
    assert gelu.has(sp.erf)


def test_sigmoid_is_inverse_exp():
    sig = lookup_form("Sigmoid")
    # Should contain exp
    assert sig.has(sp.exp)
