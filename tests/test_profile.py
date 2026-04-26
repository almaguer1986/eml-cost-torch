"""Profile tests — walk a torch.nn.Module and confirm output shape."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from eml_cost_torch import profile, profile_dict


def test_profile_sequential_returns_one_per_leaf():
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.GELU(),
        nn.Linear(4, 2),
    )
    rows = profile(model)
    assert len(rows) == 3
    assert [r.class_name for r in rows] == ["Linear", "GELU", "Linear"]


def test_profile_descends_nested_modules():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
            self.act = nn.Sigmoid()

    model = nn.Sequential(Block(), nn.Linear(4, 1))
    rows = profile(model)
    assert len(rows) == 3
    classes = [r.class_name for r in rows]
    assert classes == ["Linear", "Sigmoid", "Linear"]


def test_profile_dict_is_json_friendly():
    import json

    model = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
    rows = profile_dict(model)
    # Should be serialisable
    s = json.dumps(rows)
    assert "Linear" in s
    assert "Sigmoid" in s


def test_profile_qualified_names_are_stable():
    model = nn.Sequential(nn.Linear(4, 2), nn.GELU())
    rows = profile(model)
    # Sequential children get integer index names
    assert rows[0].name in ("0", "fc", "linear")  # tolerant
    assert all(r.name for r in rows)


def test_profile_tiny_transformer_block():
    """Exercise a realistic mini-Transformer-block-shaped composition."""
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.v = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)
            self.gelu = nn.GELU()
            self.out = nn.Linear(8, 8)

    rows = profile(TinyBlock())
    classes = [r.class_name for r in rows]
    assert "Linear" in classes
    assert "LayerNorm" in classes
    assert "GELU" in classes
    # GELU should land at higher r than Linear
    g = next(r for r in rows if r.class_name == "GELU")
    l = next(r for r in rows if r.class_name == "Linear")
    assert g.pfaffian_r > l.pfaffian_r


def test_profile_handles_unknown_layer_gracefully():
    class CustomActivation(nn.Module):
        def forward(self, x):
            return x * x

    model = nn.Sequential(nn.Linear(4, 2), CustomActivation())
    rows = profile(model)
    assert len(rows) == 2
    assert rows[1].is_unknown
    assert rows[0].class_name == "Linear"
