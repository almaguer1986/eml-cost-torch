"""Summary tests — pretty-printing format checks."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from eml_cost_torch import summary


def test_summary_contains_layer_names():
    model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    s = summary(model)
    assert "Linear" in s
    assert "GELU" in s


def test_summary_contains_axes_format():
    model = nn.Sequential(nn.Linear(2, 2))
    s = summary(model)
    # axes-tuple "p<r>-d<n>-w<m>-c<k>"
    assert "p0-d" in s


def test_summary_aggregate_section_appears():
    model = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())
    s = summary(model)
    assert "total r" in s
    assert "distinct cost classes" in s


def test_summary_no_aggregate():
    model = nn.Sequential(nn.Linear(4, 2))
    s = summary(model, show_aggregate=False)
    assert "total r" not in s


def test_summary_marks_unknown_layers():
    class CustomLayer(nn.Module):
        pass

    model = nn.Sequential(nn.Linear(2, 2), CustomLayer())
    s = summary(model)
    assert "UNKNOWN" in s or "unknown classes:" in s
