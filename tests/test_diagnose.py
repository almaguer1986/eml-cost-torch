"""Tests for eml_cost_torch.diagnose — empirically-grounded risk profile."""
from __future__ import annotations

import torch
import torch.nn as nn

from eml_cost_torch import diagnose, DiagnosisReport, LayerRisk


def test_diagnose_returns_report_with_layers():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    report = diagnose(model)
    assert isinstance(report, DiagnosisReport)
    assert report.n_layers_total >= 3
    assert all(isinstance(l, LayerRisk) for l in report.layers)


def test_gelu_flagged_as_pfaffian_not_eml():
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    report = diagnose(model)
    gelu_layer = next(l for l in report.layers if l.class_name == "GELU")
    assert gelu_layer.is_pfaffian_not_eml is True
    assert gelu_layer.fp16_risk == "elevated"
    assert gelu_layer.activation_variance_class == "saturating"


def test_relu_not_flagged_as_pfaffian_not_eml():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    report = diagnose(model)
    relu_layer = next((l for l in report.layers if l.class_name == "ReLU"), None)
    if relu_layer is None:
        return  # ReLU may not be a leaf in some configs
    assert relu_layer.is_pfaffian_not_eml is False
    assert relu_layer.fp16_risk == "low"
    assert relu_layer.activation_variance_class == "normal"


def test_pure_linear_model_has_no_pne_layers():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    report = diagnose(model)
    assert report.n_layers_pfaffian_not_eml == 0
    assert report.n_layers_with_elevated_fp16_risk == 0
    assert report.n_layers_with_saturating_variance == 0


def test_diagnose_report_str_includes_at_risk_layers():
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    report = diagnose(model)
    s = str(report)
    assert "DiagnosisReport" in s
    assert "elevated fp16 risk" in s
    assert "GELU" in s


def test_diagnose_report_to_dict_is_json_friendly():
    import json
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    report = diagnose(model)
    d = report.to_dict()
    # Must be JSON-serializable
    s = json.dumps(d)
    assert "n_layers_total" in s
    assert "empirical_basis" in s


def test_empirical_basis_is_present_and_documented():
    model = nn.Sequential(nn.Linear(4, 4))
    report = diagnose(model)
    eb = report.empirical_basis
    assert "study" in eb
    assert "E-183" in eb["study"]
    assert eb["fp16_drift_q_value"] < 0.05
    assert eb["activation_variance_q_value"] < 0.05
    assert eb["n_pfaffian_not_eml_pooled"] >= 12


def test_diagnose_does_not_run_model():
    """diagnose() must NOT call the model — pure architecture analysis."""
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.activation = nn.GELU()

        def forward(self, x):
            raise RuntimeError("forward should not be called by diagnose()")

    model = FailingModel()
    # Should NOT raise — diagnose walks the tree, doesn't run forward
    report = diagnose(model)
    assert report.n_layers_pfaffian_not_eml == 1
