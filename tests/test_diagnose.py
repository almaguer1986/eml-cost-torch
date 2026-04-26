"""Tests for eml_cost_torch.diagnose — empirically-grounded risk profile.

Updated for the 0.5.0 redesign (E-192). The diagnostic now reports
per-activation empirical fp16 drift and activation variance, with
risk bands relative to the cross-activation median. The PNE flag is
still reported for transparency but is no longer the gating signal
for the risk fields — see the diagnose module docstring for the
honest statistical context.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn

from eml_cost_torch import diagnose, DiagnosisReport, LayerRisk


def test_diagnose_returns_report_with_layers() -> None:
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    report = diagnose(model)
    assert isinstance(report, DiagnosisReport)
    assert report.n_layers_total >= 3
    assert all(isinstance(layer, LayerRisk) for layer in report.layers)


def test_gelu_flagged_as_pne_and_has_empirical_signature() -> None:
    """GELU is structurally PNE (erf) AND has an E-192 empirical row."""
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(approximate="none"))
    report = diagnose(model)
    gelu_layer = next(layer for layer in report.layers if layer.class_name == "GELU")
    assert gelu_layer.is_pfaffian_not_eml is True
    assert gelu_layer.activation_key == "GELU"
    assert gelu_layer.fp16_drift_predicted is not None
    assert gelu_layer.fp16_drift_predicted > 0.0
    assert gelu_layer.activation_variance_predicted is not None
    assert gelu_layer.activation_variance_predicted > 0.0
    assert gelu_layer.fp16_risk in {"low", "normal", "elevated"}
    assert gelu_layer.activation_variance_class in {"low", "normal", "elevated"}


def test_relu_not_pne_but_has_empirical_signature() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    report = diagnose(model)
    relu_layer = next(
        (layer for layer in report.layers if layer.class_name == "ReLU"), None
    )
    if relu_layer is None:
        return  # ReLU may be inlined in some configs
    assert relu_layer.is_pfaffian_not_eml is False
    # Empirical fields populated
    assert relu_layer.activation_key == "ReLU"
    assert relu_layer.fp16_drift_predicted is not None


def test_sigmoid_is_low_risk() -> None:
    """E-192 measured Sigmoid as the second-lowest fp16 drift activation."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid(), nn.Linear(4, 2))
    report = diagnose(model)
    sig = next(
        (layer for layer in report.layers if layer.class_name == "Sigmoid"),
        None,
    )
    if sig is None:
        return
    assert sig.fp16_risk == "low"


def test_pure_linear_model_has_no_activation_layers() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    report = diagnose(model)
    assert report.n_layers_pfaffian_not_eml == 0
    assert report.n_activation_layers == 0
    assert report.n_layers_with_elevated_fp16_risk == 0
    assert report.n_layers_with_elevated_activation_variance == 0


def test_linear_layer_has_no_empirical_lookup() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    report = diagnose(model)
    linear = next(layer for layer in report.layers if layer.class_name == "Linear")
    assert linear.activation_key is None
    assert linear.fp16_drift_predicted is None
    assert linear.activation_variance_predicted is None
    assert linear.fp16_risk == "n/a"
    assert linear.activation_variance_class == "n/a"


def test_diagnose_report_str_includes_activation_layers() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    report = diagnose(model)
    s = str(report)
    assert "DiagnosisReport" in s
    assert "fp16" in s.lower()
    assert "GELU" in s


def test_diagnose_report_to_dict_is_json_friendly() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    report = diagnose(model)
    d = report.to_dict()
    s = json.dumps(d)
    assert "n_layers_total" in s
    assert "empirical_basis" in s
    assert "n_activation_layers" in s


def test_empirical_basis_is_present_and_e192() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    report = diagnose(model)
    eb = report.empirical_basis
    assert eb["study"] == "E-192"
    assert eb["n_activations_measured"] >= 19
    assert eb["n_seeds_per_activation"] >= 5
    assert "honest_note" in eb
    # Honest note explicitly says PNE alone is not reliable
    assert "not a reliable predictor" in eb["honest_note"]


def test_diagnose_does_not_run_model() -> None:
    """diagnose() must NOT call the model — pure architecture analysis."""

    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.activation = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError(
                "forward should not be called by diagnose()"
            )

    model = FailingModel()
    report = diagnose(model)
    assert report.n_layers_pfaffian_not_eml == 1
    assert report.n_activation_layers == 1


# --------------------------------------------------------------------
# Specific E-192 expected behaviours
# --------------------------------------------------------------------


def test_geglu_classified_pne() -> None:
    """GeGLU = x * GELU(y) inherits erf -> PNE."""
    from eml_cost_torch.registry import lookup_form
    from eml_cost import analyze
    form = lookup_form("GeGLU")
    assert form is not None
    a = analyze(form)
    assert a.is_pfaffian_not_eml is True


def test_swiglu_classified_eml_elementary() -> None:
    """SwiGLU = x * SiLU(y) — both halves EML-elementary."""
    from eml_cost_torch.registry import lookup_form
    from eml_cost import analyze
    form = lookup_form("SwiGLU")
    assert form is not None
    a = analyze(form)
    assert a.is_pfaffian_not_eml is False


def test_gelu_tanh_approx_classified_eml_elementary() -> None:
    """tanh-approximation GELU is NOT PNE — distinct from exact erf form."""
    from eml_cost_torch.registry import lookup_form
    from eml_cost import analyze
    # HF tanh-approx classes
    for name in ("FastGELUActivation", "PytorchGELUTanh", "NewGELUActivation"):
        form = lookup_form(name)
        assert form is not None
        a = analyze(form)
        assert a.is_pfaffian_not_eml is False, (
            f"{name} should NOT be PNE (tanh-approximation has no erf)"
        )


def test_full_activation_zoo_diagnoses_cleanly() -> None:
    """Mixed-activation block: every activation gets an empirical signature."""
    model = nn.Sequential(
        nn.Linear(64, 64), nn.GELU(approximate="none"),
        nn.Linear(64, 64), nn.GELU(approximate="tanh"),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.Mish(),
        nn.Linear(64, 64), nn.Sigmoid(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, 64), nn.Softplus(),
        nn.Linear(64, 64), nn.Softsign(),
        nn.Linear(64, 32),
    )
    report = diagnose(model)
    # Every activation layer must have an empirical signature
    for layer in report.layers:
        if layer.class_name in {"Linear"}:
            continue
        assert layer.activation_key is not None, (
            f"{layer.class_name} not in activation survey"
        )
        assert layer.fp16_drift_predicted is not None
        assert layer.activation_variance_predicted is not None


def test_per_activation_signatures_distinct() -> None:
    """The empirical lookup distinguishes activations: GELU and Sigmoid
    get different fp16 risk bands."""
    gelu_model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    sigmoid_model = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
    g = diagnose(gelu_model).layers[1]
    s = diagnose(sigmoid_model).layers[1]
    # Sigmoid measured at ~0.000229; GELU at ~0.000536. Sigmoid should be
    # in the "low" band, GELU in "normal" or "elevated".
    assert g.fp16_drift_predicted is not None
    assert s.fp16_drift_predicted is not None
    assert g.fp16_drift_predicted > s.fp16_drift_predicted
    assert s.fp16_risk == "low"
