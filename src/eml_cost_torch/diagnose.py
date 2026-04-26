"""eml_cost_torch.diagnose — predictive risk profile for any torch.nn.Module.

Given a model, returns a per-layer diagnostic with empirically-grounded
per-activation predictions for fp16 drift and activation variance.

Empirical basis (E-192, 2026-04-26)
-----------------------------------
Controlled corpus: 19 activation functions measured on a fixed 4-block
FFN architecture (hidden=128, batch=32) over 5 random seeds.
Measurement: relative L2 fp16 drift on activation-layer outputs;
activation variance via 0.05-sigma input perturbation, 8 samples.

Key honest finding. On this controlled corpus, the symbolic
Pfaffian-not-EML classification (which fires on `erf`-based GELU and
GeGLU) is NOT a statistically reliable predictor of fp16 drift:
Mann-Whitney U PNE-vs-EML p=0.085 (fp16) and p=0.79 (variance), neither
reaching alpha=0.05. The reliable predictor is the *activation
identity itself*. We therefore ship per-activation measured signatures
and report the empirical risk band for each layer based on its class
name.

The original E-183 PNE-gated finding on heterogeneous architectures
(GPT-2 / BERT / ViT) emerged from architectural confounds — GELU layers
happened to be both the rare PNE class AND positioned where fp16 drift
is highest. The 0.5.0 diagnostic supersedes that earlier per-layer
PNE flag with the per-activation empirical lookup, which works for
all 19 measured activations regardless of PNE classification.

The PNE flag is still reported for transparency — it correctly
identifies which activations are not finite EML trees and informs
optimization-pass cost estimates — but it is no longer the gating
signal for the fp16/variance risk fields.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn

from .profile import profile_dict


_BASIS_PATH = Path(__file__).parent / "data" / "activation_empirical_basis.json"


def _load_basis() -> dict[str, Any]:
    with _BASIS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


_BASIS: dict[str, Any] = _load_basis()
_PER_ACTIVATION: dict[str, dict[str, Any]] = _BASIS["per_activation"]
_CLASS_TO_ACT: dict[str, str] = _BASIS["class_to_activation"]
_THRESHOLDS: dict[str, float] = _BASIS["thresholds"]


def _risk_band(value: float, median: float) -> str:
    """Bin a continuous risk score into low/normal/elevated using the
    cross-activation median. Bands are equal-population by construction
    of the median."""
    if value > median * 1.15:
        return "elevated"
    if value < median * 0.85:
        return "low"
    return "normal"


@dataclass
class LayerRisk:
    """Per-layer empirically-grounded risk prediction.

    Attributes
    ----------
    layer_name, class_name, axes, pfaffian_r, eml_depth, is_pfaffian_not_eml:
        Structural fields from :func:`eml_cost_torch.profile_dict`.
    activation_key:
        The activation-survey label (e.g. ``"GELU"``, ``"SwiGLU"``)
        used to look up empirical signatures, or ``None`` if this
        layer is not an activation in the survey.
    fp16_drift_predicted:
        Empirical mean of relative L2 fp16 drift measured on this
        activation in E-192. ``None`` for non-activation layers.
    fp16_drift_std:
        Across-seed std of the fp16 drift measurement.
    fp16_risk:
        Risk band: ``"low"``, ``"normal"``, or ``"elevated"`` relative
        to the cross-activation median (0.000515 in E-192).
    activation_variance_predicted:
        Empirical mean of activation variance under input perturbation.
    activation_variance_std:
        Across-seed std.
    activation_variance_class:
        Risk band: ``"low"``, ``"normal"``, or ``"elevated"``.
    notes:
        Free-form explanatory notes.
    """

    layer_name: str
    class_name: str
    pfaffian_r: Optional[int]
    eml_depth: Optional[int]
    is_pfaffian_not_eml: bool
    axes: Optional[str]

    activation_key: Optional[str] = None
    fp16_drift_predicted: Optional[float] = None
    fp16_drift_std: Optional[float] = None
    fp16_risk: str = "n/a"
    activation_variance_predicted: Optional[float] = None
    activation_variance_std: Optional[float] = None
    activation_variance_class: str = "n/a"
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Full diagnostic report for a model."""

    model_class: str
    n_layers_total: int
    n_activation_layers: int
    n_layers_pfaffian_not_eml: int
    n_layers_with_elevated_fp16_risk: int
    n_layers_with_elevated_activation_variance: int
    layers: list[LayerRisk] = field(default_factory=list)

    empirical_basis: dict[str, Any] = field(
        default_factory=lambda: {
            "study": _BASIS["meta"]["session"],
            "date": _BASIS["meta"]["date"],
            "architecture": _BASIS["meta"]["architecture"],
            "n_activations_measured": _BASIS["meta"]["n_activations_measured"],
            "n_seeds_per_activation": _BASIS["meta"]["n_seeds_per_activation"],
            "fp16_median_across_activations": _THRESHOLDS["fp16_median"],
            "activation_variance_median_across_activations":
                _THRESHOLDS["act_variance_median"],
            "honest_note": _BASIS["meta"]["honest_note"],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"DiagnosisReport for {self.model_class}",
            f"  layers total:                       {self.n_layers_total}",
            f"  activation layers (measured):       {self.n_activation_layers}",
            f"  Pfaffian-not-EML (structural):      "
            f"{self.n_layers_pfaffian_not_eml}",
            f"  elevated fp16 risk (empirical):     "
            f"{self.n_layers_with_elevated_fp16_risk}",
            f"  elevated activation variance:       "
            f"{self.n_layers_with_elevated_activation_variance}",
            "",
            f"  Empirical basis: E-192, 19 activations x 5 seeds on "
            f"controlled FFN.",
            f"  fp16 median across activations:     "
            f"{_THRESHOLDS['fp16_median']:.6f}",
            "",
            "  Activation layers:",
        ]
        for lr in self.layers:
            if lr.activation_key is not None:
                fp16 = (
                    f"{lr.fp16_drift_predicted:.6f}"
                    if lr.fp16_drift_predicted is not None
                    else "n/a"
                )
                var = (
                    f"{lr.activation_variance_predicted:.6f}"
                    if lr.activation_variance_predicted is not None
                    else "n/a"
                )
                lines.append(
                    f"    {lr.layer_name} ({lr.class_name} "
                    f"-> {lr.activation_key}): "
                    f"fp16={lr.fp16_risk} ({fp16}), "
                    f"actvar={lr.activation_variance_class} ({var})"
                )
        return "\n".join(lines)


def diagnose(model: nn.Module) -> DiagnosisReport:
    """Generate a per-layer empirically-grounded risk diagnostic.

    Parameters
    ----------
    model : torch.nn.Module
        Any PyTorch module.

    Returns
    -------
    DiagnosisReport
        Per-layer risk profile. Activation layers (ReLU, GELU, Mish,
        SwiGLU, etc.) get empirical fp16 drift and activation variance
        predictions from the E-192 controlled study. Non-activation
        layers (Linear, Conv, LayerNorm) get structural fields only;
        their fp16 / variance fields are ``None`` and risk band is
        ``"n/a"``.

    Notes
    -----
    Predictions are based on per-activation empirical lookup, not on
    symbolic Pfaffian-not-EML classification. The PNE flag is still
    reported for transparency. See module docstring for the honest
    statistical context (PNE alone is not a reliable predictor on a
    controlled corpus; activation identity is).

    For empirical re-verification on a specific architecture, run
    ``monogate-research/exploration/E-192-activation-survey/``.

    Examples
    --------
    >>> from eml_cost_torch import diagnose
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    >>> report = diagnose(model)
    >>> print(report)  # doctest: +SKIP
    """
    rows = profile_dict(model)
    layers: list[LayerRisk] = []

    n_activation_layers = 0
    n_elevated_fp16 = 0
    n_elevated_var = 0

    for r in rows:
        class_name = r.get("class_name", "unknown") or "unknown"
        is_pne = bool(r.get("is_pfaffian_not_eml"))
        risk = LayerRisk(
            layer_name=r.get("name", "") or "",
            class_name=class_name,
            pfaffian_r=r.get("pfaffian_r"),
            eml_depth=r.get("eml_depth"),
            is_pfaffian_not_eml=is_pne,
            axes=r.get("axes"),
        )

        # Empirical lookup
        act_key = _CLASS_TO_ACT.get(class_name)
        if act_key is not None and act_key in _PER_ACTIVATION:
            n_activation_layers += 1
            stats_ = _PER_ACTIVATION[act_key]
            risk.activation_key = act_key
            risk.fp16_drift_predicted = float(stats_["fp16_drift_mean"])
            risk.fp16_drift_std = float(stats_["fp16_drift_std"])
            risk.activation_variance_predicted = float(
                stats_["act_variance_mean"]
            )
            risk.activation_variance_std = float(stats_["act_variance_std"])
            risk.fp16_risk = _risk_band(
                risk.fp16_drift_predicted, _THRESHOLDS["fp16_median"]
            )
            risk.activation_variance_class = _risk_band(
                risk.activation_variance_predicted,
                _THRESHOLDS["act_variance_median"],
            )
            if risk.fp16_risk == "elevated":
                n_elevated_fp16 += 1
            if risk.activation_variance_class == "elevated":
                n_elevated_var += 1
            risk.notes.append(
                f"Activation '{class_name}' -> survey class '{act_key}': "
                f"fp16 drift {risk.fp16_drift_predicted:.6f} +- "
                f"{risk.fp16_drift_std:.6f}, "
                f"act variance {risk.activation_variance_predicted:.6f} "
                f"+- {risk.activation_variance_std:.6f} "
                f"(E-192 n=5 seeds, FFN hidden=128)."
            )
        elif is_pne:
            risk.notes.append(
                f"Class '{class_name}' is Pfaffian-not-EML (structural) "
                f"but not in the E-192 activation survey; no empirical "
                f"fp16/variance prediction available."
            )

        layers.append(risk)

    return DiagnosisReport(
        model_class=type(model).__name__,
        n_layers_total=len(layers),
        n_activation_layers=n_activation_layers,
        n_layers_pfaffian_not_eml=sum(
            1 for layer in layers if layer.is_pfaffian_not_eml
        ),
        n_layers_with_elevated_fp16_risk=n_elevated_fp16,
        n_layers_with_elevated_activation_variance=n_elevated_var,
        layers=layers,
    )
