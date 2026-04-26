"""eml_cost_torch.diagnose — predictive risk profile for any torch.nn.Module.

Given a model, returns a per-layer diagnostic report with empirically-grounded
risk predictions based on the layer's Pfaffian classification.

Empirical basis (E-183, 2026-04-26):
  - Per-layer measurements on GPT-2 small + BERT-base = 275 layers, 24 PNE.
  - Mann-Whitney U + Benjamini-Hochberg FDR.
  - 2 of 3 hypotheses survive BH-FDR alpha=0.05:

    (a) fp16_drift: PNE layers drift ~14-18% MORE than EML layers under
        fp16 cast (BH-q = 0.022, n_pne=24, n_eml=227). Effect size
        rank-biserial r = -0.30 pooled.
    (b) activation_variance: PNE layers have ~50% LESS output variance
        under input perturbation (BH-q = 2.1e-4, n_pne=24, n_eml=227).
        Effect size rank-biserial r = +0.49 pooled.

Mechanistic explanation: Pfaffian-not-EML activations (GELU, etc.) saturate
at the tails of their input distribution, producing bounded outputs with
lower magnitude variance. The same saturation curvature explains the higher
fp16 sensitivity — bf16/fp16 quantization snaps small derivatives in the
saturation region to zero.

The diagnose() function returns predictions limited to these two
empirically-validated effects. Speculative predictions (training time,
gradient stability across the net) are NOT included pending further evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch.nn as nn

from .profile import profile_dict


# Empirical constants from E-183 (GPT-2 + BERT pooled, n=275).
# These are the median-ratio observations, not modeled predictions.
_FP16_DRIFT_RATIO_PNE_OVER_EML = 1.14   # 14% more drift on PNE layers
_FP16_DRIFT_BH_Q = 0.022
_ACTVAR_RATIO_PNE_OVER_EML = 0.47       # 53% less variance on PNE layers
_ACTVAR_BH_Q = 2.1e-4


@dataclass
class LayerRisk:
    """Per-layer empirically-grounded risk prediction."""

    layer_name: str
    class_name: str
    pfaffian_r: Optional[int]
    eml_depth: Optional[int]
    is_pfaffian_not_eml: bool
    axes: Optional[str]

    fp16_risk: str = "low"
    """One of: low, elevated. 'elevated' = Pfaffian-not-EML layer with
    empirical median fp16 drift 14% higher than EML layers (q=0.022)."""

    activation_variance_class: str = "normal"
    """One of: normal, saturating. 'saturating' = Pfaffian-not-EML layer with
    empirical median activation variance 53% lower than EML layers (q=2e-4).
    Saturating activations are robust to input perturbation but contribute
    less gradient signal."""

    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Full diagnostic report for a model."""

    model_class: str
    n_layers_total: int
    n_layers_pfaffian_not_eml: int
    n_layers_with_elevated_fp16_risk: int
    n_layers_with_saturating_variance: int
    layers: list[LayerRisk] = field(default_factory=list)

    empirical_basis: dict = field(default_factory=lambda: {
        "study": "E-183 (2026-04-26)",
        "architectures": ["GPT-2 small", "BERT-base"],
        "n_layers_pooled": 275,
        "n_pfaffian_not_eml_pooled": 24,
        "fp16_drift_q_value": _FP16_DRIFT_BH_Q,
        "fp16_drift_median_ratio_pne_over_eml": _FP16_DRIFT_RATIO_PNE_OVER_EML,
        "activation_variance_q_value": _ACTVAR_BH_Q,
        "activation_variance_median_ratio_pne_over_eml": _ACTVAR_RATIO_PNE_OVER_EML,
    })

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"DiagnosisReport for {self.model_class}",
            f"  layers total:                   {self.n_layers_total}",
            f"  Pfaffian-not-EML:               {self.n_layers_pfaffian_not_eml}",
            f"  elevated fp16 risk:             {self.n_layers_with_elevated_fp16_risk}",
            f"  saturating variance:            {self.n_layers_with_saturating_variance}",
            f"",
            f"  Empirical basis: E-183 (2026-04-26), n=275 layers across",
            f"  GPT-2 small + BERT-base. fp16 risk q={_FP16_DRIFT_BH_Q:.3f}; "
            f"actvar q={_ACTVAR_BH_Q:.1e}.",
            f"",
            f"  At-risk layers:",
        ]
        for lr in self.layers:
            if lr.is_pfaffian_not_eml:
                lines.append(f"    {lr.layer_name} ({lr.class_name}): "
                              f"fp16={lr.fp16_risk}, actvar={lr.activation_variance_class}")
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
        Per-layer risk profile with predictions limited to the two
        empirically-validated effects from E-183:
          - elevated fp16 sensitivity for Pfaffian-not-EML layers
          - saturating activation variance for Pfaffian-not-EML layers

    Notes
    -----
    Predictions apply per-layer based on the symbolic Pfaffian
    classification computed from the layer class. The function does NOT
    run the model — predictions are derived from architecture, not
    measurement.

    For empirical verification on a specific model, run the
    measurement pipeline at:
      monogate-research/exploration/E-183-architecture-diagnostic/

    Examples
    --------
    >>> from eml_cost_torch import diagnose
    >>> from transformers import GPT2Config, GPT2Model
    >>> model = GPT2Model(GPT2Config())
    >>> report = diagnose(model)
    >>> print(report)
    """
    rows = profile_dict(model)

    layers = []
    for r in rows:
        is_pne = bool(r.get("is_pfaffian_not_eml"))
        risk = LayerRisk(
            layer_name=r.get("name", ""),
            class_name=r.get("class_name", "unknown"),
            pfaffian_r=r.get("pfaffian_r"),
            eml_depth=r.get("eml_depth"),
            is_pfaffian_not_eml=is_pne,
            axes=r.get("axes"),
        )
        if is_pne:
            risk.fp16_risk = "elevated"
            risk.activation_variance_class = "saturating"
            risk.notes.append(
                f"Pfaffian-not-EML class '{risk.class_name}': "
                f"~{(_FP16_DRIFT_RATIO_PNE_OVER_EML - 1) * 100:.0f}% higher fp16 drift, "
                f"~{(1 - _ACTVAR_RATIO_PNE_OVER_EML) * 100:.0f}% lower activation variance "
                f"vs other layer types (E-183, BH-q < 0.05)."
            )
        layers.append(risk)

    return DiagnosisReport(
        model_class=type(model).__name__,
        n_layers_total=len(layers),
        n_layers_pfaffian_not_eml=sum(1 for l in layers if l.is_pfaffian_not_eml),
        n_layers_with_elevated_fp16_risk=sum(1 for l in layers if l.fp16_risk == "elevated"),
        n_layers_with_saturating_variance=sum(1 for l in layers
                                               if l.activation_variance_class == "saturating"),
        layers=layers,
    )
