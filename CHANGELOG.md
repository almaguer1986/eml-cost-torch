# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — 2026-04-26 — `diagnose()` redesigned around per-activation empirical lookup (E-192)

### TL;DR

`diagnose()` no longer gates fp16-drift / activation-variance risk
predictions on the symbolic Pfaffian-not-EML classification. Instead,
it ships an empirical-basis JSON measured directly on **19 activation
functions × 5 seeds** in a controlled FFN architecture, and reports
each layer's *measured* drift and variance signature. This is more
honest *and* more useful — you can now compare GELU vs Mish vs SwiGLU
on actual fp16 sensitivity, not on a structural classification that
turned out to be a confound on heterogeneous architectures.

### What changed

  - **Activation registry expanded** from 36 to 51 named classes:
      - GELU is now correctly split into exact (erf-based, PNE) and
        tanh-approximation (NOT PNE) forms. Previously all GELU
        variants — including FastGELUActivation, PytorchGELUTanh,
        NewGELUActivation — were misclassified as PNE.
      - QuickGELUActivation now has its own sigmoid-approximation form.
      - GLU-family added: GLU, GeGLU, SwiGLU, ReGLU. GeGLU inherits
        erf -> PNE; the other three are EML-elementary.
      - Softsign, Threshold added.
  - **`diagnose()` returns per-activation predictions** keyed on torch
    class name. Each activation layer reports `fp16_drift_predicted`,
    `fp16_drift_std`, `activation_variance_predicted`,
    `activation_variance_std`, plus `fp16_risk` and
    `activation_variance_class` bands (`low` / `normal` / `elevated`)
    relative to the cross-activation median.
  - **Honest-finding note baked into the empirical-basis** field:
    Mann-Whitney U for PNE-vs-EML grouping yielded p=0.085 (fp16) and
    p=0.79 (variance) — neither reaches alpha=0.05 on the controlled
    corpus. The 0.4.0 PNE-gated diagnostic ran on heterogeneous
    architectures where PNE happened to coincide with high-fp16-risk
    positions; the new lookup is architecture-controlled and reliable.

### Per-activation rankings (E-192, fp16 drift)

Top 5 (highest drift, most fp16-sensitive):

    GeGLU       0.000736   (PNE)
    SwiGLU      0.000707
    ReGLU       0.000665
    QuickGELU   0.000602
    Softsign    0.000576

Bottom 3 (lowest drift, most fp16-stable):

    Sigmoid     0.000229
    Hardsigmoid 0.000226
    Softplus    0.000241

GELU exact lands at 0.000536 (rank 7 of 19). The 3 GLU-family
activations dominate the high-drift end despite only 1 of 3 being
classified as PNE — demonstrating the old PNE gate was missing
real risk.

### Breaking changes

  - `LayerRisk.fp16_risk` enum changed: previously {`low`, `elevated`}
    (PNE-gated); now {`low`, `normal`, `elevated`, `n/a`} (continuous,
    median-relative).
  - `LayerRisk.activation_variance_class` enum changed: previously
    {`normal`, `saturating`}; now {`low`, `normal`, `elevated`, `n/a`}.
  - `LayerRisk` gains: `activation_key`, `fp16_drift_predicted`,
    `fp16_drift_std`, `activation_variance_predicted`,
    `activation_variance_std`.
  - `DiagnosisReport.n_layers_with_saturating_variance` removed; use
    `n_layers_with_elevated_activation_variance` instead.
  - `DiagnosisReport.empirical_basis` schema changed: keyed by
    `study=E-192`, `n_activations_measured=19`, etc.
  - Now requires `eml-cost>=0.6.0` (for the expanded activation forms).

### Migration

If your code reads `fp16_risk`, swap `"elevated"` checks to either
`"elevated"` (still works, narrower scope) or `report.layers` →
`fp16_drift_predicted` for the raw measured value.

For users who relied on the PNE flag for fp16 prediction: that flag is
preserved but no longer drives the risk band. Use
`is_pfaffian_not_eml` for symbolic optimization-cost questions; use
`fp16_drift_predicted` and `fp16_risk` for empirical fp16 questions.

## [0.4.0] — 2026-04-26 — `diagnose()` predictive risk profile

### Empirical basis (replicated 2026-04-26 with ViT-B/16 added)

  - **n=387 layers across 3 transformers** (GPT-2 small + BERT-base + ViT-B/16),
    **36 PNE samples** (was 24 in initial E-183 run; replicated cleanly).
  - **fp16_drift**: BH-q = 0.020, r_rb = -0.25, ratio 1.10x.
  - **activation_variance**: BH-q = 1.5e-3, r_rb = +0.35, ratio 0.33x.
  - Controls (ResNet-18, EfficientNet-B0) returned all-N/A as expected
    (0 PNE layers each — ReLU and SiLU are EML-elementary).
  - Per-architecture activation_variance survives independently:
    GPT-2 q=8.6e-4, BERT q=2.7e-5, ViT q=2.7e-4.
  - All 24 → 36 PNE samples remain GELU variants. **GELU is currently
    the only modern activation classified as Pfaffian-not-EML** in the
    eml-cost-torch registry (SiLU = x·sigmoid(x) is EML-elementary;
    Mish, GeGLU likewise EML).

### Added

- **`eml_cost_torch.diagnose(model)`** — empirically-grounded per-layer
  risk diagnostic. Returns a `DiagnosisReport` with per-layer
  `LayerRisk` predictions:
    - `fp16_risk` ∈ {`low`, `elevated`}
    - `activation_variance_class` ∈ {`normal`, `saturating`}
  Predictions apply **only** to the two effects that survived BH-FDR
  in the E-183 architecture-diagnostic study (n=275 layers across
  GPT-2 small + BERT-base):
    - **Pfaffian-not-EML layers show ~14% higher fp16 drift** under
      cast (BH-q = 0.022)
    - **Pfaffian-not-EML layers show ~53% lower activation variance**
      under input perturbation (BH-q = 2.1×10⁻⁴)
  The function does NOT run the model — predictions derive from the
  symbolic Pfaffian classification of each layer's class.

- `DiagnosisReport.to_dict()` for JSON serialization. The
  `empirical_basis` is embedded so consumers can audit the
  data backing each prediction.

### Tests

- 8 new in `tests/test_diagnose.py`. Full suite: **35 passing.**

### Mechanistic explanation (a priori, supports the data)

GELU/SiLU/Mish saturate at input tails → bounded outputs (lower
variance) + small derivatives in saturating regions (snap to zero
under fp16 quantization → drift). Same mechanism predicted to apply
to other Pfaffian-not-EML activation classes. Only GELU was
empirically tested in the E-183 corpus — generalization to other
PNE classes pending E-183 expansion.

### Honest caveats

  - All 24 PNE samples in the E-183 corpus are GELU variants. Other
    PNE classes (SiLU, Mish, GeGLU) are theoretically covered by the
    same mechanism but empirically untested in 0.4.0.
  - Drift measurement uses CPU-emulated fp16 cast, not hardware GPU
    fp16. Real-GPU validation pending.
  - The `gradient_norm` hypothesis from the E-183 study is N/A
    (PNE layers have no parameters); not refuted, just untested.
