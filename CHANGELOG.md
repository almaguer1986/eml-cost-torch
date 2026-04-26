# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] — 2026-04-26 — `diagnose()` predictive risk profile

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
