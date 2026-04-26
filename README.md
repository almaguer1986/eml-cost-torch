# eml-cost-torch

Per-layer **Pfaffian profile** and **EML routing depth** for any `torch.nn.Module`.

Walks a PyTorch model, classifies each leaf module by mapping it to its
symbolic SymPy equivalent, and runs [`eml-cost`](https://pypi.org/project/eml-cost/)
to produce a Pfaffian chain order, EML routing depth, and canonical
axes-tuple per layer.

## Install

```bash
pip install eml-cost-torch          # core (sympy + eml-cost only)
pip install "eml-cost-torch[torch]" # adds torch>=2.0 for live model walking
```

## Quick start

```python
import torch.nn as nn
from eml_cost_torch import summary, profile

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.GELU(),
    nn.LayerNorm(32),
    nn.Linear(32, 16),
    nn.Sigmoid(),
)

print(summary(model))
```

Output:

```
==========================================================================================
  Per-layer Pfaffian profile  (5 leaf modules)
==========================================================================================
  name                              class                 axes                r     depth
  --------------------------------------------------------------------------------------
  0                                 Linear                p0-d2-w0-c0         r= 0  d=  2
  1                                 GELU                  p3-d5-w3-c0         r= 3  d=  6
  2                                 LayerNorm             p1-d4-w1-c0         r= 1  d=  4
  3                                 Linear                p0-d2-w0-c0         r= 0  d=  2
  4                                 Sigmoid               p1-d1-w1-c-1        r= 1  d=  2
  --------------------------------------------------------------------------------------
  total r (sum across leaves): 5
  max r in any leaf:           3
  max predicted_depth:         6
  distinct cost classes:       4
    p0-d2-w0-c0       x 2
    p3-d5-w3-c0       x 1
    p1-d4-w1-c0       x 1
    p1-d1-w1-c-1      x 1
```

## Programmatic access

```python
from eml_cost_torch import profile, profile_dict

# Returns list of LayerProfile dataclass instances
rows = profile(model)
for r in rows:
    print(r.class_name, r.axes, r.pfaffian_r)

# Or as JSON-friendly dicts
import json
json.dumps(profile_dict(model))
```

## What's measured per layer

| Field | Meaning |
|---|---|
| `pfaffian_r` | Total Pfaffian chain order (Khovanskii convention) |
| `max_path_r` | Chain order along the deepest path |
| `eml_depth` | EML routing tree depth |
| `predicted_depth` | Full predicted depth (chain + structural + corrections) |
| `axes` | Canonical fingerprint `p<r>-d<n>-w<m>-c<k>` |
| `is_pfaffian_not_eml` | True for Bessel, Airy, Lambert W, etc. |

## Supported torch.nn classes

60+ classes registered out of the box, including:

- **Linear / Conv** (`Linear`, `Conv1d/2d/3d`, `ConvTranspose*`)
- **Activations** (`ReLU`, `GELU` exact + tanh-approx, `Sigmoid`,
  `Tanh`, `SiLU`, `Mish`, `ELU`, `SELU`, `Softplus`, `Softsign`,
  `Hardswish`, `Hardsigmoid`, `QuickGELU`, `Softmax`, ...)
- **Gated linear units** (`GLU`, `GeGLU`, `SwiGLU`, `ReGLU` — see
  [README of e.g. LLaMA / Gemma / Mistral architectures](https://arxiv.org/abs/2002.05202))
- **Normalisation** (`LayerNorm`, `BatchNorm*`, `GroupNorm`, `RMSNorm`)
- **Pooling** (`MaxPool*`, `AvgPool*`, adaptive variants)
- **Regularisation** (`Dropout`, `AlphaDropout`)
- **Embedding** (`Embedding`, `EmbeddingBag`)
- **Attention** (`MultiheadAttention` — simplified score)

Unknown layer classes are reported as `UNKNOWN` rather than raising.

## diagnose(model) — empirical fp16 / variance signature

```python
from eml_cost_torch import diagnose
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 128), nn.GELU(approximate="none"),
    nn.Linear(128, 128), nn.SwiGLU(),
    nn.Linear(128, 32),
)
report = diagnose(model)
print(report)
```

`diagnose()` reports per-activation **measured** fp16 drift and
activation variance from the E-192 controlled study (19 activations
× 5 seeds on a fixed FFN). Each activation layer gets:

- `fp16_drift_predicted` — empirical mean relative L2 fp16 drift.
- `activation_variance_predicted` — empirical mean abs std under input
  perturbation.
- `fp16_risk` / `activation_variance_class` — `low` / `normal` /
  `elevated` band relative to the cross-activation median.

Top-5 highest-fp16-drift activations from E-192:

    GeGLU       0.000736
    SwiGLU      0.000707
    ReGLU       0.000665
    QuickGELU   0.000602
    Softsign    0.000576

Bottom-3 (most fp16-stable):

    Sigmoid     0.000229
    Hardsigmoid 0.000226
    Softplus    0.000241

**Honest caveat baked into the report.** On the controlled corpus,
the symbolic Pfaffian-not-EML classification is *not* a reliable
predictor of fp16 drift (Mann-Whitney U p=0.085). The empirical
lookup is the reliable signal. The PNE flag is still reported for
transparency and for symbolic-optimization-cost reasoning.

## See also

- [`eml-cost`](https://pypi.org/project/eml-cost/) — the underlying Pfaffian profile substrate
- [`monogate`](https://pypi.org/project/monogate/) — EML arithmetic, witnesses, CLI
- [monogate.org](https://monogate.org) — research site

## License

PROPRIETARY PRE-RELEASE — see LICENSE.
