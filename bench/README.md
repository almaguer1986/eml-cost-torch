# `eml-cost-torch` benchmarks

Reproducible benchmarks for the published `eml-cost-torch` package.

## Setup

```
pip install eml-cost eml-cost-torch torch transformers torchvision matplotlib
```

## Bench: Per-layer Pfaffian profile of production architectures

**Run:** `python bench/architectures/profile_architectures.py`

**Outputs (per architecture):**
  - `bench/architectures/{model}.json` — full per-layer profile (every
    leaf module + its Pfaffian axes-tuple)
  - `bench/architectures/{model}.png` — visualization (per-layer r and
    depth, with Pfaffian-not-EML markers)
  - `bench/architectures/summary.json` — cross-architecture comparison

**Reproducibility notebook:** `bench/architectures/reproduce.ipynb`
(run top-to-bottom in Colab or local Jupyter).

**Methodology:** instantiate each architecture from default config (no
weights download needed). Walk `model.named_modules()`, classify each
leaf via the registry, output per-layer Pfaffian profile.

**Results (4 architectures, 513 leaf modules total):**

| Architecture | Total layers | Coverage | r_max | Pfaffian-not-EML | Top axes class |
|---|---|---|---|---|---|
| GPT-2 (124M) | 124 | 51.6% | 1 | 0 | `p0-d0-w0-c0` (39), `p1-d4-w1-c0` (25) |
| BERT-base (110M) | 151 | 92.1% | 1 | 0 | `p0-d2-w0-c0` (73), `p0-d0-w0-c0` (40) |
| ResNet-50 (25M) | 126 | 100.0% | 1 | 0 | `p0-d2-w0-c0` (54), `p1-d5-w1-c0` (53) |
| **ViT-B/16 (86M)** | 112 | 100.0% | **3** | **12** | `p0-d2-w0-c0` (38), `p0-d0-w0-c0` (37) |

**What this tells you:**

  - **Layer-level vs block-level distinction.** Most leaf modules sit
    at r=0 (Linear, Conv) or r=1 (LayerNorm, BatchNorm). The cross-
    modal v3 substrate run on the *composed* Transformer block found
    `p7-d8-w3-c0` for the full block — that's the *blockwise* result,
    distinct from this *componentwise* table.
  - **ViT has 12 Pfaffian-not-EML layers.** Each is a GELU activation
    that contains erf, which is Pfaffian-class but outside the
    elementary-function family. The ViT score of 12 is the highest of
    the four architectures because ViT uses GELU exclusively in its
    MLP blocks.
  - **GPT-2's lower coverage is honest.** HuggingFace's GPT-2 uses a
    custom `Conv1D` class (not `nn.Linear`) that isn't in the registry.
    Adding it would push coverage to ~95%. This is exactly the kind of
    coverage gap the package's `is_unknown` field is designed to
    surface, not hide.
  - **ResNet-50's `p1-d5-w1-c0` (53 hits)** is the BatchNorm + ReLU
    combination — same cost class as the cross-modal `p1-d5-w1-c0`
    "protective shell" finding (Gaussian / saturating receptor).
    Honest cross-domain agreement.

**Pfaffian profile vs alternatives:**

  - **vs `torchinfo` / `ptflops` / `fvcore`**: those measure FLOPs +
    parameter count (runtime-cost proxies). `eml-cost-torch` measures
    *symbolic complexity class* — orthogonal axis. They are
    complementary, not competing.
  - **vs activation-pattern analysis (LIME, attention rollout)**: those
    require a forward pass + sample data. `eml-cost-torch` requires only
    the architecture definition; no weights, no data, no forward.

## Caveats

  - Architectures profiled with default config (random weights). The
    profile depends only on architecture, not on training, so this
    matches what `from_pretrained()` would give you.
  - Coverage gaps for custom modules (HuggingFace `Conv1D`,
    architecture-specific norms) are documented in
    `summary.json["per_architecture"][name]["unknown_class_distribution"]`.
  - Single-process, CPU; profiling is essentially instantaneous (under
    1 second per architecture).

## Reproducibility

```
git clone https://github.com/almaguer1986/eml-cost-torch
cd eml-cost-torch
pip install -e . torch transformers torchvision matplotlib
python bench/architectures/profile_architectures.py
```

Or open `bench/architectures/reproduce.ipynb` in Colab.

## See also

  - `bench/` (in `eml-cost` package) — speed benchmark
    (`analyze` vs `sympy.simplify`).
  - Cross-modal v3 substrate run in `monogate-research/exploration/`
    for the *blockwise* Transformer profile (`p7-d8-w3-c0`).
