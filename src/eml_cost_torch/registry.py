"""Known torch.nn.Module classes mapped to symbolic SymPy equivalents.

The mapping captures the *symbolic structure* of each module's elementwise
output, NOT its full tensor operation. A `nn.Linear(64, 32)` and
`nn.Linear(8, 4)` map to the same symbolic form ``x * W + b`` because
their Pfaffian profile is identical -- the cost model is invariant to
shape, only the operator structure matters.
"""
from __future__ import annotations

import sympy as sp


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
mu = sp.Symbol("mu", real=True)
sigma = sp.Symbol("sigma", positive=True)
eps = sp.Symbol("eps", positive=True)
W = sp.Symbol("W", real=True)
b = sp.Symbol("b", real=True)
alpha = sp.Symbol("alpha", positive=True)
beta = sp.Symbol("beta", positive=True)


# --- Symbolic forms for each layer family ---
LINEAR_FORM = x * W + b  # Linear / Conv*d
ZERO_FORM = x  # identity-shaped (Dropout @ inference, MaxPool, etc.)
RELU_FORM = sp.Max(0, x)  # ReLU — not analytic but symbolic
LEAKY_RELU_FORM = sp.Max(alpha * x, x)
GELU_FORM = sp.S.Half * x * (1 + sp.erf(x / sp.sqrt(2)))
SIGMOID_FORM = 1 / (1 + sp.exp(-x))
TANH_FORM = sp.tanh(x)
SOFTPLUS_FORM = sp.log(1 + sp.exp(x))
ELU_FORM = sp.Piecewise((alpha * (sp.exp(x) - 1), x < 0), (x, True))
SILU_FORM = x / (1 + sp.exp(-x))  # SiLU / Swish
MISH_FORM = x * sp.tanh(sp.log(1 + sp.exp(x)))
SOFTMAX_FORM = sp.exp(x) / (sp.exp(x) + sp.exp(y))  # 2-key approximation
LAYERNORM_FORM = (x - mu) / sp.sqrt(sigma ** 2 + eps)
BATCHNORM_FORM = (x - mu) / sp.sqrt(sigma ** 2 + eps) * alpha + beta
RMSNORM_FORM = x / sp.sqrt(sigma ** 2 + eps)
DROPOUT_FORM = x  # at inference time
EMBEDDING_FORM = W  # table lookup → opaque constant
ATTENTION_SCORE_FORM = sp.exp(x * y) / (sp.exp(x * y) + sp.exp(y))
HARDSWISH_FORM = x * sp.Piecewise(
    (0, x < -3), (1, x > 3), ((x + 3) / 6, True),
)
HARDSIGMOID_FORM = sp.Piecewise(
    (0, x < -3), (1, x > 3), ((x + 3) / 6, True),
)


# --- Registry: torch.nn class name (string) -> symbolic form ---
# Using class names instead of class objects avoids requiring torch
# at import time. Profile dispatches by ``type(module).__name__``.
TORCH_LAYER_REGISTRY: dict[str, sp.Basic] = {
    # Linear family
    "Linear": LINEAR_FORM,
    "Bilinear": LINEAR_FORM,
    # Convolutions
    "Conv1d": LINEAR_FORM,
    "Conv2d": LINEAR_FORM,
    "Conv3d": LINEAR_FORM,
    "ConvTranspose1d": LINEAR_FORM,
    "ConvTranspose2d": LINEAR_FORM,
    "ConvTranspose3d": LINEAR_FORM,
    # Pooling (treated as identity for symbolic structure)
    "MaxPool1d": ZERO_FORM,
    "MaxPool2d": ZERO_FORM,
    "MaxPool3d": ZERO_FORM,
    "AvgPool1d": ZERO_FORM,
    "AvgPool2d": ZERO_FORM,
    "AvgPool3d": ZERO_FORM,
    "AdaptiveMaxPool1d": ZERO_FORM,
    "AdaptiveMaxPool2d": ZERO_FORM,
    "AdaptiveAvgPool1d": ZERO_FORM,
    "AdaptiveAvgPool2d": ZERO_FORM,
    # Activations
    "ReLU": RELU_FORM,
    "ReLU6": RELU_FORM,
    "LeakyReLU": LEAKY_RELU_FORM,
    "PReLU": LEAKY_RELU_FORM,
    "RReLU": LEAKY_RELU_FORM,
    "GELU": GELU_FORM,
    "Sigmoid": SIGMOID_FORM,
    "LogSigmoid": SIGMOID_FORM,
    "Tanh": TANH_FORM,
    "Tanhshrink": TANH_FORM,
    "Softplus": SOFTPLUS_FORM,
    "ELU": ELU_FORM,
    "CELU": ELU_FORM,
    "SELU": ELU_FORM,
    "SiLU": SILU_FORM,
    "Mish": MISH_FORM,
    "Softmax": SOFTMAX_FORM,
    "LogSoftmax": SOFTMAX_FORM,
    "Softmin": SOFTMAX_FORM,
    "Hardtanh": LEAKY_RELU_FORM,
    "Hardswish": HARDSWISH_FORM,
    "Hardsigmoid": HARDSIGMOID_FORM,
    "Hardshrink": LEAKY_RELU_FORM,
    "Softshrink": LEAKY_RELU_FORM,
    # Normalisation
    "LayerNorm": LAYERNORM_FORM,
    "BatchNorm1d": BATCHNORM_FORM,
    "BatchNorm2d": BATCHNORM_FORM,
    "BatchNorm3d": BATCHNORM_FORM,
    "GroupNorm": BATCHNORM_FORM,
    "InstanceNorm1d": BATCHNORM_FORM,
    "InstanceNorm2d": BATCHNORM_FORM,
    "InstanceNorm3d": BATCHNORM_FORM,
    "LocalResponseNorm": LAYERNORM_FORM,
    "RMSNorm": RMSNORM_FORM,
    # Regularisation
    "Dropout": DROPOUT_FORM,
    "Dropout1d": DROPOUT_FORM,
    "Dropout2d": DROPOUT_FORM,
    "Dropout3d": DROPOUT_FORM,
    "AlphaDropout": DROPOUT_FORM,
    # Embedding
    "Embedding": EMBEDDING_FORM,
    "EmbeddingBag": EMBEDDING_FORM,
    # Attention (simplified: 2-key softmax score)
    "MultiheadAttention": ATTENTION_SCORE_FORM,
    # Identity / no-op
    "Identity": ZERO_FORM,
    "Flatten": ZERO_FORM,
    "Unflatten": ZERO_FORM,
    "Reshape": ZERO_FORM,
}


def known_layer_names() -> list[str]:
    """Return sorted list of all torch.nn layer class names with mappings."""
    return sorted(TORCH_LAYER_REGISTRY.keys())


def lookup_form(class_name: str) -> sp.Basic | None:
    """Return the symbolic form for a torch.nn class name, or None if unknown."""
    return TORCH_LAYER_REGISTRY.get(class_name)
