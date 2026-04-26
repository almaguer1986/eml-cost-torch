"""Walk a torch.nn.Module and produce per-layer Pfaffian profiles."""
from __future__ import annotations

from typing import Any

from .classify import LayerProfile, classify_layer


def _iter_named_leaves(module: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Yield (qualified_name, leaf_module) pairs.

    A leaf is a module with no children -- the actual computational
    units. Container modules (Sequential, ModuleList, ModuleDict) are
    descended into but not themselves classified.
    """
    children = list(module.named_children())
    if not children:
        return [(prefix or type(module).__name__, module)]
    out: list[tuple[str, Any]] = []
    for child_name, child in children:
        sub_prefix = f"{prefix}.{child_name}" if prefix else child_name
        out.extend(_iter_named_leaves(child, sub_prefix))
    return out


def profile(module: Any) -> list[LayerProfile]:
    """Return a per-layer Pfaffian profile of a torch.nn.Module.

    Walks the module tree, classifying each leaf module via the
    registry. Returns a list of :class:`LayerProfile` records in
    forward-pass order.

    Parameters
    ----------
    module : torch.nn.Module
        Any PyTorch module. Containers (Sequential, ModuleList, etc.)
        are descended into automatically.

    Returns
    -------
    list[LayerProfile]
        One record per leaf module.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from eml_cost_torch import profile
    >>> model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    >>> rows = profile(model)
    >>> [r.class_name for r in rows]
    ['Linear', 'GELU', 'Linear']
    """
    leaves = _iter_named_leaves(module)
    return [classify_layer(name, leaf) for name, leaf in leaves]


def profile_dict(module: Any) -> list[dict[str, Any]]:
    """Same as :func:`profile` but returns plain dicts (JSON-friendly)."""
    return [p.to_dict() for p in profile(module)]
