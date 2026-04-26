"""Pretty-print a torch.nn.Module's per-layer Pfaffian profile."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .classify import LayerProfile
from .profile import profile


def _fmt_row(name: str, class_name: str, axes: str, r: int, depth: int,
             unknown: bool) -> str:
    if unknown:
        return f"  {name:<32}  {class_name:<20}  UNKNOWN              -    -"
    return (f"  {name:<32}  {class_name:<20}  {axes:<18}  "
            f"r={r:>2}  d={depth:>3}")


def summary(module: Any, *, show_aggregate: bool = True) -> str:
    """Return a multi-line text summary of the per-layer Pfaffian profile.

    Parameters
    ----------
    module : torch.nn.Module
        Any PyTorch module.
    show_aggregate : bool, default True
        Append a per-axes-tuple counter and total-r summary at the bottom.

    Returns
    -------
    str
        Newline-separated text suitable for print().

    Examples
    --------
    >>> import torch.nn as nn
    >>> from eml_cost_torch import summary
    >>> model = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())
    >>> print(summary(model))  # doctest: +SKIP
    """
    rows = profile(module)
    return _format_rows(rows, show_aggregate=show_aggregate)


def _format_rows(rows: list[LayerProfile], *,
                 show_aggregate: bool = True) -> str:
    out = []
    out.append("=" * 90)
    out.append(f"  Per-layer Pfaffian profile  ({len(rows)} leaf modules)")
    out.append("=" * 90)
    out.append(f"  {'name':<32}  {'class':<20}  {'axes':<18}  r     depth")
    out.append("  " + "-" * 86)
    for r in rows:
        out.append(_fmt_row(
            name=r.name[:32],
            class_name=r.class_name[:20],
            axes=r.axes,
            r=r.pfaffian_r,
            depth=r.predicted_depth,
            unknown=r.is_unknown,
        ))
    if not show_aggregate:
        return "\n".join(out)
    out.append("  " + "-" * 86)
    known = [r for r in rows if not r.is_unknown]
    n_unknown = len(rows) - len(known)
    if known:
        total_r = sum(r.pfaffian_r for r in known)
        max_r = max(r.pfaffian_r for r in known)
        max_d = max(r.predicted_depth for r in known)
        ax_counts = Counter(r.axes for r in known)
        out.append(f"  total r (sum across leaves): {total_r}")
        out.append(f"  max r in any leaf:           {max_r}")
        out.append(f"  max predicted_depth:         {max_d}")
        out.append(f"  distinct cost classes:       {len(ax_counts)}")
        for ax, n in ax_counts.most_common():
            out.append(f"    {ax:<18}  x {n}")
    if n_unknown:
        out.append(f"  unknown classes: {n_unknown} "
                   f"(no registry entry; report as feature request)")
    return "\n".join(out)
