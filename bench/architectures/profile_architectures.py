"""Profile GPT-2, BERT, ResNet-50, ViT with eml-cost-torch.

Outputs:
  bench/architectures/{model}.json    — full per-layer profile
  bench/architectures/{model}.png     — visualization
  bench/architectures/summary.json    — across-architecture comparison

Models are instantiated with default configs (random weights). We do
not need to run forward; we only need the architecture tree.

Reproducibility: pip install eml-cost-torch torch transformers torchvision
matplotlib, then run this file. Output JSON is byte-stable across runs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eml_cost_torch import profile_dict


HERE = Path(__file__).parent


def load_gpt2():
    from transformers import GPT2Config, GPT2Model
    cfg = GPT2Config()  # default 124M
    return "gpt2", GPT2Model(cfg)


def load_bert():
    from transformers import BertConfig, BertModel
    cfg = BertConfig()  # default base 110M
    return "bert-base", BertModel(cfg)


def load_resnet50():
    from torchvision.models import resnet50
    return "resnet-50", resnet50(weights=None)


def load_vit():
    from torchvision.models import vit_b_16
    return "vit-b-16", vit_b_16(weights=None)


LOADERS = [load_gpt2, load_bert, load_resnet50, load_vit]


def summarize_profile(rows: list[dict]) -> dict:
    """Aggregate stats from a per-layer profile."""
    n = len(rows)
    known = [r for r in rows if not r["is_unknown"]]
    unknown = [r for r in rows if r["is_unknown"]]

    rs = [r["pfaffian_r"] for r in known]
    ds = [r["eml_depth"] for r in known]
    pds = [r["predicted_depth"] for r in known]

    axes_dist = Counter(r["axes"] for r in known)
    class_dist = Counter(r["class_name"] for r in rows)
    unknown_classes = Counter(r["class_name"] for r in unknown)

    pfaffian_not_eml_count = sum(1 for r in known if r["is_pfaffian_not_eml"])

    return {
        "n_layers_total": n,
        "n_layers_known": len(known),
        "n_layers_unknown": len(unknown),
        "coverage_pct": round(100 * len(known) / max(1, n), 1),
        "pfaffian_r": {
            "min": min(rs) if rs else None,
            "max": max(rs) if rs else None,
            "mean": round(sum(rs) / len(rs), 2) if rs else None,
            "distribution": dict(Counter(rs)),
        },
        "eml_depth": {
            "min": min(ds) if ds else None,
            "max": max(ds) if ds else None,
            "mean": round(sum(ds) / len(ds), 2) if ds else None,
            "distribution": dict(Counter(ds)),
        },
        "predicted_depth": {
            "min": min(pds) if pds else None,
            "max": max(pds) if pds else None,
            "mean": round(sum(pds) / len(pds), 2) if pds else None,
        },
        "axes_distribution": dict(axes_dist.most_common()),
        "class_distribution": dict(class_dist.most_common()),
        "unknown_class_distribution": dict(unknown_classes.most_common()),
        "pfaffian_not_eml_count": pfaffian_not_eml_count,
    }


def visualize(name: str, rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    known = [r for r in rows if not r["is_unknown"]]
    if not known:
        return

    rs = [r["pfaffian_r"] for r in known]
    ds = [r["eml_depth"] for r in known]
    layer_idx = list(range(len(known)))
    classes = [r["class_name"] for r in known]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top: per-layer pfaffian_r and eml_depth
    ax1 = axes[0]
    ax1.plot(layer_idx, rs, "o-", color="#d4a76a",
             label="pfaffian_r (chain order)", markersize=3, linewidth=0.8)
    ax1.plot(layer_idx, ds, "s-", color="#7e9bd1",
             label="eml_depth", markersize=3, linewidth=0.8, alpha=0.7)
    # Pfaffian-not-EML marker (e.g., GELU-with-erf)
    notels = [(i, rs[i]) for i, r in enumerate(known) if r["is_pfaffian_not_eml"]]
    if notels:
        nx, ny = zip(*notels)
        ax1.scatter(nx, ny, color="#e07070", marker="*", s=80, zorder=5,
                    label="Pfaffian-not-EML", edgecolors="white", linewidths=0.5)
    ax1.set_ylabel("Pfaffian / depth")
    ax1.set_title(f"{name}: per-layer Pfaffian profile (n={len(known)} known + {len(rows) - len(known)} unknown)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Bottom: class strip
    ax2 = axes[1]
    unique_classes = sorted(set(classes))
    cls_to_y = {c: i for i, c in enumerate(unique_classes)}
    ys = [cls_to_y[c] for c in classes]
    ax2.scatter(layer_idx, ys, c=ys, cmap="tab20", s=18, edgecolors="none")
    ax2.set_yticks(range(len(unique_classes)))
    ax2.set_yticklabels(unique_classes, fontsize=8)
    ax2.set_xlabel("layer index (forward order)")
    ax2.set_ylabel("layer class")
    ax2.grid(True, alpha=0.15, axis="x")

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summaries = {}
    for loader in LOADERS:
        name, model = loader()
        print(f"Profiling {name}...")
        rows = profile_dict(model)
        summary = summarize_profile(rows)
        print(f"  {summary['n_layers_total']} layers "
              f"({summary['n_layers_known']} known, "
              f"{summary['n_layers_unknown']} unknown), "
              f"coverage {summary['coverage_pct']}%")
        print(f"  pfaffian_r: min={summary['pfaffian_r']['min']}, "
              f"max={summary['pfaffian_r']['max']}, "
              f"mean={summary['pfaffian_r']['mean']}")
        print(f"  axes top-3: "
              f"{list(summary['axes_distribution'].items())[:3]}")
        print(f"  Pfaffian-not-EML layer count: "
              f"{summary['pfaffian_not_eml_count']}")

        out_json = HERE / f"{name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"name": name, "summary": summary, "rows": rows},
                      f, indent=2)
        print(f"  wrote {out_json.name}")

        out_png = HERE / f"{name}.png"
        visualize(name, rows, out_png)
        print(f"  wrote {out_png.name}")

        summaries[name] = summary
        print()

    # Cross-architecture summary
    out_summary = HERE / "summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "architectures": list(summaries.keys()),
            "per_architecture": summaries,
            "comparison_table": [
                {
                    "name": n,
                    "n_layers": s["n_layers_total"],
                    "coverage_pct": s["coverage_pct"],
                    "pfaffian_r_max": s["pfaffian_r"]["max"],
                    "eml_depth_max": s["eml_depth"]["max"],
                    "predicted_depth_max": s["predicted_depth"]["max"],
                    "pfaffian_not_eml_count": s["pfaffian_not_eml_count"],
                    "n_axes_classes": len(s["axes_distribution"]),
                    "top_axes": list(s["axes_distribution"].keys())[:3],
                }
                for n, s in summaries.items()
            ],
        }, f, indent=2)
    print(f"Wrote {out_summary.name}")


if __name__ == "__main__":
    main()
