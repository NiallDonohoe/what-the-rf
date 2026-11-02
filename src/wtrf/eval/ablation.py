"""Which augmentation closes the gap? Train with each augmentation alone, then all together,
and score every variant on the same frozen real-capture set.

Recovery = (real_acc(variant) - real_acc(none)) / (synth_acc(none) - real_acc(none)):
the fraction of the no-augmentation sim-to-real gap that the variant closes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wtrf.config import Config
from wtrf.data.augment import ORDER
from wtrf.data.dataset import IQDataset, build_splits

VARIANTS: dict[str, list[str]] = {
    "none": [],
    **{name: [name] for name in ORDER},
    "all": list(ORDER),
}


@dataclass
class AblationRow:
    variant: str
    seed: int
    run_id: str
    synth_acc: float
    real_acc: float
    real_n: int


def run_ablation(
    cfg: Config,
    real: IQDataset,
    runs_dir: Path,
    seeds: list[int],
    variants: dict[str, list[str]] | None = None,
) -> list[AblationRow]:
    from wtrf.models.train import load_run, train

    if len(real) == 0:
        raise ValueError(f"no labelled real captures for {real.classes} in {cfg.captures.dir}/")
    rows = []
    for seed in seeds:
        base = cfg.model_copy(update={"seed": seed}, deep=True)
        splits = build_splits(base)  # same data and split for every variant at this seed
        for variant, augs in (variants or VARIANTS).items():
            vcfg = base.model_copy(deep=True)
            vcfg.name = f"{cfg.name}-aug-{variant}"
            vcfg.augment.use = augs
            run = load_run(train(vcfg, runs_dir, splits=splits, verbose=False))
            synth_pred = run.probabilities(splits["test"].x).argmax(1)
            real_pred = run.probabilities(real.x).argmax(1)
            rows.append(
                AblationRow(
                    variant,
                    seed,
                    run.id,
                    float((synth_pred == splits["test"].y).mean()),
                    float((real_pred == real.y).mean()),
                    len(real),
                )
            )
            print(
                f"seed {seed}  {variant:<7} "
                f"synth {rows[-1].synth_acc:.3f}  real {rows[-1].real_acc:.3f}"
            )
    return rows


def summarise(rows: list[AblationRow]) -> list[dict[str, float | str | int]]:
    """Mean/std over seeds per variant, plus recovery against the "none" baseline."""
    by_variant: dict[str, list[AblationRow]] = {}
    for r in rows:
        by_variant.setdefault(r.variant, []).append(r)
    none = by_variant.get("none", [])
    base_gap = (
        np.mean([r.synth_acc for r in none]) - np.mean([r.real_acc for r in none])
        if none
        else np.nan
    )
    base_real = np.mean([r.real_acc for r in none]) if none else np.nan
    out: list[dict[str, float | str | int]] = []
    for variant, rs in by_variant.items():
        real = np.array([r.real_acc for r in rs])
        out.append(
            {
                "variant": variant,
                "seeds": len(rs),
                "real_acc_mean": float(real.mean()),
                "real_acc_std": float(real.std()),
                "synth_acc_mean": float(np.mean([r.synth_acc for r in rs])),
                "real_n": rs[0].real_n,
                "recovery": float((real.mean() - base_real) / base_gap)
                if base_gap > 0
                else float("nan"),
            }
        )
    return out


def write_csv(summary: list[dict[str, float | str | int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


def plot(summary: list[dict[str, float | str | int]], title: str, path: Path) -> None:
    from wtrf.eval import style

    fig, ax = style.figure(6.5, 0.9 + 0.45 * len(summary))
    names = [str(s["variant"]) for s in summary]
    means = [float(s["real_acc_mean"]) for s in summary]
    stds = [float(s["real_acc_std"]) for s in summary]
    y = np.arange(len(summary))[::-1]
    ax.barh(
        y,
        means,
        xerr=stds,
        color=style.SERIES[0],
        height=0.6,
        error_kw={"ecolor": style.TEXT_2, "lw": 1},
    )
    for yi, s in zip(y, summary, strict=True):
        rec = float(s["recovery"])
        label = f"{float(s['real_acc_mean']):.3f}" + (
            "" if np.isnan(rec) else f"  ({rec:+.0%} of gap)"
        )
        ax.annotate(
            label,
            (float(s["real_acc_mean"]), yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=style.TEXT_2,
        )
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(f"Accuracy on real captures (n={summary[0]['real_n']} windows)")
    style.finish(fig, ax, title, path)
