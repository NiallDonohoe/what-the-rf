"""One harness scores every model the same way and writes plots + tables to reports/.

Given several runs (models x seeds), it groups them by config name so SNR curves show
the spread across seeds, and the summary table lists params and training time next
to accuracy. The real-capture set is only scored here via real_vs_synth.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from wtrf.data.dataset import build_splits, load_capture_set
from wtrf.eval import calibration, confusion, real_vs_synth, snr_curves
from wtrf.eval.style import colour_for
from wtrf.models.train import RUNS_DIR, Run, load_run, predict_logits


def _nearest_mask(snr: np.ndarray, target: float) -> np.ndarray:
    levels = np.unique(snr[~np.isnan(snr)])
    return snr == levels[np.argmin(np.abs(levels - target))]


def evaluate_run(run: Run, out: Path, detailed: bool) -> dict[str, object]:
    cfg = run.cfg
    test = build_splits(cfg)["test"]
    logits = predict_logits(run.model, test.x)
    pred = logits.argmax(1)
    curve = snr_curves.accuracy_by_snr(test.y, pred, test.snr)
    (run.dir / "snr_accuracy.json").write_text(
        json.dumps([{"snr_db": s, "accuracy": a, "n_windows": n} for s, a, n in curve], indent=2)
    )

    p_raw, p_cal = calibration.softmax(logits), calibration.softmax(logits, run.temperature)
    bins = cfg.eval.ece_bins
    ece_raw, ece_cal = calibration.ece(p_raw, test.y, bins), calibration.ece(p_cal, test.y, bins)
    hi, lo = _nearest_mask(test.snr, cfg.eval.snr_high), _nearest_mask(test.snr, cfg.eval.snr_low)
    result: dict[str, object] = {
        "run": run.id,
        "label": cfg.name,
        "model": cfg.model.name,
        "seed": cfg.seed,
        "n_params": run.n_params,
        "train_time_s": run.train_time_s,
        "n_test": len(test),
        "acc": float((pred == test.y).mean()),
        "acc_high": float((pred[hi] == test.y[hi]).mean()),
        "snr_high": float(test.snr[hi][0]),
        "acc_low": float((pred[lo] == test.y[lo]).mean()),
        "snr_low": float(test.snr[lo][0]),
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "curve": curve,
    }

    if detailed:
        tag = cfg.name
        k = len(run.classes)
        for name, mask in (("high", hi), ("low", lo)):
            cm = confusion.confusion_matrix(test.y[mask], pred[mask], k)
            snr = float(test.snr[mask][0])
            confusion.write_csv(cm, run.classes, out / f"confusion_{name}_{tag}.csv")
            confusion.plot(
                cm,
                run.classes,
                f"{tag}: confusion at {snr:+g} dB SNR",
                out / f"confusion_{name}_{tag}.png",
            )
        rel_raw = calibration.reliability(p_raw, test.y, bins)
        rel_cal = calibration.reliability(p_cal, test.y, bins)
        calibration.write_csv(
            {"raw": rel_raw, "calibrated": rel_cal}, out / f"calibration_{tag}.csv"
        )
        calibration.plot(
            rel_raw,
            rel_cal,
            ece_raw,
            ece_cal,
            f"{tag}: reliability ({len(test)} windows)",
            out / f"calibration_{tag}.png",
        )

        real = load_capture_set(cfg.captures, run.classes, test.x.shape[-1])
        if len(real):
            rows = real_vs_synth.gap_table(test, pred, real, run.probabilities(real.x).argmax(1))
            real_vs_synth.write_csv(rows, out / f"real_vs_synth_{tag}.csv")
            real_vs_synth.plot(
                rows, f"{tag}: synthetic test vs real captures", out / f"real_vs_synth_{tag}.png"
            )
            result["real_acc"] = rows[-1].real_acc
            result["real_n"] = rows[-1].real_n
    return result


def _fmt(values: list[float], scale: float = 100, digits: int = 1) -> str:
    arr = np.array(values) * scale
    mean = f"{arr.mean():.{digits}f}"
    return f"{mean} ± {arr.std():.{digits}f}" if len(arr) > 1 else mean


def run_eval(runs: list[str], out: Path | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    loaded = [load_run(r, runs_dir) for r in runs]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out or Path("reports") / (loaded[0].id if len(loaded) == 1 else f"compare-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    results = []
    for run in loaded:
        detailed = run.cfg.name not in seen  # full plots for the first seed of each config
        seen.add(run.cfg.name)
        print(f"evaluating {run.id}")
        results.append(evaluate_run(run, out, detailed))

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in results:
        grouped[str(r["label"])].append(r)

    curves = {label: [r["curve"] for r in rs] for label, rs in grouped.items()}
    colours = {
        label: colour_for(str(rs[0]["model"]), i) for i, (label, rs) in enumerate(grouped.items())
    }
    snr_curves.write_csv(curves, out / "snr_curves.csv")  # type: ignore[arg-type]
    snr_curves.plot(curves, colours, out / "snr_curves.png")  # type: ignore[arg-type]

    lines = [
        f"# Evaluation report ({stamp})",
        "",
        "Accuracy in %, mean ± std over seeds. "
        "ECE is expected calibration error (lower is better).",
        "",
        "| config | model | seeds | params | train time (s) | test windows | acc | acc @ high SNR "
        "| acc @ low SNR | ECE raw | ECE calibrated | real acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, rs in grouped.items():

        def col(key: str, rs: list[dict[str, object]] = rs) -> list[float]:
            return [float(r[key]) for r in rs]  # type: ignore[arg-type]

        real = f"{_fmt(col('real_acc'))} (n={rs[0]['real_n']})" if "real_acc" in rs[0] else "–"
        lines.append(
            f"| {label} | {rs[0]['model']} | {len(rs)} | {int(col('n_params')[0]):,} "
            f"| {_fmt(col('train_time_s'), scale=1)} | {rs[0]['n_test']} | {_fmt(col('acc'))} "
            f"| {_fmt(col('acc_high'))} ({rs[0]['snr_high']:+g} dB) "
            f"| {_fmt(col('acc_low'))} ({rs[0]['snr_low']:+g} dB) "
            f"| {_fmt(col('ece_raw'), 1, 3)} | {_fmt(col('ece_cal'), 1, 3)} | {real} |"
        )
    lines += ["", "Runs: " + ", ".join(r.id for r in loaded), ""]
    (out / "summary.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\n".join(lines))
    print(f"reports written to {out}/")
    return out
