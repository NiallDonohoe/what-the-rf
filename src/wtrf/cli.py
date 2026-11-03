"""wtrf: point it at a signal, find out what the RF it is."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _train(args: argparse.Namespace) -> None:
    from wtrf.config import load_config
    from wtrf.eval.harness import run_eval
    from wtrf.models.train import train

    cfg = load_config(args.config)
    seeds = args.seeds or [cfg.seed]
    run_dirs = [train(cfg.model_copy(update={"seed": s}, deep=True)) for s in seeds]
    if args.eval:
        run_eval([str(d) for d in run_dirs])


def _eval(args: argparse.Namespace) -> None:
    from wtrf.eval.harness import run_eval

    run_eval(args.run, Path(args.out) if args.out else None)


def _ablate(args: argparse.Namespace) -> None:
    from wtrf.config import load_config
    from wtrf.data.dataset import load_capture_set
    from wtrf.data.labels import resolve_classes
    from wtrf.eval import ablation
    from wtrf.models.train import RUNS_DIR

    cfg = load_config(args.config)
    real = load_capture_set(cfg.captures, resolve_classes(cfg.data.classes), cfg.data.n_samples)
    rows = ablation.run_ablation(cfg, real, RUNS_DIR, args.seeds or [cfg.seed])
    summary = ablation.summarise(rows)
    out = Path(args.out or f"reports/ablation-{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    ablation.write_csv(summary, out / "ablation.csv")
    ablation.plot(
        summary, f"{cfg.name}: accuracy on real captures by augmentation", out / "ablation.png"
    )
    for s in summary:
        print(
            f"{s['variant']:<7} real {s['real_acc_mean']:.3f} ± {s['real_acc_std']:.3f}  "
            f"recovery {s['recovery']:+.0%}"
        )
    print(f"reports written to {out}/")


def _classify(args: argparse.Namespace) -> None:
    from wtrf.models.train import load_run
    from wtrf.pipeline import classify_capture

    run = load_run(args.run)
    c = classify_capture(run, args.capture)
    if args.json:
        print(json.dumps({"run": c.run_id, "top": c.top(3), "features": c.features.as_dict()}))
        return
    f = c.features
    print(f"{args.capture}  ({c.meta.n_windows} windows, model {c.run_id})")
    for label, p in c.top(3):
        print(f"  {label:<8} {p:6.1%}  {'█' * round(p * 30)}")
    print(
        f"  centre {f.centre_freq_hz / 1e6:.6f} MHz  bandwidth {f.occupied_bandwidth_hz:,.0f} Hz  "
        f"SNR≈{f.snr_db_est:.1f} dB  duration {f.duration_s:.2f} s"
    )


def _explain(args: argparse.Namespace) -> None:
    from wtrf.models.train import load_run
    from wtrf.pipeline import classify_capture, explain_classification

    run = load_run(args.run)
    c = classify_capture(run, args.capture)
    out = explain_classification(c, run)
    if args.json:
        print(json.dumps(out.as_dict(), indent=2))
        return
    print(out.verdict)
    for line in out.evidence:
        print(f"  - {line}")
    print(out.confidence_note)
    if out.alternatives:
        print("Alternatives:")
        for line in out.alternatives:
            print(f"  - {line}")


def _synth(args: argparse.Namespace) -> None:
    from wtrf.data import sigmf, synth
    from wtrf.data.labels import CAPTURE_TO_CANONICAL

    rng = np.random.default_rng(args.seed)
    out = str(args.out)
    if out.endswith(".npy"):
        np.save(out, synth.generate(args.cls, args.n, args.snr, rng, args.n_samples))
        print(f"wrote {args.n} x (2, {args.n_samples}) {args.cls} at {args.snr} dB to {out}")
        return
    # Anything else: a SigMF capture, handy for trying classify/explain without an SDR.
    length = args.n * args.n_samples
    z = synth.add_noise(synth.clean(args.cls, 1, rng, length), args.snr, rng)[0]
    reverse = {v: k for k, v in CAPTURE_TO_CANONICAL.items()}
    base = sigmf.write(
        out,
        z.astype(np.complex64),
        args.rate,
        args.freq,
        label=reverse.get(args.cls),
        session=f"synthetic-{args.seed}",
        sdr="synthetic",
    )
    print(f"wrote {length} samples of {args.cls} at {args.snr} dB to {base}.sigmf-meta/-data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wtrf", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("train", help="train a model; writes a run directory under runs/")
    s.add_argument("--config", required=True)
    s.add_argument(
        "--seeds", type=int, nargs="+", help="train once per seed (default: config seed)"
    )
    s.add_argument("--eval", action="store_true", help="run the evaluation harness afterwards")
    s.set_defaults(func=_train)

    s = sub.add_parser("eval", help="run the evaluation harness; writes reports/")
    s.add_argument("--run", nargs="+", required=True, help="run ids or dirs (several = compare)")
    s.add_argument("--out", help="report directory")
    s.set_defaults(func=_eval)

    s = sub.add_parser("ablate", help="train one variant per augmentation, score on real captures")
    s.add_argument("--config", required=True)
    s.add_argument("--seeds", type=int, nargs="+")
    s.add_argument("--out")
    s.set_defaults(func=_ablate)

    for name, func, help_ in (
        ("classify", _classify, "print the top classes for a SigMF capture"),
        ("explain", _explain, "classify, then explain the result from measured features"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("capture", help="path to .sigmf-meta, .sigmf-data or their base name")
        s.add_argument("--run", default="latest")
        s.add_argument("--json", action="store_true")
        s.set_defaults(func=func)

    s = sub.add_parser("synth", help="generate synthetic samples (.npy) or a SigMF capture")
    s.add_argument("--class", dest="cls", required=True)
    s.add_argument("--snr", type=float, default=10.0)
    s.add_argument("--out", required=True, help="x.npy for a batch, anything else for SigMF")
    s.add_argument("--n", type=int, default=1, help="examples (.npy) or windows of length (SigMF)")
    s.add_argument("--n-samples", type=int, default=128)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rate", type=float, default=200e3, help="SigMF sample rate, Hz")
    s.add_argument("--freq", type=float, default=100e6, help="SigMF centre frequency, Hz")
    s.set_defaults(func=_synth)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"wtrf: error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
