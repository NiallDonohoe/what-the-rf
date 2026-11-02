"""Plain PyTorch training loop: AdamW, cosine schedule, per-SNR validation every epoch.

A run directory holds everything needed to reproduce or score a result:
  config.yaml     the exact config (with seed)
  model.pt        weights, class list, fitted temperature, param count, train time
  metrics.jsonl   one line per epoch
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wtrf.config import Config, load_config, save_config
from wtrf.data import augment
from wtrf.data.dataset import IQDataset, build_splits, normalize
from wtrf.eval.calibration import fit_temperature, softmax
from wtrf.eval.snr_curves import accuracy_by_snr
from wtrf.models.registry import build_model, count_params

RUNS_DIR = Path("runs")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code that uses the global RNG
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str) -> torch.device:
    """auto = CUDA, then Apple MPS, then CPU. Results are bit-exact across reruns on CPU only."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_logits(
    model: nn.Module, x: np.ndarray, device: torch.device | None = None, batch_size: int = 1024
) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = [
        model(torch.from_numpy(x[i : i + batch_size]).to(device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class _Tracker:
    """Local JSONL always; MLflow too when the config asks for it."""

    def __init__(self, cfg: Config, run_dir: Path) -> None:
        self.file = (run_dir / "metrics.jsonl").open("w")
        self.mlflow: Any = None
        if cfg.tracking == "mlflow":
            import mlflow

            mlflow.set_experiment(cfg.name)
            mlflow.start_run(run_name=run_dir.name)
            mlflow.log_params({"model": cfg.model.name, "seed": cfg.seed, **cfg.model.params})
            self.mlflow = mlflow

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.file.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        self.file.flush()
        if self.mlflow:
            flat = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
            flat |= {f"val_acc_snr_{s:+g}": a for s, a in metrics.get("val_acc_by_snr", {}).items()}
            self.mlflow.log_metrics(flat, step=epoch)

    def close(self, run_dir: Path) -> None:
        self.file.close()
        if self.mlflow:
            self.mlflow.log_artifacts(str(run_dir))
            self.mlflow.end_run()


def _snr_dict(y: np.ndarray, pred: np.ndarray, snr: np.ndarray) -> dict[str, float]:
    return {f"{s:g}": round(a, 4) for s, a, _ in accuracy_by_snr(y, pred, snr)}


def train(
    cfg: Config,
    runs_dir: Path = RUNS_DIR,
    splits: dict[str, IQDataset] | None = None,
    verbose: bool = True,
) -> Path:
    """Train one model from `cfg` and return its run directory."""
    set_seed(cfg.seed)
    device = pick_device(cfg.train.device)
    splits = splits or build_splits(cfg)
    tr, va = splits["train"], splits["val"]

    model = build_model(cfg.model.name, len(tr.classes), tr.x.shape[-1], **cfg.model.params)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = max(1, int(np.ceil(len(tr) / cfg.train.batch_size)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs * steps_per_epoch
    )
    loss_fn = nn.CrossEntropyLoss()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"{cfg.name}-s{cfg.seed}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(cfg, run_dir / "config.yaml")
    tracker = _Tracker(cfg, run_dir)
    rng = np.random.default_rng(cfg.seed)

    t0 = time.perf_counter()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        total, seen = 0.0, 0
        for i in range(0, len(order), cfg.train.batch_size):
            idx = order[i : i + cfg.train.batch_size]
            xb = normalize(augment.apply(tr.x[idx], cfg.augment.use, cfg.augment, rng))
            xt, yt = torch.from_numpy(xb).to(device), torch.from_numpy(tr.y[idx]).to(device)
            loss = loss_fn(model(xt), yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
            seen += len(idx)

        pred = predict_logits(model, va.x, device).argmax(1)
        metrics: dict[str, Any] = {
            "train_loss": round(total / seen, 5),
            "val_acc": round(float((pred == va.y).mean()), 4),
            "val_acc_by_snr": _snr_dict(va.y, pred, va.snr),
            "lr": sched.get_last_lr()[0],
        }
        tracker.log(epoch, metrics)
        if verbose:
            by_snr = " ".join(f"{s}:{a:.2f}" for s, a in metrics["val_acc_by_snr"].items())
            print(
                f"epoch {epoch:>3}  loss {metrics['train_loss']:.4f}  "
                f"val {metrics['val_acc']:.3f}  [{by_snr}]"
            )
    train_time = time.perf_counter() - t0

    temperature = fit_temperature(predict_logits(model, va.x, device), va.y)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(tr.classes),
            "n_samples": int(tr.x.shape[-1]),
            "temperature": temperature,
            "n_params": count_params(model),
            "train_time_s": round(train_time, 2),
        },
        run_dir / "model.pt",
    )
    tracker.close(run_dir)
    if verbose:
        print(
            f"saved {run_dir}  ({count_params(model):,} params, "
            f"{train_time:.1f}s, T={temperature:.3f})"
        )
    return run_dir


@dataclass
class Run:
    dir: Path
    cfg: Config
    model: nn.Module
    classes: tuple[str, ...]
    n_samples: int
    temperature: float
    n_params: int
    train_time_s: float

    @property
    def id(self) -> str:
        return self.dir.name

    def probabilities(self, x: np.ndarray, calibrated: bool = True) -> np.ndarray:
        return softmax(predict_logits(self.model, x), self.temperature if calibrated else 1.0)


def resolve_run(run: str | Path, runs_dir: Path = RUNS_DIR) -> Path:
    """Accept a run directory, a run id, or "latest"."""
    p = Path(run)
    if p.is_dir():
        return p
    if str(run) == "latest":
        runs = sorted(
            (d for d in runs_dir.glob("*") if (d / "model.pt").exists()),
            key=lambda d: (d / "model.pt").stat().st_mtime,
        )
        if not runs:
            raise FileNotFoundError(f"no trained runs in {runs_dir}/")
        return runs[-1]
    if (runs_dir / str(run)).is_dir():
        return runs_dir / str(run)
    raise FileNotFoundError(f"run {run!r} not found (looked in {runs_dir}/)")


def load_run(run: str | Path, runs_dir: Path = RUNS_DIR, device: str = "cpu") -> Run:
    run_dir = resolve_run(run, runs_dir)
    cfg = load_config(run_dir / "config.yaml")
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = build_model(cfg.model.name, len(ckpt["classes"]), ckpt["n_samples"], **cfg.model.params)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return Run(
        dir=run_dir,
        cfg=cfg,
        model=model,
        classes=tuple(ckpt["classes"]),
        n_samples=int(ckpt["n_samples"]),
        temperature=float(ckpt["temperature"]),
        n_params=int(ckpt["n_params"]),
        train_time_s=float(ckpt["train_time_s"]),
    )
