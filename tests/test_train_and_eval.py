import json

import numpy as np

from wtrf.eval import calibration
from wtrf.eval.harness import run_eval
from wtrf.eval.snr_curves import accuracy_at, accuracy_by_snr
from wtrf.models.train import load_run, train


def test_same_config_and_seed_give_same_metrics(tiny_cfg, tmp_path):
    a = train(tiny_cfg, tmp_path / "a", verbose=False)
    b = train(tiny_cfg, tmp_path / "b", verbose=False)
    assert (a / "metrics.jsonl").read_text() == (b / "metrics.jsonl").read_text()


def test_run_round_trip_and_harness_writes_reports(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # harness looks for captures/ relative to cwd
    run_dir = train(tiny_cfg, tmp_path / "runs", verbose=False)
    run = load_run(run_dir)
    assert run.classes == ("BPSK", "AM-DSB", "WBFM") and run.n_samples == 64
    out = run_eval([str(run_dir)], tmp_path / "reports")
    for name in (
        "snr_curves.png",
        "snr_curves.csv",
        "confusion_high_tiny.png",
        "calibration_tiny.png",
        "summary.md",
    ):
        assert (out / name).exists(), name
    curve = json.loads((run_dir / "snr_accuracy.json").read_text())
    assert {row["snr_db"] for row in curve} == {10.0, 18.0}


def test_accuracy_by_snr_and_nearest_bucket():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    snr = np.array([0.0, 0.0, 10.0, np.nan])
    curve = accuracy_by_snr(y, pred, snr)
    assert curve == [(0.0, 0.5, 2), (10.0, 1.0, 1)]
    assert accuracy_at(curve, 7.0)[0] == 10.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 predictions at 0.8 confidence, 8 correct.
    probs = np.tile([0.8, 0.2], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    assert calibration.ece(probs, y) < 1e-9


def test_temperature_scaling_softens_overconfident_logits(rng):
    y = rng.integers(0, 3, 2000)
    logits = rng.standard_normal((2000, 3))
    logits[np.arange(2000), y] += 1.0
    logits *= 6  # overconfident
    t = calibration.fit_temperature(logits, y)
    assert t > 2
    assert calibration.ece(calibration.softmax(logits, t), y) < calibration.ece(
        calibration.softmax(logits), y
    )


def _fake_captures(directory, rng):
    """Synthetic stand-ins for real captures, with a carrier offset the training data lacks."""
    from wtrf.data import augment, sigmf, synth

    for cls, label in (("WBFM", "fm_broadcast"), ("OOK", "cw")):
        for session in range(2):
            z = synth.add_noise(synth.clean(cls, 1, rng, 64 * 20), 12.0, rng)
            z = augment.cfo(z, rng, augment.AugmentConfig(cfo_max=0.05))[0]
            sigmf.write(
                directory / f"{label}-{session}",
                z.astype("complex64"),
                48_000,
                0.0,
                label=label,
                session=f"{label}-{session}",
            )


def test_real_vs_synth_and_ablation_on_stand_in_captures(tiny_cfg, tmp_path, monkeypatch, rng):
    from wtrf.data.dataset import load_capture_set
    from wtrf.eval import ablation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "captures").mkdir()
    _fake_captures(tmp_path / "captures", rng)
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.data.classes = ["WBFM", "OOK"]

    out = run_eval([str(train(cfg, tmp_path / "runs", verbose=False))], tmp_path / "reports")
    gap = (out / "real_vs_synth_tiny.csv").read_text().splitlines()
    assert gap[0].startswith("class,synth_acc") and gap[-1].startswith("ALL,")

    real = load_capture_set(cfg.captures, ("WBFM", "OOK"), 64)
    assert len(real) == 80 and len(set(real.groups)) == 4
    variants = {"none": [], "cfo": ["cfo"]}
    rows = ablation.run_ablation(cfg, real, tmp_path / "runs", [0], variants)
    summary = ablation.summarise(rows)
    assert [s["variant"] for s in summary] == ["none", "cfo"]
    assert all(s["real_n"] == 80 for s in summary)
    ablation.plot(summary, "ablation", tmp_path / "ablation.png")
