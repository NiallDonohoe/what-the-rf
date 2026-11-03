"""What the RF? Streamlit demo. Run with: streamlit run app/main.py

Calls the same pipeline functions as `wtrf classify` / `wtrf explain`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from wtrf.data import sigmf  # noqa: E402
from wtrf.eval import style  # noqa: E402
from wtrf.features import spectrogram  # noqa: E402
from wtrf.models.train import RUNS_DIR, load_run  # noqa: E402
from wtrf.pipeline import classify_capture, explain_classification  # noqa: E402

EXAMPLE_DIRS = (Path("captures"), Path("examples"))

st.set_page_config(page_title="What the RF?", page_icon="📡", layout="centered")
st.title("What the RF?")
st.caption("Point it at a signal. Find out what the RF it is.")

runs = sorted(
    (d.name for d in RUNS_DIR.glob("*") if (d / "model.pt").exists()),
    key=lambda n: (RUNS_DIR / n / "model.pt").stat().st_mtime,
    reverse=True,
)
if not runs:
    st.error("No trained runs found. Train one first: `wtrf train --config configs/cnn.yaml`.")
    st.stop()

with st.sidebar:
    run_id = st.selectbox("Model run", runs)
    show_reliability = st.toggle("Show accuracy at this SNR", value=True)


@st.cache_resource
def get_run(run_id: str):  # noqa: ANN201
    return load_run(run_id)


run = get_run(run_id)

examples = [p for d in EXAMPLE_DIRS if d.exists() for p in sigmf.find_captures(d)]
source = st.radio("Capture", ["Pick an example", "Upload SigMF"], horizontal=True)
capture: sigmf.Capture | None = None
if source == "Pick an example":
    if examples:
        choice = st.selectbox("Example", examples, format_func=lambda p: str(p))
        capture = sigmf.read(choice)
    else:
        st.info("No examples yet. Try `wtrf synth --class WBFM --n 200 --out examples/wbfm`.")
else:
    files = st.file_uploader(
        "Upload the .sigmf-meta and .sigmf-data pair",
        type=["sigmf-meta", "sigmf-data"],
        accept_multiple_files=True,
    )
    if files and len(files) == 2:
        tmp = Path(tempfile.mkdtemp())
        for f in files:
            (tmp / ("upload" + Path(f.name).suffix)).write_bytes(f.getvalue())
        try:
            capture = sigmf.read(tmp / "upload")
        except (KeyError, ValueError, FileNotFoundError) as exc:
            st.error(f"Could not read that capture: {exc}")

if capture is None:
    st.stop()

with st.spinner("Classifying..."):
    try:
        result = classify_capture(run, capture)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

# Spectrogram of what the model actually saw (after resampling to the model rate).
f, t, s = spectrogram(result.iq, result.fs)
fig, ax = style.figure(7, 3.2)
ax.grid(False)
mesh = ax.pcolormesh(
    t, (f + result.features.centre_freq_hz) / 1e6, s, shading="auto", cmap=style.SEQUENTIAL
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (MHz)")
fig.colorbar(mesh, ax=ax, label="dB")
style.decorate(fig, ax, f"Spectrogram ({result.meta.n_windows} windows)")
st.pyplot(fig)
plt.close(fig)

top = result.top(3)
st.subheader(f"Predicted modulation: {top[0][0]}")
for label, p in top:
    st.progress(p, text=f"{label}: {p:.1%}")

feats = result.features
cols = st.columns(3)
cols[0].metric("Occupied bandwidth", f"{feats.occupied_bandwidth_hz / 1e3:,.1f} kHz")
cols[1].metric("Estimated SNR", f"{feats.snr_db_est:.1f} dB")
cols[2].metric("Duration", f"{feats.duration_s:.2f} s")

if show_reliability:
    if result.reliability:
        r = result.reliability
        st.sidebar.metric(
            f"Test accuracy at {r.snr_bucket_db:+g} dB",
            f"{r.accuracy:.0%}",
            help=f"{r.n_windows} synthetic test windows",
        )
    else:
        st.sidebar.info("Run `wtrf eval --run <id>` to record accuracy per SNR.")

out = explain_classification(result, run)
st.subheader("Explanation")
(st.warning if out.uncertain else st.success)(out.verdict)
st.markdown("\n".join(f"- {line}" for line in out.evidence))
st.caption(out.confidence_note)
if out.alternatives:
    st.markdown("**Alternatives**\n" + "\n".join(f"- {line}" for line in out.alternatives))
