"""
Graficos de pronostico de ChronosX y UniCA para unas pocas series.

Lee el npz que deja `run_unica.py --forecast_from ...` (historia, valor real del test y cuantiles 0.1..0.9 de cada
modelo). Por serie: un panel por modelo con la serie real (azul), el inicio del pronostico (linea punteada), la
mediana (rojo) y la banda 10%-90%. Guarda un PNG por serie.

En Colab (para que se vean en el notebook):
    %run m5cx/plot_forecasts.py --npz /content/drive/MyDrive/unica/m5_cx/forecasts/forecasts_cx_unica.npz
"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
p.add_argument("--history", type=int, default=90, help="dias de historia en el grafico")
p.add_argument("--out_dir", default=None, help="por defecto, la carpeta del npz")
p.add_argument("--layout", choices=["side", "stack"], default="side", help="side = lado a lado, stack = uno debajo del otro")
a = p.parse_args()

MODELS = ["ChronosX", "UniCA"]
TITLES = {"ChronosX": "ChronosX (IIB+OIB)", "UniCA": "UniCA"}
BLUE, RED = "#3d85c6", "#e06666"

d = np.load(a.npz)
q = list(np.round(d["q"], 2))
i10, i50, i90 = q.index(0.1), q.index(0.5), q.index(0.9)
out_dir = Path(a.out_dir or Path(a.npz).parent / "graficos")
out_dir.mkdir(parents=True, exist_ok=True)

for k, sid in enumerate(d["ids"]):
    hist = d["hist"][k][-a.history:]
    hist = hist[~np.isnan(hist)]
    y = d["y"][k]
    H = len(y)
    fstart = pd.Timestamp(str(d["fstart"][k]))
    t_hist = pd.date_range(end=fstart - pd.Timedelta(days=1), periods=len(hist), freq="D")
    t_test = pd.date_range(start=fstart, periods=H, freq="D")
    t_all = t_hist.append(t_test)
    y_all = np.concatenate([hist, y])

    present = [m for m in MODELS if m in d.files]
    if a.layout == "side":
        fig, axes = plt.subplots(1, len(present), figsize=(7 * len(present), 3.6), sharey=True)
    else:
        fig, axes = plt.subplots(len(present), 1, figsize=(10, 3.4 * len(present)), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, present):
        fq = d[m][k]
        ax.plot(t_all, y_all, color=BLUE, lw=1.6)
        ax.axvline(fstart, color="gray", ls="--", lw=1.2)
        ax.fill_between(t_test, fq[i10], fq[i90], color=RED, alpha=0.4, lw=0)
        ax.plot(t_test, fq[i50], color=RED, lw=2)
        ax.set_title(TITLES[m], fontweight="bold")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[len(present):]:
        ax.axis("off")
    fig.suptitle(str(sid), fontsize=13)
    fig.tight_layout()
    f = out_dir / f"{sid}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.show()
    print("Guardado:", f)
