"""
Compara los 4 modelos sobre el MISMO M5 (el de ChronosX) y la MISMA ventana de test, serie por serie:
    Chronos-T5 zero-shot | ChronosX | Chronos-Bolt zero-shot | UniCA

Lee solo las metricas por serie que ya guardaron los dos notebooks (no recalcula pronosticos):
    ChronosX : <cx_root>/baseline/*.npz          y <cx_root>/final/run=*/item_metrics.npz
    UniCA    : <unica_root>/baseline/*.npz       y <unica_root>/final/run=*/item_metrics.npz

Antes de comparar, VERIFICA que sean las mismas series en el mismo orden y la misma ventana: la suma de |y| en
el horizonte de cada serie ('abs') tiene que coincidir entre los dos notebooks. Si no coincide, aborta.

WQL: ChronosX guardo 19 cuantiles (0.05..0.95) y UniCA 9 (0.1..0.9). Para comparar se usa 0.1..0.9 en todos
(la columna WQL19 es la del notebook de ChronosX, solo como referencia).
MASE: estacionalidad 1 y media por serie en los dos notebooks (mismo calculo de gluonts).

Uso: python m5cx/compare_models.py --cx_root /content/drive/MyDrive/pfc1/chronosx/m5 --unica_root /content/drive/MyDrive/pfc1/unica/m5_cx
"""
import argparse
import glob
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cx_root", required=True)
p.add_argument("--unica_root", required=True)
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None, help="por defecto <unica_root>/comparacion_4_modelos.md")
a = p.parse_args()

Q9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q19 = np.arange(0.05, 1, 0.05).round(2).tolist()


def to_q9(m):
    """Deja sql con las columnas de 0.1..0.9 (de 19 o de 9 cuantiles)."""
    m = dict(m)
    if m["sql"].shape[1] == 19:
        m["sql19"] = m["sql"]
        m["sql"] = m["sql"][:, [Q19.index(q) for q in Q9]]
    assert m["sql"].shape[1] == 9, m["sql"].shape
    return m


def load_npz(path):
    return to_q9({k: v for k, v in np.load(path).items()})


def baseline(root, n=None):
    files = sorted(glob.glob(str(Path(root) / "baseline" / "*.npz")))
    ms = [(f, load_npz(f)) for f in files]
    if n is not None:
        ms = [(f, m) for f, m in ms if len(m["mase"]) == n]
    assert ms, f"no encontre un baseline en {root}/baseline" + (f" con {n} series" if n else "")
    big = max(ms, key=lambda fm: len(fm[1]["mase"]))  # el de todas las series (no un subconjunto de prueba)
    return big


def finals(root):
    out = []
    for d in sorted(Path(root, "final").glob("run=*")):
        f = d / "item_metrics.npz"
        if f.exists():
            out.append((d.name, load_npz(f)))
    assert out, f"no hay corridas finales terminadas en {root}/final"
    return out


cx_runs, ua_runs = finals(a.cx_root), finals(a.unica_root)
n = len(cx_runs[0][1]["mase"])
(f_t5, t5), (f_bolt, bolt) = baseline(a.cx_root, n), baseline(a.unica_root, n)
print("Chronos-T5 :", f_t5, "\nBolt       :", f_bolt)
print("ChronosX   :", [r for r, _ in cx_runs], "\nUniCA      :", [r for r, _ in ua_runs])

# ---------------------------------------------------------------- mismas series, misma ventana
for name, m in [("Bolt", bolt)] + [(f"UniCA {r}", m) for r, m in ua_runs] + [(f"ChronosX {r}", m) for r, m in cx_runs]:
    assert len(m["mase"]) == n, f"{name}: {len(m['mase'])} series vs {n} en Chronos-T5"
    if not np.allclose(m["abs"], t5["abs"], rtol=1e-5, atol=1e-6, equal_nan=True):
        bad = np.where(~np.isclose(m["abs"], t5["abs"], rtol=1e-5, atol=1e-6, equal_nan=True))[0]
        raise SystemExit(f"ABORTO: {name} no evalua las mismas series/ventana que Chronos-T5 "
                         f"({len(bad)} series difieren en sum|y|, p.ej. indices {bad[:5].tolist()}). "
                         f"Revisa que UniCA se haya entrenado con el M5 convertido desde m5_lean.")
print(f"\nOK: las {n} series y la ventana de test coinciden en los 4 modelos (sum|y| identico serie por serie)\n")


def avg_runs(runs):
    """Metricas por serie promediadas entre corridas (pronostico de 'semilla promedio')."""
    return {"mase": np.mean([m["mase"] for _, m in runs], axis=0),
            "sql": np.mean([m["sql"] for _, m in runs], axis=0), "abs": runs[0][1]["abs"]}


ok = np.isfinite(t5["mase"]) & np.isfinite(bolt["mase"])
for _, m in cx_runs + ua_runs:
    ok &= np.isfinite(m["mase"])


def mase(m):
    return float(np.mean(m["mase"][ok]))


def wql(m, key="sql"):
    return float(np.mean(np.nansum(m[key], axis=0) / np.nansum(m["abs"])))


def boot(base, new):
    """IC95 del cambio relativo % (new vs base): remuestreo emparejado de series."""
    rng = np.random.default_rng(a.seed)
    bm, nm = base["mase"][ok], new["mase"][ok]
    bs, ns, ab = np.nan_to_num(base["sql"][ok]), np.nan_to_num(new["sql"][ok]), np.nan_to_num(base["abs"][ok])
    k = len(bm)
    rm, rw = [], []
    for s in range(0, a.n_boot, 50):
        c = rng.multinomial(k, np.full(k, 1 / k), size=min(50, a.n_boot - s)).astype(float)
        mb, mn = c @ bm, c @ nm
        rm.append(100 * (mn - mb) / mb)
        den = c @ ab
        wb = np.mean((c @ bs) / den[:, None], axis=1)
        wn = np.mean((c @ ns) / den[:, None], axis=1)
        rw.append(100 * (wn - wb) / wb)
    rm, rw = np.concatenate(rm), np.concatenate(rw)
    pt_m = 100 * (nm.mean() - bm.mean()) / bm.mean()
    pt_w = 100 * (wql(new) - wql(base)) / wql(base)
    return pt_m, np.percentile(rm, [2.5, 97.5]), pt_w, np.percentile(rw, [2.5, 97.5]), float(np.mean(nm < bm))


def sd(x):
    return f"{np.std(x, ddof=1):.4f}" if len(x) > 1 else "n/a"


cx_avg, ua_avg = avg_runs(cx_runs), avg_runs(ua_runs)
L = [f"# Comparacion de 4 modelos en el mismo M5 ({n} series, ultimos 28 dias)\n",
     f"Series usadas para MASE (finito en todos): {int(ok.sum())} de {n}. WQL en cuantiles 0.1..0.9 para todos.\n",
     "| modelo | corridas | MASE (media) | +- entre corridas | MASE (mediana) | WQL 0.1..0.9 | +- entre corridas | WQL19 (ref. ChronosX) |",
     "|---|---|---|---|---|---|---|---|"]
rows = [("Chronos-T5 zero-shot", [t5]), ("ChronosX", [m for _, m in cx_runs]),
        ("Chronos-Bolt zero-shot", [bolt]), ("UniCA", [m for _, m in ua_runs])]
for name, ms in rows:
    mm, ww = [mase(m) for m in ms], [wql(m) for m in ms]
    med = np.median(np.mean([m["mase"][ok] for m in ms], axis=0))
    w19 = ([wql(m, "sql19") for m in ms if "sql19" in m] or [np.nan])
    L.append(f"| {name} | {len(ms)} | {np.mean(mm):.4f} | {sd(mm) if len(ms) > 1 else '-'} | {med:.4f} | "
             f"{np.mean(ww):.4f} | {sd(ww) if len(ms) > 1 else '-'} | "
             f"{np.nanmean(w19):.4f} |".replace("| nan |", "| - |"))

L += ["", "## Diferencias emparejadas (cambio relativo %, IC95 por bootstrap de series; negativo = mejor)",
      "Los modelos entrenados usan sus metricas por serie PROMEDIADAS entre corridas.\n",
      "| comparacion | cambio MASE | IC95 MASE | cambio WQL | IC95 WQL | % series con menor MASE |",
      "|---|---|---|---|---|---|"]
pairs = [("UniCA vs ChronosX", cx_avg, ua_avg),
         ("Chronos-Bolt vs Chronos-T5 (zero-shot)", t5, bolt),
         ("ChronosX vs Chronos-T5", t5, cx_avg),
         ("UniCA vs Chronos-Bolt", bolt, ua_avg),
         ("UniCA vs Chronos-T5", t5, ua_avg)]
for name, b, nw in pairs:
    pm, cim, pw, ciw, win = boot(b, nw)
    L.append(f"| {name} | {pm:+.2f}% | [{cim[0]:+.2f}, {cim[1]:+.2f}]% | {pw:+.2f}% | "
             f"[{ciw[0]:+.2f}, {ciw[1]:+.2f}]% | {100 * win:.1f}% |")
L += ["", "Si el IC95 incluye 0, la diferencia no se distingue del ruido de muestreo de series. El IC no incluye "
      "la variabilidad entre semillas (mira la columna +- entre corridas). 'UniCA vs ChronosX' compara los dos "
      "adaptadores completos (cada uno sobre su modelo base); la fila 'Chronos-Bolt vs Chronos-T5' muestra cuanto "
      "de esa diferencia ya viene del modelo base."]
txt = "\n".join(L)
out = Path(a.out or Path(a.unica_root) / "comparacion_4_modelos.md")
out.write_text(txt)
print(txt)
print("\nGuardado en", out)
