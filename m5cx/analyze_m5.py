"""
Analisis de los resultados ya guardados (sin reentrenar ni volver a pronosticar) de los 4 modelos:
    Chronos-T5 zero-shot | ChronosX | Chronos-Bolt zero-shot | UniCA

Lee:
    ChronosX : <cx_root>/baseline/*.npz, <cx_root>/final/run=*/item_metrics.npz y results.json
    UniCA    : <unica_root>/baseline/*.npz, <unica_root>/final/run=*/item_metrics.npz y results.json
    Datos    : el M5 convertido para UniCA (--data), para saber categoria, estado, historia y covariables de cada serie.

Saca:
    1. Como termino cada entrenamiento (lr, epocas, early stopping, perdida de validacion)
    2. Resultados por categoria, departamento, estado, nivel de ventas e intermitencia
    3. De que grupo de series viene la mejora de cada adaptador
    4. Empates, victorias y derrotas serie por serie
    5. MASE: media vs mediana y efecto de la escala del pronostico ingenuo
    6. Cambio por cuantil (para ver si los adaptadores mueven la mediana o tambien la incertidumbre)
    7. Que covariables cambian dentro de la ventana de test

Uso:
    python m5cx/analyze_m5.py --cx_root /content/drive/MyDrive/chronosx/m5 \
        --unica_root /content/drive/MyDrive/unica/m5_cx --data /content/drive/MyDrive/unica/data_cx/m5/M5_test
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cx_root", required=True)
p.add_argument("--unica_root", required=True)
p.add_argument("--data", default="/content/data_cx/m5/M5_test")
p.add_argument("--out", default=None, help="por defecto <unica_root>/analisis_m5.md")
a = p.parse_args()

H = 28
Q9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q19 = np.arange(0.05, 1, 0.05).round(2).tolist()
COVARIATES = [  # mismo orden que m5_to_unica.COVARIATES
    "snap_CA", "snap_TX", "snap_WI", "sell_price",
    "event_type_1_Cultural", "event_type_1_National", "event_type_1_Religious",
    "event_type_1_Sporting", "event_type_1_nan",
    "event_type_2_Cultural", "event_type_2_Religious", "event_type_2_nan",
]
EVENTS = [c for c in COVARIATES if c.startswith("event") and not c.endswith("_nan")]
MODELS = ["Chronos-T5", "ChronosX", "Chronos-Bolt", "UniCA"]
PAIRS = [("ChronosX vs T5", "Chronos-T5", "ChronosX"),
         ("UniCA vs Bolt", "Chronos-Bolt", "UniCA"),
         ("UniCA vs ChronosX", "ChronosX", "UniCA"),
         ("Bolt vs T5", "Chronos-T5", "Chronos-Bolt")]

L = []


def out(s=""):
    L.append(s)


def table(header, rows):
    out("| " + " | ".join(header) + " |")
    out("|" + "---|" * len(header))
    for r in rows:
        out("| " + " | ".join(str(x) for x in r) + " |")
    out()


def f4(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.4f}"


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:+.1f}%"


# ================================================================== metricas guardadas
def load_npz(path):
    m = {k: v for k, v in np.load(path).items()}
    if m["sql"].shape[1] == 19:
        m["sql19"] = m["sql"]
        m["sql"] = m["sql"][:, [Q19.index(q) for q in Q9]]
    return m


def baseline(root, n=None):
    ms = [(f, load_npz(f)) for f in sorted(glob.glob(str(Path(root) / "baseline" / "*.npz")))]
    if n is not None:
        ms = [(f, m) for f, m in ms if len(m["mase"]) == n]
    assert ms, f"no encontre un baseline en {root}/baseline"
    return max(ms, key=lambda fm: len(fm[1]["mase"]))[1]


def finals(root):
    runs = []
    for d in sorted(Path(root, "final").glob("run=*")):
        if (d / "item_metrics.npz").exists():
            res = json.load(open(d / "results.json")) if (d / "results.json").exists() else {}
            runs.append((d.name, load_npz(d / "item_metrics.npz"), res))
    assert runs, f"no hay corridas terminadas en {root}/final"
    return runs


def avg(runs):
    m = {"mase": np.mean([r[1]["mase"] for r in runs], axis=0),
         "sql": np.mean([r[1]["sql"] for r in runs], axis=0), "abs": runs[0][1]["abs"]}
    if all("sql19" in r[1] for r in runs):
        m["sql19"] = np.mean([r[1]["sql19"] for r in runs], axis=0)
    return m


cx_runs, ua_runs = finals(a.cx_root), finals(a.unica_root)
n = len(cx_runs[0][1]["mase"])
M = {"Chronos-T5": baseline(a.cx_root, n), "ChronosX": avg(cx_runs),
     "Chronos-Bolt": baseline(a.unica_root, n), "UniCA": avg(ua_runs)}
for k, m in M.items():
    assert len(m["mase"]) == n, f"{k}: {len(m['mase'])} series vs {n}"

# ================================================================== datos de cada serie
import datasets  # noqa: E402

print("Leyendo", a.data, "...", flush=True)
ds = datasets.load_from_disk(a.data)
assert len(ds) == n, f"el dataset tiene {len(ds)} series y las metricas {n}"


def as2d(x):
    x = np.asarray(x)
    if x.dtype == object:
        x = np.stack([np.asarray(r, np.float32) for r in x])
    return x.astype(np.float32)


ids, abs_data, mean365, zeros365, scale, test_zero = [], [], [], [], [], []
varies = np.zeros((n, len(COVARIATES)), bool)
price_chg_test, price_chg_vs_last, price_rel, price_chg_ctx = [], [], [], []
cal_days = {}  # estado -> (fechas de test, matriz 12 x 28)
i = 0
for b in ds.with_format("numpy").iter(batch_size=500):
    for j in range(len(b["item_id"])):
        sid = str(b["item_id"][j])
        y = np.asarray(b["target"][j], np.float32)
        X = as2d(b["feat_dynamic_real"][j])
        ctx, test = y[:-H], y[-H:]
        last = ctx[-365:]
        ids.append(sid)
        abs_data.append(float(np.abs(test).sum()))
        mean365.append(float(last.mean()) if len(last) else np.nan)
        zeros365.append(float((last == 0).mean()) if len(last) else np.nan)
        scale.append(float(np.mean(np.abs(np.diff(ctx)))) if len(ctx) > 1 else np.nan)
        test_zero.append(bool((test == 0).all()))
        Xt = X[:, -H:]
        varies[i] = Xt.max(1) != Xt.min(1)
        pr, pr_last = Xt[3], X[3, -H - 1] if X.shape[1] > H else np.nan
        price_chg_test.append(int((np.diff(pr) != 0).sum()))
        price_chg_vs_last.append(bool(np.any(pr != pr_last)))
        price_rel.append(float(pr.mean() / pr_last - 1) if np.isfinite(pr_last) and pr_last > 0 else np.nan)
        pc = X[3, :-H][-365:]
        price_chg_ctx.append(float((np.diff(pc) != 0).sum() * H / max(len(pc) - 1, 1)) if len(pc) > 1 else np.nan)
        state = sid.split("_")[-2]
        if state not in cal_days:
            start = np.datetime64(b["start"][j], "D")
            cal_days[state] = (start + np.arange(len(y) - H, len(y)), Xt)
        i += 1
    print(f"  {i}/{n} series", flush=True)

ids = np.array(ids)
abs_data = np.array(abs_data)
mean365, zeros365, scale = map(np.array, (mean365, zeros365, scale))
test_zero = np.array(test_zero)
price_chg_test, price_chg_vs_last = np.array(price_chg_test), np.array(price_chg_vs_last)
price_rel, price_chg_ctx = np.array(price_rel), np.array(price_chg_ctx)

if not np.allclose(abs_data, M["Chronos-T5"]["abs"], rtol=1e-4, atol=1e-3, equal_nan=True):
    bad = np.where(~np.isclose(abs_data, M["Chronos-T5"]["abs"], rtol=1e-4, atol=1e-3))[0]
    raise SystemExit(f"ABORTO: el dataset no tiene las mismas series/orden/ventana que las metricas "
                     f"({len(bad)} series difieren en sum|y| del test, p.ej. {ids[bad[:5]].tolist()})")
print("OK: el dataset coincide serie por serie con las metricas guardadas\n", flush=True)

cat = np.array([s.split("_")[0] for s in ids])
dept = np.array(["_".join(s.split("_")[:2]) for s in ids])
state = np.array([s.split("_")[-2] for s in ids])
store = np.array(["_".join(s.split("_")[-2:]) for s in ids])


def binned(x, edges, names):
    lab = np.full(len(x), "sin historia", dtype=object)
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        lab[(x >= lo) & (x < hi)] = nm
    return lab


VOL_NAMES = ["< 0.25", "0.25 - 1", "1 - 3", "3 - 10", ">= 10"]
volume = binned(mean365, [0, 0.25, 1, 3, 10, np.inf], VOL_NAMES)
INT_NAMES = ["< 25% ceros", "25 - 50%", "50 - 75%", "75 - 90%", ">= 90% ceros"]
intermit = binned(zeros365, [0, 0.25, 0.5, 0.75, 0.9, 1.01], INT_NAMES)

ok = np.ones(n, bool)
for m in M.values():
    ok &= np.isfinite(m["mase"])


# ================================================================== helpers de metricas
def mase(m, sel):
    s = sel & ok
    return float(np.mean(m["mase"][s])) if s.any() else np.nan


def wql(m, sel, key="sql"):
    den = np.nansum(m["abs"][sel])
    return float(np.mean(np.nansum(m[key][sel], axis=0) / den)) if den > 0 else np.nan


def rel(b, nw):
    return 100 * (nw - b) / b if np.isfinite(b) and b != 0 else np.nan


def series_loss(m):
    return np.nanmean(m["sql"], axis=1)  # aporte de cada serie al WQL (antes de dividir por sum|y|)


# ================================================================== 1. entrenamiento
out("# Analisis de resultados en M5")
out()
out(f"{n} series, ultimos {H} dias. Modelos entrenados: promedio por serie de sus corridas "
    f"(ChronosX: {len(cx_runs)}, UniCA: {len(ua_runs)}). WQL en cuantiles 0.1 - 0.9 para todos.")
out()
out("## 1. Como termino cada entrenamiento")
out()
for name, root, runs in [("ChronosX", a.cx_root, cx_runs), ("UniCA", a.unica_root, ua_runs)]:
    st = Path(root, "state.json")
    sweep = json.load(open(st)).get("sweep") if st.exists() else None
    out(f"{name}:")
    if sweep:
        out("- barrido de lr (perdida de validacion): " + ", ".join(f"{k}: {v:.4f}" for k, v in sweep.items()))
    for rid, _, r in runs:
        if name == "ChronosX":
            out(f"- {rid}: lr {r.get('lr')}, {r.get('max_steps')} pasos, val_loss zero-shot "
                f"{f4(r.get('val_loss_zero_shot'))} -> ChronosX {f4(r.get('val_loss_chronosx'))}")
        else:
            out(f"- {rid}: lr {r.get('lr')}, {r.get('epochs_run')} de {r.get('max_epochs')} epocas, "
                f"early stopping: {'si' if r.get('early_stopped') else 'no'}, val_loss {f4(r.get('val_loss_unica'))}")
    out()

# ================================================================== 2. por grupos
out("## 2. Resultados por grupo")
out()
out("Columnas de cambio: negativo = el segundo modelo es mejor.")
out()
GROUPS = [("Categoria", cat, None), ("Departamento", dept, None), ("Estado", state, None),
          ("Tienda", store, None),
          ("Nivel de ventas (venta media diaria, ultimo anio de historia)", volume, VOL_NAMES + ["sin historia"]),
          ("Intermitencia (% de dias con venta 0, ultimo anio de historia)", intermit, INT_NAMES + ["sin historia"]),
          ("Test", np.where(test_zero, "test todo en cero", "test con ventas"), None)]
for title, lab, order in GROUPS:
    keys = [g for g in (order or sorted(set(lab))) if (lab == g).any()]
    for metric, fn in [("MASE", mase), ("WQL", wql)]:
        out(f"### {title}: {metric}")
        out()
        rows = []
        for g in keys:
            sel = lab == g
            v = {k: fn(M[k], sel) for k in MODELS}
            rows.append([g, int(sel.sum())] + [f4(v[k]) for k in MODELS]
                        + [pct(rel(v[b], v[nw])) for _, b, nw in PAIRS[:3]])
        sel = np.ones(n, bool)
        v = {k: fn(M[k], sel) for k in MODELS}
        rows.append(["**todas**", n] + [f4(v[k]) for k in MODELS] + [pct(rel(v[b], v[nw])) for _, b, nw in PAIRS[:3]])
        table(["grupo", "series"] + MODELS + [p_[0] for p_ in PAIRS[:3]], rows)

# ================================================================== 3. de donde viene la mejora
out("## 3. De donde viene la mejora de cada adaptador")
out()
out("Que parte de la mejora total viene de cada grupo (suma 100%; un valor negativo = ese grupo empeora). "
    "Sirve para ver si la ganancia se concentra en pocas series.")
out()
for title, lab, order in GROUPS[:1] + GROUPS[4:6]:
    keys = [g for g in (order or sorted(set(lab))) if (lab == g).any()]
    rows = []
    for g in keys:
        sel = lab == g
        r = [g, f"{100 * sel.mean():.1f}%"]
        for _, b, nw in PAIRS[:2]:
            dm = np.where(ok, M[nw]["mase"] - M[b]["mase"], 0.0)
            dw = np.nan_to_num(series_loss(M[nw]) - series_loss(M[b]))
            r += [f"{100 * np.nansum(dm[sel]) / np.nansum(dm):.1f}%", f"{100 * dw[sel].sum() / dw.sum():.1f}%"]
        rows.append(r)
    out(f"### {title}")
    out()
    table(["grupo", "% de series", "ChronosX: aporte MASE", "ChronosX: aporte WQL",
           "UniCA: aporte MASE", "UniCA: aporte WQL"], rows)

# ================================================================== 4. empates
out("## 4. Gana, empata o pierde (serie por serie)")
out()
out("Empate = mismo MASE (o misma perdida de cuantiles) hasta 6 decimales.")
out()
rows = []
for name, b, nw in PAIRS:
    for metric, xb, xn, valid in [("MASE", M[b]["mase"], M[nw]["mase"], ok),
                                  ("WQL", series_loss(M[b]), series_loss(M[nw]), np.ones(n, bool))]:
        xb, xn = xb[valid], xn[valid]
        tie = np.isclose(xb, xn, rtol=1e-6, atol=1e-6)
        win, lose = (xn < xb) & ~tie, (xn > xb) & ~tie
        tz = test_zero[valid]
        rows.append([name, metric, int(valid.sum()), f"{100 * win.mean():.1f}%", f"{100 * tie.mean():.1f}%",
                     f"{100 * lose.mean():.1f}%",
                     f"{100 * win.sum() / max(win.sum() + lose.sum(), 1):.1f}%",
                     f"{100 * (tie & tz).sum() / max(tie.sum(), 1):.1f}%"])
table(["comparacion", "metrica", "series", "gana", "empata", "pierde", "gana sin contar empates",
       "empates con test todo en cero"], rows)
out(f"Series con el test (28 dias) todo en cero: {int(test_zero.sum())} ({100 * test_zero.mean():.1f}%). "
    f"Series sin MASE valido (pronostico ingenuo con error 0 en la historia): {int((~ok).sum())}.")
out()

# ================================================================== 5. MASE
out("## 5. MASE: media vs mediana y escala")
out()
rows = []
for k in MODELS:
    x = M[k]["mase"][ok]
    top = np.sort(x)[::-1]
    rows.append([k, f4(x.mean()), f4(np.median(x)), f4(np.percentile(x, 90)), f4(x.max()),
                 f"{100 * top[: len(x) // 10].sum() / x.sum():.1f}%"])
table(["modelo", "media", "mediana", "percentil 90", "maximo", "% del total que aportan el 10% peor"], rows)
out("Escala del MASE = error medio del pronostico ingenuo (y[t] - y[t-1]) en la historia. Cuanto mas chica, "
    "mas se amplifica el error. Grupos por quintiles de esa escala:")
out()
qs = np.nanpercentile(scale[ok], [20, 40, 60, 80])
edges = [-np.inf, *qs, np.inf]
rows = []
for lo, hi, nm in zip(edges[:-1], edges[1:], ["Q1 (escala mas chica)", "Q2", "Q3", "Q4", "Q5 (mas grande)"]):
    sel = (scale >= lo) & (scale < hi)
    v = {k: mase(M[k], sel) for k in MODELS}
    rows.append([nm, f"{np.nanmin(scale[sel & ok]):.3f} - {np.nanmax(scale[sel & ok]):.3f}", int((sel & ok).sum())]
                + [f4(v[k]) for k in MODELS] + [pct(rel(v[b], v[nw])) for _, b, nw in PAIRS[:2]])
table(["grupo", "escala", "series"] + MODELS + [p_[0] for p_ in PAIRS[:2]], rows)

# ================================================================== 6. por cuantil
out("## 6. Cambio por cuantil")
out()
out("Perdida de cada cuantil (suma sobre series / suma de |y|) y cambio entre modelos. Si un adaptador mejora "
    "la mediana (0.5) pero no los extremos, mueve el pronostico puntual sin mejorar la incertidumbre.")
out()
allsel = np.ones(n, bool)
den = np.nansum(M["Chronos-T5"]["abs"])
qloss = {k: np.nansum(M[k]["sql"], axis=0) / den for k in MODELS}
rows = []
for qi, q in enumerate(Q9):
    rows.append([q] + [f4(qloss[k][qi]) for k in MODELS]
                + [pct(rel(qloss[b][qi], qloss[nw][qi])) for _, b, nw in PAIRS])
table(["cuantil"] + MODELS + [p_[0] for p_ in PAIRS], rows)
for k in ("Chronos-T5", "ChronosX"):
    if "sql19" in M[k]:
        out(f"- {k}: WQL 0.1 - 0.9 = {f4(wql(M[k], allsel))} | WQL 19 cuantiles (0.05 - 0.95) = "
            f"{f4(wql(M[k], allsel, 'sql19'))}")
out()

# ================================================================== 7. covariables
out("## 7. Covariables en la ventana de test")
out()
dates = next(iter(cal_days.values()))[0]
out(f"Ventana de test: {dates[0]} a {dates[-1]}.")
out()
out("% de series en las que la covariable cambia de valor dentro de los 28 dias de test:")
out()
rows = []
for ci, c in enumerate(COVARIATES):
    r = [c, f"{100 * varies[:, ci].mean():.1f}%"]
    r += [f"{100 * varies[cat == g, ci].mean():.1f}%" for g in sorted(set(cat))]
    rows.append(r)
table(["covariable", "todas"] + sorted(set(cat)), rows)

out("Dias de test con SNAP o con evento:")
out()
rows = []
st_keys = sorted(cal_days)
for d in range(H):
    ev = [c.replace("event_type_", "tipo") for c in EVENTS
          if cal_days[st_keys[0]][1][COVARIATES.index(c), d] > 0]
    snaps = [s for s in st_keys if cal_days[s][1][COVARIATES.index(f"snap_{s}"), d] > 0]
    if ev or snaps:
        rows.append([str(dates[d]), ", ".join(ev) or "-", ", ".join(snaps) or "-"])
table(["fecha", "evento", "SNAP en"], rows)

out("Precio:")
out()
rows = []
for g in sorted(set(cat)) + ["todas"]:
    sel = cat == g if g != "todas" else np.ones(n, bool)
    pr = price_rel[sel]
    rows.append([g, int(sel.sum()), f"{100 * (price_chg_test[sel] > 0).mean():.1f}%",
                 f"{100 * price_chg_vs_last[sel].mean():.1f}%",
                 f"{np.nanmean(price_chg_ctx[sel]):.2f}",
                 f"{100 * np.nanmedian(np.abs(pr[pr != 0])):.1f}%" if np.any(pr[np.isfinite(pr)] != 0) else "-"])
table(["categoria", "series", "precio cambia dentro del test", "precio del test distinto al ultimo dia de historia",
       "cambios de precio por cada 28 dias en la historia (ultimo anio)", "tamanio mediano del cambio (si cambia)"],
      rows)

out("Resultados segun si el precio cambia en el test:")
out()
rows = []
for nm, sel in [("precio cambia", price_chg_vs_last), ("precio fijo", ~price_chg_vs_last)]:
    v = {k: mase(M[k], sel) for k in MODELS}
    w = {k: wql(M[k], sel) for k in MODELS}
    rows.append([nm, int(sel.sum())] + [pct(rel(v[b], v[nw])) for _, b, nw in PAIRS[:2]]
                + [pct(rel(w[b], w[nw])) for _, b, nw in PAIRS[:2]])
table(["grupo", "series", "ChronosX vs T5 (MASE)", "UniCA vs Bolt (MASE)", "ChronosX vs T5 (WQL)",
       "UniCA vs Bolt (WQL)"], rows)

txt = "\n".join(L)
dst = Path(a.out or Path(a.unica_root) / "analisis_m5.md")
dst.write_text(txt)
print(txt)
print("\nGuardado en", dst)
