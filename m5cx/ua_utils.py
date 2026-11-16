"""
Utilidades para reproducir UniCA en M5 con poca RAM y de forma reanudable (Colab).

  * get_repo_args      : construye el Namespace de args EXACTAMENTE con el parser del repo (utils.py),
                         para que el estimador reciba los mismos valores por defecto que main.py.
  * enable_light_cache : la cache de MixedRandomDataset (data/loader.py) guarda TODAS las series ya
                         transformadas y se duplica en cada worker -> agota los 12 GB de Colab con
                         las ~30k series de M5. Aqui se reemplaza por una cache LRU acotada.
  * metricas por serie + bootstrap emparejado (para saber si una diferencia es ruido).
  * checkpoint_problem / latest_complete_checkpoint : reanudar Lightning desde un .ckpt completo.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # los del repo (main.py y la perdida)


# ------------------------------------------------------------------ args del repo
def get_repo_args(cli: list):
    """Parsea `cli` con el parser de UniCA (utils.get_args_and_settings) sin tocar sys.argv real."""
    import torch

    import utils as unica_utils

    old = sys.argv
    sys.argv = ["main.py"] + [str(c) for c in cli]
    try:
        args, datasets = unica_utils.get_args_and_settings()
    finally:
        sys.argv = old
    torch.set_num_threads(max(1, torch.get_num_threads()))  # el repo fuerza 16 hilos; Colab tiene 2 CPU
    return args, datasets


# ------------------------------------------------------------------ memoria
class _LRU:
    """Sustituto de la lista `cache` de MixedRandomDataset: devuelve None si la serie no esta."""

    def __init__(self, n_items: int, max_items: int):
        self.n, self.max, self.d = n_items, max_items, OrderedDict()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        v = self.d.get(i)
        if v is not None:
            self.d.move_to_end(i)
        return v

    def __setitem__(self, i, v):
        self.d[i] = v
        self.d.move_to_end(i)
        while len(self.d) > self.max:
            self.d.popitem(last=False)


def enable_light_cache(max_items: int = 2000):
    """Acota la cache de series transformadas (por worker). No cambia que muestras se generan."""
    from data import loader

    cls = loader.MixedRandomDataset
    if getattr(cls, "_ua_light_cache", False):
        return
    original_init = cls.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.cache = _LRU(len(self.dataset), max_items)

    cls.__init__ = init
    cls._ua_light_cache = True


# ------------------------------------------------------------------ reanudacion
def checkpoint_problem(path):
    """None si el .ckpt de Lightning se puede cargar; si no, el motivo (Drive pudo dejarlo a medias)."""
    import torch

    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return "no existe o esta vacio"
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # zip truncado
        return f"ilegible ({type(e).__name__})"
    for k in ("state_dict", "optimizer_states", "loops", "callbacks"):
        if k not in ck:
            return f"le falta '{k}'"
    return None


def latest_complete_checkpoint(ckpt_dir):
    """(ruta o None, [(ruta, motivo)] descartados).

    Prueba last*.ckpt (del mas nuevo al mas viejo) y, si ninguno sirve, best.ckpt: tambien es un estado de
    entrenamiento completo (de una epoca anterior), asi que solo se pierden las epocas posteriores.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None, []
    by_time = (lambda ps: sorted(ps, key=lambda p: p.stat().st_mtime, reverse=True))
    cands = by_time(d.glob("last*.ckpt")) + by_time(d.glob("best*.ckpt"))
    skipped = []
    for c in cands:
        why = checkpoint_problem(c)
        if why is None:
            return c, skipped
        skipped.append((c, why))
    return None, skipped


# ------------------------------------------------------------------ metricas
def item_metrics(forecasts, test_data, quantiles=QUANTILES, batch_size=500, seasonality=1) -> dict:
    """Metricas POR SERIE con gluonts (axis=1 = se agrega solo en el tiempo).

    mase[n]   : MASE de la mediana
    ae[n]     : suma en el tiempo de |y - mediana|      (MAE agregado = sum ae / (n*H))
    se[n]     : suma en el tiempo de (y - media)^2      (MSE agregado = sum se / (n*H))
    ape[n]    : suma en el tiempo de |y - mediana|/|y| sobre los puntos con y != 0
    apen[n]   : cuantos puntos con y != 0 (MAPE agregado del repo = sum ape / sum apen)
    sql[n, Q] : suma en el tiempo de la quantile loss, por cuantil
    abs[n]    : suma de |y| en el horizonte
    El WQL (= 'mean_weighted_sum_quantile_loss', la CRPS del repo) es mean_q( sum_i sql[i,q] / sum_i abs[i] ).
    """
    from gluonts.ev.metrics import MASE, MSE, SumAbsoluteError, SumQuantileLoss, sum_absolute_label
    from gluonts.model.evaluation import evaluate_forecasts_raw

    metrics = ([MASE(), SumAbsoluteError(), MSE(forecast_type="mean")]
               + [SumQuantileLoss(q=q) for q in quantiles] + [sum_absolute_label])
    raw = evaluate_forecasts_raw(
        forecasts, test_data=test_data, metrics=metrics, axis=1, batch_size=batch_size,
        mask_invalid_label=True, allow_nan_forecast=False, seasonality=seasonality,
    )

    def arr(key):
        return np.ma.filled(np.ma.asarray(raw[key], dtype=float), np.nan).reshape(-1)

    H = len(next(iter(test_data.label))["target"])
    ape, apen = [], []
    for f, lab in zip(forecasts, test_data.label):  # MAPE como gluonts con axis=None: se ignoran y=0 y NaN
        y = np.asarray(lab["target"], dtype=float)
        ok = np.isfinite(y) & (y != 0)
        ape.append(float(np.sum(np.abs(y[ok] - f.quantile(0.5)[ok]) / np.abs(y[ok]))))
        apen.append(int(ok.sum()))
    return {
        "mase": arr("MASE[0.5]"),
        "ae": arr("sum_absolute_error[0.5]"),
        "se": arr("MSE[mean]") * H,
        "ape": np.array(ape),
        "apen": np.array(apen),
        "sql": np.stack([arr(f"sum_quantile_loss[{q}]") for q in quantiles], axis=1),
        "abs": arr("sum_absolute_label"),
        "H": np.array(H),
    }


def aggregate(m: dict, naive: dict = None) -> dict:
    H = int(m["H"])
    ok = np.isfinite(m["mase"])
    n = len(m["mase"])
    out = {
        "MASE[0.5]": float(np.mean(m["mase"][ok])),
        "mean_weighted_sum_quantile_loss": float(np.mean(np.nansum(m["sql"], axis=0) / np.nansum(m["abs"]))),
        "MAE[0.5]": float(np.nansum(m["ae"]) / (n * H)),
        "MSE[mean]": float(np.nansum(m["se"]) / (n * H)),
        "MAPE[0.5]": float(np.sum(m["ape"]) / np.sum(m["apen"])),
        "n_items": int(n),
        "n_items_mase_invalid": int((~ok).sum()),
    }
    if naive:  # normalizacion del repo (main.py): metrica / metrica del Naive en el mismo split
        out["normalized"] = {
            "MAE_norm": out["MAE[0.5]"] / naive["MAE"],
            "MSE_norm": out["MSE[mean]"] / naive["MSE"],
            "MAPE_norm": out["MAPE[0.5]"] / naive["MAPE"],
            "CRPS_norm": out["mean_weighted_sum_quantile_loss"] / naive["mean_CRPS"],
        }
    return out


def paired_bootstrap(base: dict, new: dict, n_boot: int = 1000, seed: int = 0, min_items: int = 30):
    """Bootstrap emparejado sobre series: IC95% de la diferencia (new - base) en MASE, WQL y MAE."""
    n = len(base["mase"])
    if n < min_items:
        return None
    rng = np.random.default_rng(seed)
    ok = np.isfinite(base["mase"]) & np.isfinite(new["mase"])
    d_mase = (new["mase"] - base["mase"])[ok]
    b_mase, n_mase = base["mase"][ok], new["mase"][ok]
    b_ae, n_ae = np.nan_to_num(base["ae"]), np.nan_to_num(new["ae"])

    out = {k: [] for k in ("mase_diff", "mase_rel", "wql_diff", "wql_rel", "mae_rel")}
    chunk = 50
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float64)  # (size, n)
        c_ok = counts[:, ok]
        w = c_ok.sum(1)
        mb, mn = (c_ok @ b_mase) / w, (c_ok @ n_mase) / w
        out["mase_diff"].append(mn - mb)
        out["mase_rel"].append((mn - mb) / mb)
        den_b, den_n = counts @ base["abs"], counts @ new["abs"]
        wb = np.mean((counts @ np.nan_to_num(base["sql"])) / den_b[:, None], axis=1)
        wn = np.mean((counts @ np.nan_to_num(new["sql"])) / den_n[:, None], axis=1)
        out["wql_diff"].append(wn - wb)
        out["wql_rel"].append((wn - wb) / wb)
        ab, an = counts @ b_ae, counts @ n_ae
        out["mae_rel"].append((an - ab) / ab)
    out = {k: np.concatenate(v) for k, v in out.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "n_boot": n_boot,
        "mase_diff_ci95": ci(out["mase_diff"]),
        "mase_rel_pct_ci95": [100 * v for v in ci(out["mase_rel"])],
        "wql_diff_ci95": ci(out["wql_diff"]),
        "wql_rel_pct_ci95": [100 * v for v in ci(out["wql_rel"])],
        "mae_rel_pct_ci95": [100 * v for v in ci(out["mae_rel"])],
        "prob_unica_better_mase": float(np.mean(out["mase_diff"] < 0)),
        "prob_unica_better_wql": float(np.mean(out["wql_diff"] < 0)),
        "win_rate_mase": float(np.mean(d_mase < 0)),  # % de series donde UniCA tiene menor MASE
    }


def save_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    tmp.replace(path)
