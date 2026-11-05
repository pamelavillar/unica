"""Convierte los CSV originales de M5 (Kaggle) al formato HF-datasets que usa este proyecto.

Uso:
    python data/prepare_m5.py --csv_dir /ruta/m5-forecasting-accuracy --out datos/m5/M5_test
    python data/prepare_m5.py ... --num_series 1000   # subconjunto aleatorio (pruebas rápidas)

Cada serie (item x tienda) produce una entrada con:
    item_id            id de M5, p.ej. "HOBBIES_1_001_CA_1"
    start, freq        fecha del primer día con precio, "D"
    target             ventas diarias (float32)
    feat_static_cat    [item_id, dept_id, cat_id, store_id, state_id]
    feat_dynamic_real  [sell_price, snap, event_Cultural, event_National, event_Religious, event_Sporting]
                       (todas conocidas en el futuro)

Se usa sales_train_evaluation.csv (d_1..d_1941): los últimos 28 días (d_1914..d_1941) son
el periodo de evaluación oficial de M5 y el código los usa como test.
"""
import argparse
import os
import shutil
import tempfile

import datasets
import numpy as np
import pandas as pd

EVENT_TYPES = ["Cultural", "National", "Religious", "Sporting"]
STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_series", type=int, default=None, help="subconjunto aleatorio de series")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Leyendo CSV...")
    cal = pd.read_csv(os.path.join(args.csv_dir, "calendar.csv"))
    sales = pd.read_csv(os.path.join(args.csv_dir, "sales_train_evaluation.csv"))
    prices = pd.read_csv(os.path.join(args.csv_dir, "sell_prices.csv"))

    if args.num_series is not None and args.num_series < len(sales):
        sales = sales.sample(n=args.num_series, random_state=args.seed).sort_index()

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    T = len(day_cols)
    cal = cal.iloc[:T]
    dates = pd.to_datetime(cal["date"]).to_numpy()

    # Covariables de calendario (comunes a todas las series)
    events = np.zeros((len(EVENT_TYPES), T), dtype=np.float32)
    for k, et in enumerate(EVENT_TYPES):
        events[k] = ((cal["event_type_1"] == et) | (cal["event_type_2"] == et)).to_numpy(np.float32)
    snap = {s: cal[f"snap_{s}"].to_numpy(np.float32) for s in ["CA", "TX", "WI"]}

    # Matriz de precios (serie x semana) -> (serie x día)
    weeks = np.sort(cal["wm_yr_wk"].unique())
    week_idx = np.searchsorted(weeks, cal["wm_yr_wk"].to_numpy())
    prices = prices[prices["wm_yr_wk"].isin(weeks)]
    prices["key"] = prices["item_id"] + "_" + prices["store_id"]
    price_wk = prices.pivot(index="key", columns="wm_yr_wk", values="sell_price").reindex(columns=weeks)

    target_all = sales[day_cols].to_numpy(np.float32)
    keys = (sales["item_id"] + "_" + sales["store_id"]).to_numpy()
    price_wk = price_wk.reindex(keys).to_numpy(np.float32)

    def gen():
        for i, row in enumerate(sales[STATIC_COLS + ["id"]].itertuples(index=False)):
            price = price_wk[i][week_idx]  # [T]
            has_price = ~np.isnan(price)
            if not has_price.any():
                continue
            first = int(np.argmax(has_price))  # el producto aún no se vendía antes de esto
            price = pd.Series(price[first:]).ffill().bfill().to_numpy(np.float32)
            dyn = np.vstack([price[None], snap[row.state_id][None, first:], events[:, first:]])
            yield {
                "item_id": row.id.replace("_evaluation", ""),
                "start": pd.Timestamp(dates[first]),
                "freq": "D",
                "target": target_all[i, first:],
                "feat_static_cat": [getattr(row, c) for c in STATIC_COLS],
                "feat_dynamic_real": dyn.astype(np.float32),
            }

    features = datasets.Features({
        "item_id": datasets.Value("string"),
        "start": datasets.Value("timestamp[s]"),
        "freq": datasets.Value("string"),
        "target": datasets.Sequence(datasets.Value("float32")),
        "feat_static_cat": datasets.Sequence(datasets.Value("string")),
        "feat_dynamic_real": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
    })
    print(f"Generando {len(sales)} series de hasta {T} días...")
    cache_dir = tempfile.mkdtemp()  # evita dejar una copia de ~GB en ~/.cache/huggingface
    ds = datasets.Dataset.from_generator(gen, features=features, cache_dir=cache_dir)
    # map.json / dim_params.json se regeneran a partir de los datos nuevos
    for f in ["map.json", "dim_params.json"]:
        path = os.path.join(args.out, f)
        if os.path.exists(path):
            os.remove(path)
    ds.save_to_disk(args.out)
    n = len(ds)
    del ds
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Guardado en {args.out}: {n} series")


if __name__ == "__main__":
    main()
