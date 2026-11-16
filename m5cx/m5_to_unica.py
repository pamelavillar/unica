"""
Convierte el M5 de ChronosX (salida de m5_lean.build_m5) al formato que lee UniCA (data/unimodal/data.py).

Mismo M5 que en ChronosX, serie por serie y en el mismo orden (ordenado por id):
  * target  = ventas desde la primera venta (sin ceros iniciales), igual que ChronosX
  * feat_dynamic_real = las MISMAS 12 covariables de ChronosX (snap_CA/TX/WI, sell_price, one-hot de eventos),
                        conocidas tambien en el horizonte, como en ChronosX
  * item_id = id de ChronosX (item_store), start = primer timestamp, freq = "D"
  * (opcional, --static_cats) feat_static_cat = dept_id, cat_id, store_id, state_id. ChronosX NO las usa:
    dejalo apagado para comparar con la misma informacion.
La ventana de test resulta la misma: UniCA usa los ultimos 28 dias de cada serie (split_point = 28 para m5),
igual que el offset -28 de ChronosX. compare_models.py lo verifica serie por serie.

Uso:  python m5cx/m5_to_unica.py --src /content/m5cx/m5_lean_full --dst /content/data_cx/m5/M5_test
"""
import argparse
import shutil
from pathlib import Path

import datasets
import numpy as np

COVARIATES = [  # mismo orden que m5_lean.COVARIATES
    "snap_CA", "snap_TX", "snap_WI", "sell_price",
    "event_type_1_Cultural", "event_type_1_National", "event_type_1_Religious",
    "event_type_1_Sporting", "event_type_1_nan",
    "event_type_2_Cultural", "event_type_2_Religious", "event_type_2_nan",
]
STATIC = ["dept_id", "cat_id", "store_id", "state_id"]

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--static_cats", action="store_true")
a = p.parse_args()

src = datasets.load_from_disk(a.src)
print(src)
static = a.static_cats


def convert(batch):
    out = {"item_id": batch["id"], "start": [ts[0] for ts in batch["timestamp"]],
           "freq": ["D"] * len(batch["id"]), "target": batch["target"],
           "feat_dynamic_real": [np.stack([np.asarray(batch[c][i], np.float32) for c in COVARIATES])
                                 for i in range(len(batch["id"]))]}
    if static:
        out["feat_static_cat"] = [[batch[c][i] for c in STATIC] for i in range(len(batch["id"]))]
    return out


feats = {
    "item_id": datasets.Value("string"), "start": datasets.Value("timestamp[ms]"), "freq": datasets.Value("string"),
    "target": datasets.Sequence(datasets.Value("float32")),
    "feat_dynamic_real": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
}
if static:
    feats["feat_static_cat"] = datasets.Sequence(datasets.Value("string"))

dst = Path(a.dst)
tmp = Path(str(dst) + "_tmp")
shutil.rmtree(tmp, ignore_errors=True)
out = src.map(convert, batched=True, batch_size=500, remove_columns=src.column_names,
              features=datasets.Features(feats), desc="convirtiendo")
out.save_to_disk(str(tmp))
shutil.rmtree(dst, ignore_errors=True)
tmp.rename(dst)

out = datasets.load_from_disk(str(dst))
L = np.array([len(t) for t in out.data.column("target").to_pylist()])
r = out.with_format("numpy")[0]
print(out)
print({k: getattr(v, "shape", v) for k, v in r.items()})
print(f"series: {len(out)} | largo: min {L.min()} mediana {int(np.median(L))} max {L.max()} | "
      f"series con < 2x28+1 dias: {(L < 57).sum()}")
assert not any(np.isnan(np.asarray(x)).any() for x in out.with_format("numpy")[:50]["feat_dynamic_real"]), "NaN en covariables"
print("guardado en", dst)
