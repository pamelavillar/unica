"""
Una corrida de UniCA (Chronos-Bolt-base) vs Chronos-Bolt zero-shot sobre M5 (m5/M5_test del repo).

Vive en UniCA/m5cx/ (junto a ua_utils.py); se puede lanzar desde cualquier carpeta, con DATA_PATH / MODEL_PATH
definidos (las rutas relativas de --out se resuelven desde la raiz del repo):

    python m5cx/run_unica.py --lr 5e-5 --max_epochs 100 --run_id 0 --out ./output/run0

Usa el MISMO pipeline que `main.py --model_name ts_adapter/unica` (dataset, transformaciones, muestreo
indexado, split de validacion, perdida, early stopping de 20 epocas), con estos cambios para Colab:
  * checkpoints por epoca en --out/ckpt (solo los parametros ENTRENABLES: unos MB, no los ~800 MB de
    Chronos-Bolt congelado), y reanudacion automatica desde el ultimo checkpoint completo;
  * logger CSV en vez de wandb;
  * cache acotada de series en el muestreo (ver ua_utils.enable_light_cache);
  * metricas por serie + bootstrap emparejado contra Chronos-Bolt zero-shot en la MISMA ventana.
Este repo es la version de UniCA recortada a M5: sin homogeneizador multimodal (imagen/texto), asi que UniCA
usa solo las covariables numericas/categoricas de M5 (igual que el repo original con use_satellite/use_text=False).

Pasos (cada uno se salta si ya esta hecho, asi que se puede relanzar tras un corte):
  1. Baseline Chronos-Bolt zero-shot -> metricas por serie en <baseline_dir>/<clave>.npz
  2. Entrenamiento de UniCA (reanuda desde --out/ckpt/last.ckpt si existe)
  3. Evaluacion de UniCA en la MISMA ventana de test, con IC95% (bootstrap emparejado por serie)
Con --skip_test_eval solo se entrena y se guarda la perdida de validacion (para el barrido de lr).
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"  # Colab exporta un backend inline que el matplotlib del venv no conoce
HERE = Path(__file__).resolve().parent  # UniCA/m5cx (ua_utils.py)
ROOT = HERE.parent  # UniCA/ (utils.py, data/, models/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)  # el repo abre data/dataset_properties.json y naive.json con rutas relativas

import numpy as np  # noqa: E402

np.bool = np.bool_  # models/utils.py del repo tambien lo hace (codigo viejo que usa np.bool)

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="m5/M5_test", help="carpeta dentro de DATA_PATH")
p.add_argument("--num_series", type=int, default=0, help="0 = todas las series; >0 = subconjunto aleatorio")
p.add_argument("--base_model", default="chronos_bolt_base")
p.add_argument("--device", default="cuda")
p.add_argument("--lr", type=float, default=5e-5, help="valor por defecto del repo (utils.py)")
p.add_argument("--max_epochs", type=int, default=100, help="como scripts/unimodal.sh")
p.add_argument("--num_batches_per_epoch", type=int, default=50, help="como scripts/unimodal.sh")
p.add_argument("--batch_size", type=int, default=128, help="valor por defecto del repo")
p.add_argument("--accum", type=int, default=1,
               help="acumulacion de gradiente: lote efectivo = batch_size x accum (num_batches_per_epoch se "
                    "multiplica por accum para mantener los pasos de optimizador por epoca)")
p.add_argument("--num_workers", type=int, default=2, help="el repo usa 4; Colab gratis tiene 2 CPU")
p.add_argument("--cache_items", type=int, default=2000, help="series transformadas en cache por worker")
p.add_argument("--run_id", type=int, default=0, help="semilla de entrenamiento = seed + run_id")
p.add_argument("--seed", type=int, default=42, help="semilla del repo")
p.add_argument("--eval_batch_size", type=int, default=128)
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--skip_test_eval", action="store_true")
p.add_argument("--out", default="./output/run")
p.add_argument("--baseline_dir", default="./output/baseline")
p.add_argument("--no_naive_norm", action="store_true",
               help="no normalizar con naive.json (que es del M5_test de los autores, no de otros datos)")
p.add_argument("--filter_short", action="store_true",
               help="entrenar/validar solo con series de >= 2H+1 dias, como ChronosX (el test usa todas)")
p.add_argument("--forecast_from", default=None,
               help="npz de scripts/forecast_series.py (ChronosX): pronostica esas MISMAS series con UniCA "
                    "(desde --out/ckpt/best.ckpt), guarda --forecast_out y termina. No entrena ni evalua.")
p.add_argument("--forecast_out", default=None)
p.add_argument("--subset_root", default="/content/data_subsets",
               help="donde se guarda el subconjunto de series cuando --num_series > 0")
args_cli = p.parse_args()

import torch  # noqa: E402

from ua_utils import (  # noqa: E402
    aggregate, enable_light_cache, get_repo_args, item_metrics, latest_complete_checkpoint,
    paired_bootstrap, save_json,
)

t0 = time.time()
out = Path(args_cli.out).resolve()
out.mkdir(parents=True, exist_ok=True)
if (out / "results.json").exists() and not args_cli.forecast_from:
    print(f"[skip] {out}/results.json ya existe")
    sys.exit(0)
if args_cli.skip_test_eval and (out / "train_done.json").exists():
    print(f"[skip] {out}/train_done.json ya existe")
    sys.exit(0)

timing = {}


def tick(name, t_start):
    timing[name] = round((time.time() - t_start) / 60, 2)
    print(f"[tiempo] {name}: {timing[name]} min", flush=True)


# ------------------------------------------------------------------ datos (subconjunto opcional)
def prepare_subset():
    """Copia aleatoria y reproducible de N series en <subset_root>/nN_sS/<dataset> (para pruebas)."""
    import datasets

    src = Path(os.environ["DATA_PATH"]) / args_cli.dataset
    root = Path(args_cli.subset_root) / f"n{args_cli.num_series}_s{args_cli.seed}"
    dst = root / args_cli.dataset
    if not (dst / "dataset_info.json").exists():
        ds = datasets.load_from_disk(str(src))
        idx = np.sort(np.random.default_rng(args_cli.seed).choice(len(ds), args_cli.num_series, replace=False))
        ds.select(idx.tolist()).save_to_disk(str(dst))
    return str(root)


if args_cli.num_series:
    os.environ["DATA_PATH"] = prepare_subset()

train_seed = args_cli.seed + args_cli.run_id
repo_cli = [
    "--model_name", "ts_adapter/unica", "--base_model", args_cli.base_model,
    "--with_gate", "--with_future", "--future_with_gate",
    "--datasets", "m5", "--num_workers", args_cli.num_workers,
    "--num_batches_per_epoch", args_cli.num_batches_per_epoch * args_cli.accum,
    "--max_epochs", args_cli.max_epochs, "--indexed_sample", "--split_val", "--gradient_clip", "0.1",
    "--batch_size", args_cli.batch_size, "--lr", args_cli.lr, "--seed", train_seed,
]
args, _ = get_repo_args(repo_cli)  # tambien fija las semillas como main.py

from gluonts.time_feature import get_seasonality  # noqa: E402

from utils import get_dataset  # noqa: E402

t = time.time()
print(f"Cargando {args_cli.dataset} (num_series={args_cli.num_series or 'todas'})...", flush=True)
dataset, ds_config, ds_key = get_dataset(args, args_cli.dataset, "short")
H = dataset.prediction_length
season = get_seasonality(dataset.freq)
test_data = dataset.test_data
n_series = dataset.num_series
print(f"Series: {n_series} | freq: {dataset.freq} | horizonte: {H} | ventanas de test: {len(test_data)} | "
      f"dims de covariables: {dataset.dim_params}", flush=True)
tick("carga_datos", t)

# Entrenamiento/validacion solo con series de al menos 2H+1 observaciones (lo mismo que hace ChronosX con
# has_enough_observations). Con menos, la ventana de validacion queda vacia y el sampler del repo divide por cero.
# El TEST sigue usando TODAS las series: test_data ya quedo construido sobre el dataset completo.
import pyarrow.compute as pc  # noqa: E402

_lens = pc.list_value_length(dataset.hf_dataset.data.column("target")).to_numpy()
_keep = np.where(_lens >= 2 * H + 1)[0]
if args_cli.filter_short and len(_keep) < len(_lens):
    dataset.hf_dataset = dataset.hf_dataset.select(_keep.tolist())
    print(f"Entrenamiento/validacion: {len(_keep)} de {len(_lens)} series (se excluyen {len(_lens) - len(_keep)} "
          f"con menos de {2 * H + 1} dias; en test se evaluan todas)", flush=True)

naive_all = json.load(open("naive.json"))
# naive.json es del M5_test completo de los autores: no aplica a subconjuntos ni a otros datos
naive = naive_all.get(ds_config) if not (args_cli.num_series or args_cli.no_naive_norm) else None

MODEL_DIR = os.getenv("MODEL_PATH", "./models/")
PATHS = {"chronos_bolt_base": os.path.join(MODEL_DIR, "chronos-bolt-base")}
assert args_cli.base_model in PATHS, "este notebook solo cubre chronos_bolt_base"


# ------------------------------------------------------------------ 1. baseline
base_items = None
if not args_cli.skip_test_eval and not args_cli.forecast_from:
    t = time.time()
    key = f"{args_cli.dataset.replace('/', '_')}_n{args_cli.num_series}_s{args_cli.seed}_{args_cli.base_model}"
    npz = Path(args_cli.baseline_dir) / f"{key}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Baseline: Chronos-Bolt zero-shot (sin covariables)", flush=True)
    if npz.exists():
        base_items = dict(np.load(npz))
        print(f"  (baseline cargado de {npz})", flush=True)
    else:
        from models.predictor.chronos import ChronosPredictor  # el mismo que usa el repo

        pred = ChronosPredictor(model_path=PATHS[args_cli.base_model], num_samples=args.num_samples,
                                prediction_length=H, device_map=args_cli.device)
        base_fc = pred.predict(test_data.input, batch_size=args_cli.eval_batch_size)
        base_items = item_metrics(base_fc, test_data, seasonality=season)
        np.savez(npz, **base_items)
        del pred, base_fc
        if args_cli.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("  Chronos-Bolt:", aggregate(base_items, naive), flush=True)
    tick("baseline", t)


# ------------------------------------------------------------------ 2. estimador del repo, reanudable
import lightning.pytorch as pl  # noqa: E402
from gluonts.env import env  # noqa: E402
from gluonts.torch.model.estimator import TrainOutput  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from data.sampler import RandomSampler, UniformWithStartSampler  # noqa: E402
from models.adapter.estimator import NaNCheckCallback, TSAdapterEstimator  # noqa: E402

enable_light_cache(args_cli.cache_items)
CKPT_DIR = out / "ckpt"


def use_fork_workers():
    """gluonts/torch/__init__.py hace set_start_method("spawn", force=True) si hay GPU. Con spawn, cada worker
    del DataLoader re-ejecuta este script (y recargaria M5) -> RuntimeError de multiprocessing. Los workers solo
    hacen numpy (no tocan CUDA), asi que fork es seguro y ademas comparte la memoria del proceso principal."""
    import multiprocessing

    import torch.multiprocessing as tmp

    tmp.set_start_method("fork", force=True)
    multiprocessing.set_start_method("fork", force=True)


def slim_checkpoints(module):
    """Guarda en el .ckpt solo los parametros entrenables (Chronos-Bolt congelado se recarga del disco)."""
    frozen = {n for n, prm in module.named_parameters(remove_duplicate=False) if not prm.requires_grad}

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"]):
            if k in frozen:
                del checkpoint["state_dict"][k]
        # save_hyperparameters() del repo mete en el .ckpt el ChronosWrapper entero (Chronos-Bolt pickleado,
        # ~800 MB). Para reanudar no hace falta: el modulo ya esta construido.
        checkpoint.pop("hyper_parameters", None)
        checkpoint.pop("hparams_name", None)

    module.on_save_checkpoint = types.MethodType(on_save_checkpoint, module)
    module.strict_loading = False  # al reanudar, faltan (a proposito) los pesos congelados
    return module


def load_trained(module, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = module.load_state_dict(sd, strict=False)
    trainable = {n for n, prm in module.named_parameters() if prm.requires_grad}
    lost = trainable & set(missing)
    assert not lost and not unexpected, f"checkpoint incompatible: faltan {sorted(lost)[:5]} / sobran {unexpected[:5]}"
    return module


class ResumableEstimator(TSAdapterEstimator):
    """TSAdapterEstimator.train_model del repo, con checkpoints livianos por epoca y reanudacion."""

    def train_model(self, training_data, validation_data=None, from_predictor=None,
                    shuffle_buffer_length=None, cache_data=False, ckpt_path=None, **kwargs):
        use_fork_workers()
        transformation = self.create_transformation()
        with env._let(max_idle_transforms=max(len(training_data), 100)):
            net = slim_checkpoints(self.create_lightning_module())
            train_dl = self.create_training_data_loader(
                training_data, net, transformations=transformation, shuffle_buffer_length=shuffle_buffer_length)
        with env._let(max_idle_transforms=max(len(validation_data), 100)):
            val_dl = self.create_validation_data_loader(validation_data, net, transformations=transformation)

        n_train = sum(prm.numel() for prm in net.parameters() if prm.requires_grad)
        n_all = sum(prm.numel() for prm in net.parameters())
        print(f"  parametros entrenables: {n_train:,} de {n_all:,} ({100 * n_train / n_all:.2f}%)", flush=True)

        ckpt = pl.callbacks.ModelCheckpoint(dirpath=CKPT_DIR, filename="best", monitor="val_loss", mode="min",
                                            save_top_k=1, save_last=True, verbose=True)
        early = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True,
                                           min_delta=1e-6)  # igual que el repo
        trainer = pl.Trainer(
            logger=CSVLogger(str(out), name="logs"),
            callbacks=[ckpt, early, NaNCheckCallback()],
            enable_progress_bar=False,
            log_every_n_steps=10,
            enable_model_summary=False,
            **self.trainer_kwargs,
        )
        resume, skipped = latest_complete_checkpoint(CKPT_DIR)
        for c, why in skipped:
            print(f"[resume] IGNORO {c.name}: {why}", flush=True)
        if resume is not None:
            print(f"[resume] reanudando desde {resume}", flush=True)
        trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=val_dl,
                    ckpt_path=str(resume) if resume else None)
        best = ckpt.best_model_path
        print(f"  mejor checkpoint: {best} (val_loss={float(ckpt.best_model_score):.5f}, "
              f"epocas: {trainer.current_epoch})", flush=True)
        load_trained(net, best)
        self._best_val = float(ckpt.best_model_score)
        self._epochs = int(trainer.current_epoch)
        return TrainOutput(transformation=transformation, trained_net=net, trainer=trainer,
                           predictor=self.create_predictor(transformation, net))


def build_estimator():
    """Mismos argumentos que models/utils.get_predictor (rama ts_adapter)."""
    return ResumableEstimator(
        freq=dataset.freq, lr=args.lr, prediction_length=H,
        num_batches_per_epoch=args.num_batches_per_epoch, batch_size=args.batch_size,
        item_map=dataset.item_map, weight_decay=args.weight_decay,
        model_kwargs={
            "with_gate": args.with_gate, "future_with_gate": args.future_with_gate,
            "with_future": args.with_future,
            "with_static": args.with_static, "with_past": args.with_past,
            "dropout_rate": args.dropout,
        },
        pl_logger=None, num_workers=args.num_workers,
        train_sampler=RandomSampler(min_future=H),
        validation_sampler=UniformWithStartSampler(start=-dataset.split_point, min_future=H),
        trainer_kwargs={"max_epochs": args.max_epochs,
                        "accelerator": "gpu" if args_cli.device.startswith("cuda") else "cpu",
                        "gradient_clip_val": args.gradient_clip,
                        "accumulate_grad_batches": args_cli.accum},
        indexed_sample=args.indexed_sample, imputation=args.imputation, module_name="unica", args=args,
        ckpt_info=None,
        **dataset.dim_params, model_id_or_path=PATHS[args_cli.base_model],
    )


def forecast_quantiles(forecasts):
    """(n, 9, H). UniCA (como Chronos-Bolt) devuelve sus 9 cuantiles (0.1..0.9) como 'muestras': se usan tal cual."""
    out_q = []
    for f in forecasts:
        s = np.asarray(f.samples) if hasattr(f, "samples") else None
        if s is not None and s.shape[0] == 9:
            out_q.append(np.sort(s, axis=0))
        else:
            out_q.append(np.stack([f.quantile(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]))
    return np.stack(out_q).astype(np.float32)


if args_cli.forecast_from:
    import datasets

    cxf = dict(np.load(args_cli.forecast_from))
    ids = [str(x) for x in cxf["ids"]]
    src_ids = datasets.load_from_disk(str(Path(os.environ["DATA_PATH"]) / args_cli.dataset)).data.column(
        "item_id").to_pylist()
    pos = {sid: i for i, sid in enumerate(src_ids)}
    missing = [x for x in ids if x not in pos]
    assert not missing, f"no estan en el dataset de UniCA: {missing}"
    want = {pos[x]: k for k, x in enumerate(ids)}
    sel_in, sel_y = [None] * len(ids), [None] * len(ids)
    for i, (inp, lab) in enumerate(zip(test_data.input, test_data.label)):
        if i in want:
            sel_in[want[i]], sel_y[want[i]] = inp, np.asarray(lab["target"], np.float32)
    y = np.stack(sel_y)
    assert np.allclose(y, cxf["y"], equal_nan=True), "el test de UniCA no coincide con el de ChronosX en estas series"
    print(f"Pronosticando {len(ids)} series (mismo test que ChronosX): {ids}", flush=True)

    estimator = build_estimator()
    net = load_trained(estimator.create_lightning_module(), CKPT_DIR / "best.ckpt")
    predictor = estimator.create_predictor(estimator.create_transformation(), net)
    predictor.prediction_net.eval()
    torch.manual_seed(args_cli.seed)
    ua = list(predictor.predict(sel_in))
    cxf["UniCA"] = forecast_quantiles(ua)
    dst = Path(args_cli.forecast_out or out / "forecasts.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **cxf)
    print("Guardado en", dst, flush=True)
    sys.exit(0)

print("\n[2/3] Entrenando UniCA", flush=True)
if (out / "train_done.json").exists():
    td = json.load(open(out / "train_done.json"))
    print("  (entrenamiento ya hecho)", flush=True)
    predictor = None
else:
    t = time.time()
    estimator = build_estimator()
    predictor = estimator.train(training_data=dataset.training_dataset,
                                validation_data=dataset.validation_dataset, shuffle_buffer_length=100)
    td = {
        "val_loss_unica": estimator._best_val,
        "epochs_run": estimator._epochs,
        "early_stopped": estimator._epochs < args_cli.max_epochs,
        "sec_per_step": round((time.time() - t) / max(1, estimator._epochs * args_cli.num_batches_per_epoch), 3),
        "lr": args_cli.lr, "max_epochs": args_cli.max_epochs,
        "batch_size": args_cli.batch_size, "accum": args_cli.accum,
        "run_id": args_cli.run_id, "train_seed": train_seed,
    }
    save_json(td, out / "train_done.json")
    for f in CKPT_DIR.glob("last*.ckpt"):  # best.ckpt se conserva (unos MB)
        f.unlink()
    tick("entrenamiento", t)

print(f"  val_loss UniCA (mejor epoca) = {td['val_loss_unica']:.5f} | epocas: {td['epochs_run']} | "
      f"early stopping: {td['early_stopped']} | s/paso (incluye validacion): {td.get('sec_per_step')}", flush=True)
if args_cli.skip_test_eval:
    print("[fin] --skip_test_eval: solo entrenamiento.", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ 3. evaluacion
t = time.time()
print("\n[3/3] Evaluando UniCA", flush=True)
if predictor is None:  # reconstruye desde best.ckpt (sesion nueva)
    estimator = build_estimator()
    net = load_trained(estimator.create_lightning_module(), CKPT_DIR / "best.ckpt")
    predictor = estimator.create_predictor(estimator.create_transformation(), net)
predictor.prediction_net.eval()
predictor.batch_size = args_cli.eval_batch_size
torch.manual_seed(args_cli.seed)
fc = list(predictor.predict(test_data.input))
items = item_metrics(fc, test_data, seasonality=season)
np.savez(out / "item_metrics.npz", **items)
base_agg, ua_agg = aggregate(base_items, naive), aggregate(items, naive)
boot = paired_bootstrap(base_items, items, n_boot=args_cli.n_boot, seed=args_cli.seed)
tick("evaluacion", t)

rel = lambda a, b: 100 * (b - a) / a  # noqa: E731
res = {
    "dataset": ds_config, "num_series": args_cli.num_series, "n_series": int(n_series),
    "n_items": base_agg["n_items"], "prediction_length": H,
    **{k: getattr(args_cli, k) for k in ("lr", "max_epochs", "num_batches_per_epoch", "batch_size", "accum",
                                         "run_id", "seed", "base_model")},
    **{k: td[k] for k in ("val_loss_unica", "epochs_run", "early_stopped")},
    "baseline": base_agg,
    "unica": ua_agg,
    "rel_change_pct": {
        "MASE": rel(base_agg["MASE[0.5]"], ua_agg["MASE[0.5]"]),
        "WQL": rel(base_agg["mean_weighted_sum_quantile_loss"], ua_agg["mean_weighted_sum_quantile_loss"]),
        "MAE": rel(base_agg["MAE[0.5]"], ua_agg["MAE[0.5]"]),
    },
    "paired_bootstrap": boot,
    "timing_min": {**timing, "total": round((time.time() - t0) / 60, 2)},
}
save_json(res, out / "results.json")

print("\n===== RESULTADOS (menor es mejor) =====")
print("Chronos-Bolt zero-shot:", base_agg)
print("UniCA                 :", ua_agg)
print("Cambio relativo %     :", res["rel_change_pct"])
print("Bootstrap emparejado  :", boot if boot else "n/a (menos de 30 series)")
print("Resultados guardados en", out / "results.json")
