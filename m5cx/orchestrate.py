"""
Protocolo para UniCA en M5, reducido para Colab y reanudable (misma estructura que el de ChronosX).

  Etapa A  barrido de lr: 1 corrida por lr, --sweep_epochs epocas, solo entrenamiento.
           Se elige el lr con menor perdida de validacion. (El repo NO hace barrido: usa lr=5e-5 fijo;
           para replicarlo tal cual pasa --best_lr 5e-5.)
  Etapa B  --final_runs corridas con el mejor lr y --final_epochs epocas (semillas 42, 43, ...),
           cada una evaluada en test contra Chronos-Bolt zero-shot.
  Etapa C  tabla resumen (summary.md / summary.json).

Todo lo que ya esta hecho se salta: si Colab corta, vuelve a lanzar el mismo comando.
Ejemplo (M5 completo):
    python m5cx/orchestrate.py --out_root /content/drive/MyDrive/pfc1/unica/m5_cx
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="m5/M5_test")
p.add_argument("--num_series", type=int, default=0)
p.add_argument("--out_root", required=True)
p.add_argument("--sweep_lrs", type=float, nargs="*", default=[5e-5, 5e-4])
p.add_argument("--sweep_epochs", type=int, default=30, help="30 x 50 lotes = 1500 pasos")
p.add_argument("--best_lr", type=float, default=None, help="si se da, se salta el barrido")
p.add_argument("--final_epochs", type=int, default=100, help="como scripts/unimodal.sh (con early stopping)")
p.add_argument("--final_runs", type=int, default=3)
p.add_argument("--num_batches_per_epoch", type=int, default=50)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--accum", type=int, default=1)
p.add_argument("--num_workers", type=int, default=2)
p.add_argument("--cache_items", type=int, default=2000)
p.add_argument("--base_model", default="chronos_bolt_base")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--eval_batch_size", type=int, default=128)
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--no_naive_norm", action="store_true", help="datos distintos al M5_test de los autores")
p.add_argument("--filter_short", action="store_true", help="entrenar solo con series >= 2H+1 dias (como ChronosX)")
p.add_argument("--summarize_only", action="store_true")
args = p.parse_args()

os.environ["MPLBACKEND"] = "Agg"  # lo heredan las corridas (Colab exporta un backend inline invalido en el venv)
root = Path(args.out_root)
(root / "logs").mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(label, out_dir, extra):
    cmd = [
        sys.executable, "-u", str(here / "run_unica.py"),
        "--dataset", args.dataset, "--num_series", str(args.num_series),
        "--base_model", args.base_model, "--device", args.device, "--seed", str(args.seed),
        "--num_batches_per_epoch", str(args.num_batches_per_epoch), "--batch_size", str(args.batch_size),
        "--accum", str(args.accum), "--num_workers", str(args.num_workers),
        "--cache_items", str(args.cache_items), "--eval_batch_size", str(args.eval_batch_size),
        "--n_boot", str(args.n_boot), "--out", str(out_dir), "--baseline_dir", str(root / "baseline"),
    ] + (["--no_naive_norm"] if args.no_naive_norm else []) + (["--filter_short"] if args.filter_short else []) + extra
    logfile = root / "logs" / f"{label}.log"
    log(f"{label}: lanzando (log: {logfile})")
    t = time.time()
    with open(logfile, "a") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"{label}: termino con codigo {rc} en {(time.time() - t) / 60:.1f} min")
    if rc != 0:
        log(f"ERROR en {label}. Revisa el final de {logfile}. Vuelve a lanzar el mismo comando "
            f"para reanudar (retoma desde el ultimo checkpoint).")
        sys.exit(rc)


def summarize():
    runs = []
    for d in sorted((root / "final").glob("run=*")):
        f = d / "results.json"
        if f.exists():
            runs.append(json.load(open(f)))
    if not runs:
        log("No hay corridas finales terminadas todavia.")
        return
    b = runs[0]["baseline"]
    metrics = [("MSE", "MSE[mean]"), ("MAE", "MAE[0.5]"), ("MAPE", "MAPE[0.5]"),
               ("MASE", "MASE[0.5]"), ("WQL", "mean_weighted_sum_quantile_loss")]
    u = {k: np.mean([r["unica"][k] for r in runs]) for _, k in metrics}

    L = [f"{runs[0]['dataset']} ({runs[0]['n_series']} series, horizonte {runs[0]['prediction_length']})", ""]
    L.append("UniCA:")
    L += [f"- {name}: {u[k]:.4f}" for name, k in metrics]
    L.append("")
    L.append("Chronos-Bolt:")
    L += [f"- {name}: {b[k]:.4f}" for name, k in metrics]

    (root / "summary.md").write_text("\n".join(L))
    json.dump({"runs": runs}, open(root / "summary.json", "w"), indent=1)
    print("\n".join(L))


if args.summarize_only:
    summarize()
    sys.exit(0)

state_f = root / "state.json"
state = json.load(open(state_f)) if state_f.exists() else {}

# ---------------------------------------------------------------- A. barrido
if args.best_lr is not None:
    best_lr = args.best_lr
    log(f"lr dado por el usuario: {best_lr} (sin barrido)")
elif "best_lr" in state:
    best_lr = state["best_lr"]
    log(f"barrido ya hecho: mejor lr = {best_lr}  {state['sweep']}")
else:
    sweep = {}
    for lr in args.sweep_lrs:
        d = root / "sweep" / f"lr={lr}"
        run(f"sweep_lr{lr}", d, ["--lr", str(lr), "--max_epochs", str(args.sweep_epochs),
                                  "--run_id", "0", "--skip_test_eval"])
        sweep[str(lr)] = json.load(open(d / "train_done.json"))["val_loss_unica"]
    best_lr = float(min(sweep, key=sweep.get))
    state.update({"sweep": sweep, "best_lr": best_lr, "sweep_epochs": args.sweep_epochs})
    json.dump(state, open(state_f, "w"), indent=1)
    log(f"barrido: {sweep} -> mejor lr = {best_lr}")

# ---------------------------------------------------------------- B. corridas finales
for r in range(args.final_runs):
    d = root / "final" / f"run={r}"
    run(f"final_run{r}", d, ["--lr", str(best_lr), "--max_epochs", str(args.final_epochs), "--run_id", str(r)])

# ---------------------------------------------------------------- C. resumen
summarize()
