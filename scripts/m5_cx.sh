#!/usr/bin/env bash
# UniCA + Chronos-Bolt-base sobre el M5 de ChronosX (preparacion oficial del proyecto, ver m5cx/README.md).
# Requiere: export DATA_PATH=/ruta/a/datos   (aqui se crea m5/M5_test)
#           export MODEL_PATH=/ruta/a/modelos (debe contener chronos-bolt-base)
set -e
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

# 1) Datos: M5 de ChronosX -> formato UniCA (solo la primera vez)
if [ ! -f "$DATA_PATH/m5/M5_test/dataset_info.json" ]; then
  mkdir -p ./m5_raw
  (cd m5cx && python -W ignore -c "from m5_lean import build_m5; print(build_m5(directory='../m5_raw'))")
  python -W ignore m5cx/m5_to_unica.py --src ./m5_raw/m5_lean_full --dst "$DATA_PATH/m5/M5_test"
fi

# 2) 3 corridas finales con lr=5e-4 (quita --best_lr para barrer 5e-5 y 5e-4) + resumen (reanudable: relanzar el mismo comando)
python -u -W ignore m5cx/orchestrate.py --best_lr 5e-4 --final_epochs 100 --final_runs 3 \
  --num_batches_per_epoch 50 --batch_size 64 --accum 2 --num_workers 2 --no_naive_norm --filter_short \
  --out_root ./output/m5_cx "$@"
