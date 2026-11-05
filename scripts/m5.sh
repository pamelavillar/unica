#!/usr/bin/env bash
# UniCA + Chronos-Bolt-base sobre M5.
# Requiere: export DATA_PATH=/ruta/a/datos  (debe contener m5/M5_test)
#           export MODEL_PATH=/ruta/a/modelos (opcional; si no existe chronos-bolt-base se baja de HuggingFace)
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 1) Baseline zero-shot de Chronos-Bolt (sin covariables)
python main.py --model_name chronos_bolt --datasets m5 --batch_size 128

# 2) UniCA (configuración del paper para datasets unimodales con covariables)
python main.py --model_name ts_adapter/unica \
 --datasets m5 \
 --with_gate --with_future --future_with_gate \
 --num_workers 4 \
 --num_batches_per_epoch 50 \
 --max_epochs 100 \
 --indexed_sample \
 --split_val \
 --gradient_clip 0.1 \
 --batch_size 128
