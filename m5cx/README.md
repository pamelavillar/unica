# m5cx: UniCA en el M5 de ChronosX

Preparación **oficial** del proyecto para comparar UniCA con ChronosX serie por serie. Son los mismos scripts
que usa el notebook de Colab `unica_m5_datos_chronosx.py`.

| Archivo | Qué hace |
|---|---|
| `m5_lean.py` | Arma M5 igual que ChronosX: 30,490 series desde la primera venta, d_1..d_1969, 12 covariables. Es una copia exacta de `chronosx/utils/m5_lean.py` del repo de ChronosX. |
| `m5_to_unica.py` | Convierte esa salida al formato que lee `data/unimodal/data.py` (`item_id`, `start`, `freq`, `target`, `feat_dynamic_real`). |
| `ua_utils.py` | Args del repo, caché acotada de series (RAM), reanudación desde `.ckpt`, métricas por serie y bootstrap emparejado. |
| `run_unica.py` | Una corrida: Chronos-Bolt zero-shot vs UniCA, con el mismo pipeline que `main.py --model_name ts_adapter/unica`. |
| `orchestrate.py` | Barrido de lr, corridas finales y `summary.md`. Es reanudable. |
| `compare_models.py` | Tabla de 4 modelos (Chronos-T5, ChronosX, Chronos-Bolt, UniCA), leyendo los resultados de ambos notebooks. |

La alternativa con los CSV de Kaggle (`data/prepare_m5.py`) **no** es comparable con ChronosX.

## Uso (desde la raíz `UniCA/`)

```bash
export DATA_PATH=/ruta/datos MODEL_PATH=/ruta/modelos   # MODEL_PATH/chronos-bolt-base
(cd m5cx && python -c "from m5_lean import build_m5; build_m5(directory='../m5_raw')")
python m5cx/m5_to_unica.py --src ./m5_raw/m5_lean_full --dst "$DATA_PATH/m5/M5_test"
python m5cx/orchestrate.py --best_lr 5e-4 --final_runs 3 --batch_size 64 --accum 2 \
    --no_naive_norm --filter_short --out_root ./output/m5_cx
```

`scripts/m5_cx.sh` hace todo esto.
