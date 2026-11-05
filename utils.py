import argparse
import json
import logging

import numpy as np
import torch

from data import Dataset

logger = logging.getLogger(__name__)

dataset_properties_map = json.load(open("data/dataset_properties.json"))

# Único dataset del proyecto: M5 (subconjunto de test, frecuencia diaria, horizonte 28).
datasets_dict = {
    "m5": ["m5/M5_test"],
}


def get_args_and_settings():
    parser = argparse.ArgumentParser(description="UniCA + Chronos-Bolt sobre M5")
    # Modelo
    parser.add_argument("--model_name", type=str, default="ts_adapter/unica",
                        choices=["ts_adapter/unica", "chronos_bolt", "naive"],
                        help="'ts_adapter/unica' = UniCA sobre Chronos-Bolt; 'chronos_bolt' = zero-shot")
    parser.add_argument("--datasets", type=str, nargs="+", default=["m5"], choices=datasets_dict.keys())
    parser.add_argument("--num_samples", type=int, default=16, help="Muestras para predicción zero-shot")
    # Entrenamiento
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_batches_per_epoch", "-n", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="random", choices=["random", "uniform"])
    parser.add_argument("--imputation", type=str, default="last", choices=["last", "dummy"])
    parser.add_argument("--indexed_sample", action="store_true", help="Muestreo indexado (más rápido)")
    parser.add_argument("--split_val", action="store_true", help="Usar split de validación separado")
    parser.add_argument("--wo_time", action="store_true", help="No añadir features de calendario")
    # Arquitectura UniCA
    parser.add_argument("--past_pos", type=str, default="pre", choices=["pre", "post"],
                        help="Fusión de covariables pasadas antes/después del encoder")
    parser.add_argument("--future_pos", type=str, default="post", choices=["pre", "post"],
                        help="Fusión de covariables futuras antes/después del encoder")
    parser.add_argument("--with_gate", action="store_true", help="Gate sobre covariables pasadas")
    parser.add_argument("--with_future", action="store_true", help="Usar covariables futuras (atención)")
    parser.add_argument("--future_with_gate", action="store_true", help="Gate sobre covariables futuras")
    parser.add_argument("--no_past", dest="with_past", action="store_false", default=True)
    parser.add_argument("--no_static", dest="with_static", action="store_false", default=True)
    # Salida
    parser.add_argument("--save_ckpt", action="store_true", help="Guardar el predictor entrenado")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    set_seeds(args.seed)
    torch.set_num_threads(16)
    print(vars(args))
    datasets = []
    for ds in args.datasets:
        datasets.extend(datasets_dict[ds])
    return args, datasets


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args, ds_name, term):
    # ds_name = "m5/M5_test" -> ds_key = "m5", subset = "M5_test"
    ds_key, subset = ds_name.split("/")
    ds_config = f"{ds_key}/{subset}/{term}"
    logger.info(f"Running model on dataset: {ds_config}")

    args.ds_name = ds_name
    args.term = term
    dataset = Dataset(name=ds_name, term=term, indexed_sample=args.indexed_sample)
    assert dataset.target_dim == 1
    return dataset, ds_config, ds_key
