import logging
import os
from pathlib import Path

import numpy as np
import torch
from gluonts.torch import PyTorchPredictor
from gluonts.transform import UniformSplitSampler

np.bool = np.bool_

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Única base model soportado: Chronos-Bolt (base).
CHRONOS_BOLT_DIRNAME = "chronos-bolt-base"


def get_chronos_bolt_path(model_dir: str) -> str:
    """Ruta local al checkpoint si existe; si no, el id de HuggingFace."""
    local = os.path.join(model_dir, CHRONOS_BOLT_DIRNAME)
    return local if os.path.isdir(local) else f"amazon/{CHRONOS_BOLT_DIRNAME}"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_predictor(args, dataset, season_length, ds_config, pl_logger=None, load_ckpt=False, init_weight=None, **kwargs):
    name = args.model_name
    batch_size = args.batch_size
    model_dir = os.getenv("MODEL_PATH", "./models/")
    model_path = get_chronos_bolt_path(model_dir)
    if args.save_ckpt:
        infos = dict(vars(args))
        del infos["save_ckpt"]
        del infos["datasets"]
        infos["ds_name"] = ds_config
    else:
        infos = None
    while True:
        try:
            if name.startswith("ts_adapter/"):
                # UniCA (adapter) sobre Chronos-Bolt congelado
                from .adapter.estimator import TSAdapterEstimator
                logger.info("Using TSAdapterEstimator")
                _, module_name = name.split("/")
                if args.sampler == "random":
                    from data.sampler import RandomSampler
                    logger.info("#### Using RandomSampler ####")
                    train_sampler = RandomSampler(min_future=dataset.prediction_length)
                else:
                    logger.info("#### Using UniformSplitSampler ####")
                    train_sampler = UniformSplitSampler(p=1.0, min_future=dataset.prediction_length)
                from data.sampler import UniformWithStartSampler
                val_sampler = UniformWithStartSampler(
                    start=- dataset.split_point,
                    min_future=dataset.prediction_length,
                )
                estimator = TSAdapterEstimator(
                    model_id_or_path=model_path,
                    freq=dataset.freq,
                    lr=args.lr,
                    prediction_length=dataset.prediction_length,
                    num_batches_per_epoch=args.num_batches_per_epoch,
                    batch_size=batch_size,
                    item_map=dataset.item_map,
                    weight_decay=args.weight_decay,
                    model_kwargs={
                        "with_gate": args.with_gate,
                        "future_with_gate": args.future_with_gate,
                        "with_future": args.with_future,
                        "with_static": args.with_static,
                        "with_past": args.with_past,
                        "dropout_rate": args.dropout,
                    },
                    pl_logger=pl_logger,
                    num_workers=args.num_workers,
                    train_sampler=train_sampler,
                    validation_sampler=val_sampler,
                    trainer_kwargs={"max_epochs": args.max_epochs, "accelerator": "auto",
                                    "gradient_clip_val": args.gradient_clip},
                    indexed_sample=args.indexed_sample,
                    imputation=args.imputation,
                    module_name=module_name,
                    args=args,
                    ckpt_info=infos,
                    **dataset.dim_params,
                )
                if load_ckpt:
                    assert init_weight is not None, "Please provide the path to the checkpoint"
                    predictor = PyTorchPredictor.deserialize(Path(init_weight))
                elif args.split_val:
                    predictor = estimator.train(training_data=dataset.training_dataset,
                                                validation_data=dataset.validation_dataset,
                                                shuffle_buffer_length=100)
                else:
                    predictor = estimator.train(dataset.validation_dataset,
                                                shuffle_buffer_length=100)
            elif name == "chronos_bolt":
                # Baseline zero-shot (sin covariables)
                from .predictor.chronos import ChronosPredictor
                logger.info(f"Using {name} zero-shot predictor")
                predictor = ChronosPredictor(
                    model_path=model_path,
                    num_samples=args.num_samples,
                    prediction_length=dataset.prediction_length,
                    device_map=get_device(),
                )
            elif name == "naive":
                # Referencia para normalizar métricas (naive.json), igual que en el paper
                from statsforecast.models import Naive
                from .predictor.naive import StatsForecastPredictor

                class NaivePredictor(StatsForecastPredictor):
                    ModelType = Naive

                logger.info("Using naive predictor")
                predictor = NaivePredictor(
                    dataset.prediction_length,
                    season_length=season_length,
                    freq=dataset.freq,
                    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    batch_size=512,
                )
            else:
                raise ValueError(f"Unknown model name: {name} (use 'ts_adapter/unica', 'chronos_bolt' or 'naive')")
            break
        except torch.cuda.OutOfMemoryError as e:
            print(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}")
            batch_size //= 2
            if batch_size < 1:
                raise e
    return predictor
