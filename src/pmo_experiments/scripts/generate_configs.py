"""Config generation for the experiments in pmo-experiments."""

import os

from omegaconf import DictConfig, OmegaConf

from pmo_experiments.configs.config import _merge_dict_configs, save_config
from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY

DEFAULT_DATASETS = [
    "deepeyenet",
    "kaggle_eyepacs",
    "flair",
    "pubmed_ophtha",
    "pubmed_ophtha_cfp",
    "pubmed_ophtha_cfp_matched",
    "pubmed_ophtha_decon",
]

DEFAULT_SEEDS = [106080954, 2612551882, 3796366246]

DEFAULT_MODELS = [
    "ViT-B-16",
    "ViT-B-32",
    "RN50",
    "RN50x4",
    "ViT-L-14",
]

EPOCHS = 50
PRETRAINED_NAME = "openai"
BATCH_SIZES = [2048]

MODEL_TO_GPU_COUNT_MAP = {
    "ViT-B-16": 1,
    "ViT-B-32": 1,
    "RN50": 4,
    "RN50x4": 8,
    "ViT-L-14": 2,
}

DEFAULT_LR = 0.00003
DATASET_TO_LR_MAP = {
    "deepeyenet": 0.0001,
    "kaggle_eyepacs": 0.00002,
}

MODEL_DEFAULT_CONFIG = {
    "__BASE__": None,
    "MERGE_DEFAULT": False,
    "TRAINING": {
        "WEIGHT_DECAY": 0.2,
        "WORKERS": 10,
        "PRECISION": "amp",
        "WARMUP_EPOCHS": 5,
        "ADAM_BETA1": 0.9,
        "ADAM_BETA2": 0.98,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        "EPOCHS": EPOCHS,
    },
    "MODEL": {
        "PRETRAINED_WEIGHTS": PRETRAINED_NAME,
    },
    "RESUME": False,
    "DATASET": {
        "DATASET_PATH": None,
        "CONVERTED_DATASET_PATH": None,
    },
    "GRADIENT_CHECKPOINTING": True,
    "OUTPUT": {
        "LOG_INTERVAL": 7,
        "RUN_NAME": None,
    },
}


def generate_config(
    model_name: str,
    seed: int,
    dataset_name: str,
    batch_size: int,
    output_folder_base: str = "clip_models",
    wandb_project_name: str = "pmo-experiments",
) -> DictConfig:
    """
    Generate a config for the given parameters.

    Args:
        model_name (str): Name of the model.
        seed (int): The training seed.
        dataset_name (str): Name of the training dataset.
        batch_size (int): Batch size across all devices.
        output_folder_base (str, optional): Folder base name of where the models will
            be saved. Will be appended with _{seed}. Defaults to "clip_models".
        wandb_project_name (str, optional): Base name for the wandb project. Used to
            generate the run name. Defaults to "pmo-experiments".

    Raises:
        ValueError: If the batch size is not divisible by the number of devices.

    Returns:
        DictConfig: Resulting training config.

    """
    # TODO make flexible
    num_devices = MODEL_TO_GPU_COUNT_MAP[model_name]
    lr = DATASET_TO_LR_MAP.get(dataset_name, DEFAULT_LR)

    batch_size_per_gpu = batch_size / num_devices

    if batch_size % num_devices != 0:
        raise ValueError(
            "Batch size must be divisible by number of devices: "
            f"{batch_size} % {num_devices} != 0"
        )

    default_cfg = OmegaConf.create(MODEL_DEFAULT_CONFIG)
    cfg = OmegaConf.create(
        {
            "MODEL": {
                "ARCHITECTURE": model_name,
            },
            "TRAINING": {
                "NUM_DEVICES": num_devices,
                "BATCH_SIZE_PER_GPU": int(batch_size_per_gpu),
                "LEARNING_RATE": lr,
            },
            "DATASET": {
                "NAME": dataset_name,
            },
            "OUTPUT": {
                "SAVE_DIR": f"{output_folder_base}_{seed}",
                "WANDB_PROJECT_BASE_NAME": wandb_project_name,
            },
            "SEED": seed,
        }
    )

    # Merge
    cfg = _merge_dict_configs(default_cfg, cfg)

    return cfg


def generate_all_configs(
    save_folder: str,
    model_output_base: str,
    model_names: list[str] | None,
    dataset_names: list[str] | None,
    seeds: list[int] | None,
    save: bool = True,
    wandb_project_name: str = "pmo-experiments",
) -> list[DictConfig]:
    """
    Generate all configs of the given parameterset.

    Args:
        save_folder (str): Folder to save the configs to.
        model_output_base (str): The base of the output folder for the trained models.
        model_names (list[str] | None): Names of the models. If None uses the defaults.
        dataset_names (list[str] | None): Names of the training datasets. If None uses
            the defaults.
        seeds (list[int] | None): List of training seeds. If None uses the defaults.
        save (bool, optional): If True saves the generated configs. Defaults to True.
        wandb_project_name (str, optional): Base name for the wandb project. Used to
            generate the run name. Defaults to "pmo-experiments".

    Raises:
        ValueError: _description_
        ValueError: If model or dataset name is unknown.

    Returns:
        list[DictConfig]: List of generated configs.

    """
    if model_names is None:
        model_names = DEFAULT_MODELS

    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    if seeds is None:
        seeds = DEFAULT_SEEDS

    for model_name in model_names:
        if model_name not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model '{model_name}'! Use --list-models.")

    for dataset_name in dataset_names:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{dataset_name}! Use --list-datasets.")

    if save:
        os.makedirs(save_folder, exist_ok=True)

    configs = []

    for seed in seeds:
        seed_folder = os.path.join(save_folder, str(seed))
        for dataset_name in dataset_names:
            dataset_save_folder = os.path.join(seed_folder, dataset_name)

            if save:
                os.makedirs(dataset_save_folder, exist_ok=True)

            for model_name in model_names:
                normalized_name = model_name.lower().replace("-", "_")
                for batch_size in BATCH_SIZES:
                    cfg = generate_config(
                        model_name,
                        seed,
                        dataset_name,
                        batch_size=batch_size,
                        output_folder_base=model_output_base,
                        wandb_project_name=wandb_project_name,
                    )

                    if save:
                        save_config(
                            cfg,
                            os.path.join(
                                dataset_save_folder,
                                f"config_{normalized_name}_bs{batch_size}.yaml",
                            ),
                        )

                    configs.append(cfg)

    return configs


def list_models() -> None:
    """Print the known model catalog keys."""
    print("List of registered models:")
    for name in sorted(DEFAULT_MODELS):
        print(f"\t- {name}")


def list_datasets() -> None:
    """Print the registered dataset names."""
    print("List of registered datasets:")
    for name in sorted(name for name, _ in DATASET_REGISTRY):
        print(f"\t- {name}")
