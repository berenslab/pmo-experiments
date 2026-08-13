"""Model discovery and evaluation sweep logic for CLIP model evaluation."""

import glob
import logging
import os
import re

import open_clip
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from pmo_experiments.configs.config import merge_with_default, save_config
from pmo_experiments.evaluation.full_evaluation import (
    is_fully_evaluated,
    run_full_evaluation,
)

logger = logging.getLogger(__name__)

ALL_EVAL_DATASETS = [
    # "idrid_disease_grading",
    # "idrid_segmentation",
    "rfmid_1",
    "rfmid_2",
    "joint_rfmid",
    "brset",
    "aptos2019",
    "refuge2",
    "eddfs",
]

KNN_K: dict[str, int] = {
    "idrid_disease_grading": 10,
    "rfmid_1": 10,
    "rfmid_2": 10,
    "joint_rfmid": 10,
    "idrid_segmentation": 5,
    "brset": 20,
    "aptos2019": 20,
    "refuge2": 20,
    "eddfs": 10,
}

MODEL_PRETRAIN_PREFIXES: list[tuple[str, str, str]] = [
    ("vit_b_16_biomed_clip", "vit_b_16", "biomed_clip"),
    ("vit_b_16_scratch", "vit_b_16", "scratch"),
    ("vit_b_16_openai", "vit_b_16", "openai"),
    ("vit_b_32_openai", "vit_b_32", "openai"),
    ("vit_l_14_openai", "vit_l_14", "openai"),
    ("rn50x4_openai", "rn50x4", "openai"),
    ("rn50_openai", "rn50", "openai"),
]

# Maps baseline subfolder name → (model, pretrain, train_dataset)
BASELINE_METADATA: dict[str, tuple[str, str, str]] = {
    "openai_clip": ("vit_b_16", "openai", "openai_clip"),
    "biomed_clip": ("vit_b_16", "biomed_clip", "pmc15m"),
}


def create_baseline_model_folder(
    architecture: str,
    pretrained: str | None,
    output_folder: str,
    epoch: int,
) -> None:
    """
    Materialize a baseline (non-fine-tuned) model folder.

    The open_clip baseline do not conform to the project folder structure, so this
    function creates a folder with the expected structure.

    Args:
        architecture (str): open_clip model name, e.g. "ViT-B-16", "RN50", or an
            hf-hub architecture string that bundles its own weights.
        pretrained (str | None): open_clip pretrained tag, e.g. "openai". None if the
            architecture string already bundles pretrained weights (e.g. hf-hub).
        output_folder (str): Folder to write config.yaml and the checkpoint into.
        epoch (int): Epoch number to save the checkpoint under. This is purely a
            filename convention to match whatever epoch the eval run uses — the
            baseline was not actually trained for this many epochs.

    """
    checkpoint_path = os.path.join(
        output_folder, "open_clip", "checkpoints", f"epoch_{epoch}.pt"
    )
    config_path = os.path.join(output_folder, "config.yaml")
    if os.path.exists(checkpoint_path) and os.path.exists(config_path):
        return

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(
        architecture, pretrained=pretrained, device="cpu", load_weights=True
    )
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    cfg = merge_with_default(
        OmegaConf.create(
            {
                "MERGE_DEFAULT": True,
                "MODEL": {
                    "ARCHITECTURE": architecture,
                    "PRETRAINED_WEIGHTS": pretrained,
                },
                "TRAINING": {"PRECISION": "amp"},
            }
        )
    )
    save_config(cfg, config_path)


def _normalize(s: str) -> str:
    return s.lower().replace("-", "_")


def parse_model_folder(
    path: str, run_prefix: str = "pmo-experiments"
) -> dict[str, str | None]:
    """Extract seed, model, pretrain, and train_dataset from a folder path."""
    basename = os.path.basename(path)

    seed_match = re.search(r"_s(\d+)_\d{8}_\d{6}$", basename)
    seed = seed_match.group(1) if seed_match else None

    # Strip the run_prefix and "_bs{N}_s{N}_{date}_{time}"
    # suffix.
    match_string = f"^{re.escape(run_prefix)}_"
    body = re.sub(match_string, "", basename)
    body = re.sub(r"_bs\d+_s\d+_\d{8}_\d{6}$", "", body)
    body_norm = _normalize(body)

    model: str | None = None
    pretrain: str | None = None
    train_dataset: str | None = None
    for prefix_norm, m, p in MODEL_PRETRAIN_PREFIXES:
        if body_norm.startswith(prefix_norm + "_"):
            model = m
            pretrain = p
            train_dataset = body_norm[len(prefix_norm) + 1 :]
            break

    return {
        "path": path,
        "seed": seed,
        "model": model,
        "pretrain": pretrain,
        "train_dataset": train_dataset,
    }


def parse_baseline_folders(
    baseline_folder: str, autocreate: bool = True, epoch: int = 50
) -> list[dict[str, str | None]]:
    """Return model info dicts for all known baseline model folders."""
    infos = []
    for name, (model, pretrain, train_dataset) in BASELINE_METADATA.items():
        path = os.path.join(baseline_folder, name)
        if os.path.isdir(path):
            infos.append(
                {
                    "path": path,
                    "seed": None,
                    "model": model,
                    "pretrain": pretrain,
                    "train_dataset": train_dataset,
                }
            )
        elif autocreate:
            logger.info(f"Creating baseline model folder: {path}")
            create_baseline_model_folder(
                architecture=model,
                pretrained=pretrain,
                output_folder=path,
                epoch=epoch,
            )
            infos.append(
                {
                    "path": path,
                    "seed": None,
                    "model": model,
                    "pretrain": pretrain,
                    "train_dataset": train_dataset,
                }
            )
    return infos


def _matches(
    info: dict[str, str | None],
    seeds: list[str],
    models: list[str],
    pretrains: list[str],
    train_datasets: list[str],
) -> bool:
    if len(seeds) > 0 and info["seed"] is not None and info["seed"] not in seeds:
        return False
    if len(models) > 0 and info["model"] not in models:
        return False
    if len(pretrains) > 0 and info["pretrain"] not in pretrains:
        return False
    if len(train_datasets) > 0 and info["train_dataset"] not in train_datasets:
        return False
    return True


def run_evaluation_sweep(
    model_base_folders: list[str] | None = None,
    baseline_folder: str = "clip_baseline_models",
    epoch: int = 50,
    seeds: list[str] | None = None,
    models: list[str] | None = None,
    pretrains: list[str] | None = None,
    train_datasets: list[str] | None = None,
    eval_datasets: list[str] | None = None,
    output_dir: str = "evaluation_results",
    batch_size: int = 16,
    workers: int = 16,
    device: str = "cuda",
    include_baselines: bool = False,
    eval_split: str = "test",
    force_rescore: bool = False,
    skip_baseline_autocreate: bool = False,
    run_prefix: str = "pmo-experiments",
) -> None:
    """
    Discover trained models and run evaluation over every (model, eval_dataset) pair.

    Discovers all models at a given epoch from one or more model-base folders and runs
    run_full_evaluation on each (model, eval_dataset) pair. Already-completed steps are
    skipped.

    Args:
        model_base_folders (list[str] | None, optional): Folders to search for
            trained model runs (each is globbed for '<run_prefix>*'
            subfolders). If None uses every 'clip_models_*' folder found in the
            current directory. Defaults to None.
        baseline_folder (str, optional): Folder containing baseline (non-fine-tuned)
            model folders. Defaults to "clip_baseline_models".
        epoch (int, optional): Epoch checkpoint to evaluate. Defaults to 50.
        seeds (list[str] | None, optional): Only include models with these training
            seeds. If None or empty, no filtering is applied. Defaults to None.
        models (list[str] | None, optional): Only include these model architectures.
            If None or empty, no filtering is applied. Defaults to None.
        pretrains (list[str] | None, optional): Only include these pretraining
            sources. If None or empty, no filtering is applied. Defaults to None.
        train_datasets (list[str] | None, optional): Only include models trained on
            these datasets. If None or empty, no filtering is applied. Defaults to
            None.
        eval_datasets (list[str] | None, optional): Datasets to evaluate on. If None
            uses ALL_EVAL_DATASETS. Defaults to None.
        output_dir (str, optional): Root output directory. Defaults to
            "evaluation_results".
        batch_size (int, optional): Batch size for embedding extraction. Defaults to
            16.
        workers (int, optional): Number of dataloader workers. Defaults to 16.
        device (str, optional): Device to load models onto. Defaults to "cuda".
        include_baselines (bool, optional): Also evaluate the pre-trained baseline
            models from `baseline_folder`. Defaults to False.
        eval_split (str, optional): Which split to evaluate on. Defaults to "test".
        force_rescore (bool, optional): Recompute every step even if reports exist,
            overwriting them and writing the score .npz files. Use to backfill runs
            produced before raw scores were persisted. Defaults to False.
        skip_baseline_autocreate (bool, optional): Skip auto-creating baseline model
            folders if they don't exist. If False, baseline folders are created
            on-the-fly. Defaults to False.
        run_prefix (str, optional): Prefix for the run name. Usually constructed from
            the wandb project name and metadata. Defaults to "pmo-experiments".

    """
    lr_C_range = [0.01, 0.1, 1.0, 10.0]
    if model_base_folders is None:
        model_base_folders = sorted(glob.glob("clip_models_*"))
    if seeds is None:
        seeds = []
    if models is None:
        models = []
    if pretrains is None:
        pretrains = []
    if train_datasets is None:
        train_datasets = []
    if eval_datasets is None:
        eval_datasets = ALL_EVAL_DATASETS

    # Discover and parse all model folders.
    all_infos = []
    for base in model_base_folders:
        for path in sorted(glob.glob(os.path.join(base, f"{run_prefix}*"))):
            if os.path.isdir(path):
                all_infos.append(parse_model_folder(path, run_prefix=run_prefix))

    if include_baselines:
        all_infos.extend(
            parse_baseline_folders(
                baseline_folder,
                autocreate=not skip_baseline_autocreate,
                epoch=epoch,
            )
        )

    # Warn about folders that couldn't be parsed.
    unparsed = [i for i in all_infos if i["model"] is None]
    if len(unparsed) > 0:
        logger.warning(
            f"{len(unparsed)} folder(s) could not be parsed and will be skipped:"
        )
        for u in unparsed:
            logger.warning(f"  {u['path']}")

    # Apply filters.
    filtered = [
        i
        for i in all_infos
        if i["model"] is not None
        and _matches(i, seeds, models, pretrains, train_datasets)
    ]

    logger.info(
        f"Found {len(all_infos)} model folders, {len(filtered)} after filtering."
    )

    # Build task list.
    tasks = []
    num_finished = 0
    for info in filtered:
        for dataset_name in eval_datasets:
            output_folder = os.path.join(
                output_dir,
                eval_split,
                f"{os.path.basename(info['path'])}_epoch{epoch}",
                dataset_name,
            )
            if is_fully_evaluated(
                output_folder, dataset_name, force_rescore, lr_C_range
            ):
                num_finished += 1
                continue

            tasks.append((info["path"], dataset_name, output_folder))

    logger.info(
        f"Running {len(tasks)} evaluation tasks"
        f"{'' if force_rescore else f', skipped {num_finished} finished tasks'}."
    )

    for model_path, dataset_name, output_folder in tqdm(tasks):
        os.makedirs(output_folder, exist_ok=True)
        try:
            run_full_evaluation(
                model_path=model_path,
                dataset_name=dataset_name,
                epoch=epoch,
                model_device=device,
                batch_size=batch_size,
                dataloader_workers=workers,
                output_folder=output_folder,
                knn_k=KNN_K.get(dataset_name, 20),
                test_split=eval_split,
                force_rescore=force_rescore,
                show_progress=False,
            )
        except FileNotFoundError as e:
            logger.error(f"File not found for {model_path} on {dataset_name}: {e}")
        except Exception as e:
            logger.error(f"Error evaluating {model_path} on {dataset_name}: {e}")
