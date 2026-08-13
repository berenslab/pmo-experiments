"""Wrapper script for OpenCLIP training."""

import datetime
import logging
import os
import shutil
import subprocess
import sys
from typing import Sequence

from pmo_experiments.configs.config import (
    ConfigType,
    config_from_command_line,
    get_config,
    save_config,
)
from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset
from pmo_experiments.util import get_cpu_count, get_free_port
from pmo_experiments.util.model_interface import (
    find_latest_model_folder,
)

logger = logging.getLogger(__name__)


def convert_cfg_to_open_clip_args(
    cfg: ConfigType,
    dataset: BaseDataset,
    resume_from: str | None = None,
    wds_patterns: tuple[str, str] | None = None,
) -> tuple[list[str], str]:
    """
    Convert current configuration to OpenCLIP interface.

    Args:
        cfg (ConfigType): Current configuration.
        dataset (BaseDataset): Dataset object.
        resume_from (str | None): Path to checkpoint to resume from. If None, will
            attempt to find latest checkpoint if resuming is enabled in config. Defaults
            to None.
        wds_patterns (tuple[str, str] | None): Brace-expanded shard patterns for
            (train, val) as returned by `to_webdataset()`. Required when
            USE_WEBDATASET is True.

    Returns:
        tuple[list[str], str]: OpenCLIP compatible argument list and save folder path.

    """
    effective_batch_size = (
        cfg["TRAINING"]["BATCH_SIZE_PER_GPU"]
        * cfg["TRAINING"]["NUM_DEVICES"]
        * cfg["TRAINING"]["GRADIENT_ACCUMULATION_STEPS"]
    )
    training_dataset_size, validation_dataset_size = dataset.get_dataset_size()

    assert training_dataset_size is not None and training_dataset_size > 0, (
        "Training dataset size must be known to compute number of warmup steps."
    )

    has_valid_batch_size = (
        cfg["TRAINING"].get("BATCH_SIZE_PER_GPU") is not None
        and cfg["TRAINING"]["BATCH_SIZE_PER_GPU"] > 0
    )
    assert has_valid_batch_size, (
        "Batch size per GPU must be set to compute number of warmup steps."
    )
    num_steps_per_epoch = training_dataset_size // effective_batch_size

    wandb_run_name = cfg["OUTPUT"].get("RUN_NAME")
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if wandb_run_name is None:
        pretrained_name = cfg["MODEL"]["PRETRAINED_WEIGHTS"]
        if pretrained_name is None:
            pretrained_name = "scratch"

        architecture_name = (
            cfg["MODEL"]["ARCHITECTURE"].replace("/", "-").replace(":", "-")
        )

        if "biomedclip" in architecture_name.lower():
            architecture_name = "vit_b_16"
            pretrained_name = "biomed_clip"

        wandb_run_name = (
            f"{cfg['OUTPUT']['WANDB_PROJECT_BASE_NAME']}_"
            f"{architecture_name}_"
            f"{pretrained_name}_"
            f"{cfg['DATASET']['NAME']}_"
            f"bs{effective_batch_size}_"
            f"s{cfg['SEED']}"
        )

    resume = cfg.get("RESUME", False)

    if resume:
        if resume_from is None:
            latest_model_folder = find_latest_model_folder(
                cfg["OUTPUT"]["SAVE_DIR"], wandb_run_name
            )

            if latest_model_folder is None:
                raise FileNotFoundError(
                    f"No existing model folder found for {wandb_run_name} "
                    f"in {cfg['OUTPUT']['SAVE_DIR']} to resume from."
                )

            save_folder = latest_model_folder
            wandb_run_name = os.path.basename(save_folder)
            resume_from = "latest"
        else:
            save_folder = os.path.dirname(os.path.dirname(os.path.dirname(resume_from)))
            wandb_run_name = os.path.basename(save_folder)

            # make sure save_folder exists
            if not os.path.exists(save_folder):
                raise FileNotFoundError(
                    f"Resume save folder {save_folder} does not exist."
                )

            if not os.path.exists(resume_from):
                raise FileNotFoundError(
                    f"Resume checkpoint {resume_from} does not exist."
                )

            # Make sure the structure is correct
            checkpoints_folder = os.path.join(save_folder, "open_clip", "checkpoints")
            if os.path.abspath(checkpoints_folder) != os.path.dirname(resume_from):
                raise ValueError(
                    f"Resume checkpoint {resume_from} is not in the expected "
                    f"checkpoints folder {checkpoints_folder}."
                )
        current_open_clip_folder = os.path.join(save_folder, "open_clip")
        target_open_clip_folder = os.path.join(current_open_clip_folder, wandb_run_name)

        # TODO this is so inefficient but open_clip insists on this structure
        if os.path.exists(target_open_clip_folder):
            raise FileExistsError(
                f"Target open_clip folder {target_open_clip_folder} already exists."
            )

        # Copy current_open_clip_folder to target_open_clip_folder
        shutil.copytree(current_open_clip_folder, target_open_clip_folder)

    else:
        wandb_run_name += f"_{current_timestamp}"

        save_folder = os.path.join(
            cfg["OUTPUT"]["SAVE_DIR"],
            wandb_run_name,
        )

    num_warmup_steps = cfg["TRAINING"]["WARMUP_EPOCHS"] * num_steps_per_epoch
    use_wds = cfg["TRAINING"].get("USE_WEBDATASET", False)

    if use_wds:
        assert wds_patterns is not None, (
            "wds_patterns must be provided when USE_WEBDATASET is True"
        )
        wds_dataset_args: dict[str, object] = {
            "--dataset-type": "webdataset",
            "--train-data": wds_patterns[0],
            "--val-data": wds_patterns[1],
            "--train-num-samples": training_dataset_size,
            "--val-num-samples": validation_dataset_size,
        }
    else:
        wds_dataset_args = {
            "--dataset-type": "csv",
            "--train-data": f"{dataset.converted_dataset_path}/train.csv",
            "--val-data": f"{dataset.converted_dataset_path}/val.csv",
            "--csv-img-key": dataset.get_image_path_column_name(),
            "--csv-caption-key": dataset.get_caption_column_name(),
            "--csv-sep": ",",
        }

    open_clip_args = {
        "--accum-freq": cfg["TRAINING"]["GRADIENT_ACCUMULATION_STEPS"],
        "--save-frequency": "1",
        "--report-to": "wandb",
        **wds_dataset_args,
        "--model": cfg["MODEL"]["ARCHITECTURE"],
        "--batch-size": cfg["TRAINING"]["BATCH_SIZE_PER_GPU"],
        "--epochs": cfg["TRAINING"]["EPOCHS"],
        "--lr": cfg["TRAINING"]["LEARNING_RATE"],
        "--wd": cfg["TRAINING"]["WEIGHT_DECAY"],
        "--precision": cfg["TRAINING"]["PRECISION"],
        "--workers": cfg["TRAINING"]["WORKERS"],
        "--seed": cfg["SEED"],
        "--wandb-project": cfg["OUTPUT"]["WANDB_PROJECT_BASE_NAME"],
        "--name": wandb_run_name,
        "--logs": os.path.join(save_folder, "open_clip"),  # cfg["OUTPUT"]["SAVE_DIR"],
        "--log-every-n-steps": cfg["OUTPUT"]["LOG_INTERVAL"],
        "--warmup": num_warmup_steps,
        "--beta1": cfg["TRAINING"]["ADAM_BETA1"],
        "--beta2": cfg["TRAINING"]["ADAM_BETA2"],
    }

    if resume:
        open_clip_args["--resume"] = resume_from

    if cfg["GRADIENT_CHECKPOINTING"]:
        open_clip_args["--grad-checkpointing"] = ""

    if cfg["MODEL"]["PRETRAINED_WEIGHTS"] is not None:
        open_clip_args["--pretrained"] = cfg["MODEL"]["PRETRAINED_WEIGHTS"]

    text_augmentation_path = dataset.generate_text_augmentation_dict()

    if text_augmentation_path is not None:
        open_clip_args["--text-augmentation-dict"] = text_augmentation_path
        open_clip_args["--text-augmentation-seed"] = cfg["SEED"]

    return [
        str(item)
        for sublist in open_clip_args.items()
        for item in sublist
        if item is not None and item != ""
    ], save_folder


def post_training_cleanup(final_save_folder: str) -> None:
    """
    Clean up folder and move files after training.

    Args:
        final_save_folder (str): Model save folder.

    """
    # Clean up: Move files from open_clip/<name> to open_clip
    open_clip_folder = os.path.join(final_save_folder, "open_clip")
    if os.path.exists(open_clip_folder):
        # Check for folder inside open_clip/
        inner_folder_name = os.path.join(
            open_clip_folder, os.path.basename(final_save_folder)
        )
        if os.path.exists(inner_folder_name):
            logger.info(
                f"Moving open_clip files from {inner_folder_name} to open_clip/"
            )

            params_file = os.path.join(inner_folder_name, "params.txt")
            output_log_file = os.path.join(inner_folder_name, "out.log")
            checkpoints_folder = os.path.join(inner_folder_name, "checkpoints")

            if os.path.exists(params_file):
                # Move params.txt
                shutil.move(params_file, os.path.join(open_clip_folder, "params.txt"))

            if os.path.exists(output_log_file):
                # Move out.log
                shutil.move(output_log_file, os.path.join(open_clip_folder, "out.log"))

            if os.path.exists(checkpoints_folder):
                # Iterate over all files in checkpoints folder and move them or delete
                for item in os.listdir(checkpoints_folder):
                    s = os.path.join(checkpoints_folder, item)
                    d = os.path.join(open_clip_folder, "checkpoints", item)

                    os.makedirs(os.path.dirname(d), exist_ok=True)

                    if os.path.exists(d):
                        if d.endswith(".pt"):
                            # Delete new model files
                            os.remove(s)
                            continue
                        # Overwrite existing files
                        if os.path.isfile(d):
                            os.remove(d)
                        else:
                            # TODO merge directories?
                            logger.debug(f"Skipping existing folder {d}")
                            continue
                    shutil.move(s, d)
                try:
                    os.rmdir(checkpoints_folder)
                except OSError:
                    logger.warning(
                        f"Could not remove checkpoints folder {checkpoints_folder}, "
                        f"not empty."
                    )

            # Remove now empty folder
            try:
                os.rmdir(inner_folder_name)
            except OSError:
                logger.warning(
                    f"Could not remove folder {inner_folder_name}, not empty."
                )


def run_training(
    config_path: str, resume_from: str | None, overrides: Sequence[str]
) -> None:
    """
    Run the training process.

    Args:
        config_path (str): Path to the configuration YAML file.
        resume_from (str | None): Path to a checkpoint (.pt) to resume training from.
            If None, will attempt to find latest checkpoint if resuming is enabled in
            config.
        overrides (Sequence[str]): Configuration overrides in the format key=value.

    """
    # Get rank and world_size from environment (set by torchrun) without initializing
    # Let OpenCLIP handle process group initialization

    # Only rank 0 processes the dataset and config
    cfg = get_config(config_path)
    cfg = config_from_command_line(cfg, overrides)

    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    launcher = ["python", "-m", "open_clip_train.main"]

    if cfg["TRAINING"]["NUM_DEVICES"] > 1:
        master_port = get_free_port()

        launcher = [
            "torchrun",
            "--nproc-per-node",
            str(cfg["TRAINING"]["NUM_DEVICES"]),
            f"--rdzv_endpoint=127.0.0.1:{master_port}",
            "-m",
            "open_clip_train.main",
        ]

    # Prepare dataset
    dataset_args = {
        str(key): str(value)
        for key, value in cfg["DATASET"].items()
        if value is not None
    }
    dataset = DATASET_REGISTRY.get(cfg["DATASET"]["NAME"])(**dataset_args)
    dataset.to_open_clip_format()

    if cfg["TRAINING"].get("USE_WEBDATASET", False):
        effective_batch_size = (
            cfg["TRAINING"]["BATCH_SIZE_PER_GPU"]
            * cfg["TRAINING"]["NUM_DEVICES"]
            * cfg["TRAINING"]["GRADIENT_ACCUMULATION_STEPS"]
        )
        wds_patterns = dataset.to_webdataset(
            batch_size=effective_batch_size,
            num_workers=get_cpu_count() * 2,
        )
    else:
        wds_patterns = None

    open_clip_args, final_save_folder = convert_cfg_to_open_clip_args(
        cfg, dataset, resume_from=resume_from, wds_patterns=wds_patterns
    )

    # Save final config
    # NOTE: cannot save own config directly to the open_clip save folder, because
    # open_clip complains if the folder is not empty
    if not cfg.get("RESUME", False):
        os.makedirs(final_save_folder, exist_ok=True)
        cfg_save_path = os.path.join(final_save_folder, "config.yaml")

        # Save config without __BASE__ entries
        cfg["__BASE__"] = None
        save_config(cfg, cfg_save_path)

    # Launch OpenCLIP training as a subprocess so it handles distributed setup cleanly
    cmd = launcher + open_clip_args

    logger.info(f"Running training with command:\n{' '.join(cmd)}")

    env = os.environ.copy()

    # Options to avoid back propagation timeouts in NCCL
    env["NCCL_TIMEOUT"] = "1800"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    logger.debug(
        "Using environment variables: "
        f"NCCL_TIMEOUT={env['NCCL_TIMEOUT']}, "
        f"TORCH_NCCL_ASYNC_ERROR_HANDLING={env['TORCH_NCCL_ASYNC_ERROR_HANDLING']}, "
        f"TORCH_NCCL_BLOCKING_WAIT={env['TORCH_NCCL_BLOCKING_WAIT']}"
    )

    result = subprocess.run(cmd, check=False, env=env)

    if result.returncode != 0:
        logger.error("Training process failed.")

    post_training_cleanup(final_save_folder)

    sys.exit(result.returncode)
