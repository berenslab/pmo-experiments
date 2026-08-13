"""
Module to interface with OpenCLIP models: loading models, tokenizers, and preprocessors.

Defines functions to load trained models from checkpoints and retrieve necessary
components for inference.
"""

import contextlib
import datetime
import glob
import logging
import os

import open_clip
import torch
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from pmo_experiments.configs.config import get_config


class SuppressNoPretrained(logging.Filter):
    """Logging filter to suppress specific warnings about pretrained weights."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter missing pretraining weights warnings.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the record should be logged, False if it should be suppressed.

        """
        return (
            "No pretrained weights loaded" not in record.getMessage()
            and "Model initialized randomly" not in record.getMessage()
        )


def _load_clip_model(
    folder: str, device: str | torch.device, epoch: str | int
) -> tuple[
    torch.nn.Module,
    tuple[Compose, Compose],
    (
        open_clip.tokenizer.SimpleTokenizer
        | open_clip.tokenizer.HFTokenizer
        | open_clip.tokenizer.SigLipTokenizer
    ),
]:
    """
    Load a clip model from a given folder.

    Args:
        folder (str): Path to the folder containing the model config and checkpoints.
        device (str | torch.device): Device to load the model onto.
        epoch (str | int): Epoch number to load the corresponding checkpoint.

    Raises:
        FileNotFoundError: If config or checkpoint files are not found.

    Returns:
        tuple[
            torch.nn.Module,
            tuple[Compose, Compose],
            (
                open_clip.tokenizer.SimpleTokenizer
                | open_clip.tokenizer.HFTokenizer
                | open_clip.tokenizer.SigLipTokenizer
            ),
        ]: Tuple containing the model, train and validation image preprocessor, and
            tokenizer.

    """
    config_file_path = os.path.join(folder, "config.yaml")
    epoch_checkpoint_path = os.path.join(
        folder,
        "open_clip",
        "checkpoints",
        f"epoch_{epoch}.pt",
    )
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    if not os.path.exists(epoch_checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint for epoch {epoch} not found at {epoch_checkpoint_path}"
        )

    cfg = get_config(config_file_path)

    model_name = cfg["MODEL"]["ARCHITECTURE"]
    model_precision = cfg["TRAINING"]["PRECISION"]

    # Filter weight warning
    open_clip_logger = logging.getLogger()
    open_clip_logger.addFilter(SuppressNoPretrained())
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        precision=model_precision,
        device=device,
        load_weights=False,
    )

    # Remove filter after model creation to allow warnings for other models
    open_clip_logger.removeFilter(SuppressNoPretrained())

    # Load the trained weights
    checkpoint = torch.load(epoch_checkpoint_path, map_location=device)

    # Remove 'module.' prefix from data parallel training if present
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Get the tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, (preprocess_train, preprocess_val), tokenizer  # pyright: ignore[reportReturnType]


def find_latest_model_folder(models_root_folder: str, model_name: str) -> str | None:
    """
    Find the latest model folder for the specified model_name.

    Relies on the appended timestamp.

    Args:
        models_root_folder (str): Root model directory.
        model_name (str): Name of the model.

    Returns:
        str | None: Path to the latest model folder, or None if not found.

    """
    # Find all folders in the models root folder that match
    # the model name (+ timestamp, optional)
    pattern = os.path.join(models_root_folder, f"{model_name}_*")
    candidate_folders = glob.glob(pattern)

    if len(candidate_folders) == 0:
        return None

    # Find the folder with the latest modification time
    latest_folder = max(
        candidate_folders,
        key=lambda x: datetime.datetime.strptime(
            "_".join(os.path.basename(x).split("_")[-2:]), "%Y%m%d_%H%M%S"
        ),
    )

    return latest_folder


def find_latest_checkpoint(model_folder: str) -> int:
    """
    Find the latest epoch checkpoint number in the specified model folder.

    Args:
        model_folder (str): Path to the folder with the model checkpoints.

    Returns:
        int: Latest epoch number. Returns -1 if no checkpoints are found.

    """
    checkpoints_folder = os.path.join(model_folder, "open_clip", "checkpoints")
    epoch_files = [
        f
        for f in os.listdir(checkpoints_folder)
        if f.startswith("epoch_") and f.endswith(".pt")
    ]
    if len(epoch_files) == 0:
        return -1
    epochs = [
        int(f[len("epoch_") : -len(".pt")])  # Extract epoch number from filename
        for f in epoch_files
    ]
    epoch = max(epochs)

    return epoch


def load_checkpoint(
    model_folder: str,
    epoch: int | str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[
    torch.nn.Module,
    Compose,
    (
        open_clip.tokenizer.SimpleTokenizer
        | open_clip.tokenizer.HFTokenizer
        | open_clip.tokenizer.SigLipTokenizer
    ),
]:
    """
    Load a trained CLIP model from a specified folder and epoch.

    If epoch is None the latest epoch checkpoint will be loaded.

    Args:
        model_folder (str): Path to the folder with the model config and checkpoints.
        epoch (int | str | None): Epoch number to load. If None, loads the latest epoch.
            Defaults to None.
        device (str | torch.device): Device to load the model onto. Defaults to "cpu".

    Raises:
        FileNotFoundError: If no checkpoint files are found.

    Returns:
        tuple[
            torch.nn.Module,
            Compose,
            (
                open_clip.tokenizer.SimpleTokenizer
                | open_clip.tokenizer.HFTokenizer
                | open_clip.tokenizer.SigLipTokenizer
            ),
        ]: Tuple containing the model, validation image preprocessor, and tokenizer.

    """
    if epoch is None:
        epoch = find_latest_checkpoint(model_folder)
        if epoch == -1:
            raise FileNotFoundError(
                f"No epoch checkpoint files found in {model_folder}"
            )

    model, preprocessors, tokenizer = _load_clip_model(model_folder, device, epoch)
    return model, preprocessors[1], tokenizer


def extract_image_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Extract image embeddings from a dataloader.

    Args:
        model (torch.nn.Module): Vision model to extract embeddings from.
        dataloader (torch.utils.data.DataLoader): PyTorch DataLoader providing
            image batches and labels.
        show_progress (bool, optional): Whether to display a tqdm progress bar.
            Defaults to True.

    Returns:
        torch.Tensor: Tensor of normalized image embeddings (CPU)

    """
    # TODO move to utils
    all_embeddings = []

    model_device = next(model.parameters()).device
    current_dtype = next(model.parameters()).dtype

    # Use autocast only for CUDA devices, otherwise use no-op context
    autocast_context = (
        torch.autocast(device_type=model_device.type)
        if model_device.type == "cuda"
        else contextlib.nullcontext()
    )

    for image_batch in tqdm(dataloader, disable=not show_progress):
        with torch.no_grad(), autocast_context:
            image_batch_tensor = image_batch.to(
                device=model_device,
                dtype=current_dtype,
            )

            image_features = model.encode_image(  # pyright: ignore[reportCallIssue]
                image_batch_tensor
            )
            image_embeddings = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            all_embeddings.append(image_embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings
