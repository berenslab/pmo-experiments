"""
Module for zero-shot prediction and evaluation.

This module evaluates a CLIP-style model in a *zero-shot* fashion by treating
each unique label in a target column as a text prompt (class name). It encodes
all class texts once, then encodes images in batches, and assigns the class with
highest cosine similarity.

The main entry point is :func:`evaluate_model`, which loads a dataset via
``DATASET_REGISTRY``, selects a split CSV (train/val/test), runs prediction, and
writes metrics (JSON) and confusion matrices (CSV/optional PNG) to disk.

Note:
    This implementation uses ``df[target_column].unique()`` to determine the set
    and order of classes. That order defines the columns in the returned score
    matrix and the saved confusion matrix axes.

"""

import json
import os

import numpy as np
import open_clip
import torch
from tqdm.auto import tqdm

from pmo_experiments.evaluation import (
    ColumnScores,
    PreparedData,
    ScoresResult,
    get_scores,
    prepare_dataset_and_get_embeddings,
    save_score_arrays,
)


def embed_texts(  # noqa: D417
    model: torch.nn.Module,
    tokenizer: (
        open_clip.tokenizer.SimpleTokenizer
        | open_clip.tokenizer.HFTokenizer
        | open_clip.tokenizer.SigLipTokenizer
    ),
    target_names: list[str],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Embed the target texts using CLIP.

    Args:
        model (torch.nn.Module): CLIP model.
        tokenizer (open_clip.tokenizer.SimpleTokenizer |
            open_clip.tokenizer.HFTokenizer | open_clip.tokenizer.SigLipTokenizer):
            Tokenizer corresponding to the CLIP model.
        target_names (list[str]): Texts to embed (e.g. class names).
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of dataloader workers. Defaults to 4.
        pin_memory (bool, optional): Pin the dataloader worker memory. Defaults to True.
        persistent_workers (bool, optional): Use persistent dataloader workers.
            Defaults to True.
        prefetch_factor (int, optional): Prefetch factor. Defaults to 2.
        show_progress (bool, optional): Whether to display a tqdm progress bar.
            Defaults to True.

    Returns:
        torch.Tensor: Embedded target texts as a tensor of shape (num_targets,
            embedding_dim).

    """
    # Tokenize target names
    text_tokens = tokenizer(
        target_names,
    )

    # Create dataloader for target names
    target_dataset = torch.utils.data.TensorDataset(text_tokens)

    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model_device = next(model.parameters()).device

    all_text_embeddings = []

    for batch in tqdm(
        target_loader, desc="Embedding target texts", disable=not show_progress
    ):
        with torch.no_grad():
            batch_tokens = batch[0].to(model_device)
            text_features = model.encode_text(batch_tokens)  # pyright: ignore[reportCallIssue]
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_embeddings.append(text_embeddings.cpu())

    if len(all_text_embeddings) > 0:
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    else:
        all_text_embeddings = torch.empty((0,))
    return all_text_embeddings


def calculate_stats(
    prediction_scores: np.ndarray,
    targets: np.ndarray,
    target_values: list[str],
    full_length_target_names: list[str] | None = None,
) -> ScoresResult:
    """
    Compute classification metrics from prediction scores.

    Args:
        prediction_scores (np.ndarray): Array of shape ``(N, C)`` with per-class
            scores.
        targets (np.ndarray): Array of shape ``(N,)`` with integer class indices.
        target_values (list[str]): List of all possible target class names, ordered by
            their corresponding indices.
        full_length_target_names (list[str] | None, optional): Optional list of full
            length target names added as ``"full_length_text"`` to per-class entries.
            Defaults to None.

    Returns:
        dict: See :func:`pmo_experiments.evaluation.get_scores`.

    """
    labels = list(range(len(target_values)))
    return get_scores(
        y_true=targets,
        y_scores=prediction_scores,
        labels=labels,
        class_names=target_values,
        full_length_class_names=full_length_target_names,
        include_confusion_matrix=True,
    )


def zero_shot_evaluation_on_prepared_data(
    prepared_data: PreparedData,
    output_file_path: str,
    batch_size: int = 64,
    dataloader_workers: int = 4,
    show_progress: bool = True,
):
    """
    Perform zero-shot prediction on an instance of prepared data.

    Args:
        prepared_data (PreparedData): Data prepared by
            ``prepare_dataset_and_get_embeddings`` containing embeddings, targets,
            target mapping, dataset and the loaded model.
        output_file_path (str): Json file to save evaluation statistics.
        batch_size (int, optional): Batch size for the forward pass of the texts.
            Defaults to 64.
        dataloader_workers (int, optional): Number of workers in the dataloader.
            Defaults to 4.
        show_progress (bool, optional): Whether to display a tqdm progress bar while
            embedding the target texts. Defaults to True.

    """
    # Should never be None since test_split is specified
    # and return_model=True
    assert prepared_data["test_embeddings"] is not None, (
        "Test embeddings are required for kNN evaluation"
    )
    assert prepared_data["test_targets"] is not None, (
        "Test targets are required for kNN evaluation"
    )
    assert prepared_data["loaded_model"] is not None, (
        "Loaded model is required for embedding target texts"
    )
    assert prepared_data["tokenizer"] is not None, (
        "Tokenizer is required for embedding target texts"
    )

    # embed all of the target values
    dataset = prepared_data["dataset"]
    target_value_lists = prepared_data["target_value_lists"]

    all_target_texts = []

    for col, targets in target_value_lists.items():
        for t in targets:
            all_target_texts.append(dataset.get_value_verbalization(col, t))

    target_embeddings = embed_texts(
        model=prepared_data["loaded_model"],
        tokenizer=prepared_data["tokenizer"],
        target_names=all_target_texts,
        batch_size=batch_size,
        num_workers=dataloader_workers,
        show_progress=show_progress,
    )

    idx = 0
    all_stats = {}
    column_scores: dict[str, ColumnScores] = {}
    for col, targets in target_value_lists.items():
        num_targets = len(targets)
        current_embeddings = target_embeddings[idx : idx + num_targets]
        idx += num_targets

        full_length_target_names = [
            dataset.get_value_verbalization(col, t) for t in targets
        ]

        # Do the prediction and evaluation for this target column
        cosine_similarities = prepared_data["test_embeddings"] @ current_embeddings.t()
        prediction_scores = cosine_similarities.cpu().numpy()
        targets_array = prepared_data["test_targets"][col].to_numpy()
        stats = calculate_stats(
            prediction_scores=prediction_scores,
            targets=targets_array,
            target_values=targets,
            full_length_target_names=full_length_target_names,
        )
        all_stats[col] = stats
        column_scores[col] = ColumnScores(
            y_true=targets_array,
            y_scores=prediction_scores,
            class_names=targets,
            full_length_class_names=full_length_target_names,
        )

    # Save the stats to disk
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)

    # Persist raw scores so metrics can be recomputed without re-extracting
    # embeddings (e.g. after a scoring-function change).
    scores_path = f"{os.path.splitext(output_file_path)[0]}_scores.npz"
    save_score_arrays(scores_path, column_scores)


def zero_shot_evaluation(
    dataset_name: str,
    model_path: str,
    output_file_path: str,
    dataset_kwargs: dict | None = None,
    batch_size: int = 64,
    dataloader_workers: int = 4,
    epoch: int | None = None,
    model_device: str = "cpu",
    pin_memory: bool = True,
    test_split: str = "test",
):
    """
    Evaluate a model checkpoint with zero-shot prediction on a dataset split.

    For each target column reported by the dataset, this function runs
    :func:`predict_df`, computes metrics via :func:`calculate_stats`, and writes
    results to ``output_folder``.

    Args:
        dataset_name (str): Name/key of the dataset in ``DATASET_REGISTRY``.
        model_path (str): Path to the model/checkpoint folder.
        output_file_path (str): File to save evaluation artifacts.
        dataset_kwargs (dict | None, optional): Optional kwargs forwarded to the
            dataset constructor. Defaults to None.
        batch_size (int, optional): Batch size used during inference.
            Defaults to 64.
        dataloader_workers (int, optional): Number of dataloader worker processes.
            Defaults to 4.
        epoch (int | None, optional): Number of the epoch to use. If None use the
            latest. Defaults to None.
        model_device (str, optional): Device for inference.
            Defaults to "cpu".
        pin_memory (bool, optional): Whether to enable dataloader pin-memory.
            Defaults to True.
        test_split (str, optional): Which dataset split to evaluate ("train", "val",
            or "test"). Defaults to "test".

    """
    assert output_file_path.endswith(".json"), "Output file path must end with .json"
    prepared_data = prepare_dataset_and_get_embeddings(
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_args=dataset_kwargs,
        epoch=epoch,
        model_device=model_device,
        train_split=None,
        test_split=test_split,
        batch_size=batch_size,
        num_dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        return_model=True,
    )

    zero_shot_evaluation_on_prepared_data(
        prepared_data=prepared_data,
        output_file_path=output_file_path,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
    )
