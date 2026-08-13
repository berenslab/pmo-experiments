"""
Module for running the full evaluation pipeline.

The full evaluation consists of the following steps:
1. Prepare the dataset and get embeddings for the specified model and dataset.
2. Run k-NN evaluation on the prepared data and save the results.
3. Run linear probing evaluation on the prepared data and save the results.
4. Run zero-shot evaluation on the prepared data and save the results.

Skips any of the above steps if the corresponding result files already exist to save
time when re-running the evaluation.
"""

import logging
import os
from typing import Literal, TypedDict

from pmo_experiments.evaluation import prepare_dataset_and_get_embeddings
from pmo_experiments.evaluation.knn import knn_evaluation_on_prepared_data
from pmo_experiments.evaluation.linear_probing import (
    linear_probing_evaluation_on_prepared_data,
)
from pmo_experiments.evaluation.zero_shot_prediction import (
    zero_shot_evaluation_on_prepared_data,
)

logger = logging.getLogger(__name__)


# Typed dict for the evaluation file paths.
class EvalFilePathsDict(TypedDict):
    """Return type for get_eval_file_paths function."""

    knn: str
    knn_distance_weighted: str
    linear_probing: list[str]
    linear_probing_balanced: list[str]
    zero_shot: str


def get_eval_file_paths(
    output_folder: str, dataset_name: str, lr_C_range: list[float]
) -> EvalFilePathsDict:
    """
    Get the file paths for the evaluation results.

    Args:
        output_folder (str): Folder to save the results to.
        dataset_name (str): Name of the dataset to evaluate on. Must be a registered
            dataset name in the DATASET_REGISTRY.
        lr_C_range (list[float]): List of inverse regularization strengths to try for
            linear probing.

    Returns:
        EvalFilePathsDict: Dictionary containing the file paths for the evaluation
            results.

    """
    knn_result_file = os.path.join(output_folder, f"{dataset_name}_knn_report.json")
    knn_distance_weighted_result_file = os.path.join(
        output_folder, f"{dataset_name}_knn_distance_weighted_report.json"
    )

    lp_output_folders = [
        os.path.join(output_folder, f"linear_probing_C-{lr_C}") for lr_C in lr_C_range
    ]
    lp_balanced_output_folders = [
        os.path.join(output_folder, f"linear_probing_C-{lr_C}_balanced")
        for lr_C in lr_C_range
    ]
    zero_shot_result_file = os.path.join(
        output_folder, f"{dataset_name}_zero_shot_report.json"
    )

    return {
        "knn": knn_result_file,
        "knn_distance_weighted": knn_distance_weighted_result_file,
        "linear_probing": lp_output_folders,
        "linear_probing_balanced": lp_balanced_output_folders,
        "zero_shot": zero_shot_result_file,
    }


def is_fully_evaluated(
    output_folder: str, dataset_name: str, force_rescore: bool, lr_C_range: list[float]
) -> bool:
    """
    Check whether the folder was already fully evaluated.

    Args:
        output_folder (str): Folder to save the results to.
        dataset_name (str): Name of the dataset to evaluate on. Must be a registered
            dataset name in the DATASET_REGISTRY.
        force_rescore (bool): If True, bypass the idempotency checks and recompute every
            step.
        lr_C_range (list[float]): List of inverse regularization strengths to try for
            linear probing.

    Raises:
        TypeError: Defensive check to ensure that the setting_paths are either a list or
            a string.

    Returns:
        bool: True if all evaluation results already exist and not force_rescore, False
            otherwise.

    """
    if force_rescore:
        return False

    eval_file_paths = get_eval_file_paths(output_folder, dataset_name, lr_C_range)

    for setting_paths in eval_file_paths.values():
        if isinstance(setting_paths, list):
            for path in setting_paths:
                if not os.path.exists(path):
                    return False
        elif isinstance(setting_paths, str):
            if not os.path.exists(setting_paths):
                return False
        else:
            raise TypeError(
                f"Unexpected type for setting_paths: {type(setting_paths)}. "
                f"Expected list or str."
            )

    return True


def run_full_evaluation(
    model_path: str,
    dataset_name: str,
    output_folder: str,
    dataset_args: dict | None = None,
    epoch: int | None = None,
    model_device: str = "cpu",
    batch_size: int = 64,
    dataloader_workers: int = 4,
    lr_solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs",
    lr_C_range: list[float] = [0.01, 0.1, 1.0, 10.0],
    lr_max_iter: int = 1000,
    knn_k: int = 20,
    train_split: str = "train",
    test_split: str = "test",
    force_rescore: bool = False,
    show_progress: bool = True,
):
    """
    Run the full evaluation pipeline.

    The pipeline includes:
    - kNN evaluation
    - linear probing
    - zero shot prediction

    Args:
        model_path (str): Path to the model to evaluate.
        dataset_name (str): Name of the dataset to evaluate on. Must be a registered
            dataset name in the DATASET_REGISTRY.
        output_folder (str): Folder to save the results to.
        dataset_args (dict | None, optional): Arguments for the dataset instance.
            Defaults to None.
        epoch (int | None, optional): Name of the epoch. None mean latest. Defaults to
            None.
        model_device (str, optional): Device to run on. Defaults to "cpu".
        batch_size (int, optional): Batch size. Defaults to 64.
        dataloader_workers (int, optional): Number of workers in the dataloader.
            Defaults to 4.
        lr_solver (str, optional): Solver for logistic regression. Defaults to 'lbfgs'.
        lr_C_range (list[float], optional): List of inverse regularization strengths
            to try for linear probing. Defaults to [0.01, 0.1, 1.0, 10.0].
        lr_max_iter (int, optional): Maximum iterations for logistic regression
            solver. Defaults to 1000.
        knn_k (int, optional): Number of nearest neighbors. Defaults to 20.
        train_split (str, optional): Name of the split used for training. Defaults to
            "train".
        test_split (str, optional): Name of the split used for testing. Defaults to
            "test".
        force_rescore (bool, optional): If True, bypass the idempotency checks and
            recompute every step (overwriting existing reports and re-writing the
            score ``.npz`` files). Used to backfill runs produced before raw scores
            were persisted. Defaults to False.
        show_progress (bool, optional): Whether to display tqdm progress bars for
            embedding extraction and text embedding. Defaults to True.

    """
    eval_file_paths = get_eval_file_paths(output_folder, dataset_name, lr_C_range)
    knn_result_file = eval_file_paths["knn"]
    knn_distance_weighted_result_file = eval_file_paths["knn_distance_weighted"]
    lp_output_folders = eval_file_paths["linear_probing"]
    lp_balanced_output_folders = eval_file_paths["linear_probing_balanced"]
    zero_shot_result_file = eval_file_paths["zero_shot"]

    if is_fully_evaluated(
        output_folder=output_folder,
        dataset_name=dataset_name,
        force_rescore=force_rescore,
        lr_C_range=lr_C_range,
    ):
        logger.debug(
            f"All results already exist for {model_path} on {dataset_name}"
            f" at epoch {epoch}, skipping evaluation."
        )
        return

    prepared_data = prepare_dataset_and_get_embeddings(
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_args=dataset_args,
        epoch=epoch,
        model_device=model_device,
        batch_size=batch_size,
        num_dataloader_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        return_model=True,
        train_split=train_split,
        test_split=test_split,
        show_progress=show_progress,
    )

    if force_rescore or not os.path.exists(knn_result_file):
        logger.debug(f"Running kNN on {os.path.basename(model_path)}...")
        knn_evaluation_on_prepared_data(
            prepared_data=prepared_data, output_path=knn_result_file, k=knn_k
        )

    if force_rescore or not os.path.exists(knn_distance_weighted_result_file):
        logger.debug(f"Running distance-weighted kNN {os.path.basename(model_path)}...")
        knn_evaluation_on_prepared_data(
            prepared_data=prepared_data,
            output_path=knn_distance_weighted_result_file,
            k=knn_k,
            knn_weights="distance",
        )

    for (
        lr_C,
        linear_probing_output_folder,
        linear_probing_balanced_output_folder,
    ) in zip(
        lr_C_range,
        lp_output_folders,
        lp_balanced_output_folders,
    ):
        if force_rescore or not os.path.exists(linear_probing_output_folder):
            logger.debug(f"Running linear probing {os.path.basename(model_path)}...")
            linear_probing_evaluation_on_prepared_data(
                prepared_data=prepared_data,
                output_path=linear_probing_output_folder,
                lr_solver=lr_solver,
                lr_C=lr_C,
                lr_max_iter=lr_max_iter,
                balanced=False,
            )

        if force_rescore or not os.path.exists(linear_probing_balanced_output_folder):
            logger.debug(
                f"Running balanced linear probing {os.path.basename(model_path)}..."
            )
            linear_probing_evaluation_on_prepared_data(
                prepared_data=prepared_data,
                output_path=linear_probing_balanced_output_folder,
                lr_solver=lr_solver,
                lr_C=lr_C,
                lr_max_iter=lr_max_iter,
                balanced=True,
            )

    if force_rescore or not os.path.exists(zero_shot_result_file):
        logger.debug(f"Running zero-shot evaluation {os.path.basename(model_path)}...")
        zero_shot_evaluation_on_prepared_data(
            prepared_data=prepared_data,
            output_file_path=zero_shot_result_file,
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            show_progress=show_progress,
        )
