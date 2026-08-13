"""
Linear probing evaluation for vision models.

This module provides utilities for training linear classifiers on top of frozen
vision model embeddings and evaluating their performance. It supports extracting
embeddings from a model, training logistic regression classifiers, and generating
classification reports and confusion matrices.
"""

import json
import logging
import os
from typing import Literal

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from pmo_experiments.evaluation import (
    ColumnScores,
    LinearProbingModelParameters,
    LinearProbingScores,
    PreparedData,
    get_scores,
    prepare_dataset_and_get_embeddings,
    save_score_arrays,
    to_full_width_scores,
)

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "saga",
    C: float = 1.0,
    max_iter: int = 1000,
    balanced: bool = False,
    l1_ratio: float = 0.0,
) -> tuple[LogisticRegression | DummyClassifier, np.ndarray]:
    """
    Train a logistic regression classifier on embeddings and evaluate on validation set.

    Args:
        X_train (np.ndarray): Training embeddings.
        y_train (list[int]): Training target indices.
        X_val (np.ndarray): Validation embeddings.
        solver (str, optional): Solver algorithm. 'saga' is recommended for large
            datasets and high-dimensional features. Defaults to 'saga'.
        C (float, optional): Inverse of regularization strength; smaller values
            specify stronger regularization. Defaults to 1.0.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        balanced (bool, optional): Whether to use balanced class weights. Defaults to
            False.
        l1_ratio (float, optional): The Elastic-Net mixing parameter, with
            0 <= l1_ratio <= 1. Only used if solver='saga'. Defaults to 0.

    Returns:
        tuple[LogisticRegression | DummyClassifier, np.ndarray]: Tuple containing:
            - Trained model
            - Prediction score matrix of shape ``(N, n_classes)``

    """
    if solver not in ["saga"] and l1_ratio != 0.0:
        logger.warning(
            "l1_ratio is only applicable for 'saga' solver. "
            f"Ignoring l1_ratio={l1_ratio} since solver={solver}."
        )
        l1_ratio = 0.0

    if np.unique(y_train).size < 2:
        # Create a dummy model that always predicts the single class in y_train
        model = DummyClassifier(strategy="constant", constant=y_train[0])
        model.fit(X_train, y_train)
    else:
        model = LogisticRegression(
            solver=solver,
            C=C,
            max_iter=max_iter,
            random_state=106080954,
            class_weight="balanced" if balanced else None,
            l1_ratio=l1_ratio,
            tol=1e-4 if solver != "saga" else 1e-3,
        )
        model.fit(X_train, y_train)

    y_scores: np.ndarray = model.predict_proba(X_val)  # pyright: ignore[reportAssignmentType]

    return model, y_scores


def linear_probing_on_target(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    target_values: list[str],
    lr_solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "saga",
    lr_C: float = 1.0,
    lr_max_iter: int = 1000,
    lr_l1_ratio: float = 0.0,
    balanced: bool = False,
) -> tuple[LogisticRegression | DummyClassifier, LinearProbingScores, np.ndarray]:
    """
    Perform linear probing on a specific target column.

    Extracts embeddings from train and validation sets, trains a linear classifier,
    and generates evaluation metrics.

    Args:
        train_embeddings (torch.Tensor): Tensor of training embeddings.
        val_embeddings (torch.Tensor): Tensor of validation embeddings.
        train_targets (pd.Series): Series of training target values.
        val_targets (pd.Series): Series of validation target values.
        target_values (list[str]): List of all possible target values for the column.
        lr_solver (str, optional): Logistic regression solver. Defaults to 'saga'.
        lr_C (float, optional): Inverse regularization strength. Defaults to 1.0.
        lr_max_iter (int, optional): Maximum iterations for solver. Defaults to 1000.
        lr_l1_ratio (float, optional): The Elastic-Net mixing parameter, with
            0 <= lr_l1_ratio <= 1. Only used if solver='saga'. Defaults to 0.
        balanced (bool, optional): Whether to use balanced class weights. Defaults to
            False.

    Returns:
        tuple[LogisticRegression | DummyClassifier, dict, np.ndarray]: Tuple
            containing:
            - Trained linear model
            - Linear probing report with classification metrics and confusion matrix
            - Full-width prediction score matrix of shape ``(N, len(target_values))``

    """
    all_target_idx = list(range(len(target_values)))
    linear_model, y_scores = train_logistic_regression(
        X_train=train_embeddings.numpy(),
        y_train=train_targets,
        X_val=val_embeddings.numpy(),
        solver=lr_solver,
        C=lr_C,
        max_iter=lr_max_iter,
        balanced=balanced,
        l1_ratio=lr_l1_ratio,
    )
    # predict_proba only emits columns for classes seen in training; widen to the
    # full label set so AUROC/AUPRC are computable.
    y_scores = to_full_width_scores(y_scores, linear_model.classes_, all_target_idx)
    scores = get_scores(
        y_true=val_targets,
        y_scores=y_scores,
        labels=all_target_idx,
        class_names=target_values,
        include_confusion_matrix=True,
    )

    linear_probing_report = LinearProbingScores(
        **scores,
        balanced_accuracy=scores["classification_report"]["macro avg"]["recall"],
        used_dummy_classifier=isinstance(linear_model, DummyClassifier),
        constant_prediction=target_values[int(linear_model.constant)]  # pyright: ignore[reportArgumentType,reportAttributeAccessIssue]
        if isinstance(linear_model, DummyClassifier)
        else None,
        did_convergence=bool(linear_model.n_iter_ < lr_max_iter)
        if isinstance(linear_model, LogisticRegression)
        else None,
        model_parameters=LinearProbingModelParameters(
            solver=lr_solver,
            C=lr_C,
            max_iter=lr_max_iter,
            l1_ratio=lr_l1_ratio,
            balanced=balanced,
        ),
    )

    return linear_model, linear_probing_report, y_scores


def linear_probing_evaluation_on_prepared_data(
    prepared_data: PreparedData,
    output_path: str,
    lr_solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ]
    | None = "saga",
    lr_C: float = 1.0,
    lr_max_iter: int = 1000,
    lr_l1_ratio: float = 0.0,
    balanced: bool = False,
):
    """
    Run linear probing on the prepared data.

    Args:
        prepared_data (PreparedData): Data to evaluate, including embeddings, targets,
            and target value lists. Must have train and test embeddings!
        output_path (str): Folder to save the linear probing reports and model weights
            to.
        lr_solver (str, optional): Logistic regression solver. Defaults to 'saga'.
        lr_C (float, optional): Inverse regularization strength. Defaults to 1.0.
        lr_max_iter (int, optional): Maximum iterations for solver. Defaults to 1000.
        lr_l1_ratio (float, optional): The Elastic-Net mixing parameter, with
            0 <= lr_l1_ratio <= 1. Only used if solver='saga'. Defaults to 0.
        balanced (bool, optional): Whether to use balanced class weights. Defaults to
            False.

    """
    if lr_solver is None:
        lr_solver = "saga"

    # Should never be None
    assert prepared_data["train_embeddings"] is not None, (
        "Train embeddings are required for kNN evaluation"
    )
    assert prepared_data["train_targets"] is not None, (
        "Train targets are required for kNN evaluation"
    )
    assert prepared_data["test_embeddings"] is not None, (
        "Test embeddings are required for kNN evaluation"
    )
    assert prepared_data["test_targets"] is not None, (
        "Test targets are required for kNN evaluation"
    )

    target_columns = list(prepared_data["target_value_lists"].keys())
    target_value_lists = prepared_data["target_value_lists"]
    train_embeddings = prepared_data["train_embeddings"]
    train_targets = prepared_data["train_targets"]
    test_embeddings = prepared_data["test_embeddings"]
    test_targets = prepared_data["test_targets"]

    for target_column in target_columns:
        val_targets = test_targets[target_column].to_numpy()
        linear_model, linear_probing_report, y_scores = linear_probing_on_target(
            train_embeddings=train_embeddings,
            val_embeddings=test_embeddings,
            train_targets=train_targets[target_column].to_numpy(),
            val_targets=val_targets,
            target_values=target_value_lists[target_column],
            lr_solver=lr_solver,
            lr_C=lr_C,
            lr_max_iter=lr_max_iter,
            lr_l1_ratio=lr_l1_ratio,
            balanced=balanced,
        )

        os.makedirs(output_path, exist_ok=True)

        model_file_save_path = os.path.join(
            output_path, f"{target_column}_linear_model.npz"
        )

        if isinstance(linear_model, LogisticRegression):
            np.savez(
                model_file_save_path,
                coef=linear_model.coef_,
                intercept=linear_model.intercept_,
            )

        report_path = os.path.join(output_path, f"{target_column}_report.json")
        with open(report_path, "w") as f:
            json.dump(linear_probing_report, f, indent=4, ensure_ascii=False)

        # Persist raw scores so metrics can be recomputed without re-extracting
        # embeddings (e.g. after a scoring-function change).
        scores_path = f"{os.path.splitext(report_path)[0]}_scores.npz"
        save_score_arrays(
            scores_path,
            {
                target_column: ColumnScores(
                    y_true=val_targets,
                    y_scores=y_scores,
                    class_names=target_value_lists[target_column],
                    full_length_class_names=None,
                )
            },
        )


def linear_probing_evaluation(
    model_path: str,
    dataset_name: str,
    output_path: str,
    train_split: str = "train",
    test_split: str = "test",
    epoch: int | None = None,
    dataset_args: dict | None = None,
    model_device: str = "cpu",
    batch_size: int = 64,
    dataloader_workers: int = 4,
    lr_solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "saga",
    lr_C: float = 1.0,
    lr_max_iter: int = 1000,
    lr_l1_ratio: float = 0.0,
    balanced: bool = False,
):
    """
    Run linear probing evaluation on a dataset using a trained vision model.

    Loads a model checkpoint, extracts embeddings from specified train and validation
    splits, trains linear classifiers for each target column, and saves the results.

    Args:
        model_path (str): Path to the model checkpoint folder.
        dataset_name (str): Name of the registered dataset to evaluate on.
        output_path (str): Directory to save linear model weights and reports.
        train_split (str, optional): Split to use for training
            ('train', 'val', or 'test'). Defaults to 'train'.
        test_split (str, optional): Split to use for validation
            ('train', 'val', or 'test'). Defaults to 'test'.
        epoch (int | None, optional): Specific epoch checkpoint to load. If None,
            loads the latest. Defaults to None.
        dataset_args (dict | None, optional): Additional arguments to pass to the
            dataset constructor. Defaults to None.
        model_device (str, optional): Device to load the model on. Defaults to 'cpu'.
        batch_size (int, optional): Batch size for dataloaders. Defaults to 64.
        dataloader_workers (int, optional): Number of workers for dataloaders.
            Defaults to 4.
        lr_solver (str, optional): Solver for logistic regression. 'saga' is
            recommended for large datasets. Defaults to 'saga'.
        lr_C (float, optional): Inverse regularization strength for logistic
            regression. Defaults to 1.0.
        lr_max_iter (int, optional): Maximum iterations for logistic regression
            solver. Defaults to 1000.
        lr_l1_ratio (float, optional): The Elastic-Net mixing parameter, with
            0 <= lr_l1_ratio <= 1. Only used if solver='saga'. Defaults to 0.
        balanced (bool, optional): Whether to use balanced class weights. Defaults to
            False.

    """
    prepared_data = prepare_dataset_and_get_embeddings(
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_args=dataset_args,
        epoch=epoch,
        model_device=model_device,
        train_split=train_split,
        test_split=test_split,
        batch_size=batch_size,
        num_dataloader_workers=dataloader_workers,
    )

    linear_probing_evaluation_on_prepared_data(
        prepared_data=prepared_data,
        output_path=output_path,
        lr_solver=lr_solver,
        lr_C=lr_C,
        lr_max_iter=lr_max_iter,
        lr_l1_ratio=lr_l1_ratio,
        balanced=balanced,
    )
