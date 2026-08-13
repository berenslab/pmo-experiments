"""kNN evaluation utilities for pmo-experiments."""

import json
import logging
import os
from typing import Any, Literal, Mapping

from sklearn.neighbors import KNeighborsClassifier

from pmo_experiments.evaluation import (
    ColumnScores,
    PreparedData,
    get_scores,
    prepare_dataset_and_get_embeddings,
    save_score_arrays,
    to_full_width_scores,
)

logger = logging.getLogger(__name__)


def knn_evaluation_on_prepared_data(
    prepared_data: PreparedData,
    output_path: str,
    k: int = 20,
    knn_weights: Literal["uniform", "distance"] | None = None,
):
    """
    Run kNN classification on the prepared data.

    Args:
        prepared_data (PreparedData): Data prepared by
            prepare_dataset_and_get_embeddings function. Has to contain train and test
            embeddings!
        output_path (str): Json file path to write the kNN evaluation report to.
        k (int, optional): Number of nearest neighbors. Defaults to 20.
        knn_weights (Literal['uniform', 'distance'] | None, optional): Weighting
            strategy for kNN. 'uniform' means uniform weights, 'distance' means
            weight points by the inverse of their distance. If None, defaults to
            sklearn default. Defaults to None.

    """
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

    logger.debug("Running KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=k, weights=knn_weights)  # pyright: ignore[reportArgumentType]
    knn.fit(train_embeddings.numpy(), train_targets.to_numpy())
    test_proba_list = knn.predict_proba(test_embeddings.numpy())

    # Single-output KNN returns an array; wrap for uniform handling
    if not isinstance(test_proba_list, list):
        test_proba_list = [test_proba_list]

    # ``classes_`` mirrors the proba structure: a list per output for multi-output,
    # a single array otherwise.
    knn_classes = knn.classes_
    if not isinstance(knn_classes, list):
        knn_classes = [knn_classes]

    knn_report: dict = {}
    column_scores: dict[str, ColumnScores] = {}
    for i, col in enumerate(target_columns):
        labels = list(range(len(target_value_lists[col])))
        # predict_proba only emits columns for classes seen in training; widen to the
        # full label set so AUROC/AUPRC are computable.
        y_true = test_targets[col].to_numpy()
        y_scores = to_full_width_scores(test_proba_list[i], knn_classes[i], labels)
        knn_report[col] = get_scores(
            y_true=y_true,
            y_scores=y_scores,
            labels=labels,
            class_names=target_value_lists[col],
            include_confusion_matrix=True,
        )
        column_scores[col] = ColumnScores(
            y_true=y_true,
            y_scores=y_scores,
            class_names=target_value_lists[col],
            full_length_class_names=None,
        )

    knn_report["k"] = k

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(knn_report, f, indent=4, ensure_ascii=False)

    # Persist raw scores so metrics can be recomputed without re-extracting
    # embeddings (e.g. after a scoring-function change).
    scores_path = f"{os.path.splitext(output_path)[0]}_scores.npz"
    save_score_arrays(scores_path, column_scores)


def knn_evaluation(
    model_path: str,
    dataset_name: str,
    output_path: str,
    dataset_args: Mapping[str, Any] | None = None,
    epoch: int | None = None,
    model_device: str = "cpu",
    k: int = 20,
    train_split: str = "train",
    test_split: str = "test",
    batch_size: int = 128,
    num_dataloader_workers: int = 4,
    knn_weights: Literal["uniform", "distance"] | None = None,
):
    """
    Run kNN evaluation on image embeddings extracted from a dataset split.

    Args:
        model_path (str): Path to the model checkpoint to load.
        dataset_name (str): Dataset registry name.
        output_path (str): Path to write the evaluation JSON report.
        dataset_args (Mapping[str, Any] | None): Additional dataset
            initialization arguments.
        epoch (int | None): Checkpoint epoch to load.
        model_device (str): Device to run inference on.
        k (int): Number of neighbors for kNN.
        train_split (str): Dataset split used to fit kNN.
        test_split (str): Dataset split used to evaluate kNN.
        batch_size (int): Batch size for embedding extraction.
        num_dataloader_workers (int): DataLoader workers.
        knn_weights (Literal['uniform', 'distance'] | None, optional): Weighting
            strategy for kNN. 'uniform' means uniform weights, 'distance' means
            weight points by the inverse of their distance. If None, defaults to
            sklearn default. Defaults to None.

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
        num_dataloader_workers=num_dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    knn_evaluation_on_prepared_data(
        prepared_data=prepared_data,
        output_path=output_path,
        k=k,
        knn_weights=knn_weights,
    )
