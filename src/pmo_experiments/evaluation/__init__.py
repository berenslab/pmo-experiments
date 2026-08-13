"""Module defining evaluation utilities and functions."""

import os
from typing import Any, Mapping, TypedDict

import numpy as np
import open_clip
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset
from pmo_experiments.util.dataloader import get_dataloader_from_df
from pmo_experiments.util.model_interface import (
    extract_image_embeddings,
    load_checkpoint,
)


class ScoresResult(TypedDict):
    """Typed dictionary for :func:`get_scores` output."""

    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]] | None
    auroc: float | None
    auc: float | None
    auprc: float | None
    aupr: float | None


class LinearProbingModelParameters(TypedDict):
    """Typed dictionary for linear probing model parameters."""

    solver: str
    C: float
    max_iter: int
    l1_ratio: float
    balanced: bool


class LinearProbingScores(ScoresResult):
    """Typed dictionary for linear probing evaluation output."""

    balanced_accuracy: float
    used_dummy_classifier: bool
    constant_prediction: str | None
    did_convergence: bool | None
    model_parameters: LinearProbingModelParameters


class ColumnScores(TypedDict):
    """Per-target-column inputs to :func:`get_scores`, as persisted to disk."""

    y_true: np.ndarray
    y_scores: np.ndarray
    class_names: list[str]
    full_length_class_names: list[str] | None


def contrast_scores(y_scores: np.ndarray) -> np.ndarray:
    """
    Convert per-class scores into one-vs-rest contrast scores.

    Column ``c`` becomes ``y_scores[:, c]`` minus the mean of the competing columns, so
    for a binary column this is exactly ``y_scores[:, 1] - y_scores[:, 0]``.

    This matters for ranking metrics (AUROC/AUPRC) on scores that are **not** normalised
    per row. Zero-shot prediction scores are raw cosine similarities: their magnitude is
    dominated by how generically similar an image is to any prompt, so a single column
    carries almost no class information — the discriminative signal is the contrast
    between the class prompt and its competitors. Ranking by one raw column instead
    makes the resulting AUROC effectively noise.

    Scores that already sum to 1 per row (``predict_proba`` output, as used by kNN and
    linear probing) are returned untouched: a single column is already a valid ranking
    score there, and the contrast would only be a monotone rescaling of it. Returning
    them unchanged keeps those metrics bit-identical. Applying the transform anyway
    would be *almost* harmless but not quite -- row sums are only 1 to within
    floating-point error, so subtracting a row-dependent quantity perturbs otherwise
    exactly-tied scores and silently reorders ties (measured: up to 6e-3 AUROC drift on
    kNN's discrete probabilities).

    Args:
        y_scores (np.ndarray): Array of shape ``(N, C)`` with per-class scores.

    Returns:
        np.ndarray: Array of shape ``(N, C)`` with one-vs-rest contrast scores, or
            ``y_scores`` unchanged if it is already normalised per row.

    """
    n_classes = y_scores.shape[1]
    if n_classes < 2:
        return y_scores

    totals = y_scores.sum(axis=1, keepdims=True)
    if np.allclose(totals, 1.0, rtol=0.0, atol=1e-6):
        return y_scores

    # mean of the competing columns == (row total - this column) / (C - 1)
    return y_scores - (totals - y_scores) / (n_classes - 1)


def save_score_arrays(
    output_path: str, column_scores: Mapping[str, ColumnScores]
) -> None:
    """
    Persist the raw ``get_scores`` inputs for every target column to a ``.npz``.

    Storing ``y_true``/``y_scores``/class names lets the metrics be recomputed later
    (e.g. after a scoring-function change) without re-extracting embeddings. Columns
    are indexed positionally to avoid delimiter collisions in column names.

    Args:
        output_path (str): Destination ``.npz`` path.
        column_scores (Mapping[str, ColumnScores]): Mapping of target column name to
            its scoring inputs.

    """
    arrays: dict[str, np.ndarray] = {}
    columns = list(column_scores.keys())
    arrays["columns"] = np.array(columns)
    for i, col in enumerate(columns):
        entry = column_scores[col]
        arrays[f"y_true_{i}"] = np.asarray(entry["y_true"])
        arrays[f"y_scores_{i}"] = np.asarray(entry["y_scores"])
        arrays[f"class_names_{i}"] = np.array(entry["class_names"])
        full_length = entry.get("full_length_class_names")
        if full_length is not None:
            arrays[f"full_length_class_names_{i}"] = np.array(full_length)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays)  # pyright: ignore[reportArgumentType]


def load_score_arrays(npz_path: str) -> dict[str, ColumnScores]:
    """
    Load score arrays previously written by :func:`save_score_arrays`.

    Args:
        npz_path (str): Path to the ``.npz`` file.

    Returns:
        dict[str, ColumnScores]: Mapping of target column name to its scoring inputs.

    """
    data = np.load(npz_path, allow_pickle=False)
    columns = [str(c) for c in data["columns"]]
    result: dict[str, ColumnScores] = {}
    for i, col in enumerate(columns):
        full_length_key = f"full_length_class_names_{i}"
        result[col] = ColumnScores(
            y_true=data[f"y_true_{i}"],
            y_scores=data[f"y_scores_{i}"],
            class_names=[str(c) for c in data[f"class_names_{i}"]],
            full_length_class_names=(
                [str(c) for c in data[full_length_key]]
                if full_length_key in data.files
                else None
            ),
        )
    return result


def to_full_width_scores(
    proba: np.ndarray,
    model_classes: np.ndarray | list[int],
    labels: list[int],
) -> np.ndarray:
    """
    Reindex a ``predict_proba`` matrix to the full ``(N, len(labels))`` width.

    ``predict_proba`` from scikit-learn classifiers only emits columns for the
    classes seen during training (``model.classes_``). When the training split did
    not contain every class, the resulting matrix is narrower than the full label
    set, which breaks downstream metrics. This fills the missing-class columns with
    zeros so the score matrix always aligns with ``labels``.

    Args:
        proba (np.ndarray): Score matrix of shape ``(N, len(model_classes))``.
        model_classes (np.ndarray | list[int]): Class indices corresponding to the
            columns of ``proba`` (i.e. ``model.classes_``).
        labels (list[int]): Ordered full list of class indices to align to.

    Returns:
        np.ndarray: Score matrix of shape ``(N, len(labels))`` aligned to ``labels``.

    """
    proba = np.asarray(proba)
    full = np.zeros((proba.shape[0], len(labels)), dtype=proba.dtype)
    label_to_col = {label: col for col, label in enumerate(labels)}
    for src_col, cls in enumerate(model_classes):
        dst_col = label_to_col.get(int(cls))
        if dst_col is not None:
            full[:, dst_col] = proba[:, src_col]
    return full


class PreparedData(TypedDict):
    """Typed dictionary for data preparation output."""

    train_embeddings: torch.Tensor | None
    train_targets: pd.DataFrame | None
    test_embeddings: torch.Tensor | None
    test_targets: pd.DataFrame | None
    target_value_lists: dict[str, list[str]]
    dataset: BaseDataset
    loaded_model: torch.nn.Module | None
    tokenizer: (
        open_clip.tokenizer.SimpleTokenizer
        | open_clip.tokenizer.HFTokenizer
        | open_clip.tokenizer.SigLipTokenizer
        | None
    )
    train_image_paths: list[str] | None
    val_image_paths: list[str] | None
    test_image_paths: list[str] | None
    val_targets: pd.DataFrame | None
    val_embeddings: torch.Tensor | None


def get_scores(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: list[int],
    class_names: list[str],
    full_length_class_names: list[str] | None = None,
    include_confusion_matrix: bool = True,
) -> ScoresResult:
    """
    Compute classification metrics from prediction scores.

    Args:
        y_true (np.ndarray): Integer class labels of shape ``(N,)``.
        y_scores (np.ndarray): Per-class scores of shape ``(N, C)``.
        labels (list[int]): Ordered list of class indices to evaluate.
        class_names (list[str]): Human-readable name for each entry in ``labels``.
        full_length_class_names (list[str] | None, optional): Optional longer names
            added as ``"full_length_text"`` to per-class report entries. Defaults to
            None.
        include_confusion_matrix (bool, optional): Whether to include the confusion
            matrix in the output. Defaults to True.

    Returns:
        ScoresResult: Keys: ``classification_report``, ``confusion_matrix`` (or None),
            ``auroc``, ``auc``, ``auprc``, ``aupr``.

    """
    y_pred = np.argmax(y_scores, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,  # pyright: ignore[reportArgumentType]
        output_dict=True,
    )
    assert isinstance(report, dict)

    transformed_report: dict = {}
    for key, value in report.items():
        if key in ["accuracy", "macro avg", "weighted avg"]:
            transformed_report[key] = value
        else:
            try:
                class_idx = int(key)
                entry = {**value, "text": class_names[class_idx]}
                if full_length_class_names is not None:
                    entry["full_length_text"] = full_length_class_names[class_idx]
                transformed_report[key] = entry
            except ValueError:
                transformed_report[key] = value

    conf_matrix = (
        confusion_matrix(y_true, y_pred, labels=labels).tolist()
        if include_confusion_matrix
        else None
    )

    n_classes = len(labels)
    unique_classes = np.unique(y_true)

    auroc: float | None = None
    auc: float | None = None
    auprc: float | None = None
    aupr: float | None = None

    # AUROC/AUPRC are only defined when at least two classes are present in y_true.
    # In that case we always attempt the computation; any failure now propagates
    # instead of being silently turned into None.
    if len(unique_classes) >= 2:
        # Rank on one-vs-rest contrasts rather than a single raw score column. Required
        # for un-normalised scores such as zero-shot cosine similarities; a no-op for
        # predict_proba output. See :func:`contrast_scores`.
        y_ranking = contrast_scores(y_scores)

        if n_classes == 2:
            _auroc = float(roc_auc_score(y_true, y_ranking[:, 1]))
            auroc = _auroc
            auc = _auroc

            _auprc = float(average_precision_score(y_true, y_ranking[:, 1]))
            auprc = _auprc
            aupr = _auprc
        else:
            # One-vs-rest, computed per present class. This mirrors the
            # average_precision_score multilabel route and, unlike multi_class="ovr",
            # does not require the per-row scores to sum to 1 (e.g. cosine
            # similarities). Classes with a single label present are undefined and
            # skipped.
            y_true_bin = label_binarize(y_true, classes=labels)
            supports = y_true_bin.sum(axis=0)

            per_class_auroc: list[float] = []
            per_class_auprc: list[float] = []
            per_class_support: list[int] = []
            for col in range(n_classes):
                column_classes = np.unique(y_true_bin[:, col])
                if len(column_classes) < 2:
                    # Only positives or only negatives for this class -> undefined.
                    continue
                per_class_auroc.append(
                    float(roc_auc_score(y_true_bin[:, col], y_ranking[:, col]))
                )
                per_class_auprc.append(
                    float(
                        average_precision_score(y_true_bin[:, col], y_ranking[:, col])
                    )
                )
                per_class_support.append(int(supports[col]))

            if len(per_class_auroc) > 0:
                weights = np.asarray(per_class_support, dtype=float)
                weight_sum = weights.sum()
                auroc = float(np.mean(per_class_auroc))
                auprc = float(np.mean(per_class_auprc))
                if weight_sum > 0:
                    auc = float(np.average(per_class_auroc, weights=weights))
                    aupr = float(np.average(per_class_auprc, weights=weights))
                else:
                    auc = auroc
                    aupr = auprc

    return ScoresResult(
        classification_report=transformed_report,
        confusion_matrix=conf_matrix,
        auroc=auroc,
        auc=auc,
        auprc=auprc,
        aupr=aupr,
    )


def prepare_dataset_and_get_embeddings(
    model_path: str,
    dataset_name: str,
    dataset_args: Mapping[str, Any] | None = None,
    epoch: int | None = None,
    model_device: str = "cpu",
    train_split: str | None = "train",
    val_split: str | None = None,
    test_split: str | None = "test",
    batch_size: int = 128,
    num_dataloader_workers: int = 4,
    return_model: bool = False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    show_progress: bool = True,
) -> PreparedData:
    """
    Prepare the dataset, model and embeddings.

    Args:
        model_path (str): Path to the model checkpoint to load.
        dataset_name (str): Name of the registered dataset to evaluate on.
        dataset_args (Mapping[str, Any] | None, optional): Arguments for dataset
            loading. Defaults to None.
        epoch (int | None, optional): Epoch to evaluate. None is the latest.
            Defaults to None.
        model_device (str, optional): Device to load the model on. Defaults to "cpu".
        train_split (str | None, optional): Name of the split to use for training. None
            means do not load training data. Defaults to "train".
        val_split (str | None, optional): Name of the split to use for validation. None
            means do not load validation data. Defaults to None.
        test_split (str | None, optional): Name of the split to use for testing. None
            means do not load testing data. Defaults to "test".
        batch_size (int, optional): Batch size. Defaults to 128.
        num_dataloader_workers (int, optional): Number of workers in the dataloader.
            Defaults to 4.
        return_model (bool, optional): If True also return the loaded model. Defaults
            to False.
        pin_memory (bool, optional): Whether to use pin_memory in the dataloader.
            Defaults to True.
        persistent_workers (bool, optional): Whether to use persistent_workers in the
            dataloader. Defaults to True.
        prefetch_factor (int, optional): Prefetch factor for the dataloader. Defaults
            to 2.
        show_progress (bool, optional): Whether to display a tqdm progress bar while
            extracting image embeddings. Defaults to True.

    Returns:
        PreparedData: The prepared data including embeddings, targets, target mapping,
            dataset and optionally the loaded model.

    """
    assert train_split is not None or val_split is not None or test_split is not None, (
        "At least one of train_split, val_split or test_split must be specified"
    )

    if dataset_args is None:
        dataset_args = {}

    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    dataset = dataset_cls(**dataset_args)

    model, validation_preprocess, tokenizer = load_checkpoint(
        model_path, device=model_device, epoch=epoch
    )
    model.eval()

    train_df = (
        dataset.load_split_df_for_evaluation(train_split)
        if train_split is not None
        else None
    )
    val_df = (
        dataset.load_split_df_for_evaluation(val_split)
        if val_split is not None
        else None
    )
    test_df = (
        dataset.load_split_df_for_evaluation(test_split)
        if test_split is not None
        else None
    )

    target_columns = dataset.get_eval_target_column_names()

    assert train_df is None or all(col in train_df.columns for col in target_columns), (
        f"Not all target columns {target_columns} are present in the training dataframe"
    )
    assert val_df is None or all(col in val_df.columns for col in target_columns), (
        f"Not all target columns {target_columns} are present in the validation df"
    )
    assert test_df is None or all(col in test_df.columns for col in target_columns), (
        f"Not all target columns {target_columns} are present in the test dataframe"
    )

    train_target_df = train_df[target_columns] if train_df is not None else None
    val_target_df = val_df[target_columns] if val_df is not None else None
    test_target_df = test_df[target_columns] if test_df is not None else None

    all_target_values = dataset.get_eval_target_values()

    if all_target_values is None:
        all_target_values = {col: None for col in target_columns}

    final_all_target_values: dict[str, list[str]] = {}
    for col in all_target_values.keys():
        if all_target_values[col] is None:
            final_all_target_values[col] = sorted(
                list(
                    set(
                        (
                            train_target_df[col].unique().tolist()
                            if train_target_df is not None
                            else []
                        )
                        + (
                            val_target_df[col].unique().tolist()
                            if val_target_df is not None
                            else []
                        )
                        + (
                            test_target_df[col].unique().tolist()
                            if test_target_df is not None
                            else []
                        )
                    )
                )
            )
        else:
            final_all_target_values[col] = sorted(all_target_values[col])  # pyright: ignore[reportArgumentType]

    train_targets = (
        pd.DataFrame(
            {
                col: train_target_df[col].apply(final_all_target_values[col].index)
                for col in target_columns
            }
        )
        if train_target_df is not None
        else None
    )

    val_targets = (
        pd.DataFrame(
            {
                col: val_target_df[col].apply(final_all_target_values[col].index)
                for col in target_columns
            }
        )
        if val_target_df is not None
        else None
    )

    test_targets = (
        pd.DataFrame(
            {
                col: test_target_df[col].apply(final_all_target_values[col].index)
                for col in target_columns
            }
        )
        if test_target_df is not None
        else None
    )

    # Build an ordered list of the requested splits so a single embedding pass can be
    # sliced back into per-split embeddings regardless of which subset was requested.
    present_splits = [
        (name, df)
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
        if df is not None
    ]

    # Should not happen
    assert len(present_splits) > 0, (
        "At least one of train_df, val_df or test_df must be not None"
    )

    pred_df = pd.concat([df for _, df in present_splits], ignore_index=True)

    embedding_dataloader = get_dataloader_from_df(
        pred_df,
        dataset.get_image_path_column_name(),
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        preprocess=validation_preprocess,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    all_embeddings = extract_image_embeddings(
        model, embedding_dataloader, show_progress=show_progress
    )

    split_embeddings: dict[str, torch.Tensor] = {}
    offset = 0
    for name, df in present_splits:
        split_embeddings[name] = all_embeddings[offset : offset + len(df)]
        offset += len(df)

    train_embeddings = split_embeddings.get("train")
    val_embeddings = split_embeddings.get("val")
    test_embeddings = split_embeddings.get("test")

    # Make sure that target_value_lists and target columns are in the same order
    target_value_lists = {col: final_all_target_values[col] for col in target_columns}

    return PreparedData(
        train_embeddings=train_embeddings.cpu()
        if train_embeddings is not None
        else None,
        train_targets=train_targets,
        test_embeddings=test_embeddings.cpu() if test_embeddings is not None else None,
        test_targets=test_targets,
        target_value_lists=target_value_lists,
        dataset=dataset,
        loaded_model=model if return_model else None,
        tokenizer=tokenizer if return_model else None,
        train_image_paths=train_df[dataset.get_image_path_column_name()].tolist()
        if train_df is not None
        else None,
        val_image_paths=val_df[dataset.get_image_path_column_name()].tolist()
        if val_df is not None
        else None,
        test_image_paths=test_df[dataset.get_image_path_column_name()].tolist()
        if test_df is not None
        else None,
        val_targets=val_targets,
        val_embeddings=val_embeddings.cpu() if val_embeddings is not None else None,
    )
