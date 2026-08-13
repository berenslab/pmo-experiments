"""Module for embedding evaluation and visualization using t-SNE."""

import os
import warnings

import glasbey
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from openTSNE import TSNE as openTSNE

from pmo_experiments.evaluation import (
    PreparedData,
    prepare_dataset_and_get_embeddings,
)


def perform_opentsne(
    embeddings: np.ndarray, verbose: bool = True, num_workers: int = 4
) -> np.ndarray:
    """
    Calculate the t-SNE embeddings using openTSNE.

    Args:
        embeddings (np.ndarray): High-dimensional embeddings to reduce.
        verbose (bool, optional): If true, print verbose output. Defaults to True.
        num_workers (int, optional): Number of workers to use. Defaults to 4.

    Returns:
        np.ndarray: Two-dimensional t-SNE embeddings.

    """
    tsne = openTSNE(
        n_components=2,
        random_state=106080954,
        n_jobs=num_workers,
        verbose=verbose,
    )
    reduced_embeddings = tsne.fit(embeddings).transform(embeddings)
    return reduced_embeddings


def plot_embeddings(
    embeddings: np.ndarray,
    labels: pd.Series | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the 2D t-SNE embeddings.

    If given, use the labels to color the points.
    If the labels are numeric, use a continuous color map.
    None values are plotted in gray.

    Args:
        embeddings (np.ndarray): 2D embeddings to plot.
        labels (pd.Series | None, optional): If not None a Series defining an attribute
            used for coloring. Defaults to None.

    Returns:
        tuple[Figure, Axes]: Figure and Axes objects of the plot.

    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    dataset = pd.DataFrame(
        {"Component 1": embeddings[:, 0], "Component 2": embeddings[:, 1]}
    )

    non_nan_label_mask = None
    label_name = None
    if labels is not None:
        label_name = labels.name

        if not isinstance(label_name, str):
            warnings.warn(
                "Labels provided do not have a name. Using 'label' as default name."
            )
            label_name = "label"
        dataset[label_name] = labels
        non_nan_label_mask = ~pd.isna(labels)

    sns.set_theme(style="white", palette=None)

    is_numeric = pd.api.types.is_numeric_dtype(labels) if labels is not None else False

    color_palette = None

    if is_numeric:
        color_palette = "viridis"
    elif labels is not None:
        num_unique_labels = labels.nunique()
        if num_unique_labels <= 3:
            color_palette = "husl"
        else:
            color_palette = glasbey.create_palette(palette_size=num_unique_labels)

    dot_size = 80 if len(embeddings) < 500 else 8 if len(embeddings) < 30000 else 1.5
    sns.scatterplot(
        data=dataset if non_nan_label_mask is None else dataset[non_nan_label_mask],
        x="Component 1",
        y="Component 2",
        hue=label_name,
        palette=color_palette,  # pyright: ignore[reportArgumentType]
        alpha=0.6,
        edgecolor="none",
        ax=ax,
        s=dot_size,
    )

    if (
        non_nan_label_mask is not None
        and labels is not None
        and non_nan_label_mask.sum() < len(labels)
    ):
        sns.scatterplot(
            data=dataset[~non_nan_label_mask],
            x="Component 1",
            y="Component 2",
            color="gray",
            alpha=0.3,
            edgecolor="none",
            ax=ax,
            s=dot_size,
            label="NaN",
        )

    ax_legend = ax.get_legend()

    if ax_legend is not None:
        for handle in ax_legend.legend_handles:
            if handle is None:
                continue
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([50.0])  # pyright: ignore[reportAttributeAccessIssue]
            elif hasattr(handle, "set_markersize"):
                handle.set_markersize(np.sqrt(50.0))  # pyright: ignore[reportAttributeAccessIssue]
            handle.set_alpha(1.0)

    sns.despine(trim=True)
    ax.set_title(
        "Embedding Visualization"
        if label_name is None
        else f"Embedding Visualization by {label_name}"
    )
    return fig, ax


def plot_vision_embeddings(
    prepared_data: PreparedData,
    output_path: str,
    tsne_workers: int = 1,
    vision_embeddings_save_path: str | None = None,
):
    """
    Calculate the 2D embeddings and save them to the path.

    Calculates the vision embeddings of the images in the dataframe.
    Uses image_column to determine the correct column.
    The embedding plots are saved to the file defined at output_path.
    If color_columns is defined, an additional plot per color is saved at the same
    location by appending the column name.

    Args:
        prepared_data (PreparedData): The prepared data including embeddings, targets,
            target mapping.
        output_path (str): File path to save the embedding plot.
        tsne_workers (int, optional): Number of workers in openTSNE. Defaults to
            1.
        vision_embeddings_save_path (str | None, optional): Path to save the raw t-SNE
            embeddings to. If None the embeddings are not saved. Defaults to None.

    """
    train_embeddings = prepared_data["train_embeddings"]
    train_target_indices = prepared_data["train_targets"]
    train_image_paths = prepared_data["train_image_paths"]
    target_value_lists = prepared_data["target_value_lists"]

    assert train_embeddings is not None, "Prepared data has no train embeddings."
    assert train_target_indices is not None, (
        "Prepared data has no train target indices."
    )
    assert train_image_paths is not None, "Prepared data has no train image paths."

    tsne_embeddings = perform_opentsne(
        train_embeddings.numpy(), verbose=True, num_workers=tsne_workers
    )

    if vision_embeddings_save_path is not None:
        os.makedirs(os.path.dirname(vision_embeddings_save_path), exist_ok=True)
        tsne_embeddings_df = pd.DataFrame(
            tsne_embeddings, columns=["Component 1", "Component 2"]
        )
        tsne_embeddings_df["image_path"] = train_image_paths
        tsne_embeddings_df.to_csv(vision_embeddings_save_path, index=False)

    fig, _ = plot_embeddings(tsne_embeddings)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.close(fig)

    for color_column in train_target_indices.columns:
        label_series = train_target_indices[color_column].map(
            lambda idx: target_value_lists[color_column][idx]  # pyright: ignore[reportCallIssue,reportArgumentType]
        )
        assert isinstance(label_series, pd.Series)  # for pyright type checking
        fig, _ = plot_embeddings(
            tsne_embeddings,
            labels=label_series,
        )
        color_output_path = (
            os.path.splitext(output_path)[0]
            + f"_{color_column}"
            + os.path.splitext(output_path)[1]
        )
        fig.savefig(color_output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_dataset_embeddings(
    model_path: str,
    dataset_name: str,
    output_path: str,
    batch_size: int = 64,
    dataloader_workers: int = 1,
    epoch: int | None = None,
    model_device: str = "cpu",
    vision_embeddings_save_path: str | None = None,
    dataset_args: dict | None = None,
    split: str = "train",
):
    """
    Calculate and plot the embeddings for a given dataset.

    Args:
        model_path (str): Path to the model folder.
        dataset_name (str): Name of the dataset to use.
        output_path (str): File name to save the output plot.
        batch_size (int, optional): Batch size in the forward pass. Defaults to 64.
        dataloader_workers (int, optional): Number of workers in the dataloader and in
            openTSNE. Defaults to 1.
        epoch (int | None, optional): Number of the epoch. If None use the latest.
            Defaults to None.
        model_device (str | torch.device, optional): Device to use for the model.
            Defaults to "cpu".
        vision_embeddings_save_path (str | None, optional): Path to save the raw t-SNE
            embeddings to. Defaults to None.
        dataset_args (dict | None, optional): Arguments for the dataset instantiation.
            Defaults to None.
        split (str, optional): Name of the split. Has to be one of "train", "val" and
            "test". Defaults to "train".

    Raises:
        ValueError: If the split name is invalid.

    """
    prepared_data = prepare_dataset_and_get_embeddings(
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_args=dataset_args,
        epoch=epoch,
        model_device=model_device,
        train_split=split,
        batch_size=batch_size,
        num_dataloader_workers=dataloader_workers,
    )

    plot_vision_embeddings(
        prepared_data=prepared_data,
        output_path=output_path,
        tsne_workers=dataloader_workers,
        vision_embeddings_save_path=vision_embeddings_save_path,
    )
