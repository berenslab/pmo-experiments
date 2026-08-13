"""Torch Datasets and Dataloaders for pmo-experiments."""

from typing import Any

import pandas as pd
import torch
import torchvision.io as io
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


class ImageDataset(Dataset):
    """
    Dataset for loading images from a pandas DataFrame.

    Attributes:
        paths (list[str]): List of image paths.
        preprocess (Compose | None): Preprocessing transform.
        targets (list[Any] | None): List of target values if provided.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        preprocess: Compose | None = None,
        target_col: str | None = None,
    ):
        """
        Create a new Image dataset based on a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and optional targets.
            path_col (str): Name of the column containing image paths.
            preprocess (Compose | None, optional): Preprocessing transform. Defaults to
                None.
            target_col (str | None, optional): Name of the column containing the target
                values. Defaults to None.

        """
        self.paths = df[path_col].tolist()
        self.preprocess = preprocess

        if target_col is not None:
            self.targets = df[target_col].tolist()
        else:
            self.targets = None

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, Any] | torch.Tensor | Image.Image:
        """
        Get the item at the index.

        If the dataset has a target column, returns a tuple[Image, target_value], else
        just return the Image.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple[torch.Tensor | Image.Image, Any] | torch.Tensor | Image.Image: A
                tuple of (Image, target_value) if target_col was provided, else just
                the Image. The image might be a tensor due to preprocessing.

        """
        path = self.paths[idx]

        try:
            img = Image.fromarray(
                io.read_image(path, mode=io.ImageReadMode.RGB).permute(1, 2, 0).numpy()
            )
        except RuntimeError:
            with Image.open(path) as _img:
                img = _img.convert("RGB")

        if self.preprocess is not None:
            img = self.preprocess(img)

        if self.targets is not None:
            target = self.targets[idx]

            return img, target
        return img


def get_dataloader_from_df(
    df: pd.DataFrame,
    path_col: str,
    batch_size: int,
    preprocess: Compose | None = None,
    shuffle: bool = False,
    num_workers: int = 4,
    target_col: str | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a dataloader for the images in the dataframe.

    If target_col is not None, the dataloader iterates over a batch of images and their
    target values. Else the dataloader just iterates over the image batches.

    Args:
        df (pd.DataFrame): Dataframe containing image paths and optional targets.
        path_col (str): Name of the column containing image paths.
        batch_size (int): Batch size.
        preprocess (Compose | None, optional): Preprocessing transform. Ignored if
            None. Defaults to None.
        shuffle (bool, optional): If True shuffle the samples. Defaults to False.
        num_workers (int, optional): Number of workers to use for data loading. Defaults
            to 4.
        target_col (str | None, optional): Name of the column containing the target
            value. Defaults to None.
        pin_memory (bool, optional): If True, pin the memory. Defaults to True.
        persistent_workers (bool, optional): If True, the data loader will not shut down
            the worker processes after a dataset has been consumed once. Defaults to
            True.
        prefetch_factor (int, optional): Number of samples loaded in advance by each
            worker. Defaults to 2.

    Returns:
        DataLoader: Dataloader for the images in the dataframe. If target_col was
            provided, the dataloader yields tuples of (image_batch, target_batch), else
            it yields just image_batch.

    """
    dataset = ImageDataset(df, path_col, preprocess, target_col=target_col)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return dataloader
