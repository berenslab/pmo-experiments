"""
Combined FLAIR + PubMed-Ophtha dataset for training.

This module provides a thin wrapper dataset that holds instances of the FLAIR and
PubMed-Ophtha dataset classes, delegates conversion to each, and concatenates their
resulting train/val CSVs into a single combined dataset. It is intended for training
only; evaluation methods have minimal implementations.
"""

import logging
import os

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset
from pmo_experiments.datasets.flair_dataset import FlairDataset
from pmo_experiments.datasets.pubmed_ophtha_dataset import PubMedOphthaDataSet

logger = logging.getLogger(__name__)


def _subset_args(dataset_args: dict[str, str], prefix: str) -> dict[str, str]:
    """
    Extract and un-prefix sub-dataset arguments from the combined arguments.

    Args:
        dataset_args (dict[str, str]): The combined dataset arguments.
        prefix (str): The prefix identifying arguments for a sub-dataset (e.g.
            ``"FLAIR_"``). Matching keys have the prefix stripped.

    Returns:
        dict[str, str]: The arguments for the sub-dataset, with the prefix removed.

    """
    return {
        key.removeprefix(prefix): value
        for key, value in dataset_args.items()
        if key.startswith(prefix)
    }


@DATASET_REGISTRY.register("flair_pmo")
class FlairPmoDataset(BaseDataset):
    """
    Combined FLAIR + PubMed-Ophtha dataset integration.

    Holds a :class:`FlairDataset` and a :class:`PubMedOphthaDataSet`, delegates
    conversion to each, and joins their train/val CSVs into one combined dataset.

    Sub-dataset paths default to those of the respective dataset classes. They can be
    overridden via prefixed arguments (``FLAIR_<KEY>`` / ``PMO_<KEY>``), e.g.
    ``FLAIR_DATASET_PATH`` or ``PMO_CONVERTED_DATASET_PATH``.

    Attributes:
        converted_dataset_path (str): Path to the combined converted dataset.
        webdataset_path (str): Path for the combined WebDataset shards.
        flair (FlairDataset): The wrapped FLAIR dataset.
        pmo (PubMedOphthaDataSet): The wrapped PubMed-Ophtha dataset.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize the combined FLAIR + PubMed-Ophtha dataset.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/flair_pmo/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/flair_pmo/webdataset"
        )
        # Unused, but set to satisfy the BaseDataset contract.
        self.raw_dataset_path = self.converted_dataset_path

        # Construct the sub-datasets with their own defaults. Do not forward the
        # combined args directly, as that would leak the combined
        # CONVERTED_DATASET_PATH into both children. Prefixed keys allow explicit
        # per-child overrides.
        self.flair = FlairDataset(**_subset_args(dataset_args, "FLAIR_"))
        self.pmo = PubMedOphthaDataSet(**_subset_args(dataset_args, "PMO_"))

        self.train_size = None
        self.val_size = None

    @staticmethod
    def get_caption_column_name() -> str:
        """
        Return the name of the caption column in the converted dataset.

        Returns:
            str: Name of the caption column.

        """
        return "caption"

    def _get_dataset_sizes(self) -> None:
        train_df = pd.read_csv(self.get_train_df_path(), usecols=["image_path"])
        val_df = pd.read_csv(self.get_val_df_path(), usecols=["image_path"])

        self.train_size = len(train_df)
        self.val_size = len(val_df)

    def to_open_clip_format(self) -> None:
        """Convert and join the FLAIR and PubMed-Ophtha datasets to OpenCLIP format."""
        train_csv_path = self.get_train_df_path()
        val_csv_path = self.get_val_df_path()

        if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
            logger.debug(
                "Combined train and validation CSV files already exist. Skipping join."
            )
            self._get_dataset_sizes()
            return

        # Delegate conversion to each sub-dataset.
        self.flair.to_open_clip_format()
        self.pmo.to_open_clip_format()

        caption_col = self.get_caption_column_name()
        image_col = self.get_image_path_column_name()

        os.makedirs(self.converted_dataset_path, exist_ok=True)

        for split in ["train", "val"]:
            flair_df = pd.read_csv(self.flair.get_split_df_path(split))[
                [image_col, self.flair.get_caption_column_name()]
            ].rename(columns={self.flair.get_caption_column_name(): caption_col})

            pmo_df = pd.read_csv(self.pmo.get_split_df_path(split))[
                [image_col, self.pmo.get_caption_column_name()]
            ]

            combined_df = pd.concat([flair_df, pmo_df], ignore_index=True)
            combined_df.to_csv(self.get_split_df_path(split), index=False)

        self._get_dataset_sizes()

    def get_dataset_size(self) -> tuple[int, int]:
        """
        Return the sizes of the train and val datasets.

        Raises:
            ValueError: If `to_open_clip_format` was not called before this function.

        Returns:
            tuple[int, int]: Tuple containing train and val dataset sizes.

        """
        if self.train_size is None or self.val_size is None:
            raise ValueError("Dataset sizes have not been computed yet.")

        return self.train_size, self.val_size

    def generate_text_augmentation_dict(self) -> None | str:
        """
        Generate the text augmentation dictionary for the combined dataset.

        Delegates to the FLAIR dataset so its caption placeholders (e.g.
        ``[CLS::...::]``) are resolved during training. PubMed-Ophtha captions are
        literal and contain no placeholder keys, so they are unaffected.

        Returns:
            None | str: Path to the saved dictionary file, or None.

        """
        return self.flair.generate_text_augmentation_dict()

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Preprocess the split DataFrame for evaluation.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.
            split (str): The name of the split (e.g., 'train', 'val', 'test').

        Returns:
            pd.DataFrame: The preprocessed DataFrame.

        """
        return df

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Return the list of target column names for evaluation.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of evaluation target column names.

        """
        return ["caption"]
