"""Module for dataset base class and registry."""

import logging
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import webdataset as wds  # pyright: ignore[reportMissingImports]

from pmo_experiments.util.registry import Registry

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @staticmethod
    def get_caption_column_name() -> str:
        """
        Return the name of the caption column in the converted dataset.

        Returns:
            str: Name of the caption column.

        """
        return "caption"

    @staticmethod
    def get_image_path_column_name() -> str:
        """
        Return the name of the image path column in the converted dataset.

        Returns:
            str: Name of the image path column.

        """
        return "image_path"

    @abstractmethod
    def __init__(self, **dataset_args):
        """Initialize dataset with given arguments."""
        self.converted_dataset_path: str
        self.raw_dataset_path: str
        self.webdataset_path: str

    @abstractmethod
    def to_open_clip_format(self) -> None:
        """Convert dataset to OpenCLIP format."""
        pass

    def to_webdataset(
        self,
        batch_size: int,
        n_batches_per_shard: int = 10,
        min_shards: int = 100,
        num_workers: int = 16,
    ) -> tuple[str, str]:
        """
        Pack train/val CSVs into WebDataset tar shards.

        Shard size targets `n_batches_per_shard * batch_size` samples, but is capped
        so that at least `min_shards` shards are produced for adequate shuffle
        randomness. Idempotent: skips a split if its output directory already contains
        .tar files. Images are read in parallel using `num_workers` threads; train and
        val splits are packed concurrently.

        Args:
            batch_size (int): Effective batch size (samples per training step).
            n_batches_per_shard (int, optional): Target number of batches per shard.
                Defaults to 10.
            min_shards (int, optional): Minimum number of shards per split.
                Defaults to 100.
            num_workers (int, optional): Number of threads for parallel image I/O.
                Defaults to 16.

        Returns:
            tuple[str, str]: Brace-expanded shard patterns for (train, val), e.g.
                ``/path/train/{00000..00099}.tar``. Safe to pass directly to OpenCLIP's
                ``--train-data``/``--val-data`` flags.

        """
        img_col = self.get_image_path_column_name()
        cap_col = self.get_caption_column_name()

        def _write(csv_path: str, out_dir: str) -> None:
            os.makedirs(out_dir, exist_ok=True)
            df = pd.read_csv(csv_path)
            preferred_shard_size = n_batches_per_shard * batch_size
            effective_min_shards = max(min_shards, len(df) // preferred_shard_size)
            shard_size = max(
                256,
                min(
                    preferred_shard_size,
                    math.ceil(len(df) / effective_min_shards),
                ),
            )
            approx_num_shards = math.ceil(len(df) / shard_size)
            logger.info(
                f"Packing {len(df)} samples from {csv_path} into {approx_num_shards} "
                f"shards of ~{shard_size} samples each at {out_dir}..."
            )

            def _read_group(args: tuple[Any, Any]) -> list[dict[str, Any]]:
                img_path, group = args
                with open(str(img_path), "rb") as f:
                    img_bytes = f.read()
                return [
                    {
                        "__key__": f"{row.Index:08d}",
                        "png": img_bytes,
                        "txt": str(getattr(row, cap_col)),
                    }
                    for row in group.itertuples()
                ]

            with wds.ShardWriter(
                os.path.join(out_dir, "%05d.tar"), maxcount=shard_size
            ) as sink:
                with ThreadPoolExecutor(max_workers=num_workers // 2) as pool:
                    for samples in pool.map(
                        _read_group, df.groupby(img_col, sort=False)
                    ):
                        for sample in samples:
                            sink.write(sample)

        train_dir = os.path.join(self.webdataset_path, "train")
        val_dir = os.path.join(self.webdataset_path, "val")

        train_csv_path = self.get_train_df_path()
        val_csv_path = self.get_val_df_path()

        if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
            raise FileNotFoundError(
                f"Train or val CSV not found at {train_csv_path} or {val_csv_path}. "
                "Call to_open_clip_format() first to convert the dataset."
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            if not os.path.isdir(train_dir) or not any(
                f.endswith(".tar") for f in os.listdir(train_dir)
            ):
                logger.info(
                    f"Packing train split into WebDataset shards at {train_dir}..."
                )
                futures.append(pool.submit(_write, self.get_train_df_path(), train_dir))
            if not os.path.isdir(val_dir) or not any(
                f.endswith(".tar") for f in os.listdir(val_dir)
            ):
                logger.info(f"Packing val split into WebDataset shards at {val_dir}...")
                futures.append(pool.submit(_write, self.get_val_df_path(), val_dir))
            for future in futures:
                future.result()

        def _brace(directory: str) -> str:
            names = sorted(
                (f for f in os.listdir(directory) if f.endswith(".tar")),
                key=lambda f: int(f.removesuffix(".tar")),
            )
            first = names[0].removesuffix(".tar")
            last = names[-1].removesuffix(".tar")
            return os.path.join(directory, "{" + f"{first}..{last}" + "}.tar")

        return _brace(train_dir), _brace(val_dir)

    @abstractmethod
    def get_dataset_size(self) -> tuple[int, int]:
        """
        Return the sizes of the train and val datasets.

        Returns:
            tuple[int, int]: Tuple containing train and val dataset sizes.

        """
        pass

    def get_train_df_path(self) -> str:
        """
        Return the path to the training dataframe CSV file.

        Returns:
            str: Path to the training dataframe CSV file.

        """
        return os.path.join(self.converted_dataset_path, "train.csv")

    def get_val_df_path(self) -> str:
        """
        Return the path to the validation dataframe CSV file.

        Returns:
            str: Path to the validation dataframe CSV file.

        """
        return os.path.join(self.converted_dataset_path, "val.csv")

    def get_test_df_path(self) -> str:
        """
        Return the path to the test dataframe CSV file.

        Returns:
            str: Path to the test dataframe CSV file.

        """
        return os.path.join(self.converted_dataset_path, "test.csv")

    def get_beton_path(self, split: str, target_column: str | None = None) -> str:
        """
        Return the path to the FFCV .beton file for the specified split.

        When target_column is given, the file embeds labels for that column and the
        filename includes the column name to keep separate per-target files distinct.

        Args:
            split (str): The target split. Must be one of 'train', 'val', or 'test'.
            target_column (str | None, optional): Label column name. Defaults to None.

        Returns:
            str: Path to the .beton file for the specified split.

        """
        if target_column is None:
            return os.path.join(self.converted_dataset_path, f"{split}.beton")
        return os.path.join(
            self.converted_dataset_path, f"{split}_{target_column}.beton"
        )

    def get_split_df_path(self, split: str) -> str:
        """
        Return the path to the dataframe CSV file for the specified split.

        Args:
            split (str): The target split. Must be one of 'train', 'val', or 'test'.

        Returns:
            str: Path to the dataframe CSV file for the specified split.

        """
        if split == "train":
            return self.get_train_df_path()
        elif split == "val":
            return self.get_val_df_path()
        elif split == "test":
            return self.get_test_df_path()

        else:
            raise ValueError(
                f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'."
            )

    @abstractmethod
    def preprocess_split_df_for_evaluation(
        self, df: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe for the specified split for evaluation.

        This method can be used to perform any necessary preprocessing steps on the
        dataframe before evaluation, such as filtering or formatting.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
            split (str): The target split. Must be one of 'train', 'val', or 'test'.

        """
        pass

    @staticmethod
    @abstractmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:
        """
        Return the list of target column names to be used for evaluation.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of target column names to be used for evaluation.

        """
        pass

    def load_split_df_for_evaluation(self, split: str | list[str]) -> pd.DataFrame:
        """
        Load the dataframe for the specified split for evaluation.

        This method can be used to load the dataframe for the specified split after
        it has been preprocessed for evaluation.

        Args:
            split (str | list[str]): The target split(s). Must be one of 'train',
                'val', or 'test' or a list.

        Returns:
            pd.DataFrame: The loaded dataframe for the specified split.

        """
        if isinstance(split, list):
            dfs = [self.load_split_df_for_evaluation(s) for s in split]
            return pd.concat(dfs, ignore_index=True)
        elif isinstance(split, str):
            split_df_path = self.get_split_df_path(split)

            df = pd.read_csv(split_df_path)

            return self.preprocess_split_df_for_evaluation(df, split)
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be a string or a list of strings."
            )

    def get_eval_target_values(
        self,
    ) -> dict[str, list[str] | None] | None:
        """
        Get the target values for evaluation.

        Returns:
            dict[str, list[str] | None] | None: Dictionary mapping split names to
                lists of target values, or None if evaluation target values are not
                available.

        """
        return None

    def get_value_verbalization(self, column_name: str, value: str) -> str:  # noqa: ARG002
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        return value

    def generate_text_augmentation_dict(self) -> None | str:
        """
        Generate and save the dictionary for text augmentations.

        This dictionary defines keys that are randomly replaced by the list of values.

        Returns:
            None | str: None if no text augmentation is needed, or the path to the
                saved dictionary file.

        """
        return None


DATASET_REGISTRY = Registry[BaseDataset]()
