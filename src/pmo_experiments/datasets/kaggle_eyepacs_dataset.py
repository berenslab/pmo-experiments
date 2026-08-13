"""Integration of the Kaggle EyePACS dataset."""

import glob
import logging
import os
import zipfile
from contextlib import contextmanager
from io import BytesIO
from typing import Generator

import pandas as pd
from tqdm import tqdm

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

logger = logging.getLogger(__name__)


@contextmanager
def open_split_zip(
    zip_file: zipfile.ZipFile, part_file_names: list[str], desc: str | None = None
) -> Generator[zipfile.ZipFile, None, None]:
    """
    Open a zip file that was split into multiple parts.

    Create a context manager that gets all of the zip file parts using the
    part_file_names and combines them into a single in-memory zip file.

    Args:
        zip_file (zipfile.ZipFile): Zip file to read parts from.
        part_file_names (list[str]): Names of the part files to combine.
        desc (str | None, optional): Description for the progress bar. Defaults to None.

    Yields:
        zipfile.ZipFile: Combined zip file.

    """
    with BytesIO() as combined_bytes:
        for part_file_name in tqdm(sorted(part_file_names), desc=desc):
            with zip_file.open(part_file_name) as part_file:
                combined_bytes.write(part_file.read())

        combined_bytes.seek(0)
        with zipfile.ZipFile(combined_bytes) as zf:
            yield zf


def process_train_labels(
    df: pd.DataFrame, converted_dataset_path: str, dataset_random_seed: int = 15121997
) -> pd.DataFrame:
    """
    Preprocess the train labels of the dataset.

    Split the dataset into train and validation.
    Add a template based text and the path to the images.

    Args:
        df (pd.DataFrame): Dataset to preprocess.
        converted_dataset_path (str): Path to the converted dataset.
        dataset_random_seed (int, optional): Seed to create reproducible train-val-
            split. Defaults to 15121997.

    Returns:
        pd.DataFrame: Preprocessed dataset.

    """
    df["side"] = df["image"].apply(lambda x: "left" if x.endswith("left") else "right")
    df["image_id"] = df["image"].apply(
        lambda x: int(x.removesuffix("_left").removesuffix("_right"))
    )  # Remove '_left' or '_right'

    num_unique_images = df["image_id"].nunique()
    num_train_samples = int(0.8 * num_unique_images)

    train_image_ids = (
        df["image_id"]
        .drop_duplicates()
        .sample(n=num_train_samples, random_state=dataset_random_seed)
        .tolist()
    )

    df["split"] = df["image_id"].apply(
        lambda x: "train" if x in train_image_ids else "val"
    )

    severity_to_text = {
        0: "no",
        1: "mild",
        2: "moderate",
        3: "severe",
        4: "proliferative",
    }

    label_text_template = "A fundus image showing {severity} diabetic retinopathy."
    label_text_with_side_template = (
        "A {side} eye fundus image showing {severity} diabetic retinopathy."
    )

    df["label_text"] = df["level"].apply(
        lambda x: label_text_template.format(severity=severity_to_text[x])
    )

    df["label_text_with_side"] = df.apply(
        lambda row: label_text_with_side_template.format(
            side=row["side"], severity=severity_to_text[row["level"]]
        ),
        axis=1,
    )

    df["image_path"] = df.apply(
        lambda row: os.path.join(
            converted_dataset_path, row["split"], f"{row['image']}.jpeg"
        ),
        axis=1,
    )

    df = df.sort_values(by=["split", "image_id", "side"]).reset_index(drop=True)
    return df


def process_test_labels(df: pd.DataFrame, converted_dataset_path: str) -> pd.DataFrame:
    """
    Preprocess the test labels of the dataset.

    Add a template based text and the path to the images.

    Args:
        df (pd.DataFrame): Dataset to preprocess.
        converted_dataset_path (str): Path to the converted dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.

    """
    df["side"] = df["image"].apply(lambda x: "left" if x.endswith("left") else "right")
    df["image_id"] = df["image"].apply(
        lambda x: int(x.removesuffix("_left").removesuffix("_right"))
    )  # Remove '_left' or '_right'

    severity_to_text = {
        0: "no",
        1: "mild",
        2: "moderate",
        3: "severe",
        4: "proliferative",
    }

    label_text_template = "A fundus image showing {severity} diabetic retinopathy."
    label_text_with_side_template = (
        "A {side} eye fundus image showing {severity} diabetic retinopathy."
    )

    df["label_text"] = df["level"].apply(
        lambda x: label_text_template.format(severity=severity_to_text[x])
    )

    df["label_text_with_side"] = df.apply(
        lambda row: label_text_with_side_template.format(
            side=row["side"], severity=severity_to_text[row["level"]]
        ),
        axis=1,
    )

    df["image_path"] = df.apply(
        lambda row: os.path.join(
            converted_dataset_path, "test", f"{row['image']}.jpeg"
        ),
        axis=1,
    )

    return df


@DATASET_REGISTRY.register("kaggle_eyepacs")
class KaggleEyePACSDataset(BaseDataset):
    """Kaggle EyePACS dataset integration."""

    def __init__(self, **dataset_args: str):
        """
        Initialize Kaggle EyePACS dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get(
            "DATASET_PATH", "datasets/kaggle-eyepacs/diabetic-retinopathy-detection.zip"
        )

        assert self.raw_dataset_path.endswith(".zip"), (
            "Kaggle EyePACS dataset file must be a zip file with .zip extension."
        )
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/kaggle-eyepacs/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/kaggle-eyepacs/webdataset"
        )

        self.train_size = None
        self.val_size = None

    @staticmethod
    def get_caption_column_name() -> str:
        """
        Return the name of the caption column in the converted dataset.

        Returns:
            str: Name of the caption column.

        """
        return "label_text"

    def get_dataset_size(self) -> tuple[int, int]:
        """
        Return the sizes of the train and val datasets.

        Raises:
            ValueError: If the `to_open_clip_format` was not called before this
                function.

        Returns:
            tuple[int, int]: Tuple containing train and val dataset sizes.

        """
        if self.train_size is None or self.val_size is None:
            raise ValueError("Dataset sizes have not been computed yet.")

        return self.train_size, self.val_size

    def _get_dataset_sizes(self) -> None:
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        val_csv_path = os.path.join(self.converted_dataset_path, "val.csv")

        if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
            raise ValueError(
                "Train and validation CSV files do not exist. "
                "Please run to_open_clip_format() first."
            )

        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        self.train_size = len(train_df)
        self.val_size = len(val_df)

    def to_open_clip_format(self) -> None:
        """Convert the dataset to OpenCLIP format."""
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        val_csv_path = os.path.join(self.converted_dataset_path, "val.csv")

        if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
            logger.debug(
                "Train and validation CSV files already exist. Skipping unzip."
            )

            self._get_dataset_sizes()
            return

        assert os.path.exists(self.raw_dataset_path), (
            f"Kaggle EyePACS dataset not found at {self.raw_dataset_path}. "
            "Please download it from Kaggle and place it there."
        )
        tmp_train_path = os.path.join(self.converted_dataset_path, "tmp_train.csv")
        tmp_val_path = os.path.join(self.converted_dataset_path, "tmp_val.csv")

        test_output_path = os.path.join(self.converted_dataset_path, "test")
        val_output_path = os.path.join(self.converted_dataset_path, "val")
        train_output_path = os.path.join(self.converted_dataset_path, "train")

        os.makedirs(val_output_path, exist_ok=True)
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)

        with open(self.raw_dataset_path, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                file_names = zf.namelist()

                train_part_names = [
                    name for name in file_names if name.startswith("train.zip.")
                ]
                test_part_names = [
                    name for name in file_names if name.startswith("test.zip.")
                ]
                train_labels_name = "trainLabels.csv.zip"

                existing_files_per_split = {
                    "train": {
                        os.path.basename(fp).removesuffix(".jpeg"): fp
                        for fp in glob.glob(os.path.join(train_output_path, "*.jpeg"))
                    },
                    "val": {
                        os.path.basename(fp).removesuffix(".jpeg"): fp
                        for fp in glob.glob(os.path.join(val_output_path, "*.jpeg"))
                    },
                    "test": {
                        os.path.basename(fp).removesuffix(".jpeg"): fp
                        for fp in glob.glob(os.path.join(test_output_path, "*.jpeg"))
                    },
                }

                if os.path.exists(tmp_train_path) and os.path.exists(tmp_val_path):
                    df = pd.concat(
                        [pd.read_csv(tmp_train_path), pd.read_csv(tmp_val_path)],
                        ignore_index=True,
                    )
                else:
                    # Open and read trainLabels.csv
                    with zf.open(train_labels_name) as train_labels_file:
                        with zipfile.ZipFile(train_labels_file) as train_labels_zf:
                            with train_labels_zf.open("trainLabels.csv") as csv_file:
                                df = process_train_labels(
                                    pd.read_csv(csv_file), self.converted_dataset_path
                                )

                        os.makedirs(self.converted_dataset_path, exist_ok=True)

                        self.train_size = len(df[df["split"] == "train"])
                        self.val_size = len(df[df["split"] == "val"])

                        df[df["split"] == "train"].to_csv(tmp_train_path, index=False)
                        df[df["split"] == "val"].to_csv(tmp_val_path, index=False)

                # Extract train images
                with open_split_zip(
                    zf, train_part_names, desc="Combining train parts"
                ) as train_zf:
                    train_file_names = train_zf.namelist()

                    for file_name in tqdm(
                        train_file_names, desc="Extracting train images"
                    ):
                        if not file_name.endswith(".jpeg"):
                            continue

                        img_base_name = os.path.basename(file_name).removesuffix(
                            ".jpeg"
                        )
                        if (
                            img_base_name in existing_files_per_split["train"]
                            or img_base_name in existing_files_per_split["val"]
                        ):
                            continue

                        split = df.loc[df["image"] == img_base_name, "split"].values[0]

                        if split == "train":
                            extract_path = train_output_path
                        else:
                            extract_path = val_output_path

                        output_file_path = os.path.join(
                            extract_path, f"{img_base_name}.jpeg"
                        )

                        with train_zf.open(file_name) as img_file:
                            with open(output_file_path, "wb") as out_file:
                                out_file.write(img_file.read())

                # Extract test images
                with open_split_zip(
                    zf, test_part_names, desc="Combining test parts"
                ) as test_zf:
                    test_file_names = test_zf.namelist()

                    for file_name in tqdm(
                        test_file_names, desc="Extracting test images"
                    ):
                        if not file_name.endswith(".jpeg"):
                            continue

                        img_base_name = os.path.basename(file_name)

                        if img_base_name in existing_files_per_split["test"]:
                            continue

                        with test_zf.open(file_name) as img_file:
                            with open(
                                os.path.join(test_output_path, img_base_name), "wb"
                            ) as out_file:
                                out_file.write(img_file.read())

        # Rename temporary CSV files to final names
        os.rename(tmp_train_path, train_csv_path)
        os.rename(tmp_val_path, val_csv_path)

        # Check if test labels exist
        test_labels_name = "retinopathy_solution.csv"

        if not os.path.exists(os.path.join(self.raw_dataset_path, test_labels_name)):
            logger.warning(
                "Test labels not found in the raw dataset zip. "
                "Skipping test labels extraction."
            )
            return

        process_test_labels(
            pd.read_csv(os.path.join(self.raw_dataset_path, test_labels_name)),
            self.converted_dataset_path,
        ).to_csv(
            os.path.join(self.converted_dataset_path, "test_labels.csv"), index=False
        )

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Convert target columns to text.

        Args:
            df (pd.DataFrame): DF to preprocess.
            split (str): Target split. Must be one of 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: Preprocessed DF with text targets.

        """
        severity_to_text = {
            0: "no",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }

        df["level"] = df["level"].apply(
            lambda x: severity_to_text[x] if pd.notna(x) else "Unknown"
        )

        df["side"] = df["side"].fillna("unknown side")

        return df

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:
        """
        Return the list of target column names for evaluation.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of target column names for evaluation.

        """
        if relevant_only:
            return ["level"]
        return ["level", "side"]

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value for the given column.

        """
        if column_name == "level":
            return "A fundus image showing {severity} diabetic retinopathy.".format(
                severity=value
            )
        elif column_name == "side":
            article = "an" if value[0].lower() in "aeiou" else "a"
            return "A fundus image showing {article} {side} eye.".format(
                article=article, side=value
            )
        else:
            return value


@DATASET_REGISTRY.register("kaggle_eyepacs_w_side")
class KaggleEyePACSWithSideDataset(KaggleEyePACSDataset):
    """Kaggle EyePACS dataset integration with side information in captions."""

    @staticmethod
    def get_caption_column_name() -> str:
        """
        Return the name of the caption column in the converted dataset.

        Returns:
            str: Name of the caption column.

        """
        return "label_text_with_side"
