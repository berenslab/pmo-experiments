"""Integration of the APTOS 2019 dataset."""

import logging
import os
import zipfile

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("aptos2019")
class APTOSDataset(BaseDataset):
    """APTOS 2019 dataset integration."""

    def __init__(self, **dataset_args: str):
        """
        Initialize APTOS 2019 dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get(
            "DATASET_PATH", "datasets/aptos2019/APTOS2019.zip"
        )

        assert self.raw_dataset_path.endswith(".zip"), (
            "APTOS 2019 dataset file must be a zip file with .zip extension."
        )
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/aptos2019/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/aptos2019/webdataset"
        )

        self.train_size = None
        self.val_size = None
        self.test_size = None

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
        test_csv_path = os.path.join(self.converted_dataset_path, "test.csv")

        if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
            logger.debug(
                "Train and validation CSV files already exist. Skipping unzip."
            )

            self._get_dataset_sizes()
            return

        assert os.path.exists(self.raw_dataset_path), (
            f"APTOS2019 dataset not found at {self.raw_dataset_path}. "
            "Please download it from Kaggle and place it there."
        )

        test_output_path = os.path.join(self.converted_dataset_path, "test")
        val_output_path = os.path.join(self.converted_dataset_path, "val")
        train_output_path = os.path.join(self.converted_dataset_path, "train")

        os.makedirs(val_output_path, exist_ok=True)
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)

        with zipfile.ZipFile(self.raw_dataset_path, mode="r") as zf:
            file_names = zf.namelist()

            # check for label files

            if "test.csv" not in file_names:
                raise ValueError(
                    "test.csv not found in the raw dataset zip. "
                    "Please ensure it is included in the zip file."
                )

            test_labels_df = pd.read_csv(zf.open("test.csv"))

            # Copy image files to output folders
            for file_id in test_labels_df["id_code"]:
                file_path = f"test_images/test_images/{file_id}.png"

                if file_path not in file_names:
                    raise ValueError(
                        f"{file_path} not found in the raw dataset zip. "
                        "Please ensure all test images are included in the zip file."
                    )
                with zf.open(file_path) as img_file:
                    img_bytes = img_file.read()
                    with open(
                        os.path.join(test_output_path, f"{file_id}.png"), "wb"
                    ) as out_file:
                        out_file.write(img_bytes)

            if "train_1.csv" not in file_names:
                raise ValueError(
                    "train_1.csv not found in the raw dataset zip. "
                    "Please ensure it is included in the zip file."
                )

            train_labels_df = pd.read_csv(zf.open("train_1.csv"))

            # Copy image files to output folders
            for file_id in train_labels_df["id_code"]:
                file_path = f"train_images/train_images/{file_id}.png"

                if file_path not in file_names:
                    raise ValueError(
                        f"{file_path} not found in the raw dataset zip. "
                        "Please ensure all train images are included in the zip file."
                    )
                with zf.open(file_path) as img_file:
                    img_bytes = img_file.read()
                    with open(
                        os.path.join(train_output_path, f"{file_id}.png"), "wb"
                    ) as out_file:
                        out_file.write(img_bytes)

            if "valid.csv" not in file_names:
                raise ValueError(
                    "valid.csv not found in the raw dataset zip. "
                    "Please ensure it is included in the zip file."
                )

            val_labels_df = pd.read_csv(zf.open("valid.csv"))

            # Copy image files to output folders
            for file_id in val_labels_df["id_code"]:
                file_path = f"val_images/val_images/{file_id}.png"

                if file_path not in file_names:
                    raise ValueError(
                        f"{file_path} not found in the raw dataset zip. "
                        "Please ensure all val images are included in the zip file."
                    )
                with zf.open(file_path) as img_file:
                    img_bytes = img_file.read()
                    with open(
                        os.path.join(val_output_path, f"{file_id}.png"), "wb"
                    ) as out_file:
                        out_file.write(img_bytes)

            test_labels_df["image_path"] = test_labels_df["id_code"].apply(
                lambda x: os.path.join(test_output_path, f"{x}.png")
            )
            val_labels_df["image_path"] = val_labels_df["id_code"].apply(
                lambda x: os.path.join(val_output_path, f"{x}.png")
            )
            train_labels_df["image_path"] = train_labels_df["id_code"].apply(
                lambda x: os.path.join(train_output_path, f"{x}.png")
            )

            test_labels_df.to_csv(test_csv_path, index=False)
            val_labels_df.to_csv(val_csv_path, index=False)
            train_labels_df.to_csv(train_csv_path, index=False)

            # Set dataset sizes
            self.train_size = len(train_labels_df)
            self.val_size = len(val_labels_df)
            self.test_size = len(test_labels_df)

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

        df["diagnosis"] = df["diagnosis"].apply(
            lambda x: severity_to_text[x] if pd.notna(x) else "Unknown"
        )

        return df

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Return the list of target column names for evaluation.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of target column names for evaluation.

        """
        return ["diagnosis"]

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value for the given column.

        """
        if column_name == "diagnosis":
            return "A fundus image showing {severity} diabetic retinopathy.".format(
                severity=value
            )
        else:
            return value

    def get_eval_target_values(
        self,
    ) -> dict[str, list[str] | None] | None:
        """
        Get all possible target values for evaluation.

        Returns:
            dict[str, list[str] | None] | None: A dictionary mapping target column
                names to lists of possible target values, or None.

        """
        return {"diagnosis": ["no", "mild", "moderate", "severe", "proliferative"]}
