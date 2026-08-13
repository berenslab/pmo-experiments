"""Integration of the EDDFS dataset."""

import logging
import os
import zipfile

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("eddfs")
class EDDFSDataset(BaseDataset):
    """EDDFS dataset integration."""

    def __init__(self, **dataset_args: str):
        """
        Initialize EDDFS dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get(
            "DATASET_PATH", "datasets/eddfs/OriginalImages.zip"
        )

        assert self.raw_dataset_path.endswith(".zip"), (
            "EDDFS dataset file must be a zip file with .zip extension."
        )
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/eddfs/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/eddfs/webdataset"
        )

        self.train_csv_url = "https://raw.githubusercontent.com/xia-xx-cv/EDDFS_dataset/refs/heads/main/datas/EDDFS/Annotation/train.csv"  # "https://raw.githubusercontent.com/xia-xx-cv/EDDFS_dataset/refs/heads/main/datas/EDDFS/split2/train.csv"  # noqa: E501
        self.val_csv_url = None  # "https://raw.githubusercontent.com/xia-xx-cv/EDDFS_dataset/refs/heads/main/datas/EDDFS/split2/val.csv"
        self.test_csv_url = "https://raw.githubusercontent.com/xia-xx-cv/EDDFS_dataset/refs/heads/main/datas/EDDFS/Annotation/test.csv"  # "https://raw.githubusercontent.com/xia-xx-cv/EDDFS_dataset/refs/heads/main/datas/EDDFS/split2/test.csv"  # noqa: E501

        self.train_size = None
        self.val_size = None
        self.test_size = None

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

        if (
            os.path.exists(train_csv_path)
            and os.path.exists(val_csv_path)
            and os.path.exists(test_csv_path)
        ):
            logger.debug(
                "Train and validation CSV files already exist. Skipping unzip."
            )

            self._get_dataset_sizes()
            return

        assert os.path.exists(self.raw_dataset_path), (
            f"EDDFS dataset not found at {self.raw_dataset_path}. "
            "Please download it and place it there."
        )

        train_dataframe = pd.read_csv(self.train_csv_url)
        test_dataframe = pd.read_csv(self.test_csv_url)

        if self.val_csv_url is not None:
            val_dataframe = pd.read_csv(self.val_csv_url)
        else:
            # Create stratified val split from train dataframe
            seed = 106080954

            # Columns to stratify by
            stratify_cols = [
                "normal",
                "DR",
                "AMD",
                "glaucoma",
                "myopia",
                "RVO",
                "LS",
                "hyper",
                "others",
            ]
            proxy_column = train_dataframe[stratify_cols].apply(
                lambda row: "_".join([f"{col}_{row[col]}" for col in stratify_cols]),
                axis=1,
            )

            val_fraction = 0.1
            val_dataframe = train_dataframe.groupby(
                proxy_column, group_keys=False
            ).apply(lambda x: x.sample(frac=val_fraction, random_state=seed))
            train_dataframe = train_dataframe.drop(val_dataframe.index)

        train_image_folder = os.path.join(self.converted_dataset_path, "train")
        val_image_folder = os.path.join(self.converted_dataset_path, "val")
        test_image_folder = os.path.join(self.converted_dataset_path, "test")

        os.makedirs(train_image_folder, exist_ok=True)
        os.makedirs(val_image_folder, exist_ok=True)
        os.makedirs(test_image_folder, exist_ok=True)

        train_file_names = set(train_dataframe["fnames"].tolist())
        val_file_names = set(val_dataframe["fnames"].tolist())
        test_file_names = set(test_dataframe["fnames"].tolist())

        assert train_file_names & val_file_names == set(), (
            "Train and validation sets have overlapping images."
        )
        assert train_file_names & test_file_names == set(), (
            "Train and test sets have overlapping images."
        )
        assert val_file_names & test_file_names == set(), (
            "Validation and test sets have overlapping images."
        )

        # open zip file
        with zipfile.ZipFile(self.raw_dataset_path, "r") as zip_file:
            file_names = zip_file.namelist()

            if "OriginalImages/" not in file_names[0]:
                raise ValueError(
                    "Unexpected zip file structure. "
                    "Expected files to be under 'OriginalImages/' directory."
                )

            for file_name in file_names:
                base_file_name = os.path.basename(file_name)

                if base_file_name in train_file_names:
                    output_path = os.path.join(train_image_folder, base_file_name)
                elif base_file_name in val_file_names:
                    output_path = os.path.join(val_image_folder, base_file_name)
                elif base_file_name in test_file_names:
                    output_path = os.path.join(test_image_folder, base_file_name)
                else:
                    continue  # skip files that are not in any split

                with zip_file.open(file_name) as source_file:
                    with open(output_path, "wb") as target_file:
                        target_file.write(source_file.read())

        train_dataframe["image_path"] = train_dataframe["fnames"].apply(
            lambda x: os.path.join(train_image_folder, x)
        )
        val_dataframe["image_path"] = val_dataframe["fnames"].apply(
            lambda x: os.path.join(val_image_folder, x)
        )
        test_dataframe["image_path"] = test_dataframe["fnames"].apply(
            lambda x: os.path.join(test_image_folder, x)
        )

        train_dataframe.to_csv(train_csv_path, index=False)
        val_dataframe.to_csv(val_csv_path, index=False)
        test_dataframe.to_csv(test_csv_path, index=False)

        self.train_size = len(train_dataframe)
        self.val_size = len(val_dataframe)
        self.test_size = len(test_dataframe)

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
        dr_severity_to_text = {
            0: "no",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }

        amd_severity_to_text = {
            0: "no",
            1: "early/intermediate",
            2: "advanced dry",
            3: "advanced wet",
        }

        df["normal"] = df["normal"].apply(lambda x: "normal" if x == 1 else "diseased")
        df["DR"] = df["DR"].apply(lambda x: dr_severity_to_text.get(x, "unknown"))
        df["AMD"] = df["AMD"].apply(lambda x: amd_severity_to_text.get(x, "unknown"))

        df["glaucoma"] = df["glaucoma"].apply(
            lambda x: "glaucoma" if x == 1 else "no glaucoma"
        )
        df["myopia"] = df["myopia"].apply(lambda x: "myopia" if x == 1 else "no myopia")
        df["RVO"] = df["RVO"].apply(
            lambda x: "retinal vein occlusion"
            if x == 1
            else "no retinal vein occlusion"
        )
        df["LS"] = df["LS"].apply(
            lambda x: "laser scars" if x == 1 else "no laser scars"
        )
        df["hyper"] = df["hyper"].apply(
            lambda x: "hypertensive retinopathy"
            if x == 1
            else "no hypertensive retinopathy"
        )
        df["others"] = df["others"].apply(
            lambda x: "other diseases" if x == 1 else "no other diseases"
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
        return [
            "normal",
            "DR",
            "AMD",
            "glaucoma",
            "myopia",
            "RVO",
            "LS",
            "hyper",
            "others",
        ]

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value for the given column.

        """
        if column_name == "normal":
            return f"A fundus image showing a {value} eye."

        if column_name == "DR":
            return f"A fundus image showing {value} diabetic retinopathy."
        if column_name == "AMD":
            return f"A fundus image showing {value} age-related macular degeneration."

        if column_name in [
            "glaucoma",
            "myopia",
            "RVO",
            "LS",
            "hyper",
            "others",
        ]:
            return f"A fundus image showing {value}."
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
        return {
            "normal": ["normal", "diseased"],
            "DR": ["no", "mild", "moderate", "severe", "proliferative"],
            "AMD": [
                "no",
                "early/intermediate",
                "advanced dry",
                "advanced wet",
            ],
            "glaucoma": ["glaucoma", "no glaucoma"],
            "myopia": ["myopia", "no myopia"],
            "RVO": ["retinal vein occlusion", "no retinal vein occlusion"],
            "LS": ["laser scars", "no laser scars"],
            "hyper": [
                "hypertensive retinopathy",
                "no hypertensive retinopathy",
            ],
            "others": ["other diseases", "no other diseases"],
        }
