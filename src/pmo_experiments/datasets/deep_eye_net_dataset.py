"""Integration of the DeepEyeNet dataset."""

import json
import os

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register("deepeyenet")
class DeepEyeNetDataset(BaseDataset):
    """DeepEyeNet dataset integration."""

    def __init__(self, **dataset_args: str):
        """
        Initialize DeepEyeNet dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization. Uses 'DATASET_PATH' for the raw dataset location and
                'CONVERTED_DATASET_PATH' for the converted dataset location.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/deepeyenet")
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/deepeyenet/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/deepeyenet/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.train_size = None
        self.val_size = None

    def _convert_split_to_csv(self, dataset_path: str, output_path: str) -> int:
        """
        Convert a dataset split to CSV.

        Args:
            dataset_path (str): Path of the JSON file.
            output_path (str): Path of the output CSV file.

        Returns:
            int: Number of records processed.

        """
        with open(dataset_path) as f:
            data = json.load(f)

        records = []
        for record in data:
            for image_name, caption_data in record.items():
                keywords = caption_data["keywords"].split(",") + [
                    caption_data["clinical-description"]
                ]

                records.extend(
                    [
                        {
                            "image_path": os.path.join(
                                self.raw_dataset_path, image_name
                            ),
                            "caption": caption.strip(),
                        }
                        for caption in keywords
                        if caption.strip() != ""
                    ]
                )

        df = pd.DataFrame.from_records(records)
        df.to_csv(output_path, index=False)

        return len(records)

    def to_open_clip_format(self) -> None:
        """Convert the dataset to OpenCLIP format."""
        train_dataset_path = os.path.join(
            self.raw_dataset_path, "DeepEyeNet_train.json"
        )
        val_dataset_path = os.path.join(self.raw_dataset_path, "DeepEyeNet_valid.json")

        test_dataset_path = os.path.join(self.raw_dataset_path, "DeepEyeNet_test.json")

        train_dataset_output_path = os.path.join(
            self.converted_dataset_path, "train.csv"
        )
        val_dataset_output_path = os.path.join(self.converted_dataset_path, "val.csv")
        test_dataset_output_path = os.path.join(self.converted_dataset_path, "test.csv")

        if not os.path.exists(self.converted_dataset_path):
            os.makedirs(self.converted_dataset_path)

        if not os.path.exists(train_dataset_output_path):
            self.train_size = self._convert_split_to_csv(
                train_dataset_path,
                train_dataset_output_path,
            )
        else:
            self.train_size = sum(
                len(chunk)
                for chunk in pd.read_csv(
                    train_dataset_output_path,
                    usecols=["image_path"],
                    dtype={"image_path": str},
                    chunksize=100000,
                )
            )

        if not os.path.exists(val_dataset_output_path):
            self.val_size = self._convert_split_to_csv(
                val_dataset_path,
                val_dataset_output_path,
            )
        else:
            self.val_size = sum(
                len(chunk)
                for chunk in pd.read_csv(
                    val_dataset_output_path,
                    usecols=["image_path"],
                    dtype={"image_path": str},
                    chunksize=100000,
                )
            )

        if not os.path.exists(test_dataset_output_path):
            self._convert_split_to_csv(
                test_dataset_path,
                test_dataset_output_path,
            )

        if not os.path.exists(self.all_target_value_paths):
            all_target_values = {
                "train": sorted(
                    pd.read_csv(
                        train_dataset_output_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]  # .apply(lambda x: x.strip().removesuffix("."))
                    .unique()
                    .tolist()
                ),
                "val": sorted(
                    pd.read_csv(
                        val_dataset_output_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]  # .apply(lambda x: x.strip().removesuffix("."))
                    .unique()
                    .tolist()
                ),
                "test": sorted(
                    pd.read_csv(
                        test_dataset_output_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]  # .apply(lambda x: x.strip().removesuffix("."))
                    .unique()
                    .tolist()
                ),
            }

            with open(self.all_target_value_paths, "w") as f:
                json.dump(all_target_values, f, indent=4, ensure_ascii=False)

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

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe for the specified split for evaluation.

        For DeepEyeNet, no specific preprocessing is needed for evaluation.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
            split (str): The split name (e.g., 'train', 'val', 'test').

        Returns:
            pd.DataFrame: The preprocessed dataframe, which is the same as the input
                dataframe for DeepEyeNet.

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
            list[str]: List of target column names for evaluation.

        """
        return ["caption"]
