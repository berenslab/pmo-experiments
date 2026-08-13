"""Interface for the REFUGE2 dataset (glaucoma detection)."""

import logging
import os
import zipfile

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("refuge2")
class Refuge2Dataset(BaseDataset):
    """Dataset class for the REFUGE2 dataset (glaucoma detection)."""

    def __init__(self, **dataset_args: str):
        """
        Initialize REFUGE2 dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get(
            "DATASET_PATH", "datasets/refuge2/REFUGE2.zip"
        )

        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/refuge2/converted"
        )

        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/refuge2/webdataset"
        )

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
                "Train, validation and test CSV files already exist. Skipping."
            )
            self._get_dataset_sizes()
            return

        assert os.path.exists(self.raw_dataset_path), (
            f"Refuge2 dataset not found at {self.raw_dataset_path}."
        )

        # Open the zip file
        train_image_folder = os.path.join(self.converted_dataset_path, "train")
        test_image_folder = os.path.join(self.converted_dataset_path, "test")
        val_image_folder = os.path.join(self.converted_dataset_path, "val")
        os.makedirs(self.converted_dataset_path, exist_ok=True)
        os.makedirs(train_image_folder, exist_ok=True)
        os.makedirs(test_image_folder, exist_ok=True)
        os.makedirs(val_image_folder, exist_ok=True)

        with zipfile.ZipFile(self.raw_dataset_path, "r") as zip_file:
            file_names = zip_file.namelist()

            if "REFUGE2/" not in file_names:
                raise ValueError("REFUGE2 folder not found in the zip file.")

            # Check for test files
            if "REFUGE2/Test/" not in file_names:
                raise ValueError("Test folder not found in the zip file.")

            if "REFUGE2/Test/refuge2-test/" not in file_names:
                raise ValueError("Test images not found in the zip file.")

            if "REFUGE2/Test/task1.xls" not in file_names:
                raise ValueError("Test labels not found in the zip file.")

            test_label_df = pd.read_excel(
                zip_file.open("REFUGE2/Test/task1.xls"), header=None
            )
            test_label_df.columns = ["file_name", "glaucoma_risk"]

            # Copy images
            for file_name in test_label_df["file_name"].tolist():
                zip_file_name = os.path.join("REFUGE2/Test/refuge2-test", file_name)
                if zip_file_name not in file_names:
                    raise ValueError(
                        f"Test image {file_name} not found in the zip file."
                    )

                # Extract the image and save to the converted dataset path
                with zip_file.open(zip_file_name) as image_file:
                    image_data = image_file.read()
                    output_file_name = os.path.join(test_image_folder, file_name)
                    with open(output_file_name, "wb") as output_file:
                        output_file.write(image_data)

            # Check for validation files
            if "REFUGE2/Validation/" not in file_names:
                raise ValueError("Validation folder not found in the zip file.")

            if "REFUGE2/Validation/Images/" not in file_names:
                raise ValueError("Validation images not found in the zip file.")

            if "REFUGE2/Validation/glaucoma.csv" not in file_names:
                raise ValueError("Validation labels not found in the zip file.")

            validation_label_df = pd.read_csv(
                zip_file.open("REFUGE2/Validation/glaucoma.csv")
            )
            validation_label_df = validation_label_df.rename(
                columns={"FileName": "file_name", "Glaucoma Risk": "glaucoma_risk"}
            )

            # Copy validation images
            for file_name in validation_label_df["file_name"].tolist():
                zip_file_name = os.path.join("REFUGE2/Validation/Images", file_name)
                if zip_file_name not in file_names:
                    raise ValueError(
                        f"Validation image {file_name} not found in the zip file."
                    )

                # Extract the image and save to the converted dataset path
                with zip_file.open(zip_file_name) as image_file:
                    image_data = image_file.read()
                    output_file_name = os.path.join(val_image_folder, file_name)
                    with open(output_file_name, "wb") as output_file:
                        output_file.write(image_data)

            # Check for training files
            if "REFUGE2/Train/" not in file_names:
                raise ValueError("Train folder not found in the zip file.")

            if "REFUGE2/Train/REFUGE1-test/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            if "REFUGE2/Train/REFUGE1-val/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            if "REFUGE2/Train/REFUGE1-train/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            if (
                "REFUGE2/Train/REFUGE1-test/Glaucoma_label_and_Fovea_location.xlsx"
                not in file_names
            ):
                raise ValueError("Train labels partially not found in the zip file.")

            train_test_label_df = pd.read_excel(
                zip_file.open(
                    "REFUGE2/Train/REFUGE1-test/Glaucoma_label_and_Fovea_location.xlsx"
                ),
            )

            train_test_label_df = train_test_label_df.rename(
                columns={
                    "ImgName": "file_name",
                    "Label(Glaucoma=1)": "glaucoma_risk",
                }
            ).drop(columns=["Fovea_X", "Fovea_Y", "ID"])

            if "REFUGE2/Train/REFUGE1-test/Test400/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            # Copy images
            for file_name in train_test_label_df["file_name"].tolist():
                zip_file_name = os.path.join(
                    "REFUGE2/Train/REFUGE1-test/Test400", file_name
                )
                if zip_file_name not in file_names:
                    raise ValueError(
                        f"Train test image {file_name} not found in the zip file."
                    )

                # Extract the image and save to the converted dataset path
                with zip_file.open(zip_file_name) as image_file:
                    image_data = image_file.read()
                    output_file_name = os.path.join(train_image_folder, file_name)
                    with open(output_file_name, "wb") as output_file:
                        output_file.write(image_data)

            if "REFUGE2/Train/REFUGE1-val/Fovea_locations.xlsx" not in file_names:
                raise ValueError("Train labels partially not found in the zip file.")

            train_val_label_df = pd.read_excel(
                zip_file.open("REFUGE2/Train/REFUGE1-val/Fovea_locations.xlsx"),
            )

            train_val_label_df = train_val_label_df.rename(
                columns={
                    "ImgName": "file_name",
                    "Glaucoma Label": "glaucoma_risk",
                }
            ).drop(columns=["Fovea_X", "Fovea_Y", "ID"])

            if "REFUGE2/Train/REFUGE1-val/REFUGE-Validation400/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            # Copy images
            for file_name in train_val_label_df["file_name"].tolist():
                zip_file_name = os.path.join(
                    "REFUGE2/Train/REFUGE1-val/REFUGE-Validation400", file_name
                )
                if zip_file_name not in file_names:
                    raise ValueError(
                        f"Train val image {file_name} not found in the zip file."
                    )

                # Extract the image and save to the converted dataset path
                with zip_file.open(zip_file_name) as image_file:
                    image_data = image_file.read()
                    output_file_name = os.path.join(train_image_folder, file_name)
                    with open(output_file_name, "wb") as output_file:
                        output_file.write(image_data)

            if "REFUGE2/Train/REFUGE1-train/Fovea_location.xlsx" not in file_names:
                raise ValueError("Train labels partially not found in the zip file.")

            train_train_image_names = pd.read_excel(
                zip_file.open("REFUGE2/Train/REFUGE1-train/Fovea_location.xlsx"),
            )["ImgName"].tolist()

            if "REFUGE2/Train/REFUGE1-train/Training400/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            if "REFUGE2/Train/REFUGE1-train/Training400/Glaucoma/" not in file_names:
                raise ValueError("Train images partially not found in the zip file.")

            if (
                "REFUGE2/Train/REFUGE1-train/Training400/Non-Glaucoma/"
                not in file_names
            ):
                raise ValueError("Train images partially not found in the zip file.")

            train_label_map = {}

            image_file_paths = [
                f
                for f in file_names
                if f.startswith("REFUGE2/Train/REFUGE1-train/Training400/")
                and os.path.basename(f) in set(train_train_image_names)
            ]
            for file_name in image_file_paths:
                output_file_name = os.path.join(
                    train_image_folder,
                    os.path.basename(file_name),
                )

                label = os.path.basename(os.path.dirname(file_name))
                if label == "Glaucoma":
                    glaucoma_risk = 1
                elif label == "Non-Glaucoma":
                    glaucoma_risk = 0
                else:
                    raise ValueError(
                        f"Unexpected label {label} for train train image {file_name}."
                    )

                train_label_map[os.path.basename(file_name)] = glaucoma_risk

                # Extract the image and save to the converted dataset path
                with zip_file.open(file_name) as image_file:
                    image_data = image_file.read()
                    with open(output_file_name, "wb") as output_file:
                        output_file.write(image_data)

            full_train_labels = pd.concat(
                [
                    train_test_label_df,
                    train_val_label_df,
                    pd.DataFrame(
                        {
                            "file_name": list(train_label_map.keys()),
                            "glaucoma_risk": list(train_label_map.values()),
                        }
                    ),
                ],
                ignore_index=True,
            )

            assert set(train_label_map.keys()) == set(train_train_image_names), (
                "Could not resolve all train train image files."
            )

            # Make sure the file names in the train subsets are unique
            assert (
                set(train_test_label_df["file_name"].tolist())
                & set(train_val_label_df["file_name"].tolist())
                == set()
            ), "Train test and train val file names overlap, which should not happen."

            assert (
                set(train_test_label_df["file_name"].tolist())
                & set(train_label_map.keys())
                == set()
            ), "Train test and train train file names overlap, which should not happen."

            assert (
                set(train_val_label_df["file_name"].tolist())
                & set(train_label_map.keys())
                == set()
            ), "Train val and train train file names overlap, which should not happen."

            # TODO do caption processing here

            # Update image paths
            test_label_df["image_path"] = test_label_df["file_name"].apply(
                lambda x: os.path.join(test_image_folder, x)
            )
            validation_label_df["image_path"] = validation_label_df["file_name"].apply(
                lambda x: os.path.join(val_image_folder, x)
            )
            full_train_labels["image_path"] = full_train_labels["file_name"].apply(
                lambda x: os.path.join(train_image_folder, x)
            )

            # Save the labels to CSV files
            test_label_df.to_csv(test_csv_path, index=False)
            validation_label_df.to_csv(val_csv_path, index=False)
            full_train_labels.to_csv(train_csv_path, index=False)

            # Save dataset sizes
            self.train_size = len(full_train_labels)
            self.val_size = len(validation_label_df)
            self.test_size = len(test_label_df)

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
        glaucoma_risk_to_text = {
            0: "no glaucoma",
            1: "glaucoma",
        }

        df["glaucoma_risk"] = df["glaucoma_risk"].apply(
            lambda x: glaucoma_risk_to_text.get(x, str(x))
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
        return ["glaucoma_risk"]

    def get_eval_target_values(
        self,
    ) -> dict[str, list[str] | None] | None:
        """
        Get all possible target values for evaluation.

        Returns:
            dict[str, list[str] | None] | None: A dictionary mapping target column
                names to lists of possible target values, or None.

        """
        return {"glaucoma_risk": ["no glaucoma", "glaucoma"]}

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value for the given column.

        """
        if column_name == "glaucoma_risk":
            return "A fundus image showing {glaucoma_risk}.".format(glaucoma_risk=value)
        else:
            return value
