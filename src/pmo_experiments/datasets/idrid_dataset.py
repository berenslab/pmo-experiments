"""
IDRiD dataset integration.

This module provides dataset adapters for the IDRiD dataset in two variants:

- Disease grading (DR grade + DME risk)
- Lesion segmentation (presence/absence of lesions with optional mask paths)

Both adapters convert the raw IDRiD ZIP archives into an OpenCLIP-compatible
folder layout:

- `images/` containing extracted JPEG images
- `train.csv` and `test.csv` with at least `image_path` and `caption` columns

Notes:
        - This implementation currently does not create a validation split and sets
            `val_size = 0`.
        - The raw IDRiD archives are expected to be present under `DATASET_PATH`
            with filenames `A. Segmentation.zip` and/or `B. Disease Grading.zip`.

"""

import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset


def disease_grading_row_to_text(row: pd.Series) -> str:
    """
    Create a natural-language caption from a disease grading label row.

    Args:
        row (pd.Series): Row with at least the keys `level` (int) and
            `dme_risk` (int).

    Returns:
        str: A caption describing DR severity and DME risk.

    """
    level_map = {
        0: "no diabetic retinopathy",
        1: "mild diabetic retinopathy",
        2: "moderate diabetic retinopathy",
        3: "severe diabetic retinopathy",
        4: "proliferative diabetic retinopathy",
    }

    dme_risk_map = {
        0: "no apparent hard exudates",
        1: (
            "the presence of hard exudates outside the radius of one disc diameter "
            "from the macula center"
        ),
        2: (
            "the presence of hard exudates within the radius of one disc diameter "
            "from the macula center"
        ),
    }

    text = (
        f"A color fundus photograph showing {level_map[int(row['level'])]} "
        f"and {dme_risk_map[int(row['dme_risk'])]}."
    )  # TODO use pandas typing lib

    return text


def get_segmentation_biomarker_map() -> dict[str, str]:
    """
    Return a dictionary mapping lesion abbreviations to full names.

    Returns:
        dict[str, str]: A dictionary mapping lesion abbreviations to full names.

    """
    return {
        "MA": "microaneurysms",
        "HE": "hemorrhages",
        "EX": "hard exudates",
        "SE": "soft exudates",
    }


def segmentation_row_to_text(row: pd.Series) -> str:
    """
    Create a natural-language caption from a segmentation record row.

    The caption summarizes which lesion types are present according to the
    binary indicator columns.

    Args:
        row (pd.Series): Row with binary indicator columns for lesion types.
            Expected keys include `MA`, `HE`, `EX`, and `SE`.

    Returns:
        str: A caption describing which findings are present.

    """
    text = "A color fundus photograph showing"

    findings = []

    for lesion_type, lesion_name in get_segmentation_biomarker_map().items():
        if row[lesion_type] == 1:
            findings.append(lesion_name)

    if len(findings) == 0:
        text += " no findings."
    elif len(findings) == 1:
        text += f" {findings[0]}."
    else:
        text += " " + ", ".join(findings[:-1]) + f", and {findings[-1]}."

    return text


def combine_masks(mask_paths: list[str | None]) -> np.ndarray | None:
    """
    Combine multiple segmentation masks into a single mask.

    Args:
        mask_paths (list[str | None]): List of paths to mask files.
            None values are ignored.

    Returns:
        np.ndarray | None: Combined mask as a 2D numpy array where each pixel
            contains the maximum value across all masks at that position.
            For binary masks, this is equivalent to a logical OR operation.
            Returns None if no valid masks are provided.

    """
    valid_masks = [
        path for path in mask_paths if path is not None and os.path.exists(path)
    ]

    if not valid_masks:
        return None

    # Load the first mask to initialize the combined mask
    with Image.open(valid_masks[0]) as img:
        combined = np.array(img, dtype=np.uint8)

    # Combine remaining masks using logical OR
    for mask_path in valid_masks[1:]:
        with Image.open(mask_path) as img:
            mask = np.array(img, dtype=np.uint8)
            combined = np.maximum(combined, mask)

    return combined


class IDRiDDatasetBase(BaseDataset):
    """
    Common base class for IDRiD dataset adapters.

    This stores the raw dataset path and tracks dataset sizes after conversion.

    Attributes:
        raw_dataset_path (str): Root path of the raw IDRiD dataset.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.

    """

    def __init__(self, **dataset_args):
        """
        Initialize the base IDRiD dataset adapter.

        Args:
            **dataset_args: Dataset configuration dictionary.

        """
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/idrid")

        self.train_size = None
        self.val_size = None

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


@DATASET_REGISTRY.register("idrid_disease_grading")
class IDRiDDiseaseGradingDataset(IDRiDDatasetBase):
    """
    IDRiD disease grading dataset adapter.

    Converts the IDRiD disease grading archive (`B. Disease Grading.zip`) into a
    folder containing extracted images and CSV files.

    The generated CSVs contain:
        - `image_path` (str): Path to the extracted image
        - `level` (int): Diabetic retinopathy grade (0-4)
        - `dme_risk` (int): DME risk label (0-2)
        - `caption` (str): Text description derived from `level` and `dme_risk`

    Attributes:
        raw_dataset_path (str): Root path of the raw IDRiD dataset.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.
        converted_dataset_path (str): Output folder for the converted dataset.
        disease_grading_zip_path (str): Path to the raw disease grading ZIP archive.

    """

    def __init__(self, **dataset_args):
        """
        Initialize the IDRiD disease grading dataset adapter.

        Args:
            **dataset_args: Dataset configuration dictionary.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/idrid_disease_grading/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/idrid_disease_grading/webdataset"
        )

        self.disease_grading_zip_path = os.path.join(
            self.raw_dataset_path, "B. Disease Grading.zip"
        )

    def to_open_clip_format(self):
        """
        Convert the disease grading dataset into OpenCLIP CSV + image format.

        The conversion writes:
            - `train.csv`
            - `test.csv`
            - extracted JPEG images under `images/`

        If the converted CSVs already exist, the method is a no-op besides
        setting `train_size`/`val_size`.

        """
        # Unzip to temporary directories
        disease_grading_image_path = os.path.join(self.converted_dataset_path, "images")
        disease_grading_test_csv_path = os.path.join(
            self.converted_dataset_path, "test.csv"
        )
        disease_grading_train_csv_path = os.path.join(
            self.converted_dataset_path, "train.csv"
        )

        if all(
            [
                os.path.exists(p)
                for p in [
                    disease_grading_train_csv_path,
                    disease_grading_test_csv_path,
                ]
            ]
        ):
            # TODO val split?

            self.val_size = 0
            self.train_size = len(pd.read_csv(disease_grading_train_csv_path))
            return

        assert os.path.exists(self.disease_grading_zip_path), (
            "Disease grading zip file not found."
        )

        # Process disease grading data
        os.makedirs(disease_grading_image_path, exist_ok=True)

        with zipfile.ZipFile(self.disease_grading_zip_path, "r") as seg_zip:
            # Get all training image names
            full_name_list = seg_zip.namelist()

            # Open train csv in dataframe

            with seg_zip.open(
                "B. Disease Grading/2. Groundtruths/"
                "a. IDRiD_Disease Grading_Training Labels.csv"
            ) as train_csv_file:
                train_df = pd.read_csv(train_csv_file)

                # Rename Retinopathy grade to level
                train_df = train_df.rename(
                    columns={
                        "Retinopathy grade": "level",
                        "Image name": "image_path",
                        "Risk of macular edema ": "dme_risk",
                    }
                )

                train_df["image_path"] = train_df["image_path"].apply(
                    lambda x: os.path.join(
                        disease_grading_image_path, "train_" + x + ".jpg"
                    )
                )

                # Drop all other columns
                train_df = train_df[["image_path", "level", "dme_risk"]]
                train_df["caption"] = train_df.apply(
                    disease_grading_row_to_text,
                    axis=1,
                )
                train_df.to_csv(disease_grading_train_csv_path, index=False)

            with seg_zip.open(
                "B. Disease Grading/2. Groundtruths/"
                "b. IDRiD_Disease Grading_Testing Labels.csv"
            ) as test_csv_file:
                test_df = pd.read_csv(test_csv_file)

                # Rename Retinopathy grade to level
                test_df = test_df.rename(
                    columns={
                        "Retinopathy grade": "level",
                        "Image name": "image_path",
                        "Risk of macular edema ": "dme_risk",
                    }
                )

                test_df["image_path"] = test_df["image_path"].apply(
                    lambda x: os.path.join(
                        disease_grading_image_path, "test_" + x + ".jpg"
                    )
                )

                # Drop all other columns
                test_df = test_df[["image_path", "level", "dme_risk"]]
                test_df["caption"] = test_df.apply(
                    disease_grading_row_to_text,
                    axis=1,
                )

                test_df.to_csv(disease_grading_test_csv_path, index=False)

            # Extract images
            for file_name in full_name_list:
                if file_name.startswith(
                    "B. Disease Grading/1. Original Images/a. Training Set/"
                ):
                    if not file_name.endswith(".jpg"):
                        continue
                    # Extract image
                    output_file_path = os.path.join(
                        disease_grading_image_path,
                        "train_" + os.path.basename(file_name),
                    )
                    data = seg_zip.read(file_name)
                    with open(output_file_path, "wb") as out_file:
                        out_file.write(data)

                elif file_name.startswith(
                    "B. Disease Grading/1. Original Images/b. Testing Set/"
                ):
                    if not file_name.endswith(".jpg"):
                        continue

                    # Extract image
                    output_file_path = os.path.join(
                        disease_grading_image_path,
                        "test_" + os.path.basename(file_name),
                    )
                    data = seg_zip.read(file_name)
                    with open(output_file_path, "wb") as out_file:
                        out_file.write(data)

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe for evaluation by converting codes to text labels.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
            split (str): The dataset split name (unused).

        Returns:
            pd.DataFrame: The preprocessed dataframe with text labels for level
                and DME risk.

        """
        severity_to_text = {
            0: "no",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }

        df["level"] = df["level"].apply(
            lambda x: severity_to_text.get(int(x), None) if pd.notna(x) else None
        )
        df["dme_risk"] = df["dme_risk"].apply(
            lambda x: None
            if pd.isna(x)
            else f"diabetic macular edema stage {int(x)}"
            if int(x) > 0
            else "no diabetic macular edema"
        )

        for col in IDRiDDiseaseGradingDataset.get_eval_target_column_names():
            df[col] = df[col].fillna("Unknown").astype(str)

        return df

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Get the column names for the evaluation targets.

        Returns:
            list[str]: List of evaluation target column names.

        """
        return [
            "level",
            "dme_risk",
        ]

    def get_eval_target_values(
        self,
    ) -> dict[str, list[str] | None] | None:
        """
        Get the possible evaluation target values.

        Returns:
            dict[str, list[str]]: Dictionary mapping target column names to lists of
                possible text values.

        """
        severity_to_text = {
            0: "no",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }

        dme_risk_to_text = {
            0: "no diabetic macular edema",
            1: "diabetic macular edema stage 1",
            2: "diabetic macular edema stage 2",
        }

        level_values = [severity_to_text[i] for i in range(5)]
        dme_risk_values = [dme_risk_to_text[i] for i in range(3)]

        return {
            "level": level_values,
            "dme_risk": dme_risk_values,
        }

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name == "level":
            return "A fundus image showing {severity} diabetic retinopathy.".format(
                severity=value
            )

        if column_name == "dme_risk":
            dme_risk_map = {
                0: "no apparent hard exudates",
                1: (
                    "the presence of hard exudates outside the radius of one "
                    "disc diameter from the macula center"
                ),
                2: (
                    "the presence of hard exudates within the radius of one disc "
                    "diameter from the macula center"
                ),
            }

            dme_risk_value = None

            if "no diabetic macular edema" in value:
                dme_risk_value = 0

            elif "diabetic macular edema stage 1" in value:
                dme_risk_value = 1
            elif "diabetic macular edema stage 2" in value:
                dme_risk_value = 2
            else:
                raise ValueError(f"Unknown dme risk value: {value}")

            return "A fundus image showing {dme_risk}.".format(
                dme_risk=dme_risk_map[dme_risk_value]
            )
        return value


@DATASET_REGISTRY.register("idrid_segmentation")
class IDRiDSegmentationDataset(IDRiDDatasetBase):
    """
    IDRiD lesion segmentation dataset adapter.

    Converts the IDRiD segmentation archive (`A. Segmentation.zip`) into a folder
    containing extracted images, mask files, and CSV files.

    The generated CSVs contain:
        - `image_path` (str): Path to the extracted image
        - `mask_<TYPE>` (str | None): Path to a mask file for a lesion type
        - `<TYPE>` (int): Binary indicator for lesion presence (0/1)
        - `caption` (str): Text summary derived from the indicator columns

    Where `<TYPE>` is one of `MA`, `HE`, `EX`, `SE`, or `OD`.

    Attributes:
        raw_dataset_path (str): Root path of the raw IDRiD dataset.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.
        converted_dataset_path (str): Output folder for the converted dataset.
        segmentation_zip_path (str): Path to the raw segmentation ZIP archive.

    """

    def __init__(self, **dataset_args):
        """
        Initialize the IDRiD segmentation dataset adapter.

        Args:
            **dataset_args: Dataset configuration dictionary.

        """
        super().__init__(**dataset_args)

        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/idrid_segmentation/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/idrid_segmentation/webdataset"
        )

        self.segmentation_zip_path = os.path.join(
            self.raw_dataset_path, "A. Segmentation.zip"
        )

    def to_open_clip_format(self):
        """
        Convert the segmentation dataset into OpenCLIP CSV + image format.

        The conversion writes:
            - `train.csv`
            - `test.csv`
            - extracted JPEG images under `images/`
            - extracted mask files under `masks/`

        If the converted CSVs already exist, the method is a no-op besides
        setting `train_size`/`val_size`.

        """
        # Unzip to temporary directories

        segmentation_image_path = os.path.join(self.converted_dataset_path, "images")
        segmentation_mask_path = os.path.join(self.converted_dataset_path, "masks")
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        test_csv_path = os.path.join(self.converted_dataset_path, "test.csv")

        if all(
            [
                os.path.exists(p)
                for p in [
                    train_csv_path,
                    test_csv_path,
                ]
            ]
        ):
            # TODO val split?
            self.val_size = 0
            self.train_size = len(pd.read_csv(train_csv_path))
            return

        assert os.path.exists(self.segmentation_zip_path), (
            "Segmentation zip file not found."
        )

        os.makedirs(segmentation_image_path, exist_ok=True)
        os.makedirs(segmentation_mask_path, exist_ok=True)

        # Process segmentation data
        with zipfile.ZipFile(self.segmentation_zip_path, "r") as seg_zip:
            # Get all training image names
            full_name_list = seg_zip.namelist()

            training_images = []
            test_images = []

            training_masks = {
                "MA": [],
                "HE": [],
                "EX": [],
                "SE": [],
                "OD": [],
            }

            test_masks = {
                "MA": [],
                "HE": [],
                "EX": [],
                "SE": [],
                "OD": [],
            }

            for file_name in full_name_list:
                if file_name.startswith(
                    "A. Segmentation/1. Original Images/a. Training Set/"
                ):
                    if not file_name.endswith(".jpg"):
                        continue

                    training_images.append(os.path.basename(file_name))

                    # Extract image
                    output_file_path = os.path.join(
                        segmentation_image_path, "train_" + os.path.basename(file_name)
                    )
                    data = seg_zip.read(file_name)
                    with open(output_file_path, "wb") as out_file:
                        out_file.write(data)

                elif file_name.startswith(
                    "A. Segmentation/1. Original Images/b. Testing Set/"
                ):
                    if not file_name.endswith(".jpg"):
                        continue

                    test_images.append(os.path.basename(file_name))

                    # Extract image
                    output_file_path = os.path.join(
                        segmentation_image_path, "test_" + os.path.basename(file_name)
                    )
                    data = seg_zip.read(file_name)
                    with open(output_file_path, "wb") as out_file:
                        out_file.write(data)
                elif file_name.startswith(
                    "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/"
                ):
                    if not file_name.endswith(".tif"):
                        continue
                    for mask_type, mask_type_list in training_masks.items():
                        if f"_{mask_type}.tif" in file_name:
                            mask_type_list.append(os.path.basename(file_name))

                        # Extract mask
                        output_file_path = os.path.join(
                            segmentation_mask_path,
                            "train_" + os.path.basename(file_name),
                        )
                        data = seg_zip.read(file_name)
                        with open(output_file_path, "wb") as out_file:
                            out_file.write(data)

                elif file_name.startswith(
                    "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/"
                ):
                    if not file_name.endswith(".tif"):
                        continue
                    for mask_type, mask_type_list in test_masks.items():
                        if f"_{mask_type}.tif" in file_name:
                            mask_type_list.append(os.path.basename(file_name))

                        # Extract mask
                        output_file_path = os.path.join(
                            segmentation_mask_path,
                            "test_" + os.path.basename(file_name),
                        )
                        data = seg_zip.read(file_name)
                        with open(output_file_path, "wb") as out_file:
                            out_file.write(data)

            # Create dataframes
            train_records = []

            for train_file_name in training_images:
                base_name = (
                    os.path.basename(train_file_name)
                    .replace(".jpg", "")
                    .replace(".tif", "")
                )
                record: dict[str, str | int | None] = {
                    "image_path": os.path.join(
                        segmentation_image_path,
                        "train_" + os.path.basename(train_file_name),
                    )
                }

                mask_paths = []
                disease_mask_paths = []  # Only disease masks for total
                for mask_type, mask_type_list in training_masks.items():
                    mask_file_name = f"{base_name}_{mask_type}.tif"

                    if mask_file_name in mask_type_list:
                        mask_path = os.path.join(
                            segmentation_mask_path, "train_" + mask_file_name
                        )
                        record[f"mask_{mask_type}"] = mask_path
                        record[mask_type] = 1
                        mask_paths.append(mask_path)
                        # Only include disease masks (not OD) in total
                        if mask_type != "OD":
                            disease_mask_paths.append(mask_path)
                    else:
                        record[f"mask_{mask_type}"] = None
                        record[mask_type] = 0
                        mask_paths.append(None)

                # Create and save the total mask (only disease masks)
                combined_mask = combine_masks(disease_mask_paths)
                if combined_mask is not None:
                    total_mask_filename = f"train_{base_name}_total.tif"
                    total_mask_path = os.path.join(
                        segmentation_mask_path, total_mask_filename
                    )
                    Image.fromarray(combined_mask).save(total_mask_path, format="TIFF")
                    record["mask_total"] = total_mask_path
                else:
                    record["mask_total"] = None

                train_records.append(record)

            test_records = []
            for test_file_name in test_images:
                base_name = (
                    os.path.basename(test_file_name)
                    .replace(".jpg", "")
                    .replace(".tif", "")
                )
                record: dict[str, str | int | None] = {
                    "image_path": os.path.join(
                        segmentation_image_path,
                        "test_" + os.path.basename(test_file_name),
                    )
                }

                mask_paths = []
                disease_mask_paths = []  # Only disease masks for total
                for mask_type, mask_type_list in test_masks.items():
                    mask_file_name = f"{base_name}_{mask_type}.tif"

                    if mask_file_name in mask_type_list:
                        mask_path = os.path.join(
                            segmentation_mask_path, "test_" + mask_file_name
                        )
                        record[f"mask_{mask_type}"] = mask_path
                        record[mask_type] = 1
                        mask_paths.append(mask_path)
                        # Only include disease masks (not OD) in total
                        if mask_type != "OD":
                            disease_mask_paths.append(mask_path)
                    else:
                        record[f"mask_{mask_type}"] = None
                        record[mask_type] = 0
                        mask_paths.append(None)

                # Create and save the total mask (only disease masks)
                combined_mask = combine_masks(disease_mask_paths)
                if combined_mask is not None:
                    total_mask_filename = f"test_{base_name}_total.tif"
                    total_mask_path = os.path.join(
                        segmentation_mask_path, total_mask_filename
                    )
                    Image.fromarray(combined_mask).save(total_mask_path, format="TIFF")
                    record["mask_total"] = total_mask_path
                else:
                    record["mask_total"] = None

                test_records.append(record)

            train_df = pd.DataFrame.from_records(train_records)
            train_df["caption"] = train_df.apply(
                segmentation_row_to_text,
                axis=1,
            )
            train_df.to_csv(train_csv_path, index=False)

            test_df = pd.DataFrame.from_records(test_records)
            test_df["caption"] = test_df.apply(
                segmentation_row_to_text,
                axis=1,
            )
            test_df.to_csv(test_csv_path, index=False)

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe for evaluation by converting indicator codes.

        Converts binary indicator codes to text labels for lesion presence.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
            split (str): The dataset split name (unused).

        Returns:
            pd.DataFrame: The preprocessed dataframe with text labels for
                lesion presence.

        """
        biomarker_map = get_segmentation_biomarker_map()
        biomarker_map["OD"] = "optic disc"
        for col in IDRiDSegmentationDataset.get_eval_target_column_names():
            current_biomarker_text = biomarker_map[col]
            df[col] = (
                df[col]
                .apply(
                    lambda x: "no " + current_biomarker_text
                    if x == 0
                    else current_biomarker_text
                    if pd.notna(x)
                    else None
                )
                .fillna("Unknown")
                .astype(str)
            )

        return df

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Get the column names for evaluation targeting.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of lesion type column names for evaluation.

        """
        return ["MA", "HE", "EX", "SE", "OD"]

    def get_eval_target_values(
        self,
    ) -> dict[str, list[str] | None] | None:
        """
        Get the possible evaluation target values for each lesion type.

        Returns:
            dict[str, list[str]]: Dictionary mapping lesion types to lists of
                possible text values (presence/absence).

        """
        biomarker_map = {
            key: ["no " + name, name]
            for key, name in get_segmentation_biomarker_map().items()
        }

        biomarker_map["OD"] = ["no optic disc", "optic disc"]

        # TODO weird pyright error
        return biomarker_map  # pyright: ignore[reportReturnType]

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name in IDRiDSegmentationDataset.get_eval_target_column_names():
            return f"A fundus image showing {value}."
        return value


@DATASET_REGISTRY.register("idrid_segmentation_dr")
class IDRiDSegmentationDRDataset(IDRiDSegmentationDataset):
    """
    IDRiD segmentation dataset adapter focused on DR severity classification.

    This is a variant of the segmentation dataset that overrides captions to only
    describe DR severity rather than individual lesion presence.

    """

    def __init__(self, **dataset_args):
        """
        Initialize the IDRiD segmentation DR dataset adapter.

        Args:
            **dataset_args: Dataset configuration dictionary.

        """
        converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/idrid_segmentation_dr/converted"
        )
        dataset_args["CONVERTED_DATASET_PATH"] = converted_dataset_path
        if "WEBDATASET_PATH" not in dataset_args:
            dataset_args["WEBDATASET_PATH"] = (
                "datasets/idrid_segmentation_dr/webdataset"
            )
        super().__init__(**dataset_args)

    def to_open_clip_format(self):
        """
        Convert the segmentation dataset and override captions for DR-focused training.

        Calls the parent class conversion then updates captions to focus on DR severity
        instead of individual lesion presence.

        """
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        test_csv_path = os.path.join(self.converted_dataset_path, "test.csv")

        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
            super().to_open_clip_format()

            # Update captions to only include DR severity
            train_df = pd.read_csv(train_csv_path)
            train_df["caption"] = (
                "A fundus image showing proliferative diabetic retinopathy."
            )
            train_df.to_csv(train_csv_path, index=False)

            test_df = pd.read_csv(test_csv_path)
            test_df["caption"] = (
                "A fundus image showing proliferative diabetic retinopathy."
            )
            test_df.to_csv(test_csv_path, index=False)

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Raise NotImplementedError for DR-only segmentation dataset.

        Evaluation preprocessing is not implemented for this variant.

        Args:
            df (pd.DataFrame): The dataframe to preprocess.
            split (str): The dataset split name.

        Raises:
            NotImplementedError: Always raised as this is not supported.

        """
        raise NotImplementedError(
            "Evaluation preprocessing is not implemented for the DR-only "
            "segmentation dataset."
        )

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Get the column names for evaluation targeting.

        Evaluation target column names are not defined for this variant.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Raises:
            NotImplementedError: Always raised as this is not supported.

        """
        raise NotImplementedError(
            "Evaluation target column names are not defined for the DR-only "
            "segmentation dataset."
        )

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        This method is not implemented for the DR-only segmentation dataset as there
        are no defined evaluation targets.

        Args:
            column_name (str): The name of the target column.
            value (str): The value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        raise NotImplementedError(
            "Value verbalization is not implemented for the DR-only segmentation "
            "dataset."
        )
