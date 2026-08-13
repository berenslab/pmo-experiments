"""
BRSET dataset module for diabetic retinopathy classification.

This module provides the BRSETDataset class for converting and processing
the Brazilian Retinal dataset (BRSET) to OpenCLIP compatible format.
"""

import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset


@DATASET_REGISTRY.register("brset")
class BRSETDataset(BaseDataset):
    """
    Dataset class for BRSET retinal fundus images and disease labels.

    Attributes:
        raw_dataset_path (str): Path to the raw BRSET dataset.
        converted_dataset_path (str): Path to the converted BRSET dataset in
            OpenCLIP format.
        train_size (int | None): Number of samples in the training set after conversion.
        val_size (int | None): Number of samples in the validation set after conversion.

    """

    def __init__(self, **dataset_args):
        """
        Initialize the BRSET dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization. Uses 'DATASET_PATH' for the raw dataset location and
                'CONVERTED_DATASET_PATH' for the converted dataset location.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/brset")
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/brset/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/brset/webdataset"
        )

        self.images_path = os.path.join(self.raw_dataset_path, "fundus_photos")

        self.train_size = None
        self.val_size = None

    def to_open_clip_format(self) -> None:
        """Convert BRSET to the open clip compatible format."""
        training_df_path = self.get_train_df_path()
        validation_df_path = self.get_val_df_path()
        test_df_path = self.get_test_df_path()

        os.makedirs(self.converted_dataset_path, exist_ok=True)

        if (
            not os.path.exists(training_df_path)
            or not os.path.exists(validation_df_path)
            or not os.path.exists(test_df_path)
        ):
            # Load the original dataset
            severity_to_text = {
                0: "no",
                1: "mild",
                2: "moderate",
                3: "severe",
                4: "proliferative",
            }

            template_text = "A fundus image showing {severity} diabetic retinopathy."
            base_df = pd.read_csv(
                os.path.join(self.raw_dataset_path, "labels_brset.csv")
            )

            base_df["caption"] = base_df["DR_ICDR"].apply(
                lambda x: template_text.format(severity=severity_to_text[x])
            )

            base_df["image_path"] = base_df["image_id"].apply(
                lambda x: os.path.join(self.images_path, f"{x}.jpg")
            )

            random_seed = 106080954

            # Group by patient ID stratify DR_ICDR severity
            y = base_df["DR_ICDR"]
            groups = base_df["patient_id"]

            stratified_group_kfold = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=random_seed
            )

            # Get first split (train/test)
            train_idx, test_idx = next(stratified_group_kfold.split(base_df, y, groups))

            train_df = base_df.iloc[train_idx]
            test_df = base_df.iloc[test_idx]

            # Stratified split into train and val based on the auxiliary image type
            stratified_shuffle_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.15, random_state=random_seed
            )

            y_train = y.iloc[train_idx]
            sub_train_idx, val_idx = next(
                stratified_shuffle_splitter.split(train_df, y_train)
            )

            val_df = train_df.iloc[val_idx]
            train_df = train_df.iloc[sub_train_idx]

            train_df.to_csv(training_df_path, index=False)
            val_df.to_csv(validation_df_path, index=False)
            test_df.to_csv(test_df_path, index=False)

            self.train_size = len(train_df)
            self.val_size = len(val_df)
        else:
            # Load the already converted datasets to compute sizes
            self.train_size = len(pd.read_csv(training_df_path))
            self.val_size = len(pd.read_csv(validation_df_path))

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

    @staticmethod
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Get a list evaluation target columns.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of evaluation target column names.

        """
        return [
            "patient_sex",
            "exam_eye",
            "diabetes",
            "optic_disc",
            "vessels",
            "macula",
            "DR_ICDR",
            "image_field",
            "diabetic_retinopathy",
            "macular_edema",
            "scar",
            "nevus",
            "amd",
            "vascular_occlusion",
            "hypertensive_retinopathy",
            "drusens",
            "hemorrhage",
            "retinal_detachment",
            "myopic_fundus",
            "increased_cup_disc",
            "other",
            "quality",
        ]

    def preprocess_split_df_for_evaluation(
        self,
        df: pd.DataFrame,
        split: str,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Preprocess a df for the selected split.

        Args:
            df (pd.DataFrame): Dataframe to preprocess.
            split (str): Current split (e.g., 'train', 'val', 'test').

        Returns:
            pd.DataFrame: Preprocessed dataframe.

        """
        severity_to_text = {
            0: "no",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }
        df["patient_sex"] = df["patient_sex"].apply(
            lambda x: "male" if x == 1 else "female"
        )
        df["exam_eye"] = df["exam_eye"].apply(lambda x: "right" if x == 1 else "left")
        df["image_field"] = df["image_field"].apply(lambda x: f"field {x}")

        df["diabetes"] = df["diabetes"].apply(
            lambda x: "diabetic" if x.lower() == "yes" else "non-diabetic"
        )

        df["optic_disc"] = df["optic_disc"].apply(
            lambda x: "abnormal optic disc" if x == 2 else "normal optic disc"
        )
        df["vessels"] = df["vessels"].apply(
            lambda x: "abnormal vessels" if x == 2 else "normal vessels"
        )
        df["macula"] = df["macula"].apply(
            lambda x: "abnormal macula" if x == 2 else "normal macula"
        )

        df["DR_ICDR"] = df["DR_ICDR"].apply(lambda x: f"{severity_to_text[x]}")
        df["diabetic_retinopathy"] = df["diabetic_retinopathy"].apply(
            lambda x: "diabetic retinopathy" if x == 1 else "no diabetic retinopathy"
        )
        df["amd"] = df["amd"].apply(
            lambda x: "age-related macular degeneration"
            if x == 1
            else "no age-related macular degeneration"
        )

        df["macular_edema"] = df["macular_edema"].apply(
            lambda x: "macular edema" if x == 1 else "no macular edema"
        )
        df["scar"] = df["scar"].apply(lambda x: "scar" if x == 1 else "no scar")
        df["nevus"] = df["nevus"].apply(lambda x: "nevus" if x == 1 else "no nevus")
        df["vascular_occlusion"] = df["vascular_occlusion"].apply(
            lambda x: "vascular occlusion" if x == 1 else "no vascular occlusion"
        )
        df["hypertensive_retinopathy"] = df["hypertensive_retinopathy"].apply(
            lambda x: "hypertensive retinopathy"
            if x == 1
            else "no hypertensive retinopathy"
        )
        df["drusens"] = df["drusens"].apply(
            lambda x: "drusens" if x == 1 else "no drusens"
        )
        df["hemorrhage"] = df["hemorrhage"].apply(
            lambda x: "hemorrhage" if x == 1 else "no hemorrhage"
        )
        df["retinal_detachment"] = df["retinal_detachment"].apply(
            lambda x: "retinal detachment" if x == 1 else "no retinal detachment"
        )
        df["myopic_fundus"] = df["myopic_fundus"].apply(
            lambda x: "myopic fundus" if x == 1 else "no myopic fundus"
        )
        df["increased_cup_disc"] = df["increased_cup_disc"].apply(
            lambda x: "increased cup-to-disc ratio"
            if x == 1
            else "no increased cup-to-disc ratio"
        )
        df["other"] = df["other"].apply(
            lambda x: "other abnormalities" if x == 1 else "no other abnormalities"
        )

        return df

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
            "patient_sex": ["male", "female"],
            "exam_eye": ["right", "left"],
            "diabetes": ["diabetic", "non-diabetic"],
            "optic_disc": ["abnormal optic disc", "normal optic disc"],
            "vessels": ["abnormal vessels", "normal vessels"],
            "macula": ["abnormal macula", "normal macula"],
            "DR_ICDR": [
                "no",
                "mild",
                "moderate",
                "severe",
                "proliferative",
            ],
            "diabetic_retinopathy": ["diabetic retinopathy", "no diabetic retinopathy"],
            "amd": [
                "age-related macular degeneration",
                "no age-related macular degeneration",
            ],
            "macular_edema": ["macular edema", "no macular edema"],
            "scar": ["scar", "no scar"],
            "nevus": ["nevus", "no nevus"],
            "vascular_occlusion": ["vascular occlusion", "no vascular occlusion"],
            "hypertensive_retinopathy": [
                "hypertensive retinopathy",
                "no hypertensive retinopathy",
            ],
            "drusens": ["drusens", "no drusens"],
            "hemorrhage": ["hemorrhage", "no hemorrhage"],
            "retinal_detachment": ["retinal detachment", "no retinal detachment"],
            "myopic_fundus": ["myopic fundus", "no myopic fundus"],
            "increased_cup_disc": [
                "increased cup-to-disc ratio",
                "no increased cup-to-disc ratio",
            ],
            "other": ["other abnormalities", "no other abnormalities"],
            "quality": ["Adequate", "Inadequate"],
            "image_field": ["field 1", "field 2"],
        }

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The original value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name == "DR_ICDR":
            template_text = "A fundus image showing {severity} diabetic retinopathy."
            return template_text.format(severity=value)

        if column_name in [
            "patient_sex",
            "exam_eye",
        ]:
            return f"A fundus image of a {value} eye."

        if column_name == "diabetes":
            return f"A fundus image of a {value} patient."

        if column_name in ["optic_disc", "vessels", "macula"]:
            article = "an" if value[0].lower() in "aeiou" else "a"

            return f"A fundus image showing {article} {value}."

        if column_name in [
            "diabetic_retinopathy",
            "amd",
            "macular_edema",
            "scar",
            "nevus",
            "vascular_occlusion",
            "hypertensive_retinopathy",
            "drusens",
            "hemorrhage",
            "retinal_detachment",
            "myopic_fundus",
            "increased_cup_disc",
            "other",
        ]:
            article = "an " if value[0].lower() in "aeiou" else "a "
            if value.startswith("no "):
                article = ""
            return f"A fundus image showing {article}{value}."

        if column_name == "quality":
            return f"A fundus image of {value.lower()} quality."

        return value
