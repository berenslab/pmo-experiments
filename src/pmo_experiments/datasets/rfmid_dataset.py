"""Module defining the classes for the RFMiD datasets (both 1 and 2)."""

import json
import os

import pandas as pd

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

RFMID_1_COLUMN_TO_FULL_NAME = {
    "DR": "diabetic retinopathy",
    "ARMD": "age-related macular degeneration",
    "MH": "media haze",
    "DN": "drusens",
    "MYA": "myopia",
    "BRVO": "branch retinal vein occlusion",
    "TSLN": "tessellation",
    "ERM": "epiretinal membrane",
    "LS": "laser scars",
    "MS": "macular scars",
    "CSR": "central serous retinopathy",
    "ODC": "optic disc cupping",
    "CRVO": "central retinal vein occlusion",
    "TV": "tortuous vessels",
    "AH": "asteroid hyalosis",
    "ODP": "optic disc pallor",
    "ODE": "optic disc edema",
    "ST": "shunt",
    "AION": "anterior ischemic optic neuropathy",
    "PT": "parafoveal telangiectasia",
    "RT": "retinal traction",
    "RS": "retinitis",
    "CRS": "chorioretinitis",
    "EDN": "exudation",  # spellchecker:disable-line
    "RPEC": "retinal pigment epithelium changes",
    "MHL": "macular hole",
    "RP": "retinitis pigmentosa",
    "CWS": "cotton-wool spots",
    "CB": "coloboma",
    "ODPM": "optic disc pit maculopathy",
    "PRH": "preretinal hemorrhage",
    "MNF": "myelinated nerve fibers",
    "HR": "hemorrhagic retinopathy",
    "CRAO": "central retinal artery occlusion",
    "TD": "tilted disc",
    "CME": "cystoid macular edema",
    "PTCR": "post-traumatic choroidal rupture",
    "CF": "choroidal folds",
    "VH": "vitreous hemorrhage",
    "MCA": "macroaneurysm",
    "VS": "vasculitis",
    "BRAO": "branch retinal artery occlusion",
    "PLQ": "plaque",
    "HPED": "hemorrhagic pigment epithelial detachment",
    "CL": "collateral vessels",
}

RFMID_2_COLUMN_TO_FULL_NAME = {
    "AH": "asteroid hyalosis",
    "AION": "anterior ischemic optic neuropathy",
    "ARMD": "age-related macular degeneration",
    "BRVO": "branch retinal vein occlusion",
    "CB": "coloboma",
    "CF": "choroidal folds",
    "CL": "collateral vessels",
    "CME": "cystoid macular edema",
    "CNV": "choroidal neovascularization",
    "CRAO": "central retinal artery occlusion",
    "CRS": "chorioretinitis",
    "CRVO": "central retinal vein occlusion",
    "CSR": "central serous retinopathy",
    "CWS": "cotton-wool spots",
    "CSC": "cysticercosis",
    "DN": "drusens",
    "DR": "diabetic retinopathy",
    # spellchecker:ignore-next-line
    "EDN": "exudation",
    "ERM": "epiretinal membrane",
    "GRT": "giant retinal tear",
    "HPED": "hemorrhagic pigment epithelial detachment",
    "HR": "hemorrhagic retinopathy",
    "LS": "laser scars",
    "MCA": "microaneurysm",
    "ME": "macular edema",
    "MH": "media haze",
    "MHL": "macular hole",
    "MS": "macular scar",
    "MYA": "myopia",
    "ODC": "optic disc cupping",
    "ODE": "optic disc edema",
    "ODP": "optic disc pallor",
    "ON": "optic neuritis",
    "OPDM": "optic disc pit maculopathy",
    "PRH": "pits retinal hemorrhage",
    "RD": "retinal detachment",
    "RHL": "retinal holes",
    "RTR": "retinal tears",
    "RP": "retinitis pigmentosa",
    "RPEC": "retinal pigment epithelium changes",
    "RS": "retinitis",
    "RT": "retinal traction",
    "SOFE": "silicone oil filled eye",
    "ST": "optociliary shunt",
    "TD": "tilted disc",
    "TSLN": "tessellation",
    "TV": "tortuous vessels",
    "VS": "vasculitis",
    "HTN": "hypertensive retinopathy",
    "IIH": "idiopathic intracranial hypertension",
}


JOINT_COLUMNS = [
    "AH",
    "AION",
    "ARMD",
    "BRVO",
    "CB",
    "CF",
    "CL",
    "CME",
    "CRAO",
    "CRS",
    "CRVO",
    "CSR",
    "CWS",
    "DN",
    "DR",
    "EDN",  # spellchecker:disable-line
    "ERM",
    "HPED",
    "HR",
    "LS",
    "MCA",
    "MH",
    "MHL",
    "MS",
    "MYA",
    "ODC",
    "ODE",
    "ODP",
    "ODPM",
    "PRH",
    "RP",
    "RPEC",
    "RS",
    "RT",
    "ST",
    "TD",
    "TSLN",
    "TV",
    "VS",
]


def rfmid_1_row_to_caption(row: pd.Series) -> str:
    """
    Convert a row from the RFMiD 1 dataset to a caption.

    Args:
        row (pd.Series): Row from RFMiD 1

    Returns:
        str: Caption describing the image based on the diseases present in the row.

    """
    if row["Disease_Risk"] == 0:
        return "A fundus image showing a normal eye."

    diseases = []

    for col, disease_name in RFMID_1_COLUMN_TO_FULL_NAME.items():
        if pd.notna(row[col]) and row[col] == 1:
            diseases.append(disease_name)

    if len(diseases) == 0:
        return "A fundus image showing a normal eye."

    if len(diseases) == 1:
        return f"A fundus image showing {diseases[0]}."

    diseases_str = ", ".join(diseases[:-1]) + f", and {diseases[-1]}"
    return f"A fundus image showing {diseases_str}."


def rfmid_2_row_to_caption(row: pd.Series) -> str:
    """
    Convert a row from the RFMiD 2 dataset to a caption.

    Args:
        row (pd.Series): Row from RFMiD 2

    Returns:
        str: Caption describing the image based on the diseases present in the row.

    """
    if row["WNL"] == 1:
        return "A fundus image showing a normal eye."

    diseases = []

    for col, disease_name in RFMID_2_COLUMN_TO_FULL_NAME.items():
        if pd.notna(row[col]) and row[col] == 1:
            diseases.append(disease_name)

    if len(diseases) == 0:
        return "A fundus image showing a normal eye."

    if len(diseases) == 1:
        return f"A fundus image showing {diseases[0]}."

    diseases_str = ", ".join(diseases[:-1]) + f", and {diseases[-1]}"
    return f"A fundus image showing {diseases_str}."


def joint_rfmid_row_to_caption(row: pd.Series) -> str:
    """
    Convert a row from the joint RFMID dataset to a caption.

    Args:
        row (pd.Series): Row from joint RFMID

    Returns:
        str: Caption describing the image based on the diseases present in the row.

    """
    if row["Disease_Risk"] == 0:
        return "A fundus image showing a normal eye."

    diseases = []

    for col in JOINT_COLUMNS:
        disease_name = RFMID_1_COLUMN_TO_FULL_NAME[col]
        if pd.notna(row[col]) and row[col] == 1:
            diseases.append(disease_name)

    if row["other"] == 1:
        diseases.append("other diseases")

    if len(diseases) == 0:
        return "A fundus image showing a normal eye."

    if len(diseases) == 1:
        return f"A fundus image showing {diseases[0]}."

    diseases_str = ", ".join(diseases[:-1]) + f", and {diseases[-1]}"
    return f"A fundus image showing {diseases_str}."


@DATASET_REGISTRY.register("rfmid_1")
class RFMID1Dataset(BaseDataset):
    """
    Dataset class for RFMiD 1 retinal fundus images and disease labels.

    Attributes:
        raw_dataset_path (str): Path to the raw RFMiD 1 dataset.
        converted_dataset_path (str): Path to the converted RFMiD 1 dataset in
            OpenCLIP format.
        all_target_value_paths (str): Path to the JSON file containing all target
            values for evaluation.
        train_size (int | None): Number of samples in the training set after conversion.
        val_size (int | None): Number of samples in the validation set after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize the RFMiD 1 dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization. Uses 'DATASET_PATH' for the raw dataset location and
                'CONVERTED_DATASET_PATH' for the converted dataset location.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/rfmid_1")
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/rfmid_1/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/rfmid_1/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self._training_folder = os.path.join(self.raw_dataset_path, "Training_Set")
        self._validation_folder = os.path.join(self.raw_dataset_path, "Evaluation_Set")
        self._test_folder = os.path.join(self.raw_dataset_path, "Test_Set")

        self._training_images_path = os.path.join(self._training_folder, "Training")
        self._validation_images_path = os.path.join(
            self._validation_folder, "Validation"
        )
        self._test_images_path = os.path.join(self._test_folder, "Test")

        self._training_csv_path = os.path.join(
            self._training_folder, "RFMiD_Training_Labels.csv"
        )
        self._validation_csv_path = os.path.join(
            self._validation_folder, "RFMiD_Validation_Labels.csv"
        )
        self._test_csv_path = os.path.join(
            self._test_folder, "RFMiD_Testing_Labels.csv"
        )

        self.train_size = None
        self.val_size = None

    def _convert_split_to_csv(
        self, images_path: str, base_csv_path: str, output_path: str
    ) -> int:
        """
        Convert a split from RFMiD 1 to a df.

        Args:
            images_path (str): Path to the folder containing the images for this split.
            base_csv_path (str): Path to the original CSV file containing the labels
                for this split.
            output_path (str): Path where the converted CSV file should be saved.

        Returns:
            int: Number of records processed.

        """

        def resolve_image_path(image_id: int) -> str | None:
            for ext in [".png"]:
                potential_path = os.path.join(images_path, f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    return potential_path
            return None

        df = pd.read_csv(base_csv_path)

        df["image_path"] = df["ID"].apply(resolve_image_path)

        df = df[df["image_path"].notna()].copy()

        df["caption"] = df.apply(rfmid_1_row_to_caption, axis=1)

        df.to_csv(output_path, index=False)

        return len(df)

    def to_open_clip_format(self) -> None:
        """Convert RFMiD 1 to the open clip compatible format."""
        training_df_path = self.get_train_df_path()
        validation_df_path = self.get_val_df_path()
        test_df_path = self.get_test_df_path()

        os.makedirs(self.converted_dataset_path, exist_ok=True)

        if not os.path.exists(training_df_path):
            train_size = self._convert_split_to_csv(
                images_path=self._training_images_path,
                base_csv_path=self._training_csv_path,
                output_path=training_df_path,
            )
            self.train_size = train_size
        else:
            self.train_size = len(pd.read_csv(training_df_path))

        if not os.path.exists(validation_df_path):
            val_size = self._convert_split_to_csv(
                images_path=self._validation_images_path,
                base_csv_path=self._validation_csv_path,
                output_path=validation_df_path,
            )
            self.val_size = val_size
        else:
            self.val_size = len(pd.read_csv(validation_df_path))

        if not os.path.exists(test_df_path):
            _ = self._convert_split_to_csv(
                images_path=self._test_images_path,
                base_csv_path=self._test_csv_path,
                output_path=test_df_path,
            )

        if not os.path.exists(self.all_target_value_paths):
            all_target_values = {
                "train": sorted(
                    pd.read_csv(
                        training_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "val": sorted(
                    pd.read_csv(
                        validation_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "test": sorted(
                    pd.read_csv(
                        test_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
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
        return ["Disease_Risk"] + sorted(list(RFMID_1_COLUMN_TO_FULL_NAME.keys()))

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
        df["Disease_Risk"] = df["Disease_Risk"].apply(
            lambda x: None if pd.isna(x) else ("diseased" if x == 1 else "normal")
        )

        for col, disease_text in RFMID_1_COLUMN_TO_FULL_NAME.items():
            df[col] = df[col].apply(
                lambda x: None
                if pd.isna(x)
                else (disease_text if x == 1 else f"no {disease_text}")
            )
            df[col] = df[col].fillna("Unknown").astype(str)

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
            "Disease_Risk": ["normal", "diseased"],
            **{
                col: [
                    f"no {disease_text}",
                    disease_text,
                ]
                for col, disease_text in RFMID_1_COLUMN_TO_FULL_NAME.items()
            },
        }

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name == "Disease_Risk":
            return "A fundus image showing a {} eye.".format(value)

        if column_name in RFMID1Dataset.get_eval_target_column_names():
            return "A fundus image showing {}.".format(value)

        return value


@DATASET_REGISTRY.register("rfmid_2")
class RFMID2Dataset(BaseDataset):
    """
    Dataset class for RFMiD 2 retinal fundus images and disease labels.

    Attributes:
        raw_dataset_path (str): Path to the raw RFMiD 2 dataset.
        converted_dataset_path (str): Path to the converted RFMiD 2 dataset in
            OpenCLIP format.
        all_target_value_paths (str): Path to the JSON file containing all target
            values for evaluation.
        train_size (int | None): Number of samples in the training set after conversion.
        val_size (int | None): Number of samples in the validation set after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize the RFMiD 2 dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization. Uses 'DATASET_PATH' for the raw dataset location and
                'CONVERTED_DATASET_PATH' for the converted dataset location.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/rfmid_2")
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/rfmid_2/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/rfmid_2/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self._training_folder = os.path.join(self.raw_dataset_path, "Train")
        self._validation_folder = os.path.join(self.raw_dataset_path, "Validation")
        self._test_folder = os.path.join(self.raw_dataset_path, "Test")

        self._training_csv_path = os.path.join(
            self._training_folder, "RFMiD_2_Training_labels.csv"
        )
        self._validation_csv_path = os.path.join(
            self._validation_folder, "RFMiD_2_Validation_labels.csv"
        )
        self._test_csv_path = os.path.join(
            self._test_folder, "RFMiD_2_Testing_labels.csv"
        )

        self.train_size = None
        self.val_size = None

    def _convert_split_to_csv(
        self, images_path: str, base_csv_path: str, output_path: str
    ) -> int:
        """
        Convert a split from RFMiD 2 to a df.

        Args:
            images_path (str): Path to the folder containing the images for this split.
            base_csv_path (str): Path to the original CSV file containing the labels
                for this split.
            output_path (str): Path where the converted CSV file should be saved.

        Returns:
            int: Number of records processed.

        """

        def resolve_image_path(image_id: int) -> str | None:
            for ext in [".jpg", ".JPG"]:
                potential_path = os.path.join(images_path, f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    return potential_path
            return None

        df = pd.read_csv(base_csv_path)

        df["image_path"] = df["ID"].apply(resolve_image_path)

        df = df[df["image_path"].notna()].copy()

        df["caption"] = df.apply(rfmid_2_row_to_caption, axis=1)

        df.to_csv(output_path, index=False)

        return len(df)

    def to_open_clip_format(self) -> None:
        """Convert RFMiD 2 to the open clip compatible format."""
        training_df_path = self.get_train_df_path()
        validation_df_path = self.get_val_df_path()
        test_df_path = self.get_test_df_path()

        os.makedirs(self.converted_dataset_path, exist_ok=True)

        if not os.path.exists(training_df_path):
            train_size = self._convert_split_to_csv(
                images_path=self._training_folder,
                base_csv_path=self._training_csv_path,
                output_path=training_df_path,
            )
            self.train_size = train_size
        else:
            self.train_size = len(pd.read_csv(training_df_path))

        if not os.path.exists(validation_df_path):
            val_size = self._convert_split_to_csv(
                images_path=self._validation_folder,
                base_csv_path=self._validation_csv_path,
                output_path=validation_df_path,
            )
            self.val_size = val_size
        else:
            self.val_size = len(pd.read_csv(validation_df_path))

        if not os.path.exists(test_df_path):
            _ = self._convert_split_to_csv(
                images_path=self._test_folder,
                base_csv_path=self._test_csv_path,
                output_path=test_df_path,
            )

        if not os.path.exists(self.all_target_value_paths):
            all_target_values = {
                "train": sorted(
                    pd.read_csv(
                        training_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "val": sorted(
                    pd.read_csv(
                        validation_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "test": sorted(
                    pd.read_csv(
                        test_df_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
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
        return ["WNL"] + sorted(list(RFMID_2_COLUMN_TO_FULL_NAME.keys()))

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
        df["WNL"] = df["WNL"].apply(
            lambda x: None if pd.isna(x) else ("normal" if x == 1 else "diseased")
        )

        for col, disease_text in RFMID_2_COLUMN_TO_FULL_NAME.items():
            df[col] = df[col].apply(
                lambda x: None
                if pd.isna(x)
                else (disease_text if x == 1 else f"no {disease_text}")
            )
            df[col] = df[col].fillna("Unknown").astype(str)

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
            "WNL": ["normal", "diseased"],
            **{
                col: [
                    f"no {disease_text}",
                    disease_text,
                ]
                for col, disease_text in RFMID_2_COLUMN_TO_FULL_NAME.items()
            },
        }

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name == "WNL":
            return "A fundus image showing a {} eye.".format(value)

        if column_name in RFMID1Dataset.get_eval_target_column_names():
            return "A fundus image showing {}.".format(value)

        return value


@DATASET_REGISTRY.register("joint_rfmid")
class JointRFMIDDataset(BaseDataset):
    """
    Dataset class for the joint RFMiD 1 and RFMiD 2 datasets.

    This dataset only contains columns that are present in both datasets
    plus an additional "other" column for diseases that are only present in one of the
    datasets.

    Attributes:
        rfmid_1_dataset (RFMID1Dataset): Instance of the RFMiD 1 dataset.
        rfmid_2_dataset (RFMID2Dataset): Instance of the RFMiD 2 dataset.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize the joint RFMiD dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization. Uses 'DATASET_PATH' for the raw dataset location and
                'CONVERTED_DATASET_PATH' for the converted dataset location.

        """
        rfmid_1_dataset_args = {
            key.replace("RFMID_1_", ""): value
            for key, value in dataset_args.items()
            if key.startswith("RFMID_1_")
        }

        self.rfmid_1_dataset = RFMID1Dataset(**rfmid_1_dataset_args)

        rfmid_2_dataset_args = {
            key.replace("RFMID_2_", ""): value
            for key, value in dataset_args.items()
            if key.startswith("RFMID_2_")
        }

        self.rfmid_2_dataset = RFMID2Dataset(**rfmid_2_dataset_args)

        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/joint_rfmid/converted"
        )

        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/joint_rfmid/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.train_size = None
        self.val_size = None

    def to_open_clip_format(self) -> None:
        """Convert JointRFMID to the open clip compatible format."""
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        val_csv_path = os.path.join(self.converted_dataset_path, "val.csv")
        test_csv_path = os.path.join(self.converted_dataset_path, "test.csv")

        if (
            not os.path.exists(train_csv_path)
            or not os.path.exists(val_csv_path)
            or not os.path.exists(test_csv_path)
        ):
            os.makedirs(self.converted_dataset_path, exist_ok=True)

            self.rfmid_1_dataset.to_open_clip_format()
            self.rfmid_2_dataset.to_open_clip_format()

            # Get the relevant columns
            rfmid_1_train_df = pd.read_csv(
                self.rfmid_1_dataset.get_train_df_path(),
            )
            rfmid_1_val_df = pd.read_csv(
                self.rfmid_1_dataset.get_val_df_path(),
            )
            rfmid_1_test_df = pd.read_csv(
                self.rfmid_1_dataset.get_test_df_path(),
            )

            rfmid_2_train_df = pd.read_csv(
                self.rfmid_2_dataset.get_train_df_path(),
            ).rename(columns={"OPDM": "ODPM"})
            rfmid_2_val_df = pd.read_csv(
                self.rfmid_2_dataset.get_val_df_path(),
            ).rename(columns={"OPDM": "ODPM"})
            rfmid_2_test_df = pd.read_csv(
                self.rfmid_2_dataset.get_test_df_path(),
            ).rename(columns={"OPDM": "ODPM"})

            # Get disease intersection
            rfmid_1_train_df = rfmid_1_train_df[
                ["ID", "Disease_Risk", "image_path", "caption"] + JOINT_COLUMNS
            ]
            rfmid_1_val_df = rfmid_1_val_df[
                ["ID", "Disease_Risk", "image_path", "caption"] + JOINT_COLUMNS
            ]
            rfmid_1_test_df = rfmid_1_test_df[
                ["ID", "Disease_Risk", "image_path", "caption"] + JOINT_COLUMNS
            ]

            rfmid_2_train_df = rfmid_2_train_df[
                ["ID", "WNL", "image_path", "caption"] + JOINT_COLUMNS
            ]
            rfmid_2_val_df = rfmid_2_val_df[
                ["ID", "WNL", "image_path", "caption"] + JOINT_COLUMNS
            ]
            rfmid_2_test_df = rfmid_2_test_df[
                ["ID", "WNL", "image_path", "caption"] + JOINT_COLUMNS
            ]

            # Convert WNL to disease risk
            rfmid_2_train_df["Disease_Risk"] = rfmid_2_train_df["WNL"].apply(
                lambda x: 0 if x == 1 else 1 if pd.notna(x) else None
            )
            rfmid_2_val_df["Disease_Risk"] = rfmid_2_val_df["WNL"].apply(
                lambda x: 0 if x == 1 else 1 if pd.notna(x) else None
            )
            rfmid_2_test_df["Disease_Risk"] = rfmid_2_test_df["WNL"].apply(
                lambda x: 0 if x == 1 else 1 if pd.notna(x) else None
            )

            # Drop WNL
            rfmid_2_train_df = rfmid_2_train_df.drop(columns=["WNL"])
            rfmid_2_val_df = rfmid_2_val_df.drop(columns=["WNL"])
            rfmid_2_test_df = rfmid_2_test_df.drop(columns=["WNL"])

            # Add other column for missing diseases
            rfmid_1_train_df["other"] = rfmid_1_train_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )
            rfmid_1_val_df["other"] = rfmid_1_val_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )
            rfmid_1_test_df["other"] = rfmid_1_test_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )

            rfmid_2_train_df["other"] = rfmid_2_train_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )
            rfmid_2_val_df["other"] = rfmid_2_val_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )
            rfmid_2_test_df["other"] = rfmid_2_test_df.apply(
                lambda row: 1
                if row[JOINT_COLUMNS].sum() == 0 and row["Disease_Risk"] == 1
                else 0,
                axis=1,
            )

            # Concatenate the two datasets
            train_df = pd.concat(
                [rfmid_1_train_df, rfmid_2_train_df], ignore_index=True
            )
            val_df = pd.concat([rfmid_1_val_df, rfmid_2_val_df], ignore_index=True)
            test_df = pd.concat([rfmid_1_test_df, rfmid_2_test_df], ignore_index=True)

            # Recaption for joint dataset
            train_df["caption"] = train_df.apply(joint_rfmid_row_to_caption, axis=1)
            val_df["caption"] = val_df.apply(joint_rfmid_row_to_caption, axis=1)
            test_df["caption"] = test_df.apply(joint_rfmid_row_to_caption, axis=1)

            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)

            self.train_size = len(train_df)
            self.val_size = len(val_df)

        else:
            self.train_size = len(pd.read_csv(train_csv_path))
            self.val_size = len(pd.read_csv(val_csv_path))

        if not os.path.exists(self.all_target_value_paths):
            all_target_values = {
                "train": sorted(
                    pd.read_csv(
                        train_csv_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "val": sorted(
                    pd.read_csv(
                        val_csv_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
                    .unique()
                    .tolist()
                ),
                "test": sorted(
                    pd.read_csv(
                        test_csv_path, usecols=["caption"], dtype={"caption": str}
                    )["caption"]
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
        return ["Disease_Risk"] + sorted(JOINT_COLUMNS) + ["other"]

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
        df["Disease_Risk"] = df["Disease_Risk"].apply(
            lambda x: None if pd.isna(x) else ("diseased" if x == 1 else "normal")
        )

        for col in JOINT_COLUMNS:
            disease_text = RFMID_1_COLUMN_TO_FULL_NAME[col]
            df[col] = df[col].apply(
                lambda x: None
                if pd.isna(x)
                else (disease_text if x == 1 else f"no {disease_text}")
            )
            df[col] = df[col].fillna("Unknown").astype(str)

        df["other"] = df["other"].apply(
            lambda x: None
            if pd.isna(x)
            else ("other diseases" if x == 1 else "no other diseases")
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
            "Disease_Risk": ["normal", "diseased"],
            **{
                col: [
                    f"no {RFMID_1_COLUMN_TO_FULL_NAME[col]}",
                    RFMID_1_COLUMN_TO_FULL_NAME[col],
                ]
                for col in JOINT_COLUMNS
            },
            "other": ["no other diseases", "other diseases"],
        }

    def get_value_verbalization(self, column_name: str, value: str) -> str:
        """
        Get the verbalization of a target value for a given column.

        Args:
            column_name (str): The name of the target column.
            value (str): The target value to verbalize.

        Returns:
            str: The verbalization of the target value.

        """
        if column_name == "Disease_Risk":
            return "A fundus image showing a {} eye.".format(value)

        if column_name in JointRFMIDDataset.get_eval_target_column_names():
            return "A fundus image showing {}.".format(value)

        return value
