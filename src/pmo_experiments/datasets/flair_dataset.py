"""FLAIR dataset integration."""

import ast
import contextlib
import json
import logging
import os
import random

import pandas as pd
from flair.modeling.dictionary import (
    definitions as flair_definitions,
)
from flair_datasets import (
    prepare_partitions as prepare_flair_partitions,
)

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset

logger = logging.getLogger(__name__)

DATASET_NAME_TO_FLAIR_PATH = {
    "eyepacs": "01_EYEPACS",
    "messidor": "02_MESSIDOR",
    "1000x39": "05_1000x39",
    "lag": "07_LAG",
    "odir-5k": "08_ODIR-5K",
    "papila": "09_PAPILA",
    "paraguay": "10_PARAGUAY",
    "stare": "11_STARE",
    "aria": "12_ARIA",
    "fives": "13_FIVES",
    "agar300": "14_AGAR300",
    "aptos": "15_APTOS",
    "fund-oct": "16_FUND-OCT",
    "diabretdb1": "17_DiaRetDB1",
    "drions-db": "18_DRIONS-DB",
    "drishti-gs1": "19_Drishti-GS1",
    "e-ophta": "20_E-ophta",
    "g1020": "21_G1020",
    "hei-med": "22_HEI-MED",
    "hrf": "23_HRF",
    "origa": "24_ORIGA",
    "refuge": "25_REFUGE",
    "roc": "26_ROC",
    "oia-ddr": "28_OIA-DDR",
    "airogs": "29_AIROGS",
    "sustech-sysu": "30_SUSTech-SYSU",
    "jichi": "31_JICHI",
    "chaksu": "32_CHAKSU",
    "dr1-2": "33_DR1-2",
    "cataract": "34_Cataract",
    "acrima": "36_ACRIMA",
    "deepdrid": "37_DeepDRiD",
    "mmac": "38_MMAC",
}


DATASET_NAME_TO_FUNCTION_MAP = {
    "eyepacs": prepare_flair_partitions.adequate_01_eyepacs,
    "messidor": prepare_flair_partitions.adequate_02_messidor,
    "1000x39": prepare_flair_partitions.adequate_05_1000x39,
    "lag": prepare_flair_partitions.adequate_07_lag,
    "odir-5k": prepare_flair_partitions.adequate_08_odir5k,
    "papila": prepare_flair_partitions.adequate_09_papila,
    "paraguay": prepare_flair_partitions.adequate_10_paraguay,
    "stare": prepare_flair_partitions.adequate_11_stare,
    "aria": prepare_flair_partitions.adequate_12_aria,
    "fives": prepare_flair_partitions.adequate_13_fives,
    "agar300": prepare_flair_partitions.adequate_14_agar300,
    "aptos": prepare_flair_partitions.adequate_15_aptos,
    "fund-oct": prepare_flair_partitions.adequate_16_fundoct,
    "diabretdb1": prepare_flair_partitions.adequate_17_diaretdb1,
    "drions-db": prepare_flair_partitions.adequate_18_drions_db,
    "drishti-gs1": prepare_flair_partitions.adequate_19_drishtigs1,
    "e-ophta": prepare_flair_partitions.adequate_20_e_ophta,
    "g1020": prepare_flair_partitions.adequate_21_g1020,
    # "hei-med": prepare_flair_partitions.adequate_22_hei_med,
    "hrf": prepare_flair_partitions.adequate_23_hrf,
    "origa": prepare_flair_partitions.adequate_24_origa,
    "refuge": prepare_flair_partitions.adequate_25_refuge,
    "roc": prepare_flair_partitions.adequate_26_roc,
    "oia-ddr": prepare_flair_partitions.adequate_28_OIA,
    "airogs": prepare_flair_partitions.adequate_29_airogs,
    "sustech-sysu": prepare_flair_partitions.adequate_30_sustech,
    "jichi": prepare_flair_partitions.adequate_31_jichi,
    "chaksu": prepare_flair_partitions.adequate_32_chaksu,
    "dr1-2": prepare_flair_partitions.adequate_33_dr,
    "cataract": prepare_flair_partitions.adequate_34_cataract,
    "acrima": prepare_flair_partitions.adequate_36_acrima,
    # "deepdrid": prepare_flair_partitions.adequate_37_deepdrid,
    # "mmac": prepare_flair_partitions.adequate_38_mmac,
}


@DATASET_REGISTRY.register("flair")
class FlairDataset(BaseDataset):
    """Flair dataset integration."""

    def __init__(self, **dataset_args: str):
        """
        Initialize Flair dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get("DATASET_PATH", "datasets/flair/raw")

        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/flair/converted"
        )
        self.dict_output_path = os.path.join(
            self.converted_dataset_path, "dictionary.json"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/flair/webdataset"
        )

        self.converted_flair_dataset_path = dataset_args.get(
            "CONVERTED_FLAIR_DATASET_PATH", "datasets/flair/converted"
        )

        # NOTE: flair has no val split
        self.selected_training_datasets = None  # dataset_args.get(
        #     "SELECTED_TRAINING_DATASETS", None
        # )

        if self.selected_training_datasets is None:
            # TODO load default
            self.selected_training_datasets = [
                "eyepacs",
                "1000x39",
                "lag",
                "odir-5k",
                "papila",
                "paraguay",
                "stare",
                "aria",
                "agar300",
                "fund-oct",
                "diabretdb1",
                "drions-db",
                "drishti-gs1",
                "e-ophta",
                "g1020",
                "hrf",
                "origa",
                "roc",
                "oia-ddr",
                "airogs",
                "sustech-sysu",
                "jichi",
                "chaksu",
                "dr1-2",
                "cataract",
            ]

        for dataset_name in self.selected_training_datasets:
            if dataset_name not in DATASET_NAME_TO_FLAIR_PATH:
                raise ValueError(
                    f"Dataset name '{dataset_name}' is not recognized. "
                    f"Valid options are: {list(DATASET_NAME_TO_FLAIR_PATH.keys())}"
                )

        # A [ATR] fundus photograph of [CLS]
        self.desc_base_prefix = "A "
        self.desc_base_suffix = "fundus photograph of "
        self.desc_base_suffix_short = "fundus photograph"
        self.desc_base_prefix_key = "[DESC_BASE_PREFIX]"
        self.desc_base_suffix_key = "[DESC_BASE_SUFFIX]"
        self.desc_base_suffix_short_key = "[DESC_BASE_SUFFIX_SHORT]"
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

    def _preprocess_loaded_dfs(
        self, train_dfs: list[pd.DataFrame], val_dfs: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        final_train_df = pd.concat(train_dfs, ignore_index=True)
        final_val_df = pd.concat(val_dfs, ignore_index=True)

        return final_train_df, final_val_df

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
            f"Flair dataset not found at {self.raw_dataset_path}."
        )

        # Prepare inputs for conversion

        flair_partition_input = prepare_flair_partitions.DatasetPreparationConfig(
            datasets_base_path=self.raw_dataset_path,
            pretrain_dataset_output_path=os.path.join(
                self.converted_flair_dataset_path, "pretrain"
            ),
            transferability_dataset_output_path=os.path.join(
                self.converted_flair_dataset_path, "transferability"
            ),
            classification_transferability_subfolder="classification",
            segmentation_transferability_subfolder="segmentation",
        )

        # Create directories

        os.makedirs(self.converted_dataset_path, exist_ok=True)
        os.makedirs(flair_partition_input.pretrain_dataset_output_path, exist_ok=True)
        os.makedirs(
            flair_partition_input.transferability_dataset_output_path, exist_ok=True
        )
        os.makedirs(
            os.path.join(
                flair_partition_input.transferability_dataset_output_path,
                flair_partition_input.classification_transferability_subfolder,
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                flair_partition_input.transferability_dataset_output_path,
                flair_partition_input.segmentation_transferability_subfolder,
            ),
            exist_ok=True,
        )

        # First run FLAIR conversions
        assert self.selected_training_datasets is not None, (
            "Selected training datasets should not be None at this point."
        )
        for dataset_name in self.selected_training_datasets:
            logger.info(f"Processing dataset {dataset_name}...")

            if dataset_name not in DATASET_NAME_TO_FUNCTION_MAP:
                raise ValueError(
                    f"Dataset name '{dataset_name}' is not recognized. "
                    f"Valid options are: {list(DATASET_NAME_TO_FUNCTION_MAP.keys())}"
                )

            if os.path.exists(
                os.path.join(
                    flair_partition_input.pretrain_dataset_output_path,
                    DATASET_NAME_TO_FLAIR_PATH[dataset_name] + ".csv",
                )
            ):
                logger.debug(
                    f"CSV file for dataset '{dataset_name}' already exists. Skipping."
                )
                continue

            conversion_function = DATASET_NAME_TO_FUNCTION_MAP[dataset_name]

            # Suppress print output of conversion function
            with contextlib.redirect_stdout(None):
                conversion_function(flair_partition_input)

        train_fraction = 0.9
        full_train_dfs = []
        full_val_dfs = []

        for dataset_name in self.selected_training_datasets:
            dataset_csv_path = os.path.join(
                flair_partition_input.pretrain_dataset_output_path,
                DATASET_NAME_TO_FLAIR_PATH[dataset_name] + ".csv",
            )

            if not os.path.exists(dataset_csv_path):
                raise ValueError(
                    f"Expected CSV file for dataset '{dataset_name}' not found "
                    f"at {dataset_csv_path}. "
                    "Please check the conversion step for this dataset."
                )

            df = pd.read_csv(dataset_csv_path).drop(
                columns=["Unnamed: 0"], errors="ignore"
            )

            if df.empty:
                raise ValueError(
                    f"CSV file for dataset '{dataset_name}' is empty. "
                    "Please check the conversion step for this dataset."
                )

            # Load and explode categories
            df["label_category"] = df["categories"].apply(ast.literal_eval)

            # Ignore spelling mistake from FLAIR
            # spellchecker:ignore-next-line
            df["label_attribute"] = df["atributes"].apply(ast.literal_eval)
            df = (
                df.explode("label_category")
                .explode("label_attribute")
                .reset_index(drop=True)
            )

            df["dataset_name"] = dataset_name

            # Preprocess df
            # [DESC_BASE] = A fundus photograph of
            # [CLS::<val>::] = target from dictionary
            # FLAIR excludes "06_DEN" (DeepEyeNet), "11_STARE", "08_ODIR-5K",
            # "31_JICHI" from CLS-substitution because their raw category text is
            # free-form

            def create_label_text(row):
                if row["dataset_name"] in [
                    "06_DEN",
                    "11_STARE",
                    "08_ODIR-5K",
                    "31_JICHI",
                    "den",
                    "stare",
                    "odir-5k",
                    "jichi",
                ]:
                    return (
                        self.desc_base_prefix
                        + self.desc_base_suffix
                        + " "
                        + row["label_category"].strip()
                    ).strip()
                attribute_text = (
                    row["label_attribute"].strip() + " "
                    if pd.notna(row["label_attribute"])
                    else ""
                )
                category_text = (
                    "[CLS::" + row["label_category"] + "::]"
                    if pd.notna(row["label_category"])
                    and not row["label_category"].strip() == ""
                    else ""
                )

                if category_text == "":
                    output_label_text = (
                        f"{self.desc_base_prefix_key}{attribute_text}"
                        f"{self.desc_base_suffix_short_key}".strip()
                    )
                else:
                    output_label_text = (
                        f"{self.desc_base_prefix_key}{attribute_text}"
                        f"{self.desc_base_suffix_key}{category_text}".strip()
                    )
                return output_label_text

            df["label_text"] = df.apply(create_label_text, axis=1)

            df["image_path"] = df["image"].apply(
                lambda x: os.path.join(self.raw_dataset_path, x.removeprefix("/"))
            )

            # Split into train and val group by image
            dataset_seed = 106080954
            unique_images = df["image_path"].unique()
            rng = random.Random(dataset_seed)
            shuffled_images = unique_images.tolist()
            rng.shuffle(shuffled_images)
            train_size = int(len(unique_images) * train_fraction)
            train_images = shuffled_images[:train_size]
            val_images = shuffled_images[train_size:]

            full_train_dfs.append(df[df["image_path"].isin(train_images)])
            full_val_dfs.append(df[df["image_path"].isin(val_images)])

        # Concatenate all datasets and save
        final_train_df, final_val_df = self._preprocess_loaded_dfs(
            full_train_dfs, full_val_dfs
        )

        final_train_df.to_csv(train_csv_path, index=False)
        final_val_df.to_csv(val_csv_path, index=False)

        # Save dataset sizes
        self.train_size = len(final_train_df)
        self.val_size = len(final_val_df)

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
        raise NotImplementedError("This method should be implemented.")
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
    def get_eval_target_column_names(relevant_only: bool = False) -> list[str]:  # noqa: ARG004
        """
        Return the list of target column names for evaluation.

        Args:
            relevant_only (bool): If True, return only relevant target columns. If
                False, return all target columns.

        Returns:
            list[str]: List of target column names for evaluation.

        """
        raise NotImplementedError("This method should be implemented.")
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
        raise NotImplementedError("This method should be implemented.")
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

    def generate_text_augmentation_dict(self) -> str:
        """
        Generate and save the dictionary for text augmentations.

        This dictionary defines keys that are randomly replaced by the list of values.
        Uses the FLAIR definitions for the replacement values.

        Returns:
            str: None if no text augmentation is needed, or the path to the
                saved dictionary file.

        """
        if os.path.exists(self.dict_output_path):
            return self.dict_output_path

        flair_dict = {
            self.desc_base_prefix_key: [self.desc_base_prefix],
            self.desc_base_suffix_key: [self.desc_base_suffix],
            self.desc_base_suffix_short_key: [self.desc_base_suffix_short],
        }

        for key, value_list in flair_definitions.items():
            updated_key = f"[CLS::{key}::]"
            flair_dict[updated_key] = value_list

        with open(self.dict_output_path, "w") as f:
            json.dump(flair_dict, f, indent=4, ensure_ascii=False)

        return self.dict_output_path
