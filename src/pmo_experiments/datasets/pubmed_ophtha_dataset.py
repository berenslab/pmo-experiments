"""
Integration of the PubMed-Ophtha dataset.

This module provides dataset adapters for the PubMed-Ophtha dataset and its
variants (CFP, fuzzy CFP, in-text mentions, etc.), including conversion to
OpenCLIP-compatible CSV/image format and robust image extraction from Parquet.
"""

import json
import logging
import os

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

from pmo_experiments.datasets.base_dataset import DATASET_REGISTRY, BaseDataset
from pmo_experiments.util import get_cpu_count
from pmo_experiments.util.dataset_decontamination import (
    detect_dataset_contamination,
)

logger = logging.getLogger(__name__)


def preprocess_df(
    dataframe: pd.DataFrame,
    image_folder_path: str,
    include_in_text_mentions: bool = False,
    include_image_types: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame of image/caption pairs for OpenCLIP training/eval.

    Args:
        dataframe (pd.DataFrame): Input metadata DataFrame (one row per panel).
        image_folder_path (str): Path to folder containing extracted images.
        include_in_text_mentions (bool, optional): If True, add in-text mentions as
            captions. Defaults to False.
        include_image_types (bool, optional): If True, add image type description as
            caption. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with columns
            - panel_id
            - image_path
            - caption
            - caption_origin.

    """
    if len(dataframe) == 0:
        # return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=["panel_id", "image_path", "caption", "caption_origin"]
        )
    output_records = []

    for _, row in dataframe.iterrows():
        panel_id = row["panel_id"]

        image_path = os.path.join(image_folder_path, f"{panel_id}.png")

        if not os.path.exists(image_path):
            logger.warning(f"Image not found at path: {image_path}")
            continue

        if pd.isna(row["subcaption_text"]) or row["subcaption_text"] == "":
            logger.warning(f"Missing subcaption_text for panel_id {panel_id}")
            continue

        output_records.append(
            {
                "panel_id": panel_id,
                "image_path": image_path,
                "caption": row["subcaption_text"],
                "caption_origin": "subcaption_text",
            }
        )

        if include_in_text_mentions:
            # Load in_text_mention
            in_text_mentions = row["in_text_mention"]
            if (
                pd.isna(in_text_mentions)
                or in_text_mentions == "[]"
                or in_text_mentions == ""
            ):
                continue

            # Load json list of in-text mentions
            try:
                in_text_mentions_list = json.loads(in_text_mentions)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding in_text_mention for panel_id {panel_id}: {e}"
                )
                continue

            for context_text in in_text_mentions_list:
                output_records.append(
                    {
                        "panel_id": panel_id,
                        "image_path": image_path,
                        "caption": context_text,
                        "caption_origin": "in_text_mention",
                    }
                )

        if include_image_types:
            detected_image_types = []

            if row["contains_cfp"]:
                detected_image_types.append("CFPs")
            if row["contains_oct"]:
                detected_image_types.append("OCTs")
            if row["contains_retinal"]:
                detected_image_types.append("retinal images")
            if row["contains_other"]:
                detected_image_types.append("other images")

            is_marked = row["contains_marked"]

            image_type_text = (
                detected_image_types[0]
                if len(detected_image_types) == 1
                else ", ".join(detected_image_types[:-1])
                + f", and {detected_image_types[-1]}"
            )

            if is_marked:
                image_type_text += "with marks indicating important features"

            output_records.append(
                {
                    "panel_id": panel_id,
                    "image_path": image_path,
                    "caption": f"A panel in a figure depicting {image_type_text}.",
                    "caption_origin": "image_type_description",
                }
            )

    return pd.DataFrame.from_records(output_records)


@DATASET_REGISTRY.register("pubmed_ophtha")
class PubMedOphthaDataSet(BaseDataset):
    """
    PubMed-Ophtha dataset integration.

    Attributes:
        raw_dataset_path (str): Path to the raw PubMed-Ophtha dataset.
        converted_dataset_path (str): Path to the converted OpenCLIP-compatible dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        images_folder (str): Path to the folder with extracted images.
        all_target_value_paths (str): Path to the JSON file with all target values.
        dataset_parquet_path (str): Path to the Parquet file with panel metadata.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.
        test_size (int | None): Number of test samples after conversion. Set lazily by
            `to_open_clip_format`, unlike `train_size`/`val_size` it is not
            initialized in `__init__`.
        include_in_text_mentions (bool): Whether to include in-text mentions as
            captions.
        include_image_types (bool): Whether to include image type description as
            caption.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize PubMed-Ophtha dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        assert all([isinstance(v, str) for v in dataset_args.values()])

        # Initialize dataset with args
        self.raw_dataset_path = dataset_args.get(
            "DATASET_PATH", "datasets/pubmed_ophtha"
        )
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha/webdataset"
        )

        self.images_folder = dataset_args.get(
            "IMAGES_FOLDER", os.path.join(self.converted_dataset_path, "images")
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.dataset_parquet_path = os.path.join(
            self.raw_dataset_path, "pubmed_ophtha.parquet"
        )

        self._hf_repo_path = (
            "hf://datasets/pubmed-ophtha/PubMed-Ophtha/pubmed_ophtha.parquet"
        )

        self.train_size = None
        self.val_size = None

        self.include_in_text_mentions = False
        self.include_image_types = False

    def prepare_train_test_val_dfs(
        self, meta_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits from the metadata DataFrame.

        Args:
            meta_data_df (pd.DataFrame): DataFrame with panel metadata.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        """
        # Create the train test val splits
        # Split into test based on article_id
        random_seed = 106080954

        def get_auxiliary_img_type(row):
            img_types = []
            if row["contains_cfp"]:
                img_types.append("cfp")
            elif row["contains_oct"]:
                img_types.append("oct")
            elif row["contains_retinal"]:
                img_types.append("retinal")
            elif row["contains_other"]:
                img_types.append("other")

            if row["contains_marked"]:
                img_types.append("marked")
            else:
                img_types.append("unmarked")

            return "_".join(img_types)

        auxiliary_img_type = meta_data_df.apply(get_auxiliary_img_type, axis=1)

        # Stratified split into train and test based on the auxiliary image type and
        # article id as group
        X = meta_data_df
        y = auxiliary_img_type
        groups = meta_data_df["article_id"]

        stratified_group_kfold = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=random_seed
        )

        # Get first split (train/test)
        train_idx, test_idx = next(stratified_group_kfold.split(X, y, groups))

        train_df = meta_data_df.iloc[train_idx]
        test_df = meta_data_df.iloc[test_idx]

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

        return train_df, val_df, test_df

    def _download_dataset_from_hf(self) -> None:
        logger.info(
            "Downloading PubMed-Ophtha dataset from HuggingFace to "
            f"{self.dataset_parquet_path}"
        )
        df = pd.read_parquet(self._hf_repo_path)
        df.to_parquet(self.dataset_parquet_path, index=False)

    def to_open_clip_format(self) -> None:
        """
        Convert the PubMed-Ophtha dataset to OpenCLIP CSV/image format.

        This extracts images from the Parquet file, checks for corruption, and writes
        train/val/test CSVs and a target values JSON. Uses the instance's
        include_in_text_mentions, and include_image_types flags.
        """
        train_csv_path = os.path.join(self.converted_dataset_path, "train.csv")
        val_csv_path = os.path.join(self.converted_dataset_path, "val.csv")
        test_csv_path = os.path.join(self.converted_dataset_path, "test.csv")

        if (
            not os.path.exists(train_csv_path)
            or not os.path.exists(val_csv_path)
            or not os.path.exists(test_csv_path)
        ):
            if not os.path.exists(self.dataset_parquet_path):
                # Download the dataset from HuggingFace if not present
                self._download_dataset_from_hf()

            pf = pq.ParquetFile(self.dataset_parquet_path)

            cols = [c for c in pf.schema_arrow.names if c != "panel_image_bytes"]

            meta_data_df = pd.read_parquet(self.dataset_parquet_path, columns=cols)

            train_df, val_df, test_df = self.prepare_train_test_val_dfs(meta_data_df)

            # Save the images as png from the parquet file
            batch_size = 1000
            columns = ["panel_id", "panel_image_bytes"]

            relevant_panel_ids = set(
                pd.concat([train_df, val_df, test_df], ignore_index=True)[
                    "panel_id"
                ].to_list()
            )

            num_batches = (len(meta_data_df) + batch_size - 1) // batch_size
            for batch in tqdm(
                pf.iter_batches(batch_size=batch_size, columns=columns),
                total=num_batches,
                desc="Saving images",
            ):
                batch_df = batch.to_pandas()

                batch_df = batch_df[batch_df["panel_id"].isin(relevant_panel_ids)]

                if batch_df.empty:
                    continue

                batch_df["image_path"] = batch_df["panel_id"].apply(
                    lambda panel_id: os.path.join(
                        self.images_folder,
                        f"{panel_id}.png",
                    )
                )

                for _, row in batch_df.iterrows():
                    image_path = row["image_path"]

                    if pd.isna(row["panel_image_bytes"]):
                        logger.warning(
                            f"Missing panel_image_bytes for panel_id {row['panel_id']}"
                        )
                        continue
                        # raise ValueError(
                        #  f"Missing panel_image_bytes for panel_id {row['panel_id']}"
                        # )

                    if os.path.exists(image_path):
                        continue

                    os.makedirs(os.path.dirname(image_path), exist_ok=True)

                    with open(image_path, "wb") as f:
                        f.write(row["panel_image_bytes"])

                    # Try to open the image to check if it's corrupted
                    try:
                        with Image.open(image_path) as img:
                            img.convert("RGB").load()
                    except (OSError, SyntaxError, UnidentifiedImageError) as e:
                        logger.warning(f"Image at path {image_path} is corrupted: {e}")
                        os.remove(image_path)

            # Preprocess the dataframes to create the captions
            train_df = preprocess_df(
                train_df,
                self.images_folder,
                include_in_text_mentions=self.include_in_text_mentions,
                include_image_types=self.include_image_types,
            )
            val_df = preprocess_df(
                val_df,
                self.images_folder,
                include_in_text_mentions=self.include_in_text_mentions,
                include_image_types=self.include_image_types,
            )
            test_df = preprocess_df(
                test_df,
                self.images_folder,
                include_in_text_mentions=self.include_in_text_mentions,
                include_image_types=self.include_image_types,
            )

            os.makedirs(self.converted_dataset_path, exist_ok=True)

            # Save the dataframes as csv files
            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)

            # Create the target values json file
            target_values = {
                "train": {
                    "caption": train_df["caption"].unique().tolist(),
                },
                "val": {
                    "caption": val_df["caption"].unique().tolist(),
                },
                "test": {
                    "caption": test_df["caption"].unique().tolist(),
                },
            }

            with open(self.all_target_value_paths, "w") as f:
                json.dump(target_values, f, indent=4, ensure_ascii=False)

        elif not os.path.exists(self.all_target_value_paths):
            # Create the target values json file
            target_values = {
                "train": {
                    "caption": pd.read_csv(
                        train_csv_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]
                    .unique()
                    .tolist(),
                },
                "val": {
                    "caption": pd.read_csv(
                        val_csv_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]
                    .unique()
                    .tolist(),
                },
                "test": {
                    "caption": pd.read_csv(
                        test_csv_path,
                        usecols=["caption"],
                        dtype={"caption": str},
                    )["caption"]
                    .unique()
                    .tolist(),
                },
            }

            with open(self.all_target_value_paths, "w") as f:
                json.dump(target_values, f, indent=4, ensure_ascii=False)

        # Read dataset sizes
        self.train_size = sum(
            len(chunk)
            for chunk in pd.read_csv(
                train_csv_path,
                usecols=["image_path"],
                dtype={"image_path": str},
                chunksize=100000,
            )  # type: ignore[reportCallIssue]
        )

        self.val_size = sum(
            len(chunk)
            for chunk in pd.read_csv(
                val_csv_path,
                usecols=["image_path"],
                dtype={"image_path": str},
                chunksize=100000,
            )  # type: ignore[reportCallIssue]
        )

        self.test_size = sum(
            len(chunk)
            for chunk in pd.read_csv(
                test_csv_path,
                usecols=["image_path"],
                dtype={"image_path": str},
                chunksize=100000,
            )  # type: ignore[reportCallIssue]
        )

    def get_dataset_size(self) -> tuple[int, int]:
        """
        Return the sizes of the train and val datasets.

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
        Preprocess the split DataFrame for evaluation.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.
            split (str): The name of the split (e.g., 'train', 'val', 'test').

        Returns:
            pd.DataFrame: The preprocessed DataFrame.

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
            list[str]: List of evaluation target column names.

        """
        return ["caption"]


@DATASET_REGISTRY.register("pubmed_ophtha_cfp")
class PubMedOphthaCFPDataSet(PubMedOphthaDataSet):
    """
    PubMed-Ophtha CFP-only dataset integration.

    Attributes:
        converted_dataset_path (str): Path to the converted dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        all_target_value_paths (str): Path to the JSON file with all target values.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize PubMed-Ophtha CFP-only dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha_cfp/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha_cfp/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.train_size = None
        self.val_size = None

    def prepare_train_test_val_dfs(
        self, meta_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits, keeping only CFP panels.

        Args:
            meta_data_df (pd.DataFrame): DataFrame with panel metadata.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        """
        train_df, val_df, test_df = super().prepare_train_test_val_dfs(meta_data_df)  # pyright: ignore[reportAttributeAccessIssue]

        # Drop rows that don't contain CFPs
        # Works since the splits are stratified
        train_mask = (
            train_df["contains_cfp"]
            & ~train_df["contains_other"]
            & ~train_df["contains_oct"]
            & ~train_df["contains_retinal"]
        )
        val_mask = (
            val_df["contains_cfp"]
            & ~val_df["contains_other"]
            & ~val_df["contains_oct"]
            & ~val_df["contains_retinal"]
        )
        test_mask = (
            test_df["contains_cfp"]
            & ~test_df["contains_other"]
            & ~test_df["contains_oct"]
            & ~test_df["contains_retinal"]
        )

        train_df = train_df[train_mask]
        val_df = val_df[val_mask]
        test_df = test_df[test_mask]

        return train_df, val_df, test_df


@DATASET_REGISTRY.register("pubmed_ophtha_in_text_mentions")
class PubMedOphthaInTextDataSet(PubMedOphthaDataSet):
    """
    PubMed-Ophtha dataset with in-text mentions integration.

    Attributes:
        converted_dataset_path (str): Path to the converted dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        include_in_text_mentions (bool): Whether to include in-text mentions as
            captions.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize PubMed-Ophtha FP in-text dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha_in_text/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha_in_text/webdataset"
        )

        self.include_in_text_mentions = True


@DATASET_REGISTRY.register("pubmed_ophtha_decon")
class PubMedOphthaDeconDataset(PubMedOphthaDataSet):
    """
    PubMed-Ophtha dataset decontaminated against known test datasets.

    Attributes:
        converted_dataset_path (str): Path to the converted dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        article_body_path (str): Path to the folder where downloaded article XML
            bodies are stored for contamination checking.
        delete_article_bodies (bool): Whether to delete `article_body_path` after the
            contamination check has run.
        all_target_value_paths (str): Path to the JSON file with all target values.
        contaminated_panels_path (str): Path to the JSON file listing panel IDs
            flagged as contaminated (i.e. overlapping with known test datasets).
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize PubMed-Ophtha decontaminated dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha_decon/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha_decon/webdataset"
        )

        self.article_body_path = dataset_args.get(
            "ARTICLE_BODY_PATH", "datasets/pubmed_ophtha_decon/article_bodies"
        )

        self.use_cached_contamination_results = (
            dataset_args.get("USE_CACHED_CONTAMINATION_RESULTS", "True").lower()
            == "true"
        )

        self.delete_article_bodies = (
            dataset_args.get("DELETE_ARTICLE_BODIES", "False").lower() == "true"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        # Use contaminated_panels.json in the repository if
        # USE_CACHED_CONTAMINATION_RESULTS is True
        self.contaminated_panels_path = (
            os.path.join(self.converted_dataset_path, "contaminated_panels.json")
            if not self.use_cached_contamination_results
            else os.path.join(os.path.dirname(__file__), "contaminated_panels.json")
        )

        self.train_size = None
        self.val_size = None

    def prepare_train_test_val_dfs(
        self, meta_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits with contaminated panels removed.

        Runs `detect_dataset_contamination` (caching results at
        `contaminated_panels_path`) to identify panels overlapping with known test
        datasets, then drops those panels from the stratified splits produced by the
        parent class.

        Args:
            meta_data_df (pd.DataFrame): DataFrame with panel metadata.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        """
        # Detect contaminated samples based on article body presence

        if not os.path.exists(self.contaminated_panels_path):
            detect_dataset_contamination(
                dataset_path=self.dataset_parquet_path,  # pyright: ignore[reportAttributeAccessIssue]
                article_body_folder=self.article_body_path,
                delete_body_folder=self.delete_article_bodies,
                output_file=self.contaminated_panels_path,
                num_workers=get_cpu_count(),
            )

        with open(self.contaminated_panels_path) as f:
            contaminated_panels = json.load(f)

        contaminated_panel_set = {
            panel_id for entry in contaminated_panels for panel_id in entry["panel_ids"]
        }

        train_df, val_df, test_df = super().prepare_train_test_val_dfs(meta_data_df)  # pyright: ignore[reportAttributeAccessIssue]

        # Drop rows that don't contain CFPs
        # Works since the splits are stratified
        train_mask = ~train_df["panel_id"].isin(contaminated_panel_set)
        val_mask = ~val_df["panel_id"].isin(contaminated_panel_set)
        test_mask = ~test_df["panel_id"].isin(contaminated_panel_set)

        train_df = train_df[train_mask]
        val_df = val_df[val_mask]
        test_df = test_df[test_mask]

        return train_df, val_df, test_df


@DATASET_REGISTRY.register("pubmed_ophtha_cfp_matched")
class PubMedOphthaCFPMatched(PubMedOphthaCFPDataSet):
    """
    PubMed-Ophtha CFP dataset sampled to the size of DeepEyeNet.

    Attributes:
        converted_dataset_path (str): Path to the converted dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        all_target_value_paths (str): Path to the JSON file with all target values.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize PubMed-Ophtha CFP matched dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha_cfp_matched/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha_cfp_matched/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.train_size = None
        self.val_size = None

    def prepare_train_test_val_dfs(
        self, meta_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits, keeping only CFP panels.

        Args:
            meta_data_df (pd.DataFrame): DataFrame with panel metadata.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        """
        train_df, val_df, test_df = super().prepare_train_test_val_dfs(meta_data_df)  # pyright: ignore[reportAttributeAccessIssue]

        seed = 106080954

        # Sample down to DEN sizes
        val_size = 3142
        test_size = 3140
        train_size = 9428

        unique_train_images = train_df["panel_id"].unique()
        unique_val_images = val_df["panel_id"].unique()
        unique_test_images = test_df["panel_id"].unique()

        if len(unique_train_images) > train_size:
            train_df = train_df[
                train_df["panel_id"].isin(
                    pd.Series(unique_train_images).sample(
                        n=train_size, random_state=seed
                    )
                )
            ]
        if len(unique_val_images) > val_size:
            val_df = val_df[
                val_df["panel_id"].isin(
                    pd.Series(unique_val_images).sample(n=val_size, random_state=seed)
                )
            ]

        if len(unique_test_images) > test_size:
            test_df = test_df[
                test_df["panel_id"].isin(
                    pd.Series(unique_test_images).sample(n=test_size, random_state=seed)
                )
            ]

        return train_df, val_df, test_df


@DATASET_REGISTRY.register("pubmed_ophtha_full")
class PubMedOphthaFull(PubMedOphthaDataSet):
    """
    PubMed-Ophtha (full) dataset integration.

    This version has no test and only a 5% validation split, to allow for maximum
    training data.
    It is intended for use in pretraining and fine-tuning of models, rather than
    evaluation.

    Attributes:
        raw_dataset_path (str): Path to the raw PubMed-Ophtha dataset.
        converted_dataset_path (str): Path to the converted OpenCLIP-compatible dataset.
        webdataset_path (str): Path to the folder for WebDataset shards.
        images_folder (str): Path to the folder with extracted images.
        all_target_value_paths (str): Path to the JSON file with all target values.
        dataset_parquet_path (str): Path to the Parquet file with panel metadata.
        train_size (int | None): Number of training samples after conversion.
        val_size (int | None): Number of validation samples after conversion.
        test_size (int | None): Number of test samples after conversion. Set lazily by
            `to_open_clip_format`, unlike `train_size`/`val_size` it is not
            initialized in `__init__`.
        include_in_text_mentions (bool): Whether to include in-text mentions as
            captions.
        include_image_types (bool): Whether to include image type description as
            caption.

    """

    def __init__(self, **dataset_args: str):
        """
        Initialize the full PubMed-Ophtha dataset with given arguments.

        Args:
            **dataset_args (str): Arbitrary keyword arguments for dataset
                initialization.

        """
        super().__init__(**dataset_args)
        self.converted_dataset_path = dataset_args.get(
            "CONVERTED_DATASET_PATH", "datasets/pubmed_ophtha_full/converted"
        )
        self.webdataset_path = dataset_args.get(
            "WEBDATASET_PATH", "datasets/pubmed_ophtha_full/webdataset"
        )

        self.all_target_value_paths = os.path.join(
            self.converted_dataset_path, "target_values.json"
        )

        self.train_size = None
        self.val_size = None

        self.include_in_text_mentions = False
        self.include_image_types = False

    def prepare_train_test_val_dfs(
        self, meta_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train and validation splits without a test split.

        Args:
            meta_data_df (pd.DataFrame): DataFrame with panel metadata.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        """
        # Mini val split for evaluation, no test split
        val_split = 0.05
        random_seed = 106080954

        def get_auxiliary_img_type(row):
            img_types = []
            if row["contains_cfp"]:
                img_types.append("cfp")
            elif row["contains_oct"]:
                img_types.append("oct")
            elif row["contains_retinal"]:
                img_types.append("retinal")
            elif row["contains_other"]:
                img_types.append("other")

            if row["contains_marked"]:
                img_types.append("marked")
            else:
                img_types.append("unmarked")

            return "_".join(img_types)

        auxiliary_img_type = meta_data_df.apply(get_auxiliary_img_type, axis=1)

        X = meta_data_df
        y = auxiliary_img_type

        stratified_shuffle_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_split, random_state=random_seed
        )

        train_idx, val_idx = next(stratified_shuffle_splitter.split(X, y))

        val_df = X.iloc[val_idx]
        train_df = X.iloc[train_idx]

        # No test split
        test_df = pd.DataFrame(columns=meta_data_df.columns)

        return train_df, val_df, test_df
