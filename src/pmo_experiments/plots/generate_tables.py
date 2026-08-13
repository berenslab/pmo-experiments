"""Generate the tables from the evaluation results for the paper."""

import os

import pandas as pd

from pmo_experiments.evaluation.statistics import tost_wilcoxon_vs_reference
from pmo_experiments.plots._common import (
    LABELS,
    SETTING_TITLES,
    filter_analysis_frame,
    get_medical_targets,
)

DR_COLUMNS = [
    "aptos2019::diagnosis",
    "brset::DR_ICDR",
    "brset::diabetic_retinopathy",
    "eddfs::DR",
    "rfmid_1::DR",
    "rfmid_2::DR",
]

AMD_COLUMNS = [
    "brset::amd",
    "eddfs::AMD",
    "rfmid_1::ARMD",
    "rfmid_2::ARMD",
]

GLAUCOMA_COLUMNS = [
    "eddfs::glaucoma",
    "refuge2::glaucoma_risk",
]

MYOPIA_COLUMNS = [
    "brset::myopic_fundus",
    "eddfs::myopia",
    "rfmid_1::MYA",
    "rfmid_2::MYA",
]

HYPERTENSIVE_RETINOPATHY_COLUMNS = [
    "brset::hypertensive_retinopathy",
    "eddfs::hyper",
    "rfmid_2::HTN",
]

AH_COLUMNS = ["rfmid_1::AH", "rfmid_2::AH"]

MACULAR_SCAR_COLUMNS = ["rfmid_1::MS", "rfmid_2::MS"]

DISEASE_RISK_COLUMNS = ["eddfs::normal", "rfmid_1::Disease_Risk", "rfmid_2::WNL"]

MODEL_ORDER = ["kaggle_eyepacs", "flair", "deepeyenet", "pubmed_ophtha", "pmc15m"]

TOST_REFERENCE = "pubmed_ophtha"
TOST_VARIANTS = [
    "pubmed_ophtha_cfp",
    "pubmed_ophtha_cfp_matched",
    "pubmed_ophtha_decon",
]
TOST_MARGIN = 2.0

VARIANT_NAME_MAP = {
    "pubmed_ophtha_cfp": "PubMed-Ophtha-CFP",
    "pubmed_ophtha_cfp_matched": "PubMed-Ophtha-CFP-matched",
    "pubmed_ophtha_decon": "PubMed-Ophtha-decontaminated",
}


def generate_disease_table(df: pd.DataFrame, output_path: str):
    """
    Generate a table for each eval setting and save as csv.

    Creates one table per setting (knn, zero-shot, linear probing).
    Each table has the task groups as well as the average across all tasks.

    Args:
        df (pd.DataFrame): Evaluation result df.
        output_path (str): Folder to save the tables in.

    """
    filtered_df = (
        df[df["training_dataset"].isin(MODEL_ORDER)]
        .drop(columns=["model_architecture", "batch_size", "pretraining_weights"])
        .copy()
    )

    target_columns = get_medical_targets(filtered_df)

    for setting in SETTING_TITLES.keys():
        setting_df = (
            filtered_df[filtered_df["eval_setting"] == setting]
            .drop(columns=["eval_setting"])
            .set_index("training_dataset")
        ) * 100

        dr_scores = setting_df[DR_COLUMNS].mean(axis=1).rename("DR")
        amd_scores = setting_df[AMD_COLUMNS].mean(axis=1).rename("AMD")
        glaucoma_scores = setting_df[GLAUCOMA_COLUMNS].mean(axis=1).rename("Glaucoma")
        myopia_scores = setting_df[MYOPIA_COLUMNS].mean(axis=1).rename("Myopia")
        hypertensive_retinopathy_scores = (
            setting_df[HYPERTENSIVE_RETINOPATHY_COLUMNS]
            .mean(axis=1)
            .rename("Hypertensive Retinopathy")
        )
        asteroid_hyalosis_scores = (
            setting_df[AH_COLUMNS].mean(axis=1).rename("Asteroid Hyalosis")
        )
        macular_scar_scores = (
            setting_df[MACULAR_SCAR_COLUMNS].mean(axis=1).rename("Macular Scar")
        )
        disease_risk_scores = (
            setting_df[DISEASE_RISK_COLUMNS].mean(axis=1).rename("Disease Risk")
        )

        eval_setting_series = pd.Series(
            [setting] * len(dr_scores), index=dr_scores.index, name="eval_setting"
        )

        average_scores = setting_df[target_columns].mean(axis=1).rename("Average Score")

        disease_df = pd.concat(
            [
                eval_setting_series,
                disease_risk_scores,
                amd_scores,
                glaucoma_scores,
                myopia_scores,
                hypertensive_retinopathy_scores,
                asteroid_hyalosis_scores,
                macular_scar_scores,
                dr_scores,
                average_scores,
            ],
            axis=1,
        ).reset_index()

        disease_df = disease_df[disease_df["training_dataset"].isin(LABELS)]
        disease_df["training_dataset"] = disease_df["training_dataset"].map(
            lambda x: LABELS[x]  # pyright: ignore[reportArgumentType]
        )

        # sort by general models order
        disease_df["training_dataset"] = pd.Categorical(
            disease_df["training_dataset"],
            categories=list(LABELS.values()),
            ordered=True,
        )
        disease_df = disease_df.sort_values("training_dataset").reset_index(drop=True)

        disease_df.to_csv(
            os.path.join(output_path, f"disease_table_{setting}.csv"), index=False
        )


def generate_variants_table(
    df: pd.DataFrame, output_path: str, margin: float = TOST_MARGIN
):
    """
    Generate a table for the TOST of the variants and save as csv.

    Creates one table per setting (knn, zero-shot, linear probing).
    Each table has the average score across all tasks in the setting as well as the Holm
    corrected p value of the TOST against the margin.

    Args:
        df (pd.DataFrame): Evaluation result df.
        output_path (str): Folder to save the tables in.
        margin (float): Margin for the TOST. Defaults to TOST_MARGIN.

    """
    filtered_df = (
        df[df["training_dataset"].isin([TOST_REFERENCE] + TOST_VARIANTS)]
        .drop(columns=["model_architecture", "batch_size", "pretraining_weights"])
        .copy()
    )

    target_columns = get_medical_targets(filtered_df)

    for setting in SETTING_TITLES.keys():
        setting_df = (
            filtered_df[filtered_df["eval_setting"] == setting]
            .drop(columns=["eval_setting"])
            .set_index("training_dataset")
        ) * 100

        average_scores = setting_df[target_columns].mean(axis=1)

        reference_vector = setting_df.loc[TOST_REFERENCE, target_columns].to_numpy(
            dtype=float
        )
        competitor_vectors = {
            variant: setting_df.loc[variant, target_columns].to_numpy(dtype=float)
            for variant in TOST_VARIANTS
        }
        tost_result = tost_wilcoxon_vs_reference(
            reference_vector, competitor_vectors, margin=margin
        ).set_index("competitor")

        tost_df = average_scores.rename("Average Score").to_frame()
        tost_df["p_holm"] = tost_result["p_holm"]
        tost_df["signif"] = tost_result["signif"]
        tost_df = tost_df.reindex([TOST_REFERENCE] + TOST_VARIANTS)
        tost_df = tost_df.reset_index(names="training_dataset")
        tost_df["eval_setting"] = setting

        tost_df["training_dataset"] = tost_df["training_dataset"].map(
            lambda x: VARIANT_NAME_MAP.get(x, x) if x != TOST_REFERENCE else LABELS[x]  # pyright: ignore[reportArgumentType,reportCallIssue]
        )

        tost_df.to_csv(
            os.path.join(output_path, f"variant_table_{setting}.csv"), index=False
        )


def generate_tables(
    csv_path: str,
    output_dir: str = "eval_tables",
):
    """
    Generate the disease and variant tables from the evaluation results.

    Args:
        csv_path (str): Path to the evaluation results csv.
        output_dir (str, optional): Folder to save the tables in. Defaults to
            "eval_tables".

    """
    df = filter_analysis_frame(pd.read_csv(csv_path))

    os.makedirs(output_dir, exist_ok=True)

    generate_disease_table(df, output_dir)
    generate_variants_table(df, output_dir)
