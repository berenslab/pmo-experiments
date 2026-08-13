"""Shared building blocks for the figure-plotting functions in this package."""

import os
import subprocess

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ._colors import INK

LABELS = {
    "pubmed_ophtha": "PubMed-Ophtha",
    "deepeyenet": "DeepEyeNet",
    "flair": "FLAIR",
    "kaggle_eyepacs": "Kaggle EyePACS",
    "pmc15m": "BiomedCLIP",
}
GENERAL_CORPORA = ["pubmed_ophtha", "deepeyenet", "flair", "kaggle_eyepacs", "pmc15m"]

NON_MEDICAL = [
    "brset::diabetes",
    "brset::exam_eye",
    "brset::image_field",
    "brset::patient_sex",
    "brset::quality",
]
JOINT_RFMID_PREFIX = "joint_rfmid::"

# User font directory
_LOCAL_FONT_DIR = os.path.expanduser("~/fonts")

SETTING_TITLES = {
    "knn_distance_weighted": "k-NN",
    "linear_probing_balanced": "Linear probing",
    "zero_shot": "Zero-shot",
}

TICK_LABELS = {
    "pubmed_ophtha": "PubMed-\nOphtha",
    "deepeyenet": "Deep-\nEyeNet",
    "flair": "FLAIR",
    "kaggle_eyepacs": "Kaggle\nEyePACS",
    "pmc15m": "Biomed-\nCLIP",
}

PANEL_ORDER = ["knn_distance_weighted", "zero_shot", "linear_probing_balanced"]


A4_SHORT_SIDE_IN = 8.27  # print target width: A4's short side (210 mm)


def ensure_font() -> None:
    """
    Register Arial with matplotlib, expanding with user dir if necessary.

    Raises:
        RuntimeError: If Arial cannot be found or matplotlib fails to register it.

    """
    if any(f.name == "Arial" for f in fm.fontManager.ttflist):
        return
    out = subprocess.run(
        ["fc-list", ":family=Arial", "--format=%{file}\n"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    paths = [p for p in out.splitlines() if p]
    if len(paths) == 0 and os.path.isdir(_LOCAL_FONT_DIR):
        for p in fm.findSystemFonts(fontpaths=[_LOCAL_FONT_DIR]):
            try:
                if fm.get_font(p).family_name == "Arial":
                    paths.append(p)
            except Exception:  # noqa: BLE001 - unreadable/broken font file
                continue
    if len(paths) == 0:
        raise RuntimeError("Font 'Arial' is not installed; install it and retry.")
    for p in paths:
        fm.fontManager.addfont(p)
    if not any(f.name == "Arial" for f in fm.fontManager.ttflist):
        raise RuntimeError(
            "Found Arial font files but matplotlib did not register them."
        )


def apply_style() -> None:
    """
    Apply the shared figure style to matplotlib.

    Registers Arial via ensure_font().
    """
    ensure_font()
    sns.set_theme(style="ticks", font="Arial")
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": "large",
            "axes.labelsize": "medium",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "axes.linewidth": 1.0,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "legend.fancybox": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def get_medical_targets(score_df: pd.DataFrame) -> list[str]:
    """
    Return the per-target AUROC columns, dropping non-medical and joint_rfmid ones.

    Args:
        score_df (pd.DataFrame): Raw per-target AUROC scores, one column per
            "dataset::target" metric.

    Returns:
        list[str]: Column names to treat as evaluation targets.

    """
    metric_columns = [c for c in score_df.columns if "::" in c]
    targets = [
        c
        for c in metric_columns
        if c not in NON_MEDICAL and not c.startswith(JOINT_RFMID_PREFIX)
    ]
    return targets


def filter_analysis_frame_arch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to batch size 2048, openai weights preferred, drop flair_mini (all archs).

    Same as `_common.filter_analysis_frame` but without the ViT-B/16 restriction, so
    it keeps every architecture in the cross-architecture comparison (panel D).

    Args:
        df (pd.DataFrame): Raw per-run score table.

    Returns:
        pd.DataFrame: The filtered subset used for the pooled-architecture comparison.

    """
    return df[
        ((df["batch_size"].isna()) | (df["batch_size"] == 2048))
        & (df["training_dataset"] != "flair_mini")
        & ((df["batch_size"].isna()) | (df["pretraining_weights"] == "openai"))
    ].copy()


def filter_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to relevant entries.

    Args:
        df (pd.DataFrame): Raw per-run score table with one row per trained
            (training_dataset, eval_setting) configuration.

    Returns:
        pd.DataFrame: The filtered subset used for the ViT-B/16 comparisons.

    """
    df = filter_analysis_frame_arch(df)
    return df[df["model_architecture"] == "ViT-B/16"].copy()


def get_target_vector(
    df: pd.DataFrame, targets: list[str], training_data_source: str, setting: str
) -> np.ndarray:
    """
    Get the per-target score vector for one training data source under one eval setting.

    Args:
        df (pd.DataFrame): Score table to look up the row in.
        targets (list[str]): Evaluation target columns to read.
        training_data_source (str): Value of the "training_dataset" column to select.
        setting (str): Value of the "eval_setting" column to select.

    Returns:
        np.ndarray: Per-target scores for that (training_data_source, setting) pair.

    """
    row = df[
        (df["training_dataset"] == training_data_source)
        & (df["eval_setting"] == setting)
    ]

    return row[targets].iloc[0].astype(float).values  # pyright: ignore[reportReturnType]


def save(
    fig: Figure, name: str, output_dir: str, dpi: int = 400
) -> tuple[str, str, str]:
    """
    Save fig as SVG, PNG, PDF under output_dir.

    Args:
        fig (Figure): Figure to save.
        name (str): Base filename (without extension) for the three outputs.
        output_dir (str): Directory to save into; created if missing.
        dpi (int): Resolution passed to each Figure.savefig call.

    Returns:
        tuple[str, str, str]: Paths to the saved SVG, PNG, and PDF files.

    """
    os.makedirs(output_dir, exist_ok=True)
    svg = os.path.join(output_dir, f"{name}.svg")
    png = os.path.join(output_dir, f"{name}.png")
    pdf = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(svg, bbox_inches="tight", dpi=dpi)
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf, bbox_inches="tight", dpi=dpi)
    return svg, png, pdf
