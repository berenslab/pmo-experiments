"""
General-case AUROC figure.

PMO vs. four fine-tuning corpora across three eval protocols.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from pmo_experiments.evaluation.statistics import (
    bootstrap_mean_ci,
    directional_wilcoxon_vs_reference,
)
from pmo_experiments.plots._colors import (
    ACCENT_LIGHT,
    INK,
    MUTED,
    MUTED_BLUE_GREY,
)
from pmo_experiments.plots._common import (
    A4_SHORT_SIDE_IN,
    GENERAL_CORPORA,
    LABELS,
    PANEL_ORDER,
    SETTING_TITLES,
    TICK_LABELS,
    apply_style,
    filter_analysis_frame,
    get_medical_targets,
    get_target_vector,
    save,
)

ORDER = ["kaggle_eyepacs", "flair", "deepeyenet", "pubmed_ophtha", "pmc15m"]

NUMBER_PAD = 0.4  # gap from the CI's upper whisker to the number label
STAR_HEADROOM = (
    3.0  # y-axis headroom for the number + its significance star, above the tallest CI
)
PANEL_LABEL_OFFSET_PT = (
    13.86  # gap between each panel's bold letter (A/B/C) and its axis top
)


def sig_vs(
    reference: str,
    others: list[str],
    df: pd.DataFrame,
    targets: list[str],
    setting: str,
) -> dict[str, str]:
    """
    Compute Holm-corrected one-sided significance stars, reference vs each corpus.

    Args:
        reference (str): Corpus to treat as the reference (PMO).
        others (list[str]): Competitor corpora to compare against.
        df (pd.DataFrame): Score table to look up rows in.
        targets (list[str]): Evaluation target columns.
        setting (str): Eval setting to compare under.

    Returns:
        dict[str, str]: Maps each competitor's display label (via LABELS) to its
            Holm-corrected significance star.

    """
    reference_vector = get_target_vector(df, targets, reference, setting)
    competitor_vectors = {o: get_target_vector(df, targets, o, setting) for o in others}
    result = directional_wilcoxon_vs_reference(reference_vector, competitor_vectors)
    return dict(zip(result["competitor"].map(LABELS), result["signif"]))


def compute_group_ylim(
    df: pd.DataFrame,
    targets: list[str],
    settings: list[str],
    corpora: list[str] = GENERAL_CORPORA,
) -> tuple[float, float]:
    """
    Compute the bottom/top of a y-axis shared by every (corpus, setting) pair.

    Args:
        df (pd.DataFrame): Score table to look up rows in.
        targets (list[str]): Evaluation target columns.
        settings (list[str]): Eval settings to include in the shared axis.
        corpora (list[str]): Corpora to include in the shared axis.

    Returns:
        tuple[float, float]: (bottom, top) y-limits with headroom for the number
            and significance-star annotations above the tallest bootstrap CI.

    """
    los, his = [], []
    for s in settings:
        for c in corpora:
            lo, hi = bootstrap_mean_ci(100 * get_target_vector(df, targets, c, s))
            los.append(lo)
            his.append(hi)
    bottom = 5 * math.floor(min(los) / 5)
    top = max(his) + NUMBER_PAD + STAR_HEADROOM
    return bottom, top


def general_case_panel(
    ax: Axes,
    df: pd.DataFrame,
    targets: list[str],
    setting: str,
    order: list[str],
    title: str,
    ylim: tuple[float, float],
    draw_yaxis: bool = True,
) -> None:
    """
    Draw one single-panel bar chart with significance stars onto `ax`.

    Args:
        ax (Axes): Axes to draw the panel onto.
        df (pd.DataFrame): Score table to look up rows in.
        targets (list[str]): Evaluation target columns.
        setting (str): Eval setting this panel represents.
        order (list[str]): Corpora, left to right.
        title (str): Panel title.
        ylim (tuple[float, float]): Shared y-axis limits (see `compute_group_ylim`).
        draw_yaxis (bool): Whether this panel owns its y-axis, or shares one drawn
            by a panel to its left.

    """
    reference = "pubmed_ophtha"
    others = [c for c in GENERAL_CORPORA if c != reference]
    significance = sig_vs(reference, others, df, targets, setting)

    labels = [LABELS[c] for c in order]
    values = [
        100 * float(np.nanmean(get_target_vector(df, targets, c, setting)))
        for c in order
    ]
    cis = [
        bootstrap_mean_ci(100 * get_target_vector(df, targets, c, setting))
        for c in order
    ]
    colors = [ACCENT_LIGHT if c == reference else MUTED_BLUE_GREY for c in order]

    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="none", zorder=3)
    base_fs = matplotlib.rcParams["font.size"]
    for bar, corpus, val, (lo, hi) in zip(bars, order, values, cis):
        x = bar.get_x() + bar.get_width() / 2
        ax.errorbar(
            x,
            val,
            yerr=[[val - lo], [hi - val]],
            fmt="none",
            ecolor=INK,
            elinewidth=matplotlib.rcParams["axes.linewidth"],
            capsize=4,
            zorder=4,
        )
        ax.text(
            x,
            hi + NUMBER_PAD,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=base_fs,
            fontweight="medium",
            color=INK,
            zorder=5,
        )
        if corpus != reference:
            sig_text = significance[LABELS[corpus]].replace("ns", "n.s.")
            ax.annotate(
                sig_text,
                xy=(x, hi + NUMBER_PAD),
                xytext=(0, base_fs * 1.3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=base_fs,
                color=MUTED,
                zorder=5,
                annotation_clip=False,
            )
        else:
            ax.annotate(
                "ref",
                xy=(x, hi + NUMBER_PAD),
                xytext=(0, base_fs * 1.3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=base_fs,
                color=MUTED,
                zorder=5,
                annotation_clip=False,
            )

    if "pmc15m" in order:
        sep_x = order.index("pmc15m") - 0.5
        sep_top = max(hi for _, hi in cis) + NUMBER_PAD
        ax.vlines(sep_x, ylim[0], sep_top, color=MUTED, lw=0.8, zorder=1)

    ax.set_ylim(*ylim)
    ax.set_xlabel("")
    ax.tick_params(axis="both", colors=INK)
    ax.set_axisbelow(True)
    if draw_yaxis:
        ax.set_ylabel("AUROC (%)", fontweight="medium", color=INK)
        sns.despine(ax=ax, trim=True, offset=6)
    else:
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True, trim=True, offset=6)
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xticklabels([TICK_LABELS[c] for c in order], rotation=0, ha="center")
    for sp in ax.spines.values():
        sp.set_color(INK)
    ax.set_title(title, fontweight="semibold", color=INK, loc="left")


def plot_general_case_figure(
    csv_path: str, output_dir: str = "figures", dpi: int = 400
) -> None:
    """
    Build and save the 2-row (A+B on top, C below) general-case AUROC figure.

    Args:
        csv_path (str): Path to the per-target AUROC CSV (one row per trained
            (training_dataset, model_architecture, eval_setting) configuration).
        output_dir (str): Directory to save the SVG/PNG/PDF outputs into.
        dpi (int): Resolution used when saving.

    """
    score_df = pd.read_csv(csv_path)
    targets = get_medical_targets(score_df)
    analysis_df = filter_analysis_frame(score_df)

    apply_style()

    fig_height_in = 9.0 * (A4_SHORT_SIDE_IN / 11.0)
    fig = plt.figure(figsize=(A4_SHORT_SIDE_IN, fig_height_in), layout="constrained")
    grid_wspace = 0.32
    subfig_top, subfig_bottom = fig.subfigures(  # pyright: ignore[reportGeneralTypeIssues]
        nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.12
    )
    gs_top = subfig_top.add_gridspec(nrows=1, ncols=4, wspace=grid_wspace)
    gs_bottom = subfig_bottom.add_gridspec(nrows=1, ncols=4, wspace=grid_wspace)

    ax_a = subfig_top.add_subplot(gs_top[0, 0:2])
    ax_b = subfig_top.add_subplot(gs_top[0, 2:4], sharey=ax_a)
    ax_c = subfig_bottom.add_subplot(gs_bottom[0, 1:3], sharey=ax_a)

    ylim = compute_group_ylim(analysis_df, targets, PANEL_ORDER)

    general_case_panel(
        ax_a,
        analysis_df,
        targets,
        "knn_distance_weighted",
        ORDER,
        SETTING_TITLES["knn_distance_weighted"],
        ylim=ylim,
        draw_yaxis=True,
    )
    general_case_panel(
        ax_b,
        analysis_df,
        targets,
        "zero_shot",
        ORDER,
        SETTING_TITLES["zero_shot"],
        ylim=ylim,
        draw_yaxis=False,
    )
    general_case_panel(
        ax_c,
        analysis_df,
        targets,
        "linear_probing_balanced",
        ORDER,
        SETTING_TITLES["linear_probing_balanced"],
        ylim=ylim,
        draw_yaxis=True,
    )

    for ax, letter in zip([ax_a, ax_b, ax_c], ["A", "B", "C"]):
        letter_text = ax.annotate(
            letter,
            xy=(-0.12, 1.0),
            xycoords="axes fraction",
            xytext=(0, PANEL_LABEL_OFFSET_PT),
            textcoords="offset points",
            fontsize=matplotlib.rcParams["font.size"] + 4,
            fontweight="bold",
            color=INK,
            va="bottom",
            ha="left",
        )
        if ax is ax_b:
            letter_text.set_in_layout(False)

    save(fig, "figure_wilcoxon_auroc", output_dir, dpi=dpi)
    plt.close(fig)
