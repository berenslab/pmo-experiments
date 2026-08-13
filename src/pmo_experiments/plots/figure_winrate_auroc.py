"""PMO win-rate figure across eval protocols and architectures."""

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from pmo_experiments.evaluation.statistics import paired_winrate_test
from pmo_experiments.plots._colors import (
    ACCENT_LIGHT,
    INK,
    LIGHT_GREY,
    LIGHT_PINK,
    MAGENTA,
    MUTED,
    MUTED_BLUE_GREY,
)
from pmo_experiments.plots._common import (
    A4_SHORT_SIDE_IN,
    LABELS,
    SETTING_TITLES,
    TICK_LABELS,
    apply_style,
    filter_analysis_frame,
    filter_analysis_frame_arch,
    get_medical_targets,
    get_target_vector,
    save,
)

GENERAL_CORPORA_ARCH = ["pubmed_ophtha", "deepeyenet", "flair", "kaggle_eyepacs"]
ARCHITECTURES = ["RN50", "RN50x4", "ViT-B/16", "ViT-B/32", "ViT-L/14"]

# Fixed competitor order
COMPETITOR_ORDER = ["kaggle_eyepacs", "flair", "deepeyenet", "pmc15m"]
D_COMPETITOR_ORDER = [
    c for c in COMPETITOR_ORDER if c != "pmc15m"
]  # BiomedCLIP is ViT-B/16-only


D_SHADES = [
    LIGHT_PINK,
    ACCENT_LIGHT,
    MAGENTA,
]

TITLE_PAD = (
    14  # clearance (points) between each panel's plot area and its title/letter row
)
DEFAULT_TITLE_PAD = (
    6.0  # matplotlib's rcParams["axes.titlepad"], figure_auroc.py relies on this
)
PANEL_LABEL_OFFSET_PT = 13.86 + (TITLE_PAD - DEFAULT_TITLE_PAD)


def vec_pooled(
    df: pd.DataFrame, targets: list[str], corpus: str, setting: str
) -> np.ndarray:
    """
    Concatenate per-target vectors across architectures.

    Args:
        df (pd.DataFrame): Score table (architecture-pooled) to look up rows in.
        targets (list[str]): Evaluation target columns to read.
        corpus (str): Value of the "training_dataset" column to select.
        setting (str): Value of the "eval_setting" column to select.

    Returns:
        np.ndarray: Per-target scores for `corpus`/`setting`, one block per
            architecture in ARCHITECTURES, concatenated in that order.

    """
    return np.concatenate(
        [vec_arch(df, targets, corpus, setting, arch) for arch in ARCHITECTURES]
    )


def vec_arch(
    df: pd.DataFrame, targets: list[str], corpus: str, setting: str, arch: str
) -> np.ndarray:
    """
    Look up the per-target score vector for one corpus / eval setting / architecture.

    Args:
        df (pd.DataFrame): Score table to look up the row in.
        targets (list[str]): Evaluation target columns to read.
        corpus (str): Value of the "training_dataset" column to select.
        setting (str): Value of the "eval_setting" column to select.
        arch (str): Value of the "model_architecture" column to select.

    Returns:
        np.ndarray: Per-target scores for that (corpus, setting, arch) triple.

    """
    row = df[
        (df["training_dataset"] == corpus)
        & (df["eval_setting"] == setting)
        & (df["model_architecture"] == arch)
    ]
    assert len(row) == 1, (
        f"expected exactly 1 row for {corpus}/{setting}/{arch}, got {len(row)}"
    )
    return row[targets].iloc[0].astype(float).values  # pyright: ignore[reportReturnType]


def head_to_head_vitb16(
    reference: str,
    others: list[str],
    df: pd.DataFrame,
    targets: list[str],
    setting: str,
) -> list[dict]:
    """
    PMO win rate vs each competitor (ViT-B/16 only).

    Wraps `evaluation.statistics.paired_winrate_test` (Wilson CI, Holm-corrected
    binomial test) and reshapes its fractional [0, 1] output into the row-dict
    format (percentages) the panel-drawing functions expect.

    Args:
        reference (str): Corpus to treat as the reference (PMO).
        others (list[str]): Competitor corpora to compare against.
        df (pd.DataFrame): ViT-B/16-filtered score table.
        targets (list[str]): Evaluation target columns.
        setting (str): Eval setting to compare under.

    Returns:
        list[dict]: One dict per competitor with "corpus", "win_rate", "ci_low",
            "ci_high" (all in percent), "p_holm", and "signif".

    """
    reference_vector = get_target_vector(df, targets, reference, setting)
    competitor_vectors = {o: get_target_vector(df, targets, o, setting) for o in others}
    _assert_equal_pairing(reference_vector, competitor_vectors, setting, len(targets))
    result = paired_winrate_test(reference_vector, competitor_vectors)
    return [
        {
            "corpus": r["competitor"],
            "win_rate": 100 * r["win_rate"],
            "ci_low": 100 * r["ci_low"],
            "ci_high": 100 * r["ci_high"],
            "p_holm": r["p_holm"],
            "signif": r["signif"],
        }
        for _, r in result.iterrows()
    ]


def _assert_equal_pairing(
    reference_vector: np.ndarray,
    competitor_vectors: dict[str, np.ndarray],
    setting: str,
    expected_n: int,
) -> None:
    """
    Assert every competitor is paired with the reference over the same n targets.

    Every competitor is scored on the same targets, so the count of non-NaN
    (reference, competitor) pairs should be identical across competitors and
    equal to `expected_n` — only the tie count (excluded from wins/losses by
    `paired_winrate_test`) should ever differ between them.
    """
    totals = {
        o: int(np.sum(~np.isnan(reference_vector - v)))
        for o, v in competitor_vectors.items()
    }
    assert len(set(totals.values())) == 1, (
        f"{setting}: total paired-comparison count differs across competitors: {totals}"
    )
    n = next(iter(totals.values()))
    assert n == expected_n, f"{setting}: expected {expected_n} targets, got {n}"


def head_to_head_pooled(
    reference: str,
    others: list[str],
    df: pd.DataFrame,
    targets: list[str],
    setting: str,
) -> list[dict]:
    """
    PMO win rate vs each competitor, pooled over (target x architecture) pairs.

    Same wrapping as `head_to_head_vitb16`, but paired over `vec_pooled` vectors
    instead of `vec`, so each comparison spans every architecture in ARCHITECTURES.

    Args:
        reference (str): Corpus to treat as the reference (PMO).
        others (list[str]): Competitor corpora to compare against.
        df (pd.DataFrame): Architecture-pooled, filtered score table.
        targets (list[str]): Evaluation target columns.
        setting (str): Eval setting to compare under.

    Returns:
        list[dict]: One dict per competitor with "corpus", "win_rate", "ci_low",
            "ci_high" (all in percent), "p_holm", and "signif".

    """
    reference_vector = vec_pooled(df, targets, reference, setting)
    competitor_vectors = {o: vec_pooled(df, targets, o, setting) for o in others}
    _assert_equal_pairing(
        reference_vector, competitor_vectors, setting, len(targets) * len(ARCHITECTURES)
    )
    result = paired_winrate_test(reference_vector, competitor_vectors)
    return [
        {
            "corpus": r["competitor"],
            "win_rate": 100 * r["win_rate"],
            "ci_low": 100 * r["ci_low"],
            "ci_high": 100 * r["ci_high"],
            "p_holm": r["p_holm"],
            "signif": r["signif"],
        }
        for _, r in result.iterrows()
    ]


def _finish_shared_yaxis(ax: Axes, draw_yaxis: bool) -> None:
    """Draw or hide this panel's own y-axis, depending on whether it shares one."""
    if draw_yaxis:
        ax.set_ylabel("PubMed-Ophtha win rate (%)", fontweight="medium", color=INK)
        sns.despine(ax=ax, offset=6)
    else:
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True, offset=6)
        ax.tick_params(axis="y", left=False, labelleft=False)
    for sp in ax.spines.values():
        sp.set_color(INK)


def winrate_panel(
    ax: Axes,
    df: pd.DataFrame,
    targets: list[str],
    setting: str,
    title: str,
    draw_yaxis: bool = True,
) -> None:
    """
    Draw one panel (A/B/C): PMO head-to-head win rate vs each standard competitor.

    ViT-B/16 only. Bars use ACCENT_LIGHT where the win rate is significantly
    different from chance (Holm-corrected) and MUTED_BLUE_GREY otherwise.

    Args:
        ax (Axes): Axes to draw the panel onto.
        df (pd.DataFrame): ViT-B/16-filtered score table.
        targets (list[str]): Evaluation target columns.
        setting (str): Eval setting this panel represents.
        title (str): Panel title.
        draw_yaxis (bool): Whether this panel owns its y-axis (see
            `_finish_shared_yaxis`).

    """
    by_corpus = {
        r["corpus"]: r
        for r in head_to_head_vitb16(
            "pubmed_ophtha", COMPETITOR_ORDER, df, targets, setting
        )
    }
    rows = [by_corpus[c] for c in COMPETITOR_ORDER]
    fs = matplotlib.rcParams["font.size"]

    for xi, r in enumerate(rows):
        wr = r["win_rate"]
        color = ACCENT_LIGHT if r["p_holm"] < 0.05 else MUTED_BLUE_GREY
        ax.bar(xi, wr, 0.62, color=color, zorder=3)
        ax.errorbar(
            xi,
            wr,
            yerr=[[wr - r["ci_low"]], [r["ci_high"] - wr]],
            fmt="none",
            ecolor=INK,
            elinewidth=matplotlib.rcParams["axes.linewidth"],
            capsize=4,
            zorder=4,
        )
        ax.text(
            xi,
            r["ci_high"] + 1.5,
            r["signif"].replace("ns", "n.s."),
            ha="center",
            va="bottom",
            fontsize=fs,
            color=MUTED,
            zorder=4,
        )
        ax.text(
            xi,
            2.5,
            f"{wr:.0f}%",
            ha="center",
            va="bottom",
            fontsize=fs,
            fontweight="medium",
            color=INK,
            zorder=5,
        )

    sep_x = len(rows) - 1 - 0.5
    sep_top = max(r["ci_high"] for r in rows) + 1.5
    ax.vlines(sep_x, 0, sep_top, color=LIGHT_GREY, lw=0.8, zorder=1)

    ax.axhline(50, ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([TICK_LABELS[r["corpus"]] for r in rows])
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.tick_params(axis="both", colors=INK)
    ax.set_axisbelow(True)
    _finish_shared_yaxis(ax, draw_yaxis)
    ax.set_title(title, fontweight="semibold", color=INK, loc="left", pad=TITLE_PAD)


def pooled_winrate_panel(
    ax: Axes,
    df: pd.DataFrame,
    targets: list[str],
    draw_yaxis: bool = True,
) -> None:
    """
    Draw panel D: pooled (across all 5 architectures) PMO win rate per protocol.

    One group of bars per protocol, one bar per competitor
    in D_COMPETITOR_ORDER, shaded from light pink to brand magenta (D_SHADES).

    Args:
        ax (Axes): Axes to draw the panel onto.
        df (pd.DataFrame): Architecture-pooled, filtered score table.
        targets (list[str]): Evaluation target columns.
        draw_yaxis (bool): Whether this panel owns its y-axis (see
            `_finish_shared_yaxis`).

    """
    fs = matplotlib.rcParams["font.size"]
    per_setting = {
        s: {
            r["corpus"]: r
            for r in head_to_head_pooled(
                "pubmed_ophtha", D_COMPETITOR_ORDER, df, targets, s
            )
        }
        for s in SETTING_TITLES.keys()
    }
    n = len(D_COMPETITOR_ORDER)
    width = 0.8 / n
    x = np.arange(len(SETTING_TITLES))
    for k, o in enumerate(D_COMPETITOR_ORDER):
        xpos = x + (k - (n - 1) / 2) * width
        for xi, s in zip(xpos, list(SETTING_TITLES.keys())):
            r = per_setting[s][o]
            wr = r["win_rate"]
            ax.bar(xi, wr, width, color=D_SHADES[k % len(D_SHADES)], zorder=3)
            ax.errorbar(
                xi,
                wr,
                yerr=[[wr - r["ci_low"]], [r["ci_high"] - wr]],
                fmt="none",
                ecolor=INK,
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            ax.text(
                xi,
                r["ci_high"] + 1.5,
                r["signif"].replace("ns", "n.s."),
                ha="center",
                va="bottom",
                fontsize=fs - 1,
                color=MUTED,
                zorder=4,
            )
    ax.axhline(50, ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(SETTING_TITLES.values()))
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.tick_params(axis="both", colors=INK)
    ax.set_axisbelow(True)
    _finish_shared_yaxis(ax, draw_yaxis)
    ax.set_title(
        "Pooled across architectures",
        fontweight="semibold",
        color=INK,
        loc="left",
        pad=TITLE_PAD,
    )


def plot_winrate_figure(
    csv_path: str, output_dir: str = "figures", dpi: int = 400
) -> None:
    """
    Build and save the 2x2 (A-D) PMO win-rate figure.

    Args:
        csv_path (str): Path to the per-target AUROC CSV (one row per trained
            (training_dataset, model_architecture, eval_setting) configuration).
        output_dir (str): Directory to save the SVG/PNG/PDF outputs into.
        dpi (int): Resolution used when saving.

    """
    score_df = pd.read_csv(csv_path)
    targets = get_medical_targets(score_df)
    analysis_df_vitb16 = filter_analysis_frame(score_df)
    analysis_df_arch = filter_analysis_frame_arch(score_df)

    apply_style()

    fig_width_in = A4_SHORT_SIDE_IN
    fig_height_in = fig_width_in * 0.8
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), layout="constrained")
    gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.08, hspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], sharey=ax_a)
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)

    winrate_panel(
        ax_a,
        analysis_df_vitb16,
        targets,
        "knn_distance_weighted",
        SETTING_TITLES["knn_distance_weighted"],
        draw_yaxis=True,
    )
    winrate_panel(
        ax_b,
        analysis_df_vitb16,
        targets,
        "zero_shot",
        SETTING_TITLES["zero_shot"],
        draw_yaxis=False,
    )
    winrate_panel(
        ax_c,
        analysis_df_vitb16,
        targets,
        "linear_probing_balanced",
        SETTING_TITLES["linear_probing_balanced"],
        draw_yaxis=True,
    )
    pooled_winrate_panel(ax_d, analysis_df_arch, targets, draw_yaxis=False)

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        ax.annotate(
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

    legend_handles = [
        matplotlib.lines.Line2D(
            [], [], ls=(0, (4, 3)), lw=1.0, color=MUTED, label="Random (50%)"
        ),
    ] + [
        matplotlib.patches.Patch(color=D_SHADES[k], label=f"D: vs {LABELS[o]}")
        for k, o in enumerate(D_COMPETITOR_ORDER)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=matplotlib.rcParams["font.size"] - 1,
        frameon=False,
    )

    save(fig, "figure_winrate_auroc", output_dir, dpi=dpi)
    plt.close(fig)
