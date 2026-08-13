"""Module for statistical evaluation of model performance across different groups."""

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


def signif_label(p: float) -> str:
    """
    Convert p-value into significance label.

    Args:
        p (float): P-value to convert.

    Returns:
        str: Significance label corresponding to the p-value.

    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def wilcoxon_test(
    df: pd.DataFrame, index_column_names: list[str], eval_column_names: list[str]
) -> pd.DataFrame:
    """
    Perform Friedman test and pairwise Wilcoxon signed-rank tests on the given df.

    Args:
        df (pd.DataFrame): Dataframe containing the evaluation results. Must include
            columns specified in index_column_names and eval_column_names.
        index_column_names (list[str]): List of column names to use as indices for
            grouping (e.g., training_dataset, disease). If multiple columns are
            provided, they will be combined into a single index.
        eval_column_names (list[str]): List of column names containing the evaluation
            metrics to compare across groups.

    Returns:
        pd.DataFrame: Dataframe containing the pairwise Wilcoxon test results,
            including the statistic, raw p-value, adjusted p-value, and significance
            label for each pair of groups. If multiple index columns were provided, the
            group identifiers will be split back into their original components.

    """
    assert len(index_column_names) > 0, "Need at least one index column"
    assert len(eval_column_names) > 0, "Need at least one eval column"
    assert all(col in df.columns for col in index_column_names), (
        f"Index columns not all in dataframe: {index_column_names}"
    )
    assert all(col in df.columns for col in eval_column_names), (
        f"Eval columns not all in dataframe: {eval_column_names}"
    )
    # Subset: training_dataset + disease columns
    index_column_name = "model_setting"
    column_name_map = None
    if len(index_column_names) == 1:
        index_column_name = index_column_names[0]
    else:
        df[index_column_name] = df[index_column_names].agg("_".join, axis=1)
        unique_combinations = df[index_column_names].drop_duplicates()
        column_name_map = {
            "_".join(combination): combination
            for _, combination in unique_combinations.iterrows()
        }
    df = df[[index_column_name] + eval_column_names]

    all_index_values = sorted(set(df[index_column_name]))

    # Filter and order datasets
    df[index_column_name] = pd.Categorical(
        df[index_column_name], categories=all_index_values, ordered=True
    )
    df = df.sort_values(index_column_name)

    # Pivot to long
    long = df.melt(
        id_vars=index_column_name,
        value_vars=eval_column_names,
        var_name="evaluation_target",
        value_name="value",
    )
    long = long.dropna(subset=["value"])

    # --- Friedman test ---
    # Pivot to wide
    wide = long.pivot_table(
        index="evaluation_target",
        columns=index_column_name,
        values="value",
        aggfunc="mean",
        observed=False,
    )
    wide = wide.dropna()
    arrays = [wide[ds].values for ds in all_index_values]
    friedman_stat, friedman_p = stats.friedmanchisquare(*arrays)

    # --- Pairwise Wilcoxon (paired, same diseases) ---
    pairs = list(combinations(all_index_values, 2))
    wilcoxon_stats = []
    p_values = []
    for ds1, ds2 in pairs:
        a = wide[ds1].values
        b = wide[ds2].values
        w_stat, p_val = stats.wilcoxon(a, b)
        wilcoxon_stats.append(w_stat)
        p_values.append(p_val)

    _, p_adj, _, _ = multipletests(p_values, method="holm")

    rows = []
    for (ds1, ds2), w_stat, p_raw, p_a in zip(pairs, wilcoxon_stats, p_values, p_adj):  # pyright: ignore[reportArgumentType]
        rows.append(
            {
                "group1": ds1,
                "group2": ds2,
                "statistic": w_stat,
                "p": p_raw,
                "p_adj": p_a,
                "p_adj_signif": signif_label(p_a),
            }
        )
    wilcoxon_df = pd.DataFrame(rows)

    if len(index_column_names) > 1 and column_name_map is not None:
        # Undo the combined index column
        for col in index_column_names:
            wilcoxon_df[f"group1_{col}"] = wilcoxon_df["group1"].map(
                lambda x: column_name_map[x][index_column_names.index(col)]  # pyright: ignore[reportArgumentType]
            )
            wilcoxon_df[f"group2_{col}"] = wilcoxon_df["group2"].map(
                lambda x: column_name_map[x][index_column_names.index(col)]  # pyright: ignore[reportArgumentType]
            )
        wilcoxon_df = wilcoxon_df.drop(columns=["group1", "group2"])
        wilcoxon_df = wilcoxon_df[
            ["group1_" + col for col in index_column_names]
            + ["group2_" + col for col in index_column_names]
            + ["statistic", "p", "p_adj", "p_adj_signif"]
        ]
    else:
        wilcoxon_df = wilcoxon_df.rename(
            columns={
                "group1": f"group1_{index_column_name}",
                "group2": f"group2_{index_column_name}",
            }
        )

    return wilcoxon_df


def directional_wilcoxon_vs_reference(
    reference_vector: np.ndarray,
    competitor_vectors: Mapping[str, np.ndarray],
    alternative: str = "greater",
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """
    Compare a reference vector against several competitors with a directional test.

    Runs a paired one-sided Wilcoxon signed-rank test between reference_vector and
    each vector in competitor_vectors, then Holm-corrects the resulting p-values
    across all competitors.

    Args:
        reference_vector (np.ndarray): Per-target scores for the reference group.
        competitor_vectors (Mapping[str, np.ndarray]): Per-target scores for each
            competitor group, keyed by competitor name. Each vector must be paired
            element-wise with reference_vector (same targets, same order).
        alternative (str): Alternative hypothesis passed to scipy.stats.wilcoxon.
            Defaults to "greater" (reference_vector tends to exceed the competitor).
        zero_method (str): Zero-handling method passed to scipy.stats.wilcoxon.

    Returns:
        pd.DataFrame: One row per competitor with columns "competitor", "n_pairs",
            "p" (raw p-value), "p_holm" (Holm-adjusted p-value), and "signif"
            (star-notation significance label of p_holm).

    """
    names = list(competitor_vectors.keys())
    n_pairs = []
    p_values = []
    for name in names:
        diff = np.asarray(reference_vector) - np.asarray(competitor_vectors[name])
        diff = diff[~np.isnan(diff)]
        n_pairs.append(diff.size)
        if np.any(diff != 0):
            p = stats.wilcoxon(
                diff, alternative=alternative, zero_method=zero_method
            ).pvalue  # pyright: ignore[reportAttributeAccessIssue]
        else:
            p = float("nan")
        p_values.append(p)

    _, p_holm, _, _ = multipletests(list(p_values), method="holm")
    return pd.DataFrame(
        {
            "competitor": names,
            "n_pairs": n_pairs,
            "p": p_values,
            "p_holm": p_holm,
            "signif": [signif_label(p) for p in p_holm],  # pyright: ignore[reportOptionalIterable]
        }
    )


def tost_wilcoxon_vs_reference(
    reference_vector: np.ndarray,
    competitor_vectors: Mapping[str, np.ndarray],
    margin: float,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """
    Test equivalence of a reference vector to several competitors via TOST.

    For each competitor, runs the two one-sided tests (TOST) procedure: two paired
    one-sided Wilcoxon signed-rank tests bounding the difference
    (reference_vector - competitor_vector) within [-margin, margin], taking the
    larger of the two p-values as the TOST p-value. The resulting p-values are then
    Holm-corrected across all competitors. A significant p is a positive claim that
    the difference lies within the margin, not merely a failure to reject "no
    difference".

    Args:
        reference_vector (np.ndarray): Per-target scores for the reference group.
        competitor_vectors (Mapping[str, np.ndarray]): Per-target scores for each
            competitor group, keyed by competitor name. Each vector must be paired
            element-wise with reference_vector (same targets, same order).
        margin (float): Equivalence margin, in the same units as the scores.
        zero_method (str): Zero-handling method passed to scipy.stats.wilcoxon.

    Returns:
        pd.DataFrame: One row per competitor with columns "competitor", "n_pairs",
            "p" (raw TOST p-value), "p_holm" (Holm-adjusted p-value), and "signif"
            (star-notation significance label of p_holm).

    """
    names = list(competitor_vectors.keys())
    n_pairs = []
    p_values = []
    for name in names:
        diff = np.asarray(reference_vector) - np.asarray(competitor_vectors[name])
        diff = diff[~np.isnan(diff)]
        n_pairs.append(diff.size)

        lower_diff = diff + margin
        if np.any(lower_diff != 0):
            p_lower = stats.wilcoxon(
                lower_diff, alternative="greater", zero_method=zero_method
            ).pvalue  # pyright: ignore[reportAttributeAccessIssue]
        else:
            p_lower = float("nan")

        upper_diff = diff - margin
        if np.any(upper_diff != 0):
            p_upper = stats.wilcoxon(
                upper_diff, alternative="less", zero_method=zero_method
            ).pvalue  # pyright: ignore[reportAttributeAccessIssue]
        else:
            p_upper = float("nan")

        p_values.append(max(p_lower, p_upper))

    _, p_holm, _, _ = multipletests(list(p_values), method="holm")
    return pd.DataFrame(
        {
            "competitor": names,
            "n_pairs": n_pairs,
            "p": p_values,
            "p_holm": p_holm,
            "signif": [signif_label(p) for p in p_holm],  # pyright: ignore[reportOptionalIterable]
        }
    )


def paired_winrate_test(
    reference_vector: np.ndarray,
    competitor_vectors: Mapping[str, np.ndarray],
    alternative: str = "two-sided",
    ci_method: str = "wilson",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compare a reference vector against several competitors with a win-rate test.

    Args:
        reference_vector (np.ndarray): Per-target scores for the reference group.
        competitor_vectors (Mapping[str, np.ndarray]): Per-target scores for each
            competitor group, keyed by competitor name. Each vector must be paired
            element-wise with reference_vector (same targets, same order).
        alternative (str): Alternative hypothesis passed to scipy.stats.binomtest.
        ci_method (str): Confidence interval method passed to
            scipy.stats.binomtest(...).proportion_ci.
        confidence_level (float): Confidence level for the win-rate interval.

    Returns:
        pd.DataFrame: One row per competitor with columns "competitor", "wins",
            "losses", "n_pairs", "win_rate", "ci_low", "ci_high", "p" (raw
            p-value), "p_holm" (Holm-adjusted p-value), and "signif" (star-notation
            significance label of p_holm).

    """
    names = list(competitor_vectors.keys())
    rows = []
    p_values = []
    for name in names:
        diff = np.asarray(reference_vector) - np.asarray(competitor_vectors[name])
        diff = diff[~np.isnan(diff)]
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        n_pairs = wins + losses
        binom_result = stats.binomtest(wins, n_pairs, 0.5, alternative=alternative)
        ci = binom_result.proportion_ci(
            confidence_level=confidence_level, method=ci_method
        )
        p_values.append(binom_result.pvalue)
        rows.append(
            {
                "competitor": name,
                "wins": wins,
                "losses": losses,
                "n_pairs": n_pairs,
                "win_rate": wins / n_pairs,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "p": binom_result.pvalue,
            }
        )

    _, p_holm, _, _ = multipletests(list(p_values), method="holm")
    result = pd.DataFrame(rows)
    result["p_holm"] = p_holm
    result["signif"] = [signif_label(p) for p in p_holm]  # pyright: ignore[reportOptionalIterable]
    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci_level: float = 95.0,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a percentile bootstrap confidence interval for the mean of values.

    Args:
        values (np.ndarray): Samples to bootstrap over (e.g. per-target scores).
        n_boot (int): Number of bootstrap resamples.
        ci_level (float): Confidence level of the interval, in percent.
        seed (int): Seed for the random number generator.

    Returns:
        tuple[float, float]: Lower and upper bounds of the confidence interval.

    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(
        axis=1
    )
    alpha = (100 - ci_level) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(lo), float(hi)
