import pandas as pd
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def get_paired_differences(
    df,
    metric,
    method_index,
    reference,
    paired_indices
):
    """
    Compute paired differences between a reference method and all
    other methods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a MultiIndex.

    metric : str
        Column containing the metric to compare.

    method_index : str
        Name of the MultiIndex level identifying the methods.

    reference : str
        Reference method.

    paired_indices : list of str
        MultiIndex levels identifying paired observations.

    Returns
    -------
    pd.DataFrame
        One row per paired observation and one column per method.
        Values correspond to:

            reference - method
    """

    data = (
        df[[metric]]
        .reset_index()
        .pivot(
            index=paired_indices,
            columns=method_index,
            values=metric
        )
    )

    if reference not in data.columns:
        raise ValueError(
            f"Reference method '{reference}' not found."
        )

    methods = [
        method for method in data.columns
        if method != reference
    ]

    differences = pd.DataFrame(
        {
            method: data[reference] - data[method]
            for method in methods
        },
        index=data.index
    )

    return differences

def paired_statistical_analysis(
    df,
    metric,
    method_index,
    reference,
    paired_indices,
    analysis_indices,
    n_boot=10000,
    confidence=0.95,
    random_state=42
):
    """
    Paired statistical comparison of a reference method against
    all other methods.

    Differences are defined as:

        reference - method

    Repeated observations not included in analysis_indices are
    averaged before statistical inference.
    """

    # ---------------------------------------------------------
    # Paired differences
    # ---------------------------------------------------------

    differences = get_paired_differences(
        df=df,
        metric=metric,
        method_index=method_index,
        reference=reference,
        paired_indices=paired_indices
    )

    # ---------------------------------------------------------
    # Aggregate repeated observations
    # ---------------------------------------------------------

    block_diff = (
        differences
        .groupby(level=analysis_indices)
        .mean()
    )

    # ---------------------------------------------------------
    # Statistical analysis
    # ---------------------------------------------------------

    rng = np.random.default_rng(random_state)

    results = []

    alpha = 1.0 - confidence

    for method in block_diff.columns:

        values = block_diff[method].dropna().to_numpy()

        # Wilcoxon signed-rank test
        W, p = wilcoxon(
            values,
            alternative="two-sided"
        )

        # Bootstrap confidence interval of the mean difference
        boot_means = np.empty(n_boot)

        for i in range(n_boot):

            sample = rng.choice(
                values,
                size=len(values),
                replace=True
            )

            boot_means[i] = sample.mean()

        ci_low, ci_high = np.quantile(
            boot_means,
            [alpha / 2, 1 - alpha / 2]
        )

        results.append({
            "Comparison": reference + " - " + method,
            # "N": len(values),
            "Mean difference": values.mean(),
            # "Median difference": np.median(values),
            "95% CI": [float(ci_low.round(3)), float(ci_high.round(3))],
            # "CI high": ci_high,
            # "W": W,
            "p": p
        })

    results = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Holm correction
    # ---------------------------------------------------------

    _, p_holm, _, _ = multipletests(
        results["p"],
        method="holm"
    )

    results["Adjusted p-value"] = p_holm
    results.pop("p")

    return results, differences, block_diff