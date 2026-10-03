import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, kendalltau
import argparse
from analysis_utils import create_dictionary_from_results, dict_to_df_fold_seed_metric

def plot_dataset_heatmaps(
    df,
    metrics=("F1-score", "AUC"),
    preprocessings=("Bl", "A1"),
    lesion="Masses",
    figsize=(9, 12)
):
    """
    Generate a 2x2 heatmap figure:

                        Metric 1       Metric 2
        Standard          H              H
        Exp.-weighted     H              H

    Rows within each metric are ordered according to the
    Standard metric evaluated on the complete dataset ('all').

    Standard and explainability-weighted heatmaps share the
    same color scale within each metric.
    """

    # ---------------------------------------------------------
    # 1. Prepare DataFrame
    # ---------------------------------------------------------

    data = df.copy()

    data.index = data.index.set_names([
        "Dataset",
        "Preprocessing",
        "Model",
        "Seed",
        "Fold"
    ])

    data = data.reset_index()

    datasets = [
        "all",
        "VinDr",
        "INbreast"
    ]

    models = [
        "DenseNet",
        "EfficientNet",
        "MobileNet",
        "ResNet18",
        "ResNet50"
    ]

    data = data[
        data["Preprocessing"].isin(preprocessings)
        & data["Dataset"].isin(datasets)
    ]

    # Standard + explainability-weighted columns
    metric_columns = []

    for metric in metrics:
        metric_columns.extend([
            metric,
            f"{metric}_exp"
        ])

    # Mean across folds and seeds
    means = (
        data
        .groupby([
            "Dataset",
            "Model",
            "Preprocessing"
        ])[metric_columns]
        .mean()
    )

    configurations = pd.MultiIndex.from_product(
        [models, preprocessings],
        names=["Model", "Preprocessing"]
    )

    # ---------------------------------------------------------
    # 2. Prepare values
    # ---------------------------------------------------------

    values_by_metric = {}

    for metric in metrics:

        for column in [
            metric,
            f"{metric}_exp"
        ]:

            values = (
                means[column]
                .unstack("Dataset")
                .reindex(
                    configurations,
                    columns=datasets
                )
            )

            if values.isna().any().any():
                raise ValueError(
                    f"Missing results for {column}"
                )

            values_by_metric[column] = values

    # ---------------------------------------------------------
    # 3. Figure
    #
    # Two additional narrow columns are used for independent
    # colorbars for F1 and AUC.
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=figsize,
        layout="constrained"
    )

    gs = GridSpec(
        2, 4,
        figure=fig,
        width_ratios=[
            1, 0.035,
            1, 0.035
        ],
        wspace=0.08,
        hspace=0.08
    )

    axes = np.empty(
        (2, 2),
        dtype=object
    )

    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0])

    axes[0, 1] = fig.add_subplot(gs[0, 2])
    axes[1, 1] = fig.add_subplot(gs[1, 2])

    # One colorbar for each metric
    cbar_axes = [
        fig.add_subplot(gs[:, 1]),
        fig.add_subplot(gs[:, 3])
    ]

    # ---------------------------------------------------------
    # 4. Draw each metric
    # ---------------------------------------------------------

    for j, metric in enumerate(metrics):

        standard_column = metric
        exp_column = f"{metric}_exp"

        standard_values = values_by_metric[
            standard_column
        ]

        exp_values = values_by_metric[
            exp_column
        ]

        # # -----------------------------------------------------
        # # Same row order in both panels:
        # # Standard / All, descending
        # # -----------------------------------------------------

        # order = (
        #     standard_values["all"]
        #     .sort_values(
        #         ascending=False
        #     )
        #     .index
        # )

        # -----------------------------------------------------
        # Common color scale for Standard and Exp.
        # within this metric
        # -----------------------------------------------------

        all_values = np.concatenate([
            standard_values.to_numpy().ravel(),
            exp_values.to_numpy().ravel()
        ])

        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)

        # -----------------------------------------------------
        # Standard and Explainability-aware
        # -----------------------------------------------------

        panels = [
            (
                standard_values,
                "Standard"
            ),
            (
                exp_values,
                "Explainability-aware"
            )
        ]

        for i, (values, evaluation) in enumerate(
            panels
        ):

            ax = axes[i, j]

            # -----------------------------------------------------
            # Each panel is independently ordered according to
            # its performance on the complete dataset ('all')
            # -----------------------------------------------------

            order = (
                values["all"]
                .sort_values(ascending=False)
                .index
            )


            heat_values = values.loc[
                order,
                datasets
            ].copy()

            heat_values.index = [
                f"{model} ({prep})"
                for model, prep in order
            ]

            heat_values.columns = [
                "All",
                "VinDr",
                "INbreast"
            ]

            sns.heatmap(
                heat_values,
                ax=ax,
                cmap="YlGnBu",
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                cbar=(i == 0),
                cbar_ax=(
                    cbar_axes[j]
                    if i == 0
                    else None
                ),
                annot_kws={
                    "fontsize": 8
                }
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.tick_params(
                axis="x",
                rotation=0,
                labelsize=12
            )

            ax.tick_params(
                axis="y",
                rotation=0,
                labelsize=10
            )

            # Column title only in first row
            if i == 0:
                ax.set_title(
                    metric,
                    fontsize=14
                )

            # Row label only on left-hand panels
            if j == 0:
                ax.set_ylabel(
                    evaluation,
                    fontsize=14,
                    labelpad=12
                )

    # ---------------------------------------------------------
    # 5. Figure title
    # ---------------------------------------------------------

    fig.suptitle(
        f"Dataset-stratified performance ranking for {lesion}",
        fontsize=18
    )

    return (
        fig,
        axes,
        values_by_metric
    )

def dataset_configuration_correlations(
    df,
    lesion,
    metrics=("F1-score", "auc"),
    preprocessings=("Bl", "A1"),
    dataset1="VinDr",
    dataset2="INbreast"
):
    """
    Calculate Spearman and Pearson correlations between two
    datasets using the mean performance of the 10
    Model x Preprocessing configurations.

    Fold and seed results are first averaged for each
    configuration.

    Spearman:
        Measures consistency in the relative ordering of
        configurations.

    Pearson:
        Measures the linear relationship between metric
        values in the two datasets.
    """

    data = df.copy()

    data.index = data.index.set_names([
        "Dataset",
        "Preprocessing",
        "Model",
        "Seed",
        "Fold"
    ])

    data = data.reset_index()

    models = [
        "DenseNet",
        "EfficientNet",
        "MobileNet",
        "ResNet18",
        "ResNet50"
    ]

    data = data[
        data["Dataset"].isin([
            dataset1,
            dataset2
        ])
        & data["Preprocessing"].isin(
            preprocessings
        )
    ]

    metric_columns = []

    for metric in metrics:
        metric_columns.extend([
            metric,
            f"{metric}_exp_025",
            f"{metric}_exp_050",
            f"{metric}_exp_075"            
        ])

    # Mean across folds and seeds
    means = (
        data
        .groupby([
            "Dataset",
            "Model",
            "Preprocessing"
        ])[metric_columns]
        .mean()
    )

    configurations = pd.MultiIndex.from_product(
        [models, preprocessings],
        names=["Model", "Preprocessing"]
    )

    results = []

    for metric in metrics:

        columns = {
            "Standard": metric,
            "Explainability-weighted_025":
                f"{metric}_exp_025",
            "Explainability-weighted_050":
                f"{metric}_exp_050",
            "Explainability-weighted_075":
                f"{metric}_exp_075"
        }

        corr_results = []
        for evaluation, column in columns.items():

            values = (
                means[column]
                .unstack("Dataset")
                .reindex(
                    configurations,
                    columns=[
                        dataset1,
                        dataset2
                    ]
                )
            )

            if values.isna().any().any():
                raise ValueError(
                    f"Missing results for {column}"
                )

            x = values[dataset1].to_numpy()
            y = values[dataset2].to_numpy()

            # Spearman correlation
            spearman_r, spearman_p = spearmanr(
                x,
                y
            )

            kendall_v, kendall_p = kendalltau(
                x,
                y
            )

            if spearman_p>=0.001 or kendall_p>=0.001:
                print(f'No significant correlation in {evaluation}-{metric}')

            corr_results.append({
                "Evaluation": evaluation,
                "Spearman": spearman_r,
                "Spearman p-value": spearman_p,
                "Kendall": kendall_v,
                "Kendall p-value": kendall_p

            })

        metric_results = {"Lesion": lesion, "Metric": metric}
        metric_results |= {"Spearman_"+r["Evaluation"]: r["Spearman"] for r in corr_results}
        metric_results |= {"Kendall_"+r["Evaluation"]: r["Kendall"] for r in corr_results}

        results.append(metric_results)        


    df = pd.DataFrame(results)

    df = df.set_index(
        ["Lesion", "Metric"]
    )

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path_all", type=str, required=True,
                    help="Path to the metrics file using the entire dataset")
    parser.add_argument("--metrics_path_vindr", type=str, required=True,
                        help="Path to the metrics file using VinDr only")
    parser.add_argument("--metrics_path_inbreast", type=str, required=True,
                        help="Path to the metrics file using INbreast only")
    parser.add_argument("--positive_class", type=str, required=True,
                            help="Lesion name")
    parser.add_argument("--metrics", type=str, nargs='+', required=True,
                            help="Metric name (F1-score, AUC, Precision, ...)")
    
    parser.add_argument("--data_configurations", type=str, nargs='+', required=True,
                                help="Data configurations (Bl, P1, P2, A1, A2)")
    

    args = parser.parse_args()

    path_to_metrics_all = args.metrics_path_all
    path_to_metrics_VinDr = args.metrics_path_vindr
    path_to_metrics_INbreast = args.metrics_path_inbreast
    lesion=args.positive_class
    metrics=args.metrics
    lesion_title = "Mass" if lesion=='Nodulo' else 'Microcalcifications'
    dconfig = args.data_configurations

    experiment_results = {}

    experiment_results['all'] = create_dictionary_from_results(path_to_metrics_all, lesion, 0.75)
    experiment_results['VinDr'] = create_dictionary_from_results(path_to_metrics_VinDr, lesion, 0.75, 'VinDr')
    experiment_results['INbreast'] = create_dictionary_from_results(path_to_metrics_INbreast, lesion, 0.75, 'INbreast')

    df_std = {}
    df_exp = {}
    for dname in experiment_results:
        df_std[dname] = dict_to_df_fold_seed_metric(experiment_results[dname]["standard"]).loc[dconfig]
        df_exp[dname] = dict_to_df_fold_seed_metric(experiment_results[dname]["explain"]).loc[dconfig]

    df_std = pd.concat(df_std, names=['Dataset'])
    df_exp = pd.concat(df_exp, names=['Dataset'])


    df_exp_75 = df_exp

    experiment_results_25 = {}
    experiment_results_25['all'] = create_dictionary_from_results(path_to_metrics_all, lesion, 0.25)
    experiment_results_25['VinDr'] = create_dictionary_from_results(path_to_metrics_VinDr, lesion, 0.25, 'VinDr')
    experiment_results_25['INbreast'] = create_dictionary_from_results(path_to_metrics_INbreast, lesion, 0.25, 'INbreast')    
    experiment_results_50 = {}
    experiment_results_50['all'] = create_dictionary_from_results(path_to_metrics_all, lesion, 0.50)
    experiment_results_50['VinDr'] = create_dictionary_from_results(path_to_metrics_VinDr, lesion, 0.50, 'VinDr')
    experiment_results_50['INbreast'] = create_dictionary_from_results(path_to_metrics_INbreast, lesion, 0.50, 'INbreast')    

    df_exp_25 = {}
    df_exp_50 = {}
    for dname in experiment_results:
        df_exp_25[dname] = dict_to_df_fold_seed_metric(experiment_results_25[dname]["explain"]).loc[dconfig]
        df_exp_50[dname] = dict_to_df_fold_seed_metric(experiment_results_50[dname]["explain"]).loc[dconfig]

    df_exp_25 = pd.concat(df_exp_25, names=['Dataset'])        
    df_exp_50 = pd.concat(df_exp_50, names=['Dataset'])        


    df_exp_temp = df_exp_75.rename(columns={"Precision":"Precision_exp", "Recall":"Recall_exp", 
                                         "F1-score":"F1-score_exp", "Acc":"Acc_exp", "AUC":"AUC_exp", "AUPRC":"AUPRC_exp"})

    df_all = pd.concat([df_std, df_exp_temp], axis=1)

    fig, axes, values = plot_dataset_heatmaps(df_all, lesion=lesion_title, preprocessings=tuple(dconfig))

    plt.savefig("ranking_per_dataset_"+lesion+".png")
    plt.show()

    df_exp_075 = df_exp_75.rename(columns={"Precision":"Precision_exp_075", "Recall":"Recall_exp_075", 
                                         "F1-score":"F1-score_exp_075", "Acc":"Acc_exp_075", "AUC":"AUC_exp_075", "AUPRC":"AUPRC_exp_075"})
    df_exp_050 = df_exp_50.rename(columns={"Precision":"Precision_exp_050", "Recall":"Recall_exp_050", 
                                         "F1-score":"F1-score_exp_050", "Acc":"Acc_exp_050", "AUC":"AUC_exp_050", "AUPRC":"AUPRC_exp_050"})
    df_exp_025 = df_exp_25.rename(columns={"Precision":"Precision_exp_025", "Recall":"Recall_exp_025", 
                                         "F1-score":"F1-score_exp_025", "Acc":"Acc_exp_025", "AUC":"AUC_exp_025", "AUPRC":"AUPRC_exp_025"})


    df_all_thresholds = pd.concat([df_std, df_exp_025, df_exp_050, df_exp_075], axis=1)
    df_corr = dataset_configuration_correlations(
        df_all_thresholds,
        lesion=lesion,
        metrics=tuple(metrics),
        preprocessings=tuple(dconfig)
    )

    print(
        df_corr.round(3).to_string(
        )
    )
    
    latex_table = df_corr.round(3).to_latex(escape=False)
    print(latex_table) 
    
