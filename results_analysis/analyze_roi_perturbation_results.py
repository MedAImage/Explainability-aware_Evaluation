import json
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy

def read_jsonl(file):
    json_obj = []
    buffer = ""
    for l in file:
        if l=="}{\n":
            l = "}\n"
        buffer += l
        try:
            obj = json.loads(buffer)
            json_obj.append(obj)
            buffer = "{\n"
        except json.JSONDecodeError:
            pass
    return json_obj


def create_dictionary_from_results(path_to_metrics):
    files = os.listdir(path_to_metrics)

    metrics_files = [f for f in files if f.endswith(".jsonl")]

    model_name_tr = {"CustomDenseNet": "DenseNet", "CustomMobileNetV3": "MobileNet",
                    "CustomResNetBinary50": "ResNet50", "CustomResNetBinary": "ResNet18",
                    "EfficientNetB0": "EfficientNet"}

    lesion_dict = {'Nodulo':{}, 'Microcalcificaciones':{}}
    experiments = {'size perturbation': copy.deepcopy(lesion_dict), 'position perturbation': copy.deepcopy(lesion_dict)}
    for f in metrics_files:
        lesion = f.split('_')[-1].split('.jsonl')[0]
        with open(os.path.join(path_to_metrics, f)) as jfile:
            exp_results = read_jsonl(jfile)
        for r in exp_results:
            arch = model_name_tr[r['Architecture']]
            seed = r['Seed']
            K = r['Model File'].split('.')[0].split('_')[-1]
            if arch not in experiments['size perturbation'][lesion]:
                experiments['size perturbation'][lesion][arch] = {}
                experiments['position perturbation'][lesion][arch] = {}
            if seed not in experiments['size perturbation'][lesion][arch]:
                experiments['size perturbation'][lesion][arch][seed] = {}
                experiments['position perturbation'][lesion][arch][seed] = {}
            if K not in experiments['size perturbation'][lesion][arch][seed]:
                experiments['size perturbation'][lesion][arch][seed][K] = {}
                experiments['position perturbation'][lesion][arch][seed][K] = {}
            experiments['size perturbation'][lesion][arch][seed][K] = r['ROI Perturbation']['perturbed_size']                
            experiments['position perturbation'][lesion][arch][seed][K] = r['ROI Perturbation']['perturbed_position']

    return experiments

def dict_to_df_perturbation_ROI(d):
    rows = []

    for lesion, models in d.items():
        for model, seeds in models.items():
            for seed, folds in seeds.items():
                for fold, alphas in folds.items():
                        for alpha, energy_ths in alphas.items():
                            for energy_th, metrics in energy_ths.items():
                                row = {
                                    "Lesion": lesion,
                                    "Model": model,
                                    "Seed": seed,
                                    "Fold": fold,
                                    "Alpha": alpha,
                                    "Energy Threshold": energy_th
                                }

                                row.update(metrics)

                                rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Lesion", "Model", "Seed", "Fold", "Alpha", "Energy Threshold"]
    ).sort_index()

    return df


def plot_roi_perturbation(
        df,
        title=None,
        alpha_ticks=None,
        lesion_labels=None,
        figsize=(20, 10)):
    """
    Plot ROI perturbation sensitivity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MultiIndex levels:
        Lesion, Model, Seed, Fold, Alpha, Energy Threshold.

        Expected columns include:
        - mean_fR_diff_with_0
        - std_fR_diff_with_0

    title : str, optional
        Figure title.

    alpha_ticks : list of float, optional
        Alpha values displayed as x-axis tick labels.
        All available alpha values are still used to draw the curves.

    lesion_labels : dict, optional
        Mapping between lesion names in the DataFrame and the labels
        displayed in the figure. For example:

        {
            "Microcalcificaciones": "Microcalcifications",
            "Nodulo": "Masses"
        }

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    """

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    data = df.reset_index()

    # Make sure Alpha and Energy Threshold are numeric
    data["Alpha"] = data["Alpha"].astype(float)
    data["Energy Threshold"] = data["Energy Threshold"].astype(float)

    # Sort data explicitly
    data = data.sort_values(
        [
            "Lesion",
            "Model",
            "Seed",
            "Fold",
            "Energy Threshold",
            "Alpha"
        ]
    )

    lesions = list(data["Lesion"].unique())
    models = list(data["Model"].unique())

    thresholds = sorted(
        data["Energy Threshold"].unique()
    )

    alpha_values = sorted(
        data["Alpha"].unique()
    )

    # If alpha_ticks is not specified, show all alpha values
    if alpha_ticks is None:
        alpha_ticks = alpha_values

    # Default lesion labels
    if lesion_labels is None:
        lesion_labels = {}

    # ---------------------------------------------------------
    # Global Y limits
    #
    # Same scales are used for all lesions and architectures
    # within this figure.
    # ---------------------------------------------------------

    mean_min = data["mean_fR_diff_with_0"].min()
    mean_max = data["mean_fR_diff_with_0"].max()

    mean_range = mean_max - mean_min

    if mean_range == 0:
        mean_range = 1.0

    mean_margin = 0.05 * mean_range

    std_max = data["std_fR_diff_with_0"].max()

    if std_max == 0:
        std_max = 1.0

    std_margin = 0.05 * std_max

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        nrows=len(lesions),
        ncols=len(models),
        wspace=0.20,
        hspace=0.30
    )

    # Store axes in first column to position lesion row labels
    row_axes = []

    # Information for the common legend
    legend_handles = None
    legend_labels = None

    # ---------------------------------------------------------
    # Draw grid
    # ---------------------------------------------------------

    for i_lesion, lesion in enumerate(lesion_labels.keys()):

        for i_model, model in enumerate(models):

            # Two vertically stacked plots inside each
            # lesion/model cell
            inner = outer[i_lesion, i_model].subgridspec(
                2,
                1,
                height_ratios=[1, 1],
                hspace=0.08
            )

            ax_mean = fig.add_subplot(inner[0])
            ax_std = fig.add_subplot(
                inner[1],
                sharex=ax_mean
            )

            # Store axes from first column for row label positioning
            if i_model == 0:
                row_axes.append((ax_mean, ax_std))

            subset = data[
                (data["Lesion"] == lesion) &
                (data["Model"] == model)
            ]

            # -------------------------------------------------
            # Upper plot: mean Delta f_R
            #
            # Line:
            #   mean across fold-seed runs
            #
            # Shaded region:
            #   SD across fold-seed runs
            # -------------------------------------------------

            sns.lineplot(
                data=subset,
                x="Alpha",
                y="mean_fR_diff_with_0",
                hue="Energy Threshold",
                hue_order=thresholds,
                marker="o",
                palette= sns.color_palette(),
                errorbar="sd",
                ax=ax_mean
            )

            # Reference line: no change in f_R
            ax_mean.axhline(
                0,
                linewidth=0.8,
                linestyle="--"
            )

            # Save legend information only once
            if legend_handles is None:
                legend_handles, legend_labels = (
                    ax_mean.get_legend_handles_labels()
                )

            # Remove local legend
            if ax_mean.get_legend() is not None:
                ax_mean.get_legend().remove()

            # -------------------------------------------------
            # Lower plot: SD of Delta f_R across images
            #
            # Line:
            #   mean of the within-run SD across fold-seed runs
            # -------------------------------------------------

            sns.lineplot(
                data=subset,
                x="Alpha",
                y="std_fR_diff_with_0",
                hue="Energy Threshold",
                hue_order=thresholds,
                marker="o",
                palette= sns.color_palette(),
                errorbar=None,
                ax=ax_std,
                legend=False
            )

            # -------------------------------------------------
            # Common Y scales
            # -------------------------------------------------

            ax_mean.set_ylim(
                mean_min - mean_margin,
                mean_max + mean_margin
            )

            ax_std.set_ylim(
                0,
                std_max + std_margin
            )

            # -------------------------------------------------
            # Architecture titles
            # -------------------------------------------------

            if i_lesion == 0:
                ax_mean.set_title(
                    model,
                    fontsize=16
                )

            # -------------------------------------------------
            # Y labels
            # -------------------------------------------------

            if i_model == 0:

                ax_mean.set_ylabel(
                    r"Mean $\Delta f_R$", fontsize=16
                )

                ax_std.set_ylabel(
                    r"SD $\Delta f_R$", fontsize=16
                )

            else:

                ax_mean.set_ylabel("")
                ax_std.set_ylabel("")

            # -------------------------------------------------
            # X axis
            # -------------------------------------------------

            # Do not show tick labels on upper plots
            ax_mean.tick_params(
                axis="x",
                labelbottom=False
            )

            # Show only selected alpha tick labels
            ax_std.set_xticks(alpha_ticks)

            # X label only on the bottom lesion row
            if i_lesion == len(lesions) - 1:

                ax_std.set_xlabel(
                    r"$\alpha$", fontsize=14
                )

            else:

                ax_std.set_xlabel("")

    # ---------------------------------------------------------
    # Figure title
    # ---------------------------------------------------------

    if title is not None:

        fig.suptitle(
            title,
            fontsize=18,
            y=0.98
        )

    # ---------------------------------------------------------
    # Adjust figure margins BEFORE positioning row labels
    # ---------------------------------------------------------

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.15
    )

    # ---------------------------------------------------------
    # Lesion labels
    #
    # Each label is vertically centered with respect to the
    # two plots belonging to that lesion row.
    # ---------------------------------------------------------

    for lesion, (ax_mean, ax_std) in zip(
            lesion_labels.keys(),
            row_axes):

        pos_mean = ax_mean.get_position()
        pos_std = ax_std.get_position()

        y_center = (
            pos_mean.y1 + pos_std.y0
        ) / 2

        label = lesion_labels.get(
            lesion,
            lesion
        )

        fig.text(
            0.02,
            y_center,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold"
        )

    # ---------------------------------------------------------
    # Common legend at bottom
    # ---------------------------------------------------------

    fig.legend(
        legend_handles,
        legend_labels,
        title="Energy threshold",
        title_fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(thresholds),
        fontsize=14,
        frameon=False
    )

    return fig

if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    
    experiment_results = create_dictionary_from_results(path_to_metrics)
    df_size = dict_to_df_perturbation_ROI(experiment_results['size perturbation'])
    df_position = dict_to_df_perturbation_ROI(experiment_results['position perturbation'])

    lesion_labels = {
        "Nodulo": "Mass",
        "Microcalcificaciones": "Microcalcifications"        
    }

    fig = plot_roi_perturbation(
        df_size,
        title="Sensitivity to ROI size perturbations",
        alpha_ticks=[-0.4, -0.2, 0, 0.2, 0.4],
        lesion_labels=lesion_labels
    )

    plt.savefig("Sensitivity_ROI_size_perturbations.png")
    plt.show()
    
    fig = plot_roi_perturbation(
        df_position,
        title="Sensitivity to ROI position perturbations",
        alpha_ticks=[0, 0.2, 0.4],
        lesion_labels=lesion_labels
    )

    plt.savefig("Sensitivity_ROI_position_perturbations.png")
    plt.show()