import json
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_utils import create_xai_perturbation_dict_from_results, dict_to_df_XAI_perturbation_metric


def plot_delta_auc(df, palette, figsize=(18, 4.5)):

    data = df.reset_index()

    models = data["Model"].unique()
    methods = data["Method"].unique()

    legend_handles = None

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=figsize,
        sharey=True
    )

    for ax, model in zip(axes, models):

        subset = data[data["Model"] == model]

        sns.boxplot(
            data=subset,
            x="Method",
            y="Delta AUC",
            hue='Method',
            order=methods,
            ax=ax,
            showfliers=False,
            palette=palette,
            legend=True
        )

        if legend_handles is None:
            legend_handles, legend_labels = (
                ax.get_legend_handles_labels()
            )

        # if ax.get_legend() is not None:
        #     ax.get_legend().remove()

        # ax.set_xlabel("")
        # ax.set_xticklabels([])        


        sns.stripplot(
            data=subset,
            x="Method",
            y="Delta AUC",
            order=methods,
            ax=ax,
            size=3,
            alpha=0.5
        )

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xlabel("")
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False
        )
 

        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.8
        )

        ax.set_title(model, fontsize=16)

        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta$AUC", fontsize=14)
        else:
            ax.set_ylabel("")


        # ax.tick_params(
        #     axis="x",
        #     rotation=45,
        # )

    fig.suptitle(
        "Perturbation-based evaluation of explanation maps", fontsize=18
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(methods),
        fontsize=14,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0.1, 1, 1])

    return fig

if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    
    experiment_results = create_xai_perturbation_dict_from_results(path_to_metrics)
    df_xai = dict_to_df_XAI_perturbation_metric(experiment_results)    

    df_xai = df_xai.loc[:,:,:,["$M_C^{+}$", 'Att', 'EC-Bk', 'EC-Prj', 'GC-Bk', 'GC-Prj',
                                                 'GC-Att', 'SC-Bk', 'SC-Prj', 'SC-Att']]

    method_colors = {"$M_C^{+}$": "red",
                     'Att': "orange",
                     'EC-Bk': "blue",
                     'EC-Prj': "lightskyblue",
                     'GC-Bk': "lime",
                     'GC-Prj': "palegreen",
                     'GC-Att': 'forestgreen',
                     'SC-Bk': "magenta",
                     'SC-Prj': "pink",
                     'SC-Att': 'deeppink'}

    fig = plot_delta_auc(df_xai, palette = method_colors)

    plt.savefig("Perturbation_xai_evaluation.png")    
    plt.show()
    

