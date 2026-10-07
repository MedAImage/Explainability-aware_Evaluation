import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
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
    parser = argparse.ArgumentParser(description="Anaysis of the bias results")
    parser.add_argument('--perturbation_learned_bias', type=str, required=True, help='File containing the perturbation results of models with learned bias')
    parser.add_argument('--perturbation_ignored_bias', type=str, required=True, help='File containing the perturbation results of models ignoring the bias')
    parser.add_argument('--perturbation_zero_bias', type=str, required=True, help='File containing the perturbation results of models with zero bias')
    # parser.add_argument('--general_learned_bias', type=str, required=True, help='File containing the general results of models with learned bias')
    # parser.add_argument('--general_ignored_bias', type=str, required=True, help='File containing the general results of models ignoring the bias')
    # parser.add_argument('--general_zero_bias', type=str, required=True, help='File containing the general results of models with zero bias')
    
    args = parser.parse_args()

    perturbation_learned_bias = args.perturbation_learned_bias
    perturbation_ignored_bias = args.perturbation_ignored_bias
    perturbation_zero_bias = args.perturbation_zero_bias

    perturbation_files = {'Learned bias': perturbation_learned_bias, 'Ignored bias': perturbation_ignored_bias,
                          'Zero bias': perturbation_zero_bias}
    perturbation_results = {}

    for exp, file in perturbation_files.items():
        experiment_results = create_xai_perturbation_dict_from_results(file)
        perturbation_results[exp] = dict_to_df_XAI_perturbation_metric(experiment_results).loc[:,:,:,["$M_C^{+}$"]]    

    df_perturbation_results = pd.concat(perturbation_results, axis=1)

    print(df_perturbation_results)

    