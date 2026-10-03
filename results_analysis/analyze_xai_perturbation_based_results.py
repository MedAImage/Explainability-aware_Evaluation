import json
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    x_method_name_tr = {"contribution": "$M_C^{+}$", "attention": "Att", "grad_cam_cnn": "GC-Bk", 
                        "grad_cam_proj": "GC-Prj", "grad_cam_att": "GC-Att",
                        "eigen_cam_cnn": "EC-Bk", "eigen_cam_proj": "EC-Prj", "eigen_cam_att": "EC-Att",
                        "shapley_cam_cnn": "SC-Bk", "shapley_cam_proj": "SC-Prj", "shapley_cam_att": "SC-Att"}

    experiments = {}
    for f in metrics_files:
        with open(os.path.join(path_to_metrics, f)) as jfile:
            exp_results = read_jsonl(jfile)
        for r in exp_results:
            arch = model_name_tr[r['Architecture']]
            seed = r['Seed']
            K = r['Model File'].split('.')[0].split('_')[-1]
            if arch not in experiments:
                experiments[arch] = {}
            if seed not in experiments[arch]:
                experiments[arch][seed] = {}
            if K not in experiments[arch][seed]:
                experiments[arch][seed][K] = {}
            for method, metrics in r['Faithfulness'].items():
                experiments[arch][seed][K][x_method_name_tr[method]] = {'AUC MoRF': metrics['AUC MoRF'][0], 
                                                                        'AUC LeRF': metrics['AUC LeRF'][0],
                                                                        'Delta AUC': metrics['Delta AUC'][0]}
    return experiments

def dict_to_df_fold_seed_XAI_metric(d):
    rows = []

    for model, seeds in d.items():
        for seed, folds in seeds.items():
            for fold, methods in folds.items():
                    for method, metrics in methods.items():
                        row = {
                            "Model": model,
                            "Seed": seed,
                            "Fold": fold,
                            "Method": method
                        }

                        row.update(metrics)

                        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Model", "Seed", "Fold", "Method"]
    ).sort_index()

    return df

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
    
    experiment_results = create_dictionary_from_results(path_to_metrics)
    df_xai = dict_to_df_fold_seed_XAI_metric(experiment_results)    

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
    

