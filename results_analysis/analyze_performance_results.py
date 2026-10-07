import json
import sys
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from analysis_utils import create_dictionary_from_results, dict_to_df_fold_seed_metric
from scipy.stats import spearmanr, variation

def _darken(color, factor=0.65):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def performance_with_without_expl_plot_bars(
    ax,
    df_runs_orig: pd.DataFrame,
    df_runs_expl: pd.DataFrame,
    metric: str = "AUC",
    title: str | None = None,
    figsize=(10, 5),
    inner_factor=0.65,
    outer_width=0.16,
    inner_width=0.09,
):

    dfo = df_runs_orig.copy().reset_index()
    dfe = df_runs_expl.copy().reset_index()

    # Compute mean on folds-seeds
    mean_orig = (
        dfo.groupby(["Experiment", "Model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value_orig"})
    )
    mean_expl = (
        dfe.groupby(["Experiment", "Model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value_expl"})
    )

    # Align standard and explained results
    d = mean_orig.merge(mean_expl, on=["Experiment", "Model"], how="inner")

    models = sorted(d["Model"].unique())
    exps = ['Bl', 'P1', 'P2', 'A1', 'A2']

    x = np.arange(len(models))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    exp_color = {exp: color_cycle[i % len(color_cycle)] for i, exp in enumerate(exps)}

    offsets = (np.arange(len(exps)) - (len(exps) - 1) / 2) * outer_width

    if ax is None:
        print('ax is None')
        fig, ax = plt.subplots(figsize=figsize)

    for j, exp in enumerate(exps):
        c = exp_color[exp]
        c_inner = _darken(c, factor=inner_factor)

        de = d[d["Experiment"] == exp].set_index("Model")

        y_orig = [de.loc[m, "value_orig"] if m in de.index else np.nan for m in models]
        y_expl = [de.loc[m, "value_expl"] if m in de.index else np.nan for m in models]

        xpos = x + offsets[j]

        ax.bar(xpos, y_orig, width=outer_width, label=exp, color=c, edgecolor="none")
        ax.bar(xpos, y_expl, width=inner_width, color=c_inner, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.legend(title="Experiment", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_model_threshold_heatmap(
    df,
    metric_prefix,
    ax,
    exp_order=None,
    model_order=None,
    thresholds=('0.25', '0.5', '0.75'),
    cmap='viridis',
    vmin=None,
    vmax=None,
    annot=False,
    cbar=False,
    cbar_ax=None
):
    dfr = df.reset_index()

    blocks = []
    for model in model_order:
        sub = dfr[dfr['Model'] == model].copy().set_index('Experiment')
        cols = [f'{metric_prefix}_p_{t}' for t in thresholds]
        sub = sub[cols]

        if exp_order is not None:
            sub = sub.loc[exp_order]

        sub.columns = pd.MultiIndex.from_product([[model], thresholds])
        blocks.append(sub)

    full = pd.concat(blocks, axis=1)

    sns.heatmap(
        full,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f",
        cbar=cbar,
        cbar_ax=cbar_ax
    )
    if cbar:
        cbar_ax.tick_params(labelsize=12)

    lower_labels = [str(t) for _m in model_order for t in thresholds]
    ax.set_xticklabels(lower_labels, rotation=0)
    ax.set_ylabel("Data configuration", y=0.4, fontsize=12)
    ax.set_xlabel("Energy threshold", fontsize=12)
    ax.tick_params(axis='both', labelsize = 12)

    n_thr = len(thresholds)
    for k in range(1, len(model_order)):
        ax.axvline(k * n_thr, color='white', lw=2)

    y_top = -0.2
    for i, model in enumerate(model_order):
        center = i * n_thr + n_thr / 2
        ax.text(center, y_top, model, ha='center', va='bottom', fontsize=12)

    ax.set_ylim(len(full.index), -1.0)

def df_to_latex_with_min_max(df, minmax_rows, minmax_columns):
    df_str = df.round(2).astype(str)
    # print(minmax_rows)
    # print(minmax_columns)
    for mMr in minmax_rows:
        idx_min = df.loc(axis=0)[mMr,:].idxmin()
        idx_max = df.loc(axis=0)[mMr,:].idxmax()    
        for col, row_idx in idx_min.items():
            if col in minmax_columns:
                val = df.loc[row_idx, col]
                new_val = f"{{\\color{{red}} {val:.2f} }}"
                df_str.loc[row_idx, col] = new_val

        for col, row_idx in idx_max.items():
            if col in minmax_columns:
                val = df.loc[row_idx, col]
                new_val = f"{{\\color{{mygreen}} {val:.2f} }}"
                df_str.loc[row_idx, col] = new_val
            else:
                val = df.loc[row_idx, col]
                new_val = f"\\textbf{{{val:.2f}}}"
                df_str.loc[row_idx, col] = new_val
                

    latex_table = df_str.to_latex(escape=False)
    latex_table = re.sub(' +', ' ', latex_table)
    # print(latex_table) #float_format="%.2f"))

    return latex_table

def compute_df_correlations(df_std, df_exp, metrics, backbones, lesion):
    index = pd.MultiIndex.from_product([[lesion], metrics, backbones], names=["Lesion", "Metric", "Backbone"])

    columns = []
    for th in df_exp:
        columns.append(("Rank correlation ($\rho$)", "Standard vs. explainability-aware", f"$\tau={th}$"))
    columns.append(("Rank correlation ($\rho$)", "Threshold stability", r"$0.25$ vs. $0.75$"))
    columns+=[("Relative score dispersion", "Standard",""),("Relative score dispersion", "Explainability-aware",f"$\tau=0.25$"), 
              ("Relative score dispersion", "Explainability-aware",f"$\tau=0.5$"), ("Relative score dispersion","Explainability-aware",f"$\tau=0.75$")]
    columns = pd.MultiIndex.from_tuples(columns)

    df_corr = pd.DataFrame(index=index, columns=columns, dtype=float)    
    for metric in metrics:
        for bk in backbones:
            if bk=="Global":
                values_std = df_std[metric].to_numpy()
            else:
                values_std = df_std.loc(axis=0)[:,bk][metric].to_numpy()
            CV_std = variation(values_std)                
            df_corr.loc[(lesion, metric, bk), ("Relative score dispersion", "Standard","")] = CV_std
            for th in df_exp:
                if bk=="Global":
                    values_exp = df_exp[th][metric].to_numpy()
                else:
                    values_exp = df_exp[th].loc(axis=0)[:,bk][metric].to_numpy()
                spearman_r, _ = spearmanr(
                    values_std,
                    values_exp
                )
                df_corr.loc[(lesion, metric, bk), ("Rank correlation ($\rho$)", "Standard vs. explainability-aware", f"$\tau={th}$")] = spearman_r
                CV_exp = variation(values_exp)
                df_corr.loc[(lesion, metric, bk), ("Relative score dispersion", "Explainability-aware",f"$\tau={th}$")] = CV_exp

            if bk=="Global":
                values_exp25 = df_exp[0.25][metric].to_numpy()
                values_exp75 = df_exp[0.75][metric].to_numpy()        
            else:
                values_exp25 = df_exp[0.25].loc(axis=0)[:,bk][metric].to_numpy()
                values_exp75 = df_exp[0.75].loc(axis=0)[:,bk][metric].to_numpy()        

            spearman_r, _ = spearmanr(
                values_exp25,
                values_exp75
            )
            df_corr.loc[(lesion, metric, bk), ("Rank correlation ($\rho$)", "Threshold stability", r"$0.25$ vs. $0.75$")] = spearman_r
    return df_corr

if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    lesion=sys.argv[2]

    experiment_results = create_dictionary_from_results(path_to_metrics, lesion, 0.75)
    df_std = dict_to_df_fold_seed_metric(experiment_results["standard"])
    df_exp = dict_to_df_fold_seed_metric(experiment_results["explain"])

    df_exp_75 = df_exp
    experiment_results_25 = create_dictionary_from_results(path_to_metrics, lesion, 0.25)
    df_exp_25 = dict_to_df_fold_seed_metric(experiment_results_25["explain"])
    experiment_results_50 = create_dictionary_from_results(path_to_metrics, lesion, 0.5)
    df_exp_50 = dict_to_df_fold_seed_metric(experiment_results_50["explain"])


    df_exp_temp = df_exp_75.rename(columns={"Precision":"P_exp", "Recall":"recall_exp", 
                                         "F1-score":"f1_exp", "Acc":"acc_exp", "AUC":"auc-roc_exp", "AUPRC":"auprc_exp"})


    df_penalty_all = []
    metrics = df_std.columns.to_list()
    df_exp_th = [df_exp_25, df_exp_50, df_exp_75]    
    for idx_m, metric in enumerate(metrics):
        thresholds = [0.25, 0.50, 0.75]

        df_metric = df_std[metric]
        df_metric_penalty = []
        for idx in range(len(df_exp_th)):
            df_metric_explained = df_exp_th[idx][metric]
            df_penalty = 1-df_metric_explained/df_metric
            df_penalty = df_penalty.to_frame(name=metric+'_p_'+str(thresholds[idx]))

            df_metric_penalty.append(df_penalty)

        df_penalty_all.append(pd.concat(df_metric_penalty, axis=1))

    df_penalty_all = pd.concat(df_penalty_all, axis=1)


    penalty_metrics = df_penalty_all.columns.to_list()
    penalty_075 = [m for m in penalty_metrics if m.endswith('0.75')]

    df_all = pd.concat([df_std, df_exp_temp, df_penalty_all[penalty_075]], axis=1)

    df_mean_metrics = df_all.groupby(["Experiment","Model"]).agg("mean")

    print(df_mean_metrics)

    backbone_order = ["DenseNet", "EfficientNet", "MobileNet", "ResNet18", "ResNet50"]
    dconfig_order = ['Bl', 'P1', 'P2', 'A1', 'A2']
    df_mean_metrics = df_mean_metrics.reorder_levels(["Model", "Experiment"]).reindex(backbone_order, level=0).reindex(dconfig_order, level=1)
    minmax_rows = backbone_order
    minmax_columns = df_std.columns.to_list() + df_exp_temp.columns.to_list()

    # print(df_mean_metrics)

    # latex_table = df_to_latex_with_min_max(df_mean_metrics, minmax_rows=minmax_rows, minmax_columns=minmax_columns)

    # print(latex_table) 
    
    
    # Correlations and dispersions
    
    df_mean_std = df_std.groupby(["Experiment","Model"]).agg("mean")
    df_mean_exp_25 = df_exp_25.groupby(["Experiment","Model"]).agg("mean")
    df_mean_exp_50 = df_exp_50.groupby(["Experiment","Model"]).agg("mean")
    df_mean_exp_75 = df_exp_75.groupby(["Experiment","Model"]).agg("mean")
    
    df_mean_exp = {0.25: df_mean_exp_25, 0.50: df_mean_exp_50, 0.75: df_mean_exp_75}

    metrics = ['F1-score', 'AUC']
    backbones = ["DenseNet", "EfficientNet", "MobileNet", "ResNet18", "ResNet50", "Global"]

    lesion_name = "Mass" if lesion=="Nodulo" else "Microcalcificaciones"
    df_correlations = compute_df_correlations(df_mean_std, df_mean_exp, metrics, backbones, lesion_name)
    print(df_correlations)
    
    # latex_corr_table = df_correlations.round(2).to_latex(escape=False)    
    
    # print(latex_corr_table)


    #Plots for standard vs. explainability-aware performance
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes = axes.reshape(1, -1)

    
    axes_flat = axes.flatten()

    figure_metrics = ["Recall", "F1-score", "AUC", "AUPRC"]
    for idx, metric in enumerate(figure_metrics):
        axes_flat[idx] = performance_with_without_expl_plot_bars(axes_flat[idx], df_std, df_exp, metric = metric)

        axes_flat[idx].set_ylabel(metric, fontsize=11)
        axes_flat[idx].set_xlabel("")
        axes_flat[idx].tick_params(axis='both', labelsize = 12)


        axes_flat[idx].legend_.remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.005),
        fontsize = 12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("performance_metrics_"+sys.argv[2]+".png")
    plt.show()


    #Plots for explainability penalty

    vmin = 0
    vmax = 1

    exp_order = ['Bl', 'P1', 'P2', 'A1', 'A2']
    model_order = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']
    thresholds = ['0.25', '0.5', '0.75']

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.subplots_adjust(right=0.88)
    axes_flat = axes.flatten()

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    metrics = ['F1-score', 'AUC']
    titles = ['F1-score penalty', 'AUC penalty']
    mean_penalty = (
            df_penalty_all.groupby(["Experiment", "Model"])
            .mean()
        )

    thresholds = [0.25, 0.50, 0.75]
    for idx_m, metric in enumerate(metrics):
        df_penalty_metric = mean_penalty[[metric+'_p_'+str(th) for th in thresholds]]

        plot_model_threshold_heatmap(
            df_penalty_metric,
            metric_prefix=metric,
            ax=axes_flat[idx_m],
            exp_order=exp_order,
            model_order=model_order,
            thresholds=thresholds,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            cbar=idx_m==0,
            cbar_ax=cbar_ax if idx_m==0 else None
        )

        axes_flat[idx_m].set_title(titles[idx_m], fontsize = 14)


    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig("metrics_penalization_"+sys.argv[2]+".png")    
    plt.show()

