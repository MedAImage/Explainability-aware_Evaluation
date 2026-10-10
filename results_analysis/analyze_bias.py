import json
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from analysis_utils import read_jsonl

def create_dictionary_from_dir_results(dir_metrics):

    experiments = dict()
    contrib = {"contribution": "use_bias", "contrib_no_bias": "ignore_bias"}
    energy = {"energy_0.25": "energy 0.25", "energy_0.5": "energy 0.50", "energy_0.75": "energy 0.75"}
    correlations = ['spearman', 'pearson', 'ccc'] #, 'pearson']

    files = os.listdir(dir_metrics)
    metrics_files = [f for f in files if f.endswith('.jsonl')]
    for file in metrics_files:
        with open(os.path.join(dir_metrics,file)) as jfile:
            exp_results = read_jsonl(jfile)

        bias = file.split('.jsonl')[0].split('_B')[-1]
        bias = float(bias)
        for exp in exp_results:
            xai_metrics = exp["explainable_metrics"]
            seed = exp["cuantitative_metrics"]["Seed"]
            if bias not in experiments:
                experiments[bias] = dict()
            experiments[bias][seed] = dict()
            for c in contrib:
                c_name = contrib[c]
                if c_name not in experiments[bias][seed]:
                    experiments[bias][seed][c_name] = dict()
                for corr in correlations:
                    experiments[bias][seed][c_name][corr] = xai_metrics[corr][c][0]
                for en in energy:
                    experiments[bias][seed][c_name][en] = xai_metrics[en][c]

    return experiments


def bias_dict_to_df(d):
    rows = []

    for bias, seeds in d.items():
        for seed, map_types in seeds.items(): 
            for map_type, metrics in map_types.items():
                    row = {
                        "Bias": bias,
                        "Seed": seed,
                        "Map type": map_type,
                    }

                    row.update(metrics)

                    rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Bias", "Seed", "Map type"]
    ).sort_index()

    return df


if __name__ == "__main__":

    dir_metrics = sys.argv[1]

    experiment_results = create_dictionary_from_dir_results(dir_metrics)

    df_results = bias_dict_to_df(experiment_results)

    df_results_use_bias = df_results.loc[:,:,"use_bias"]
    df_results_ignore_bias = df_results.loc[:,:,"ignore_bias"]

    df_results_B0 = df_results_use_bias.loc[0.0]

    df_diff_B0 = df_results_use_bias-df_results_B0

    df_diff_ignoreB = df_results_ignore_bias-df_results_use_bias

    fig = plt.figure(figsize=(10, 7), layout='constrained')

    subfigs = fig.subfigures(2, 1, hspace=0.08)

    axes_top = subfigs[0].subplots(1, 2)

    axes_top[0] = sns.lineplot(df_diff_ignoreB, x = 'Bias', y="ccc", ax = axes_top[0])
    axes_top[0].set_ylim(-0.3, 0.3)
    axes_top[0].set_ylabel(r'$\Delta CCC$', fontsize=12)
    axes_top[1] = sns.lineplot(df_diff_ignoreB, x = 'Bias', y="energy_0.75", ax = axes_top[1])
    axes_top[1].set_ylabel(r'$\Delta f_{\mathcal{R}}$', fontsize=12)

    subfigs[0].suptitle(
        'Bias omission effect (ignored − included)', fontsize=14
    )
    subfigs[0].supxlabel('Fixed bias value')

    axes_bottom = subfigs[1].subplots(1, 2)

    axes_bottom[0] = sns.lineplot(df_diff_B0, x = 'Bias', y="ccc", ax = axes_bottom[0])
    axes_bottom[0].set_ylim(-0.3, 0.3)
    axes_bottom[0].set_ylabel(r'$\Delta CCC$', fontsize=12)
    axes_bottom[1] = sns.lineplot(df_diff_B0, x = 'Bias', y="energy_0.75", ax = axes_bottom[1])
    axes_bottom[1].set_ylabel(r'$\Delta f_{\mathcal{R}}$', fontsize=12)

    subfigs[1].suptitle(
        'Fixed-bias effect (bias b - bias 0)', fontsize=14
    )
    subfigs[1].supxlabel('Fixed bias value')

    for ax in list(axes_top)+list(axes_bottom):
        ax.axhline(
            0,
            linewidth=0.8,
            linestyle="--"
        )
        ax.set_xlabel("")

    plt.savefig("bias_analysis.png")
    plt.show()
