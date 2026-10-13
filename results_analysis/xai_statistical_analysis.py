import sys
from analysis_utils import create_dictionary_from_results, dict_to_df_fold_seed_XAI_metric
from analysis_utils import create_xai_perturbation_dict_from_results, dict_to_df_XAI_perturbation_metric
from statistical_utils import paired_statistical_analysis
import pandas as pd


if __name__ == "__main__":

    path_to_general_xai_metrics = sys.argv[1]
    path_to_perturbation_xai_metrics = sys.argv[2]
    lesion=sys.argv[3]

    general_experiment_results = create_dictionary_from_results(path_to_general_xai_metrics, lesion)
    df_xai = dict_to_df_fold_seed_XAI_metric(general_experiment_results["XAI"])

    df_xai = df_xai.loc(axis=0)['Bl']

    methods = ["$M_C^{+}$", 'Att', 'EC-Bk', 'GC-Bk', 'SC-Bk']

    df_xai = df_xai.loc[:,:,:,methods]

    perturbation_experiment_results = create_xai_perturbation_dict_from_results(path_to_perturbation_xai_metrics)
    df_xai_perturbation = dict_to_df_XAI_perturbation_metric(perturbation_experiment_results)
    df_xai_perturbation = df_xai_perturbation.loc[:,:,:,methods]

    metrics = {'spearman': {'df': df_xai, 'name': 'Spearman'},
               'Delta AUC': {'df': df_xai_perturbation, 'name': 'Delta AUC'},
               'PG_1-top': {'df': df_xai, 'name': 'Pointing Game'},
               'energy_0.25': {'df': df_xai, 'name': r'$f_R$ ($\tau=0.25$)'},
               'energy_0.5': {'df': df_xai, 'name': r'$f_R$ ($\tau=0.50$)'},
               'energy_0.75': {'df': df_xai, 'name': r'$f_R$ ($\tau=0.75$)'}}

    all_stats_results = []

    for metric, info_metric in metrics.items():
        stats_results, _, _ = paired_statistical_analysis(
                df=info_metric['df'],
                metric=metric,
                method_index="Method",
                reference=r"$M_C^{+}$",
                paired_indices=["Model", "Fold", "Seed"],
                analysis_indices=["Model", "Fold"]
            )
        stats_results["Metric"] = info_metric['name']
        all_stats_results.append(stats_results)
        
    stats_all_metrics = pd.concat(all_stats_results)
    stats_all_metrics = stats_all_metrics.set_index(["Metric", "Comparison"]).round(3)
    print(stats_all_metrics)

    latex_table = stats_all_metrics.to_latex(float_format = "%.4f", escape=False)
    print(latex_table) 

