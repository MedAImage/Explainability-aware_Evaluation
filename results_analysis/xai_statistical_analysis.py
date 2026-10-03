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

    print(df_xai_perturbation)
    stats_deltaAUC, _, _ = paired_statistical_analysis(
        df=df_xai_perturbation,
        metric="Delta AUC",
        method_index="Method",
        reference=r"$M_C^{+}$",
        paired_indices=["Model", "Fold", "Seed"],
        analysis_indices=["Model", "Fold"]
    )
    stats_deltaAUC["Metric"] = 'Delta AUC'


    stats_pg, _, _ = paired_statistical_analysis(
        df=df_xai,
        metric="PG_1-top",
        method_index="Method",
        reference=r"$M_C^{+}$",
        paired_indices=["Model", "Fold", "Seed"],
        analysis_indices=["Model", "Fold"]
    )
    stats_pg["Metric"] = 'Pointing Game'

    stats_fR_025, _, _ = paired_statistical_analysis(
        df=df_xai,
        metric="energy_0.25",
        method_index="Method",
        reference=r"$M_C^{+}$",
        paired_indices=["Model", "Fold", "Seed"],
        analysis_indices=["Model", "Fold"]
    )
    stats_fR_025["Metric"] = '$f_R$ ($\tau=0.25$)'

    stats_fR_050, _, _ = paired_statistical_analysis(
        df=df_xai,
        metric="energy_0.5",
        method_index="Method",
        reference=r"$M_C^{+}$",
        paired_indices=["Model", "Fold", "Seed"],
        analysis_indices=["Model", "Fold"]
    )
    stats_fR_050["Metric"] = '$f_R$ ($\tau=0.50$)'

    stats_fR_075, _, _ = paired_statistical_analysis(
        df=df_xai,
        metric="energy_0.75",
        method_index="Method",
        reference=r"$M_C^{+}$",
        paired_indices=["Model", "Fold", "Seed"],
        analysis_indices=["Model", "Fold"]
    )
    stats_fR_075["Metric"] = '$f_R$ ($\tau=0.75$)'

    stats_all_metrics = pd.concat([stats_deltaAUC, stats_pg, stats_fR_025, stats_fR_050, stats_fR_075])
    stats_all_metrics = stats_all_metrics.set_index(["Metric", "Comparison"]).round(3)
    print(stats_all_metrics)

    latex_table = stats_all_metrics.to_latex(escape=False)
    print(latex_table) 

