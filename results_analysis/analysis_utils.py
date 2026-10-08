import json
import os
import pandas as pd


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

def dict_to_df(d):
    df = (
        pd.DataFrame.from_dict(
            {(exp, model): metrics
            for exp, models in d.items()
            for model, metrics in models.items()},
            orient="index"
        )
    )

    df.index = pd.MultiIndex.from_tuples(
        df.index, names=["Experiment", "Model"]
    )
    return df

def dict_to_df_fold_seed_metric(d):
    rows = []

    for exp, models in d.items():
        for model, seeds in models.items():
            for seed, folds in seeds.items():
                for fold, metrics in folds.items():

                    row = {
                        "Experiment": exp,
                        "Model": model,
                        "Seed": seed,
                        "Fold": fold,                        
                    }

                    row.update(metrics)

                    rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Experiment", "Model", "Seed", "Fold"]
    ).sort_index()

    df = df.reindex(['Bl', 'P1', 'P2', 'A1', 'A2'], level=0)

    return df

def dict_to_df_fold_seed_XAI_metric(d):
    rows = []

    for exp, models in d.items():
        for model, seeds in models.items():
            for seed, folds in seeds.items():
                for fold, methods in folds.items():
                        for method, metrics in methods.items():
                            row = {
                                "Experiment": exp,
                                "Model": model,
                                "Seed": seed,
                                "Fold": fold,
                                "Method": method
                            }

                            row.update(metrics)

                            rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Experiment", "Model", "Seed", "Fold", "Method"]
    ).sort_index()

    df = df.reindex(['Bl', 'P1', 'P2', 'A1', 'A2'], level=0)

    return df

def dict_to_df_XAI_perturbation_metric(d):
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


def create_dictionary_from_results(path_to_metrics, lesion=None, energy_threshold=0.75, dataset_name=None):
    if os.path.isdir(path_to_metrics):
        files = os.listdir(path_to_metrics)
        metrics_files = [f for f in files if f.startswith("Final_metrics_runs")]
    else:
        path_to_metrics, file = os.path.split(path_to_metrics)
        metrics_files = [file]

    metrics_type = {"standard": "cuantitative_metrics", "explain": "explainable_weighted_metrics",
                    "XAI": "explainable_metrics"}

    model_name_tr = {"CustomDenseNet": "DenseNet", "CustomMobileNetV3": "MobileNet",
                    "CustomResNetBinary50": "ResNet50", "CustomResNetBinary": "ResNet18",
                    "EfficientNetB0": "EfficientNet"}
    model_names = list(model_name_tr.keys())    
    exp_name_tr = {"copy_copy_copy": "Bl",  "copy_clahe_enhance_AUGM": "A2",
                   "copy_clahe_tophat5x5_AUGM": "A2", "copy_copy_clahe": "P1", "copy_clahe_enhance": "P2",
                   "copy_clahe_tophat5x5":"P2", "expand_flip": "A1",}
    experiment_names = list(exp_name_tr.keys())
    x_method_name_tr = {"contribution": "$M_C^{+}$", "attention": "Att", "grad_cam_cnn": "GC-Bk", 
                        "grad_cam_proj": "GC-Prj", "grad_cam_att": "GC-Att",
                        "eigen_cam_cnn": "EC-Bk", "eigen_cam_proj": "EC-Prj", "eigen_cam_att": "EC-Att",
                        "shapley_cam_cnn": "SC-Bk", "shapley_cam_proj": "SC-Prj", "shapley_cam_att": "SC-Att"}

    experiments = {"standard": dict(), "explain": dict(), 'XAI': dict()}
    if dataset_name!=None:
        dataset_name = dataset_name + '_'
    else:
        dataset_name = ''
    for f in metrics_files:
        # if not f.endswith('.jsonl'):
        #     continue
        # print(f)

        with open(os.path.join(path_to_metrics, f)) as jfile:
            exp_results = read_jsonl(jfile)


        for mtype in metrics_type:
            mtype_name = metrics_type[mtype]

            for r in exp_results:
                long_model_name = r["cuantitative_metrics"]["Model-Run"]
                model = next((model_name for model_name in model_names if model_name in long_model_name), None)
                if model==None:
                    continue
                exp = next((exp_name for exp_name in experiment_names if exp_name in long_model_name), None)
                if exp==None:
                    continue
                fields = long_model_name.split('.')[0].split('_')
                K = next((field for field in fields if field.startswith('K')), None)
                if K==None:
                    continue
                seed = r["cuantitative_metrics"]["Seed"]
                model_name = model_name_tr[model]
                exp = exp_name_tr[exp]
                if exp not in experiments[mtype].keys():
                    experiments[mtype][exp] = dict()

                if model_name not in experiments[mtype][exp]:
                    experiments[mtype][exp][model_name] = dict()
                if seed not in experiments[mtype][exp][model_name]:
                    experiments[mtype][exp][model_name][seed] = dict()
                if K not in experiments[mtype][exp][model_name][seed]:
                    experiments[mtype][exp][model_name][seed][K] = dict()

                XAI_methods = False
                if mtype == 'standard':                
                    pred_results = r[mtype_name]
                elif mtype == 'explain':
                    pred_results = r[mtype_name][str(energy_threshold)]
                else:
                    XAI_methods = True

                if XAI_methods:
                    XAI_metrics = r[mtype_name]
                    experiments[mtype][exp][model_name][seed][K] = {x_method_name_tr["contribution"]: {}, x_method_name_tr["attention"]: {},
                                                                    x_method_name_tr["grad_cam_cnn"]: {}, x_method_name_tr["grad_cam_proj"]: {}, 
                                                                    x_method_name_tr["grad_cam_att"]: {}, x_method_name_tr["eigen_cam_cnn"]: {}, 
                                                                    x_method_name_tr["eigen_cam_proj"]: {}, x_method_name_tr["eigen_cam_att"]: {},
                                                                    x_method_name_tr["shapley_cam_cnn"]: {}, x_method_name_tr["shapley_cam_proj"]: {},
                                                                    x_method_name_tr["shapley_cam_att"]: {}}
                    for x_metric in XAI_metrics:
                        for x_method, x_res in XAI_metrics[x_metric].items():
                            if x_method not in x_method_name_tr.keys():
                                continue
                            x_method_tr = x_method_name_tr[x_method]
                            if type(x_res) is list:
                                x_res = x_res[0]
                            experiments[mtype][exp][model_name][seed][K][x_method_tr][x_metric] = x_res
                else:
                    experiments[mtype][exp][model_name][seed][K]["Precision"] = pred_results["Precision"]
                    experiments[mtype][exp][model_name][seed][K]["Recall"] = pred_results["Recall"]
                    experiments[mtype][exp][model_name][seed][K]["F1-score"] = pred_results["F1 Score"]
                    experiments[mtype][exp][model_name][seed][K]["Acc"] = pred_results["Accuracy"]
                    experiments[mtype][exp][model_name][seed][K]["AUC"] = pred_results["AUC-ROC"]
                    experiments[mtype][exp][model_name][seed][K]["AUPRC"] = pred_results["AUPRC"]
    return experiments

def create_xai_perturbation_dict_from_results(path_to_metrics):
    if os.path.isdir(path_to_metrics):
        files = os.listdir(path_to_metrics)
        metrics_files = [f for f in files if f.startswith("Final_metrics_runs")]
    else:
        path_to_metrics, file = os.path.split(path_to_metrics)
        metrics_files = [file]

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
            fields = r['Model File'].split('.')[0].split('_')
            K = next((field for field in fields if field.startswith('K')), None)
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

