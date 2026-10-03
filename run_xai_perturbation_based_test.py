import subprocess
import re
import argparse
import os

PROJECT_DIRPATH = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("--models_path", type=str, default=None,
                    help="Path to the models directory")
parser.add_argument("--positive_class", type=str, default=None,
                    help="Name of the positive class")

parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
parser.add_argument("--data_split_path", type=str, default=".", help="Path to the K-Fold split")


args = parser.parse_args()

models_path = args.models_path

models = os.listdir(models_path)
models.sort()

positive_class = args.positive_class
dataroot = args.dataroot
data_split_path = args.data_split_path


for modelfile in models:

    model_fields = modelfile.split('.pth')[0].split('_')
    model_name = model_fields[0]
    seed = model_fields[2]
    suffix = '_'.join(model_fields[3:-1])
    K = model_fields[-1]
    # Bias = model_fields[-1]
    print(modelfile)
    print(model_fields)
    print(suffix)
    print(K)
    # print('Bias', Bias)
    # print('----')

    if suffix!='copy_copy_copy':
        print('Skipping model', modelfile)
        continue

    print('Processing model', modelfile)

    testset = f'{data_split_path}/{K}/joined_dataset_test_{positive_class}_76014.json'

    augment_config = "augment_transform_"+suffix+".yaml"

    model_weights_path = os.path.join(models_path, modelfile)

    cmd = [
            "python3", "get_xai_perturbation_based_results.py",
            "--testset", testset,
            "--seed", seed, ###
            "--model", model_name, ###
            "--model_weights_path", model_weights_path,
            "--dataroot", dataroot,
            "--positive_classes", positive_class,
            "--augmentation_config_path", '../configuration_files/augment_transform_/'+augment_config,
            "--json_suffix", "_"+model_name+"_"+positive_class+"_"+"faithfulness50",
            "--compare_cam"
        ]

    print(f"Running: {' '.join(cmd)}\n")
                
    process = subprocess.run(
                cmd, 
                stderr=subprocess.STDOUT, 
                text=True, 
                check=True
            )
                
