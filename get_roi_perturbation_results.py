
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from dataset_load.dataset import lesionDataset, collate_test
from models.models import  EfficientNetB0, CustomResNetBinary, CustomResNetBinary50, CustomDenseNet, CustomMobileNetV3
from metrics import get_roi_energy_fraction, get_ground_truth_mask, normalize_map
import argparse
import json
import yaml
import copy
from tqdm import tqdm

def perturbe_rois_in_size(rois, size_alpha_values):
    new_rois = {}
    for alpha in size_alpha_values:
        new_rois[alpha] = []
        for r in rois:
            cx, cy = r[0]+r[2]/2, r[1]+r[3]/2
            w, h = r[2], r[3]
            w = (1+alpha)*w
            h = (1+alpha)*h
            x, y = cx-w/2, cy-h/2
            p_roi = [x, y, w, h]
            new_rois[alpha].append(p_roi)
    return new_rois


def perturbe_rois_in_position(rois, position_alpha_values,
                              image_width, image_height):

    new_rois = {}


    directions = [
        (-1, 0),                         # left
        (-1, -1),           # upper-left
        (0, -1),                         # up
        (1, -1),            # upper-right
        (1, 0),                          # right
        (1, 1),             # lower-right
        (0, 1),                          # down
        (-1, 1),            # lower-left
    ]
    for alpha in position_alpha_values:

        # Independent list for every direction
        new_rois[alpha] = [
            [None] * len(rois)
            for _ in range(8)
        ]

        for i_r, r in enumerate(rois):

            x1, y1, w, h = r[0], r[1], r[2], r[3]

            displacement_x = alpha*w
            displacement_y = alpha*h

            for i_direction, (ux, uy) in enumerate(directions):

                dx = displacement_x * ux
                dy = displacement_y * uy

                new_x1 = x1 + dx
                new_y1 = y1 + dy

                new_x1 = max(0, min(new_x1, image_width - w))
                new_y1 = max(0, min(new_y1, image_height - h))

                new_rois[alpha][i_direction][i_r] = [
                    new_x1,
                    new_y1,
                    w,
                    h
                ]

    return new_rois


def get_roi_perturbation_results(testDataset, positive_classes, modelName, bestModelPth, dataroot='.', transformsConfig=None, limit = 10000):

    print(f"Best model path: {bestModelPth}")
    bestmodel = bestModelPth.split("/")[-1]
    print(f"Best model: {bestmodel}")

    #SETTING THE ARCHITECTURE OF THE MODEL
    if modelName == "CustomResNetBinary":
        model = CustomResNetBinary()
    elif modelName == "CustomResNetBinary50":
        model = CustomResNetBinary50()
    elif modelName == "EfficientNetB0":
        model = EfficientNetB0()
    elif modelName == "CustomDenseNet":
        model = CustomDenseNet()
    elif modelName == "CustomMobileNetV3":
        model = CustomMobileNetV3()
    else:
        raise ValueError(f"Model {modelName} not recognized. Please choose a valid model.")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    state_dict = torch.load(bestModelPth, map_location = device)
    model.load_state_dict(state_dict)
    model = model.to(device) 

    
    DatasetLesion = lesionDataset(dataPath = testDataset ,positive_classes = positive_classes, transforms_config=transformsConfig, dataroot=dataroot, limit=limit)

    test_dataset = DatasetLesion
    print(f"Test Dataset Size: {len(test_dataset)}")
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_test)


    #INFERENCE  
    model.eval()
    all_predictions_explained_W = {}
    
    LContrib_Th = [0.25, 0.5, 0.75]
    for th in LContrib_Th:
        all_predictions_explained_W[th] = []
    
    
    map_results = {}
    map_types = ['contribution']

    energy = {}
    for th in LContrib_Th:
        energy[th] = []

    size_alpha_values = [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]        
    size_perturbation = {}
    for s_alpha in size_alpha_values:
        size_perturbation[s_alpha] = copy.deepcopy(energy)

    position_alpha_values = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]        
    position_perturbation = {}
    for p_alpha in position_alpha_values:
        position_perturbation[p_alpha] = copy.deepcopy(energy)

    for t in map_types:
        map_results[t] = {'map': None, 'norm_map': None, 'gray_map': None, # temporal. Used for the current sample
                          'perturbed_size': copy.deepcopy(size_perturbation), 'perturbed_position': copy.deepcopy(position_perturbation)}

    with torch.no_grad():
        for inputs, rois, _, labels, _ in tqdm(test_data_loader):
            if labels[0]!=1:
                continue
            inputs, labels = inputs.to(device), labels.to(device)

            _, _, contrib_map = model(inputs)

            for img, batch_rois in zip(inputs.tolist(), rois):

                cvimg = torch.permute(torch.tensor(img), (1, 2, 0)).cpu().numpy()
                cvimg = (cvimg*255).astype(np.uint8)

                img_size = cvimg.shape

                cv_contrib_map = torch.permute(contrib_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach().numpy()
                cv_contrib_map_norm, grayscale_contrib_map = normalize_map(cv_contrib_map)
                map_results['contribution']['map'] = cv_contrib_map
                map_results['contribution']['norm_map'] = cv_contrib_map_norm
                map_results['contribution']['gray_map'] = grayscale_contrib_map
                

                size_perturbed_rois = perturbe_rois_in_size(batch_rois[positive_classes[0]], size_alpha_values)
                for mtype in map_types:
                    resized_map = cv2.resize(map_results[mtype]['norm_map'], (img_size[1], img_size[0]))                                        
                    for s_alpha in size_alpha_values:
                        p_rois = size_perturbed_rois[s_alpha]
                        ground_truth_mask = get_ground_truth_mask(img_size, img_size,p_rois)
                        for th in LContrib_Th:
                            map_results[mtype]['perturbed_size'][s_alpha][th].append(get_roi_energy_fraction(resized_map, ground_truth_mask, th_zero=th))

                position_perturbed_rois = perturbe_rois_in_position(batch_rois[positive_classes[0]], position_alpha_values, img_size[1], img_size[0])                            
                for mtype in map_types:
                    resized_map = cv2.resize(map_results[mtype]['norm_map'], (img_size[1], img_size[0]))                                        
                    for p_alpha in position_alpha_values:
                        p_rois = position_perturbed_rois[p_alpha]
                        pos_perturbation_results = {th: [] for th in LContrib_Th}
                        for r in p_rois:
                            ground_truth_mask = get_ground_truth_mask(img_size, img_size,r)
                            for th in LContrib_Th:
                                pos_perturbation_results[th].append(get_roi_energy_fraction(resized_map, ground_truth_mask, th_zero=th))
                        for th in LContrib_Th:
                            map_results[mtype]['perturbed_position'][p_alpha][th].append(np.mean(pos_perturbation_results[th]))

                # print(map_results)

    mtype = 'contribution'
    roi_perturbation_results = {'perturbed_size': {}, 'perturbed_position': {}}
    for s_alpha in size_alpha_values:
        roi_perturbation_results['perturbed_size'][s_alpha] = {th: {} for th in LContrib_Th}
        for th in LContrib_Th:
            ref_array = np.array(map_results[mtype]['perturbed_size'][0][th])
            roi_perturbation_results['perturbed_size'][s_alpha][th]['mean_fR'] = float(np.mean(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])))
            roi_perturbation_results['perturbed_size'][s_alpha][th]['std_fR'] = float(np.std(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])))
            roi_perturbation_results['perturbed_size'][s_alpha][th]['mean_fR_diff_with_0'] = float(np.mean(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])-ref_array))
            roi_perturbation_results['perturbed_size'][s_alpha][th]['std_fR_diff_with_0'] = float(np.std(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])-ref_array))
            roi_perturbation_results['perturbed_size'][s_alpha][th]['mean_abs_fR_diff_with_0'] = float(np.mean(np.absolute(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])-ref_array)))
            roi_perturbation_results['perturbed_size'][s_alpha][th]['std_abs_fR_diff_with_0'] = float(np.std(np.absolute(np.array(map_results[mtype]['perturbed_size'][s_alpha][th])-ref_array)))
    for p_alpha in position_alpha_values:
        roi_perturbation_results['perturbed_position'][p_alpha] = {th: {} for th in LContrib_Th}
        for th in LContrib_Th:
            ref_array = np.array(map_results[mtype]['perturbed_position'][0][th])
            roi_perturbation_results['perturbed_position'][p_alpha][th]['mean_fR'] = float(np.mean(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])))
            roi_perturbation_results['perturbed_position'][p_alpha][th]['std_fR'] = float(np.std(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])))
            roi_perturbation_results['perturbed_position'][p_alpha][th]['mean_fR_diff_with_0'] = float(np.mean(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])-ref_array))
            roi_perturbation_results['perturbed_position'][p_alpha][th]['std_fR_diff_with_0'] = float(np.std(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])-ref_array))
            roi_perturbation_results['perturbed_position'][p_alpha][th]['mean_abs_fR_diff_with_0'] = float(np.mean(np.absolute(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])-ref_array)))
            roi_perturbation_results['perturbed_position'][p_alpha][th]['std_abs_fR_diff_with_0'] = float(np.std(np.absolute(np.array(map_results[mtype]['perturbed_position'][p_alpha][th])-ref_array)))
            
    return roi_perturbation_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MedImage binary classification test")
    parser.add_argument('--testset', type=str, required=True, help='Test set')
    parser.add_argument("--seed", type=int, default=51, help="Torch seed")
    parser.add_argument("--model", type=str, default="CustomResNetBinary50", help="Selection of the model to test")
    parser.add_argument("--model_weights_path", type=str, default="/bestModels", help="Path to the model weights")
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--metrics_run_path", type=str, help="Path to the metrics run folder")
    parser.add_argument("--json_suffix", type=str, default=None, help="Suffix to append to metrics jsonl filename")
    parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")
    
    args = parser.parse_args()
    json_suffix = args.json_suffix

    with open(args.augmentation_config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    transformsConfig = config

    roi_perturbation_results = get_roi_perturbation_results(args.testset, args.positive_classes, args.model , args.model_weights_path, 
                                                            dataroot=args.dataroot, transformsConfig=transformsConfig, limit = 100000)

    final_metrics = { "Seed": args.seed,
                      "Architecture": args.model,
                      "Model File": args.model_weights_path.split('/')[-1],
                      "Lesion type": args.positive_classes[0],
                      "ROI Perturbation": roi_perturbation_results
                    }

    json_name = "Final_ROI_perturbation_results.jsonl" if not json_suffix else f"Final_ROI_perturbation_{json_suffix}.jsonl"
    with open(json_name, "a") as f:
        json.dump(final_metrics, f, indent=4)
