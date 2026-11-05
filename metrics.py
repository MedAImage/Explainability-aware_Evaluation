import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(l_true, l_predict, threshold = 0.5):
    l_predict_bin = (l_predict>threshold).astype(int)
    precision = precision_score(l_true,l_predict_bin, average="binary", zero_division=0)
    recall = recall_score(l_true,l_predict_bin, average="binary",zero_division=0)
    f1 = f1_score(l_true,l_predict_bin, average="binary",zero_division=0)
    auc_roc = roc_auc_score(l_true,l_predict)
    acc = accuracy_score(l_true,l_predict_bin)

    return acc, precision, recall, f1, auc_roc

def get_ground_truth_mask(orig_size, map_size, rois):
    H_orig, W_orig = orig_size[0], orig_size[1]
    H_map, W_map = map_size[0], map_size[1]
    ground_truth_mask = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)

    H_scale, W_scale = H_map/H_orig, W_map/W_orig
    for r in rois:
        w, h = max(int(r[2]*W_scale+1), 1), max(int(r[3]*H_scale+1), 1)
        x, y = max(0, int(r[0]*W_scale)), max(0, int(r[1]*H_scale))
        x2, y2 = min(x+w, map_size[1]), min(y+h, map_size[0]) 
        ground_truth_mask[y:y2, x:x2] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    ground_truth_mask = cv2.dilate(ground_truth_mask, kernel)

    return ground_truth_mask

def get_roi_energy_fraction(map, ground_truth_mask, th_zero = 0.5):
    norm_map = np.copy(map)
    norm_map[norm_map<th_zero] = 0

    rois_energy = np.sum(norm_map[ground_truth_mask==1])

    total_energy = np.sum(norm_map)
    fR = rois_energy/(total_energy+np.finfo(np.float32).eps)

    return fR

def pointing_game_topK(map, rois, img_size, K=1):
    H_orig, W_orig = img_size[0], img_size[1]
    H_scale, W_scale = map.shape[0]/H_orig, map.shape[1]/W_orig

    hit = 0
    max_indices = np.argsort(map, axis=None)[-K:]
    for r in rois:
        cx, cy = r[0]+r[2]/2, r[1]+r[3]/2
        w, h = max(int(r[2]*W_scale), 1), max(int(r[3]*H_scale), 1)
        for idx in max_indices:
            max_y = idx // map.shape[1]
            max_x = idx % map.shape[1]
            if abs(max_x-cx)<w/2 and abs(max_y-cy)<h/2:
                hit = 1
                break
    return hit

def get_map_mass(map):
    map_mass = map.sum()
    return map_mass


def normalize_map(map):
    map_norm = (map-np.min(map))
    diff_max_min = (np.max(map)-np.min(map))
    if diff_max_min>0:
        map_norm = map_norm/diff_max_min
    grayscale_map = (map_norm*255).astype(np.uint8)

    return map_norm, grayscale_map

def concordance_corrcoef(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
