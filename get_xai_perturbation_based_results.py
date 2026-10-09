import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import copy
from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


from models.models import  EfficientNetB0, CustomResNetBinary, CustomResNetBinary50, CustomDenseNet, CustomMobileNetV3

from dataset_load.dataset import lesionDataset, normal_transform

class CamWrapperAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out,(tuple, list)):
            out = out[0]
        return out

def find_last_spatial_layer(model, input_size=(1, 3, 224, 224), device="cpu"):
    model = model.to(device).eval()
    x = torch.zeros(input_size).to(device)

    last_spatial_module = None

    def hook(module, inp, out):
        nonlocal last_spatial_module
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            last_spatial_module = module

    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return last_spatial_module
    

def get_gradCam_map(cam_model, target_layers, inputs):
    with torch.enable_grad():
        gradcam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = GradCAMPlusPlus(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                cam_map = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                gradcam_maps.append(cam_map)
                
                cam.activations_and_grads.activations = []
                cam.activations_and_grads.gradients = []

    return gradcam_maps

def get_eigenCam_map(cam_model, target_layers, inputs):
    with torch.enable_grad():
        eigencam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = EigenCAM(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                cam_map = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                eigencam_maps.append(cam_map)

                cam.activations_and_grads.activations = []
                cam.activations_and_grads.gradients = []

    return eigencam_maps

def get_shapleyCam_map(cam_model, target_layers, inputs):
    with torch.enable_grad():
        shapleycam_maps = []
        for i, layer in enumerate(target_layers):
            cam_ctx = ShapleyCAM(model=cam_model, target_layers=[layer])
            with cam_ctx as cam:
                cam_map = cam(input_tensor=inputs, targets=[BinaryClassifierOutputTarget(1)])[0]
                shapleycam_maps.append(cam_map)

                cam.activations_and_grads.activations = []
                cam.activations_and_grads.gradients = []

    return shapleycam_maps

# Image perturbation

def gaussian_kernel2d(kernel_size, sigma, device, dtype):
    """Creates a normalized 2-D Gaussian kernel."""
    coords = torch.arange(
        kernel_size, device=device, dtype=dtype
    ) - (kernel_size - 1) / 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    return kernel


def gaussian_filter(x, kernel_size, sigma):
    """
    Gaussian filtering of a [C,H,W] tensor.
    Each channel is filtered independently.
    """
    C = x.shape[0]

    kernel = gaussian_kernel2d(
        kernel_size, sigma, x.device, x.dtype
    )

    kernel = kernel[None, None].repeat(C, 1, 1, 1)

    x = x.unsqueeze(0)

    y = F.conv2d(
        x,
        kernel,
        padding=kernel_size // 2,
        groups=C
    )

    return y.squeeze(0)


@torch.no_grad()
def deletion_curve(
    model,
    image,
    explanation_map,
    percentages=None,
    sigma=40.0,
    kernel_size=101,
    target_fn=None,
    MoRF = True,
    blur_type = 'mean'
):
    """
    Computes a MoRF or LeRF deletion curve using mean blur as perturbation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.

    image : torch.Tensor
        Input image with shape [C, H, W].

    explanation_map : torch.Tensor
        Spatial explanation map with shape [h, w].
        Higher values indicate higher relevance.

    percentages : sequence of float, optional
        Fractions of the image to perturb, in [0, 1].
        Default: 0, 0.05, 0.10, ..., 1.0.

    sigma : float
        Standard deviation of the Gaussian kernel.

    kernel_size : int
        Size of the kernel. Must be odd.

    target_fn : callable, optional
        Function that receives the model output and returns the
        positive-class logit for every sample in the batch.

    MoRF : boolean
        Perturbation based on the most relevant pixels (True) or on the least relevant pixels (False)


    Returns
    -------
    percentages : torch.Tensor
        Perturbed fractions.

    logits : torch.Tensor
        Positive-class logits for each perturbation level.
    """

    if percentages is None:
        percentages = torch.linspace(0, 1, 21, device=image.device)
    else:
        percentages = torch.as_tensor(
            percentages, dtype=torch.float32, device=image.device
        )

    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W]")

    if explanation_map.ndim != 2:
        raise ValueError("explanation_map must have shape [h, w]")

    C, H, W = image.shape
    h, w = explanation_map.shape

    # ------------------------------------------------------------
    # 1. Generate the blurred reference image
    # ------------------------------------------------------------
    blurred = blur_image(image.unsqueeze(0),
                            sigma=sigma,
                            kernel_size=kernel_size, blur_type=blur_type)[0]


    # ------------------------------------------------------------
    # 2. Rank explanation-map cells from most to least relevant
    # ------------------------------------------------------------
    relevance = explanation_map.flatten()

    order = torch.argsort(relevance, descending=MoRF)

    n_regions = relevance.numel()

    # rank[i] gives the position of cell i in the relevance ranking
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n_regions, device=image.device)

    rank = rank.reshape(h, w)

    # ------------------------------------------------------------
    # 3. Construct all deletion masks
    # ------------------------------------------------------------
    masks = []

    for p in percentages:

        n_delete = round(float(p) * n_regions)

        if n_delete == 0:
            mask = torch.zeros(
                (h, w),
                dtype=torch.float32,
                device=image.device
            )
        else:
            mask = (rank < n_delete).float()

        # Resize the mask to input-image resolution.
        # nearest preserves the region boundaries.
        mask = F.interpolate(
            mask[None, None],
            size=(H, W),
            mode="nearest"
        )[0]
        
        masks.append(mask)

    # [P, 1, H, W]
    masks = torch.stack(masks)
    
 
    # ------------------------------------------------------------
    # 4. Generate all perturbed images
    # ------------------------------------------------------------
    original = image.unsqueeze(0)      # [1,C,H,W]
    blurred = blurred.unsqueeze(0)     # [1,C,H,W]
    
    perturbed = (
        original * (1.0 - masks)
        + blurred * masks
    )
    
    # ------------------------------------------------------------
    # 5. Evaluate all perturbation levels in one batch
    # ------------------------------------------------------------
 
    model.eval()
    
    output, _, _ = model(perturbed)

    if target_fn is not None:
        logits = target_fn(output)
    else:
        logits = output.squeeze(-1)

    return percentages.detach().cpu(), logits.detach().cpu(), perturbed.detach().cpu(), masks.detach().cpu()


def blur_image(image, sigma=40.0, kernel_size=101, blur_type = 'mean'):
    """
    blur implemented in PyTorch.

    image: [B,C,H,W]
    """

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    device = image.device
    dtype = image.dtype

    if blur_type=='gaussian':
        x = torch.arange(
            kernel_size,
            device=device,
            dtype=dtype
        ) - kernel_size // 2

        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    elif blur_type=='mean':        
        kernel_2d = torch.ones((kernel_size, kernel_size), device=device)
        kernel_2d /=kernel_2d.sum()
    else:
        raise ValueError("blur_type must be gaussian or mean")
    
    C = image.shape[1]

    kernel = kernel_2d.expand(C, 1, kernel_size, kernel_size)

    return F.conv2d(
        image,
        kernel,
        padding=kernel_size // 2,
        groups=C
    )



def perturbation_auc(logits, percentages):
    """
    Computes the normalized AUC of a perturbation curve.

    Parameters
    ----------
    logits : torch.Tensor
        Positive-class logits for each perturbation level.
        Shape: [N].

    percentages : torch.Tensor or sequence
        Perturbation fractions corresponding to each logit.
        Example: [0.0, 0.01, 0.02, 0.05, ..., 0.30].

    Returns
    -------
    auc : float
        Normalized area under the probability-vs-perturbation curve.
        The result is in [0, 1].
    """

    logits = torch.as_tensor(logits, dtype=torch.float32)
    percentages = torch.as_tensor(
        percentages,
        dtype=torch.float32,
        device=logits.device
    )

    # Convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Area under the curve
    auc = torch.trapz(probs, percentages)

    # Normalize by the length of the perturbation interval
    interval = percentages[-1] - percentages[0]

    if interval <= 0:
        raise ValueError("percentages must span a non-zero interval.")

    auc = auc / interval

    return auc.item()

def get_cam_maps(cam_model, target_layer, inputs):
    cam_maps = {}
    # results from EIGEN-CAM
    eigen_cam_types = ['eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att']
    sal_maps_eigencam = get_eigenCam_map(cam_model, target_layer, inputs)
    for imap, tmap in enumerate(eigen_cam_types):
        cam_maps[tmap] = sal_maps_eigencam[imap]

    # results from GRAD-CAM++
    grad_cam_types = ['grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att']
    sal_maps_gradcam = get_gradCam_map(cam_model, target_layer, inputs)
    for imap, tmap in enumerate(grad_cam_types):
        cam_maps[tmap] = sal_maps_gradcam[imap]

    # results from SHAPLEY-CAM
    shapley_cam_types = ['shapley_cam_cnn', 'shapley_cam_proj', 'shapley_cam_att']
    sal_maps_shapleycam = get_shapleyCam_map(cam_model, target_layer, inputs)
    for imap, tmap in enumerate(shapley_cam_types):
        cam_maps[tmap] = sal_maps_shapleycam[imap]

    return cam_maps

def evaluate_faithfulness(testDataset, positive_classes, modelName, bestModelPth, dataroot='.', transformsConfig=None, limit=100000, compare_cam=False):

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

    target_layer = [find_last_spatial_layer(model.base_model), model.head.proj, model.head.attn[-1]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    state_dict = torch.load(bestModelPth, map_location = device)
    model.load_state_dict(state_dict)
    model = model.to(device) 

    model.eval()

    cam_model = CamWrapperAdapter(model).to(device)

    normal_data = normal_transform()

    DatasetLesion = lesionDataset(dataPath = testDataset ,positive_classes = positive_classes, transform_with_class = normal_data, transforms_config=transformsConfig, dataroot=dataroot, limit=limit)

    test_dataset = DatasetLesion
    print(f"Test Dataset Size: {len(test_dataset)}")
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_test)

    map_types = ['contribution', 'attention']

    if compare_cam:
        map_types += ['grad_cam_cnn', 'grad_cam_proj', 'grad_cam_att', 
                    'eigen_cam_cnn', 'eigen_cam_proj', 'eigen_cam_att', 
                    'shapley_cam_cnn', 'shapley_cam_proj', 'shapley_cam_att']

    map_results = {}
    for t in map_types:
        map_results[t] = {'map': None, # temporal. Used for the current sample
                          'auc_morf': [], 'auc_lerf': [], 'delta_auc': []}

    with torch.no_grad():
        for inputs, rois, _, labels, img_name in tqdm(test_data_loader):

            img_name = str(img_name[0])
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, att_map, contrib_map = model(inputs)
            outputs = outputs.detach()
            att_map = att_map.detach()
            contrib_map = contrib_map.detach()
            
            probabilities = torch.sigmoid(outputs)
            
            for img, _ ,label, prob, logits in zip(inputs.tolist(), rois ,labels.tolist(), probabilities.tolist(), outputs.tolist()):

                if label[0]!=1 or logits[0]<=0:
                    continue

                # print(f'LABEL {label[0]} - LOGITS {logits[0]}')

                cvimg = torch.permute(torch.tensor(img), (1, 2, 0)).cpu().numpy()
                cvimg = (cvimg*255).astype(np.uint8)

                img_size = cvimg.shape

                cv_contrib_map = torch.permute(contrib_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach().numpy()
                cv_contrib_map = cv2.resize(cv_contrib_map, (img_size[1], img_size[0])) 
                map_results['contribution']['map'] = cv_contrib_map

                cv_att_map = torch.permute(att_map.squeeze(dim=0), (1, 2, 0)).squeeze().cpu().detach().numpy()
                cv_att_map = cv2.resize(cv_att_map, (img_size[1], img_size[0]))
                map_results['attention']['map'] = cv_att_map

                if compare_cam:                
                    cam_maps = get_cam_maps(cam_model, target_layer, inputs)
                    for tmap, map in cam_maps.items():
                        map_results[tmap]['map'] = map

                for tmap in map_results:
                    # print('----------------------', tmap, '------------------------')
                    percentages_morf, logits_morf, perturbed_images, mask_images = deletion_curve(model, inputs[0], torch.tensor(map_results[tmap]['map'], device=device), MoRF=True)
        
                    auc_morf = perturbation_auc(logits_morf, percentages_morf)

                    percentages_lerf, logits_lerf, _, _ = deletion_curve(model, inputs[0], torch.tensor(map_results[tmap]['map'], device=device), MoRF=False)
        
                    # print(percentages_lerf, logits_lerf)

                    auc_lerf = perturbation_auc(logits_lerf, percentages_lerf)

                    # print(f'AUC MoRF {auc_morf}, AUC LeRF {auc_lerf}, Delta AUC {auc_lerf-auc_morf}')
                    map_results[tmap]['auc_morf'].append(auc_morf)
                    map_results[tmap]['auc_lerf'].append(auc_lerf)
                    map_results[tmap]['delta_auc'].append(auc_lerf-auc_morf)

                    # if tmap=='contribution':
                    #     perturbed = perturbed_images
                    #     percentages = percentages_morf
                        

                # idxs = [0, 1, 2, 4, 10]   # 0%, 5%, 10%, 20%, 50%

                # fig, axes = plt.subplots(1, len(idxs), figsize=(15, 5))

                # for ax, idx in zip(axes, idxs):
                #     img = perturbed[idx]

                #     # Assuming the three channels contain the same baseline image.
                #     ax.imshow(img[1], cmap="gray")
                #     ax.set_title(f"{percentages[idx]*100:.0f}%")
                #     ax.axis("off")

                # plt.tight_layout()
                # plt.show()

    n_images = len(map_results['contribution']['auc_morf'])
    print('NUMBER of IMAGES', n_images)
    faithfulness_results = {}
    for tmap in map_results:
        mean_auc_MoRF = float(np.mean(np.array(map_results[tmap]['auc_morf'], dtype=np.float32)))
        std_auc_MoRF = float(np.std(np.array(map_results[tmap]['auc_morf'], dtype=np.float32)))
        mean_auc_LeRF = float(np.mean(np.array(map_results[tmap]['auc_lerf'], dtype=np.float32)))
        std_auc_LeRF = float(np.std(np.array(map_results[tmap]['auc_lerf'], dtype=np.float32)))
        mean_delta_auc = float(np.mean(np.array(map_results[tmap]['delta_auc'], dtype=np.float32)))
        std_delta_auc = float(np.std(np.array(map_results[tmap]['delta_auc'], dtype=np.float32)))

        map_metrics = {'AUC MoRF': [mean_auc_MoRF, std_auc_MoRF],
                       'AUC LeRF': [mean_auc_LeRF, std_auc_LeRF],
                       'Delta AUC':[mean_delta_auc, std_delta_auc]}
        faithfulness_results[tmap] = map_metrics

        # print(tmap, 'AUC MoRF', np.mean(np.array(map_results[tmap]['auc_morf'], dtype=np.float32)))
        # print(tmap, 'AUC LeRF', np.mean(np.array(map_results[tmap]['auc_lerf'], dtype=np.float32)))
        # print(tmap, 'Delta AUC', np.mean(np.array(map_results[tmap]['delta_auc'], dtype=np.float32)))


    return faithfulness_results, n_images


def collate_test(batch):
    """
    batch: list of (img, label)
      - img should be [3,H,W], variable H,W
    """
    imgs, image_rois, img_types, labels, img_names = zip(*batch)
    B = len(imgs)

    proc_imgs = []
    sizes = []
    for x in imgs:
        assert x.dim() == 3, "img must be [C,H,W]"
        C, H, W = x.shape
        sizes.append((H, W))
        proc_imgs.append(x)
        H_pad, W_pad = H, W



    batch_imgs = torch.zeros(B, 3, H_pad, W_pad, dtype=proc_imgs[0].dtype)

    for i, x in enumerate(proc_imgs):
        _, H, W = x.shape
        batch_imgs[i, :, :H, :W] = x

    labels = torch.stack(labels).view(B, 1)
    img_types = torch.stack(img_types).view(B, 4)

    return batch_imgs, list(image_rois) ,img_types, labels, list(img_names)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Faithfulness test")
    parser.add_argument('--testset', type=str, required=True, help='Test set')
    parser.add_argument("--model", type=str, default="CustomResNetBinary50", help="Selection of the model to test")
    parser.add_argument("--model_weights_path", type=str, default="/bestModels", help="Path to the model weights")
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")
    parser.add_argument("--seed", type=int, required=True, help="Torch seed")
    parser.add_argument("--json_suffix", type=str, default=None, help="Suffix to append to metrics jsonl filename")
    parser.add_argument("--compare_cam", action='store_true', help="Compare with CAM methods")    

    args = parser.parse_args()
    json_suffix = args.json_suffix

    with open(args.augmentation_config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    transformsConfig = config

    faithfulness_metrics_report, n_images = evaluate_faithfulness(args.testset, args.positive_classes, args.model , args.model_weights_path, 
                      dataroot=args.dataroot, transformsConfig=transformsConfig,limit = 10000, compare_cam = args.compare_cam)

    final_metrics = { "Seed": args.seed,
                      "Architecture": args.model,
                      "Model File": args.model_weights_path.split('/')[-1],
                      "Lesion type": args.positive_classes[0],
                      "Number of images": n_images,
                      "Faithfulness": faithfulness_metrics_report
                    }

    json_name = "Final_metrics_runs.jsonl" if not json_suffix else f"Final_metrics_runs_{json_suffix}.jsonl"
    with open(json_name, "a") as f:
        json.dump(final_metrics, f, indent=4)
