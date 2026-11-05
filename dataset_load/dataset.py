import os
import json
import sys
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abs_path not in sys.path:
    sys.path.append(abs_path)
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps 
from torchvision import transforms
import numpy as np
import cv2
import random
import math
from utils.transforms import apply_clahe, apply_top_hat
from utils.enhance_uniform import enhance_uniform
from typing import List, Tuple
import copy


def data_augmentation_transform():
    return transforms.Compose([
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast = (0.8, 1.5)),
    transforms.RandomRotation(degrees = 10),
])

################## LESION DATASET ################## 
class lesionDataset(Dataset):
    def __init__(self, dataPath, positive_classes, transform_with_class = None, limit = 1000000, testDebug = False, seed = None,transforms_config = None, dataroot='.', withLTimeAugmentation=False):
        
        self.dataPath = dataPath
        self.dataroot = dataroot
        self.testDebug = testDebug
        self.withLTimeAugmentation = withLTimeAugmentation

        cfg = transforms_config or {}

        # Flags y canales desde YAML
        self.load_flipped   = bool(cfg.get("flipped",  False))
        self.load_expanded  = bool(cfg.get("expanded", False))
        self.channels  = cfg.get("channels", [])



        self.transforms_config = cfg.get("transformations", None)

        print(f"[dataset] flipped(V2): {self.load_flipped} | expanded(EXP): {self.load_expanded}")
        print(f"[dataset] channels: {self.channels if self.channels else 'None'}")


        self.data = []
        self.labels =[]
        self.rois =[]
        self.image_sizes=[]
        self.CLASSES = dict()

        for id_class, class_name in enumerate(positive_classes):
            self.CLASSES[class_name] = id_class
        self.NUM_CLASSES = len(positive_classes)
        
        self.transform_with_class = transform_with_class

        if os.path.isfile(dataPath) and dataPath.endswith(".json"):
            self.load_dataset_from_json(dataPath,limit)
        else:
            print('No valid format for loading the dataset', dataPath)
            exit()   
        self.datasetSize = len(self.data)

    def load_dataset_from_json(self, dataPath, limit):
        discarded=0 
        imageDontreaded = 0
        with open(dataPath, 'r') as f:
            dataset = json.load(f)
        print(f"Number of samples in the dataset:{len(dataset)}")
        for _, d in dataset.items():
            if len(self.data) > limit:
                break

            #Flipped y expanded
            basename = os.path.basename(d["image"])
            is_flipped  = "_V2_" in basename          
            is_expanded = "_EXP"  in basename       

            if not self.withLTimeAugmentation and (is_flipped or is_expanded):
                continue

            is_original = (not is_flipped) and (not is_expanded)
            is_pure_v2  = is_flipped and (not is_expanded)
            is_pure_exp = is_expanded and (not is_flipped)
            is_combined = is_flipped and is_expanded  

            include = False
            if is_original:
                include = True  
            else:
                if self.load_expanded and self.load_flipped:
                    include = is_pure_v2 or is_pure_exp or is_combined                      
                elif self.load_expanded and not self.load_flipped:
                    include = is_pure_exp           
                elif self.load_flipped and not self.load_expanded:
                    include = is_pure_v2                
                else:
                    include = False                    

            if not include:
                continue

            if not os.path.exists(os.path.join(self.dataroot,d["image"])):
                discarded += 1
                print(f"Invalid path: {d['image']}")
                continue
            label = [0] * self.NUM_CLASSES
            for l, v in d["label"].items():
                if l in self.CLASSES:
                    if len(v) > 0:
                        label[self.CLASSES[l]] = 1
                    else:
                        label[self.CLASSES[l]] = 0

            file_name = d['image'].split('.')[-2]
            left_or_right = file_name.split('_')[-2]
            ml_or_cc = file_name.split('_')[-1]

            if is_flipped:
                if left_or_right == 'L':
                    left_or_right = 'R'
                elif left_or_right == 'R':
                    left_or_right = 'L'

            type_dict = {
                ('L', 'ML'): [1, 0, 0, 0],
                ('L', 'CC'): [0, 1, 0, 0],
                ('R', 'ML'): [0, 0, 1, 0],
                ('R', 'CC'): [0, 0, 0, 1]
            }
            image_type = np.array(type_dict.get((left_or_right, ml_or_cc), [0, 0, 0, 0]))

            image_type = torch.tensor(image_type, dtype=torch.float32)            

            image = cv2.imread(os.path.join(self.dataroot,d["image"]))
            if image is None:
                print(f"Imagen not found: {d['image']}")
                imageDontreaded += 1
                continue
            rois = d["label"]
            self.data.append((os.path.join(self.dataroot,d["image"]), rois, image_type, label))

            
            self.labels.append(label)
            shape = image.shape
            
            self.rois.append(rois)            
            self.image_sizes.append(shape)
        print(f"Wrong images {imageDontreaded}")
        print(f"Discarded images: {discarded}")
        print(f"Number of samples: {len(self.data)}")


    def get_transform_image(self, name, base):
        if name == "Copy":
            return base
        elif name == "Clahe":
            return apply_clahe(base)
        elif name == "TopHat":
            return apply_top_hat(base)
        elif name == "TopHat5x5":
            return apply_top_hat(base, (5, 5))
        elif name == 'EnhanceUniform':
            return enhance_uniform(base)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_pth, image_rois, image_type, labels = self.data[index]
        scale_factor = 0.2

        image = cv2.imread(image_pth, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        scaled_rois = copy.deepcopy(image_rois)

        scaled_rois = {
            roi_type: [[box[0] * scale_factor, box[1] * scale_factor, box[2] * scale_factor, box[3] * scale_factor]for box in boxes]
        for roi_type, boxes in scaled_rois.items()
        }


        img_name = os.path.basename(image_pth)
        
        image_array_pil = Image.fromarray(image)
        
        if self.transform_with_class:
            image_array = self.transform_with_class(image_array_pil)
        else:
            image_array = image_array_pil

        base = np.array(image_array)

        if getattr(self, "channels", None):
            channels_out = []       
            for processing in self.channels:
                if len(channels_out) >= 3:
                    break
                arr = self.get_transform_image(processing, base)

                channels_out.append(np.clip(arr, 0, 255).astype(np.uint8))

            finalImage = np.stack(channels_out, axis=0) if channels_out else base[None, ...]
            image_array = (torch.from_numpy(finalImage).float()) / 255.0

        else:
            image_array = transforms.ToTensor()(image_array)


        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return image_array, scaled_rois, image_type, label_tensor, img_name
    

# Collate and help functions

def round_up(x: int, base: int) -> int:
    return int(math.ceil(x / base)) * base

def make_pad_to_multiple(sizes: List[Tuple[int,int]], multiple: int = 32) -> Tuple[int,int]:
    H_max = max(h for h, w in sizes)
    W_max = max(w for h, w in sizes)
    return round_up(H_max, multiple), round_up(W_max, multiple)


# Training collate: pad variable-sized images to create batches.
# Round up to multiples of 32 so even the largest images may receive padding, for greater consistency across the dataset.
def collate_pad_to32(batch):
    imgs, image_rois, img_types, labels, img_names = zip(*batch)
    B = len(imgs)

    proc_imgs = []
    sizes = []
    for x in imgs:
        assert x.dim() == 3, "img must be [C,H,W]"
        C, H, W = x.shape
        sizes.append((H, W))
        proc_imgs.append(x)
    H_pad, W_pad = make_pad_to_multiple(sizes, multiple=32)

    batch_imgs = torch.zeros(B, 3, H_pad, W_pad, dtype=proc_imgs[0].dtype)

    for i, x in enumerate(proc_imgs):
        _, H, W = x.shape
        batch_imgs[i, :, :H, :W] = x

    labels = torch.stack(labels).view(B, 1)
    img_types = torch.stack(img_types).view(B, 4)

    return batch_imgs, list(image_rois) ,img_types, labels, list(img_names)

# Test collate: no padding is applied; all images in a batch must have the same size.
# Avoids artificial regions affecting spatial explanations.
def collate_test(batch):
    imgs, image_rois, img_types, labels, img_names = zip(*batch)
    B = len(imgs)

    batch_imgs = torch.stack(imgs)

    labels = torch.stack(labels).view(B, 1)
    img_types = torch.stack(img_types).view(B, 4)

    return batch_imgs, list(image_rois) ,img_types, labels, list(img_names)

