import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from models.models import  EfficientNetB0, CustomResNetBinary,  CustomResNetBinary50,  CustomDenseNet, CustomMobileNetV3
import numpy as np
from dataset_load.dataset import lesionDataset, data_augmentation_transform, collate_pad_to32
import os
import argparse
import yaml
from tqdm import tqdm
import random
from utils.pos_weight_samples import pos_weight_samples

probability_threshold = 0.5

def load_yaml_config(file_path):
    if not file_path or file_path is None:
        return None
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch = 0):
    model.train()
    running_loss = 0.0
    for inputs, _,types, labels, _ in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        types = types.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(inputs) #ignore maps
        loss = criterion_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_data_loader)
    return avg_train_loss


def validation (epoch = 0):
    model.eval()
    val_labels=[]
    val_predictions=[]
    val_loss = 0.0

    with torch.no_grad():
        for inputs, _,types, labels, _ in val_data_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            types = types.to(device)
            outputs, _, _= model(inputs) #ignore maps
            probabilities = torch.sigmoid(outputs)
            loss = criterion_loss(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            predicted_class = (probabilities >= probability_threshold ).float()
            val_labels.append(labels.cpu().numpy())
            val_predictions.append(predicted_class.cpu().numpy())
        avgValLoss = val_loss / len(val_data_loader.dataset)
        return avgValLoss, val_labels, val_predictions


if __name__=="__main__":
    #ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="MedImage binarty classification training")
    parser.add_argument("--config_file", type=str, default=None, help="Path to a config file")
    parser.add_argument('--trainset', type=str, required=True, help='Training set')
    parser.add_argument('--valset', type=str, required=True, help='Validation set')
    parser.add_argument("--seed", type=int, default=51, help="Seed torch for training")
    parser.add_argument("--seed_split", type=str, default=None, help="Seed split for saving the model")
    parser.add_argument("--number_of_epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--model", type=str, default="CustomResNetBinary", help="Selection of the model to train")
    parser.add_argument("--dataroot", type=str, default=".", help="Root path to the dataset")
    parser.add_argument("--positive_classes", type=str, nargs='+', help='List of positive classes (Nodulo, Calc_tip_benig)')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., cuda:1, cpu)")
    parser.add_argument("--model_save_path", type=str, default="bestModels/", help="Path to save the best model")
    parser.add_argument("--augmentation_config_path", type=str, default="augment_transform.yaml", help="Path to the augmentation config file")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix to add to the model file name")  
    args = parser.parse_args()
    print(args)

    #CONFIGURATION WITH ARGUMENTS
    EPOCHS = args.number_of_epochs
    PATIENCE = args.patience
    batchSamples = args.batch_size
    numChannels = 3
    numtypes = 4
    learning_rate = args.learning_rate
    positive_classes = args.positive_classes
    print(f"Probability threshold for classification: {probability_threshold}")
    fix_seed(args.seed)

    # LOAD YAML CONFIG (usar solo channels)
    config = load_yaml_config(args.augmentation_config_path) or {}
    transformsConfig = config  # pasamos el dict al dataset

    channels_list = config.get("channels", [])
    print(f"[train] YAML cargado de: {args.augmentation_config_path}")
    print(f"[train] channels: {channels_list}")


    assert isinstance(channels_list, list) and len(channels_list) == 3, 'The number of channels should be 3. Review your configuration file'

    inchannels = 3 
    
    model_architectures = {
        "CustomResNetBinary": CustomResNetBinary,
        "CustomResNetBinary50": CustomResNetBinary50,
        "EfficientNetB0": EfficientNetB0,
        "CustomDenseNet": CustomDenseNet,
        "CustomMobileNetV3": CustomMobileNetV3
    }

    if args.model not in model_architectures:
        raise ValueError(f"Model {args.model} not recognized. Valid models are: {list(model_architectures.keys())}")
    
    model = model_architectures[args.model]()

    positive_classes_str = '__'.join(args.positive_classes)
    save_completeBestModel_path = os.path.join(args.model_save_path, args.seed_split, positive_classes_str)
    os.makedirs(save_completeBestModel_path, exist_ok=True)
    print(save_completeBestModel_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device) 
    
    #LOADING THE DATASET:
    TrainDatasetJson = args.trainset
    ValDatasetJson = args.valset
    print(f"[DEBUG] train.py | dataroot: {args.dataroot}")
    print(f"[DEBUG] train.py | TrainDatasetJson: {TrainDatasetJson}")
    print(f"[DEBUG] train.py | ValDatasetJson: {ValDatasetJson}")

    data_augmentation = data_augmentation_transform()

    train_dataset = lesionDataset(dataPath = TrainDatasetJson, positive_classes = positive_classes, transform_with_class=data_augmentation, seed = args.seed,transforms_config=transformsConfig ,dataroot=args.dataroot, withLTimeAugmentation=True)
    val_dataset = lesionDataset(dataPath = ValDatasetJson, positive_classes = positive_classes, seed = args.seed, transforms_config=transformsConfig ,dataroot=args.dataroot)

    train_data_loader = DataLoader(train_dataset, batch_size=batchSamples, shuffle=True, num_workers=4, collate_fn = collate_pad_to32)
    val_data_loader = DataLoader(val_dataset, batch_size=batchSamples, shuffle=False, num_workers=4, collate_fn = collate_pad_to32)

    train_labels = np.array(train_dataset.labels)
    val_labels = np.array(val_dataset.labels)

 
    pos_weight = pos_weight_samples(train_labels, device)
    criterion_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight = pos_weight)   
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        
    train_loss = []
    validation_loss =[]
    loss_increase_steps = 0
    best_loss = np.inf

    save_name = os.path.join(save_completeBestModel_path, args.model)
    if args.suffix:
        save_route = f'{save_name}_seed_{args.seed}_{args.suffix}.pth'
    else:
        save_route = f'{save_name}_seed_{args.seed}.pth'

    #TRAINING/VALIDATION LOOP
    for epoch in tqdm(range(EPOCHS)):
        trainloss = train(epoch)
        avgValLoss, val_labels, val_predictions = validation(epoch)
        train_loss.append(trainloss)
        validation_loss.append(avgValLoss)
        print(f"Epoch [{epoch+1}/{EPOCHS}]||Train Loss: {trainloss:.4f}||Validation Loss: {avgValLoss:.4f}\n")
        if avgValLoss < best_loss:
            best_loss = avgValLoss
            print("Saving the model...")
            torch.save(model.state_dict(), f'{save_route}')
            loss_increase_steps = 0
        else:
            loss_increase_steps += 1
            print("Not saving. Waiting for the model to improve with patience", loss_increase_steps)
            if loss_increase_steps>PATIENCE:
                break

