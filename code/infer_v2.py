import utils.classifier_utils as classifier_utils

import time
import yaml

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

import os

def test(args):
    test_dataset = classifier_utils.loadDataset(datadir = args['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    model = torchvision.models.__dict__[args['model']](weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.DataParallel(model, args['gpus'])
    model.cuda()

    # model_files = os.listdir(args['checkpoints'])
    # model_files = [file for file in model_files if file.endswith(".pth")]
    # indices = [int(file.split("_")[1].split(".")[0]) for file in model_files]
    # highest_index = max(indices)
    # best_model_file = f"model_{highest_index}.pth"
    # best_model_file = os.path.join(args['checkpoints'], best_model_file)

    best_model_file = args['model_path']

    checkpoint = torch.load(best_model_file, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    correct_preds = 0
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (image, target) in enumerate(test_data_loader):
            image, target = image.cuda(), target.cuda()
            target = target.type(torch.float)
            
            output = model(image)
            output = output.squeeze()

            loss = criterion(output, target)

            test_loss += loss.item()
            y_pred = torch.round(torch.sigmoid(output))
            correct_preds += torch.sum(y_pred == target).cpu()
    
    acc = correct_preds / len(test_data_loader.dataset)
    print("Test Accuracy : {}".format(acc))
    print("")

if __name__ == "__main__":
    with open('configs/train_config.yaml') as f:
        config = yaml.safe_load(f)

    test(config)