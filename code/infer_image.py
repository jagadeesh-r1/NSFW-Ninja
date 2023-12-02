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

from argparse import ArgumentParser
import os
from PIL import Image


def inference(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])


    model = torchvision.models.__dict__['resnet18'](weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.DataParallel(model, [0,1])
    model.cuda()

    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    imgs_list = os.listdir(args.folder)
    with torch.no_grad():
        for img_name in imgs_list:
            img_name = os.path.join(args.folder, img_name)
            img = Image.open(img_name).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0).cuda()
            output = model(img)
            y_pred = torch.round(torch.sigmoid(output))
            y_pred = y_pred.cpu().numpy()[0][0]
            print("Image - {} : {}".format(img_name, y_pred))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-f", "--folder", dest="folder", help="inference folder containing images")
    parser.add_argument("-m", "--model", dest="model", help="model path")
    args = parser.parse_args()

    inference(args)