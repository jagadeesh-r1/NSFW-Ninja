import utils.classifier_utils as classifier_utils

import time
import yaml

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import ResNet101_Weights
import torch.nn.functional as F

from argparse import ArgumentParser

def inference(args):
    # test_dataset = classifier_utils.loadDataset(datadir = args.folder, train=False)
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1,
    #     shuffle=False, num_workers=16, pin_memory=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])


    model = torchvision.models.__dict__['resnet101'](weights=ResNet101_Weights.IMAGENET1K_V1)
    checkpoint = torch.load(args.model, map_location='gpu')
    model.load_state_dict(checkpoint['model'])

    imgs_list = os.listdir(args.folder)
    for img_name in imgs_list:
        img_name = os.path.join(args.folder, img_name)
        img = Image.open(img_name).convert('RGB')
        img = transform(img)

        output = model(img)
        print(output.shape)




    # correct_preds = 0
    # test_loss = 0.0
    # with torch.no_grad():
    #     for i, (image) in enumerate(test_data_loader, 0):
    #         image = image.cuda()
    #         print(image.shape)
            
            # output = model(image)
            # output = output.squeeze()

            # loss = criterion(output, target)

            # test_loss += loss.item()
            # y_pred = torch.round(torch.sigmoid(output))
            # correct_preds += torch.sum(y_pred == target).cpu()
    
    # acc = correct_preds / len(val_data_loader.dataset)
    # print("Epoch : {}   Test Accuracy : {}".format(acc))
    # print("")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-f", "--folder", dest="folder", help="inference folder containing images")
    parser.add_argument("-m", "--model", dest="model", help="model path")
    args = parser.parse_args()

    inference(args)