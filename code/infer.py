from __future__ import print_function
import os
import time
import math

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

def infer(model, criterion, data_loader, epoch, step, args):
    epoch_start = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    print('Test data num: {}'.format(epoch_data_len))

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            batch_start = time.time()
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            loss_ = loss.item() * image.size(0) # this batch loss
            correct_ = torch.sum(preds == target.data) # this batch correct number

            running_loss += loss_
            running_corrects += correct_

            batch_end = time.time()
            if i % args['print_freq'] == 0:
                print('[VAL] Epoch: {}/{}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(step,
                      epoch, args['epochs'], i, math.ceil(epoch_data_len/args['batch_size']), correct_.double()/image.size(0),
                      loss_/image.size(0), batch_end-batch_start))

        epoch_loss = running_loss / epoch_data_len
        epoch_acc = running_corrects.double() / epoch_data_len
        epoch_end = time.time()
        print('[Val@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}'.format(epoch,
              args['epochs'], epoch_acc, epoch_loss, epoch_end-epoch_start))
        print()
    return epoch_acc


def test(args):
    testdir = args['test_dir']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    st = time.time()

    test_dataset = torchvision.datasets.ImageFolder(
                testdir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,]))
    
    test_data_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args['batch_size'],
                shuffle=True, num_workers=args['workers'], pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    model = torchvision.models.__dict__[args['model']](pretrained=True)
    checkpoint = torch.load(args['model_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    infer(model, criterion, test_data_loader, args)

if __name__ == "__main__":
    with open('configs.yaml') as f:
        config = yaml.safe_load(f)

    test(config)