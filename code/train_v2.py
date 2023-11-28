import utils.classifier_utils as dataloader, accuracy

import time
import yaml

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import ResNet101_Weights
import torch.nn.functional as F



def run_one_epoch(model, criterion, optimizer, data_loader, epoch, args):
    # pass
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    print('Total Training Data: {}'.format(epoch_data_len))

    for i, (image, target) in enumerate(data_loader):
        print("Image shape : ", image.shape)
        print("Target Shape : ", target)

        image, target = image.cuda(), target.cuda()
        target = target.type(torch.float)
        
        output = model(image)
        output = output.squeeze()
        print("Output : ", output.shape)
        loss = criterion(output, target)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # _, preds = torch.max(output, 1)
        preds = torch.round(torch.sigmoid(output))


def train(args):
    train_dataset = dataloader.loadDataset(datadir = args['train_dir'], train=True)
    # val_dataset = dataloader.loadDataset(datadir = args['val_dir'], train=False)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'],
        shuffle=True, num_workers=args['num_workers'], pin_memory=args['pin_memory'])
    
    # val_data_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args['batch_size'],
    #     shuffle=False, num_workers=args['workers'], pin_memory=args['pin_memory'])


    model = torchvision.models.__dict__[args['model']](weights=ResNet101_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    model = nn.DataParallel(model, args['gpus'])
    model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['optim_milestones'], gamma=args['lr_gamma'])


    if args['resume']:
        checkpoint = torch.load(args['model_path'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    # Training
    # start_time = time.time()
    for epoch in range(args['epochs']):
        run_one_epoch(model, criterion, optimizer, train_data_loader, epoch, args)
        lr_scheduler.step()
    
    # end_time = time.time()
    # print("Total Time Taken : {}".format(end_time - start_time))


if __name__ == "__main__":
    with open('configs/train_config.yaml') as f:
        config = yaml.safe_load(f)


    # if not os.path.exists(config['checkpoints']):
    #     os.mkdir(config['checkpoints'])

    # g_val_accs = {}

    train(config)