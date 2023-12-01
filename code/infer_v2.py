def test(args):
    test_dataset = classifier_utils.loadDataset(datadir = args['test_dir'], train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    model = torchvision.models.__dict__[args['model']](weights=ResNet101_Weights.IMAGENET1K_V1)
    checkpoint = torch.load(args['resume'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['optim_milestones'], gamma=args['lr_gamma'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    correct_preds = 0
    test_loss = 0.0
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
    
    acc = correct_preds / len(val_data_loader.dataset)
    print("Epoch : {}   Test Accuracy : {}".format(acc))
    print("")

if __name__ == "__main__":
    with open('configs/train_config.yaml') as f:
        config = yaml.safe_load(f)

    test(config)