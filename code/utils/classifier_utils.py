import torchvision
from torchvision import transforms
import torch

def loadDataset(datadir=None, train=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        normalize]))
    else:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize]))


def loadInceptionDataset(datadir=None, train=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                        transforms.Resize((342, 342)),
                        transforms.RandomCrop(299),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        normalize]))
    else:
        return torchvision.datasets.ImageFolder(
                datadir,
                transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    normalize]))

def accuracy(y, y_pred):
    y_pred = torch.round(torch.sigmoid(y_pred))
    return torch.sum(y_pred == y) / y.shape[0]            