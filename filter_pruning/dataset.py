import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from utils import Cutout

#Load CIFAR100 dataset
def load_cifar100(batch_size,workers,cutout,cutout_length):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset=datasets.CIFAR100(root='dataset', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=batch_size, num_workers=workers, shuffle=False)
    
    return train_loader, test_loader

#Load CIFAR10 dataset
def load_cifar10(batch_size,workers,cutout,cutout_length):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader

#Load Imagenet dataset
def load_imagenet():
    traindir='/home/DATA/ImageNet_raw/train'
    valdir='/home/shalinis/ImageNet_raw/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=6,  pin_memory=True, shuffle=True)
    
    test_dataset=datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,]))
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)

    return train_dataset, test_dataset

#Load MNIST dataset
def load_mnist():
    train_data= torch.utils.data.DataLoader(datasets.MNIST(root='dataset',train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), download=True),
                                            batch_size=batch_size, num_workers=6, shuffle=True)
    test_loader= torch.utils.data.DataLoader(datasets.MNIST(root='dataset',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), download=True),
                                            batch_size=batch_size, num_workers=4, shuffle=False)
    return train_data, test_loader

if __name__ == "__main__":
    train_data, test_loader=load_imagenet()