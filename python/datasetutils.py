import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

dataset_CIFAR10_train = datasets.CIFAR10(root='./data/cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

dataset_CIFAR10_test = datasets.CIFAR10(root='./data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

dataset_CIFAR100_train = datasets.CIFAR100(root='./data/cifar100', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

dataset_CIFAR100_test = datasets.CIFAR100(root='./data/cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))


dataset_IMAGENET_train = datasets.ImageFolder('/home/julien/dataset/cv/imagenet/train', transforms.Compose([
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
]))

dataset_IMAGENET_test = datasets.ImageFolder('/home/julien/dataset/cv/imagenet/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            normalize,
]))
