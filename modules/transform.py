import torchvision
import torch
from torchvision.transforms import Lambda

class Transforms:
    def __init__(self, mean=None, std=None):
        self.train_transform = []
        self.train_transform.append(torchvision.transforms.ToTensor())
        self.train_transform.append(Lambda(torch.flatten))
        self.test_transform = [
            torchvision.transforms.ToTensor(),
            Lambda(torch.flatten)
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
