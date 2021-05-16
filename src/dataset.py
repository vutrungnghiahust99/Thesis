from PIL import Image

import torch
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self,
                 root: str,
                 transform):
        super(MNIST, self).__init__()
        self.data, self.targets, self.heads = torch.load(root)
        self.transform = transform

    def __getitem__(self, index: int):
        img, target, heads = self.data[index], int(self.targets[index]), self.heads[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, heads

    def __len__(self) -> int:
        return len(self.data)


class CIFAR10(Dataset):
    def __init__(
            self,
            root: str,
            transform=None):

        super(CIFAR10, self).__init__()
        s = torch.load(root)
        self.data, self.labels, self.mapping = s['data'], s['labels'], s['mapping']
        self.transform = transform

    def __getitem__(self, index: int):
        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, self.mapping[index]

    def __len__(self) -> int:
        return len(self.data)


class CIFAR10_TEST(Dataset):
    def __init__(
            self,
            root: str,
            transform=None):

        super(CIFAR10, self).__init__()
        s = torch.load(root)
        self.data, self.labels = s['data'], s['labels']
        self.transform = transform

    def __getitem__(self, index: int):
        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
