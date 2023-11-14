import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms


class TrainMelanomaDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        target = tensor(int(self.data.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
        return image, target


class TestMelanomaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data[idx])

        image = Image.open(img_name)
        label = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image, label


def init_train_data(csv_path, root_dir, batch_size, seed, validation_split=0.15):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])
    }
    df = pd.read_csv(csv_path)

    train_data, test_data = train_test_split(df, test_size=validation_split, shuffle=True, random_state=seed)

    # extract labels from train_data
    train_labels = np.array(train_data['target'])

    # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

    train_dataset = TrainMelanomaDataset(data=train_data,
                                         root_dir=root_dir,
                                         transform=data_transforms['train'])

    val_dataset = TrainMelanomaDataset(data=test_data,
                                       root_dir=root_dir,
                                       transform=data_transforms['val'])
    class_weights = [(1 / len(train_data[train_data['target'] == 0])), (1 / len(train_data[train_data['target'] == 1]))]
    print("Class weights: ", class_weights)
    samples_weights = [class_weights[t] for t in train_labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=3)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, class_weights


def init_test_data(root_dir, batch_size):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = TestMelanomaDatasetTest(root_dir=root_dir,
                                      transform=data_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return loader

