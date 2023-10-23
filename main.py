import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        target = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, target





def batch_mean_and_sd(loader, total_size, batch_size=100):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    i = 0
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        i+=1
        if i%100 == 0:
            print(i, " batches processed over ", total_size/batch_size, " batches")
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std

def init():
    csv_file = 'D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-labels.csv'
    image_dir = 'D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-resized'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8209, 0.6367, 0.5940], std=[0.1507, 0.1888, 0.2152])
    ])

    # Create an instance of the MelanomaDataset
    dataset = MelanomaDataset(csv_file=csv_file, root_dir=image_dir, transform=transform)
    print("dataset loaded")
    batch_size = 100
    total_size = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=10)
    mean, std = batch_mean_and_sd(loader,total_size, batch_size=batch_size)
    print("mean and std: \n", mean, std)
# Create a DataLoader to batch and shuffle the data
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    init()