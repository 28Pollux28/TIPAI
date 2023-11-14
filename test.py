import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MelanomaDatasetTest(Dataset):
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

if __name__ == '__main__':
    model = main.init_model()
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])

    dataset = MelanomaDatasetTest(root_dir='D:/STUDIES/TC4/TIP/IA/isic-2020-resized/test-resized/test-resized/',
                                    transform=data_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    checkpoint = torch.load('models/run_23/checkpoint-59.pth')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.to(device)
    with torch.no_grad():
        # Generate CSV file with image_name,target
        with open('submission.csv', 'w') as f:
            f.write('image_name,target\n')
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                f.write(labels[0] + ',' + str(preds[0][1].item()) + '\n')
                # print(labels[0] + ',' + str(preds[0][1].item()) + '\n')




