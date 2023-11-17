import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

# Load the pre-trained model
model = alexnet(weights=AlexNet_Weights.DEFAULT)  # Initialize a new AlexNet model
model.classifier[6] = nn.Linear(4096, 2)  # Modify the classifier to match the number of classes

# Load the trained weights from the fine-tuned model
model.load_state_dict(torch.load('fine_tuned_alexnet.pth'))

# Assuming your model is in evaluation mode
model.eval()

# Define the transformation for your test data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8209, 0.6367, 0.5940], std=[0.1507, 0.1888, 0.2152])
])

# Specify the path to your test data folder
test_data_path = 'C:/Users/jeanp/Desktop/TIP projet melanome/test-resized'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Create a dataset from the folder
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = sorted(os.listdir(folder_path))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return self.file_list[idx], image

# Create a DataLoader for the test dataset
test_dataset = CustomDataset(test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Test the model on the data and store the outputs in a DataFrame
all_data = []

with torch.no_grad():
    for filenames, images in tqdm(test_loader):
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        batch_data = {'image_name': filenames[0], 'target': probabilities.numpy()[1]}
        all_data.append(pd.DataFrame([batch_data]))

# Concatenate DataFrames and save to CSV
results_df = pd.concat(all_data, ignore_index=True)
results_csv_path = 'predictions2.csv'
results_df.to_csv(results_csv_path, index=False)

print(f'Predictions saved to {results_csv_path}')
