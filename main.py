import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
from sklearn.metrics import f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


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
        i += 1
        if i % 100 == 0:
            print(i, " batches processed over ", total_size / batch_size, " batches")
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


def init(device):
    csv_file = 'C:/Users/jeanp/Desktop/TIP projet melanome/train-labels.csv'
    image_dir = 'C:/Users/jeanp/Desktop/TIP projet melanome/train-resized'

    transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8209, 0.6367, 0.5940], std=[0.1507, 0.1888, 0.2152])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8209, 0.6367, 0.5940], std=[0.1507, 0.1888, 0.2152])
        ])
    }

    # Create an instance of the MelanomaDataset
    dataset = MelanomaDataset(csv_file=csv_file, root_dir=image_dir, transform=transform['train'])
    print("Dataset loaded")
    batch_size = 100

    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 2)
    model.to(device)
    model.load_state_dict(torch.load('fine_tuned_alexnet.pth'))

    # Calculate class weights
    num_samples = len(dataset)
    data = pd.read_csv(csv_file)
    class_weights = compute_class_weight('balanced', classes=data['target'].unique(), y=data['target'])
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Create a data loader with weighted loss
    sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights, num_samples=num_samples, replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        #sampler=sampler
    )

    #mean, std = batch_mean_and_sd(loader, total_size, batch_size=batch_size)
    #print("Mean and std: \n", mean, std)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {loss}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_alexnet.pth')
    print("Model saved")
    accuracy, f1, precision = test(model, loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    return accuracy, f1, precision


def test_on_test_data(device):
    # Specify the path to your test dataset
    test_image_dir = 'C:/Users/jeanp/Desktop/TIP projet melanome/train-resized'

    # Create a test dataset
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8209, 0.6367, 0.5940], std=[0.1507, 0.1888, 0.2152])
        ])
    #test_dataset = MelanomaDataset(csv_file=test_csv_file, root_dir=test_image_dir, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root=test_image_dir,
        train=False,
        download=False,
        transform=transform
    )

    # Create a test data loader
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=10, shuffle=True)

    # Load the pre-trained model
    model = alexnet(weights=AlexNet_Weights.DEFAULT)  # Initialize a new AlexNet model
    model.classifier[6] = nn.Linear(4096, 2)  # Modify the classifier to match the number of classes
    model.to(device)

    # Load the trained weights from the fine-tuned model
    model.load_state_dict(torch.load('fine_tuned_alexnet.pth'))

    #accuracy, f1, precision = test(model, test_loader, device)
    #print(f'Test Accuracy: {accuracy:.2f}%')
    #print(f'F1 Score: {f1:.2f}')
    #print(f'Precision: {precision:.2f}')


if __name__ == '__main__':
    print("Is CUDA available? ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init(device)
    test_on_test_data(device)
