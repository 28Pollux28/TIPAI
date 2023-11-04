import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch import nn, optim, tensor
from tensorboardX import SummaryWriter
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print("Running on", device)


class MelanomaDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        target = tensor(int(self.data.iloc[idx, 1])).to(device)

        if self.transform:
            image = self.transform(image)
            image = image.to(device)
        return image, target


def batch_mean_and_sd(loader, total_size, batch_size=100):
    cnt = 0
    i = 0
    total_sum = torch.empty(3).to(device)
    total_sum_of_square = torch.empty(3).to(device)
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        total_sum += sum_
        total_sum_of_square += sum_of_square
        cnt += nb_pixels
        i+=1
        if i%100 == 0:
            print(images.shape)
            print(i, " batches processed over ", total_size/batch_size, " batches")
    mean = total_sum / cnt
    snd_moment = total_sum_of_square / cnt
    std = torch.sqrt(snd_moment - mean ** 2)
    return mean, std


def init():
    csv_file = 'D:\STUDIES/TC4/TIP/IA/isic-2020-resized/train-labels.csv'
    image_dir = 'D:\\STUDIES\\TC4\\TIP\\IA\\isic-2020-resized\\train-resized\\'

    transform = transforms.Compose([
        # transforms.PILToTensor(),
        # transforms.ToTensor(),
        # transforms.AutoAugment(),
        transforms.ToTensor(),
        # transforms.RandomAutocontrast(),
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
        num_workers=1)
    mean, std = batch_mean_and_sd(loader,total_size, batch_size=batch_size)
    print("mean and std: \n", mean, std)
    #
    # import matplotlib.pyplot as plt
    #
    # for images, i in loader:
    #     image_to_display = images
    #     break
    # for image in images:
    #
    #     # Convert the tensor to a NumPy array
    #     image_to_display = image.permute(1, 2, 0).numpy()
    #     # Display the image using Matplotlib
    #     plt.imshow(image_to_display)
    #     plt.pause(0.5)
# Create a DataLoader to batch and shuffle the data
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def init_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(device)

    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = True

    # Customize the final classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 2)  # 2 output classes (melanoma and non-melanoma)
    ).to(device)
    return model


def param_model(model, class_weights):
    # Set up L2 regularization (weight decay) in the optimizer
    weight_decay = 1e-2  # Adjust the weight decay coefficient as needed
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=weight_decay)

    # Set up class weights (adjust as needed)
    class_weights = torch.tensor([1.0, 1.0]).to(device)  # Example: Weight class 1 less and class 2 more

    # Set up loss and move it to the GPU
    # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(device)

    return optimizer, criterion


def load_data(batch_size=16):
    # Data augmentation and loading
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.8209, 0.6367, 0.5940], [0.1507, 0.1888, 0.2152])  # precomputed
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.8209, 0.6367, 0.5940], [0.1507, 0.1888, 0.2152])
        ])
    }
    df = pd.read_csv('D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-labels.csv')
    validation_split = 0.1
    train_data, test_data = train_test_split(df, test_size=validation_split, shuffle=True, random_state=seed)

    # extract labels from train_data
    train_labels = np.array(train_data['target'])

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print("class_weights: ", class_weights)
    class_weights = torch.from_numpy(class_weights)

    # Create an instance of the MelanomaDataset class
    train_dataset = MelanomaDataset(data=train_data,
                                    root_dir='D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-resized',
                                    transform=data_transforms['train'])

    val_dataset = MelanomaDataset(data=test_data,
                                    root_dir='D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-resized',
                                    transform=data_transforms['val'])
    # Compute the class weights
    n_non_malignant = len(train_data[train_data['target'] == 0])
    n_malignant = len(train_data[train_data['target'] == 1])
    weight_non_malignant =1 / n_non_malignant
    weight_malignant = 1 / n_malignant
    print("Total Train Samples: ", len(train_data))
    print("Non-malignant Train Samples: ", n_non_malignant)
    print("Malignant Train Samples: ", n_malignant)

    print("Total Validation Samples: ", len(test_data))
    print("Non-malignant Validation Samples: ", len(test_data[test_data['target'] == 0]))
    print("Malignant Validation Samples: ", len(test_data[test_data['target'] == 1]))

    # Create a custom data sampler to ensure equal class representation in each batch
    class_count = [weight_non_malignant, weight_malignant]  # Specify the number of samples for each class
    weight = torch.tensor(class_count, dtype=torch.float).to(device)
    samples_weight = np.array([class_count[t] for t in train_data['target']])

    sampler = WeightedRandomSampler(samples_weight, len(train_dataset), replacement=True)

    # Set up data loaders for training and validation data, and move data to the GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, weight, len(train_dataset), len(val_dataset), batch_size


def train(model, optimizer, criterion, train_loader, val_loader, run, num_epochs=10, start_epoch=0, train_dataset_len=0, val_dataset_len=0, batch_size=16):
    # Training loop
    writer = SummaryWriter('runs/ISIIC-experiement-'+str(run))
    for epoch in range(start_epoch,start_epoch+num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        num_classes = 2  # Replace with the actual number of classes
        class_correct_train = [0] * num_classes
        class_correct_val = [0] * num_classes
        class_total_train = [0] * num_classes
        class_total_val = [0] * num_classes
        total_loss = 0.0
        num_batches = 0
        model.train()
        for i, data in enumerate(train_loader,0):
            if i % int(train_dataset_len/batch_size/20) == 0:
                print('Training batch progress %2d %%' % (100 * i / int(train_dataset_len/batch_size)))
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches+=1

            _, predicted = torch.max(outputs, 1)

            # Calculate batch accuracy per class
            for c in range(num_classes):
                class_correct_train[c] += (predicted[labels == c] == c).sum().item()
                class_total_train[c] += (labels == c).sum().item()

                # if labels.sum().item() > 0:
                #     print("labels: ", labels)
                #     print("outputs: ", outputs)
                #     print("predicted: ", predicted)
                #     print("class_correct_train: ", class_correct_train)
                #     print("class_total_train: ", class_total_train)

        print("Finished Training Epoch %d" % (epoch + 1))
        # Validation loop
        total_val_loss = 0.0
        num_val_batches = 0
        model.eval()
        with torch.inference_mode():
            for i, data in enumerate(val_loader,0):
                if i % int(val_dataset_len / batch_size / 20) == 0:
                    print('Validation batch progress %2d %%' % (100 * i / int(val_dataset_len / batch_size)))
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                num_val_batches += 1

                _, predicted = torch.max(outputs, 1)
                # Calculate batch accuracy per class
                for c in range(num_classes):
                    class_correct_val[c] += (predicted[labels == c] == c).sum().item()
                    class_total_val[c] += (labels == c).sum().item()

        batch_train_loss = total_loss / num_batches
        batch_val_loss = total_val_loss / num_val_batches
        writer.add_scalar('Loss/train_batch', batch_train_loss, epoch)
        writer.add_scalar('Loss/val_batch', batch_val_loss, epoch)
        for c in range(num_classes):
            if class_total_train[c] > 0:
                print('Accuracy of train class %s : %2f %%' % (
                    c, 100 * class_correct_train[c] / class_total_train[c]))
                class_acc = class_correct_train[c] / class_total_train[c]
                writer.add_scalar(f'Accuracy/train_class_{c}', class_acc, epoch)
            if class_total_val[c] > 0:
                print('Accuracy of validation class %s : %2f %%' % (
                    c, 100 * class_correct_val[c] / class_total_val[c]))
                class_acc = class_correct_val[c] / class_total_val[c]
                writer.add_scalar(f'Accuracy/val_class_{c}', class_acc, epoch)
        print('Global accuracy of train: %2f %%' % (
            100 * sum(class_correct_train) / sum(class_total_train)))
        print('Global accuracy of validation: %2f %%' % (
            100 * sum(class_correct_val) / sum(class_total_val)))
        writer.add_scalar('Accuracy/train', sum(class_correct_train) / sum(class_total_train), epoch)
        writer.add_scalar('Accuracy/val', sum(class_correct_val) / sum(class_total_val), epoch)
        if epoch != start_epoch and epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, 'models/'+str(run)+'/checkpoint-'+str(epoch)+'.pth')
        if epoch == start_epoch:
            print('Dataset size: %d, Batch size: %d, Batches per epoch: %d' % (len(train_loader.dataset), len(train_loader), num_batches))
            print('Saw %d non-melanoma images, training accuracy of non-melanoma: %2f %%' % (
                class_total_train[0], 100 * class_correct_train[0] / class_total_train[0]))
            print('Saw %d melanoma images, training accuracy of melanoma: %2f %%' % (
                class_total_train[1], 100 * class_correct_train[1] / class_total_train[1]))
    checkpoint = {
        'epoch': start_epoch+num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'models/' + str(run) + '/checkpoint-' + str(start_epoch+num_epochs-1) + '.pth')

if __name__ == '__main__':
    # Get run number based on existing runs in models folder or from cmd args
    if len(sys.argv) > 1:
        run = sys.argv[1]
    else:
        runs = os.listdir('models')
        run = 'run_1'
        if len(runs) > 0:
            run = 'run_' + str(len(runs) + 1)
        os.mkdir('models/' + run)
        print("run: ", run)

    train_loader, val_loader, weights, train_dataset_len, val_dataset_len, batch_size = load_data()
    # If run exists, get last checkpoint (file checkpoint-N.pth) and load it. N can be larger than 10 so we need to find the largest N
    checkpoints = os.listdir('models/' + run)
    if len(checkpoints) > 0:
        checkpoints = [int(f.split('-')[1].split('.')[0]) for f in checkpoints]
        checkpoints.sort()
        last_checkpoint = checkpoints[-1]
        checkpoint = torch.load('models/' + run + '/checkpoint-' + str(last_checkpoint) + '.pth')

        model = init_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        epoch = checkpoint['epoch']
    else:
        model = init_model()
        optimizer, criterion = param_model(model, class_weights=weights.to(device))
        epoch = 0

    print("weights: ", weights)
    train(model, optimizer, criterion, train_loader, val_loader, run, num_epochs=20, start_epoch=epoch, train_dataset_len=train_dataset_len, val_dataset_len=val_dataset_len, batch_size=batch_size)



    # init()
    # model = init_model()

    # optimizer, criterion = param_model(model, class_weights=weights.to(device))
    # train(model, optimizer, criterion, train_loader, val_loader, num_epochs=20)
