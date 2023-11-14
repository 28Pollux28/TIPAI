import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

import MelanomaDataset as mDataset
import models
import optimizer as optim
import utils


def train(train_data, train_val, model, optimizer, loss_fn, epochs, run, device, num_classes, scheduler):
    writer = SummaryWriter('runs/ISIIC-experiement-'+str(run))

    train_data_size = len(train_data.dataset)
    val_data_size = len(train_val.dataset)
    train_data_batch_size = train_data.batch_size
    num_batch = train_data_size / train_data_batch_size
    print("Train data size: ", train_data_size)
    print("Val data size: ", val_data_size)
    print("Train data batch size: ", train_data_batch_size)

    for epoch in range(epochs):
        ###Stats###
        if num_classes == 2:
            epoch_total_class = [0,0]
            epoch_correct_class = [0,0]
        epoch_loss = []
        total_distance = 0
        ############


        model.train()
        print("Epoch: ", epoch)
        for i, (inputs, labels) in enumerate(train_data):
            if i % int(num_batch / 20) == 0:
                print("Training Progress: %d%%" % (i * train_data_batch_size * 100 / train_data_size))
            inputs = inputs.to(device)
            # convert labels to float
            optimizer.zero_grad()
            outputs = model(inputs)
            if num_classes == 1:
                # Output is [16, 1] and labels is [16] so we need to reshape labels
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)

            labels = labels.to(device)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            ###Stats###
            if num_classes == 2:
                _, predicted = torch.max(outputs, 1)
                # Calculate batch accuracy per class
                for c in range(num_classes):
                    epoch_total_class[c] += (labels == c).sum().item()
                    epoch_correct_class[c] += (predicted[labels == c] == c).sum().item()
            elif num_classes == 1:
                # Measure distance between output and label
                total_distance += torch.dist(outputs, labels).item()
        if args.use_scheduler:
            scheduler.step(sum(epoch_loss))

        if num_classes == 2:
            print("Training Class 0 accuracy: ", epoch_correct_class[0] / epoch_total_class[0])
            print("Training Class 1 accuracy: ", epoch_correct_class[1] / epoch_total_class[1])
            writer.add_scalar('Accuracy/train/class_0', epoch_correct_class[0] / epoch_total_class[0], epoch)
            writer.add_scalar('Accuracy/train/class_1', epoch_correct_class[1] / epoch_total_class[1], epoch)

        print("Mean Training Loss: ", sum(epoch_loss) / num_batch)
        print("Median Training Loss: ", sorted(epoch_loss)[int(num_batch / 2)])
        print("Total distance: ", total_distance)

        writer.add_scalar('Loss/train/mean', sum(epoch_loss) / num_batch, epoch)
        writer.add_scalar('Loss/train/median', sorted(epoch_loss)[int(num_batch / 2)], epoch)
        writer.add_scalar('Loss/train/total_distance', total_distance, epoch)

        # Make a histogram of the distribution of the loss per batch
        writer.add_histogram('Loss/train/distribution', np.array(epoch_loss), epoch)
        writer.add_histogram('Model/fc/parameters', model.fc[0].weight.flatten(), epoch, bins='tensorflow')

        ### Validate ###
        validate(train_val, model, loss_fn, device, num_classes, writer, epoch)
        if epoch+1 % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'runs/ISIIC-experiement-{}/models/checkpoint-{}.pth'.format(run, epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'runs/ISIIC-experiement-{}/models/checkpoint-last.pth'.format(run))

def validate(train_val, model, loss_fn, device, num_classes, writer, epoch):
    model.eval()
    train_val_size = len(train_val.dataset)
    train_val_batch_size = train_val.batch_size
    num_batch = (train_val_size / train_val_batch_size)
    if epoch ==0:
        print("Val data size: ", train_val_size)
        print("Val data batch size: ", train_val_batch_size)
    ###Stats###
    if num_classes == 2:
        epoch_total_class = [0, 0]
        epoch_correct_class = [0, 0]
    epoch_loss = []
    total_distance = 0
    ############
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train_val):
            if i % int(num_batch / 20) == 0:
                print("Validation Progress: %d%%" % (i * train_val_batch_size * 100 / train_val_size))
            inputs = inputs.to(device)
            if num_classes == 1:
                # Output is [16, 1] and labels is [16] so we need to reshape labels
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            epoch_loss.append(loss.item())
            ###Stats###
            if num_classes == 2:
                _, predicted = torch.max(outputs, 1)
                # Calculate batch accuracy per class
                for c in range(num_classes):
                    epoch_total_class[c] += (labels == c).sum().item()
                    epoch_correct_class[c] += (predicted[labels == c] == c).sum().item()
            elif num_classes == 1:
                # Measure distance between output and label
                total_distance += torch.dist(outputs, labels).item()
        if num_classes == 2:
            print("Validation Class 0 accuracy: ", epoch_correct_class[0] / epoch_total_class[0])
            print("Validation Class 1 accuracy: ", epoch_correct_class[1] / epoch_total_class[1])
            writer.add_scalar('Accuracy/val/class_0', epoch_correct_class[0] / epoch_total_class[0], epoch)
            writer.add_scalar('Accuracy/val/class_1', epoch_correct_class[1] / epoch_total_class[1], epoch)

        print("Mean Validation Loss: ", sum(epoch_loss) / num_batch)
        print("Median Validation Loss: ", sorted(epoch_loss)[int(num_batch / 2)])
        print("Total distance: ", total_distance)

        writer.add_scalar('Loss/val/mean', sum(epoch_loss) / num_batch, epoch)
        writer.add_scalar('Loss/val/median', sorted(epoch_loss)[int(num_batch / 2)], epoch)
        writer.add_scalar('Loss/val/total_distance', total_distance, epoch)

        # Make a histogram of the distribution of the loss per batch
        writer.add_histogram('Loss/val/distribution', np.array(epoch_loss), epoch)
    model.train()


def test(test_data, model, device, num_classes):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    model = models.init_resnet50_1_class(device=device, fine_tune=False)
    dataset = mDataset.TestMelanomaDataset(root_dir='D:/STUDIES/TC4/TIP/IA/isic-2020-resized/test-resized/test-resized/',
                                  transform=data_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)
    checkpoint = torch.load('./runs/ISIIC-experiement-12/models/checkpoint-last.pth')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.to(device)
    with torch.inference_mode():
        # Generate CSV file with image_name,target
        with open('submission.csv', 'w') as f:
            f.write('image_name,target\n')
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                f.write(labels[0].split('.')[0] + ',' + str(outputs[0].item()) + '\n')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get the arguments
    args = utils.get_args()
    print(args)
    ### Init Run ###

    ############# PARAMS ##################
    args.csv = "D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-labels.csv"
    args.data = 'D:/STUDIES/TC4/TIP/IA/isic-2020-resized/train-resized'
    optimizer_name = "adam"
    use_weight = False
    ############# ACTION ##################
    if args.action == "train":
        run = utils.init_run(args.run)
        model = models.init_model(args.model, args.class_number, args.fine_tune, device)
        train_data, train_val, _ = mDataset.init_train_data(args.csv, args.data, args.batch_size, 42)
        optimizer, scheduler = optim.init_optimizer(model, optimizer_name, args.lr, args.wd, args.use_scheduler)
        weight = torch.tensor([1.0, 1.15]).to(device)
        if args.class_number == 2:
            loss_fn = optim.init_loss("cross_entropy", weight=weight if use_weight else None, device=device)
        else:
            loss_fn = optim.init_loss("binary_cross_entropy", weight=weight if use_weight else None, device=device)

        ### Write params to File ###
        with open('runs/ISIIC-experiement-'+str(run)+'/params.txt', 'w') as f:
            f.write('Model: '+args.model+'\n')
            f.write('Optimizer: '+optimizer_name+'\n')
            f.write('Learning rate: '+str(args.lr)+'\n')
            f.write('Use weight: '+str(use_weight)+'\n')
            f.write('Weight: '+str(weight)+'\n')
            f.write('Weight decay: '+str(args.wd)+'\n')
            f.write('Fine tuning: '+str(args.fine_tune)+'\n')
            f.write('Class number: '+str(args.class_number)+'\n')
            f.write('Batch size: '+str(args.batch_size)+'\n')
            f.write('Epochs: '+str(args.epochs)+'\n')
            f.write('Use scheduler: '+str(args.use_scheduler)+'\n')

        train(train_data, train_val, model, optimizer, loss_fn, args.epochs, run, device, args.class_number, scheduler)


    elif args.action == "test":
        # test the model
        test(args.data, args.model_file, device, args.class_number)
