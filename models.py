import torch
from torchvision import models


def init_resnet50_2_class(device, fine_tune=False):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 2),
        torch.nn.Softmax(dim=1)
    )
    return model.to(device)


def init_resnet50_1_class(device, fine_tune=False):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1),
        torch.nn.Sigmoid()
    )
    return model.to(device)


def init_efficientnet_b4_1_class(device, fine_tune=False):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 1),
        torch.nn.Sigmoid()
    )
    return model.to(device)


def init_efficientnet_b4_2_class(device, fine_tune=False):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 2),
        torch.nn.Softmax(dim=1)
    )
    return model.to(device)


def init_efficientnet_v2_m_1_class(device, fine_tune=False):
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 1),
        torch.nn.Sigmoid()
    )
    return model.to(device)


def init_efficientnet_v2_m_2_class(device, fine_tune=False):
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 2),
        torch.nn.Softmax(dim=1)
    )
    return model.to(device)


def init_model(model_name, class_number=2, finetune=False, device='cpu'):
    print("Init model:", model_name)
    if class_number == 2:
        if model_name == "resnet_50":
            model = init_resnet50_2_class(device=device, fine_tune=finetune)
        elif model_name == "efficientnet_b4":
            model = init_efficientnet_b4_2_class(device=device, fine_tune=finetune)
        elif model_name == "efficientnet_v2_m":
            model = init_efficientnet_v2_m_2_class(device=device, fine_tune=finetune)
        else:
            print("Model not supported")
            exit(1)
    elif class_number == 1:
        if model_name == "resnet_50":
            model = init_resnet50_1_class(device=device, fine_tune=finetune)
        elif model_name == "efficientnet_b4":
            model = init_efficientnet_b4_1_class(device=device, fine_tune=finetune)
        elif model_name == "efficientnet_v2_m":
            model = init_efficientnet_v2_m_1_class(device=device, fine_tune=finetune)
        else:
            print("Model not supported")
            exit(1)
    else:
        print("Class number not supported")
        exit(1)
    return model
