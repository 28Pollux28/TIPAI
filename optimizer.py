import torch


def init_optimizer(model, optimizer_name, lr, weight_decay=0.0, use_scheduler=False,model_name=None):
    if model_name == "resnet50":
        params = model.fc.parameters()
    elif model_name == "efficientnet_b4":
        params = model.classifier.parameters()
    elif model_name == "efficientnet_v2_m":
        params = model.classifier.parameters()
    else:
        params = model.parameters()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer_name))
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
        return optimizer, scheduler
    return optimizer, None

def init_loss(loss, weight=None, device='cpu'):
    if loss == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    elif loss == 'binary_cross_entropy':
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return loss_fn.to(device)
