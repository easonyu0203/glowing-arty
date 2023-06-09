import os
from dataclasses import dataclass

from logger import LoggerContext, WandbLogger, ConsoleLogger, Logger
import utils
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from models.resnet50Wrapper import ResNet50Wrapper
from models.resnet18Wrapper import ResNet18Wrapper
from models.mobilenet_small import MobileNetSmallWrapper
from models.stn_mobile_small import STNMobileNetSmall
import numpy as np
import wandb
from tqdm import tqdm
from typing import Tuple

from utils.my_train import EarlyStopper


def initialize_globals():
    global DEVICE, PROJECT_NAME, LOGGER_TYPE, DATA_ROOT, MODEL_ROOT
    DEVICE = utils.my_torch.detect_device(verbose=True)

    PROJECT_NAME = "Arty"
    LOGGER_TYPE = "wandb"
    DATA_ROOT = "data"
    MODEL_ROOT = "trained_models"


@dataclass
class Hyperparameters:
    # model
    model: str = 'stn_mobilenet_small'

    # Data
    val_ratio: float = 0.2
    data_ratio: float = 1

    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 200
    early_stop_patience: int = 30
    target_val_acc: float = 100.0  # 100%

    # Optimizer & LR scheme
    opt_name: str = 'sgd'
    momentum: float = 0.9
    lr: float = 0.0001
    lr_scheduler_name: str = 'cosine'
    lr_warmup_epochs: int = 10
    lr_warmup_method: str = 'linear'
    lr_warmup_decay: float = 0.01

    # Regularization and Augmentation
    weight_decay: float = 2e-05
    norm_weight_decay: float = 0.0
    label_smoothing: float = 0.1

    # Resizing
    train_crop_size: int = 176
    val_crop_size: int = 224

    def to_dict(self):
        return self.__dict__



def setup_training(config: Hyperparameters) -> Tuple[
    DataLoader, DataLoader, nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    print("splitting dataset...")
    train_dir, val_dir = utils.my_data.split_dataset(DATA_ROOT, val_ratio=config.val_ratio,
                                                     data_ratio=config.data_ratio)

    print("Loading training data...")
    train_preprocess = T.Compose([
        T.Resize([232, ]),
        T.CenterCrop(config.train_crop_size),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(root=train_dir, transform=train_preprocess)

    print("Loading validation data...")
    val_preprocess = T.Compose([
        T.Resize([232, ]),
        T.CenterCrop(config.val_crop_size),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(root=val_dir, transform=val_preprocess)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=10)

    print("Creating model...")
    if config.model == 'resnet18':
        model = ResNet18Wrapper(num_classes=len(train_dataset.classes))
    elif config.model == 'resnet50':
        model = ResNet50Wrapper(num_classes=len(train_dataset.classes))
    elif config.model == 'mobile_small':
        model = MobileNetSmallWrapper(num_classes=len(train_dataset.classes))
    elif config.model == 'stn_mobilenet_small':
        model = STNMobileNetSmall(num_classes=len(train_dataset.classes))
    else:
        raise NotImplementedError(f"Model {config.model} is not implemented")
    model.to(DEVICE)

    print("Creating criterion...")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    print("Creating optimizer...")
    if config.opt_name != 'sgd': raise NotImplementedError("Only SGD is supported")
    parameters = utils.my_train.set_weight_decay(
        model,
        config.weight_decay,
        norm_weight_decay=config.norm_weight_decay
    )
    optimizer = torch.optim.SGD(
        parameters,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    print("Creating learning rate scheduler...")
    if config.lr_scheduler_name != 'cosine': raise NotImplementedError("Only cosine is supported")
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        config.epochs - config.lr_warmup_epochs,
        eta_min=0
    )

    if config.lr_warmup_epochs > 0:
        if config.lr_warmup_method != 'linear': raise NotImplementedError("Only linear is supported")
        warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.lr_warmup_decay,
            total_iters=config.lr_warmup_epochs
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return train_loader, val_loader, model, criterion, optimizer, lr_scheduler


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, logger: Logger):
    model.train()
    data_loader_progress = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    example_ct = epoch * len(data_loader.dataset)

    for i, (images, labels) in enumerate(data_loader_progress):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # wandb log every 10 batches
        if i % 10 == 0:
            example_ct += data_loader.batch_size
            acc1, acc3 = utils.my_train.accuracy(outputs, labels, topk=(1, 3))
            loss_value = loss.item()
            lr = optimizer.param_groups[0]['lr']
            logger.log({
                "train/loss": loss_value,
                "train/acc1": acc1,
                "train/acc3": acc3,
                "train/lr": lr,
                "train/epoch": epoch,
                "train/example_ct": example_ct,
            })

            # update tqdm progress bar description with current metrics
            data_loader_progress.set_description(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc1: {acc1:.4f} | Acc3: {acc3:.4f} | LR: {lr:.6f}"
            )


def evaluate(model, criterion, data_loader, example_ct, logger: Logger):
    model.eval()
    acc1_avg, acc3_avg, loss_avg = utils.my_train.AverageMeter(), utils.my_train.AverageMeter(), utils.my_train.AverageMeter()
    with torch.inference_mode():
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # calculate metrix
            acc1, acc3 = utils.my_train.accuracy(outputs, labels, topk=(1, 3))
            loss = loss.item()
            acc1_avg.update(acc1, images.size(0))
            acc3_avg.update(acc3, images.size(0))
            loss_avg.update(loss, images.size(0))

        # Create a list to store logged images
        logged_images = []

        # random sample 4 images for data_loader
        sample_indices = np.random.choice(len(data_loader.dataset), 4, replace=False)
        for i in sample_indices:
            images, labels = data_loader.dataset[i]
            images = images.unsqueeze(0).to(DEVICE)
            outputs = model(images)
            if hasattr(model, 'stn'):
                stn_img = model.stn(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.item()
            label = data_loader.dataset.classes[labels]
            pred = data_loader.dataset.classes[preds]
            pred_confidence = torch.softmax(outputs, dim=1)[0, preds].item()
            # Append logged image to the list
            logged_images.append(wandb.Image(images[0], caption=f"pred: {pred}({pred_confidence:.2f}), label: {label}"))
            if hasattr(model, 'stn'):
                logged_images.append(wandb.Image(stn_img[0], caption=f"STN Transformed Image"))

        # log metrics
        logger.log({
            "val/loss": loss_avg.avg,
            "val/acc1": acc1_avg.avg,
            "val/acc3": acc3_avg.avg,
            "val/sample": logged_images,
            "val/example_ct": example_ct,
        })

    return acc1_avg.avg, acc3_avg.avg, loss_avg.avg


def save_best_model(model, acc1, best_acc1):
    if acc1 >= best_acc1:
        print(f"New best accuracy: {acc1:.2f}, previous: {best_acc1:.2f}")
        model_dir = os.path.join(MODEL_ROOT, model.__class__.__name__)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"best_acc_{acc1:.2f}.pth")
        torch.save(model.state_dict(), model_path)
        return acc1
    return best_acc1


def pipline(hyperparameters: Hyperparameters, logger: Logger = WandbLogger()):
    with LoggerContext(logger, PROJECT_NAME, hyperparameters.to_dict()) as logger:
        config: Hyperparameters = hyperparameters
        train_loader, val_loader, model, criterion, optimizer, lr_scheduler = setup_training(config)

        print("Start training...")
        logger.watch(model, criterion, log="all", log_freq=10)
        best_acc1 = 0.0
        early_stopper = EarlyStopper(config.early_stop_patience, config.target_val_acc)
        for epoch in range(config.epochs):
            train_one_epoch(model, criterion, optimizer, train_loader, epoch, logger)  # Add logger as an argument
            lr_scheduler.step()
            example_ct = (epoch + 1) * len(train_loader.dataset)
            acc1, acc3, loss = evaluate(model, criterion, val_loader, example_ct, logger)  # Add logger as an argument
            best_acc1 = save_best_model(model, acc1, best_acc1)

            # Check for early stopping
            stop = early_stopper.check(acc1)
            if stop:
                print("Early stopping triggered")
                break


def main():
    initialize_globals()
    hyperparameters = Hyperparameters()

    # Change the logger_type to 'console' if you want to use ConsoleLogger
    if LOGGER_TYPE == 'wandb':
        logger = WandbLogger()
    elif LOGGER_TYPE == 'console':
        logger = ConsoleLogger()
    else:
        raise ValueError("Invalid logger_type")

    pipline(hyperparameters, logger)


if __name__ == "__main__":
    main()
