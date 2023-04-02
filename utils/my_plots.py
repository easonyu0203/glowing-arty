from torchvision.transforms import transforms as T
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import random
import numpy as np
import torch.nn as nn
import math
import platform

system = platform.system()

if system == 'Windows':
    # Use Microsoft YaHei font for Chinese text on Windows
    font_name = 'Microsoft YaHei'
else:
    # Use SimHei font for Chinese text on other systems (e.g. macOS, Linux)
    font_name = 'SimHei'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False


def random_plot_dataset(dataset: datasets, n_example: int = 5, reverse_norm: bool = True, figsize: int = 5):
    """Plot n random images from the dataset."""
    # random select examples
    indices = random.sample(range(len(dataset)), n_example)
    examples = [dataset[i] for i in indices]

    # Plot the images in a row
    fig, ax = plt.subplots(1, n_example, figsize=(n_example * figsize, figsize))
    for i, (img, label) in enumerate(examples):
        # Reverse the normalization
        if reverse_norm:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std[:, None, None] + mean[:, None, None]
            img = torch.clamp(img, 0, 1)
        # Plot the image and title
        ax[i].imshow(img.permute(1, 2, 0))
        ax[i].set_title(dataset.classes[label], fontsize=20)
        ax[i].axis('off')

    plt.show()


def random_plot_preds(dataset, model, n_example=5, figsize=5):
    """Plot n random images and their predictions from the dataset."""
    # random select examples
    indices = random.sample(range(len(dataset)), n_example)
    examples = [dataset[i] for i in indices]
    have_stn = hasattr(model, 'stn')

    # Calculate the number of rows and columns
    ncols = math.ceil(math.sqrt(n_example))  # Multiply by 2 to accommodate both original and STN images
    nrows = math.ceil(n_example / ncols)
    if have_stn:
        nrows *= 2

    # Plot the images in a grid
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * figsize, nrows * figsize),
                           constrained_layout=True)  # Improve the layout by adding space between subplots
    ax = np.ravel(ax)

    model.eval()
    with torch.inference_mode():

        for i, (images, labels) in enumerate(examples):
            images = images.unsqueeze(0)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.item()
            label = dataset.classes[labels]
            pred = dataset.classes[preds]
            pred_confidence = torch.softmax(outputs, dim=1)[0, preds].item()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = images.squeeze(0) * std[:, None, None] + mean[:, None, None]
            img = torch.clamp(img, 0, 1)

            # Plot the original image and title
            index = i * 2 if have_stn else i
            ax[index].imshow(img.permute(1, 2, 0))
            ax[index].set_title(f"Original\npred: {pred}({pred_confidence:.2f})\nlabel: {label}", fontsize=14)
            ax[index].axis('off')

            # Plot the STN transformed image if applicable
            if have_stn:
                stn_img = model.stn(images).squeeze(0) * std[:, None, None] + mean[:, None, None]
                stn_img = torch.clamp(stn_img, 0, 1)

                ax[i*2 + 1].imshow(stn_img.permute(1, 2, 0))
                ax[i*2 + 1].set_title(f"STN Transformed\npred: {pred}({pred_confidence:.2f})\nlabel: {label}", fontsize=14)
                ax[i*2 + 1].axis('off')

        # Turn off remaining unused axes
        for i in range(n_example * 2, nrows * ncols):
            ax[i].axis('off')

    plt.show()


if __name__ == '__main__':
    train_preprocess = T.Compose([
        T.Resize([232, ]),
        T.CenterCrop(224),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load the dataset
    dataset = datasets.ImageFolder('data', transform=train_preprocess)
    # Plot some examples
    random_plot_dataset(dataset)
