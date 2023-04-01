import matplotlib.pyplot as plt
import random
from torchvision import datasets
import numpy as np
from torchvision.transforms import transforms as T
import torch
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
