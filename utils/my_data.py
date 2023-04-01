import os
import random
import shutil
from tqdm import tqdm


def split_dataset(data_dir, val_ratio, data_ratio=1.0):
    """
    Splits the images in a directory into training and validation sets.

    Args:
        data_dir (str): The directory containing the image data.
        val_ratio (float): The ratio of validation images to total images.

    Returns:
        (str, str): The paths to the training and validation directories.
    """
    # Create train and val directories if they don't exist
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(train_dir) or os.path.exists(val_dir):
        answer = input(
            f'The directories {train_dir} and/or {val_dir} already exist. Do you want to delete them? [y/N] ')
        if answer.lower() == 'y':
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(val_dir, ignore_errors=True)
        else:
            print('Aborting.')
            return train_dir, val_dir
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # Loop through each class directory
    for class_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, class_dir)):
            continue

        if class_dir == 'train' or class_dir == 'val':
            continue

        # Create train and val directories for this class
        class_train_dir = os.path.join(train_dir, class_dir)
        class_val_dir = os.path.join(val_dir, class_dir)
        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)
        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)

        # Loop through each image in this class directory
        class_images = os.listdir(os.path.join(data_dir, class_dir))
        random.shuffle(class_images)
        num_images_to_use = int(len(class_images) * data_ratio)
        class_images = class_images[:num_images_to_use]
        num_val_images = int(len(class_images) * val_ratio)
        with tqdm(total=len(class_images), desc=f'Splitting {class_dir} data') as pbar:
            for i, image in enumerate(class_images):
                src = os.path.join(data_dir, class_dir, image)
                if i < num_val_images:
                    dst = os.path.join(class_val_dir, image)
                else:
                    dst = os.path.join(class_train_dir, image)
                shutil.copy(src, dst)
                pbar.update(1)

    return train_dir, val_dir


if __name__ == '__main__':
    # Split the dataset
    train_dir, val_dir = split_dataset('../data', 0.2)

    # Print the number of images in each directory
    print(f'Training images: {len(os.listdir(train_dir))}')
    print(f'Validation images: {len(os.listdir(val_dir))}')
