import os
from typing import Any, Callable, List, Optional, Tuple, Union
import PIL
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from .utils import set_seed

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# Step 4: Create a custom dataset
class SortedMNISTDataset(Dataset):
    def __init__(self, data_label_pairs, transform=None):
        self.data_label_pairs = data_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_label_pairs)

    def __getitem__(self, idx):
        img = self.data_label_pairs[idx][0]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        target = self.data_label_pairs[idx][1]
        return img, target


def stratified_split(args, dataset, num_classes, train_val_split=0.9):
    indices_dir = args.train_val_indices_dir
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)

    seed = args.seed
    train_indices_path = os.path.join(indices_dir, f'train_indices_{seed}.npy')
    val_indices_path = os.path.join(indices_dir, f'val_indices_{seed}.npy')

    if not os.path.exists(train_indices_path) or not os.path.exists(val_indices_path):
        # Set a fixed random seed for reproducibility
        set_seed(args)

        # Define the number of samples in the training set
        num_train_samples = len(dataset)

        # Define the stratified split parameters
        train_size = train_val_split
        val_size = 1 - train_val_split

        # Calculate the number of samples in each class
        class_counts = torch.zeros(num_classes)
        for i in range(num_train_samples):
            _, label = dataset[i]
            class_counts[label] += 1

        # Calculate the number of samples in each class for the training and validation sets
        train_class_counts = (class_counts * train_size).long()
        val_class_counts = (class_counts * val_size).long()

        # Create a list to store the indices for each class
        train_indices = []
        val_indices = []
        for i in range(num_classes):
            class_indices = [j for j in range(num_train_samples) if dataset[j][1] == i]
            train_indices.extend(class_indices[:train_class_counts[i]])
            val_indices.extend(class_indices[train_class_counts[i]:train_class_counts[i] + val_class_counts[i]])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        np.save(train_indices_path, train_indices)
        np.save(val_indices_path, val_indices)

    print('\n===> Found Train/val Split Indices!\n')
    train_indices = list(np.load(train_indices_path))
    val_indices = list(np.load(val_indices_path))

    return train_indices, val_indices


def get_data_loader(args,
                    preprocess=None,
                    sort=False,
                    train_val_split=0.9):
    if args.dataset == 'mnist':
        img_specs = {
            'num_classes': 10,
            'num_channel': 1,
            'width': 28,
            'height': 28
        }

        data_dir = os.path.join(args.data_dir, 'mnist')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=preprocess)

        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform_test)

        if sort:
            # Step 2: Extract data and labels and sort them
            # Convert dataset into a list of (data, label) for sorting
            data_label_pairs = [(data, label) for data, label in zip(train_dataset.data, train_dataset.targets)]

            # Step 3: Sort by label
            data_label_pairs.sort(key=lambda x: x[1])

            train_dataset = SortedMNISTDataset(data_label_pairs, transform=preprocess)

    elif args.dataset == 'cifar10':
        img_specs = {
            'num_classes': 10,
            'num_channel': 3,
            'width': 32,
            'height': 32
        }
        data_dir = os.path.join(args.data_dir, 'cifar10')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    else:
        raise NotImplementedError('Not matching dataset found')

    # split train dataset into train and val
    if train_val_split < 1:
        train_indices, val_indices = stratified_split(args,
                                                      train_dataset,
                                                      num_classes=img_specs['num_classes'],
                                                      train_val_split=train_val_split)
        ## Option 1: Create the subset random samplers for the training and validation sets
        # train_sampler = SubsetRandomSampler(train_indices)
        # val_sampler = SubsetRandomSampler(val_indices)

        ## Option 2: Create the subset datasets for the training and validation sets
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

        val_loader = DataLoader(val_dataset,
                                batch_size=args.eval_batch_size,
                                pin_memory=True,
                                num_workers=0)
    else:
        # We want to compare the effect hyperparamter tuning on test set vs. val set
        # To ensure a fair comparison, we use the same training set for experiments on test set and val set

        # load saved train/val split indices
        train_indices, val_indices = stratified_split(args,
                                                      train_dataset,
                                                      num_classes=img_specs['num_classes'],
                                                      train_val_split=train_val_split)
        train_dataset = Subset(train_dataset, train_indices)
        val_loader = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                              num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.eval_batch_size,
                             pin_memory=True,
                             num_workers=2)

    return train_loader, val_loader, test_loader, img_specs


if __name__ == '__main__':
    from time import time
    import multiprocessing as mp

    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(train_reader, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
