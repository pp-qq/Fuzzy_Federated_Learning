import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.stats import dirichlet
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

dir_path = Path("MNIST_g_test/")
random.seed(1)
test_set_size = 3000


def main():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    test_path = dir_path / "g_test"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get MNIST data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = torchvision.datasets.MNIST(
        root=dir_path / "rawdata",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    print(type(dataset_image))
    print(dataset_image.shape)  # (10000, 28, 28)
    print(type(dataset_label))
    print(dataset_label.shape)  # (10000,)

    # Partition data
    X, y = dataset_image, dataset_label

    # データ数をランダムな3000個にする
    random_index = random.sample(range(len(X)), test_set_size)
    X = X[random_index]
    y = y[random_index]

    print(X.shape)

    X_g_test = np.array(X)
    y_g_test = np.array(y)

    g_test_data = (X_g_test, y_g_test)
    with open(test_path / "g_test.pkl", "wb") as f:
        pickle.dump(g_test_data, f)


if __name__ == "__main__":
    main()
