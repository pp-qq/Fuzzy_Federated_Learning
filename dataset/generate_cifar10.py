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
from sklearn.model_selection import train_test_split

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
num_clusters = 5
class_per_client = 3
partission_method = "multi"  # random, multi, prob
data_num_per_client = 3000

dir_path = Path("Cifar10/")


def separate_data(
    images,
    labels,
    num_clients,
    num_classes,
    class_per_client,
    partission_method,
):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]

    if partission_method == "random":
        for i in range(num_clients):
            for _ in range(data_num_per_client):
                idx = np.random.randint(len(images))
                X[i].append(images[idx])
                y[i].append(labels[idx])
        cluster_results = None

    elif partission_method == "multi":
        cluster_results = {
            "clusters": [[] for _ in range(num_clusters)],
            "labels": [[] for _ in range(num_clusters)],
        }

        classes_per_client = [[] for _ in range(num_clients)]

        i = 0
        for client in range(num_clients):
            classes_per_client[client] = [
                i % num_classes,
                (i + 1) % num_classes,
                (i + 2) % num_classes,
            ]
            i += 3
            i %= num_classes

        print(classes_per_client)

        # for i in range(num_clients):
        #     cluster_results["clusters"][i % num_clusters].append(i)

        indices_by_class = [[] for _ in range(num_classes)]

        for i in range(len(labels)):
            indices_by_class[labels[i]].append(i)

        for client_id in range(num_clients):
            for _ in range(data_num_per_client):
                label = np.random.choice(classes_per_client[client_id])
                idx = np.random.choice(indices_by_class[label])
                X[client_id].append(images[idx])
                y[client_id].append(labels[idx])

    elif partission_method == "prob":
        client_label_probabilities = [
            [1, 2, 3, 4, 3, 2, 1, 1, 1, 1],
            [1, 2, 3, 4, 3, 2, 1, 1, 1, 1],
            [1, 1, 2, 3, 4, 3, 2, 1, 1, 1],
            [1, 1, 2, 3, 4, 3, 2, 1, 1, 1],
            [1, 1, 1, 2, 3, 4, 3, 2, 1, 1],
            [1, 1, 1, 2, 3, 4, 3, 2, 1, 1],
            [1, 1, 1, 1, 2, 3, 4, 3, 2, 1],
            [1, 1, 1, 1, 2, 3, 4, 3, 2, 1],
            [1, 1, 1, 1, 1, 2, 3, 4, 3, 2],
            [1, 1, 1, 1, 1, 2, 3, 4, 3, 2],
            [2, 1, 1, 1, 1, 1, 2, 3, 4, 3],
            [2, 1, 1, 1, 1, 1, 2, 3, 4, 3],
            [3, 2, 1, 1, 1, 1, 1, 2, 3, 4],
            [3, 2, 1, 1, 1, 1, 1, 2, 3, 4],
            [4, 3, 2, 1, 1, 1, 1, 1, 2, 3],
            [4, 3, 2, 1, 1, 1, 1, 1, 2, 3],
            [3, 4, 3, 2, 1, 1, 1, 1, 1, 2],
            [3, 4, 3, 2, 1, 1, 1, 1, 1, 2],
            [2, 3, 4, 3, 2, 1, 1, 1, 1, 1],
            [2, 3, 4, 3, 2, 1, 1, 1, 1, 1],
        ]
        client_label_probabilities = np.array(client_label_probabilities).astype(
            np.float32
        )
        client_label_probabilities /= np.sum(client_label_probabilities, axis=1)[
            :, np.newaxis
        ]
        print(client_label_probabilities)

        indices_by_class = [[] for _ in range(num_classes)]

        for i in range(len(labels)):
            indices_by_class[labels[i]].append(i)

        for client_id in range(num_clients):
            for _ in range(data_num_per_client):
                label = np.random.choice(
                    num_classes, p=client_label_probabilities[client_id]
                )
                idx = np.random.choice(indices_by_class[label])
                X[client_id].append(images[idx])
                y[client_id].append(labels[idx])
        cluster_results = None

    for i in range(num_clients):
        X[i] = np.array(X[i])
        y[i] = np.array(y[i])
        print(f"Client {i}: {len(X[i])}")
    print(X[0].shape)
    print(y[0].shape)

    return X, y, cluster_results


def main():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    train_path = dir_path / "train"
    test_path = dir_path / "test"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path / "rawdata",
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=dir_path / "rawdata",
        train=False,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    print(type(dataset_image))
    print(dataset_image.shape)
    print(type(dataset_label))
    print(dataset_label.shape)

    X, y, statistic = separate_data(
        dataset_image,
        dataset_label,
        num_clients,
        num_classes,
        class_per_client,
        partission_method,
    )

    # Save train/test data
    for i in range(num_clients):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], test_size=0.2, random_state=42
        )

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        train_data = (X_train, y_train)
        test_data = (X_test, y_test)

        with open(train_path / f"train{i}.pkl", "wb") as f:
            pickle.dump(train_data, f)
        with open(test_path / f"test{i}.pkl", "wb") as f:
            pickle.dump(test_data, f)

    # Save statistic
    with open(dir_path / "statistic.json", "w") as f:
        json.dump(statistic, f, indent=4)


if __name__ == "__main__":
    main()
