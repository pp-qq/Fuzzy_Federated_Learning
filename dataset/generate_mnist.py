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

num_clients = 20
num_classes = 10
num_clusters = 5
partission_method = "probabilistic"  # "multi-class", "probabilistic", "random"

# this parameter is used only when partission_method is "multi-class"
class_per_client = 3

dir_path = Path("MNIST/")
random.seed(1)


def separate_data(
    images,
    labels,
    num_clients,
    num_classes,
    class_per_client,
    partition_method,
):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]

    # Separate data by class
    if partition_method == "multi-class":
        cluster_results = {
            "clusters": [[] for _ in range(num_clusters)],
            "labels": [[] for _ in range(num_clusters)],
        }

        data_idxs_for_each_clusters = [[] for _ in range(num_clusters)]
        data_idxs_for_each_clients = [[] for _ in range(num_clients)]

        i = 0
        for cluster in range(num_clusters):
            # # クラスターごとのラベルをランダムに決める
            # cluster_results["labels"][cluster] = random.sample(
            #     range(num_classes), class_per_client
            # )

            # クラスターごとのラベルを順番に決める
            cluster_results["labels"][cluster] = [
                i % num_classes,
                (i + 1) % num_classes,
                (i + 2) % num_classes,
            ]
            i += 3
            i %= num_classes

        for client in range(num_clients):
            # # クライアントごとにクラスターをランダムに決める
            # cluster_idx = random.randint(0, num_clusters - 1)
            # cluster_results["clusters"][cluster_idx].append(client)

            # # クライアント2つを1つのクラスターに割り当てる
            # cluster_results["clusters"][client % num_clusters].append(client)

            # クライアントを順番にクラスターに割り当てる
            cluster_results["clusters"][client % num_clusters].append(client)

        for i in range(len(labels)):
            for cluster in range(num_clusters):
                if labels[i] in cluster_results["labels"][cluster]:
                    for client in cluster_results["clusters"][cluster]:
                        X[client].append(images[i])
                        y[client].append(labels[i])

        print("X", X[0][0].shape)
        for i in range(len(X)):
            print(len(X[i]))
        # return X, y, cluster_results

    # Separate data by cluster
    elif partition_method == "probabilistic":
        data_num_per_client = 3000
        # Sample Probabilities
        # client_label_probabilities = [
        #     [1, 4, 6, 10, 6, 4, 1, 1, 1, 1],
        #     [1, 4, 6, 10, 6, 4, 1, 1, 1, 1],
        #     [1, 1, 4, 6, 10, 6, 4, 1, 1, 1],
        #     [1, 1, 4, 6, 10, 6, 4, 1, 1, 1],
        #     [1, 1, 1, 4, 6, 10, 6, 4, 1, 1],
        #     [1, 1, 1, 4, 6, 10, 6, 4, 1, 1],
        #     [1, 1, 1, 1, 4, 6, 10, 6, 4, 1],
        #     [1, 1, 1, 1, 4, 6, 10, 6, 4, 1],
        #     [1, 1, 1, 1, 1, 4, 6, 10, 6, 4],
        #     [1, 1, 1, 1, 1, 4, 6, 10, 6, 4],
        #     [2, 1, 1, 1, 1, 1, 4, 6, 10, 6],
        #     [2, 1, 1, 1, 1, 1, 4, 6, 10, 6],
        #     [3, 2, 1, 1, 1, 1, 1, 4, 6, 10],
        #     [3, 2, 1, 1, 1, 1, 1, 4, 6, 10],
        #     [4, 3, 2, 1, 1, 1, 1, 1, 4, 6],
        #     [4, 3, 2, 1, 1, 1, 1, 1, 4, 6],
        #     [3, 4, 3, 2, 1, 1, 1, 1, 1, 4],
        #     [3, 4, 3, 2, 1, 1, 1, 1, 1, 4],
        #     [2, 3, 4, 3, 2, 1, 1, 1, 1, 1],
        #     [2, 3, 4, 3, 2, 1, 1, 1, 1, 1],
        # ]
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

        cluster_results = {
            "clusters": [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
                [16, 17],
                [18, 19],
            ],
        }

        client_label_probabilities = np.array(
            client_label_probabilities
        ).astype(np.float64)
        for i in range(len(client_label_probabilities)):
            sum = np.sum(client_label_probabilities[i])
            client_label_probabilities[i] = client_label_probabilities[i] / sum

        print(client_label_probabilities)

        indexes_per_label = [[] for _ in range(num_classes)]
        for i in range(len(labels)):
            indexes_per_label[labels[i]].append(i)

        for client in range(num_clients):
            for i in range(data_num_per_client):
                # randomly choose a label from client_label_probabilities
                label = np.random.choice(
                    num_classes, p=client_label_probabilities[client]
                )
                # randomly choose a data from indexes_per_label
                index = random.choice(indexes_per_label[label])
                X[client].append(images[index])
                y[client].append(labels[index])

    # Randomly assign data to clients
    elif partition_method == "random":
        # Randomly assign data to clients
        for i in range(num_clients):
            start = random.randint(0, num_clients - 1)
            X[i] = images[start::num_clients]
            y[i] = labels[start::num_clients]
        cluster_results = None

    # Convert to numpy array
    for i in range(num_clients):
        X[i] = np.array(X[i])
        y[i] = np.array(y[i])

    # return X, y, cluster_results
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

    # Get MNIST data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root=dir_path / "rawdata",
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.MNIST(
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
    print(dataset_image.shape)  # (70000, 28, 28)
    print(type(dataset_label))
    print(dataset_label.shape)  # (70000,)

    classes = trainset.classes

    # Partition data
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
