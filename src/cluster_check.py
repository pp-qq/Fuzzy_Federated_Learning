import pickle
import random
import time
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import torch
from client import Client
from nets import LeNet_Cifar10, LeNet_MNIST
from options import args_parser
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor


def plot_features(args, labels, w_clients_comp):
    if args.dataset == "mnist_multi":
        grand_truth = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    elif args.dataset == "mnist_prob":
        grand_truth = [
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            8,
            8,
            9,
            9,
        ]

    fig = plt.figure()
    plot1 = fig.add_subplot(1, 2, 1)
    plot1.scatter(
        w_clients_comp[:, 0],
        w_clients_comp[:, 1],
        c=grand_truth,
        cmap="rainbow",
        alpha=0.5,
    )
    plot1.set_title("Ground Truth")
    plot1.set_xlabel("PC1")
    plot1.set_ylabel("PC2")

    plot2 = plt.figure().add_subplot(1, 2, 2)
    plot2.scatter(
        w_clients_comp[:, 0],
        w_clients_comp[:, 1],
        c=labels,
        cmap="rainbow",
        alpha=0.5,
    )
    plot2.set_title("Clustering Result")
    plot2.set_xlabel("PC1")
    plot2.set_ylabel("PC2")

    plt.show()


def weights_flatten(w_clients):
    w_clients_flatten = []
    for i in range(len(w_clients)):
        w_client_flatten = []
        for key in w_clients[i].keys():
            w_client_flatten.append(
                # valがtensorなのでnumpyに変換し，flattenする
                w_clients[i][key]
                .to("cpu")
                .detach()
                .numpy()
                .flatten()
            )
        w_client_flatten = np.concatenate(w_client_flatten)
        w_clients_flatten.append(w_client_flatten)
    w_clients_flatten = np.array(w_clients_flatten)
    return w_clients_flatten


def main(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clients = []
    for i in range(args.num_clients):
        clients.append(Client(id=i, args=args))

    for i in range(args.num_clients):
        for j in range(args.epochs):
            clients[i].train()
            clients[i].test()
            print(f"Client {i} epoch {j} finished")
            print(f"Client {i} accuracy: {clients[i].test_acc[-1]}")

    # クラスタリングテスト
    w_list = [client.model.state_dict() for client in clients]
    w_flatten_list = weights_flatten(w_list)

    # 中間層のパラメータを取得
    w_hidden_list = []
    for i in range(len(w_list)):
        w_hidden_list.append(w_list[i]["fc1.weight"])
    w_hidden_list = np.array(w_hidden_list)
    w_hidden_list = w_hidden_list.reshape(w_hidden_list.shape[0], -1)
    w_hidden_list = sklearn.preprocessing.normalize(w_hidden_list, norm="l2", axis=0)

    w_flatten_list = sklearn.preprocessing.normalize(w_flatten_list, norm="l2", axis=0)

    # クラスタリング
    w_clients_comp = PCA(n_components=2, random_state=args.seed).fit_transform(
        w_flatten_list
    )
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(
        w_clients_comp
    )

    w_hidden_comp = PCA(n_components=2, random_state=args.seed).fit_transform(
        w_hidden_list
    )
    kmeans_hidden = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(
        w_hidden_comp
    )

    labels = kmeans.labels_
    print(labels)

    labels_hidden = kmeans_hidden.labels_
    print(labels_hidden)

    # クラスタリング結果を図示
    # plot_features(args, labels, w_clients_comp)


if __name__ == "__main__":
    args = args_parser()
    main(args)
