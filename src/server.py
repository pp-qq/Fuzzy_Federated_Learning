import copy
import os
import random
import time
from collections import OrderedDict, defaultdict

import numpy as np
import sklearn
import torch
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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


class ServerFedAvg(object):
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.num_clients = args.num_clients
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.model = None
        self.clients = []
        self.uploaded_weights = []

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            self.clients.append(clientObj(args=self.args, id=i))

    def save_global_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_global_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def aggregate_parameters(self, client_models):
        client_models = list(client_models.values())
        global_dict = {}
        for k in client_models[0].state_dict().keys():
            global_dict[k] = torch.stack(
                [
                    client_models[i].state_dict()[k].float()
                    for i in range(len(client_models))
                ],
                0,
            ).mean(0)
        return global_dict

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # aggregate
            global_dict = self.aggregate_parameters(
                client_models={
                    client.id: client.model for client in self.clients
                }
            )
            # update global model
            for client in self.clients:
                client.load_model_weights(model_dict=global_dict)

            # test
            for client in self.clients:
                client.test()

            # save global model
            # self.save_global_model(model=global_dict, path=os.path.join(self.args.save_path, 'global_model.pth'))
            # print("Global model saved.")

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()


class ServerFedKM(object):
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.num_clients = args.num_clients
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.num_clusters = args.num_clusters

        self.model = None
        self.clients = []
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.model_clusters = [None for _ in range(self.num_clusters)]

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            self.clients.append(clientObj(args=self.args, id=i))

    def save_global_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_global_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def aggregate_parameters(self, client_models):
        client_models = list(client_models.values())
        cluster_dict = {}
        for k in client_models[0].state_dict().keys():
            cluster_dict[k] = torch.stack(
                [
                    client_models[i].state_dict()[k].float()
                    for i in range(len(client_models))
                ],
                0,
            ).mean(0)
        return cluster_dict

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # クラスタリング
            w_list = [client.model.state_dict() for client in self.clients]
            w_clients_flatten = weights_flatten(w_list)
            w_clients_flatten = sklearn.preprocessing.normalize(
                w_clients_flatten, norm="l2", axis=0
            )

            # k-means
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
                w_clients_flatten
            )
            w_clients_comp = PCA(
                n_components=2, random_state=0, svd_solver="randomized"
            ).fit_transform(w_clients_flatten)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
                w_clients_comp
            )
            for i in range(len(kmeans.labels_)):
                self.clusters[kmeans.labels_[i]].append(self.clients[i])

            # TODO: クラスタを再割り当てする
            # self.clustersを構成すれば良い

            # aggregate
            # クラスタごとにパラメータを集約
            for i in range(self.num_clusters):
                for client in self.clusters[i]:
                    self.model_clusters[i] = self.aggregate_parameters(
                        client_models={
                            client.id: client.model
                            for client in self.clusters[i]
                        }
                    )

            # クラスタごとにパラメータを更新
            for i in range(self.num_clusters):
                for client in self.clusters[i]:
                    client.load_model_weights(
                        model_dict=self.model_clusters[i]
                    )

            # test
            for client in self.clients:
                client.test()

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()


class ServerFedFCM(object):
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.num_clients = args.num_clients
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.num_clusters = args.num_clusters

        self.model = None
        self.clients = []
        self.fcm_labels = []
        self.model_clusters = [None for _ in range(self.num_clusters)]

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            self.clients.append(clientObj(args=self.args, id=i))

    def save_global_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_global_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def aggregate_parameters(self, client_models, fcm_labels):
        # TODO: FCMで，クライアントがそのクラスタに属する確率を踏まえて集約を行う
        # 具体的には((i番目のクライアントがAのクラスタに属する確率)/(Aのクラスタに属する確率の和))*i番目のクライアントのパラメータを足し合わせる

        for i in range(self.num_clusters):
            sum_prob = 0
            for j in range(self.num_clients):
                sum_prob += fcm_labels[j][i]
            cluster_model_dict = {}
            for j in range(self.num_clients):
                # client_models[j]はj番目のクライアントのモデルのOrderedDict
                # client_moded_dictにはj番目のクライアントのモデルのOrderedDictの各valueにfcm_labels[j][i]をかけたものを足し合わせたものを格納する
                for k in client_models[j].keys():
                    if k in cluster_model_dict:
                        cluster_model_dict[k] += (
                            client_models[j][k] * fcm_labels[j][i] / sum_prob
                        )
                    else:
                        cluster_model_dict[k] = (
                            client_models[j][k] * fcm_labels[j][i] / sum_prob
                        )
            self.model_clusters[i] = cluster_model_dict

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # クラスタリング
            w_list = [client.send_model_weights() for client in self.clients]
            w_clients_flatten = weights_flatten(w_list)
            w_clients_flatten = sklearn.preprocessing.normalize(
                w_clients_flatten, norm="l2", axis=0
            )

            # Fuzzy C-Means
            w_clients_comp = PCA(
                n_components=2, random_state=0, svd_solver="randomized"
            ).fit_transform(w_clients_flatten)
            fcm = FCM(n_clusters=self.num_clusters, random_state=0)
            fcm.fit(w_clients_comp)
            fcm_labels = fcm.u

            # aggregate
            self.aggregate_parameters(w_list, fcm_labels)

            # クライアントごとにパラメータを更新
            # TODO: FCMで，クライアントごとにパラメータを更新する
            # 具体的には，各クラスタに属する確率*各クラスタのパラメータを足し合わせる
            for i in range(self.num_clients):
                new_model_dict = {}
                for j in range(self.num_clusters):
                    for k in self.model_clusters[j].keys():
                        if k in new_model_dict:
                            new_model_dict[k] += (
                                self.model_clusters[j][k] * fcm_labels[i][j]
                            )
                        else:
                            new_model_dict[k] = (
                                self.model_clusters[j][k] * fcm_labels[i][j]
                            )
                self.clients[i].load_model_weights(model_dict=new_model_dict)

            # test
            for client in self.clients:
                client.test()

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()
