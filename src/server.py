import copy
import json
import os
import pickle
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
        self.pre_train_epochs = args.pre_train_epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.model = None
        self.clients = []
        self.uploaded_weights = []

        self.start_time = time.time()  # インスタンス生成時刻

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

    def save_stuts(self):
        """self.clientsのself.train_loss, self.test_loss, self.train_acc, self.test_acc, self.test_macro_f1を保存する"""

        # './result/FedAVG_<yyyymmddss>'のディレクトリを作成
        result_dir = os.path.join(
            f'./result/FedAVG_{self.dataset}{time.strftime("%Y-%m-%d-%H-%M", time.localtime(self.start_time))}'
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, client in enumerate(self.clients):
            # train_loss, test_loss, train_acc, test_acc, test_macro_f1をカラムとするcsvファイルを作成
            # 1行目にはカラム名を記載
            # 2行目以降には各エポックの値を記載
            # clientのidをファイル名とする
            with open(os.path.join(result_dir, f"{client.id}.csv"), "w") as f:
                f.write(
                    "train_loss,test_loss,train_acc,test_acc,test_macro_f1,test_w_macro_f1,g_test_acc,g_test_loss,g_test_macro_f1\n"
                )
                for j in range(self.epochs + self.pre_train_epochs):
                    f.write(
                        f"{client.train_loss[j]},{client.test_loss[j]},{client.train_acc[j]},{client.test_acc[j]},{client.test_macro_f1[j]},{client.test_w_macro_f1[j]},{client.g_test_acc[j]},{client.g_test_loss[j]},{client.g_test_macro_f1[j]}\n"
                    )

        # argsをconfig.jsonファイルに保存
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.num_clients):
            self.clients[i].pre_train()

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # aggregate
            global_dict = self.aggregate_parameters(
                client_models={client.id: client.model for client in self.clients}
            )
            # update global model
            for client in self.clients:
                client.load_model_weights(model_dict=global_dict)

            # test
            print()
            for client in self.clients:
                test_loss, test_acc = client.test()
                print(
                    f"Client {client.id} test loss: {test_loss:.4f}, test acc: {test_acc:.4f}"
                )

            # global test
            print()
            for client in self.clients:
                g_test_loss, g_test_acc, g_test_macro_f1 = client.global_test()
                print(
                    f"Client {client.id} global test loss: {g_test_loss:.4f}, global test acc: {g_test_acc:.4f}, global test macro f1: {g_test_macro_f1:.4f}"
                )
                print()

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()

        print("Total Time cost:", time.time() - self.start_time)
        self.save_stuts()


class ServerFedKM(object):
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.num_clients = args.num_clients
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.pre_train_epochs = args.pre_train_epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.n_comp = args.n_comp
        self.cluster_based_on = args.cluster_based_on

        self.num_clusters = args.num_clusters

        self.model = None
        self.clients = []
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.model_clusters = [None for _ in range(self.num_clusters)]

        self.start_time = time.time()  # インスタンス生成時刻

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

    def save_stuts(self):
        # self.clientsのself.train_loss, self.test_loss, self.train_acc, self.test_acc, self.test_macro_f1を保存する

        # './result/FedAVG_<yyyymmddss>'のディレクトリを作成
        result_dir = os.path.join(
            f'./result/FedKM_{self.dataset}{time.strftime("%Y-%m-%d-%H-%M", time.localtime(self.start_time))}'
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, client in enumerate(self.clients):
            # train_loss, test_loss, train_acc, test_acc, test_macro_f1をカラムとするcsvファイルを作成
            # 1行目にはカラム名を記載
            # 2行目以降には各エポックの値を記載
            # clientのidをファイル名とする
            with open(os.path.join(result_dir, f"{client.id}.csv"), "w") as f:
                f.write(
                    "train_loss,test_loss,train_acc,test_acc,test_macro_f1,test_w_macro_f1,g_test_acc,g_test_loss,g_test_macro_f1\n"
                )
                for j in range(self.epochs + self.pre_train_epochs):
                    f.write(
                        f"{client.train_loss[j]},{client.test_loss[j]},{client.train_acc[j]},{client.test_acc[j]},{client.test_macro_f1[j]},{client.test_w_macro_f1[j]},{client.g_test_acc[j]},{client.g_test_loss[j]},{client.g_test_macro_f1[j]}\n"
                    )

        # argsをconfig.jsonファイルに保存
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.num_clients):
            self.clients[i].pre_train()

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # クラスタリング
            w_list = [client.model.state_dict() for client in self.clients]
            if self.cluster_based_on == "weight":
                w_clients_flatten = weights_flatten(w_list)
                w_clients_flatten = sklearn.preprocessing.normalize(
                    w_clients_flatten, norm="l2", axis=0
                )

                # k-means
                w_clients_comp = PCA(
                    n_components=self.n_comp, random_state=0, svd_solver="randomized"
                ).fit_transform(w_clients_flatten)
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
                    w_clients_comp
                )
                for i in range(len(kmeans.labels_)):
                    self.clusters[kmeans.labels_[i]].append(self.clients[i])

            elif self.cluster_based_on == "hidden":
                w_hidden_list = []
                for i in range(len(w_list)):
                    w_hidden_list.append(w_list[i]["fc1.weight"])
                w_hidden_list = np.array(w_hidden_list)
                w_hidden_list = w_hidden_list.reshape(w_hidden_list.shape[0], -1)
                w_hidden_list = sklearn.preprocessing.normalize(
                    w_hidden_list, norm="l2", axis=0
                )

                # k-means
                w_hidden_comp = PCA(
                    n_components=self.n_comp, random_state=0, svd_solver="randomized"
                ).fit_transform(w_hidden_list)
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
                    w_hidden_comp
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
                            client.id: client.model for client in self.clusters[i]
                        }
                    )

            # クラスタごとにパラメータを更新
            for i in range(self.num_clusters):
                for client in self.clusters[i]:
                    client.load_model_weights(model_dict=self.model_clusters[i])

            # test
            print()
            for client in self.clients:
                test_loss, test_acc = client.test()
                print(
                    f"Client {client.id} test loss: {test_loss:.4f}, test acc: {test_acc:.4f}"
                )

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()

            # global test
            print()
            for client in self.clients:
                g_test_loss, g_test_acc, g_test_macro_f1 = client.global_test()
                print(
                    f"Client {client.id} global test loss: {g_test_loss:.4f}, global test acc: {g_test_acc:.4f}, global test macro f1: {g_test_macro_f1:.4f}"
                )
                print()

        print("Total Time cost:", time.time() - self.start_time)
        self.save_stuts()


class ServerFedFCM(object):
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.num_clients = args.num_clients
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.pre_train_epochs = args.pre_train_epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.num_clusters = args.num_clusters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.seed = args.seed

        self.num_clusters = args.num_clusters
        self.n_comp = args.n_comp
        self.degree_of_fuzziness = args.degree_of_fuzziness
        self.cluster_based_on = args.cluster_based_on
        self.fuzzy_ratio = args.fuzzy_ratio
        self.fuzzy_agg_type = args.fuzzy_agg_type
        self.fuzzy_add_global_model_epoch = args.fuzzy_add_global_model_epoch

        self.model = None
        self.clients = []
        self.fcm_labels = []
        self.model_clusters = [None for _ in range(self.num_clusters)]

        self.past_models = []
        self.past_labels = []

        self.start_time = time.time()  # インスタンス生成時刻

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            self.clients.append(clientObj(args=self.args, id=i))

    def save_global_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_global_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def aggregate_parameters(self, client_models, fcm_labels):
        # クラスターモデルの集約
        # 具体的には((i番目のクライアントがAのクラスタに属する確率)/(Aのクラスタに属する確率の和))*i番目のクライアントのパラメータを足し合わせる
        for i in range(self.num_clusters):
            sum_prob = 0
            for j in range(self.num_clients):
                sum_prob += fcm_labels[j][i]
            cluster_model_dict = {}
            for j in range(self.num_clients):
                # client_models[j]はj番目のクライアントのモデルのOrderedDict
                # cluster_model_dictにはj番目のクライアントのモデルのOrderedDictの各valueにfcm_labels[j][i]をかけたものを足し合わせたものを格納する
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

        # クライアントモデルの更新
        # 具体的には，各クラスタに属する確率*各クラスタのパラメータを足し合わせたモデルにfuzzy-ratio，
        # グローバルモデルに1-fuzzy_ratioで重み付けしたものを足し合わせたモデルをクライアントにロードする
        global_dict = {}
        for k in client_models[0].keys():
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))],
                0,
            ).sum(0)
        for i in range(self.num_clients):
            tmp_model_dict = {}
            for j in range(self.num_clusters):
                for k in self.model_clusters[j].keys():
                    if k in tmp_model_dict:
                        tmp_model_dict[k] += (
                            self.model_clusters[j][k] * fcm_labels[i][j]
                        )
                    else:
                        tmp_model_dict[k] = self.model_clusters[j][k] * fcm_labels[i][j]
            new_model_dict = OrderedDict(
                {
                    k: tmp_model_dict[k] * self.fuzzy_ratio
                    + global_dict[k] * (1 - self.fuzzy_ratio)
                    for k in tmp_model_dict.keys()
                }
            )
            self.clients[i].load_model_weights(model_dict=new_model_dict)

    def aggregate_parameters2(self, client_models, fcm_labels):
        # グローバルテストの精度を実際の環境で測定することはできないため，使えない手法

        # クラスターモデルの集約
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

        # クライアントモデルの更新
        # クラスターモデル * (1 - fuzzy_ratio) + (各クラスタに属する確率*各クラスタのパラメータを足し合わせたモデル) * fuzzy-ratio，
        for i in range(self.num_clients):
            tmp_model_dict = {}
            tmp_cluster_model_dict = copy.deepcopy(
                self.model_clusters[np.argmax(fcm_labels[i])]
            )
            for j in range(self.num_clusters):
                for k in self.model_clusters[j].keys():
                    if k in tmp_model_dict:
                        tmp_model_dict[k] += (
                            self.model_clusters[j][k] * fcm_labels[i][j]
                        )
                    else:
                        tmp_model_dict[k] = self.model_clusters[j][k] * fcm_labels[i][j]
            new_model_dict = OrderedDict(
                {
                    k: tmp_model_dict[k] * self.fuzzy_ratio
                    + tmp_cluster_model_dict[k] * (1 - self.fuzzy_ratio)
                    for k in tmp_model_dict.keys()
                }
            )
            self.clients[i].load_model_weights(model_dict=new_model_dict)

    def aggregate_parameters3(self, client_models, fcm_labels, epoch):
        # クラスターモデルの集約
        # 具体的には((i番目のクライアントがAのクラスタに属する確率)/(Aのクラスタに属する確率の和))*i番目のクライアントのパラメータを足し合わせる
        for i in range(self.num_clusters):
            sum_prob = 0
            for j in range(self.num_clients):
                sum_prob += fcm_labels[j][i]
            cluster_model_dict = {}
            for j in range(self.num_clients):
                # client_models[j]はj番目のクライアントのモデルのOrderedDict
                # cluster_model_dictにはj番目のクライアントのモデルのOrderedDictの各valueにfcm_labels[j][i]をかけたものを足し合わせたものを格納する
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

        # クライアントモデルの更新
        # 具体的には，各クラスタに属する確率*各クラスタのパラメータを足し合わせたモデルにfuzzy-ratio，
        # グローバルモデルに1-fuzzy_ratioで重み付けしたものを足し合わせたモデルをクライアントにロードする
        if epoch < self.fuzzy_add_global_model_epoch:
            fuzzy_ratio = self.fuzzy_ratio
        else:
            fuzzy_ratio = 1.0
        global_dict = {}
        for k in client_models[0].keys():
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))],
                0,
            ).sum(0)
        for i in range(self.num_clients):
            tmp_model_dict = {}
            for j in range(self.num_clusters):
                for k in self.model_clusters[j].keys():
                    if k in tmp_model_dict:
                        tmp_model_dict[k] += (
                            self.model_clusters[j][k] * fcm_labels[i][j]
                        )
                    else:
                        tmp_model_dict[k] = self.model_clusters[j][k] * fcm_labels[i][j]
            new_model_dict = OrderedDict(
                {
                    k: tmp_model_dict[k] * fuzzy_ratio
                    + global_dict[k] * (1 - fuzzy_ratio)
                    for k in tmp_model_dict.keys()
                }
            )
            self.clients[i].load_model_weights(model_dict=new_model_dict)

    def aggregate_parameters4(self, client_models, fcm_labels):
        # グローバルテストの精度を実際の環境で測定することはできないため，使えない手法

        # クラスターモデルの集約
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

        # クライアントモデルの更新
        # クラスターモデル * (1 - fuzzy_ratio) + (各クラスタに属する確率*各クラスタのパラメータを足し合わせたモデル) * fuzzy-ratio
        for i in range(self.num_clients):
            tmp_model_dict = {}
            tmp_cluster_model_dict = copy.deepcopy(
                self.model_clusters[np.argmax(fcm_labels[i])]
            )
            for j in range(self.num_clusters):
                for k in self.model_clusters[j].keys():
                    if k in tmp_model_dict:
                        tmp_model_dict[k] += (
                            self.model_clusters[j][k] * fcm_labels[i][j]
                        )
                    else:
                        tmp_model_dict[k] = self.model_clusters[j][k] * fcm_labels[i][j]
            new_model_dict = OrderedDict(
                {
                    k: tmp_model_dict[k] * self.fuzzy_ratio
                    + tmp_cluster_model_dict[k] * (1 - self.fuzzy_ratio)
                    for k in tmp_model_dict.keys()
                }
            )
            self.clients[i].load_model_weights(model_dict=new_model_dict)

    def save_stuts(self):
        # self.clientsのself.train_loss, self.test_loss, self.train_acc, self.test_acc, self.test_macro_f1を保存する

        # './result/FedAVG_<yyyymmddss>'のディレクトリを作成
        result_dir = os.path.join(
            f'./result/FedFCM_{self.dataset}{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(self.start_time))}'
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, client in enumerate(self.clients):
            # train_loss, test_loss, train_acc, test_acc, test_macro_f1, g_test_acc, g_test_loss, g_test_macro_f1をカラムとするcsvファイルを作成
            # 1行目にはカラム名を記載
            # 2行目以降には各エポックの値を記載
            # clientのidをファイル名とする
            with open(os.path.join(result_dir, f"{client.id}.csv"), "w") as f:
                f.write(
                    "train_loss,test_loss,train_acc,test_acc,test_macro_f1,test_w_macro_f1,g_test_acc,g_test_loss,g_test_macro_f1\n"
                )
                for j in range(self.epochs + self.pre_train_epochs):
                    f.write(
                        f"{client.train_loss[j]},{client.test_loss[j]},{client.train_acc[j]},{client.test_acc[j]},{client.test_macro_f1[j]},{client.test_w_macro_f1[j]},{client.g_test_acc[j]},{client.g_test_loss[j]},{client.g_test_macro_f1[j]}\n"
                    )

        # argsをconfig.jsonファイルに保存
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

        # past_modelsをpickleファイルに保存
        with open(os.path.join(result_dir, "past_models_and_labels.pkl"), "wb") as f:
            pickle.dump(self.past_models, f)
            pickle.dump(self.past_labels, f)

    def train(self):
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(self.num_clients):
            self.clients[i].pre_train()

        for i in range(self.epochs):
            start_time = time.time()
            print("Epoch: {}/{}".format(i + 1, self.epochs))

            for client in self.clients:
                client.train()

            # クラスタリング
            w_list = [client.send_model_weights() for client in self.clients]
            if self.cluster_based_on == "weight":
                w_clients_flatten = weights_flatten(w_list)
                self.past_models.append(w_clients_flatten)
                w_clients_flatten = sklearn.preprocessing.normalize(
                    w_clients_flatten, norm="l2", axis=0
                )

                # Fuzzy C-Means
                w_clients_comp = PCA(
                    n_components=self.n_comp, random_state=0, svd_solver="randomized"
                ).fit_transform(w_clients_flatten)
                fcm = FCM(
                    n_clusters=self.num_clusters,
                    random_state=0,
                    m=self.degree_of_fuzziness,
                )
                fcm.fit(w_clients_comp)
                fcm_labels = fcm.u
                self.past_labels.append(fcm_labels)

            elif self.cluster_based_on == "hidden":
                w_hidden_list = []
                for i in range(len(w_list)):
                    w_hidden_list.append(w_list[i]["fc1.weight"])
                w_hidden_list = np.array(w_hidden_list)
                w_hidden_list = w_hidden_list.reshape(w_hidden_list.shape[0], -1)
                self.past_models.append(w_hidden_list)
                w_hidden_list = sklearn.preprocessing.normalize(
                    w_hidden_list, norm="l2", axis=0
                )

                # Fuzzy C-Means
                w_hidden_comp = PCA(
                    n_components=self.n_comp, random_state=0, svd_solver="randomized"
                ).fit_transform(w_hidden_list)
                fcm = FCM(
                    n_clusters=self.num_clusters,
                    random_state=0,
                    m=self.degree_of_fuzziness,
                )
                fcm.fit(w_hidden_comp)
                fcm_labels = fcm.u
                self.past_labels.append(fcm_labels)

            # aggregate
            if self.fuzzy_agg_type == 1:
                self.aggregate_parameters(w_list, fcm_labels)
            elif self.fuzzy_agg_type == 2:
                self.aggregate_parameters2(w_list, fcm_labels)
            elif self.fuzzy_agg_type == 3:
                self.aggregate_parameters3(w_list, fcm_labels, i)
            elif self.fuzzy_agg_type == 4:
                self.aggregate_parameters4(w_list, fcm_labels)

            # test
            print()
            for client in self.clients:
                test_loss, test_acc = client.test()
                print(
                    f"Client {client.id} test loss: {test_loss:.4f}, test acc: {test_acc:.4f}"
                )

            end_time = time.time()
            print("Time cost: {}".format(end_time - start_time))
            print()

            # global test
            print()
            for client in self.clients:
                g_test_loss, g_test_acc, g_test_macro_f1 = client.global_test()
                print(
                    f"Client {client.id} global test loss: {g_test_loss:.4f}, global test acc: {g_test_acc:.4f}, global test macro f1: {g_test_macro_f1:.4f}"
                )
                print()

        print("Total Time cost:", time.time() - self.start_time)
        self.save_stuts()
