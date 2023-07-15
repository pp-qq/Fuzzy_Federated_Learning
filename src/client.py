import copy
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import LeNet_MNIST


class Client(object):
    def __init__(self, args, id):
        self.args = args
        self.id = id
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.momentum = args.momentum
        self.seed = args.seed

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_time = []

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # if torch.backends.mps.is_available() and self.device != 'cuda:0':
        #     self.device = torch.device('mps')

        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.load_model()
        self.load_train_data()
        self.load_test_data()

    def save_model(self):
        pass

    def load_model(self):
        if self.model_name == "lenet" and self.dataset == "mnist":
            self.model = LeNet_MNIST()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )

    def send_model_weights(self):
        return self.model.state_dict()

    def load_model_weights(self, model_dict):
        self.model.load_state_dict(model_dict)

    def load_train_data(self):
        if self.dataset == "mnist":
            train_data_pkl = os.path.join(
                "data", "MNIST", "train", f"train{self.id}.pkl"
            )

        train_data = pickle.load(open(train_data_pkl, "rb"))
        train_x, train_y = train_data
        train_x = torch.from_numpy(train_x).float()
        train_y = torch.from_numpy(train_y).long()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=1,
            shuffle=True,
        )
        self.train_loader = train_loader

    def load_test_data(self):
        if self.dataset == "mnist":
            test_data_pkl = os.path.join(
                "data", "mnist", "test", f"test{self.id}.pkl"
            )

        test_data = pickle.load(open(test_data_pkl, "rb"))
        test_x, test_y = test_data
        test_x = torch.from_numpy(test_x).float()
        test_y = torch.from_numpy(test_y).long()
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_x, test_y),
            batch_size=1,
            shuffle=True,
        )
        self.test_loader = test_loader

    def train(self):
        device = self.device
        self.model.train()
        for epoch in range(self.local_ep):
            print(f"Client {self.id} training epoch {epoch}")
            start_time = time.time()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            end_time = time.time()
            print(
                f"Client {self.id} training epoch {epoch} time {end_time - start_time}"
            )

    def test(self):
        device = self.device
        self.model.eval()
        test_loss = 0
        acc = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                acc += pred.eq(target.view_as(pred)).sum().item()
        print(
            f"Client {self.id} test loss {test_loss / len(self.test_loader)}"
        )
        print(f"Client {self.id} test acc {acc / len(self.test_loader)}")


# class ClientKM(object):
#     def __init__(self, args, id, train_loader, test_loader):
#         self.args = args
#         self.id = id
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.model_name = args.model_name
#         self.dataset = args.dataset
#         self.local_ep = args.local_ep
#         self.local_bs = args.local_bs
#         self.lr = args.lr
#         self.momentum = args.momentum
#         self.seed = args.seed

#         self.model = None
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
#         self.train_loss = []
#         self.test_loss = []
#         self.train_acc = []
#         self.test_acc = []
#         self.train_time = []

#     def save_model(self):
#         pass

#     def load_model(self):
#         pass

#     def send_parameters(self):
#         pass

#     def load_train_data(self):
#         pass

#     def load_test_data(self):
#         pass

#     def train(self):
#         trainloader = self.load_train_data()
#         self.model.train()
#         for epoch in range(self.local_ep):
#             # train
#             pass

#         # test
#         testloader = self.load_test_data()
#         self.model.eval()
#         pass

#         # save model
#         pass
