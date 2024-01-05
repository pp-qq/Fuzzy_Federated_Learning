import json
import os
import pickle

import fcmeans as FCM
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA


def main():
    DIR = "/Users/tk/Documents/M2/codes/Fuzzy_Federated_Learning/src/result/success/compare_comp_dimention_in_mnist_multi/fedfcm_n_comp_2/fedfcm_w_5.0/FedFCM_mnist_multi2023-12-22-11-56-19/"

    json_path = DIR + "config.json"
    with open(json_path, "r") as f:
        config = json.load(f)

    if not os.path.exists(DIR + "cluster_images/"):
        os.mkdir(DIR + "cluster_images/")

    with open(DIR + "past_models_and_labels.pkl", "rb") as f:
        past_models = pickle.load(f)
        past_labels = pickle.load(f)

    for i, i_epoch_models in enumerate(past_models):
        i_epoch_models = sklearn.preprocessing.normalize(i_epoch_models, norm="l2")
        i_epoch_models_comp = PCA(n_components=2).fit_transform(i_epoch_models)
        fcm = FCM.FCM(n_clusters=5, m=config["degree_of_fuzziness"], random_state=0)
        fcm.fit(i_epoch_models_comp)

        plt.figure(figsize=(10, 10))
        plt.scatter(
            i_epoch_models_comp[:, 0],
            i_epoch_models_comp[:, 1],
            c=fcm.u.argmax(axis=1),
        )
        plt.title(f"Epoch {i}")
        plt.savefig(DIR + f"cluster_images/epoch_{i}.png")


if __name__ == "__main__":
    main()
