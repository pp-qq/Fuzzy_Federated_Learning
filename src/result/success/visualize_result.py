import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # TARGET_DIR = Path("./mnist_random_client20_epoch20/")
    # TARGET_DIR = Path("./mnist_prob_client20_epoch20/")
    # TARGET_DIR = Path("./mnist_multi_client20_epoch20/")
    # TARGET_DIR = Path("./mnist_random_2_client20_epoch20/")
    # TARGET_DIR = Path("./mnist_prob_2_client20_epoch20/")
    TARGET_DIR = Path("./mnist_multi_4/")

    # TARGET_DIR配下のディレクトリを取得
    alg_dirs = [p for p in TARGET_DIR.iterdir() if p.is_dir()]
    alg_dirs.sort(key=lambda x: x.name.split("_")[0])
    print(alg_dirs)
    # alg_dirs[1]を末尾に移動
    # alg_dirs.append(alg_dirs.pop(1))

    plt.figure(figsize=(30, 10))
    for i, alg_dir in enumerate(alg_dirs):
        plt.subplot(1, 3, i + 1)
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")

        csv_paths = glob.glob(os.path.join(alg_dir, "*.csv"))
        # 0.csv, 1.csv, 2.csv, ..., 9.csv, 10.csv, 11.csv, ..., 19.csvに並び替える
        csv_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        # csvのカラムはtrain_loss,test_loss,train_acc,test_acc,test_macro_f1
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            plt.plot(
                df["test_acc"].values,
                label=csv_path.split("/")[-1].split(".")[0],
            )

        plt.title(f"{alg_dir.name.split('_')[0]}")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"./{TARGET_DIR.name}/test_acc.png")

    plt.figure(figsize=(30, 10))
    for i, alg_dir in enumerate(alg_dirs):
        plt.subplot(1, 3, i + 1)
        plt.ylim(0, 5)
        plt.xlabel("Epoch")
        plt.ylabel("Test Loss")

        csv_paths = glob.glob(os.path.join(alg_dir, "*.csv"))
        # 0.csv, 1.csv, 2.csv, ..., 9.csv, 10.csv, 11.csv, ..., 19.csvに並び替える
        csv_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        # csvのカラムはtrain_loss,test_loss,train_acc,test_acc,test_macro_f1
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            plt.plot(
                df["test_loss"].values,
                label=csv_path.split("/")[-1].split(".")[0],
            )

        plt.title(f"{alg_dir.name.split('_')[0]}")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"./{TARGET_DIR.name}/test_loss.png")

    plt.figure(figsize=(30, 10))
    for i, alg_dir in enumerate(alg_dirs):
        plt.subplot(1, 3, i + 1)
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Test Macro F1")

        csv_paths = glob.glob(os.path.join(alg_dir, "*.csv"))
        # 0.csv, 1.csv, 2.csv, ..., 9.csv, 10.csv, 11.csv, ..., 19.csvに並び替える
        csv_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        # csvのカラムはtrain_loss,test_loss,train_acc,test_acc,test_macro_f1
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            plt.plot(
                df["test_macro_f1"].values,
                label=csv_path.split("/")[-1].split(".")[0],
            )

        plt.title(f"{alg_dir.name.split('_')[0]}")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"./{TARGET_DIR.name}/test_macro_f1.png")


if __name__ == "__main__":
    main()
