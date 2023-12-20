import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_columns(column_name, dirs, target_dir, is_legend=True):
    plt.figure(figsize=(30, 10))
    for i, alg_dir in enumerate(dirs):
        plt.subplot(1, 3, i + 1)
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        if column_name == "test_macro_f1":
            plt.ylabel("Test Macro F1")
        elif column_name == "test_acc":
            plt.ylabel("Test Accuracy")
        elif column_name == "test_loss":
            plt.ylabel("Test Loss")
        elif column_name == "g_test_acc":
            plt.ylabel("Global Data Test Accuracy")
        elif column_name == "g_test_macro_f1":
            plt.ylabel("Global Data Test Macro F1")

        csv_paths = glob.glob(os.path.join(alg_dir, "*.csv"))
        # 0.csv, 1.csv, 2.csv, ..., 9.csv, 10.csv, 11.csv, ..., 19.csvに並び替える
        csv_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        # csvのカラムはtrain_loss,test_loss,train_acc,test_acc,test_macro_f1
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            plt.plot(
                df[column_name].values,
                label=csv_path.split("/")[-1].split(".")[0],
            )

        plt.title(f"{alg_dir.name.split('_')[0]}")
        if is_legend:
            plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"./{str(target_dir)}/{column_name}.png")


def main():
    parent_dir = Path("./mnist_weight_1")
    target_dirs = glob.glob(os.path.join(parent_dir, "*"))
    target_dirs = [Path(target_dir) for target_dir in target_dirs]
    print(target_dirs)

    for target_dir in target_dirs:
        # TARGET_DIR配下のディレクトリを取得
        alg_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
        alg_dirs.sort(key=lambda x: x.name.split("_")[0])
        print(alg_dirs)
        # alg_dirs[1]を末尾に移動
        alg_dirs.append(alg_dirs.pop(1))

        plot_columns("test_macro_f1", alg_dirs, target_dir)
        plot_columns("test_acc", alg_dirs, target_dir)
        plot_columns("test_loss", alg_dirs, target_dir)
        plot_columns("g_test_acc", alg_dirs, target_dir, is_legend=False)
        plot_columns("g_test_macro_f1", alg_dirs, target_dir, is_legend=False)


if __name__ == "__main__":
    main()
