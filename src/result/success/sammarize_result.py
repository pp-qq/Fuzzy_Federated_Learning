import glob
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    sammary_type = 1
    DIR = Path("")
    metric = "test_acc"
    metrics = [
        "test_w_macro_f1",
        "test_acc",
        "g_test_macro_f1",
        "g_test_acc",
    ]
    over_metric = 0.85

    if sammary_type == 0:
        # 複数回実行した最終結果の平均と標準偏差を出力
        parent_dir = DIR
        target_dirs = glob.glob(os.path.join(parent_dir, "*"))
        target_dirs = [Path(target_dir) for target_dir in target_dirs]

        for met in metrics:
            metric = met
            print(metric)
            for target_dir in target_dirs:
                metric_list = []
                sub_target_dirs = glob.glob(os.path.join(target_dir, "*"))
                for sub_target_dir in sub_target_dirs:
                    for i in range(20):
                        csv_path = sub_target_dir + f"/{i}.csv"
                        df = pd.read_csv(csv_path)
                        metric_list.append(df[metric].values[-1])
                print(target_dir)
                print(np.mean(metric_list))
                print(statistics.pstdev(metric_list))
                print()

            for target_dir in target_dirs:
                sub_target_dirs = glob.glob(os.path.join(target_dir, "*"))
                for i in range(20):
                    metric_list = []
                    for sub_target_dir in sub_target_dirs:
                        for j in range(20):
                            csv_path = sub_target_dir + f"/{j}.csv"
                            df = pd.read_csv(csv_path)
                            metric_list.append(df[metric].values[i])
                    if np.mean(metric_list) > over_metric:
                        print(target_dir, i)
                        print(np.mean(metric_list))
                        print(statistics.pstdev(metric_list))
                        print()
                        break

    elif sammary_type == 1:
        # 特定の1回の実行結果を出力
        target_dir = DIR
        csv_paths = glob.glob(os.path.join(target_dir, "*.csv"))
        csv_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        for met in metrics:
            metric = met
            metric_list = []
            for csv_path in csv_paths:
                df = pd.read_csv(csv_path)
                metric_list.append(df[metric].values[-1])
            print(target_dir)
            print(metric)
            print(np.mean(metric_list))
            print(statistics.pstdev(metric_list))

            for i in range(20):
                metric_list = []
                for j in range(20):
                    csv_path = str(target_dir) + f"/{j}.csv"
                    df = pd.read_csv(csv_path)
                    metric_list.append(df[metric].values[i])
                if np.mean(metric_list) > over_metric:
                    print(target_dir, i)
                    print(np.mean(metric_list))
                    print(statistics.pstdev(metric_list))
                    break

    else:
        # 各データ分布での実行結果の平均を出力
        parent_dir = DIR
        target_dirs = glob.glob(os.path.join(parent_dir, "*"))
        target_dirs = [Path(target_dir) for target_dir in target_dirs]
        print(target_dirs)

        for target_dir in target_dirs:
            print("\n", target_dir)
            # TARGET_DIR配下のディレクトリを取得
            alg_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
            alg_dirs.sort(key=lambda x: x.name.split("_")[0])
            # alg_dirs[1]を末尾に移動
            # alg_dirs.append(alg_dirs.pop(1))

            for alg_dir in alg_dirs:
                algorithm = alg_dir.name.split("_")[0]

                test_acc = []
                test_macro_f1 = []
                global_acc = []
                global_macro_f1 = []
                for i in range(20):
                    csv_path = os.path.join(alg_dir, f"{i}.csv")
                    df = pd.read_csv(csv_path)
                    test_acc.append(df["test_acc"].values[-1])
                    test_macro_f1.append(df["test_macro_f1"].values[-1])
                    global_acc.append(df["g_test_acc"].values[-1])
                    global_macro_f1.append(df["g_test_macro_f1"].values[-1])

                print(f"{algorithm}     test_acc:       {np.mean(test_acc)}")
                print(f"{algorithm}     test_macro_f1:  {np.mean(test_macro_f1)}")
                print(f"{algorithm}     global_acc:     {np.mean(global_acc)}")
                print(f"{algorithm}     global_macro_f1:{np.mean(global_macro_f1)}")


if __name__ == "__main__":
    main()
