import glob
import os

import pandas as pd


def main():
    data_dir_1 = "/Users/tk/Documents/M2/codes/Fuzzy_Federated_Learning/src/result/success/data_for_honbun/mnist_random/fedavg"
    data_dir_2 = "/Users/tk/Documents/M2/codes/Fuzzy_Federated_Learning/src/result/success/data_for_honbun/mnist_random/fedfcm"
    metric = "test_acc"
    p_threshold = 0.05

    data_1 = []
    data_2 = []

    data_sub_dirs_1 = glob.glob(os.path.join(data_dir_1, "*"))
    data_sub_dirs_2 = glob.glob(os.path.join(data_dir_2, "*"))

    for data_sub_dir in data_sub_dirs_1:
        for i in range(20):
            csv_path = data_sub_dir + f"/{i}.csv"
            df = pd.read_csv(csv_path)
            data_1.append(df[metric].values[-1])

    for data_sub_dir in data_sub_dirs_2:
        for i in range(20):
            csv_path = data_sub_dir + f"/{i}.csv"
            df = pd.read_csv(csv_path)
            data_2.append(df[metric].values[-1])

    # 正規分布に従うかどうかを調べるため，ヒストグラムを描画
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.hist(data_1, bins=100)
    plt.title("FedAvg")
    plt.subplot(2, 1, 2)
    plt.hist(data_2, bins=100)
    plt.title("FedFCM")
    plt.tight_layout()
    plt.show()

    # F検定
    from scipy import stats

    result_f_test = stats.f_oneway(data_1, data_2)

    if result_f_test.pvalue < p_threshold:
        print("有意差あり")
    else:
        print("有意差なし")

    # t検定
    result_t_test = stats.ttest_ind(data_1, data_2, equal_var=False)
    print(result_t_test)


if __name__ == "__main__":
    main()
