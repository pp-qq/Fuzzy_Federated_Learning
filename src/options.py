import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Select Federated Learning algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="fedavg",
        help="name of federated learning algorithm. fedavg, fedkm, fedfcm",
    )

    # Federated Learning arguments
    parser.add_argument("--epochs", type=int, default=20, help="rounds of training")
    parser.add_argument(
        "--num_clients", type=int, default=2, help="number of clients: K"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="the number of local epochs: E"
    )
    parser.add_argument(
        "--pre_train_epochs",
        type=int,
        default=0,
        help="the number of local pre-epochs: E",
    )
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="SGD momentum (default: 0.5)",
    )

    # Clusterd Federated Learning arguments
    parser.add_argument(
        "--num_clusters", type=int, default=5, help="number of clusters: C"
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="lenet", help="model name")

    # Other arguments
    parser.add_argument(
        "--dataset", type=str, default="cifar10_random", help="name of dataset"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    args = parser.parse_args()
    return args
