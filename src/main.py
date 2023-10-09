from client import Client
from options import args_parser
from server import ServerFedAvg, ServerFedFCM, ServerFedKM


def main(args):
    if args.algorithm == "fedavg":
        server = ServerFedAvg(args)
    elif args.algorithm == "fedkm":
        server = ServerFedKM(args)
    elif args.algorithm == "fedfcm":
        server = ServerFedFCM(args)
    server.set_clients(Client)
    server.train()


if __name__ == "__main__":
    args = args_parser()
    main(args)
