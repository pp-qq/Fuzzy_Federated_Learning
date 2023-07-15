from client import Client
from options import args_parser
from server import ServerFedAvg, ServerFedFCM, ServerFedKM


def main(args):
    # server = ServerFedAvg(args)
    # server = ServerFedKM(args)
    server = ServerFedFCM(args)
    server.set_clients(Client)
    server.train()


if __name__ == "__main__":
    args = args_parser()
    main(args)
