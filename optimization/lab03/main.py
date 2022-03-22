import pickle
import argparse
import os

from torch.utils.data import DataLoader
import torch.optim as optim

from utils.metrics import *
from models import *
from datasets import *
from client import *
from server import *


INDICES_PATH = "data/indices"
MNIST_PATH = "data/mnist"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_rounds',
        help='number of rounds;',
        type=int
    )
    parser.add_argument(
        '--run_fedavg',
        help='if selected fedavg is used otherwise parameter server is used',
        action='store_true'
    )
    parser.add_argument(
        '--n_local_epochs',
        help='number of local epochs; only used with FedAvg;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--lr',
        help='number of clients;',
        type=float,
        default=1e-2
    )
    parser.add_argument(
        '--local_rank',
        type=int
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=1234,
        required=False
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    dist.init_process_group(backend='gloo', init_method="env://")

    rank = dist.get_rank()

    # TODO: precise model, criterion, metric and device
    model = None
    criterion = None
    metric = None
    device = None

    optimizer = optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=5e-4
    )

    if rank == 0:
        # Initialize gradients to 0
        for param in model.parameters():
            param.grad = torch.zeros_like(param)

        # create test loader for MNIST dataset
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        dataset = FashionMNIST(MNIST_PATH, train=False, transform=transform)
        test_loader = DataLoader(dataset, shuffle=False, batch_size=128)

        server = Server(model, test_loader, optimizer, criterion, metric, device)

        if args.run_fedavg:
            server.run_fedavg(n_rounds=args.n_rounds)
        else:
            server.run_ps(n_rounds=args.n_rounds)

    else:
        indices_path = os.path.join(INDICES_PATH, f"client_{rank}.pkl")
        with open(indices_path, "rb") as f:
            indices = pickle.load(f)

        dataset = SubMNIST(MNIST_PATH, indices)
        loader = DataLoader(dataset, shuffle=True, batch_size=32)

        if args.run_fedavg:
            client = \
                Client(
                    model=model,
                    loader=loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    n_local_epochs=args.n_local_epochs,
                    device=device
                )
            client.run_fedavg(n_rounds=args.n_rounds)
        else:
            client = Client(model=model, loader=loader, criterion=criterion, optimizer=optimizer, device=device)
            client.run_ps(n_rounds=args.n_rounds)


if __name__ == "__main__":
    main()
