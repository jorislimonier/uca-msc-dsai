import os
import time
import random
import pickle
import argparse

import numpy as np

from torchvision.datasets import FashionMNIST


MNIST_PATH = "mnist"
INDICES_PATH = "indices"
N_SAMPLES = 60000
N_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_clients',
        help='number of clients;',
        type=int
    )
    parser.add_argument(
        '--n_iid',
        help='if selected the dataset is slit in a non i.i.d. fashion, otherwise it is i.i.d',
        action='store_true'
    )
    parser.add_argument(
        '--n_classes_per_client',
        help='number of classes associated to a given client; only used with `--n_iid`; default is `2`',
        default=2
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=1234,
        required=False
    )

    return parser.parse_args()


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follows:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.

    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def generate_data(n_clients, seed=1234):
    dataset = FashionMNIST(MNIST_PATH, download=True)

    os.makedirs(INDICES_PATH, exist_ok=True)
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    all_indices = list(range(N_SAMPLES)) 
    rng.shuffle(all_indices)

    if args.n_iid:
        clients_indices = non_iid_split(
            dataset=dataset,
            n_classes=N_CLASSES,
            n_clients=args.n_clients,
            n_classes_per_client=args.n_classes_per_client,
            seed=args.seed
        )
    else:
        clients_indices = iid_divide(all_indices, n_clients)

    for client_id, indices in enumerate(clients_indices):
        client_path = os.path.join(INDICES_PATH, "client_{}.pkl".format(client_id+1))
        save_data(indices, os.path.join(client_path))


if __name__ == "__main__":
    args = parse_args()
    generate_data(n_clients=args.n_clients, seed=args.seed)

