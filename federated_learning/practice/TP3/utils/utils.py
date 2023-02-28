from models import *
from trainer import *

from datasets.mnist import *

from client import *

from aggregator import *

from activity_simulator import *
from clients_sampler import *

from .optim import *
from .metrics import *
from .constants import *

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize


def experiment_not_implemented_message(experiment_name):
    error = f"{experiment_name} is not available! " \
            f"Possible are: 'mnist'."

    return error


def get_model(experiment_name, device):
    """
    create model

    Parameters
    ----------
    experiment_name: str

    device: str
        either cpu or cuda


    Returns
    -------
        model (torch.nn.Module)

    """

    if experiment_name == "mnist":
        model = LinearLayer(input_dim=784, output_dim=10, bias=True)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = model.to(device)

    return model


def get_trainer(experiment_name, device, optimizer_name, lr, seed):
    """
    constructs trainer for an experiment for a given seed

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    device: str
        used device; possible `cpu` and `cuda`

    optimizer_name: str

    lr: float
        learning rate

    seed: int

    Returns
    -------
        Trainer

    """
    torch.manual_seed(seed)

    if experiment_name == "mnist":
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = \
        get_model(experiment_name=experiment_name, device=device)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr=lr,
        )

    return Trainer(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        is_binary_classification=is_binary_classification
    )


def get_loader(experiment_name, client_data_path, batch_size, train):
    """

    Parameters
    ----------
    experiment_name: str

    client_data_path: str

    batch_size: int

    train: bool

    Returns
    -------
        * torch.utils.data.DataLoader

    """

    if experiment_name == "mnist":
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNIST(root=client_data_path, train=train, transform=transform)

    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def init_client(args, client_type, client_id, data_dir, logger):
    """initialize one client


    Parameters
    ----------
    args:

    client_type: str

    client_id: int

    data_dir: str

    logger:

    Returns
    -------
        * Client

    """
    train_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.train_bz,
        train=True,
    )

    val_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.test_bz,
        train=False,
    )

    test_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.test_bz,
        train=False,
    )

    trainer = \
        get_trainer(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.local_optimizer,
            lr=args.local_lr,
            seed=args.seed
        )

    if client_type == "perfedavg":
        client = PerFedAvgClient(
            client_id=client_id,
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            local_steps=args.local_steps,
            logger=logger
        )
    else:
        client = Client(
            client_id=client_id,
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            local_steps=args.local_steps,
            logger=logger
        )

    return client


def get_aggregator(
        aggregator_type,
        clients_dict,
        clients_weights_dict,
        global_trainer,
        logger,
        verbose,
        seed
):
    """
    Parameters
    ----------
    aggregator_type: str
        possible are {"centralized", "no_communication"}

    clients_dict: Dict[int: Client]

    clients_weights_dict: Dict[int: Client]

    global_trainer: Trainer

    logger: torch.utils.tensorboard.SummaryWriter

    verbose: int

    seed: int


    Returns
    -------
        * Aggregator
    """
    if aggregator_type == "centralized":
        return CentralizedAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=logger,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=logger,
            verbose=verbose,
            seed=seed
        )
    else:
        error_message = f"{aggregator_type} is not a possible aggregator type, possible are: "
        for type_ in AGGREGATOR_TYPES:
            error_message += f" {type_}"


def get_clients_weights(objective_type, n_samples_per_client):
    """compute the weights to be associated to every client

    If objective_type is "average", clients receive the same weight

    If objective_type is "weighted", clients receive weight proportional to the number of samples


    Parameters
    ----------
    objective_type: str
        type of the objective function; possible are: {"average", "weighted"}

    n_samples_per_client: Dict[int: float]


    Returns
    -------
        * Dict[int: float]

    """
    weights_dict = dict()
    n_clients = len(n_samples_per_client)

    if objective_type == "average":
        for client_id in n_samples_per_client:
            weights_dict[int(client_id)] = 1 / n_clients

    elif objective_type == "weighted":
        total_num_samples = 0

        for client_id in n_samples_per_client:
            total_num_samples += n_samples_per_client[client_id]

        for client_id in n_samples_per_client:
            weights_dict[int(client_id)] = n_samples_per_client[client_id] / total_num_samples

    else:
        raise NotImplementedError(
            f"{objective_type} is not an available objective type. Possible are 'average' and `weighted"
        )

    return weights_dict


def get_activity_simulator(all_clients_cfg, rng):
    clients_ids = []
    availability_types = []
    availabilities = []
    stability_types = []
    stabilities = []

    for client_id in all_clients_cfg.keys():
        clients_ids.append(client_id)
        availability_types.append(all_clients_cfg[client_id]["availability_type"])
        availabilities.append(all_clients_cfg[client_id]["availability"])
        stability_types.append(all_clients_cfg[client_id]["stability_type"])
        stabilities.append(all_clients_cfg[client_id]["stability"])

    clients_ids = np.array(clients_ids, dtype=np.int32)
    availabilities = np.array(availabilities, dtype=np.float32)
    stabilities = np.array(stabilities, dtype=np.float32)

    activity_simulator = \
        ActivitySimulator(clients_ids, availability_types, availabilities, stability_types, stabilities, rng=rng)

    return activity_simulator


def get_clients_sampler(
        sampler_type,
        activity_simulator,
        clients_weights_dict,
        rng
):
    if sampler_type == "unbiased":
        return UnbiasedClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    elif sampler_type == "biased":
        return BiasedClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    else:
        error_message = f"{sampler_type} is not an available sampler type, possible are:"

        for t in SAMPLER_TYPES:
            error_message += f"{t},"

        raise NotImplementedError(error_message)