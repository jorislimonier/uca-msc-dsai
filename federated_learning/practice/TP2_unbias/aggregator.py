import random
import time
from abc import ABC, abstractmethod
from client import Client
import numpy as np
from tqdm import tqdm
from utils.torch_utils import *


class Aggregator(ABC):
  r"""Base class for Aggregator.

  `Aggregator` dictates communications between clients_dict

  Attributes
  ----------
  clients_dict: Dict[int: Client]

  clients_weights_dict: Dict[int: Client]

  global_trainer: List[Trainer]

  n_clients:

  model_dim: dimension if the used model

  c_round: index of the current communication round

  verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

  sampling_rate: proportion of clients used at each round; default is `1.`

  sample_with_replacement: is True, client are sampled with replacement; default is False

  n_clients_per_round:

  sampled_clients:

  logger: SummaryWriter

  rng: random number generator

  np_rng: numpy random number generator

  Methods
  ----------
  __init__

  mix

  update_clients

  write_logs

  save_state

  load_state

  """

  def __init__(
    self,
    clients_dict: dict[int, Client],
    clients_weights_dict,
    global_trainer,
    sampling_rate: float,
    heterogeneous_participation: bool,
    logger,
    sample_with_replacement: bool = False,
    verbose=0,
    seed=None,
  ):
    """

    Parameters
    ----------
    clients_dict: Dict[int: Client]

    clients_weights_dict: Dict[int: Client]

    global_trainer: Trainer

    logger: SummaryWriter

    verbose: int

    sampling_rate: float

    sample_with_replacement: bool

    heterogeneous_participation: bool

    seed: int

    """
    rng_seed = seed if (seed is not None and seed >= 0) else int(time.time())
    self.rng = random.Random(rng_seed)
    self.np_rng = np.random.default_rng(rng_seed)

    self.clients_dict = clients_dict
    self.n_clients = len(clients_dict)

    self.clients_weights = []
    self.clients_weights_dict = clients_weights_dict
    for idx in range(self.n_clients):
      self.clients_weights.append(self.clients_weights_dict[idx])

    self.global_trainer = global_trainer
    self.device = self.global_trainer.device

    self.verbose = verbose
    self.logger = logger

    self.model_dim = self.global_trainer.model_dim

    # TODO: Initialize parameters
    self.sampling_rate = sampling_rate
    self.sample_with_replacement = sample_with_replacement
    # Hint: the number of clients per round is related to the sampling_rate and must be at least one
    self.n_clients_per_round = int(self.sampling_rate * self.n_clients)

    self.sampled_clients_ids = list()
    self.sampled_clients = list()

    self.c_round = 0

  @abstractmethod
  def mix(self):
    """mix sampled clients according to weights

    Parameters
    ----------

    Returns
    -------
        None
    """
    pass

  @abstractmethod
  def update_clients(self):
    """
    send the new global model to the clients
    """
    pass

  def write_logs(self):
    global_train_loss = 0.0
    global_train_metric = 0.0
    global_test_loss = 0.0
    global_test_metric = 0.0

    for client_id, client in self.clients_dict.items():

      train_loss, train_metric, test_loss, test_metric = client.write_logs(
        counter=self.c_round
      )

      if self.verbose > 1:

        tqdm.write("*" * 30)
        tqdm.write(f"Client {client_id}..")

        tqdm.write(
          f"Train Loss: {train_loss:.3f} | Train Metric: {train_metric :.3f}|", end=""
        )
        tqdm.write(f"Test Loss: {test_loss:.3f} | Test Metric: {test_metric:.3f} |")

        tqdm.write("*" * 30)

      global_train_loss += self.clients_weights_dict[client_id] * train_loss
      global_train_metric += self.clients_weights_dict[client_id] * train_metric
      global_test_loss += self.clients_weights_dict[client_id] * test_loss
      global_test_metric += self.clients_weights_dict[client_id] * test_metric

    if self.verbose > 0:

      tqdm.write("+" * 50)
      tqdm.write("Global..")
      tqdm.write(
        f"Train Loss: {global_train_loss:.3f} | Train Metric: {global_train_metric:.3f} |",
        end="",
      )
      tqdm.write(
        f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_metric:.3f} |"
      )
      tqdm.write("+" * 50)

    self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
    self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
    self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
    self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)
    self.logger.flush()

  # TODO: write method
  def sample_clients(self):
    """
    sample a list of clients
    """

    # Hint: use rng to write self.sampled_clients_ids
    self.sampled_clients_ids = np.random.choice(
      self.n_clients, self.n_clients_per_round, replace=self.sample_with_replacement
    )

    # TODO: print the sampled clients
    print(f"{self.sampled_clients_ids=}")

    # TODO: return client objects for sampled clients
    self.sampled_clients = [self.clients_dict[idx] for idx in self.sampled_clients_ids]


class NoCommunicationAggregator(Aggregator):
  r"""Clients do not communicate. Each client work locally"""

  def mix(self):
    self.sample_clients()

    for client in self.sampled_clients:
      client.step()

    trainers_deltas = [
      self.clients_dict[idx].trainer - self.global_trainer
      for idx in range(self.n_clients)
    ]

    self.global_trainer.optimizer.zero_grad()

    self.c_round += 1

  def update_clients(self):
    pass


class CentralizedAggregator(Aggregator):
  r"""Standard Centralized Aggregator.

  Clients get fully synchronized with the average client.

  """

  def mix(self):

    # TODO: the aggregator samples clients
    self.sample_clients()

    clients_weights = torch.tensor(self.clients_weights, dtype=torch.float32)

    # TODO: only the sampled clients train
    for idx in range(self.n_clients):
      self.clients_dict[idx].step()

    # TODO: only the sampled clients send their deltas to the aggregator
    trainers_deltas = [
      client.trainer - self.global_trainer for client in self.sampled_clients
    ]

    self.global_trainer.optimizer.zero_grad()

    aggregation_weights = clients_weights[self.sampled_clients_ids] / self.sampling_rate

    # TODO: use the proper aggregation weight
    average_models(
      trainers_deltas,
      target_trainer=self.global_trainer,
      weights=aggregation_weights,
      average_params=False,
      average_gradients=True,
    )

    self.global_trainer.optimizer.step()

    # assign the updated model to all clients_dict
    self.update_clients()

    self.c_round += 1

  def update_clients(self):
    for client_id, client in self.clients_dict.items():

      copy_model(client.trainer.model, self.global_trainer.model)
