import time
import json

from abc import ABC, abstractmethod


class ClientsSampler(ABC):
    r"""Base class for clients_dict sampler

    Attributes
    ----------
    activity_simulator: ActivitySimulator

    clients_ids:

    n_clients: int

    clients_weights_dict: Dict[int: float]
        maps clients ids to their corresponding weight/importance in the true objective function

    _availability_dict: Dict[int: float]
        maps clients ids to their stationary participation probability

    _stability_dict: Dict[int: float]
        maps clients ids to the spectral gap of their corresponding markov chains

    history: Dict[int: Dict[str: List]]
        stores the active and sampled clients and their weights at every time step

    _time_step: int
        tracks the number of steps

    rng:

    Methods
    ----------
    __init__

    _update_estimates

    sample_clients

    step

    save_history

    """

    def __init__(
            self,
            activity_simulator,
            clients_weights_dict,
            rng=None,
            *args,
            **kwargs
    ):
        """

        Parameters
        ----------
        activity_simulator: ActivitySimulator

        clients_weights_dict: Dict[int: float]

        rng:

        """

        self.activity_simulator = activity_simulator

        self.clients_ids = list(clients_weights_dict.keys())
        self.n_clients = len(self.clients_ids)

        self.clients_weights_dict = clients_weights_dict

        self._availability_types_dict, self._availability_dict, self._stability_types_dict, self._stability_dict = \
            self._gather_clients_parameters()

        self.history = dict()

        self._time_step = -1

        self._debug = True

        if rng is None:
            seed = int(time.time())
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def _gather_clients_parameters(self):
        availability_types_dict = dict()
        availability_dict = dict()
        stability_types_dict = dict()
        stability_dict = dict()

        for idx, client_id in enumerate(self.activity_simulator.clients_ids):
            availability_types_dict[int(client_id)] = str(self.activity_simulator.availability_types[idx])
            availability_dict[int(client_id)] = float(self.activity_simulator.availabilities[idx])
            stability_types_dict[int(client_id)] = str(self.activity_simulator.stability_types[idx])
            stability_dict[int(client_id)] = float(self.activity_simulator.stabilities[idx])

        return availability_types_dict, availability_dict, stability_types_dict, stability_dict

    def get_active_clients(self):
        return self.activity_simulator.get_active_clients()

    def step(self, active_clients, sampled_clients_ids, sampled_clients_weights):
        """update the internal state of the clients sampler

        Parameters
        ----------
        active_clients: List[int]

        sampled_clients_ids: List[int]

        sampled_clients_weights: Dict[int: float]


        Returns
        -------
            None
        """
        self.activity_simulator.step()
        self._time_step += 1

        current_state = {
            "active_clients": active_clients,
            "sampled_clients_ids": sampled_clients_ids,
            "sampled_clients_weights": sampled_clients_weights
        }

        self.history[self._time_step] = current_state

    def save_history(self, json_path):
        """save history and clients ids

        save a dictionary with:
            * history: stores the active and sampled clients and their weights at every time step
            * clients_ids: list of clients ids stored as integers
            * true_weights_dict: dictionary mapping clients ids to their true weights

        Parameters
        ----------
        json_path: path of a .json file

        Returns
        -------
            None
        """
        metadata = {
            "history": self.history,
            "clients_ids": self.clients_ids,
            "clients_true_weights": self.clients_weights_dict,
            "clients_availability_types": self._availability_types_dict,
            "clients_true_availability": self._availability_dict,
            "clients_stability_types": self._stability_types_dict,
            "clients_true_stability": self._stability_dict
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f)

    @abstractmethod
    def sample(self, active_clients):
        """sample clients_dict

        Parameters
        ----------
        active_clients: List[int]

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * LIst[float]: weights to be associated to the sampled clients_dict
        """
        pass


class UnbiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients_dict
    """

    def sample(self, active_clients):
        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self._availability_dict[client_id]
            )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class BiasedClientsSampler(ClientsSampler):
    """
    Samples only the more available clients
    """
    def __init__(
            self,
            activity_simulator,
            clients_weights_dict,
            rng=None
    ):
        super(BiasedClientsSampler, self).__init__(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )

        # get more available clients
        self._available_clients = []
        for client_id in self._availability_types_dict:
            if self._availability_types_dict[client_id] == "available":
                self._available_clients.append(client_id)

    def sample(self, active_clients):
        sampled_clients_ids = list(set(active_clients) & set(self._available_clients))

        sampled_clients_weights = []
        for client_id in sampled_clients_ids:
            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self._availability_dict[client_id]
            )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights
