import torch
import torch.distributed as dist


class Client(object):
    r"""Implement client

    Attributes
    ----------
    model
    loader
    criterion
    device
    rank
    world_size

    Methods
    ----------
    __init__

    get_batch

    compute_gradients

    fit_epoch

    push_model

    push_gradients

    pull_model

    run_ps

    run_fedavg
    
    """
    def __init__(self, model, loader, criterion, optimizer, n_local_epochs=None, device=None):
        self.model = model
        self.loader = loader
        self.iterator = iter(self.loader)
        self.criterion = criterion
        self.optimizer = optimizer

        self.n_local_epochs = n_local_epochs

        if device is None:
            self.device = torch.device("cpu")

        self.device = device

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def get_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        return batch

    def compute_gradients(self, batch):
        """
        :param batch: tuple of inputs and labels
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.model.zero_grad()

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)
        loss.backward()

    def fit_epoch(self):
        self.model.train()

        n_samples = len(self.loader.dataset)

        global_loss = 0.

        for x, y in self.loader:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x).squeeze()

            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            global_loss += loss.item() * len(y)

        return global_loss / n_samples

    def push_model(self):
        for param in self.model.parameters():
            pass
            # TODO: push model parameters to the server (remember that server has `rank=0`)
            # TODO: you can use `dist.reduce`

    def push_gradients(self):
        for param in self.model.parameters():
            pass
            # TODO: push gradients to the server (remember that server has `rank=0`)
            # TODO: you can use `dist.reduce`

    def pull_model(self):
        for param in self.model.parameters():
            pass
            # TODO: pull model parameters from the server (remember that server has `rank=0`)
            # TODO: you can use `dist.braodcast()`

    def run_ps(self, n_rounds):
        for _ in range(n_rounds):
            self.pull_model()
            batch = self.get_batch()
            self.compute_gradients(batch)
            self.push_gradients()

    def run_fedavg(self, n_rounds):
        for _ in range(n_rounds):
            self.pull_model()

            for _ in range(self.n_local_epochs):
                self.fit_epoch()

            self.push_model()
