import torch
import torch.distributed as dist


class Server(object):
    r"""Implements server

    Attributes
    ----------
    model
    test_loader
    criterion
    optimizer
    metric
    logger
    log_freq
    device
    rank
    world_size

    Methods
    ----------
    __init__
    evaluate_loader
    push_model
    pull_gradients
    step
    run

    """
    def __init__(self, model, test_loader, optimizer, criterion, metric, device, logger=None, log_freq=5):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        assert self.rank == 0, "Server should have rank=0"

        self.model = model
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.logger = logger
        self.log_freq = log_freq
        self.device = device

    def evaluate_loader(self):
        """
        evaluate learner on `iterator`
        :return
            global_loss and  global_metric accumulated over the iterator
        """
        n_samples = len(self.test_loader.dataset)

        self.model.eval()

        global_loss = 0.
        global_metric = 0.

        for x, y in self.test_loader:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            with torch.no_grad():
                y_pred = self.model(x).squeeze()
                global_loss += self.criterion(y_pred, y).item() * len(y_pred)
                global_metric += self.metric(y_pred, y).item() * len(y_pred)

        return global_loss / n_samples, global_metric / n_samples

    def broadcast_model(self):
        for param in self.model.parameters():
            # TODO: broadcast the model to all workers
            # TODO: you can use `dist.broadcast
            pass

    def reduce_gradients(self):
        for param in self.model.parameters():
            # TODO: receive gradients from all workers
            # TODO: you can use `reduce`
            pass

    def reduce_models(self):
        for param in self.model.parameters():
            param_old = param.data.clone()
            # TODO: receive models from all workers
            # TODO: you can use `reduce`
            param.data -= param_old.data
            param.data /= (self.world_size - 1)

    def step(self):
        self.optimizer.step()

    def run_ps(self, n_rounds):
        for round_idx in range(n_rounds):
            self.broadcast_model()
            self.model.zero_grad()
            self.reduce_gradients()
            self.step()

            if round_idx % self.log_freq == self.log_freq - 1:
                loss, acc = self.evaluate_loader()
                print(f"Round {round_idx} | Loss: {loss:.3f} | Acc.: {(acc*100):.3f}%")

    def run_fedavg(self, n_rounds):
        for round_idx in range(n_rounds):
            self.broadcast_model()
            self.reduce_models()

            if round_idx % self.log_freq == self.log_freq - 1:
                loss, acc = self.evaluate_loader()
                print(f"Round {round_idx} | Loss: {loss:.3f} | Acc.: {(acc*100):.3f}%")
