import torch
from copy import deepcopy


class Trainer:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model: nn.Module
        the model trained by the learner

    criterion: torch.nn.modules.loss
        loss function used to train the `model`

    metric: fn
        function to compute the metric, should accept as input two vectors and return a scalar

    device : str or torch.device)

    optimizer: torch.optim.Optimizer

    is_binary_classification: bool

    is_ready: bool

    Methods
    -------

    fit_epoch: perform several optimizer steps on all batches drawn from `loader`

    fit_epochs: perform multiple training epochs

    evaluate_loader: evaluate `model` on a loader

    get_param_tensor: get `model` parameters as a unique flattened tensor

    get_grad_tensor: get `model` gradients as a unique flattened tensor

    set_grad_tensor:

    __sub__: differentiate trainers

    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            is_binary_classification
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer

        self.is_binary_classification = is_binary_classification

        self.model_dim = int(self.get_param_tensor().shape[0])

        self.is_ready = True

    def fit_epoch(self, loader):
        """
        perform several optimizer steps on all batches drawn from `loader`
        
        Parameters
        ----------
        loader: torch.utils.data.DataLoader

        Returns
        -------
            None
        """
        self.model.train()

        for x, y in loader:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device).type(torch.long)

            if self.is_binary_classification:
                y = y.to(self.device).type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            outs = self.model(x)

            loss = self.criterion(outs, y)

            loss.backward()

            self.optimizer.step()

    def fit_epochs(self, loader, n_epochs):
        """
        perform multiple training epochs


        Parameters
        ----------
        loader: torch.utils.data.DataLoader

        n_epochs: int

        Returns
        -------
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(loader)

    def evaluate_loader(self, loader):
        """
        evaluate learner on loader
        
        Parameters
        ----------
        loader: torch.utils.data.DataLoader

        Returns
        -------
            float: loss
            float: accuracy

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device).type(torch.long)

                if self.is_binary_classification:
                    y = y.to(self.device).type(torch.float32).unsqueeze(1)

                outs = self.model(x)

                global_loss += self.criterion(outs, y).item() * y.size(0)
                global_metric += self.metric(outs, y).item() * y.size(0)

                n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        Returns
        -------
            * torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        Returns
        -------
            * torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_grad_tensor(self, grad_tensor):
        """

        Parameters
        ----------
        grad_tensor: torch.tensor of shape (`self.model_dim`,)

        Returns
        -------
            None

        """
        grad_tensor = grad_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            if param.grad is None:
                param.grad = param.data.clone()

            param.grad.data = \
                deepcopy(grad_tensor[current_index: current_index + current_dimension].reshape(param_shape))

            current_index += current_dimension

    def __sub__(self, other):
        """differentiate trainers

        returns a Trainer object with the same parameters
        as self and gradients equal to the difference with respect to the parameters of "other"

        Remark: returns a copy of self, and self is modified by this operation

        Parameters
        ----------
        other: Trainer

        Returns
        -------
            * Trainer
        """
        params = self.get_param_tensor()
        other_params = other.get_param_tensor()

        self.set_grad_tensor(other_params - params)

        return self





