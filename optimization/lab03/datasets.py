from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import FashionMNIST


class SubMNIST(Dataset):
    """
    Constructs a subset of MNIST dataset from list of indices;

    Attributes
    ----------
    indices: iterable of integers

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, root_path, indices):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        mnist_train = FashionMNIST(root=root_path,
                                   download=True,
                                   train=True,
                                   transform=self.transform)

        self.indices = indices

        self.data = mnist_train.data
        self.targets = mnist_train.targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
