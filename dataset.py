import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import yaml
import tensorflow as tf

with open(r'cofig.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    params = yaml.load(file, Loader=yaml.FullLoader)

dataset = params['dataset']
batch_size = params['batch_size']

class DataPipeline :
    def __init__(self, dataset_name : str, batch_size : int) -> None:
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        if dataset_name == 'FashionMNIST' :
            self.training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )

            self.test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        elif self.model_name == 'MNIST':
            self.training_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )

            self.test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        elif self.model_name == 'CIFAR10':
            self.training_data = datasets.CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )

            self.test_data = datasets.CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        elif self.model_name == 'CIFAR10':
            self.training_data = datasets.CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )

            self.test_data = datasets.CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        else :
            raise NotImplementedError(f'{self.model_name} is not implemented yet.')



        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
            

