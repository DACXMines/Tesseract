import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import tensorflow as tf

'''
with open(r'cofig.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    params = yaml.load(file, Loader=yaml.FullLoader)

dataset = params['dataset']
batch_size = params['batch_size']
'''


class DataPipeline :
    def __init__(self, dataset_name : str, batch_size : int) -> None:
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        if self.dataset_name == 'FashionMNIST' :
            self.training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])
            )

            self.test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])
            )
        elif self.dataset_name == 'MNIST':
            self.training_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
            )

            self.test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
            )
        elif self.dataset_name == 'CIFAR10':
            self.training_data = datasets.CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            )

            self.test_data = datasets.CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            )
        
        else :
            raise NotImplementedError(f'{self.dataset_name} dataset not available yet.')



        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
            

