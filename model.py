import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ClientModel:
    def __init__(self, config, id: str):
        self.pretrained = config["pretrained"]
        self.model = self.init_model(config=config)
        self.id = id
        if config["model_name"] == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def init_model(self, config):
        if config["model_name"] == 'ResNet':
            return models.resnet18(pretrained=config["pretrained"])
        elif config["model_name"] == 'DNN':
            return models.densenet161(pretrained=config["pretrained"])
        elif config["model_name"] == 'GRU':
            return AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        elif config["model_name"] == 'CNN':
            return DNN()
        else :
            raise NotImplementedError(f'{config["model_name"]} is not implemented yet.')
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    
class DNN(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=1)
        self.fc1 = nn.Linear(50*7*7, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x