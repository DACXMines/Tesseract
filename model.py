import torch
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ClientModel:
    def __init__(self, config, id: str):
        self.pretrained = config.pretrained
        self.model = self.init_model(config=config)
        self.id = id
        if config.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def init_model(self, config):
        if config.model_name == 'ResNet':
            return models.resnet18(pretrained=config.pretrained)
        elif config.model_name == 'DNN':
            return models.densenet161(pretrained=config.pretrained)
        elif config.model_name == 'GRU':
            return AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        else :
            raise NotImplementedError(f'{config.model_name} is not implemented yet.')
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")


