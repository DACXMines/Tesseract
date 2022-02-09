import torch 
import torch.nn as nn 
import numpy as np 


#State of the art defense systems : 
class DefenseSystem() : 
    
    def __init__(self, grads) : 
        self.grads = grads
        self.n_clients = len(grads)

    def fed_sgd(self):
        n = self.n_clients 
        weighted_mean = np.mean(self.grads)
        

class Krum():
    

class Bulyan() : 


class Trimmed_mean(): 


class foolsgold() : 


class faba():  


#Tesseract : 
