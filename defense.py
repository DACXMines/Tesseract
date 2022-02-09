import torch 
import torch.nn as nn 
import numpy as np 


#State of the art defense systems : 
class DefenseSystem() : 
    
    def __init__(self, grads, lr) : 
        self.grads = grads
        self.n_clients, self.size_grad = grads.size()
        self.lr = lr 

    def fed_sgd(self, net, weights, attack): #weights c'est dans le main et c'est 
        # /le nombre de data sample for each client, spécifique à SGD 

        n = self.n_clients 
        params = torch.stack([torch.cat([xx for xx in x], dim=0) for x in self.grads])
        #size = n_clients*size_grad
        params = attack(params)

        params_weighted = torch.matmul(torch.transpose(params,0,1),weights) #weighted means = global_params
        #size params_weighted = size_grad*1 = (PARAMS) size_grad*n_clients x (WEIGHTS) n_clients*1 

        ## TODO : weight by number of data sample each client holds 
        with torch.no_grad():
            idx = 0 
            param = net.parameters()
            if param[idx].requires_grad == True : 
                param[idx] +=  self.lr*params_weighted[idx]
        return net 

    
    def Krum(self):
    
    def Bulyan(self): 
        
    def Trimmed_mean(self): 
    
    def foolsgold(self) : 
        
    def faba(self):  

    def tesseract(self) : 