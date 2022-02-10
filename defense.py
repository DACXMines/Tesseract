import torch 
import torch.nn as nn 
import numpy as np 


#State of the art defense systems : 
class DefenseSystem() : 
    
    def __init__(self, grads, lr, cmax) : 
        self.grads = grads
        self.n_clients, self.size_grad = grads.size()
        self.lr = lr 
        self.cmax = cmax

    def fed_sgd(self, net, weights, attack): #weights c'est dans le main et c'est 
        # /le nombre de data sample for each client, spécifique à SGD 

        n = self.n_clients 
        params = self.grads.reshape(0,-1)
        #torch.stack([torch.cat([xx for xx in x], dim=0) for x in self.grads])
        #size = n_clients*size_grad
        params = attack(self.lr, params, self.cmax)
        params_weighted = torch.matmul(torch.transpose(params,0,1),weights) #weighted means = global_params
        #size params_weighted = size_grad*1 = (PARAMS) size_grad*n_clients x (WEIGHTS) n_clients*1 

        ## TODO : weight by number of data sample each client holds 
        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*params_weighted[idx]
        return net 
    
    def Trimmed_mean(self, net, attack):
        params = self.grads.reshape(0,-1) #n_clients*grad_size 
        _, L = params.shape
        sorted_params = torch.sort(params, axis=0)

        trimmed_mean = torch.mean(sorted_params[:,self.cmax:L-self.cmax], axis=0) #n_clients*(grad_size-2*cmax)
        grad_final_update = torch.mean(trimmed_mean[:,:], axis=1) #1*(grad_size-2*cmax)

        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*grad_final_update[idx]
        return net 



    def Krum(self, net):
        params = self.grads.reshape(0,-1) #n_clients*grad_size 
        _, grad_size = params.shape
        distances = torch.tensor(self.n_clients, self.n_clients-1)
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i!=j:
                    distances[i][j]+=torch.cdist(params[i], params[j], 2) #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                    #distances[i][j] = distance entre gradient du client 1 = g1 et g2 
        sorted_distances = distances.sort(axis=0)
        closest_neighbors = self.n_clients - self.cmax - 2 

        closest = torch.sum(sorted_distances[:,:closest_neighbors], axis=0) #n_clients*1

        selected_grad = torch.argmax(closest, axis=1) #retourne l'indice du gradient à choisir 

        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*params[selected_grad][idx]
        return net 

    
    def Bulyan(self): 
        
    
    def foolsgold(self) : 
        
    def faba(self):  

    def tesseract(self) : 