import torch 
import torch.nn as nn 
import numpy as np 
import copy
import statistics


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

    
    def Bulyan (self, net): 
        #step 1 : chose t = n_clients - 2 * cmax gradients according to Krum

        params = self.grads.reshape(0,-1) #n_clients*grad_size 
        clients = self.n_clients
        received_set = copy.copy(params)
        selected_set = torch.zeros(self.n_clients - 2 * self.cmax)
        
        for it in range (self.n_clients - 2 * self.cmax) :
            distances = torch.tensor(clients, clients-1)
            for i in range(clients):
                for j in range(clients):
                    if i!=j:
                        distances[i][j]+=torch.cdist(received_set[i], received_set[j], 2) #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                        #distances[i][j] = distance entre gradient du client 1 = g1 et g2 
            sorted_distances = distances.sort(axis=0)
            closest_neighbors = clients - self.cmax - 2 

            closest = torch.sum(sorted_distances[:,:closest_neighbors], axis=0) #clients*1

            selected_grad = torch.argmax(closest, axis=1).item() #retourne l'indice du gradient à choisir 

            #the chosen gradient is added to the list, and the corresponding client is removed
            selected_set[it] = selected_grad
            received_set = torch.cat([received_set[0:selected_grad], received_set[selected_grad+1:]])
            clients -= 1

        
        #step 2 : compute the resulting gradient

        #as put in the paper : each i-th coordinate of G is equal to the average of the
        #b = n_clients - 4 * cmax closest i-th coordinates to the median i-th coordinate of the t selected gradients

        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                #compute the median for each coordinate
                selected_coordinates = []
                for selected_grad in selected_set :
                    selected_coordinates.append(self.lr*params[selected_grad][idx])
                median_coordinate = statistics.median(selected_coordinates)

                #select the b closest coordinates
                b = self.n_clients - 4 * self.cmax
                final_coordinates = []
                for selected_coordinate in selected_coordinates :
                    if len(final_coordinates) < b :
                        final_coordinates.append(selected_coordinate - median_coordinate)
                    else :
                        max_diff = max(final_coordinates, key=abs)
                        if abs(selected_coordinate - median_coordinate) < abs(max) :
                            final_coordinates.remove(max_diff)
                            final_coordinates.append(selected_coordinate - median_coordinate)

                #compute the mean over the b coordinates
                final_mean_coordinate = statistics.mean(final_coordinates) + median_coordinate

                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*final_mean_coordinate
        return net 
    
    def foolsgold(self, net) : 
        params = self.grads.reshape(0,-1) #n_clients*grad_size 
        _, grad_size = params.shape
        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cs_map = torch.zeros((self.n_clients, self.n_clients)) #cs_map[i][j] = cosine similarity between i & j
        for i in range (self.n_clients):
            for j in range (i):
                #fill the cosine simiarity map
                cs_map[i,j] = cos_sim(params[i], params[j])
                cs_map[j,i] = cs_map[i,j]
        #rescaling of all similarities
        v = torch.zeros(self.n_clients)        
        for i in range (self.n_clients):
            v[i] = torch.max(cs_map[i])

        alphas = torch.zeros(self.n_clients)
        #Pardoning
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i == j:
                    continue
                if v[i] < v[j]:
                    cs_map[i][j] = cs_map[i][j] * v[i] / v[j]
            alphas[i] = 1 - torch.max(cs_map[i])
        
        #Logits
        alphas[alphas > 1] = 1
        alphas[alphas < 0] = 0

        alphas = alphas / np.max(alphas)
        alphas[(alphas == 1)] = .99
        
        alphas = (np.log(alphas / (1 - alphas)) + 0.5)
        alphas[(np.isinf(alphas) + alphas > 1)] = 1
        alphas[(alphas < 0)] = 0

        aggregated_params = torch.matmul(torch.transpose(params, 0, 1), alphas.reshape(-1,1))
        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*aggregated_params[idx]
        return net
        
    def faba(self, net):  
        params = self.grads.reshape(0,-1) #n_clients*grad_size 
        params_faba = copy.copy(params)
        #faba algo : cmax iters, take out client furthest away from mean
        for _ in range (self.cmax) :
            g0 = torch.mean(params_faba, dim=0)
            dist_g0 = torch.zeros(len(params_faba))
            for i in range (len(params_faba)) :
                dist_g0[i] = torch.norm(g0 - params_faba[i]).item()
            client_max_dist = int(torch.argmax(dist_g0)).item()
            params_faba = torch.cat([params_faba[0:client_max_dist], params_faba[client_max_dist+1:]])
        #the aggregated gradient
        g0 = torch.mean(params_faba, dim=0)
        with torch.no_grad():
            global_param = net.parameters()
            for idx, _ in enumerate(global_param) : 
                if global_param[idx].requires_grad == True : 
                    global_param[idx] +=  self.lr*g0[idx]
        return net 


    def tesseract(self) : 
    
    