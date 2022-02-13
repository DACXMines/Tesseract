import torch 
import torch.nn as nn 
import numpy as np 
import copy
import statistics


#State of the art defense systems : 
class DefenseSystem() : 
    
    def __init__(self, grads, lr, cmax) : 
        self.grads = grads
        '''
        self.n_clients = len(grads)
        self.size_grad = len(grads[0])
        '''
        self.n_clients, self.size_grad = grads.size()
        self.lr = lr 
        self.cmax = cmax
    
    def aggregate(self, config, net) :
        aggregation_model = config["aggregation_model"]
        if aggregation_model == "trimmed_mean" :
            return self.Trimmed_mean(net)
        elif aggregation_model == "krum" :
            return self.Krum(net)
        elif aggregation_model == "bulyan" :
            return self.Bulyan(net)
        elif aggregation_model == "foolsgold" :
            return self.foolsgold(net)
        elif aggregation_model == "faba" :
            return self.faba(net)

    def fed_sgd(self, net, weights): #weights c'est dans le main et c'est 
        # /le nombre de data sample for each client, spécifique à SGD 

        params = self.grads
        params_weighted = torch.matmul(torch.transpose(params,0,1),weights) #weighted means = global_params
        #size params_weighted = size_grad*1 = (PARAMS) size_grad*n_clients x (WEIGHTS) n_clients*1 

        ##Weight by number of data sample each client holds 
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += params_weighted[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()  
        del params, params_weighted
        return net 
    
    def Trimmed_mean(self, net): 
        params = self.grads
        _, L = params.shape
        sorted_params = torch.sort(params, axis=0)

        trimmed_mean = torch.mean(sorted_params[0][self.cmax:L-self.cmax,:], axis=0) #1*(grad_size)

        with torch.no_grad():
            idx = 0
            for _, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += trimmed_mean[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()  
                    
        del params, sorted_params, trimmed_mean
        return net  

    def Krum(self, net):
        params = self.grads
        distances = torch.zeros([self.n_clients, self.n_clients-1])
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i!=j:
                    if i > j :
                        distances[i][j]+=torch.dist(params[i], params[j], 2).item() #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                    else :
                        distances[i][j - 1]+=torch.dist(params[i], params[j], 2).item() #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                    #distances[i][j] = distance entre gradient du client 1 = g1 et g2 
        sorted_distances, _ = distances.sort(axis=0)
        closest_neighbors = self.n_clients - self.cmax - 2 

        closest = torch.sum(sorted_distances[:,:closest_neighbors], axis=0) #n_clients*1

        selected_grad = torch.argmax(closest, axis=-1) #retourne l'indice du gradient à choisir 

        with torch.no_grad():
            idx = 0
            for _, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += params[selected_grad][idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()  
        del params
        return net 

    def Bulyan (self, net): 
        #step 1 : chose t = n_clients - 2 * cmax gradients according to Krum

        params = self.grads
        clients = self.n_clients
        received_set = copy.copy(params)
        selected_set = torch.zeros(self.n_clients - 2 * self.cmax)

        remaining_map = {} #map between remaining clients and their original index in the params map
        
        for it in range (self.n_clients - 2 * self.cmax) :
            distances = torch.zeros([clients, clients-1])
            for i in range(clients):
                for j in range(clients):
                    if i!=j:
                        if i > j :
                            distances[i][j]+=torch.dist(params[i], params[j], 2).item() #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                        else :
                            distances[i][j - 1]+=torch.dist(params[i], params[j], 2).item() #Pour chaque param[i], on calcule la 2-norme distance entre les 2 
                        #distances[i][j] = distance entre gradient du client 1 = g1 et g2 
            sorted_distances, _ = distances.sort(axis=0)
            closest_neighbors = clients - self.cmax - 2 

            closest = torch.sum(sorted_distances[:,:closest_neighbors], axis=0) #n_clients*1

            selected_grad = torch.argmax(closest, axis=-1).item() #retourne l'indice du gradient à choisir 

            #the chosen gradient is added to the list, and the corresponding client is removed
            selected_set[it] = remaining_map.get(selected_grad, selected_grad)

            received_set = torch.cat([received_set[0:selected_grad], received_set[selected_grad+1:]])

            clients -= 1

            remaining_map_prev = copy.deepcopy(remaining_map)

            for remaining_client in range (clients) :
                if remaining_client >= selected_grad :
                    remaining_map[remaining_client] = remaining_map_prev.get(remaining_client + 1, remaining_client + 1)
                else :
                    remaining_map[remaining_client] = remaining_map_prev.get(remaining_client, remaining_client)
        #step 2 : compute the resulting gradient

        #as put in the paper : each i-th coordinate of G is equal to the average of the
        #b = n_clients - 4 * cmax closest i-th coordinates to the median i-th coordinate of the t selected gradients

        with torch.no_grad():
            idx = 0
            for _, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param_data = torch.zeros(param[1].nelement())
                    for i in range (param[1].nelement()) :
                        #compute the median for each coordinate
                        selected_coordinates = []
                        for selected_grad in selected_set :
                            selected_grad = int(selected_grad.item())
                            selected_coordinates.append(self.lr*params[selected_grad][idx+i].item())
                        median_coordinate = statistics.median(selected_coordinates)

                        #select the b closest coordinates
                        b = self.n_clients - 4 * self.cmax
                        final_coordinates = []
                        for selected_coordinate in selected_coordinates :
                            if len(final_coordinates) < b :
                                final_coordinates.append(selected_coordinate - median_coordinate)
                            else :
                                max_diff = max(final_coordinates, key=abs)
                                if abs(selected_coordinate - median_coordinate) < abs(max_diff) :
                                    final_coordinates.remove(max_diff)
                                    final_coordinates.append(selected_coordinate - median_coordinate)

                        #compute the mean over the b coordinates
                        final_mean_coordinate = statistics.mean(final_coordinates) + median_coordinate
                        param_data[i] = final_mean_coordinate
                    param[1].data += param_data.reshape(param[1].shape)  
                    idx += param[1].nelement()                  
        return net 
    
    def foolsgold(self, net) : 
        params = self.grads
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

        alphas = alphas / torch.max(alphas).item()
        alphas[(alphas == 1)] = .99
        
        alphas = (torch.log(alphas / (1 - alphas)) + 0.5)
        alphas[(torch.isinf(alphas) + alphas > 1)] = 1
        alphas[(alphas < 0)] = 0

        aggregated_params = torch.matmul(torch.transpose(params, 0, 1), alphas.reshape(-1,1))
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += aggregated_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()  
        del params, aggregated_params
        return net
        
    def faba(self, net):  
        params = self.grads
        params_faba = copy.copy(params)
        #faba algo : cmax iters, take out client furthest away from mean
        for _ in range (self.cmax) :
            g0 = torch.mean(params_faba, dim=0)
            dist_g0 = torch.zeros(len(params_faba))
            for i in range (len(params_faba)) :
                dist_g0[i] = torch.norm(g0 - params_faba[i]).item()
            client_max_dist = int(torch.argmax(dist_g0).item())
            params_faba = torch.cat([params_faba[0:client_max_dist], params_faba[client_max_dist+1:]])
        #the aggregated gradient
        g0 = torch.mean(params_faba, dim=0)

        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += g0[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement()

        return net 

    def tesseract(self, net, prev_direction, reputation) : 
        #params = self.grads.reshape(0,-1) #n_clients*grad_size
        params = self.grads

        flip_clients = torch.zeros(self.n_clients) #flip-score vector

        prev_direction = prev_direction.view(-1)

        #Computing flip-score
        for i in range (self.n_clients) :
            client_compare = torch.eq(torch.sign(params[i]), torch.sign(prev_direction)) #where the client goes in the same direction as the model
            flip_clients[i] = torch.matmul(torch.square(params[i]), torch.where(client_compare, 0, 1).type(torch.FloatTensor))
        
        #Update the reputation scores
        penalty = 1.0 - 2*self.cmax/len(params) 
        reward = 2*self.cmax/len(params)
        argsorted = torch.argsort(flip_clients)
        if (self.cmax > 0):
            reputation[argsorted[self.cmax:-self.cmax]] = reputation[argsorted[self.cmax:-self.cmax]] + reward
            reputation[argsorted[:self.cmax]] = reputation[argsorted[:self.cmax]] - penalty
            reputation[argsorted[-self.cmax:]] = reputation[argsorted[-self.cmax:]] - penalty  
        argsorted = torch.argsort(reputation)

        #Normalize reputations weights
        weights = torch.exp(reputation)/torch.sum(torch.exp(reputation))

        #Aggregate gradients
        tesseract_params = torch.matmul(torch.transpose(params, 0, 1), weights.reshape(-1,1))

        #Update global direction
        global_direction = torch.sign(tesseract_params)

        #updating parameters
        with torch.no_grad():
            idx = 0
            for j, (param) in enumerate(net.named_parameters()):
                if param[1].requires_grad:
                    param[1].data += tesseract_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                    idx += param[1].nelement() 
        
        del params, tesseract_params

        return net, global_direction, reputation
