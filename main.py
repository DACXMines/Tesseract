#import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import gc
import copy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from dataset import DataPipeline
from defense import DefenseSystem
from model import ClientModel



def load_conf(path) :
    with open(path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)    

def main(config) :
    nb_clients = config['nb_clients']
    n_epochs = config["n_epochs"]
    cmax = config["cmax"]
    writer = SummaryWriter(log_dir=f"dataset_"+config["dataset_name"]+"_attack_"+config["attack"]+
        "_defense_"+config["aggregation_model"]+"_cmax_"+str(cmax)+"_nbclients_"+str(nb_clients))

    #Data : load and distribute amongst clients
    data_pipeline = DataPipeline(config)
    train_dataloader = data_pipeline.train_dataloader
    client_train_data = [[] for _ in range(nb_clients)]
    client_train_label = [[] for _ in range(nb_clients)] 
    for _, (data, label) in enumerate(train_dataloader):
        for (x, y) in zip(data, label):
            selected_worker = np.random.randint(nb_clients)
            client_train_data[selected_worker].append(x)
            client_train_label[selected_worker].append(y)
    # concatenate the data for each clent
    client_train_data = [(torch.stack(client_data, dim=0)).squeeze(0) for client_data in client_train_data] 
    client_train_label = [(torch.stack(client_label, dim=0)).squeeze(0) for client_label in client_train_label] 

    if config["attack"] == "data_poisoning" :
        for i in range (cmax) :
            idx = torch.randperm(client_train_label[i].nelement())
            client_train_label[i] = client_train_label[i].view(-1)[idx].view(client_train_label[i].size())

    test_dataloader = data_pipeline.test_dataloader

    #For FEDSGD
    wts = torch.zeros(len(client_train_data))
    for i in range(len(client_train_data)):
        wts[i] = len(client_train_data[i])
    wts = wts/torch.sum(wts)

    #Attack
    attack = Attack(config)

    #Models
    global_model = ClientModel(config, "global")
    
    #Evaluation of the model
    lr = config["lr"]
    batch_size = config["batch_size"]
    criterion = nn.CrossEntropyLoss()
    P = 0
    P_global = 0
    for param in global_model.model.parameters():
        if param.requires_grad:
            P = P + param.nelement()
            P_global += 1
    direction = torch.zeros(P)
    reputation = torch.zeros(nb_clients)

    #Train & test  
    test_acc = np.empty(n_epochs)
    batch_idx = np.zeros(nb_clients)
    for epoch in range(n_epochs):
        grad_list = []
        for client_id in range(nb_clients):
            local_model = copy.deepcopy(global_model.model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            optimizer.zero_grad()
            
            #sample local dataset in a round-robin manner
            if (batch_idx[client_id]+batch_size < client_train_data[client_id].shape[0]):
                minibatch = np.asarray(list(range(int(batch_idx[client_id]),int(batch_idx[client_id])+batch_size)))
                batch_idx[client_id] = batch_idx[client_id] + batch_size
            else: 
                minibatch = np.asarray(list(range(int(batch_idx[client_id]),client_train_data[client_id].shape[0]))) 
                batch_idx[client_id] = 0


            output = local_model(client_train_data[client_id][minibatch])
            loss = criterion(output, client_train_label[client_id][minibatch])
            loss.backward()
            optimizer.step()
                    
            #append all gradients in a list
            grad_list.append([(x-y).detach() for x, y in zip(local_model.parameters(), global_model.model.parameters()) if x.requires_grad != 'null'])

        
        #Attack
        grads = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
        grads = attack.attack_grads(grads)

        #Aggregation
        aggregation = DefenseSystem(grads, lr, cmax)
        if config["aggregation_model"] == "tesseract" :
            global_model.model, direction, reputation = aggregation.tesseract(global_model.model, direction, reputation)
        elif config["aggregation_model"] == "fed_sgd" :
            global_model.model = aggregation.fed_sgd(global_model.model, wts)
        else :
            global_model.model = aggregation.aggregate(config, global_model.model)
        
        del grads, grad_list
        gc.collect()
    
        #Test
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_dataloader:
                outputs = global_model.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc[epoch] = correct/total                
            print ('Iteration: %d, test_acc: %f' %(epoch, test_acc[epoch]))

        writer.add_scalar('test/accuracy', test_acc[epoch], epoch)


        for i in range (cmax) :
            writer.add_scalar('reputation/attacker_'+str(i), reputation[i], epoch)
        for i in range (cmax, nb_clients) :
            writer.add_scalar('reputation/client_'+str(i), reputation[i], epoch)
        writer.add_scalar('test/accuracy', test_acc[epoch], epoch)
        
if __name__ == '__main__' :
    config = load_conf("config.yaml")
    main(config)