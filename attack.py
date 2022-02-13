import torch 
import torch.nn as nn 
import numpy as np
from copy import deepcopy
import math

### In all attacks, we suppose that the c_max first indexes refers to the parameters of malicious clients.

def full_trim_attack(lr, params, c_max):
    if c_max == 0:
        return params
    final_params = deepcopy(params)
    direction = torch.sign(torch.sum(-params, axis=0))
    min_param = torch.min(-params, axis=0)
    max_param = torch.max(-params, axis=0)
    param_directions = (direction > 0) * min_param + (direction < 0) * max_param
    for i in range(c_max) :
        noise = 1 + torch.rand(params.shape[0])
        final_params[i] = -(param_directions * ((direction * param_directions > 0) * noise + (direction * param_directions < 0) * noise))
    return final_params

def adaptive_trim_attack(lr, params, c_max, fs_max, fs_min, old_direction, n):
    return None

def adaptive_krum_attack(le, params, c_max, fs_max, fs_min, old_direction, n):
    return None

def lambda_max(c, params, device):
    '''
    Compute noisy lambda
    :param c: number of malicious clients
    :param params:
    :return: mxm matrix of lambda
    '''
    n = params.shape[0]
    dist = torch.zeros((n, n))
    for i in range(dist.shape[0]):
        for j in range(i):
            dist[i][j] = torch.norm(params[i] - params[j])
            dist[j][i] = dist[i][j]
    sane_dist = dist[c:][c:]
    min_sane_dist = torch.min(sane_dist, axis=0).item()
    global_dist = torch.zeros(n-c).to(device)
    #On calque la distribution des gradients pour les clients malicious sur celle des clients sains
    for i in range(c, n):
        global_dist[i-c] = torch.norm(params[i])
    max_global_dist = torch.max(global_dist).item()
    scale = 1.0/(len(params[0]))
    # On concatène les deux matrices de distribution
    return (math.sqrt(scale/(n-2*c-1))*min_sane_dist) + math.sqrt(scale)*max_global_dist

def find_lambda():
    return None

def full_krum_attack(lr, params, c_max):
    if c_max == 0:
        return params
    final_params = deepcopy(params)
    noise_coef = 0.0001/c_max
    direction = torch.sign(torch.sum(-params, axis=0))
    lambda_mx = lambda_max(c_max, params)
    final_lambda = find_lambda()
    if (final_lambda>0):
        final_params[0] = -(direction*final_lambda)
        for i in range(c_max):
            noise = torch

    return None





