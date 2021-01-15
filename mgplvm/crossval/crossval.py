import os
import numpy as np
import copy
import mgplvm
import torch
from mgplvm import kernels, rdist, models, training
from mgplvm.manifolds import Torus, Euclid, So3
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_1samp
from . import train_model, load_model
torch.set_default_dtype(torch.float64)

def not_in(arr, inds):
    mask = np.ones(arr.size, dtype=bool)
    mask[inds] = False
    return arr[mask]

def update_params(params, **kwargs):
    newps = copy.copy(params)
    for key, value in kwargs.items():
        newps[key] = value
    return newps

def train_cv(Y,
            device,
            model_ps,
            train_ps,
            T1 = None,
            N1 = None,
            nt_train = None,
            nn_train = None,
            test = True):
    """
    Parameters
    ----------
    Y : array
        data with dimensionality (n x m x n_samples)
    device : torch.device
        GPU/CPU device on which to run the calculations
    model_ps : dict
        dictionary of model parameters. Constructed by crossval.model_params()
    train_ps : dict
        dictionary of training parameters. Constructed by crossval.training_params()
    T1 [optional] : int list
        indices of the conditions to use for training
    N1 [optional] : int list
        indices of the neurons to use for training
    nt_train [optional] : int
        number of randomly selected conditions to use for training
    nn_train [optional] : int
        number of randomly selected neurons to use for training

    Returns
    -------
    mod : mgplvm.svgplvm
        model trained via crossvalidation

    """

    n, m = Y.shape[:2]
    nt_train = int(round(m/2)) if nt_train is None else nt_train
    nn_train = int(round(n/2)) if nn_train is None else nn_train
    
    if T1 is None: # random shuffle of timepoints
        T1 = np.random.permutation(np.arange(m))[:nt_train]
    if N1 is None: # random shuffle of neurons
        N1 = np.random.permutation(np.arange(n))[:nn_train]
    Y1, Y2 = Y[:, T1, :], Y[N1, :, :]
    split = {'Y': Y, 'N1': N1, 'T1': T1}
    
    mod = load_model(model_ps)
    train_ps1 = update_params(train_ps, batch_pool = T1)
    mod = train_model(mod, Y, device, train_ps1)
    
    ### construct a mask for some of the time points ####
    def hook(grad):
        ''' used to 'mask' some gradients for cv'''
        grad[T1, ...] *= 0
        return grad
    train_ps2 = update_params(train_ps, neuron_idxs = N1, hook = hook)
    
    #### no gradients for the remaining parameters ####
    for p in mod.parameters():
        if not (((p.shape == mod.lat_dist.gamma.shape) and torch.all(p == mod.lat_dist.gamma))
                or
                ((p.shape == mod.lat_dist.manif.mu.shape) and torch.all(p == mod.lat_dist.manif.mu))):
            p.requires_grad = False

    mod = train_model(mod, Y, device, train_ps2)
    
    if test:
        test_cv(mod, split, device, n_mc = train_ps['n_mc'])
        
    return mod, split
        


def test_cv(mod, split, device, n_mc = 32):
    Y, T1, N1 = split['Y'], split['T1'], split['N1']
    n, m = Y.shape[:2]
    
    ##### assess the CV quality ####
    T2, N2 = not_in(np.arange(m), T1), not_in(np.arange(n), N1)

    #generate prediction for held out data#
    Ytest = Y[N2, :, 0][:, T2]
    latents = mod.lat_dist.manif.prms.detach()[T2, ...]
    Ypred, var = mod.svgp.predict(latents.T[None, ...], False)
    Ypred = Ypred.detach().cpu().numpy()[0, N2, :, 0]
    MSE = np.mean((Ypred - Ytest)**2)
    print('MSE', MSE, np.sqrt(np.mean(np.var(Ytest, axis = 1))))

    #/((m-nt_train)*(n-nn_train)

    ### compute crossvalidated log likelihood ###
    svgp_elbo, kl = mod.elbo(torch.tensor(Y).to(device), n_mc, batch_idxs=T2, neuron_idxs = N2)
    
    svgp_elbo = svgp_elbo.sum(-1).sum(-1) #(n_mc)
    LLs = svgp_elbo - kl  # LL for each batch (n_mc, )
    LL = (torch.logsumexp(LLs, 0) - np.log(n_mc)).detach().cpu().numpy()
    LL = LL/(len(T2)*len(N2))
    
    
    print('LL', LL)
    
    
    return MSE, LL
    
    