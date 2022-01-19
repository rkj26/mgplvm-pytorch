import numpy as np
import copy
import torch
from .train_model import train_model
from .crossval import not_in, update_params, test_cv
from ..manifolds import Euclid
from ..likelihoods import Gaussian, NegativeBinomial, Poisson
from ..rdist import GP_circ, GP_diag
from ..lpriors import Null
from ..models import Lvgplvm, Lgplvm


def train_cv_bgpfa_trials(Y,
             device,
             train_ps,
             fit_ts,
            d_fit,
            ell,
            train_trials,
            lat_scale = 1,
            rel_scale = 1,
            likelihood = 'Gaussian',
            model = 'bgpfa',
            ard = True,
            Bayesian = True):
    
    
    _, n, m = Y.shape

    Y1 = Y[:train_trials, :, :]
    
    ##### fit the first model!!!! ####
    n_samples, n, T = Y1.shape
    
    manif = Euclid(T, d_fit)
    lprior = Null(manif)
    lat_dist = GP_circ(manif, T, n_samples, fit_ts, _scale=lat_scale, ell = ell) #initial ell ~200ms
    
    
    if model in ['bgpfa', 'bGPFA', 'gpfa', 'GPFA"']: ###Bayesian GPFA!
        if likelihood == 'Gaussian':
            lik = Gaussian(n, Y=Y1, d = d_fit)
        elif likelihood == 'NegativeBinomial':
            lik = NegativeBinomial(n, Y=Y1)
        elif likelihood == 'Poisson':
            #print('poisson lik')
            lik = Poisson(n)
        
        mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, ard = ard, learn_scale = (not ard), Y = Y1, rel_scale = rel_scale,
                     Bayesian = Bayesian).to(device)
    
    train_model(mod, torch.tensor(Y1).to(device), train_ps) ###initial training####
    
    ### fit second model and copy over parameters ###
    Y2 = Y
    n_samples, n, T = Y2.shape
    
    ###rdist: ell
    manif = Euclid(T, d_fit)
    lprior = Null(manif)
    ell0 = mod.lat_dist.ell.detach().cpu()
    lat_dist = GP_circ(manif, T, n_samples, fit_ts, _scale=lat_scale, ell = ell0)
    
    if model in ['bgpfa', 'bGPFA', 'gpfa', 'GPFA']: ###Bayesian GPFA!!!
        if likelihood == 'Gaussian':
            ###lik: sigma
            sigma = mod.obs.likelihood.sigma.detach().cpu()
            lik = Gaussian(n, sigma = sigma)
        elif likelihood == 'NegativeBinomial':
            #lik: c, d, total_count
            c, d, total_count = [val.detach().cpu() for val in [mod.obs.likelihood.c, mod.obs.likelihood.d, mod.obs.likelihood.total_count]]
            lik = NegativeBinomial(n, c=c, d=d, total_count=total_count)
        elif likelihood == 'Poisson':
            #print('poisson lik')
            c, d = [val.detach().cpu() for val in [mod.obs.likelihood.c, mod.obs.likelihood.d]]
            lik = Poisson(n, c = c, d = d)
        
        if Bayesian:
            #print('bayesian')
            ###obs: q_mu, q_sqrt, _scale, _dim_scale, _neuron_scale
            q_mu, q_sqrt = mod.obs.q_mu.detach().cpu(), mod.obs.q_sqrt.detach().cpu()
            scale, dim_scale, neuron_scale = mod.obs.scale.detach().cpu(), mod.obs.dim_scale.detach().cpu().flatten(), mod.obs.neuron_scale.detach().cpu().flatten()
            mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, ard = ard, learn_scale = (not ard),
                        q_mu = q_mu, q_sqrt = q_sqrt, scale = scale, dim_scale = dim_scale, neuron_scale = neuron_scale,
                         Bayesian = True).to(device)
        
        else:
            #print('not bayesian')
            ###obs: C
            lat_C = mod.obs.C.detach().cpu()
            mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, lik, C = lat_C, Bayesian = False).to(device)
    
    torch.cuda.empty_cache
    
    for p in mod.parameters():  #no gradients for the remaining parameters
        p.requires_grad = False

    mod.lat_dist._nu.requires_grad = True #latent variational mean
    mod.lat_dist._scale.requires_grad = True #latent variational covariance
    if 'circ' in mod.lat_dist.name:
        mod.lat_dist._c.requires_grad = True #latent variational covariance

    train_ps2 = update_params(train_ps, max_steps = int(round(train_ps['max_steps'])))
    train_model(mod, torch.tensor(Y2).to(device), train_ps2)

    return mod
