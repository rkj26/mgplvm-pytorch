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



def train_cv_behavior(Y,
             device,
             train_ps,
             fit_ts,
            d_fit,
            ell,
            T1,
            N1,
            lat_scale = 1,
            rel_scale = 1,
            ard = True,
            Bayesian = True):
    
    
    #print('training')
    
    _, n, m = Y.shape
    
    ##### fit the first model!!!! ####
    Y1 = Y[..., T1]
    n_samples, n, T = Y1.shape
    y, b = Y1[:, :-2, :], Y1[:, -2:, :]
    manif = Euclid(T, d_fit)
    lprior = Null(manif)
    lat_dist = GP_circ(manif, T, n_samples, fit_ts[..., T1], _scale=lat_scale, ell = ell) #initial ell ~200ms
    behaviour_lik = Gaussian(b.shape[-2], Y = b, d = d_fit)
    spike_lik = NegativeBinomial(y.shape[-2], Y=y)

        
    mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, [spike_lik, behaviour_lik], ard = ard, learn_scale = (not ard), Y = Y1, rel_scale = rel_scale, Bayesian = Bayesian).to(device)
    
    train_model(mod, torch.tensor(Y1).to(device), train_ps) ###initial training####
    
    ### fit second model and copy over parameters ###
    Y2 = Y
    n_samples, n, T = Y2.shape
    
    ###rdist: ell
    manif = Euclid(T, d_fit)
    lprior = Null(manif)
    ell0 = mod.lat_dist.ell.detach().cpu()
    lat_dist = GP_circ(manif, T, n_samples, fit_ts, _scale=lat_scale, ell = ell0)
    b_sigma = mod.obs.behavior_likelihood.sigma.detach().cpu()
    behaviour_lik = Gaussian(b.shape[-2], sigma=b_sigma)

    c, d, total_count = [val.detach().cpu() for val in [mod.obs.spike_likelihood.c, mod.obs.spike_likelihood.d, mod.obs.spike_likelihood.total_count]]
    spike_lik = NegativeBinomial(n, c=c, d=d, total_count=total_count)

    q_mu, q_sqrt = mod.obs.q_mu.detach().cpu(), mod.obs.q_sqrt.detach().cpu()
    scale, dim_scale, neuron_scale = mod.obs.scale.detach().cpu(), mod.obs.dim_scale.detach().cpu().flatten(), mod.obs.neuron_scale.detach().cpu().flatten()
    mod = Lvgplvm(n, T, d_fit, n_samples, lat_dist, lprior, [spike_lik, behaviour_lik], ard = ard, learn_scale = (not ard),
                q_mu = q_mu, q_sqrt = q_sqrt, scale = scale, dim_scale = dim_scale, neuron_scale = neuron_scale,
                    Bayesian = True).to(device)
        
    torch.cuda.empty_cache
    
    for p in mod.parameters():  #no gradients for the remaining parameters
        p.requires_grad = False

    mod.lat_dist._nu.requires_grad = True #latent variational mean
    mod.lat_dist._scale.requires_grad = True #latent variational covariance
    if 'circ' in mod.lat_dist.name:
        mod.lat_dist._c.requires_grad = True #latent variational covariance

    train_ps2 = update_params(train_ps, neuron_idxs=N1, max_steps = int(round(train_ps['max_steps'])))
    train_model(mod, torch.tensor(Y2).to(device), train_ps2)


    return mod