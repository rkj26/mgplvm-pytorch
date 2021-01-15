import mgplvm
from mgplvm import lpriors, kernels, models
from mgplvm.manifolds import Euclid, Torus
import torch
import pickle
import numpy as np


def model_params(n, m, d, n_z, **kwargs):
    
    params = {
        'n': n, 'm': m, 'd': d, 'n_z': n_z,
        'manifold': 'euclid',
        'kernel': 'RBF',
        'prior': 'Uniform',
        'likelihood': 'Gaussian',
        'initialization': 'pca',
        'Y': None,
        'latent_sigma': 1,
        'diagonal': True,
        'learn_linear_weights': False,
        'learn_linear_alpha': True,
        'RBF_alpha': None,
        'RBF_ell': None,
        'arp_p': 1,
        'arp_eta': np.ones(d)*0.3,
        'arp_learn_eta': True,
        'arp_learn_c': False,
        'arp_learn_phi': True,
        'lik_gauss_std': None,
        'device': None
    }
    
    for key, value in kwargs.items():
        params[key] = value
    
    return params

def load_model(params):

    likelihoods = {'GP': lpriors.GP}
    
    n, m, d, n_z = params['n'], params['m'], params['d'], params['n_z']
    
    #### specify manifold ####
    if params['manifold'] == 'euclid':
        manif = Euclid(m, d, initialization = params['initialization'], Y = params['Y'][:, :, 0])
    elif params['manifold'] == 'torus':
        manif = Torus(m, d, initialization = params['initialization'], Y = params['Y'][:, :, 0])
        
    #### specify latent distribution ####
    lat_dist = mgplvm.rdist.ReLie(manif, m, sigma=params['latent_sigma'], diagonal = params['diagonal'])
    
    #### specify kernel ####
    if params['kernel'] == 'linear':
        kernel = kernels.Linear(n, manif.linear_distance, d, learn_weights = params['learn_linear_weights'],
                                learn_alpha = params['learn_linear_alpha'], Y = params['Y'])
    elif params['kernel'] == 'RBF':
        ell = None if params['RBF_ell'] is None else np.ones(n)*params['RBF_ell']
        kernel = kernels.QuadExp(n, manif.distance, Y = params['Y'],
                                 alpha = params['RBF_alpha'], ell = ell)
        
    #### speciy prior ####
    if params['prior'] == 'GP':
        lprior_kernel = kernels.QuadExp(d, manif.distance, learn_alpha = False, ell = np.ones(n)*m/20)
        lprior = lpriors.GP(manif, lprior_kernel, n_z = n_z, tmax = m)
    elif params['prior'] == 'ARP':
        lprior = lpriors.ARP(params['arp_p'], manif, ar_eta = torch.tensor(params['arp_eta']),
                         learn_eta = params['arp_learn_eta'], learn_c = params['arp_learn_c'])
    else:
        lprior = lpriors.Uniform(manif)

    #### specify likelihood ####
    if params['likelihood'] == 'Gaussian':
        likelihood = mgplvm.likelihoods.Gaussian(n, variance=np.square(params['lik_gauss_std']))
    elif params['likelihood'] == 'Poisson':
        likelihood = mgplvm.likelihoods.Poisson(n)
    elif params['likelihood'] == 'NegBinom':
        likelihood = mgplvm.likelihoods.NegativeBinomial(n)
        
    #### specify inducing points ####
    z = manif.inducing_points(n, n_z)
    
    #### construct model ####
    device = (mgplvm.utils.get_device() if params['device'] is None else params['device'])
    mod = models.SvgpLvm(n,
                     z,
                     kernel,
                     likelihood,
                     lat_dist,
                     lprior).to(device)
    
    return mod
    
    
    