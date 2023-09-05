# Testing PDMP

#from src.models.normal_model import normal
from Gaussian_model import Gaussian
#from src.scoring_rules.scoring_rules import EnergyScore
from abcpy.approx_lhd import EnergyScore 
#from src.pdmp.boomerang import boomerang_sampler, boomerang_sampler_gibbs
from abcpy.inferences import boomerang_sampler_gibbs
import tqdm as tqdm
# from src.sampler.sgMCMC import SGMCMC
import torch
import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_sparse_coded_signal
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from jax import random
import functools

torch.set_default_dtype(torch.float64) #Necessary for numerical precision

# Set seeds
torch.manual_seed(0)


y, X, beta = make_sparse_coded_signal(n_samples=1,
                                   n_components=5,
                                   n_features=100,
                                   n_nonzero_coefs=2,
                                   random_state=0)


# add some noise to y
y = y + 0.1 * np.random.randn(len(y))

def gradient_fn(theta, y , x):
    #theta: tuple pair of gibbs and non gibbs variable (x, alpha)
    #alpha: (tau, lambda_1,.., lambda_p) for a (p+1)-dim vector
    #betas: (beta_1,.., beta_p) for a p-dim vector()
    betas, alpha = theta
    alpha = torch.tensor(alpha)
    betas = torch.tensor(betas, requires_grad=True)
    x = torch.tensor(x)
    y = torch.tensor(y)

    sigma_2 = .1 ** 2
    sigma_2_inv = 1 / sigma_2

    sigma_beta = sigma_2 * alpha[0] ** 2 * torch.diag(alpha[1:] ** 2)
    sigma_beta_inv = torch.inverse(sigma_beta)


    # Compute potential fn U
    U = 0.5 * sigma_2_inv * torch.t(y - x @ betas) @ (y - x @ betas) + 0.5 * torch.t(betas) @ sigma_beta_inv @ betas
    #U = 0.5 * sigma_2_inv * (y - x @ betas).T @ (y - x @ betas) + 0.5 * (betas).T @ sigma_beta_inv @ betas
    U.backward()

    # Compute gradient
    grad = betas.grad.detach().numpy() - theta[0]
    
    return grad

grad_func = functools.partial(gradient_fn, y=y, x=X)


def gibbs_fn(theta):
    #theta: tuple pair of gibbs and non gibbs variable (x, alpha)
    betas, alpha = theta
    tau = alpha[0]
    lambdas = alpha[1:]
    sigma = 0.1

    new_tau = draw_tau(lambdas, betas, sigma, tau)
    new_lambda = draw_lambda(lambdas, betas, sigma, new_tau)
    
    return np.hstack([new_tau, new_lambda])

import numpy as np
from scipy.stats import gamma, uniform, expon

def draw_tau(lambda_, beta, sigma, tau):
    shape_tau = 0.5 * (len(lambda_) + 1)
    gamma_t = 1 / tau**2
    u1 = uniform.rvs(0, 1 / (1 + gamma_t), size=1)
    trunc_limit_tau = (1 - u1) / u1
    mu2_tau = np.sum((beta / (sigma * lambda_))**2)
    rate_tau = (mu2_tau / 2)
    ub_tau = gamma.cdf(trunc_limit_tau, shape_tau, scale=1/rate_tau)
    u2 = uniform.rvs(0, ub_tau, size=1)
    gamma_t = gamma.ppf(u2, shape_tau, scale=1/rate_tau)
    return 1 / np.sqrt(gamma_t)

def draw_lambda(lambda_, beta, sigma, tau):
    gamma_l = 1 / lambda_**2
    u1 = uniform.rvs(0, 1 / (1 + gamma_l), size=len(lambda_))
    trunc_limit = (1 - u1) / u1
    mu2_j = (beta / (sigma * tau))**2
    rate_lambda = (mu2_j / 2)
    ub_lambda = expon.cdf(trunc_limit, scale=1/rate_lambda)
    u2 = uniform.rvs(0, ub_lambda, size=len(ub_lambda))
    gamma_l = expon.ppf(u2, scale=1/rate_lambda)
    return 1 / np.sqrt(gamma_l)

initial_pos = (np.ones(5), np.ones(6))
sampler = boomerang_sampler_gibbs(sigma_ref=np.eye(5), mu_ref=np.zeros(5), gradient=grad_func, initial_pos=initial_pos, niter=10000, lr=1.0, initial_t_max_guess=1.0, seed=0, noisy_gradient=False, d=5, q=0.9, gibbs_sampler=gibbs_fn, gibbs_ref =1.)
sampler.sample()


