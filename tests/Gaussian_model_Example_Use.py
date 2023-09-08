import numpy as np
from abcpy.approx_lhd import SynLikelihood, EnergyScore, KernelScore
from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal, LogNormal
from abcpy.inferences import adSGLD, SGLD
from abcpy.statistics import Identity
from Gaussian_model import Gaussian

# setup backend
dummy = BackendDummy()

# define a uniform prior distribution
mu = Normal([5, 1], name='mu')
sigma = LogNormal([1,1], name='sigma')
model = Gaussian([mu, sigma])

stat_calc = Identity(degree=2, cross=False)

dist_calc = EnergyScore(stat_calc, model, 1)

y_obs = model.forward_simulate([6,1], 100, rng=np.random.RandomState(8))  # Correct

sampler = adSGLD([model], [dist_calc], dummy, seed=1)

journal = sampler.sample([y_obs], 100, 100, 2000, step_size=0.0001, w_val = 300, diffusion_factor=0.01, path_to_save_journal="tmp.jnl")

journal.plot_posterior_distr(path_to_save="posterior.png")
journal.traceplot()
