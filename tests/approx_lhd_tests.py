import unittest

import numpy as np

from abcpy.approx_lhd import EnergyScore # PenLogReg, SynLikelihood, SemiParametricSynLikelihood, EnergyScore
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity
from Gaussian_model import Gaussian


class PenLogRegTests(unittest.TestCase):
    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.model_bivariate = Uniform([[0, 0], [1, 1]], name="model")
        self.stat_calc = Identity(degree=2, cross=1)
        self.likfun = PenLogReg(self.stat_calc, [self.model], n_simulate=100, n_folds=10, max_iter=100000, seed=1)
        self.likfun_wrong_n_sim = PenLogReg(self.stat_calc, [self.model], n_simulate=10, n_folds=10, max_iter=100000,
                                            seed=1)
        self.likfun_bivariate = PenLogReg(self.stat_calc, [self.model_bivariate], n_simulate=100, n_folds=10,
                                          max_iter=100000, seed=1)

        self.y_obs = self.model.forward_simulate(self.model.get_input_values(), 1, rng=np.random.RandomState(1))
        self.y_obs_bivariate = self.model_bivariate.forward_simulate(self.model_bivariate.get_input_values(), 1,
                                                                     rng=np.random.RandomState(1))
        self.y_obs_double = self.model.forward_simulate(self.model.get_input_values(), 2, rng=np.random.RandomState(1))
        self.y_obs_bivariate_double = self.model_bivariate.forward_simulate(self.model_bivariate.get_input_values(), 2,
                                                                            rng=np.random.RandomState(1))
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))
        self.y_sim_bivariate = self.model_bivariate.forward_simulate(self.model_bivariate.get_input_values(), 100,
                                                                     rng=np.random.RandomState(1))

    def test_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

        # create observed data
        comp_likelihood = self.likfun.loglikelihood(self.y_obs, self.y_sim)
        expected_likelihood = 9.77317308598673e-08
        # This checks whether it computes a correct value and dimension is right. Not correct as it does not check the
        # absolute value:
        # self.assertLess(comp_likelihood - expected_likelihood, 10e-2)
        self.assertAlmostEqual(comp_likelihood, np.log(expected_likelihood))

        # check if it returns the correct error when n_samples does not match:
        self.assertRaises(RuntimeError, self.likfun_wrong_n_sim.loglikelihood, self.y_obs, self.y_sim)

        # try now with the bivariate uniform model:
        comp_likelihood_biv = self.likfun_bivariate.loglikelihood(self.y_obs_bivariate, self.y_sim_bivariate)
        expected_likelihood_biv = 0.999999999999999
        self.assertAlmostEqual(comp_likelihood_biv, np.log(expected_likelihood_biv))

    def test_likelihood_multiple_observations(self):
        comp_likelihood = self.likfun.likelihood(self.y_obs_double, self.y_sim)
        expected_likelihood = 7.337876253225462e-10
        self.assertAlmostEqual(comp_likelihood, expected_likelihood)

        expected_likelihood_biv = 0.9999999999999979
        comp_likelihood_biv = self.likfun_bivariate.likelihood(self.y_obs_bivariate_double, self.y_sim_bivariate)
        self.assertAlmostEqual(comp_likelihood_biv, expected_likelihood_biv)

    def test_loglikelihood_additive(self):
        comp_loglikelihood_a = self.likfun.loglikelihood([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.loglikelihood([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.loglikelihood(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)


class SynLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.stat_calc = Identity(degree=2, cross=False)
        self.likfun = SynLikelihood(self.stat_calc)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))

    def test_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]
        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -0.6434435652263701
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -1.2726154993040115
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.loglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.loglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.loglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)


class SemiParametricSynLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.stat_calc_1 = Identity(degree=1, cross=False)
        self.likfun_1 = SemiParametricSynLikelihood(self.stat_calc_1)
        self.stat_calc = Identity(degree=2, cross=False)
        self.likfun = SemiParametricSynLikelihood(self.stat_calc)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))

    def test_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]

        # check whether it raises correct error with input of wrong size
        self.assertRaises(RuntimeError, self.likfun_1.loglikelihood, y_obs, self.y_sim)

        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -2.3069321875272815
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -3.7537571275591683
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.loglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.loglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.loglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)


class EnergyScoreTests(unittest.TestCase):
    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.stat_calc_1 = Identity(degree=1, cross=False)
        self.likfun_1 = EnergyScore(self.stat_calc_1, self.model, 1.5)
        self.stat_calc = Identity(degree=2, cross=False)
        self.likfun = EnergyScore(self.stat_calc, self.model, 1.5)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))
        # print(self.y_sim)

    # def test_likelihood(self):
    #     # Checks whether wrong input type produces error message
    #     self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
    #     self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

    #     # create observed data
    #     y_obs = self.model.forward_simulate([7, 1], 50, rng=np.random.RandomState(1)) #[1.8]

    #     # check whether it raises correct error with input of wrong size
    #     #self.assertRaises(RuntimeError, self.likfun_1.loglikelihood, y_obs, self.y_sim)

    #     # calculate the statistics of the observed data
    #     for i in range(0,80):
    #         self.y_sim = self.model.forward_simulate([5 + 0.05*i, 1], 50, rng=np.random.RandomState(1))
    #         self.y_sim_grad = self.model.grad_forward_simulate([5 + 0.05*i, 1], 50, rng=np.random.RandomState(1))
    #         # print(self.y_sim)
    #         comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
    #         grad_comp_loglikelihood = self.likfun.gradloglikelihood(y_obs, self.y_sim_grad)
    #         print(str(comp_loglikelihood) + str(" <---- loglik ") + str(5 + 0.05*i))
    #         print(str(grad_comp_loglikelihood) + str(" <---- gradloglik ") + str(5 + 0.05*i))
    #     expected_loglikelihood = -2.3069321875272815
    #     # This checks whether it computes a correct value and dimension is right
    #     self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]

        # check whether it raises correct error with input of wrong size
        self.assertRaises(RuntimeError, self.likfun_1.loglikelihood, y_obs, self.y_sim)

        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -2.3069321875272815                                            # Change this value
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -3.7537571275591683
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.loglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.loglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.loglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

    def test_grad_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.gradloglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.gradloglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]

        # check whether it raises correct error with input of wrong size
        self.assertRaises(RuntimeError, self.likfun_1.gradloglikelihood, y_obs, self.y_sim)

        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.gradloglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -2.3069321875272815                                            # Change this value
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_grad_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.gradloglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -3.7537571275591683
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_grad_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.gradloglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.gradloglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.gradloglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

class KernelScoreTests(unittest.TestCase):
    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.stat_calc_1 = Identity(degree=1, cross=False)
        self.beta = 1.5
        self.likfun_1 = KernelScore(self.stat_calc_1, self.model, self.BetaNorm)
        self.stat_calc = Identity(degree=2, cross=False)
        self.likfun = KernelScore(self.stat_calc, self.model, self.BetaNorm)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))
        # print(self.y_sim)

    def test_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.loglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.loglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]

        # check whether it raises correct error with input of wrong size
        self.assertRaises(RuntimeError, self.likfun_1.loglikelihood, y_obs, self.y_sim)

        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -2.3069321875272815                                            # Change this value
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.loglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -3.7537571275591683
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.loglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.loglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.loglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

    def test_grad_likelihood(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.gradloglikelihood, 3.4, [2, 1])
        self.assertRaises(TypeError, self.likfun.gradloglikelihood, [2, 4], 3.4)

        # create observed data
        y_obs = [1.8]

        # check whether it raises correct error with input of wrong size
        self.assertRaises(RuntimeError, self.likfun_1.gradloglikelihood, y_obs, self.y_sim)

        # calculate the statistics of the observed data
        comp_loglikelihood = self.likfun.gradloglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -2.3069321875272815                                            # Change this value
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_grad_likelihood_multiple_observations(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood = self.likfun.gradloglikelihood(y_obs, self.y_sim)
        expected_loglikelihood = -3.7537571275591683
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_loglikelihood, expected_loglikelihood)

    def test_grad_loglikelihood_additive(self):
        y_obs = [1.8, 0.9]
        comp_loglikelihood_a = self.likfun.gradloglikelihood([y_obs[0]], self.y_sim)
        comp_loglikelihood_b = self.likfun.gradloglikelihood([y_obs[1]], self.y_sim)
        comp_loglikelihood_two = self.likfun.gradloglikelihood(y_obs, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

    def BetaNorm(self, x1, x2):
        assert len(x2.shape) == 1, "x2 should be a 1D tensor"
        assert x1.shape[1:] == x2.shape, "The last dimensions of x1 and x2 should match"
        
        # Subtract x2 from all entries in x1 and compute the beta norm
        diff = x1 - x2
        norm_beta = torch.sum(torch.abs(diff).pow(2), dim=-1).pow(self.beta/2)
        return norm_beta










if __name__ == '__main__':
    # unittest.main()
    tests = EnergyScoreTests()
    tests.setUp()
    tests.test_likelihood()
