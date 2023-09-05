##import torch
##
##def g_and_k_quantile(y, A, B, g, k):
##    c1 = 1/torch.sqrt(torch.tensor(2.0))
##    c2 = -1/2
##    c3 = -1/(6*torch.sqrt(torch.tensor(2.0)))
##    c4 = -1/24
##    
##    return A + B * (1 + c1*y + c2*y**2 + c3*y**3 + c4*y**4) * (1 + y**2)**k * torch.exp(g*y)
##
##def simulate_g_and_k(n, A, B, g, k):
##    # Sample from standard normal
##    y = torch.randn(n)
##    
##    # Return quantile values
##    return g_and_k_quantile(y, A, B, g, k)
##
##def get_gradients(n, A, B, g, k):
##    A.requires_grad_(True)
##    B.requires_grad_(True)
##    g.requires_grad_(True)
##    k.requires_grad_(True)
##
##    samples = simulate_g_and_k(n, A, B, g, k)
##    grads = []
##
##    for s in samples:
##        s.backward(retain_graph=True)
##        grads.append((A.grad.item(), B.grad.item(), g.grad.item(), k.grad.item()))
##        A.grad.zero_()
##        B.grad.zero_()
##        g.grad.zero_()
##        k.grad.zero_()
##
##    return samples, grads
##
### Example usage
##A, B, g, k = torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5)
##samples, gradients = get_gradients(1000, A, B, g, k)
##print(samples)
##print(gradients)
##



import torch
import logging
from numbers import Number

import numpy as np
import scipy

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector

class G_and_K(ProbabilisticModel, Continuous):

    def __init__(self, parameters, name='G_and_K'):
        # We expect input of type parameters = [mu, sigma]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 4:
            raise RuntimeError('Input list must be of length 4, containing [A, B, g, k].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.ordered_transforms = [False, torch.exp, False, False]
        self.ordered_inverse_transforms = [False, torch.log, False, False]

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 4:
            raise ValueError('Number of parameters of Normal model must be 4.')

        # Check whether input is from correct domain
        if input_values[1] < 0:
            return False

        return True
    
    def _check_output(self, values):
        return True
    
    def get_output_dimension(self):
        return 1 

    def g_and_k_quantile(self, y, A, B, g, k):
        c1 = 1/torch.sqrt(torch.tensor(2.0))
        c2 = -1/2
        c3 = -1/(6*torch.sqrt(torch.tensor(2.0)))
        c4 = -1/24
        
        return A + B * (1 + c1*y + c2*y**2 + c3*y**3 + c4*y**4) * (1 + y**2)**k * torch.exp(g*y)

    def forward_simulate(self, params, n, rng=None, to_list=True):
        # Sample from standard normal
        y = torch.randn(n)
        
        # Return quantile values
        if to_list:
            return self.g_and_k_quantile(y, params[0], params[1], params[2],params[3]).tolist()
        else:
            return self.g_and_k_quantile(y, params[0], params[1], params[2],params[3])
        
    def grad_forward_simulate(self,params, n, rng=None):
        
        A, B, g, k = torch.tensor(params[0]), torch.tensor(params[1]), torch.tensor(params[2]), torch.tensor(params[3])
        A.requires_grad_(True)
        B.requires_grad_(True)
        g.requires_grad_(True)
        k.requires_grad_(True)

        samples = self.forward_simulate([A, B, g, k], n, to_list=False)
        grads = []

        for s in samples:
            s.backward(retain_graph=True)
            grads.append((A.grad.item(), B.grad.item(), g.grad.item(), k.grad.item()))
            A.grad.zero_()
            B.grad.zero_()
            g.grad.zero_()
            k.grad.zero_()

        return samples.tolist() + grads

    def transform_list(self):
        return self.ordered_transforms

    def inverse_transform_list(self):
        return self.ordered_inverse_transforms
    
    def jacobian_list(self):
        return self.ordered_transforms
    
    def transform_variables(self, variables):
        # Takes as input: [np.array(theta1), ...... , np.array(theta_n)]
        # returns : list of transformed variables in correct space : [T(theta1), ..... , T(thetan)]
        variables = self.to_tensor(variables)
        transformed = variables
        for index in range(0,len(variables)):
            if self.ordered_transforms[index]:
                transformed[index] = self.ordered_transforms[index](variables[index])
        return transformed.tolist()
    
    def to_tensor(self,variables):
        arg = torch.tensor(variables, dtype=torch.float)#, requires_grad = True)
        return arg
# Example usage
#A, B, g, k = torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5)
#A, B, g, k = 0.0, 1.0, 0.5, 0.5
#samples, gradients = grad_forward_simulate(100, A, B, g, k)
#print(samples)
#print(gradients)

    def transform(self, variables):
        return transform_variables(variables)
    

    
    def inverse_transform(self, variables):
        variables = self.to_tensor(variables)
        transformed = variables
        for index in range(0,len(variables)):
            if self.ordered_transforms[index]:
                transformed[index] = self.ordered_inverse_transforms[index](variables[index])
        return transformed.tolist()

    def transform_jacobian(self, variables): # variables is probably not needed here
        # Takes as input: [theta1, ..... , thetan]
        # returns: np.array of the jacobian of the variable input wrt to the parameter transformations
        # [dT(theta1)/dtheta1, ..... dT(theta_n)/dthetan] along the diagonal
        array = []
        for element_index, element in enumerate(variables):
            x = torch.tensor(float(variables[element_index]), requires_grad = True)
            if self.ordered_transforms[element_index]:
                y = self.ordered_transforms[element_index](x)
                y.backward()
                dx = x.grad
                array.append(dx)
            else:
                array.append(1)
        return np.diag(array)