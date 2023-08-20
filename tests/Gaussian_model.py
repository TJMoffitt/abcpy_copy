import logging
from numbers import Number

import numpy as np
import scipy

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector


import torch
from torch.autograd.functional import jacobian

class Gaussian(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Gaussian'):
        # We expect input of type parameters = [mu, sigma]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 2:
            raise RuntimeError('Input list must be of length 2, containing [mu, sigma].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.ordered_transforms = [False, torch.exp]

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters of Normal model must be 2.')

        # Check whether input is from correct domain
        mu = input_values[0]
        sigma = input_values[1]
        if sigma < 0:
            return False

        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        # input_values = self.transform_variables(input_values) # do this outside in inference. 
        mu = input_values[0]
        sigma = input_values[1]

        # Do the actual forward simulation
        #vector_of_k_samples = np.array(rng.normal(mu, sigma, k))

        # Format the output to obey API   
        result = self.normal_model_pytorch([float(input_value) for input_value in input_values], k)#[np.array([x]) for x in vector_of_k_samples]
        return result

    def normal_model_pytorch(self, input_values, n, return_grad = False):
        values = []
        for n in range(0,n):
            value = []
            mu = torch.tensor(input_values[0], requires_grad = True)
            sigma = torch.tensor(input_values[1], requires_grad = True)
            variables = [mu,sigma]

            yval = torch.randn(1)*sigma + mu
            value.append(yval.item())
            #yval.backward()
            #for var in variables:
             #   value.append(var.grad.item())
            values.append(np.array(value))
        return values

    def grad_forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Takes input in the form:  [a,....,z]
        #print(input_values)
        # Outputs: array: [x1, x2, ...... ,xn, [dx1/dtheta1, dx1/dtheta2], ...... [dxn/dtheta1, dxn/dtheta2],]

        mu = input_values[0]
        sigma = input_values[1]
  
        result = self.grad_normal_model_pytorch([float(input_value) for input_value in input_values], k)#[np.array([x]) for x in vector_of_k_samples]
        #print(result)
        return result

    def grad_normal_model_pytorch(self, input_values, n, return_grad = False):
        values = []
        gradvalues = []

        for n in range(0,n):

            mu = torch.tensor(input_values[0], requires_grad = True)
            sigma = torch.tensor(input_values[1], requires_grad = True)
            z = torch.randn(1)
            variables = [mu,sigma]

            yval = z*sigma + mu    # Check if this should be grad_log_normal_ or rename grad_normal
            ########################################
            #grad_mu = ((yval.item()-input_values_transformed[0])/(input_values_transformed[1]**2))*scipy.stats.norm(input_values_transformed[0], input_values_transformed[1]).pdf(yval.item())
            #grad_sigma = ((((yval.item()-input_values_transformed[0])**2)/(input_values_transformed[1]**3)) -  1/(input_values_transformed[1]) )*scipy.stats.norm(input_values_transformed[0], input_values_transformed[1]).pdf(yval.item())
            #gradvalue = []
            #gradvalue.append(grad_mu)
            #gradvalue.append(grad_sigma)
            #print(yval.item(), input_values_transformed[0], grad_mu)
            #print("^ Mu")
            #print(abs(yval.item()-input_values_transformed[0]), input_values_transformed[1], grad_sigma)
            #print(abs(yval.item()-input_values_transformed[0])-input_values_transformed[1], grad_sigma )
            #print(" ^ Sigma( if 1 is greater than two the gradient should be )")
            #########################################
            values.append(yval.item())
            yval.backward()
            gradvalue = []
            for var in variables:
                gradvalue.append(var.grad.item())
            gradvalues.append(gradvalue)
            
            # print(variables)
            # print(yval.item())
            # print(np.array(gradvalue))
        return values + gradvalues
      

    def _check_output(self, values):
        if not isinstance(values, Number):
            raise ValueError('Output of the normal distribution is always a number.')

        # At this point values is a number (int, float); full domain for Normal is allowed
        return True
    

    # def transform_variables(self, variables):
    #     mu = torch.exp(variables[0])
    #     sigma = variables[1]
    #     return [mu, sigma]


    # def transform_jacobian(self, variables): # variables is probably not needed here
    #     jacobian_of_inputs = jacobian(self.transfrom_variables, variables)
    #     return jacobian_of_inputs

    def get_output_dimension(self):
        return 1  

    
    def to_tensor(self,variables):
        arg = torch.tensor(variables, dtype=torch.float)#, requires_grad = True)
        return arg
    
    def transform_variables(self, variables):
        # Takes as input: [np.array(theta1), ...... , np.array(theta_n)]
        # returns : list of transformed variables in correct space : [T(theta1), ..... , T(thetan)]
        variables = self.to_tensor(variables)
        transformed = variables
        for index in range(0,len(variables)):
            if self.ordered_transforms[index]:
                transformed[index] = self.ordered_transforms[index](variables[index])
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
  


normalmodel = Gaussian([10,2])
#print(normalmodel.transform_variables([10,10]))
#print(normalmodel.transform_variables([5,0]))
#print(normalmodel.transform_variables([0, -10]))
#print(normalmodel.transform_variables([-5,10]))
print(normalmodel.transform_jacobian([10,2]))
print(normalmodel.transform_jacobian([10,3]))
print(normalmodel.transform_jacobian([10,4]))


#print(np.sum(normalmodel.forward_simulate([10,2],1000))/1000)
#print(np.std(normalmodel.forward_simulate([10,2],1000)))
#print(np.sum(normalmodel.grad_forward_simulate([10,2],10)))
#print(normalmodel.grad_forward_simulate([10,2],1000)[1000:1010])
#print(np.std(normalmodel.grad_forward_simulate([10,2],1000)[0:1000]))