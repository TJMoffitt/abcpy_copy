import numpy as np

from abcpy.probabilisticmodels import Hyperparameter, ModelResultingFromOperation

import torch


class GraphTools:
    """This class implements all methods that will be called recursively on the graph structure."""

    def sample_from_prior(self, model=None, rng=np.random.RandomState()):
        """
        Samples values for all random variables of the model.
        Commonly used to sample new parameter values on the whole graph.

        Parameters
        ----------
        model: abcpy.ProbabilisticModel object
            The root model for which sample_from_prior should be called.
        rng: Random number generator
            Defines the random number generator to be used
        """
        if model is None:
            model = self.model
        # If it was at some point not possible to sample (due to incompatible parameter values provided by the parents), we start from scratch
        while not (self._sample_from_prior(model, rng=rng)):
            self._reset_flags(model)

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        self._reset_flags(model)

    def _sample_from_prior(self, models, is_not_root=False, was_accepted=True, rng=np.random.RandomState()):
        """
        Recursive version of sample_from_prior. Commonly called from within sample_from_prior.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            Defines the models for which, together with their parents, new parameters will be sampled
        is_root: boolean
            Whether the probabilistic models provided in models are root models.
        was_accepted: boolean
            Whether the sampled values for all previous/parent models were accepted.
        rng: Random number generator
            Defines the random number generator to be used

        Returns
        -------
        boolean:
            Whether it was possible to sample new values for all nodes of the graph.
        """

        # If it was so far possible to sample parameters for all nodes, the current node as well as its parents are sampled, using depth-first search
        if was_accepted:
            for model in models:
                for parent in model.get_input_models():
                    if not parent.visited:
                        parent.visited = True
                        was_accepted = self._sample_from_prior([parent], is_not_root=True, was_accepted=was_accepted,
                                                               rng=rng)
                        if not was_accepted:
                            return False

                if is_not_root and not (model._forward_simulate_and_store_output(rng=rng)):
                    return False

                model.visited = True

        return was_accepted

    def _reset_flags(self, models=None):
        """
        Resets all flags that say that a probabilistic model has been updated. Commonly used after actions on the whole
        graph, to ensure that new actions can take place.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel
            The models for which, together with their parents, the flags should be reset. If no value is provided, the
            root models are assumed to be the model of the inference method.
        """
        if not models:
            models = self.model

        # For each model, the flags of the parents get reset recursively.
        for model in models:
            for parent in model.get_input_models():
                self._reset_flags([parent])
            model.visited = False
            model.calculated_pdf = None

    def pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
        """
        Calculates the joint probability density function of the prior of the specified models at the given parameter values.
        Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            Defines the models for which the pdf of their prior should be evaluated
        parameters: python list
            The parameters at which the pdf should be evaluated
        mapping: list of tuples
            Defines the mapping of probabilistic models and index in a parameter list.
        is_root: boolean
            A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

        Returns
        -------
        list
            The resulting pdf,as well as the next index to be considered in the parameters list.
        """
        self.set_parameters(parameters)
        result = self._recursion_pdf_of_prior(models, parameters, mapping, is_root)
        return result

    def _recursion_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
        """
        Calculates the joint probability density function of the prior of the specified models at the given parameter values.
        Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            Defines the models for which the pdf of their prior should be evaluated
        parameters: python list
            The parameters at which the pdf should be evaluated
        mapping: list of tuples
            Defines the mapping of probabilistic models and index in a parameter list.
        is_root: boolean
            A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

        Returns
        -------
        list
            The resulting pdf,as well as the next index to be considered in the parameters list.
        """
        # At the beginning of calculation, obtain the mapping#
        if is_root:
            mapping, garbage_index = self._get_mapping()

        # The pdf of each root model is first calculated separately
        result = [1.] * len(models)

        for i, model in enumerate(models):
            # If the model is not a root model, the pdf of this model, given the prior, should be calculated
            if not is_root and not (isinstance(model, ModelResultingFromOperation)):
                # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
                relevant_parameters = []

                for mapped_model, model_index in mapping:
                    if mapped_model == model:
                        parameter_index = model_index
                        # for j in range(model.get_output_dimension()):
                        relevant_parameters.append(parameters[parameter_index])
                        # parameter_index+=1
                        break
                if len(relevant_parameters) == 1:
                    relevant_parameters = relevant_parameters[0]
                else:
                    relevant_parameters = np.array(relevant_parameters)
            else:
                relevant_parameters = []

            # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation.
            visited_parents = [False for j in range(len(model.get_input_models()))]
            # For each parent, the pdf of this parent has to be calculated as well.
            for parent_index, parent in enumerate(model.get_input_models()):
                # Only calculate the pdf if the parent has never been visited for this model
                if not (visited_parents[parent_index]):
                    pdf = self._recursion_pdf_of_prior([parent], parameters, mapping=mapping, is_root=False)
                    input_models = model.get_input_models()
                    for j in range(len(input_models)):
                        if input_models[j][0] == parent:
                            visited_parents[j] = True
                    result[i] *= pdf
            if not is_root:
                if model.calculated_pdf is None:
                    result[i] *= model.pdf(model.get_input_values(), relevant_parameters)
                else:
                    result[i] *= 1

                    # Multiply the pdfs of all roots together to give an overall pdf.
        temporary_result = result
        result = 1.
        for individual_result in temporary_result:
            result *= individual_result

        return result
    
    # def pdf_of_prior_transformed(self, models, parameters, mapping=None, is_root=True):
    #     """
    #     Calculates the joint probability density function of the prior of the specified models at the given parameter values.
    #     Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

    #     Parameters
    #     ----------
    #     models: list of abcpy.ProbabilisticModel objects
    #         Defines the models for which the pdf of their prior should be evaluated
    #     parameters: python list
    #         The parameters at which the pdf should be evaluated
    #     mapping: list of tuples
    #         Defines the mapping of probabilistic models and index in a parameter list.
    #     is_root: boolean
    #         A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

    #     Returns
    #     -------
    #     list
    #         The resulting pdf,as well as the next index to be considered in the parameters list.
    #     """
    #     self.set_parameters(parameters)
    #     parameters = self.apply_full_transform(parameters)
    #     result = self._recursion_pdf_of_prior_transformed(models, parameters, mapping, is_root)
    #     return result

    # def _recursion_pdf_of_prior_transformed(self, models, parameters, mapping=None, is_root=True):
    #     """
    #     Calculates the joint probability density function of the prior of the specified models at the given parameter values.
    #     Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

    #     Parameters
    #     ----------
    #     models: list of abcpy.ProbabilisticModel objects
    #         Defines the models for which the pdf of their prior should be evaluated
    #     parameters: python list
    #         The parameters at which the pdf should be evaluated
    #     mapping: list of tuples
    #         Defines the mapping of probabilistic models and index in a parameter list.
    #     is_root: boolean
    #         A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

    #     Returns
    #     -------
    #     list
    #         The resulting pdf,as well as the next index to be considered in the parameters list.
    #     """
    #     # At the beginning of calculation, obtain the mapping#
    #     if is_root:
    #         mapping, garbage_index = self._get_mapping()

    #     # The pdf of each root model is first calculated separately
    #     result = [1.] * len(models)

    #     for i, model in enumerate(models):
    #         # If the model is not a root model, the pdf of this model, given the prior, should be calculated
    #         if not is_root and not (isinstance(model, ModelResultingFromOperation)):
    #             # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
    #             relevant_parameters = []

    #             for mapped_model, model_index in mapping:
    #                 if mapped_model == model:
    #                     parameter_index = model_index
    #                     # for j in range(model.get_output_dimension()):
    #                     relevant_parameters.append(parameters[parameter_index])
    #                     # parameter_index+=1
    #                     break
    #             if len(relevant_parameters) == 1:
    #                 relevant_parameters = relevant_parameters[0]
    #             else:
    #                 relevant_parameters = np.array(relevant_parameters)
    #         else:
    #             relevant_parameters = []

    #         # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation.
    #         visited_parents = [False for j in range(len(model.get_input_models()))]
    #         # For each parent, the pdf of this parent has to be calculated as well.
    #         for parent_index, parent in enumerate(model.get_input_models()):
    #             # Only calculate the pdf if the parent has never been visited for this model
    #             if not (visited_parents[parent_index]):
    #                 pdf = self._recursion_pdf_of_prior_transformed([parent], parameters, mapping=mapping, is_root=False)
    #                 input_models = model.get_input_models()
    #                 for j in range(len(input_models)):
    #                     if input_models[j][0] == parent:
    #                         visited_parents[j] = True
    #                 result[i] *= pdf
    #         if not is_root:
    #             if model.calculated_pdf is None:
    #                 result[i] *= model.pdf(self.apply_local_transform(model, model.get_input_values()), relevant_parameters)
    #             else:
    #                 result[i] *= 1

    #                 # Multiply the pdfs of all roots together to give an overall pdf.
    #     temporary_result = result
    #     result = 1.
    #     for individual_result in temporary_result:
    #         result *= individual_result

    #     return result
    
    # def grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
    #     """
    #     Calculates the gradient of the joint log probability density function of the prior of the specified models at the given parameter values.
    #     Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

    #     Parameters
    #     ----------
    #     models: list of abcpy.ProbabilisticModel objects
    #         Defines the models for which the pdf of their prior should be evaluated
    #     parameters: python list
    #         The parameters at which the pdf should be evaluated
    #     mapping: list of tuples
    #         Defines the mapping of probabilistic models and index in a parameter list.
    #     is_root: boolean
    #         A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

    #     Returns
    #     -------
    #     list
    #         The resulting pdf,as well as the next index to be considered in the parameters list.
    #     """
    #     self.set_parameters(parameters)
    #     print(" TESTING ")
    #     result = self._recursion_grad_log_pdf_of_prior(models, parameters, mapping, is_root)
    #     return result

    # def _recursion_grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
    #     """
    #     Calculates the joint gradient of the log probability density function of the prior of the specified models at the given parameter values.
    #     Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

    #     Parameters
    #     ----------
    #     models: list of abcpy.ProbabilisticModel objects
    #         Defines the models for which the pdf of their prior should be evaluated
    #     parameters: python list
    #         The parameters at which the pdf should be evaluated
    #     mapping: list of tuples
    #         Defines the mapping of probabilistic models and index in a parameter list.
    #     is_root: boolean
    #         A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

    #     Returns
    #     -------
    #     list
    #         The resulting grad log pdf,as well as the next index to be considered in the parameters list.
    #     """
    #     # At the beginning of calculation, obtain the mapping
    #     if is_root:
    #         mapping, garbage_index = self._get_mapping()

    #     # The pdf of each root model is first calculated separately
    #     result = [0.0] * len(models)

    #     for i, model in enumerate(models):
    #         # If the model is not a root model, the pdf of this model, given the prior, should be calculated
    #         if not is_root and not (isinstance(model, ModelResultingFromOperation)):
    #             # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
    #             relevant_parameters = []

    #             for mapped_model, model_index in mapping:
    #                 if mapped_model == model:
    #                     parameter_index = model_index
    #                     # for j in range(model.get_output_dimension()):
    #                     relevant_parameters.append(parameters[parameter_index])
    #                     # parameter_index+=1
    #                     break
    #             if len(relevant_parameters) == 1:
    #                 relevant_parameters = relevant_parameters[0]
    #             else:
    #                 relevant_parameters = np.array(relevant_parameters)
    #         else:
    #             relevant_parameters = []

    #         # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation.
    #         visited_parents = [False for j in range(len(model.get_input_models()))]
    #         # For each parent, the pdf of this parent has to be calculated as well.
    #         for parent_index, parent in enumerate(model.get_input_models()):
    #             # Only calculate the pdf if the parent has never been visited for this model
    #             if not (visited_parents[parent_index]):
    #                 grad_log_pdf = self._recursion_grad_log_pdf_of_prior([parent], parameters, mapping=mapping, is_root=False)
    #                 input_models = model.get_input_models()
    #                 for j in range(len(input_models)):
    #                     if input_models[j][0] == parent:
    #                         visited_parents[j] = True
    #                 result[i] += grad_log_pdf
    #         if not is_root:
    #             if model.calculated_grad_log_pdf is None:
    #                 result[i] += model.grad_log_pdf(model.get_input_values(), relevant_parameters)
    #             else:
    #                 result[i] += 0.0

    #                 # Multiply the pdfs of all roots together to give an overall pdf.
    #     temporary_result = result
    #     result = 0.0
    #     for individual_result in temporary_result:
    #         result += individual_result

    #     return result

    # def _recursion_grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
    #     """
    #     Calculates the joint gradient of the log probability density function of the prior of the specified models at the given parameter values.
    #     Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

    #     Parameters
    #     ----------
    #     models: list of abcpy.ProbabilisticModel objects
    #         Defines the models for which the pdf of their prior should be evaluated
    #     parameters: python list
    #         The parameters at which the pdf should be evaluated
    #     mapping: list of tuples
    #         Defines the mapping of probabilistic models and index in a parameter list.
    #     is_root: boolean
    #         A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

    #     Returns
    #     -------
    #     list
    #         The resulting grad log pdf,as well as the next index to be considered in the parameters list.
    #     """
    #     print(" 1")
    #     # At the beginning of calculation, obtain the mapping
    #     if is_root:
    #         mapping, garbage_index = self._get_mapping()
    #     print(" 2")
    #     # The pdf of each root model is first calculated separately
    #     # result = [0.0] * len(models)
    #     print(str(" Length of model parameters :") + str(len(parameters)))
    #     #result = [[0.0] * len(parameters[modelindex])] for modelindex, model in enumerate(models)] # This only works in one dimension, 
    #     result = [0.0 * len(parameters)]

    #     for i, model in enumerate(models):
    #         # If the model is not a root model, the pdf of this model, given the prior, should be calculated
    #         if not is_root and not (isinstance(model, ModelResultingFromOperation)):
    #             # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
    #             relevant_parameters = []

    #             for mapped_model, model_index in mapping:
    #                 if mapped_model == model:
    #                     parameter_index = model_index
    #                     # for j in range(model.get_output_dimension()):
    #                     relevant_parameters.append(parameters[parameter_index])
    #                     # parameter_index+=1
    #                     break
    #             if len(relevant_parameters) == 1:
    #                 relevant_parameters = relevant_parameters[0]
    #             else:
    #                 relevant_parameters = np.array(relevant_parameters)
    #         else:
    #             relevant_parameters = []

    #         # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation.
    #         visited_parents = [False for j in range(len(model.get_input_models()))]
    #         # For each parent, the pdf of this parent has to be calculated as well.
    #         for parent_index, parent in enumerate(model.get_input_models()):
    #             # Only calculate the pdf if the parent has never been visited for this model
    #             if not (visited_parents[parent_index]):
    #                 grad_log_pdf = self._recursion_grad_log_pdf_of_prior([parent], parameters, mapping=mapping, is_root=False)
    #                 input_models = model.get_input_models()
    #                 for j in range(len(input_models)):
    #                     if input_models[j][0] == parent:
    #                         visited_parents[j] = True
    #                 result[i] += grad_log_pdf
    #         if not is_root: 
    #             if model.calculated_pdf is None:
                    
    #                 result[i] += model.gradlogpdf(model.get_input_values(), relevant_parameters)
    #                 print(result[i])
    #                 print(" ^ A Grad Log PDF ")
    #             else:
    #                 result[i] += 0.0
    #                 print(result[i])
    #                 print(" ^ A Grad Log PDF ")

    #                 # Multiply the pdfs of all roots together to give an overall pdf.
    #     print(" --- ")
    #     print(parameters)
    #     print(result)
    #     print(" --- ")
    #     #temporary_result = result
    #     #result = 0.0
    #     #for individual_result in temporary_result:
    #     #    result += individual_result

    #     return result

    # # def _recursion_grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
    # #     for model in models:
    # #         print(model.get_input_models())
    # #         result = [0.0] * len(parameters)
    # #         for parent_index, parent in enumerate(model.get_input_models()):
    # #             print(self.jacobian())
    # #             #print(parent_index)
    # #             #print(parent)
    # #             #print(parent.get_input_values())
    # #             print(" =========== ")
    # #             print(parent_index)
    # #             print(parent.get_input_values())
    # #             print(parameters[parent_index])
    # #             result[parent_index] += parent.gradlogpdf(parent.get_input_values(), parameters[parent_index])
    # #             if parent not in model.get_input_models():
    # #                 print(parent.r_jacobian())
    # #             print(result)
    # #             print(" =========== ")
    # #         return result
    # def grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
    #     self.set_parameters(parameters)
    #     for model in models:
    #         jacobians = []
    #         mapping, garbage_index = self._get_mapping()
    #         print(mapping)
    #         jacobians_list = [0]*len(mapping)
    #         grad_log_prior_array = [0.0]*len(mapping)
    #         for element in mapping:
    #             print(element)
    #             print(" ^ 1")
    #             print(element[0].get_input_values())
    #             grad_log_prior_array[element[1]] = element[0].gradlogpdf(element[0].get_input_values(), parameters[element[1]])
    #             #try:
    #             jacobian_list = element[0].jacobian_list()
    #             for parent_index, parent in enumerate(element[0].get_input_models()):
    #                 #print(parent)
    #                 #print(" ^ 2")
    #                 for listelement in mapping:
    #                     if listelement[0] == parent:
    #                         jacobians_list[listelement[1]] = jacobian_list[parent_index]
    #                         #print(listelement[0])
    #                         ##print(parent)
    #                         #print(" *** ")
    #             #print(jacobians_list)

    #             #print(jacobians)
    #             #except:
    #                 #print(str(element[0])+ "has no jacobian list")
    #         #print(grad_log_prior_array)
    #         #print(jacobians)
    #             #print(element[0])
    #             #print(element[1])
    #             #print(element[0].get_input_values())
    #         #print(garbage_index)
    #         # models_unordered = self._recursion_grad_log_pdf_of_prior(model, parameters)
    #         # print(models_unordered)
    #         # for element in models_unordered:
    #         #     print(element.get_input_values())

            
    #     #return result

    def grad_log_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
        # self.set_parameters(parameters)
        for model in models:
            mapping, garbage_index = self._get_mapping()
            grad_log_prior_array = [0.0]*len(mapping)
            for element in mapping:
                #print(element)
                #print(" ^ 1")
                #print(element[0].get_input_values())
                theta_values = element[0].get_input_values()
                #print(str(theta_values) + " < Theta Values Got from chain")
                transformed = self.apply_local_transform(element[0], theta_values)
                #print(str(transformed) + " < Transformed Values through T(theta)")
                grad_log_prior_array[element[1]] = element[0].gradlogpdf(transformed, parameters[element[1]])
                #try:
                #jacobian_list = element[0].jacobian_list()
                #for parent_index, parent in enumerate(element[0].get_input_models()):
                ##    #print(parent)
                #    #print(" ^ 2")
                #    for listelement in mapping:
                #        if listelement[0] == parent:
                #            jacobians_list[listelement[1]] = jacobian_list[parent_index]
                
            #print(grad_log_prior_array)
            return grad_log_prior_array

    def apply_local_transform(self,element,values):
        output_array = [0.0] * len(values)
        try:
            transforms = element.transform_list()
        except:
            return values
        inputmodels = element.get_input_models()
        for index, element in enumerate(values):
            transform = transforms[index]
            output_array[index] = element
            if transform != False and ("Hyperparameter" not in str(inputmodels[index])):
                output_array[index] = transforms[index](torch.tensor(element)).item()
            #else:
            #    output_array[index] = element

        return output_array
    # def _recursion_grad_log_pdf_of_prior(self, node, parameters, mapping=None, is_root=True):
    #     functions = []
    #     if "abcpy.probabilisticmodels.Hyperparameter" not in str(type(node)): # CHANGE THIS!!! complete hack
    #         print(type(node))
    #         functions.append(node)
    #     print(node)
    #     for parent in node.get_input_models():
    #         parent_functions = self._recursion_grad_log_pdf_of_prior(parent, parameters)
    #         if parent_functions != []:
    #             for element in parent_functions:
    #                 functions.append(element)
    #     return functions

    # def recursive_leaf(self, leaf):
    #     parent_index, parent in enumerate(leaf.get_input_models()):
    #         if [(type(element) == float) for element in parent.get_input_values()]:
    #             return [parent for parent in ]
    #         else:


    def _get_mapping(self, models=None, index=0, is_not_root=False):
        """Returns a mapping of model and first index corresponding to the outputs in this model in parameter lists.

        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects
        index: integer
            Next index to be mapped in a parameter list
        is_not_root: boolean
            Specifies whether the models specified are root models.

        Returns
        -------
        list
            A list containing two entries. The first entry corresponds to the mapping of the root models, including their parents. The second entry corresponds to the next index to be considered in a parameter list.
        """

        if models is None:
            models = self.model

        mapping = []

        for model in models:
            # If this model corresponds to an unvisited free parameter, add it to the mapping
            if is_not_root and not model.visited and not (isinstance(model, Hyperparameter)) and not (
            isinstance(model, ModelResultingFromOperation)):
                mapping.append((model, index))
                index += 1  # model.get_output_dimension()
            # Add all parents to the mapping, if applicable
            for parent in model.get_input_models():
                parent_mapping, index = self._get_mapping([parent], index=index, is_not_root=True)
                parent.visited = True
                for mappings in parent_mapping:
                    mapping.append(mappings)

            model.visited = True

        # At the end of the algorithm, reset all flags such that another method can act on the graph freely.
        if not is_not_root:
            self._reset_flags()

        return [mapping, index]

    def _get_names_and_parameters(self):
        """
        A function returning the name of each model and the corresponding parameters to this model

        Returns
        -------
        list:
            Each entry is a tuple, the first entry of which is the name of the model and the second entry is the parameter values associated with it
        """
        mapping = self._get_mapping()[0]

        return_value = []

        for model, index in mapping:
            return_value.append(
                (model.name, self.accepted_parameters_manager.get_accepted_parameters_bds_values([model])))

        return return_value

    def get_parameters(self, models=None, is_root=True):
        """
        Returns the current values of all free parameters in the model. Commonly used before perturbing the parameters
        of the model.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            The models for which, together with their parents, the parameter values should be returned. If no value is
            provided, the root models are assumed to be the model of the inference method.
        is_root: boolean
            Specifies whether the current models are at the root. This ensures that the values corresponding to
            simulated observations will not be returned.

        Returns
        -------
        list
            A list containing all currently sampled values of the free parameters.
        """
        parameters = []

        # If we are at the root, we set models to the model attribute of the inference method
        if is_root:
            models = self.model

        for model in models:
            # If we are not at the root, the sampled values for the current node should be returned
            if is_root == False and not isinstance(model, (ModelResultingFromOperation, Hyperparameter)):
                parameters.append(model.get_stored_output_values())
                model.visited = True

            # Implement a depth-first search to return also the sampled values associated with each parent of the model
            for parent in model.get_input_models():
                if not parent.visited:
                    parameters += self.get_parameters(models=[parent], is_root=False)
                    parent.visited = True

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if is_root:
            self._reset_flags()

        return parameters

    def set_parameters(self, parameters, models=None, index=0, is_root=True):
        """
        Sets new values for the currently used values of each random variable.
        Commonly used after perturbing the parameter values using a kernel.

        Parameters
        ----------
        parameters: list
            Defines the values to which the respective parameter values of the models should be set
        model: list of abcpy.ProbabilisticModel objects
             Defines all models for which, together with their parents, new values should be set. If no value is provided, the root models are assumed to be the model of the inference method.
        index: integer
            The current index to be considered in the parameters list
        is_root: boolean
            Defines whether the current models are at the root. This ensures that only values corresponding to random variables will be set.

        Returns
        -------
        list: [boolean, integer]
            Returns whether it was possible to set all parameters and the next index to be considered in the parameters list.
        """

        # If we are at the root, we set models to the model attribute of the inference method
        if is_root:
            models = self.model

        for model in models:
            # New parameters should only be set in case we are not at the root
            if not is_root and not isinstance(model, ModelResultingFromOperation):
                # new_output_values = np.array(parameters[index:index + model.get_output_dimension()])
                new_output_values = np.array(parameters[index]).reshape(-1, )
                if not model.set_output_values(new_output_values):
                    return [False, index]
                index += 1  # model.get_output_dimension()
                model.visited = True

            # New parameters for all parents are set using a depth-first search
            for parent in model.get_input_models():
                if not parent.visited and not isinstance(parent, Hyperparameter):
                    is_set, index = self.set_parameters(parameters, models=[parent], index=index, is_root=False)
                    if not is_set:
                        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
                        if is_root:
                            self._reset_flags()
                        return [False, index]
            model.visited = True

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if is_root:
            self._reset_flags()

        return [True, index]

    def get_correct_ordering(self, parameters_and_models, models=None, is_root=True):
        """
        Orders the parameters returned by a kernel in the order required by the graph.
        Commonly used when perturbing the parameters.

        Parameters
        ----------
        parameters_and_models: list of tuples
            Contains tuples containing as the first entry the probabilistic model to be considered and as the second entry the parameter values associated with this model
        models: list
            Contains the root probabilistic models that make up the graph. If no value is provided, the root models are assumed to be the model of the inference method.

        Returns
        -------
        list
            The ordering which can be used by recursive functions on the graph.
        """
        ordered_parameters = []

        # If we are at the root, we set models to the model attribute of the inference method
        if is_root:
            models = self.model

        for model in models:
            if not model.visited:
                model.visited = True

                # Check all entries in parameters_and_models to determine whether the current model is contained within it
                for corresponding_model, parameter in parameters_and_models:
                    if corresponding_model == model:
                        for param in parameter:
                            ordered_parameters.append(param)
                        break

                # Recursively order all the parents of the current model
                for parent in model.get_input_models():
                    if not parent.visited:
                        parent_ordering = self.get_correct_ordering(parameters_and_models, models=[parent],
                                                                    is_root=False)
                        for parent_parameters in parent_ordering:
                            ordered_parameters.append(parent_parameters)

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if is_root:
            self._reset_flags()

        return ordered_parameters

    def simulate(self, n_samples_per_param, rng=np.random.RandomState(), npc=None):
        """Simulates data of each model using the currently sampled or perturbed parameters.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list
            Each entry corresponds to the simulated data of one model.
        """
        result = []
        for model in self.model:
            try: 
                modelparams = self.apply_local_transform(model, model.get_input_values())
                #modelparams = model.transform_variables(model.get_input_values())
            except:
                modelparams = model.get_input_values()
            parameters_compatible = model._check_input(modelparams) # Changed to model.transform_variables(model.get_input_values()) from model.get_input_values() 
            if parameters_compatible:
                if npc is not None and npc.communicator().Get_size() > 1:
                    simulation_result = npc.run_nested(model.forward_simulate, modelparams,
                                                       n_samples_per_param, rng=rng)
                else:
                    simulation_result = model.forward_simulate(modelparams, n_samples_per_param, rng=rng)
                result.append(simulation_result)
            else:
                return None
        return result
    

    def gradsimulate(self, n_samples_per_param, rng=np.random.RandomState(), npc=None):
        """Simulates data of each model using the currently sampled or perturbed parameters.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list
            Each entry corresponds to the simulated data of one model.
        """
        result = []
        for model in self.model:
            #print(model.get_input_values())
            try:
                modelparams = self.apply_local_transform(model, model.get_input_values())
                #modelparams = model.transform_variables(model.get_input_values()) #self.transform()[0]
            except:
                modelparams = model.get_input_values()
            parameters_compatible = model._check_input(modelparams)
            if parameters_compatible:
                if npc is not None and npc.communicator().Get_size() > 1:
                    simulation_result = npc.run_nested(model.grad_forward_simulate, modelparams,
                                                       n_samples_per_param, rng=rng)
                else:
                    simulation_result = model.grad_forward_simulate(modelparams, n_samples_per_param, rng=rng)

                result.append(simulation_result)
            else:
                return None
        return result
    
    def transform(self):
        """
        Takes as input a n dimensional array of theta values from R^{DxN}
        Returns the transformed variables in their corresponding space in the model
        """
        return [self.apply_local_transform(model, model.get_input_values()) for model in self.model]
        #return [model.transform_variables(model.get_input_values()) for model in self.model]
    
    def transform_post(self, model, variables):
        """
        Takes as input a n dimensional array of theta values from R^{DxN}
        Returns the transformed variables in their corresponding space in the model
        """
        return [np.array(variable) for variable in self.apply_local_transform(model, variables)]
        #return [np.array(variable) for variable in model.transform_variables(variables)]
    

    # def jacobian(self):
    #     print(" ////////")
    #     print(self.jacobian_old())
    #     print(self.jacobian_new())
    #     print(" ////////")        
    #     return self.jacobian_new()

    # def jacobian_old(self):
    #     # print(self.model[0].get_input_values())
    #     # print(" ^ input value")
    #     for model in self.model:
    #         #print(str(model.get_input_values()) + str(" < - Jacobian input value"))
    #         return model.transform_jacobian(model.get_input_values())


    def jacobian(self):
        element = self.model[0]
        values = element.get_input_values()
        output_array = [1.0] * len(values)
        try:
            transforms = element.jacobian_list()
        except:
            return values
        inputmodels = element.get_input_models()
        for index, element in enumerate(values):
            transform = transforms[index]
            # output_array[index] = element
            if transform != False and ("Hyperparameter" not in str(inputmodels[index])):
                output_array[index] = transforms[index](torch.tensor(element)).item()
            #else:
            #    output_array[index] = element

        return np.diag(output_array)
    
    def current_input_values(self):
        for model in self.model:
            return model.get_input_values()
        
    def full_grad_log_prior(self, input_array):
        output_array = [0.0] * len(input_array)
        mapping, garbage_index = self._get_mapping()
        for model in mapping:
            input_values = model[0].get_input_values()
            input_transformed = model[0].transform(input_values)
            x_val = input_array[model[1]]
            output_array[model[1]] = model[0].gradlogpdf(input_values, x_val)
        return output_array
    
    def full_jacobian(self):
        mapping, garbage_index = self._get_mapping()
        output_array = [0.0] * len(mapping)
        mapping_list = dict((x, y) for x, y in mapping)
        for element in mapping_list:
            jacobian = element.jacobian_list()   # This should the diagonal
            for index, parent in enumerate(element.get_input_models()):
                try:
                    location = mapping_list[parent]
                    jac_element = jacobian[index]
                    output_array[location] = jac_element
                except:
                    pass

        # Now we need to add any transformations into the final model!
        model = self.model[0]
        model_jacobian = model.jacobian_list()
        for index, parent in enumerate(model.get_input_models()):
                try:
                    #print(" --- ")
                    location = mapping_list[parent]
                    #print(location)
                    jac_element = model_jacobian[index]
                    #print(index)
                    #print(jac_element)
                    output_array[location] = jac_element
                except:
                    pass


        #print(output_array)
        return output_array
    
    def apply_jacobian(self,values):
        output_array = [0.0] * len(values)
        jacobian = self.full_jacobian()

        for index, element in enumerate(values):
            transform = jacobian[index]
            if transform != False:
                output_array[index] = jacobian[index](torch.tensor(element)).item()
            else:
                output_array[index] = 1

        return output_array

    def full_inverse_transform(self):
        mapping, garbage_index = self._get_mapping()
        output_array = [0.0] * len(mapping)
        mapping_list = dict((x, y) for x, y in mapping)
        for element in mapping_list:
            inverse_transform_list = element.inverse_transform_list()   # This should the diagonal
            for index, parent in enumerate(element.get_input_models()):
                try:
                    location = mapping_list[parent]
                    inverse_element = inverse_transform_list[index]
                    output_array[location] = inverse_element
                except:
                    pass

        # Now we need to add any transformations into the final model!
        model = self.model[0]
        model_inverse_transform = model.inverse_transform_list()
        for index, parent in enumerate(model.get_input_models()):
                try:
                    location = mapping_list[parent]
                    model_inverse_element = model_inverse_transform[index]
                    output_array[location] = model_inverse_element
                except:
                    pass

        return output_array
    
    def apply_full_inverse_transform(self,values):
        output_array = [0.0] * len(values)
        transforms = self.full_inverse_transform()
        for index, element in enumerate(values):
            transform = transforms[index]
            if transform != False:
                output_array[index] = np.array([transforms[index](torch.tensor(element)).item()])
            else:
                output_array[index] = element
        return output_array

    def full_transform(self):
        mapping, garbage_index = self._get_mapping()
        output_array = [0.0] * len(mapping)
        mapping_list = dict((x, y) for x, y in mapping)
        for element in mapping_list:
            transform_list = element.transform_list()   # This should the diagonal
            for index, parent in enumerate(element.get_input_models()):
                try:
                    location = mapping_list[parent]
                    transform_element = transform_list[index]
                    output_array[location] = transform_element
                except:
                    pass

        # Now we need to add any transformations into the final model!
        model = self.model[0]
        model_transform = model.transform_list()
        for index, parent in enumerate(model.get_input_models()):
                try:
                    location = mapping_list[parent]
                    model_inverse_element = model_transform[index]
                    output_array[location] = model_inverse_element
                except:
                    pass

        return output_array
    
    def apply_full_transform(self,values):
        output_array = [0.0] * len(values)
        transforms = self.full_transform()
        for index, element in enumerate(values):
            transform = transforms[index]
            if transform != False:
                output_array[index] = np.array([transforms[index](torch.tensor(element)).item()])
            else:
                output_array[index] = element
        return output_array


    # def add_nulls_scoringrule(self, scoring_rule):
    #     scoring_rule = scoring_rule.tolist()
    #     mapping = self._get_mapping()[0]
    #     array_n = [0.0] * len(mapping)
    #     model = self.model[0]
    #     base_models = model.get_input_models()
    #     mapping_list = dict((x, y) for x, y in mapping)
    #     for index, element in enumerate(base_models):
    #         array_n[mapping_list[element]] = scoring_rule[index]
    #     return np.array(array_n)







