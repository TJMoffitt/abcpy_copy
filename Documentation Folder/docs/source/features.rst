Added functions
=====

A description of the added functions and their use.
------------

Each of the continuous models is extended for use in the developed samplers by the addition of the following functions:

.. class:: ContinuousModel

    .. py:function:: gradlogpdf(Input_values: List[float], X: float) -> float

        This function computes the gradient of the logarithm of the pdf. The gradient is explicitly defined for each class within the continuous models.

        :param Input_values: A list containing the corresponding parameters for the chosen model. The variables should be provided in their true (post-transformation) form for accuracy. Below, we'll include a list of each mathematical variable as input for every model and its common name in the respective pdf. We'll also provide the explicit form of the pdf for clarity.
        :param X: A single float value representing the point at which the gradient of the pdf is calculated. It should be provided in its post-transformation form, ensuring the gradlogpdf is computed directly on the given parameters.
        :return: Gradient of the logarithm of the pdf at the point X.
        
    .. py:function:: transform_list() -> List[Callable]

        Provides a list of transformations which will map values from the real line (where our sgld algorithms operate for each of the d parameters) to the appropriate space for our model pdf. For example, to R+ in the case of a Gaussian variance. These transformations, defined in PyTorch, are returned as a list in the order of the input parameters for each model. The transformations are utilized in [Name functions] from the graphtools class, enabling the operation of sgld algorithms over the entire real line.

        :return: A list of transformation functions for model parameters.
        
    .. py:function:: inverse_transform_list() -> List[Callable]

        Defines the inverse transformations corresponding to the functions in `transform_list`. The inverses are sequenced in the same order as the transformations. Each inverse function, designed in PyTorch, is used in specific functions (you can list them as [Add the names here]) in the graphtools class.

        :return: Inverses of the transformations from `transform_list`.


.. class:: Inference

    Inference class contains methods and utilities to perform parameter inference. The two algorithms added are the SGLD and ADSGLD.

    .. class:: SGLD

        Stochastic Gradient Langevin Dynamics algorithm (SGLD). It is often outperformed by ADSGLD, so while SGLD is included for completeness, it's recommended to use ADSGLD for most practical implementations.

        .. note::
           For more information on SGLD, refer to:
           Welling, Max, and Yee W. Teh. 2011. "Bayesian Learning via Stochastic Gradient Langevin Dynamics."

        .. py:method:: __init__(root_models: List[Model], gradloglikfuns: List[Function], backend: BackendType, kernel=None, seed=None)

            Initializes the SGLD Class.

            :param root_models: User-defined model for which parameters are to be inferred. Given as a list containing the model class.
            :param gradloglikfuns: Likelihood function used in the model. For SGLD algorithms, this would typically be the energy or kernel score defined.
            :param backend: Backend used for parallelization inherited from the abcpy structure. Currently, neither SGLD nor ADSGLD support parallelization; thus, `BackendDummy()` from abcpy.backends should be used.
            :param kernel: Perturbation kernel. Not used for either of the algorithms, defaults to None.
            :param seed: Random seed for the kernel. Not used and should remain as None by default.

        .. py:method:: sample(self, observations: List[float], n_samples: int, n_samples_per_param: int = 100, burnin: int = 1000, step_size: float = 0.001, iniPoint=None, w_val: int = 3, bounds=None, speedup_dummy: bool = True, n_groups_correlated_randomness=None, use_tqdm: bool = True, journal_file=None, path_to_save_journal=None) -> Journal

            Simulates likely parameter values for the user-defined model given observed values.

            :param observations: List of observations for which model parameters should be inferred.
            :param n_samples: Number of posterior samples to generate post burn-in.
            :param n_samples_per_param: Number of samples generated for each iteration of the algorithm.
            :param burnin: The number of initial sampled parameters that should be discarded as their accuracy might be low.
            :param step_size: Defines the size of the change for each step in the algorithm.
            :param iniPoint: Initial point for the simulation. 
            :param w_val: Balances the trade-off between the scoring rule and the prior pdfs.
            :param bounds: Bounds for the parameters.
            :param speedup_dummy: If True, speedup measures are applied.
            :param n_groups_correlated_randomness: Number of groups for correlated randomness.
            :param use_tqdm: If True, displays a progress bar.
            :param journal_file: Path to a pre-existing journal file.
            :param path_to_save_journal: Path where the resulting journal should be saved.
            :return: A custom data structure “Journal” inherited from the abcpy framework.

        .. note::
           The SGLD algorithm takes time to converge. It's advisable to visualize the values of your posteriors as a time series to assess the appropriate burn-in period.


    .. class:: ADSGLD

        Adjusted Stochastic Gradient Langevin Dynamics algorithm (ADSGLD). This outperforms the SGLD in most cases. It is advised to use the ADSGLD algorithm for most practical implementations, although in simpler cases, it may exhibit somewhat slower convergence compared to SGLD.

        .. note::
           For more information on ADSGLD, refer to:
           Welling, Max, and Yee W. Teh. 2011. "Bayesian Learning via Stochastic Gradient Langevin Dynamics."

        .. py:method:: __init__(root_models: List[Model], gradloglikfuns: List[Function], backend: BackendType, kernel=None, seed=None)

            Initializes the ADSGLD Class.

            :param root_models: User-defined model for which parameters are to be inferred. Given as a list containing the model class.
            :param gradloglikfuns: Likelihood function used in the model. For ADSGLD algorithms, this would typically be the energy or kernel score defined.
            :param backend: Backend used for parallelization inherited from the abcpy structure. Currently, neither ADSGLD nor SGLD support parallelization; thus, `BackendDummy()` from abcpy.backends should be used.
            :param kernel: Perturbation kernel. Not used for either of the algorithms, defaults to None.
            :param seed: Random seed for the kernel. Not used and should remain as None by default.

        .. py:method:: sample(self, observations: List[float], n_samples: int, n_samples_per_param: int = 100, burnin: int = 1000, diffusion_factor: float = 0.01, step_size: float = 0.001, iniPoint=None, w_val: int = 3, bounds=None, speedup_dummy: bool = True, n_groups_correlated_randomness=None, use_tqdm: bool = True, journal_file=None, path_to_save_journal=None) -> Journal

            Simulates likely parameter values for the user-defined model given observed values.

            :param observations: List of observations for which model parameters should be inferred.
            :param n_samples: Number of posterior samples to generate post burn-in.
            :param n_samples_per_param: Number of samples generated for each iteration of the algorithm.
            :param burnin: The number of initial sampled parameters that should be discarded as their accuracy might be low.
            :param diffusion_factor: Initializes the adaptive thermostat (xi) and sets the random noise level added at each step for momentum.
            :param step_size: Defines the size of the change for each step in the algorithm.
            :param iniPoint: Initial point for the simulation. 
            :param w_val: Balances the trade-off between the scoring rule and the prior pdfs.
            :param bounds: Bounds for the parameters.
            :param speedup_dummy: If True, speedup measures are applied.
            :param n_groups_correlated_randomness: Number of groups for correlated randomness.
            :param use_tqdm: If True, displays a progress bar.
            :param journal_file: Path to a pre-existing journal file.
            :param path_to_save_journal: Path where the resulting journal should be saved.
            :return: A custom data structure “Journal” inherited from the abcpy framework.

        .. note::
           The ADSGLD algorithm, while advanced, also takes time to converge. As always, practitioners are advised to monitor convergence carefully.


.. class:: Approx_Lhd

    In `approx_lhd`, we introduce scoring rules for parameter inference. We currently support energy and kernel score methods. Each method computes both the log_score and the gradient of the log score.

    .. class:: EnergyScore

        Energy score is one of two scoring rules defined in the introduction. It computes the log of the energy score and its gradient for a given set of values.

        .. note::
           Energy score reference: [Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.]

        .. py:method:: __init__(Statistics_calc=None, Model=None, Beta: float = [Default], Mean: bool = False)

            Initializes the EnergyScore class.

            :param Statistics_calc: Not used. This parameter might be removed in future versions.
            :param Model: User-defined model for which parameters are being calculated. This parameter is not used and might be removed in future versions.
            :param Beta: Beta value for the beta norm of the energy score. Acceptable range is [0, 2] inclusive.
            :param Mean: If set to True, returns the mean of the energy score over observed values. Default is False.

        .. py:method:: Loglikelihood(Y_obs: List[float], Y_Sim: List[float]) -> float

            Computes the log of the energy score for given values.

            :param Y_obs: List of observed values.
            :param Y_Sim: List of simulated values.
            :return: Float representing the energy score for the given observed and simulated values.

        .. py:method:: GradLoglikelihood(Y_obs: List[float], Y_Sim: List[List[float]]) -> List[float]

            Computes the gradient of the energy score with respect to each of the parameters of the user-defined model.

            :param Y_obs: List of observed values.
            :param Y_Sim: Y_sim consists of simulated values and a list of lists of their gradients with respect to each of the parameters of the model.
            :return: List of gradients of the energy score with respect to each parameter of the user-defined model in the order defined in that model.


    .. class:: KernelScore

        Kernel score is the second of two scoring rules defined in the introduction. It computes the log of the kernel score and its gradient for a given set of values.

        .. note::
        Kernel score reference: [Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.]

        .. py:method:: __init__(Statistics_calc=None, Model=None, Kernel: Callable, Mean: bool = False)

            Initializes the KernelScore class.

            :param Statistics_calc: Not used. This parameter might be removed in future versions.
            :param Model: User-defined model for which parameters are being calculated. This parameter is not used and might be removed in future versions.
            :param Kernel: A PyTorch function that defines the kernel for computing the scoring function. For instance, the energy score can be obtained by supplying the beta_norm function from the energy score (with a negated returned value). Another example is the RBF function [cite].
            :param Mean: If set to True, returns the mean of the kernel score over observed values. Default is False.

        .. py:method:: Loglikelihood(Y_obs: List[float], Y_Sim: List[float]) -> float

            Computes the log of the kernel score for given values.

            :param Y_obs: List of observed values.
            :param Y_Sim: List of simulated values.
            :return: Float representing the kernel score for the given observed and simulated values.

        .. py:method:: GradLoglikelihood(Y_obs: List[float], Y_Sim: List[List[float]]) -> List[float]

            Computes the gradient of the kernel score with respect to each of the parameters of the user-defined model using the autograd capability in PyTorch.

            :param Y_obs: List of observed values.
            :param Y_Sim: Y_sim consists of simulated values and a list of lists of their gradients with respect to each of the parameters of the model. 
            :return: List of gradients of the kernel score with respect to each parameter of the user-defined model in the order defined in that model.

.. class:: graphtools

   A litany of helper functions are defined in Graph Tools, bridging the gap between continuous priors, likelihood functions (scoring rules), user-defined models, and inference algorithms. Though these functions are primarily for internal use, ensuring the seamless operation of the package, they are documented here for transparency, further development, and understanding of the package's internals.

.. function:: GradSimulate(N_samples_per_param: int, Rng=None)

   Simulates gradients. 

   :param N_samples_per_param: Number of values and corresponding gradients to simulate.
   :param Rng: Defaults to ``np.random.RandomState()``. However, this might be deprecated in future versions.
   :return: [Needs Description]
   
   .. note::
      This function distinguishes itself by calling the ``grad_forward_simulate`` method instead of the usual ``forward_simulate``. Additionally, it applies the ``transform_variables`` function from the user-defined model, ensuring correct parameter value translation.

.. function:: grad_log_pdf_of_prior(Models: List, Parameters: List[np.array]) -> np.array

   Computes the gradient of the logarithm of the PDFs for the priors.

   :param Models: List containing the variable for the user-defined model.
   :param Parameters: List of numpy arrays representing the parameter values.
   :return: Array of floats representing the gradients of the log of the PDFs based on the input.

   .. note::
      Current simulation algorithms are limited to one model at a time, though the input allows for a list to maintain consistency with existing abcpy structure.

.. function:: apply_local_transform(model_prior, values) -> [ReturnType]

   Transforms parent parameters into the correct space.

   :param model_prior: [Needs Description]
   :param values: [Needs Description]
   :return: Transformed parameters.

   .. note::
      This function uses the transformations provided by ``transform_list()``. 

.. function:: full_transform() -> List

   Collates all transformations from hierarchical prior models.

   :return: Array containing transformations in the correct order.

.. function:: apply_full_transform(values: List) -> List

   Applies all transformations to a given set of values.

   :param values: Array of values.
   :return: Transformed values array.

.. function:: full_inverse_transform() -> List

   Compiles all inverse transformations from hierarchical prior models.

   :return: Array containing inverse transformations in the correct order.

.. function:: apply_full_inverse_transform(values: List) -> List

   Applies all inverse transformations to a given set of values.

   :param values: Array of values.
   :return: Transformed values array.

