#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.config import DEFAULT_ALPHA, DEFAULT_BAYESIAN_INFERENCE_METHOD
from abra.inference.inference_base import InferenceProcedure
from abra.stats import Samples
from abra.vis import Traces
from abra.inference.bayesian import get_stan_model, get_model_datatype, get_model_data
from abra.inference.bayesian.results import BayesianTestResults
import numpy as np


SAMPLING_DEFAULT_INFERENCE_KWARGS = dict(iter=2000, chains=4, n_jobs=-1, seed=1,
                                         control=dict(stepsize=0.01,
                                                      adapt_delta=0.99))
VB_DEFAULT_INFERENCE_KWARGS = dict(iter=10000, seed=1)


class BayesianDelta(InferenceProcedure):
    """
    Runs Bayesian inference procedure to test for the difference between two
    samples. Uses Stan to run inference proceduress.
    Parameters
    ----------
    model_name: str
        The name of the model to use:
            "gaussian", "g"       : Gaussian model (symetric, continuous)
            "student-t", "t"      : Student's t model (symetric, continuous)
            "binomial", "b"       : Binomial model (binary)
            "beta-binomial", "bb" : Beta-Binomial model (binary)
            "bernoulli", "bn"     : Bernoulli model (binary)
            "poisson", "p"        : Poisson model (counts)
            "gamma-poisson", "gp" : Gamma-Poisson model (counts)
    model_params: dict
        Arguments specific for initializing each type of Bayesian model.
        See bayesian inference README for details on specifications for each model.
    hypothesis: str
        The alternative hypothesis, or how we think the variation is related to
        the control.
    """

    def __init__(self,
                 model_name,
                 model_params={},
                 hypothesis='larger',
                 *args, **kwargs):

        super(BayesianDelta, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.data_type = get_model_datatype(model_name)
        self.hypothesis = hypothesis
        self.model_params = model_params
        self.bm = get_stan_model(model_name, **model_params)
        self.trace = None

    def ensure_dtypes(self, data, name=None):
        if len(data) == 0:
            raise ValueError("No observations provided for {!r} treatment".format(name))
        return np.array(data, dtype=self.data_type)

    def inference_inputs(self, control, variation):
        """
        Processes the control and variations data for the particular model
        """
        control_observations = self.ensure_dtypes(control.data, 'control')
        variation_observations = self.ensure_dtypes(variation.data, 'variation')
        return get_model_data(control_observations, variation_observations, self.model_name, self.model_params)

    @property
    def summary(self):
        return self.traces.summary

    def plot_trace(self, variable, *args, **kwargs):
        return self.traces.plot(variable, *args, **kwargs)

    def run(self, control_samples, variation_samples,
            alpha=DEFAULT_ALPHA, inference_kwargs=None):
        """
        Run the inference procedure over the samples using a particular
        inference method.
        control_samples: instance of Samples
            the samples for the control condition
        variation_smaples: instance of Samples
            the samples for the varitation condition
        alpha: float
            One minus the credible mass under the posterior, used to calculate
            the credible interval. Roughly analogous to the alpha in frequentist
            tests.
        inference_kwargs : dict
            Additional kwargs that used by the Stan sampling or variational
            inference methods. For details, see:
              - http://mc-stan.org/rstan/reference/stanmodel-method-sampling.html
              - http://mc-stan.org/rstan/reference/stanmodel-method-vb.html
        """
        if isinstance(control_samples, (list, np.ndarray)):
            control_samples = Samples(control_samples)

        if isinstance(variation_samples, (list, np.ndarray)):
            variation_samples = Samples(variation_samples)

        self.alpha = alpha
        self.credible_mass = 1 - alpha
        self.control = control_samples
        self.variation = variation_samples
        inference_data, hyperparameters = self.inference_inputs(control_samples, variation_samples)
        self.hyperparameters = {k:v for k, v in list(hyperparameters.items()) if k!='recompile'}

        # check for `inference_method` in inference_kwargs
        if inference_kwargs is not None:
            if 'inference_method' in inference_kwargs:
                inference_method = inference_kwargs['inference_method']
                del inference_kwargs['inference_method']
            else:
                inference_method = DEFAULT_BAYESIAN_INFERENCE_METHOD
        else:
            inference_method = DEFAULT_BAYESIAN_INFERENCE_METHOD

        self.inference_method = inference_method

        # run inference
        if inference_method[0] == "s":  # Gibs sampling
            if inference_kwargs is None:
                inference_kwargs = SAMPLING_DEFAULT_INFERENCE_KWARGS
            inference_results = self.bm.sampling(data=inference_data, **inference_kwargs)

        elif inference_method[0] == "v":  # variational bayes
            if inference_kwargs is None:
                inference_kwargs = VB_DEFAULT_INFERENCE_KWARGS
            inference_results = self.bm.vb(data=inference_data, **inference_kwargs)
        else:
            raise ValueError('Unknown inference procedure {!r}'.format(inference_method))

        self.traces = Traces(self.extract_traces(inference_results))

        return self.traces

    def extract_traces(self, inference_results):
        if issubclass(inference_results.__class__, dict):
            traces = {}
            for ii in range(len(inference_results['sampler_param_names'])):
                param_name = inference_results['sampler_param_names'][ii]
                param_value = np.array(inference_results['sampler_params'][ii])
                traces[param_name] = param_value
        else:
            traces = inference_results.extract()
        return traces

    @property
    def stats(self):
        summary = self.summary
        delta = summary.loc['delta', 'mean']
        delta_relative = summary.loc['delta_relative', 'mean']
        effect_size = summary.loc['effect_size', 'mean']
        prob_greater = self.traces.delta.prob_greater_than(0.)
        return delta, delta_relative, effect_size, prob_greater

    def hdi(self, alpha, variable='delta'):
        """
        Calculate the highest density interval for the delta posterior.
        """
        values = getattr(self.traces, variable).hdi(alpha)
        percentiles = (alpha, 1 - alpha)
        return (values, percentiles)

    @property
    def hypothesis_text(self):
        variation_name = self.variation.name
        variation_name = variation_name if variation_name else 'variation'
        if self.hypothesis == 'larger':
            return "{} is larger".format(variation_name)
        elif self.hypothesis == 'smaller':
            return "{} is smaller".format(variation_name)
        else:
            return "{} != {}".format(variation_name, self.control.name)

    def make_results(self):
        """
        Package up inference results
        """
        delta, delta_relative, effect_size, prob_greater = self.stats
        hdi = self.hdi(self.alpha)
        hdi_relative = self.hdi(self.alpha, 'delta_relative')
        return BayesianTestResults(control=self.control,
                                   variation=self.variation,
                                   delta=delta,
                                   delta_relative=delta_relative,
                                   effect_size=effect_size,
                                   alpha=self.alpha,
                                   traces=self.traces,
                                   hdi=hdi,
                                   hdi_relative=hdi_relative,
                                   prob_greater=prob_greater,
                                   model_name=self.model_name,
                                   hypothesis=self.hypothesis_text,
                                   inference_method=self.inference_method,
                                   data_type=self.data_type,
                                   inference_procedure=self,
                                   hyperparameters=self.hyperparameters
                                   )
