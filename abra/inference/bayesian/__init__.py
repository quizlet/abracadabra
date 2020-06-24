import os
import sys
import logging
from pystan import StanModel
import pickle as pickle
from abra.config import STAN_MODEL_CACHE


HERE = os.path.dirname(os.path.abspath(__file__))
STAN_MODEL_HOME = os.path.join(HERE, 'models')

DEFAULT_BETA_ALPHA = 1.
DEFAULT_BETA_BETA = 1.
DEFAULT_MAX_STD = 30.
DEFAULT_PRECISION = 30.


if not os.path.isdir(STAN_MODEL_CACHE):
    os.makedirs(STAN_MODEL_CACHE)


def get_stan_model_code(model_name, **model_args):
    clean_model_name = model_name.replace("-", "_").replace(" ", "_")

    # Continuous models
    if clean_model_name in ('normal', 'gaussian', 'g'):
        from .models.continuous import gaussian as model_code_function
    elif clean_model_name in ('student_t', 't', 'exp_student_t', 'est'):
        from .models.continuous import exp_student_t as model_code_function

    # Binary / proportions models
    elif clean_model_name in ('binomial', 'b', 'beta_binomial', 'bb'):
        from .models.binary import beta_binomial as model_code_function
    elif clean_model_name in ('bernoulli', 'bn'):
        from .models.binary import bernoulli as model_code_function

    # Counts / rate models
    elif clean_model_name in ('poisson', 'p', 'gamma_poisson', 'gp'):
        from .models.counts import gamma_poisson as model_code_function

    return model_code_function()


def get_stan_model(model_name, recompile=False, **model_params):
    """
    Compile a Stan probabilistic model, or load pre-compiled model from cache,
    if available.

    Parameters
    ----------
    model_name : str
        Then name of the model used
            "bernoulli", "b": Bernoulli likelihood flat prior on probabilities
            "beta-binomial", "bb": Binomial likelihood and Beta prior
    recompile : boolean
        If set to to True, always recompile the model, otherwise try to use
        the cached pickle of the model.
    """

    python_version = 'python{0[0]}.{0[1]}'.format(sys.version_info)

    filename = "-".join([_f for _f in [model_name, python_version] if _f])
    compiled_model_file = os.path.join(STAN_MODEL_CACHE, filename + ".pickle")

    if os.path.isfile(compiled_model_file) and not recompile:
        with open(compiled_model_file, 'rb') as m:
            model = pickle.load(m)
    else:
        model_code = get_stan_model_code(model_name)
        model = StanModel(model_code=model_code)
        logging.info('Saving model to {}'.format(compiled_model_file))
        with open(compiled_model_file, 'wb') as f:
            pickle.dump(model, f)

    return model


def get_model_datatype(model_name):
    clean_model_name = model_name.replace("-", "_").replace(" ", "_")
    if clean_model_name in ('gaussian', 'g',
                            'student_t', 'st',
                            'exp_student_t', 'est'):
        return float
    elif clean_model_name in ('bernoulli', 'bn',
                              'beta_binomial', 'bb',
                              'binomial', 'b',
                              'poisson', 'p',
                              'gamma_poisson', 'gp'):
        return int
    raise ValueError('Unknown model name {!r}'.format(model_name))


def get_model_data(control_observations, variation_observations, model_name, model_params={}):
    """
    Performs any model-specific preprocessing for observations or model
    hyperparameters. We deal with hyperprams as model data here so that there
    is no need to re-compile Stan models for different model configurations.

    Parameters
    ----------
    control_observations : ndarray
        The observattions vector for the control treatment
    variation_observations : ndarray
        The observattions vector for the variation treatment
    model_name : str
        The name of one of the supported Bayesian Stan models
    model_params : dict
        Any additional model keyword arguments (e.g. hyperparameters) to be
        included as input data to the Stan model. For details on model-specific
        arguments see the model definitions in the ./models/* directory

    Returns
    -------
    iput_data : dict
        Data variable for input to the PyStan model interface.
    """

    clean_model_name = model_name.replace("-", "_").replace(" ", "_")

    # # Deal with input data
    # Models that use raw sequence inputs
    if clean_model_name in ('bernoulli',
                            'gaussian',
                            'student_t',
                            'exp_student_t',
                            'gamma_poisson'):
        input_data = dict(
            n_control=int(len(control_observations)),
            n_variation=int(len(variation_observations)),
            control=control_observations,
            variation=variation_observations
        )
    # Models that use summary statistics inputs
    elif clean_model_name in ('beta_binomial', 'bb', 'binomial'):
        input_data = dict(
            n_control=int(len(control_observations)),
            n_variation=int(len(variation_observations)),
            s_control=int(sum(control_observations)),
            s_variation=int(sum(variation_observations)),
        )

    # # Deal with model-specific hyperparameters passed in `model_params`
    hyperparameters = {}
    if clean_model_name in ('bernoulli', 'beta_binomial', 'binomial', 'gamma_poisson'):
        alpha = model_params.get('alpha', DEFAULT_BETA_ALPHA)
        beta = model_params.get('beta', DEFAULT_BETA_BETA)
        hyperparameters.update(dict(alpha_=float(alpha), beta_=float(beta)))

    elif clean_model_name in ('gaussian', 'student_t', 'exp_student_t'):
        std_max = model_params.get('std_max',  DEFAULT_MAX_STD)
        hyperparameters.update(dict(std_max=std_max))

        if clean_model_name in ('student_t', 'exp_student_t'):
            precision = model_params.get('precision', DEFAULT_PRECISION)
            hyperparameters.update(dict(precision=precision))

    input_data.update(hyperparameters)

    return input_data, hyperparameters
