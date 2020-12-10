from abra.inference.inference_base import InferenceProcedure, FrequentistProcedure


def get_inference_procedure(method, **infer_params):
    _method = method.lower().replace('-', '').replace('_', '').replace(' ', '')
    if _method in ('meansdelta'):
        from abra import MeansDelta as IP

    elif _method in ('proportionsdelta'):
        from abra import ProportionsDelta as IP

    elif _method in ('ratesratio'):
        from abra import RatesRatio as IP

    elif _method in ('bootstrap'):
        from abra import BootstrapDelta as IP

    elif method in (
        'gaussian',
        'bernoulli',
        'binomial',
        'beta_binomial',
        'gamma_poisson',
        'student_t',
        'exp_student_t'
    ):

        from abra import BayesianDelta as IP
        infer_params.update({"model_name": method})
    else:
        raise ValueError('Unknown inference method {!r}'.format(method))

    return IP(method=method, **infer_params)


__all__ = [
    "InferenceProcedure",
    "FrequentistProcedure"
]
