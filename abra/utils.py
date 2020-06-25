#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import string
from pandas import DataFrame


def dict_to_object(item):
    """
    Recursively convert a dictionary to an object.
    """
    def convert(item):
        if isinstance(item, dict):
            return type('DictToObject', (), {k: convert(v) for k, v in item.items()})
        if isinstance(item, list):
            def yield_convert(item):
                for index, value in enumerate(item):
                    yield convert(value)
            return list(yield_convert(item))
        else:
            return item
    return convert(item)


def ensure_dataframe(data, data_attr='data'):
    """
    Check if an object is a dataframe, and if not, check if it has an
    attribute that is a dataframe.
    """
    if not isinstance(data, DataFrame):
        if hasattr(data, data_attr) and isinstance(getattr(data, data_attr), DataFrame):
            data = getattr(data, data_attr)
        else:
            raise ValueError('`data` is incorrect format, must be a DataFrame')

    return data


def set_backend():
    """
    Set supported matplotlb backend depending on platform.
    """
    from sys import platform
    import matplotlib as mpl
    backend = 'pdf' if platform == 'darwin' else 'agg'
    mpl.use(backend)
    return backend


def generate_fake_observations(
    n_observations=10000,
    n_treatments=2,
    n_attributes=2,
    distribution='bernoulli',
    seed=123
):
    """
    Create a dataframe of artificial observations to be used for testing and demos.
    Treatments have different means, but segments defined by the attributes have
    no effect.

    Parameters
    -----------
    n_observations: int
        number of unique observations (e.g. users)
    n_treatments: int
        the number of simulated treatments. Will create an equivalent
        number of cohorts. Each successive treatment, by definition will
        have a systematically larger mean metric values (see metric
        distribution below). Note: the maximum number of treatments allowed
        is 6.
    n_attributes: int
        the number of attribute columns simulated. The number of distinct
        values taken by each attribute column is sampled uniformly between
        1 and 4. The segments defined by n_attributes have no effect on the
        sample means.
    distribution: str
        the type of metric distributions simulated
            - 'bernoulli': the mean increases from .5 by .1 for each successive treatment
            - 'gaussian': the mean increases from 0 by 1 for each successive treatment
            - 'poisson': the mean increases from 0 by 10 for each successive treatment
    seed: int
        The random number generator seed.
    """
    np.random.seed(seed)

    letters = string.ascii_uppercase
    n_treatments = min(n_treatments, 6)

    data = pd.DataFrame()
    data['id'] = list(range(n_observations))

    # add treatments
    treatments = list(letters[:n_treatments])
    data['treatment'] = np.random.choice(treatments, size=n_observations)

    # add attributes (attributes should have no effect)
    attribute_columns = ['attr_{}'.format(i) for i in range(n_attributes)]
    for ai, attr in enumerate(attribute_columns):
        attr_vals = ['A{}{}'.format(ai, a.lower()) for a in list(letters[:np.random.randint(1, 4)])]
        data[attr] = np.random.choice(attr_vals, size=n_observations)

    # add measurements, each treatment has successively larger means
    for delta, tr in enumerate(treatments):
        tr_mask = data.treatment == tr
        n_tr = sum(tr_mask)
        if 'gauss' in distribution:
            data.loc[tr_mask, 'metric'] = delta + np.random.randn(n_tr)
        elif 'bern' in distribution:
            data.loc[tr_mask, 'metric'] = list(map(bool, np.round(.1 * delta + np.random.random(n_tr))))
        elif 'poiss' in distribution:
            data.loc[tr_mask, 'metric'] = list(map(int, np.random.poisson(1 + delta , size=n_tr) ))

    return data


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    Usage
    -----
    with suppress_stdout_stderr():
        function_with_stdout_stderr()

    Notes
    -----
    This *should* not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited.

    For details see:
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *args, **kwargs):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all pointers
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        os.close(self.save_fds[0])
        os.close(self.save_fds[1])


class run_context(suppress_stdout_stderr):
    """
    Context manager with optional input to allow or suppress stdout and stderr.

    Parameters
    ----------
    suppress : bool
        Whether or not to suppress stdout and stderr
    """
    def __init__(self, suppress=True, *args, **kwargs):
        if suppress:
            super(run_context, self).__init__(*args, **kwargs)

        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            super(run_context, self).__enter__()

    def __exit__(self, *args, **kwargs):
        if self.suppress:
            super(run_context, self).__exit__()


def safe_isnan(val):
    if val is not None:
        return np.isnan(val)
    return False


def safe_cast_json(data, mapping):
    """
    Apply safe casting of common problem data types (see TYPE_MAPPING).
    """
    _apply = lambda x: _safe_cast_json(x, mapping)
    if isinstance(data, (str, bool)):
        return data
    elif isinstance(data, Mapping):
        return type(data)({k: _apply(v) for k, v in list(data.items())})
    elif isinstance(data, Sequence):
        # # additional sequence processing, no None in sequences
        # _data = [_apply(v) for v in data]
        # _data = [d if d is not None else 0 for d in _data]
        return type(data)(_apply(v) for v in data)
    else:
        return mapping.get(data, data) if not safe_isnan(data) else mapping[np.nan]
